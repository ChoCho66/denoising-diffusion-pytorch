from denoising_diffusion_pytorch.my_function import *
import pandas as pd
import os
import math
# import copy
from pathlib import Path
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms
# from torchvision import transforms as utils
from torchvision import utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.my_fid_evaluation import FIDEvaluation_label

from denoising_diffusion_pytorch.version import __version__

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions


def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

# prob = 1 <=> all ones.    prob = 0 <=> all zeros.


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        # 如果左邊的值 < prob 則為 True
        # 所以這裡相當於 Bern(prob)
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    """
    input shape: (b, 1)
    output shape: (b, dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        
        # Ex:
        # half_dim = 64
        # emb = ~e
        emb = math.log(10000) / (half_dim - 1)
        
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb = [ exp{~e}, exp{-2~e}, ..., ] = [ e1, e2, ..., e63 ]
        
        
        emb = time[:, None] * emb[None, :]
        # time[:, None] :  [[ t1 ], 
        #                   [ t2 ],
        #                   [ t3 ]]
        # embeddings[None, :] :  [ [e1, e2, ..., e63] ]
        # emb = [ [t1e1, t1e2, ..., t1e63],
        #         [t2e1, t2e2, ..., t2e63],
        #         [t3e1, t3e2, ..., t3e63] ]
        
        
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        # 主要在做 group normalization
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        # x -> self.proj -> self.norm -> self.act
        # - 使用 self.proj 調整維度
        # - 使用 self.norm 做 GroupNorm 
        # - 使用 self.act 做 SiLU 
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """
    input: x with shape (b,c,h,w)\n
    output: shape (b,c,h,w)\n
    heads: the number of (single) attention\n
    dim_head: the hidden dimension of each q, k, v (in each attention)
    """

    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # the shape of self.to_qkv(x) = (b, 3 * dim_head * heads, h, w)
        # 把 self.to_qkv(x) 在 dim=1 的地方拆成 (q,k,v) 這 tuple,
        # 其中 q,k,v 每個的 shape 皆為 (b, dim_head * heads, h, w)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 讓 q,k,v 的 shape: (b, heads * dim_head, h, w) -> (b, heads, dim_head, h * w)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        # inner product of q and k (in dim_head)
        # sim = similarity of (q,k)
        # the shape of sim = (b, heads, h * w, h * w)
        sim = einsum('b h d i, b h d j -> b h i j', q, k)

        # the score of (q,k) = attn = sim.softmax(dim = -1)
        # score 恆正 所以取 softmax
        # the shape of attn = (b, heads, h * w, h * w)
        attn = sim.softmax(dim=-1)

        # the product of score(q,k) and v
        # the shape of attn = (b, heads, h * w, h * w)
        # the shape of v = (b, heads, dim_head, h * w)
        # the shape of out = (b, heads, h * w, dim_head)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        # shape: (b, heads, h * w, dim_head) -> (b, heads * dim_head, h, w)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        # shape: (b, heads * dim_head, h, w) -> (b, dim, h, w)
        return self.to_out(out)

# model


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes: int,
        cond_drop_prob=0.5,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attn_dim_head=32,
        attn_heads=4
    ):
        super().__init__()

        self.num_classes = num_classes

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        self.dim_mults = dim_mults
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim,
                            classes_emb_dim=classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(
            mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(
            mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(
            mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out,
                            time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_out + dim_in, dim_out,
                            time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(
            dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    # Sampling 時的 $\widetilde{\varepsilon_{\theta}}(x_t,c)$
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=1.,
        rescaled_phi=0.,
        **kwargs
    ):

        # cond_drop_prob = 0 <=> keep conditioning information
        # cond_drop_prob = the probability of forgetting conditioning information

        # - `logits` = $\varepsilon_{\theta}(z_{\lambda},c)$
        # - `null_logits` = $\varepsilon_{\theta}(z_{\lambda},\emptyset)$
        # - `scaled_logits` = $\widetilde{\varepsilon_{\theta}}(z_{\lambda},c)$
        # - `rescaled_logits` = $\widetilde{\varepsilon_{\theta}}(z_{\lambda},c)\cdot \dfrac{\mathtt{std}(\varepsilon_{\theta}(z_{\lambda},c))}{\mathtt{std}(\widetilde{\varepsilon_{\theta}}(z_{\lambda},c))}$
        # - `return`: a linear combination of `scaled_logits` and `rescaled_logits`

        # logits = eps_the(z_lam,c)

        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        # cond_scale = s in article
        # s = w + 1

        if cond_scale == 1:
            return logits

        # null_logits = eps_the(z_lam)

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        # scaled_logits = til(eps_the)(z_lam,c)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim=tuple(
            range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * \
            (std_fn(logits) / std_fn(scaled_logits))

        # return

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    # training 時用的 noise predictor
    # x : (b,c,h,w)
    # time : (b,1)
    # classes : (b,)
    def forward(
        self,
        x,
        time,
        classes,
        cond_drop_prob=None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance

        # classes_emb 為 classes 的 embedding in R^dim
        classes_emb = self.classes_emb(classes)

        # cond_drop_prob = 0 <=> keep conditioning information
        # cond_drop_prob = the probability of forgetting conditioning information

        if cond_drop_prob > 0:
            # prob = 1 <=> all ones.    prob = 0 <=> all zeros.
            # def prob_mask_like(shape, prob, device):
            #     return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
            
            # keep_mask 為 (b,) 的 tensor，裡面每個元素都為 Bernoulli(prob)
            keep_mask = prob_mask_like(
                (batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(
                self.null_classes_emb, 'd -> b d', b=batch)

            # For each batch, classes_emb 裡的每個 x 的資訊，只有 keep_mask 為 1 的才會保留，其他的都會被替換成 null_classes_emb
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # U-net

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

# gaussian diffusion trainer class


def extract(a, t: torch, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) /
                      tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# class Dataset_condition_attr(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         list_attr_celeba_path,
#         condition_attr,
#         exts = ['jpg', 'jpeg', 'png', 'tiff'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

#         self.transform = transforms.Compose([
#             transforms.Lambda(maybe_convert_fn),
#             transforms.Resize(image_size),
#             transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor()
#         ])

#         df_attr = pd.read_csv(list_attr_celeba_path)
#         df_attr = df_attr[['image_id']+condition_attr]
#         df_attr['condition_attr'] = df_attr.copy().groupby(condition_attr).ngroup()
#         self.df_attr = df_attr[['image_id','condition_attr']]
#         self.num_classes = len(self.df_attr['condition_attr'].unique())

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         path = self.paths[index]
#         img = Image.open(path)
#         image_id = str(path).split("/")[-1]
#         classes = self.df_attr.iloc[index,-1]
#         return self.transform(img), classes, image_id+".jpg"


class MyDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        list_attr_celeba_path,
        condition_attr,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        image_paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]
        image_paths.sort()
        self.paths = image_paths
        self.images_we_have = [os.path.basename(p) for p in self.paths]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(
            convert_image_to) else nn.Identity()

        self.transform = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_neg_one_to_one),
            # transforms.Lambda(lambda t: t * 2 - 1),
        ])

        df_attr = pd.read_csv(list_attr_celeba_path)
        df_attr = df_attr[df_attr['image_id'].isin(self.images_we_have)]
        df_attr = df_attr[['image_id']+condition_attr]
        df_attr['condition_attr'] = df_attr.copy().groupby(
            condition_attr).ngroup()
        self.df_attr = df_attr[['image_id', 'condition_attr']]
        self.num_classes = len(self.df_attr['condition_attr'].unique())

        # mean_classes = []
        # for c in range(self.num_classes):
        #     images_c = []
        #     df_mean_cond_c = self.df_attr[self.df_attr['condition_attr'] == c]
        #     image_path_list = [os.path.join(self.folder , image_id) for image_id in list(df_mean_cond_c['image_id'])]
        #     for img_path in image_path_list:
        #         images_c.append(self.transform(Image.open((img_path))))
        #     images_c = torch.stack(images_c, dim=0)
        #     images_c = images_c.to(cuda_or_cpu)
        #     mean_c = torch.mean(images_c, dim=0)
        #     mean_classes.append(mean_c.to("cpu"))
        #     torch.cuda.empty_cache()
        # self.mean_classes = mean_classes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_id = os.path.basename(path).split(".")[0]
        classes = self.df_attr.iloc[index, -1]
        return self.transform(img), classes, image_id+".jpg"

    def img(self, index):
        img = self[index][0]
        img = unnormalize_to_zero_to_one(img)
        img = img.clamp(0, 1)
        return tensor2pil(img)

    # def img_mean_classes(self, classes):
    #     img = self.mean_classes[classes]
    #     img = unnormalize_to_zero_to_one(img)
    #     img = img.clamp(0,1)
    #     return tensor_to_pil(img)


class MyDataset1(Dataset):
    def __init__(
        self,
        folder_list,
        image_size,
        allowed_extensions=['.jpg', '.jpeg', '.png', '.webp', '.tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()
        self.folder_list = folder_list
        self.image_size = image_size
        self.paths = []
        self.targets = []
        self.num_classes = len(folder_list)

        for class_idx, folder_path in enumerate(self.folder_list):
            folder_path = Path(folder_path)
            for item in folder_path.glob(f'**/*'):
                _, ext = os.path.splitext(item.name)
                if ext.lower() in allowed_extensions:
                    self.paths.append(str(item))
                    self.targets.append(class_idx)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(
            convert_image_to) else nn.Identity()

        self.transform = transforms.Compose([
            transforms.Lambda(maybe_convert_fn),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_neg_one_to_one),
            # transforms.Lambda(lambda t: t * 2 - 1),
        ])

    def _add_images_from_folder(self, folder_path, class_idx, allowed_extensions):
        for item in folder_path.iterdir():
            if item.is_dir():
                self._add_images_from_folder(
                    item, class_idx, allowed_extensions)
            else:
                _, ext = os.path.splitext(item.name)
                if ext.lower() in allowed_extensions:
                    self.paths.append(str(item))
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        img 範圍: -1~1
        """
        img_path = self.paths[idx]
        target = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target, img_path

    def img(self, idx):
        img = self[idx][0]
        img = unnormalize_to_zero_to_one(img)
        img = img.clamp(0, 1)
        display(self[idx][1])
        display(tensor2pil(img))

    def count_mean(self, filename="data_mean.pt"):
        mean_classes = []
        indices_classes = [[index for index, value in enumerate(
            self.targets) if value == c] for c in range(self.num_classes)]
        for c in range(self.num_classes):
            images_c = []
            image_path_list = [self.paths[idx] for idx in indices_classes[c]]
            for img_path in image_path_list:
                images_c.append(self.transform(Image.open((img_path))))
            images_c = torch.stack(images_c, dim=0)
            # images_c = images_c.to(cuda_or_cpu)
            mean_c = torch.mean(images_c, dim=0)
            mean_classes.append(mean_c.to("cpu"))
            torch.cuda.empty_cache()
        self.data_mean = torch.stack(mean_classes, dim=0)
        torch.save(self.data_mean, filename)

    def load_mean(self, filename="data_mean.pt"):
        self.data_mean = torch.load(filename)

    def img_mean(self, idx):
        img = self.data_mean[idx]
        img = unnormalize_to_zero_to_one(img)
        img = img.clamp(0, 1)
        display(tensor2pil(img))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=1.,
        offset_noise_strength=0.,
        min_snr_loss_weight=False,
        min_snr_gamma=5
    ):
        # is_minus_mean = True,
        super().__init__()
        assert not (
            type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.num_classes = model.num_classes
        # self.mean_classes = dataset.mean_classes
        # self.is_minus_mean = is_minus_mean

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # sig_t^2

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        # mu_t(x_t,x_0) = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        (x_0, x_t, t) -> (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale=6., rescaled_phi=0.7, clip_x_start=False):
        """
        return pred_noise, pred_x0
        """
        model_output = self.model.forward_with_cond_scale(
            x, t, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            """
            model(xt,t) = pred_noise
            pred_x0 = pred_x0(xt,t,pred_noise)
            """
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            """
            model(xt,t) = pred_x0
            pred_noise = pred_noise(xt,t,pred_x0)
            """
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            """
            model(xt,t) = pred_v
            pred_x0 = pred_x0(xt,t,pred_v)
            pred_noise = pred_noise(xt,t,pred_x0)
            """
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised=True):
        """
        (xt, t, classes, cond_scale, rescaled_phi) \n
        -> μ_θ(xt,t), Σ_t, log Σ_t, pred x0(xt)

        t shape: (b,1)
        """
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, classes: torch.int, cond_scale=6., rescaled_phi=0.7, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, classes=classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, classes, shape, cond_scale=6., rescaled_phi=0.7,
                      return_all_timesteps=False,
                      return_predict_x0=False):
        """
        有 unnormalize_to_zero_to_one 
        """
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None
        x_starts = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(
                img, t, classes, cond_scale, rescaled_phi)
            imgs.append(img)
            x_starts.append(x_start)

        # if self.is_minus_mean:
        #     img = img + self.mean_by_given_classes(classes).to(device)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret2 = x_start if not return_all_timesteps else torch.stack(
            x_starts, dim=1)

        ret = unnormalize_to_zero_to_one(ret)
        ret2 = unnormalize_to_zero_to_one(ret2)

        if return_predict_x0:
            return ret, ret2
        else:
            return ret

    @torch.inference_mode()
    def ddim_sample(self, classes, shape, cond_scale=6., rescaled_phi=0.7, clip_denoised=True,
                    return_all_timesteps=False,
                    return_predict_x0=False):
        """
        shape: (batch_size, channels, image_size, image_size)
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None
        x_starts = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                x_starts.append(x_start)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) *
                           (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            imgs.append(img)
            x_starts.append(x_start)

        # if self.is_minus_mean:
        #     img = img + self.mean_by_given_classes(classes).to(device)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret2 = x_start if not return_all_timesteps else torch.stack(
            x_starts, dim=1)

        ret = unnormalize_to_zero_to_one(ret)
        ret2 = unnormalize_to_zero_to_one(ret2)

        if return_predict_x0:
            return ret, ret2
        else:
            return ret

    def sample2gif(self, i0=0, classes=None, is_ddim=False, is_sample=True, cond_scale=6., rescaled_phi=0.7, save_gif_path=None, show_results=False, n = 5):
        with torch.inference_mode():
            if is_sample == True:
                if classes is None:
                    classes = torch.randint(
                        0, self.num_classes, (16,)).to(self.device)
                batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
                sample_fn = self.p_sample_loop if not is_ddim else self.ddim_sample
                image_list = sample_fn(classes,
                                       (batch_size, channels,
                                        image_size, image_size),
                                       cond_scale, rescaled_phi,
                                       return_all_timesteps=True,
                                       return_predict_x0=True,
                                       )
                """
                The shape of image_list[0], image_list[1] is
                torch.Size([16, 200, 3, 64, 64])
                """
                torch.save(image_list[0], 'backward_process_gif.pth')
                torch.save(image_list[1], 'pred_x0_gif.pth')
            else:
                image_list = ['', '']
                image_list[0] = torch.load('backward_process_gif.pth')
                image_list[1] = torch.load('pred_x0_gif.pth')

            A = image_list[0][i0].permute(0, 2, 3, 1).cpu()[1:].clamp(0, 1)
            B = image_list[1][i0].permute(0, 2, 3, 1).cpu().clamp(0, 1)

            if is_ddim:
                def update(frame):
                    imgA.set_array(A[frame])
                    imgB.set_array(B[frame])
                    axA.set_title(
                        f'Backward process: xti, i={self.sampling_timesteps-1-frame}')
                    return imgA, imgB
            else:
                def update(frame):
                    imgA.set_array(A[frame])
                    imgB.set_array(B[frame])
                    axA.set_title(
                        f'Backward process: xt, t={self.num_timesteps-1-frame}')
                    return imgA, imgB

            fig, (axA, axB) = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))

            imgA = axA.imshow(A[0])
            axA.set_title('Backward process: xt')

            imgB = axB.imshow(B[0])
            axB.set_title('predict x0')

            animation = FuncAnimation(
                fig, update, frames=len(A), interval=100, blit=True)
            if save_gif_path:
                animation.save(filename=save_gif_path, writer="pillow")
            display(HTML(animation.to_jshtml()))
            plt.close()

            if show_results:
                I0, J0 = 2, 8
                fig, axes = plt.subplots(I0, J0, figsize=(16, 4))

                for i in range(I0):
                    for j in range(J0):
                        img = image_list[0][i*J0 + j][-1]
                        img = tensor2pil(img)
                        axes[i, j].imshow(img)
                        axes[i, j].axis('off')

                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.show()

    @torch.inference_mode()
    def sample(self,
               classes=None,
               cond_scale=6., rescaled_phi=0.7):
        if classes is None:
            classes = torch.randint(0, self.num_classes, (1,)).to(self.device)
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels

        # p_sample_loop and ddim_sample 最後都有 unnormalize_to_zero_to_one
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, channels, image_size, image_size), cond_scale, rescaled_phi)

    @torch.inference_mode()
    def interpolate(self, x1, x2, classes, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, _ = self.p_sample(img, i, classes)

        img = unnormalize_to_zero_to_one(img)
        return img

    # TODO: classes 要改？
    @torch.inference_mode()
    def interpolate_from_x1_to_x2(self, x1, x2, classes_int: int, t: int = None, s=(2, 8), save_gif_path=None, power_f=1):
        """
        s + 1 等分 \n
        x1,x2 with shape (1,c,w,h) (batch = 1, single image)
        classes = torch.tensor([class] * (s0*s1+1) ).cuda()
        """
        b, *_, device = *x1.shape, x1.device
        assert b == 1

        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        x1t, x2t = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        ss = s[0] * s[1]

        classes = torch.tensor([classes_int] * (ss+1)).to(device)

        # 建立等差數列 lam
        lam = torch.linspace(-1, 1, ss+1).to(device) ** power_f
        if power_f % 2 == 0:
            lam[:(ss+1)//2] *= -1
        lam = (lam + 1)/2

        lam = lam.view(-1, 1, 1, 1)

        # 使用廣播執行內插
        img = (1 - lam) * x1t + lam * x2t

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, _ = self.p_sample(img, i, classes)

        img = unnormalize_to_zero_to_one(img)
        img_list = list(img.clamp(0, 1).cpu())

        if save_gif_path:
            images2gif(img_list, save_path=save_gif_path, duration=200)

        # 创建一个2x5的子图布局
        _, axs = plt.subplots(s[0], s[1], figsize=(2 * s[1], 2 * s[0]))

        # 将每张图像显示在对应的子图中
        for i in range(s[0]):
            for j in range(s[1]):
                index = i * s[1] + j
                axs[i, j].imshow(img_list[index].permute(1, 2, 0))
                # axs[i, j].axis('off')  # 关闭坐标轴

        # 调整布局，以免重叠
        plt.tight_layout()
        plt.show()

    @torch.inference_mode()
    def new_interpolate_from_x1_to_x2(self, x1, x2, classes_int: int, t=None, s=16, save_gif_path=None, power_f=1):
        """
        s + 1 等分 \n
        x1,x2 with shape (b,c,w,h) (b=1, single image)
        """
        b, *_, device = *x1.shape, x1.device
        assert b == 1

        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        x1t, x2t = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        classes = torch.tensor([classes_int] * (s+1)).to(device)

        # 建立等差數列 lam
        lam = torch.linspace(-1, 1, s+1).to(device) ** power_f
        if power_f % 2 == 0:
            lam[:(s+1)//2] *= -1
        lam = (lam + 1)/2

        lam = lam.view(-1, 1, 1, 1)

        # 使用廣播執行內插
        img = (1 - lam) * x1t + lam * x2t

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, _ = self.p_sample(img, i, classes)

        img = unnormalize_to_zero_to_one(img)
        img = img.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        img_list = list(img)
        # img_list = list(img.clamp(0,1).cpu())

        def update(frame):
            imgA.set_array(img_list[frame])
            axA.set_title(f'Interpolation of (x1,x2), s={frame}/{s}')
            return imgA,
        fig, (ax1, axA, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(6, 3))
        fig.subplots_adjust(left=0.01, bottom=0.1,
                            right=0.99, top=0.9, wspace=0.1)
        img1 = ax1.imshow(unnormalize_to_zero_to_one(
            x1[0]).permute(1, 2, 0).cpu())
        imgA = axA.imshow(img_list[0])
        axA.set_title('Interpolation of (x1,x2), s=0')
        img2 = ax2.imshow(unnormalize_to_zero_to_one(
            x2[0]).permute(1, 2, 0).cpu())
        animation = FuncAnimation(fig, update, frames=len(
            img_list), interval=100, blit=True)
        if save_gif_path:
            animation.save(filename=save_gif_path, writer="pillow")
        display(HTML(animation.to_jshtml()))
        plt.close()

    # def forward_backwrad_process(self, x_start:torch, classes:torch, num_timesteps:int, b:int=7, contains_backward=True, cond_scale = 6., rescaled_phi = 0.7):
    def forward_backwrad_process(self, ds: Dataset, num_timesteps: int, classes_int: int = -1, b: int = 7, contains_backward=True, cond_scale=6., rescaled_phi=0.7, n=5):
        """
        x_start with shape (b, w, h, c)
        """
        with torch.inference_mode():
            _ = DataLoader(ds, batch_size=b, shuffle=True)
            _ = next(iter(_))
            x_start = _[0].clone()
            if classes_int < 0:
                classes = _[1].to(self.device)
            else:
                classes = torch.ones(b, dtype=torch.int).to(self.device) * classes_int
            noise_list = []
            xt_list = []
            noise_list.append(torch.zeros_like(x_start[0]))
            xt_list.append(unnormalize_to_zero_to_one(x_start))
            x_t = x_start.to(self.device)
            for t in range(num_timesteps):
                noise_t = torch.randn_like(x_t[0]).to(self.device)
                noise_t = torch.stack([noise_t.clone().detach()
                                      for __ in range(b)]).to(self.device)
                x_t = torch.sqrt(1-self.betas[t]) * x_t + \
                    torch.sqrt(self.betas[t]) * noise_t
                if t % n == 0:
                    xt_list.append(unnormalize_to_zero_to_one(x_t))
                    noise_list.append(unnormalize_to_zero_to_one(noise_t[0]))
            for t in tqdm(reversed(range(0, num_timesteps)), desc='Backward process', total=num_timesteps):
                noise_t = torch.zeros_like(x_t[0])
                x_t, _ = self.p_sample(
                    x_t, t, classes, cond_scale, rescaled_phi)
                if t % n == 0:
                    noise_list.append(unnormalize_to_zero_to_one(noise_t))
                    xt_list.append(unnormalize_to_zero_to_one(x_t))
            xt_list = [i.permute(0, 2, 3, 1).cpu().clamp(0, 1)
                       for i in xt_list]
            noise_list = [i.permute(1, 2, 0).cpu().clamp(0, 1)
                          for i in noise_list]

            def update(frame):
                img0.set_array(xt_list[frame][0])
                img1.set_array(xt_list[frame][1])
                img2.set_array(xt_list[frame][2])
                img3.set_array(xt_list[frame][3])
                img4.set_array(xt_list[frame][4])
                img5.set_array(xt_list[frame][5])
                img6.set_array(xt_list[frame][6])
                img_noise.set_array(noise_list[frame])

                if n*frame < num_timesteps:
                    ax3.set_title(f'Forward process, t={n*frame}')
                    return img0, img1, img2, img3, img4, img5, img6, img_noise
                else:
                    ax3.set_title(
                        f'Backward process, t={2*num_timesteps - n*frame}')
                    return img0, img1, img2, img3, img4, img5, img6, img_noise

            _fig, (_ax0, _ax1, _ax2, _ax3, _ax4, _ax5, _ax6, _ax_noise) = plt.subplots(
                nrows=1, ncols=8, figsize=(12, 3))
            _fig.subplots_adjust(left=0.01, bottom=0.1,
                                 right=0.99, top=0.9, wspace=0.1)
            _ax0.imshow(xt_list[0][0])
            _ax1.imshow(xt_list[0][1])
            _ax2.imshow(xt_list[0][2])
            _ax3.imshow(xt_list[0][3])
            _ax4.imshow(xt_list[0][4])
            _ax5.imshow(xt_list[0][5])
            _ax6.imshow(xt_list[0][6])
            _ax_noise.imshow(noise_list[0])
            _ax3.set_title('x0')

            fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax_noise) = plt.subplots(
                nrows=1, ncols=8, figsize=(12, 3))
            fig.subplots_adjust(left=0.01, bottom=0.1,
                                right=0.99, top=0.9, wspace=0.1)
            img0 = ax0.imshow(xt_list[0][0])
            img1 = ax1.imshow(xt_list[0][1])
            img2 = ax2.imshow(xt_list[0][2])
            img3 = ax3.imshow(xt_list[0][3])
            img4 = ax4.imshow(xt_list[0][4])
            img5 = ax5.imshow(xt_list[0][5])
            img6 = ax6.imshow(xt_list[0][6])
            img_noise = ax_noise.imshow(noise_list[0])

            ax3.set_title('Forward and Backward process, t=0')
            ax_noise.set_title('Noise')
            animation = FuncAnimation(fig, update, frames=len(
                xt_list), interval=100, blit=True)
            # if save_gif_path:
            #     animation.save(filename=save_gif_path, writer="pillow")
            display(HTML(animation.to_jshtml()))
            plt.close()

            __fig, (__ax0, __ax1, __ax2, __ax3, __ax4, __ax5, __ax6,
                    __ax_noise) = plt.subplots(nrows=1, ncols=8, figsize=(12, 3))
            __fig.subplots_adjust(left=0.01, bottom=0.1,
                                  right=0.99, top=0.9, wspace=0.1)
            __ax0.imshow(xt_list[-1][0])
            __ax1.imshow(xt_list[-1][1])
            __ax2.imshow(xt_list[-1][2])
            __ax3.imshow(xt_list[-1][3])
            __ax4.imshow(xt_list[-1][4])
            __ax5.imshow(xt_list[-1][5])
            __ax6.imshow(xt_list[-1][6])
            __ax_noise.imshow(noise_list[-1])
            __ax3.set_title('Final x0')

    @autocast(enabled=False)
    def q_sample(self, x_start, t: torch, noise=None):
        """
        t shape: (b,)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += self.offset_noise_strength * \
                rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t: torch, classes, noise=None):
        b, c, h, w = x_start.shape
        """
        t shape: (b,)
        """
        # noise sample
        noise = default(noise, lambda: torch.randn_like(x_start))

        # xt ~ q(xt)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        # For the case pred_noise
        # predict noise = eps_theta(x_t,t)
        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # For the case pred_noise
        # loss = || predict_noise - real_noise ||_2
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    # def forward(self, img, classes, *args, **kwargs):
    # Training 時的 loss = diffusion(training_images, classes = image_classes)
    # training_images ~ q(x0)
    def forward(self, img, classes):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # t ~ unif({0,1,...,T})
        # t shape: (b,)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img)
        # if self.is_minus_mean == True:
        #     img = img - self.mean_by_given_classes(classes).to(device)

        return self.p_losses(img, t, classes)

    # def mean_by_given_classes(self, classes):
    #     return torch.concat([self.mean_classes[c].unsqueeze(dim=0) for c in classes], dim=0)


def divisible_by(numer, denom):
    return (numer % denom) == 0

# trainer class


class Trainer(object):
    def __init__(
        self,
        dataset,
        diffusion_model,
        results_folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        amp=True,
        mixed_precision_type='fp16',
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.,
        num_fid_samples=10000,
        save_best_and_latest_only=False
    ):
        super().__init__()

        # fid
        self.fid = None

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = dataset

        # assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation_label(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=self.results_folder,
                device=self.device,
                num_fid_samples=50000,
                inception_block_idx=2048
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    X, y = next(self.dl)[:2]
                    X, y = X.to(device), y.to(device)

                    with self.accelerator.autocast():
                        # self.model = diffusion_model
                        loss = self.model(X, classes=y)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(
                                self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(
                                classes=torch.randint(
                                    0, self.ds.num_classes, (n,)).to(device)
                            ), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images, str(
                            self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def fid_score(self,
                  batch_size=128,
                  num_fid_samples=50000,
                  inception_block_idx=2048,
                  ):
        self.fid_scorer = FIDEvaluation_label(
            batch_size=batch_size,
            dl=self.dl,
            sampler=self.ema.ema_model,
            channels=self.channels,
            accelerator=self.accelerator,
            stats_dir=self.results_folder,
            device=self.device,
            num_fid_samples=num_fid_samples,
            inception_block_idx=inception_block_idx
        )
        accelerator = self.accelerator
        fid_score = self.fid_scorer.fid_score()
        accelerator.print(f'fid_score: {fid_score}')
        self.fid = fid_score


class Trainer_stu(object):
    def __init__(
        self,
        dataset,
        diffusion_model_teacher,
        diffusion_model,
        results_folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        amp=True,
        mixed_precision_type='fp16',
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.,
        num_fid_samples=10000,
        save_best_and_latest_only=False
    ):
        super().__init__()

        # fid
        self.fid = None

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.diffusion_model_teacher = diffusion_model_teacher
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = dataset

        # assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size=train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation_label(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=self.results_folder,
                device=self.device,
                num_fid_samples=50000,
                inception_block_idx=2048
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    X, y = next(self.dl)[:2]
                    X, y = X.to(device), y.to(device)

                    with self.accelerator.autocast():
                        # self.model = diffusion_model
                        
                        b = X.shape[0]
                        t = torch.randint(0, self.model.num_timesteps, (b,), device=device).long()
                        Xt = self.model.q_sample(X,t)
                        model_out = self.model.model(Xt, t, y)
                        target = self.diffusion_model_teacher.model(Xt, t, y)
                        
                        loss = F.mse_loss(model_out, target, reduction='none')
                        loss = reduce(loss, 'b ... -> b', 'mean')

                        # loss = loss * extract(self.loss_weight, t, loss.shape)
                        loss = loss.mean()

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(
                                self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(
                                classes=torch.randint(
                                    0, self.ds.num_classes, (n,)).to(device)
                            ), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images, str(
                            self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def fid_score(self,
                  batch_size=128,
                  num_fid_samples=50000,
                  inception_block_idx=2048,
                  ):
        self.fid_scorer = FIDEvaluation_label(
            batch_size=batch_size,
            dl=self.dl,
            sampler=self.ema.ema_model,
            channels=self.channels,
            accelerator=self.accelerator,
            stats_dir=self.results_folder,
            device=self.device,
            num_fid_samples=num_fid_samples,
            inception_block_idx=inception_block_idx
        )
        accelerator = self.accelerator
        fid_score = self.fid_scorer.fid_score()
        accelerator.print(f'fid_score: {fid_score}')
        self.fid = fid_score
        


# example

if __name__ == '__main__':
    num_classes = 10

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000
    ).cuda()

    training_images = torch.randn(8, 3, 128, 128).cuda() # images are normalized from 0 to 1
    image_classes = torch.randint(0, num_classes, (8,)).cuda()    # say 10 classes

    loss = diffusion(training_images, classes = image_classes)
    loss.backward()

    # do above for many steps

    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = 6.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    sampled_images.shape # (8, 3, 128, 128)

    # interpolation

    interpolate_out = diffusion.interpolate(
        training_images[:1],
        training_images[:1],
        image_classes[:1]
    )

