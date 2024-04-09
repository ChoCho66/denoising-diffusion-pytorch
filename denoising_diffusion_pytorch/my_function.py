import torch
from torchvision import transforms
from PIL import Image
# 定义转换
# def pil_to_tensor(x):
def pil2tensor(x:Image):
    return transforms.ToTensor()(x)
# pil_to_tensor = transforms.ToTensor()
# tensor_to_pil = transforms.ToPILImage()
# def tensor_to_pil(x:torch):
def tensor2pil(x:torch):
    return transforms.ToPILImage()(x.clamp(0,1))

def add_to_class(Class):
  def wrapper(obj):
    setattr(Class, obj.__name__, obj)
  return wrapper

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

from sys import getsizeof
cuda_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def images2gif(
    images: list,
    save_path: str = "images.gif",
    loop=0, duration=40
):
    """
    images: list of tensors with length L \n
    image: tensor with shape (c,w,h)
    """
    images = [tensor2pil(i) for i in images]
    return images[0].save(save_path, save_all=True, append_images=images[1:], loop=loop, duration=duration)
  
def save_info(path,model,diffusion,trainer):
    info_to_save = {
        "diffusion.image_size": diffusion.image_size,
        "model.dim_mults": model.dim_mults,
        "diffusion.num_timesteps": diffusion.num_timesteps,
        "diffusion.beta_schedule": diffusion.beta_schedule,
        "diffusion.objective": diffusion.objective,
        "trainer.batch_size": trainer.batch_size,
        "self.num_fid_samples": trainer.num_fid_samples,
        "trainer.fid": trainer.fid,
    }

    with open(path, 'w') as file:
        for title, value in info_to_save.items():
            file.write(f"{title}: {value}\n")

def fix_seed(seed: int = 66):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # print(f"Random seed set as {seed}")
