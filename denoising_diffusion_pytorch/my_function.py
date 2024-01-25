import torch
from torchvision import transforms
# 定义转换
def pil_to_tensor(x):
    return transforms.ToTensor()(x)
# pil_to_tensor = transforms.ToTensor()
# tensor_to_pil = transforms.ToPILImage()
def tensor_to_pil(x:torch):
    return transforms.ToPILImage()(x.clamp(0,1))

def add_to_class(Class):
  def wrapper(obj):
    setattr(Class, obj.__name__, obj)
  return wrapper

from sys import getsizeof
cuda_or_cpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def images_to_gif(
    images: list,
    save_path: str = "images.gif",
    loop=0, duration=40
):
    """
    images: list of tensors with length L \n
    images.shape = [L,c,w,h]
    """
    images = [tensor_to_pil(i) for i in images]
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
