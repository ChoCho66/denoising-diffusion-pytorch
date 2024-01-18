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
    loop=0, duration=400
):
    """
    images: list of tensors with length L \n
    images.shape = [L,c,w,h]
    """
    images = [tensor_to_pil(i) for i in images]
    return images[0].save(save_path, save_all=True, append_images=images[1:], loop=loop, duration=duration)