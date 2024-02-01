from denoising_diffusion_pytorch.my_denoising_diffusion_pytorch import *

model = Unet(
  dim=64,
  channels=3,
  dim_mults=(1,2,4,8),
  flash_attn=True,
)

diffusion = GaussianDiffusion(
  model,
  image_size=64,
  timesteps=200,
  sampling_timesteps=150,
  objective='pred_v'
)

x1 = torch.arange(1,3,64,64)
x2 = torch.arange(1,3,64,64)

diffusion.interpolate_from_x1_to_x2(
  x1.unsqueeze(0),
  x2.unsqueeze(0),
  t = 2,
)