import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.io import load_image, save_image
from utils.functionals import crop_mid
from model.pipeline import Pipeline


pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4')

color = load_image('./datasets/polyhaven_examples/brick_wall_09_4k/brick_wall_09_diff_4k.png', to_batch=True)
tiled_color = torch.concat([color] * 2, dim=-1)
tiled_color = torch.concat([tiled_color] * 2, dim=-2)

save_dir = './eval/tiles'
os.makedirs(save_dir, exist_ok=True)
save_image(color, os.path.join(save_dir, 'color.png'))
save_image(crop_mid(tiled_color, (128, 128)), os.path.join(save_dir, 'tiled_color.png'))

latents = pipeline.image2latents(color)

color_recon = pipeline.latents2image(latents)
color_recon = torch.concat([color_recon] * 2, dim=-1)
color_recon = torch.concat([color_recon] * 2, dim=-2)
save_image(crop_mid(color_recon, (128, 128)), os.path.join(save_dir, 'tiled_recon.png'))

tiled_latents = torch.concat([latents] * 2, dim=-1)
tiled_latents = torch.concat([tiled_latents] * 2, dim=-2)
tiled_latents = pipeline.latents2image(tiled_latents)
save_image(crop_mid(tiled_latents, (128, 128)), os.path.join(save_dir, 'tiled_latents.png'))

tiled_latents = tiled_latents[..., :tiled_latents.shape[-2] // 2, :tiled_latents.shape[-1] // 2]
tiled_latents = torch.concat([tiled_latents] * 2, dim=-1)
tiled_latents = torch.concat([tiled_latents] * 2, dim=-2)
save_image(crop_mid(tiled_latents, (128, 128)), os.path.join(save_dir, 'tiled_latents_cropped.png'))

