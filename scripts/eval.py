import torch
import sys
import os
import tqdm
from omegaconf import OmegaConf
from diffusers import DDPMScheduler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.pipeline import Pipeline
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor,
    device='cuda',
    num_inference_steps=50,
    verbose=False,
    cond_scale=3.,
):    
    pipeline.unet.eval()

    # pipeline.scheduler = scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    bsz = masked_ref_latents.shape[0]
    latents_shape = masked_ref_latents.shape[-2:]
    latents_start = torch.randn(bsz, 16, *latents_shape, device=masked_ref_latents.device)

    encoder_hidden_states = None
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(latents_start.shape[0])

        latents_input = torch.concat([latents_start, masked_ref_latents], dim=1)
        latents_input_ref = masked_ref_latents

        if cond_scale is not None:
            latents_input_null = torch.concat([latents_start, torch.zeros_like(masked_ref_latents)], dim=1)
            latents_input = torch.concat([latents_input, latents_input_null], dim=0)

        noise_pred = pipeline.unet(
            latents_input,
            latents_input_ref,
            t, 
            encoder_hidden_states, 
            return_dict=False, 
        )[0]

        if cond_scale is not None:
            noise_pred_cond, noise_pred_null = noise_pred.chunk(2, dim=0)
            # cfg
            noise_pred = noise_pred_null + cond_scale * (noise_pred_cond - noise_pred_null)

        t = t[:1]
        latents_start = pipeline.scheduler.step(noise_pred, t, latents_start, return_dict=False)[0]
    
    color = pipeline.latents2image(latents_start[:, :4, ...])
    normal = pipeline.latents2image(latents_start[:, 4:8, ...])
    metallic = pipeline.latents2image(latents_start[:, 8:12, ...])
    roughness = pipeline.latents2image(latents_start[:, 12:, ...])

    return color, normal, metallic, roughness


if __name__ == '__main__':
    # prepare
    device = 'cuda'

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )
    scheduler.set_timesteps(1000)

    pipeline: Pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4', scheduler=scheduler)
    unet = SiameseUnet.from_config('./configs/unet/render_all', config_main='config_main.json', config_ref='config_ref.json')
    pipeline.unet = unet
    pipeline.frozen()

    pipeline.to(device)
    
    

