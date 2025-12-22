import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from logging import Logger
import os
from typing import Union
import tqdm
from utils.functionals import check_prompt
from .Pipeline import Pipeline


class BlendingPipeline(Pipeline):
    def __init__(self, pipeline: Union[StableDiffusionPipeline, str]=None, device='cpu', **kwargs):
        super().__init__(pipeline, device, **kwargs)
        return
    
    @staticmethod
    def from_pretrained(model_path: str, **kwargs):
        for key, value in kwargs.copy().items():
            # kwargs set to None isn't allowed to pass to from_pretrained
            if value is None:
                kwargs.pop(key)

        pipeline = DiffusionPipeline.from_pretrained(model_path, **kwargs)
        pipeline = BlendingPipeline(pipeline)

        return pipeline
    
    @torch.no_grad()
    def edit_image(self, 
                   image: torch.Tensor,
                   mask: torch.Tensor, 
                   prompts: Union[str, list[str]],
                   num_inference_steps=50,
                   guidance_scale=7.5,
                   blending_percentage=0.25,
                   generator=None):
        batch_size = image.shape[0]
        if isinstance(prompts, str):
            prompts = [prompts] * batch_size
        else: 
            assert len(prompts) == batch_size

        source_latents = self.image2latents(image)
        latents = torch.randn(source_latents.shape, device=self.device, generator=generator)
        w, h = latents.shape[-2:]
        mask = mask[:, 0, :, :].unsqueeze(1)
        mask_latents = nn.functional.interpolate(mask, size=(w, h))

        self.scheduler.set_timesteps(num_inference_steps)
        text_embeddings = self.prepare_prompt_embeddings(prompts, batch_size)

        with tqdm.tqdm(self.scheduler.timesteps[int(num_inference_steps * blending_percentage):], desc='Blending') as progressbar:
            for t in progressbar:
                noised_source_latents = self.add_noise(source_latents, t, text_embeddings, guidance_scale=guidance_scale)
                latents = self.remove_noise(latents, t, text_embeddings, guidance_scale=guidance_scale)
                latents = latents * mask_latents + noised_source_latents * (1 - mask_latents)
        
        # latents = latents * mask_latents + source_latents * (1 - mask_latents)
        image = self.latents2image(latents)

        return image
