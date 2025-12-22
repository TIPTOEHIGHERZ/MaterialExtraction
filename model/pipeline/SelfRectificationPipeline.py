import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from logging import Logger
import os
from typing import Union
from diffusers.models.attention_processor import Attention
from model.KVInjection import KVInjection
import tqdm
from utils.functionals import check_prompt
from .Pipeline import Pipeline


class SelfRectificationPipeline(Pipeline):
    def __init__(self, pipeline: Union[StableDiffusionPipeline, str]=None, device='cpu', **kwargs):
        super().__init__(pipeline, device, **kwargs)

        return
    
    def invert(self,
               image: torch.Tensor,
               num_inference_steps,
               prompt='',
               verbose=True,
               desc='DDIM Inverting',
               is_latents=True,
               interval=None,
               use_conditional_guidance=False,
               return_intermidiate_latents=True,
               latents_ref_list=None,
               **cross_attention_kwargs):
        assert interval is None or (len(interval) == 2 and interval[1] >= interval[0])
        if interval is None:
            interval = [0, num_inference_steps]

        batch_size = image.shape[0]
        device = image.device
        self.scheduler.set_timesteps(num_inference_steps)
        latents = self.image2latents(image) if not is_latents else image
        timesteps = reversed(self.scheduler.timesteps[interval[0]: interval[1]])
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps

        latents_list = list()
        for i, timestep in enumerate(iteration):
            if latents_ref_list is not None:
                latents_ref = latents_ref_list[-1 - i]
                latents = torch.concat([latents, latents_ref], dim=0)

            batch_size = latents.shape[0]
            prompt = check_prompt(prompt, batch_size)
            text_embeddings = self.prepare_prompt_embeddings(prompt, batch_size, use_conditional_guidance=use_conditional_guidance)
            latents = self.add_noise(latents, timestep, text_embeddings, **cross_attention_kwargs, use_conditional_guidance=use_conditional_guidance)

            if latents_ref_list is not None:
                latents = latents[0].unsqueeze(0)

            if return_intermidiate_latents:
                latents_list.append(latents)

        if return_intermidiate_latents:
            return latents, latents_list

        return latents
    
    def sampling(self,
                 latents: torch.Tensor=None,
                 width=512,
                 height=512,
                 num_inference_steps=50,
                 prompt='',
                 verbose=True,
                 desc='DDIM Sampling',
                 eta=0.,
                 return_latents=False,
                 use_conditional_guidance=False,
                 latents_ref_list=None,
                 **cross_attention_kwargs):
        device = self.vae.device
        batch_size = latents.shape[0] if latents is not None else 1

        self.scheduler.set_timesteps(num_inference_steps)

        latents = torch.randn([1, 4, width // self.pipeline.vae_scale_factor, height // self.pipeline.vae_scale_factor], device=device) \
        if latents is None else latents

        timesteps = self.scheduler.timesteps
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps

        if latents_ref_list is not None:
            assert len(latents_ref_list) == num_inference_steps

        for i, timestep in enumerate(iteration):
            if latents_ref_list is not None:
                lantents_ref = latents_ref_list[-1 - i]
                latents = torch.concat([latents, lantents_ref], dim=0)
            
            batch_size = latents.shape[0]
            prompt = check_prompt(prompt, batch_size)
            text_embeddings = self.prepare_prompt_embeddings(prompt, batch_size, use_conditional_guidance=use_conditional_guidance)
            latents = self.remove_noise(latents, timestep, text_embeddings, None, 
                                        eta, use_conditional_guidance=use_conditional_guidance,
                                        **cross_attention_kwargs)
            latents = latents[0].unsqueeze(0)
            

        if return_latents:
            return latents

        image = self.latents2image(latents)

        return image
