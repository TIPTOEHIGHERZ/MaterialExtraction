import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from typing import Union
import tqdm

from .Pipeline import Pipeline
from model.KVInjection.KVInjection import register_kv_injection, KVSaver, KVInjectionAgent


class MyPipeline(Pipeline):
    def __init__(self, pipeline: Union[StableDiffusionPipeline, str] = None, device='cpu', **kwargs):
        super().__init__(pipeline, device, **kwargs)

        return

    @torch.no_grad()
    def regenerate(self,
                   image: torch.Tensor,
                   mask: torch.Tensor,
                   reference_image: torch.Tensor,
                   num_inference_steps=50,
                   guidance_scale=7.5,
                   blending_percentage=0.25):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        batch_size = image.shape[0]
        prompts = [''] * batch_size

        source_latents = self.image2latents(image)
        # latents, noised_source_latents = self.invert(source_latents, num_inference_steps,
        #                       is_latents=True, save_kv=False, use_injection=False, save_period=1)
        latents = self.invert(source_latents, num_inference_steps,
                              is_latents=True, save_kv=False, use_injection=False)
        # get kv features
        self.invert(reference_image, num_inference_steps, save_kv=True, use_injection=False)
        w, h = latents.shape[-2:]
        if mask.ndim == 4:
            mask = mask[:, 0, :, :].unsqueeze(1)
        else:
            mask = mask[0, :, :].unsqueeze(0).unsqueeze(0)
        mask_latents = nn.functional.interpolate(mask, size=(w, h))

        self.scheduler.set_timesteps(num_inference_steps)
        text_embeddings = self.prepare_prompt_embeddings(prompts, batch_size)
        
        latents = self.sample(source_latents, latents, text_embeddings, mask_latents, desc='Coarse sampling')
        self.restart_savers()
        # self.clear_savers()

        latents = self.invert(latents, num_inference_steps,
                              is_latents=True, save_kv=False, use_injection=False)
        # self.invert(reference_image, num_inference_steps, save_kv=True, use_injection=False)     

        latents = self.sample(source_latents, latents, text_embeddings, mask_latents, desc='Fine sampling')
        
        image = self.latents2image(latents)

        return image

    def register_kv_injection(self,
                              kv_injection_agent: KVInjectionAgent,
                              num_inference_steps: int,
                              register_name=''):
        register_kv_injection(self.unet, kv_injection_agent, num_inference_steps, register_name)
        return
    
    def clear_savers(self):
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and name.endswith('attn1'):
                module.kv_saver.clear()
        return
    
    def restart_savers(self):
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and name.endswith('attn1'):
                module.kv_saver.restart()
        return

    def sample(self,
               source_latents: Union[torch.Tensor, list[torch.Tensor]],
               latents: torch.Tensor,
               text_embeddings: torch.Tensor,
               mask_latents: torch.Tensor,
               desc: str,
               guidance_scale=7.5):
        with tqdm.tqdm(self.scheduler.timesteps, desc=desc) as progress_bar:
            for i, t in enumerate(progress_bar):
                if isinstance(source_latents, torch.Tensor):
                    noised_source_latents = self.add_noise(source_latents, t, text_embeddings,
                                                        guidance_scale=guidance_scale, 
                                                        save_kv=False, use_injection=False)
                else:
                    noised_source_latents = source_latents[len(self.scheduler.timesteps) - i - 1]
                
                latents = self.remove_noise(latents, t, text_embeddings, guidance_scale=guidance_scale,
                                            save_kv=False, use_injection=True)
                latents = latents * mask_latents + noised_source_latents * (1 - mask_latents)
        
        return latents
