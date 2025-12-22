import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline, DDPMScheduler
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from logging import Logger
import os
import sys
from typing import Union
from diffusers.models.attention_processor import Attention
import tqdm
from typing import Callable

sys.path.append(os.getcwd())

from utils.functionals import check_prompt


class Pipeline:
    def __init__(self, pipeline: Union[StableDiffusionPipeline, str] = None, device='cpu', **kwargs):
        self.logger = self.get_logger(self.__class__.__name__)
        self.model_path = ''

        if isinstance(pipeline, str):
            self.model_path = pipeline
            model_path = pipeline
            for key, value in kwargs.copy().items():
                # kwargs set to None isn't allowed to pass to from_pretrained
                if value is None:
                    kwargs.pop(key)

            pipeline = DiffusionPipeline.from_pretrained(model_path, **kwargs)

        self.device = device
        # self.unet: UNet2DConditionModel = pipeline.unet
        # self.scheduler: DDIMScheduler = pipeline.scheduler
        # self.vae: AutoencoderKL = pipeline.vae

        if pipeline is not None:
            self.text_encoder: CLIPTextModel = pipeline.text_encoder
            self.tokenizer: CLIPTokenizer = pipeline.tokenizer
            self.unet: UNet2DConditionModel = pipeline.unet
            self.scheduler: DDIMScheduler | DDPMScheduler = pipeline.scheduler
            self.vae: AutoencoderKL = pipeline.vae

        return

    def to(self, device: str):
        modules = [
            self.text_encoder,
            self.unet,
            self.vae
        ]

        for module in modules:
            module.to(device)

        self.scheduler.alphas = self.scheduler.alphas.to(device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.scheduler.betas = self.scheduler.betas.to(device)

        self.device = device
        return
    
    def scheduler_to(self, device):
        self.scheduler.alphas = self.scheduler.alphas.to(device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.scheduler.betas = self.scheduler.betas.to(device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        return
    
    def cuda(self):
        self.to('cuda')
        return self
    
    def cpu(self):
        self.to('cpu')
        return self

    def init(self, pipeline: StableDiffusionPipeline):
        self.unet: UNet2DConditionModel = pipeline.unet
        self.scheduler: DDIMScheduler | DDPMScheduler = pipeline.scheduler
        self.vae: AutoencoderKL = pipeline.vae

        return

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        for key, value in kwargs.copy().items():
            # kwargs set to None isn't allowed to pass to from_pretrained
            if value is None:
                kwargs.pop(key)

        pipeline = DiffusionPipeline.from_pretrained(model_path, **kwargs)
        pipeline.model_path = model_path
        pipeline = cls(pipeline)
        
        return pipeline

    @staticmethod
    def get_logger(logger_name: str) -> Logger:
        return logging.getLogger(logger_name)

    @torch.no_grad()
    def image2latents(self, image: torch.Tensor, mask: torch.Tensor=None):
        # normalize
        if image.ndim == 3:
            image.unsqueeze_(0)
        image = image * 2 - 1

        if mask is not None:
            image = image * mask

        latents = self.vae.encode(image, return_dict=False)[0].mean
        latents *= self.vae.config.scaling_factor
        # latents *= 0.18215

        return latents

    def latents2image(self, latents: torch.Tensor):
        # denormalize
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        # image = self.vae.decode(latents / 0.18215, return_dict=False)[0]
        image = image.clamp(-1, 1)
        image = (image + 1) / 2

        return image

    def prepare_prompt_embeddings(self, prompt, batch_size=1, use_conditional_guidance=False):
        device = self.device

        cond_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        cond_embeddings = self.text_encoder(cond_tokens['input_ids'].to(device))[0]

        if not use_conditional_guidance:
            return cond_embeddings

        uncond_tokens = self.tokenizer(
            [''] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_tokens['input_ids'].to(device))[0]

        return torch.cat([uncond_embeddings, cond_embeddings])
        # return uncond_embeddings

    @torch.no_grad()
    def add_noise(self,
                  sample: torch.Tensor,
                  timestep,
                  encoder_hidden_states: torch.Tensor,
                  noise: torch.Tensor = None,
                  guidance_scale=7.5,
                  use_conditional_guidance=False,
                  **cross_attention_kwargs):
        b = sample.shape[0]
        num_train_steps, num_inference_steps = len(self.scheduler.alphas), self.scheduler.num_inference_steps
        # print(timestep, num_inference_steps)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).to(sample.device)
        timestep = timestep if timestep.shape[0] == b else torch.concat([timestep] * b, dim=0)
        next_step, _ = torch.min(torch.concat([(timestep + num_train_steps // num_inference_steps).unsqueeze(1),
                                            torch.tensor([num_train_steps - 1] * timestep.shape[0], device=timestep.device).unsqueeze(1)], dim=-1), 
                                            dim=1)
        # next_step[i] = min(timestep[i] + num_train_steps // num_inference_steps, num_train_steps - 1)

        model_input = torch.concat([sample] * 2) if use_conditional_guidance else sample
        # model_input = sample
        if noise is None:
            noise_pred: torch.Tensor = self.unet(model_input, timestep, encoder_hidden_states,
                                                cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
            if use_conditional_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise

        alpha = self.scheduler.alphas_cumprod

        alpha_t = alpha[timestep]
        alpha_t = alpha_t.view(alpha_t.shape[0], *[1] * (noise_pred.ndim - 1))
        beta_t = 1 - alpha_t
        alpha_next = alpha[next_step]
        alpha_next = alpha_next.view(alpha_next.shape[0], *[1] * (noise_pred.ndim - 1))
        beta_next = 1 - alpha_next

        x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        x_next = alpha_next.sqrt() * x_0 + beta_next.sqrt() * noise_pred

        return x_next
    
    def remove_noise(self, 
                     sample: torch.Tensor, 
                     timestep, 
                     encoder_hidden_states: torch.Tensor,
                     noise=None,
                     eta=0.,
                     guidance_scale=7.5,
                     use_conditional_guidance=False,
                     attention_mask=None,
                     num_inference_steps=None,
                     **cross_attention_kwargs):
        num_train_steps = len(self.scheduler.alphas)
        num_inference_steps = self.scheduler.num_inference_steps if num_inference_steps is None else num_inference_steps
        pre_step = torch.max(timestep - num_train_steps // num_inference_steps, torch.zeros_like(timestep))

        alpha = self.scheduler.alphas_cumprod
        alpha_t = alpha[timestep]
        alpha_t = alpha_t.reshape(alpha_t.shape[0], 1, 1, 1)
        beta_t = 1 - alpha_t

        alpha_pre = alpha[pre_step]
        alpha_pre = alpha_pre.reshape(alpha_pre.shape[0], 1, 1, 1)
        beta_pre = 1 - alpha_pre
        
        sigma_t = eta * (beta_pre / beta_t).sqrt() * (1 - alpha_t / alpha_pre).sqrt()

        # latens_input = torch.concat([sample] * 2)
        # encoder_hidden_states = torch.concat([encoder_hidden_states] * 2)
        latens_input = torch.concat([sample] * 2) if use_conditional_guidance else sample
        # encoder_hidden_states = torch.concat([encoder_hidden_states] * 2, dim=0) \
        #     if encoder_hidden_states.shape[0] != latens_input.shape[0] else encoder_hidden_states
        if noise is None:
            noise_pred: torch.Tensor = self.unet(latens_input, timestep, encoder_hidden_states, attention_mask=attention_mask,
                                                cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
        else:
            noise_pred = noise

        if use_conditional_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        # x_pre = self.scheduler.step(noise_pred, timestep, sample).prev_sample
        x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        x_pre = alpha_pre.sqrt() * x_0 + (beta_pre - sigma_t ** 2).sqrt() * noise_pred + \
                sigma_t * torch.randn_like(sample)
        return x_pre
    
    def get_x0(
        self, 
        sample: torch.Tensor, 
        timestep, 
        encoder_hidden_states: torch.Tensor,
        noise=None,
        eta=0.,
        guidance_scale=7.5,
        use_conditional_guidance=False,
        attention_mask=None,
        num_inference_steps=None,
        **cross_attention_kwargs
    ):
        num_train_steps = len(self.scheduler.alphas)
        num_inference_steps = self.scheduler.num_inference_steps if num_inference_steps is None else num_inference_steps
        pre_step = torch.max(timestep - num_train_steps // num_inference_steps, torch.zeros_like(timestep))

        alpha = self.scheduler.alphas_cumprod
        alpha_t = alpha[timestep]
        alpha_t = alpha_t.reshape(alpha_t.shape[0], 1, 1, 1)
        beta_t = 1 - alpha_t

        alpha_pre = alpha[pre_step]
        alpha_pre = alpha_pre.reshape(alpha_pre.shape[0], 1, 1, 1)
        beta_pre = 1 - alpha_pre
        
        # latens_input = torch.concat([sample] * 2)
        # encoder_hidden_states = torch.concat([encoder_hidden_states] * 2)
        latens_input = torch.concat([sample] * 2) if use_conditional_guidance else sample
        # encoder_hidden_states = torch.concat([encoder_hidden_states] * 2, dim=0) \
        #     if encoder_hidden_states.shape[0] != latens_input.shape[0] else encoder_hidden_states
        if noise is None:
            noise_pred: torch.Tensor = self.unet(latens_input, timestep, encoder_hidden_states, attention_mask=attention_mask,
                                                cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
        else:
            noise_pred = noise

        if use_conditional_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        # x_pre = self.scheduler.step(noise_pred, timestep, sample).prev_sample
        x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()

        return x_0

    @torch.no_grad()
    def prompt2embeddings(self, prompt: Union[str, list[str]]) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = [prompt]
        
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        embeddings = self.text_encoder(tokens['input_ids'].to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def invert(self,
               image: torch.Tensor,
               num_inference_steps,
               prompt='',
               verbose=True,
               desc='DDIM Inverting',
               is_latents=True,
               return_intermidiate=True,
               use_conditional_guidance=False,
               **cross_attention_kwargs):

        batch_size = image.shape[0]
        device = image.device
        self.scheduler.set_timesteps(num_inference_steps)
        prompt = check_prompt(prompt, batch_size)
        latents = self.image2latents(image) if not is_latents else image
        timesteps = reversed(self.scheduler.timesteps)
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps

        text_embeddings = self.prepare_prompt_embeddings(prompt, batch_size, use_conditional_guidance=use_conditional_guidance)
        latents_list = list()
        for i, timestep in enumerate(iteration):
            latents = self.add_noise(latents, timestep, text_embeddings, **cross_attention_kwargs, use_conditional_guidance=use_conditional_guidance)
            if return_intermidiate:
                latents_list.append(latents.detach().cpu())

        if return_intermidiate:
            return latents, latents_list

        return latents

    @torch.no_grad()
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
                 **cross_attention_kwargs):
        device = self.vae.device
        batch_size = latents.shape[0] if latents is not None else 1
        prompt = check_prompt(prompt, batch_size)

        self.scheduler.set_timesteps(num_inference_steps)
        # latents = self.vae.encode(image, return_dict=False)[0].mode()
        latents = torch.randn([1, 4, width // self.pipeline.vae_scale_factor, height // self.pipeline.vae_scale_factor], device=device) \
        if latents is None else latents

        timesteps = self.scheduler.timesteps
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps

        text_embeddings = self.prepare_prompt_embeddings(prompt, batch_size, use_conditional_guidance=use_conditional_guidance)
        # encoder_hidden_states, _ = self.pipeline.encode_prompt(prompt, device, 1, True)
        for timestep in iteration:
            latents = self.remove_noise(latents, timestep, text_embeddings, None, 
                                        eta, use_conditional_guidance=use_conditional_guidance,
                                        **cross_attention_kwargs)

        # image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        # image = image.clamp(-1, 1)
        # image = (image + 1) / 2
        if return_latents:
            return latents

        image = self.latents2image(latents)
        # image = self.latents2image(latents)
        # do_denormalize = [True] * image.shape[0]
        # image = self.pipeline.image_processor.postprocess(image, do_denormalize=do_denormalize, output_type='pt')

        return image

    def predict_x_prev(self, x_t: torch.Tensor, t, noise_pred: torch.Tensor, eta=0.):
        batch_size = noise_pred.shape[0]

        alpha = self.scheduler.alphas_cumprod

        train_steps = alpha.shape[0]

        t_prev = t - train_steps // self.scheduler.num_inference_steps
        alpha_t = alpha[t].repeat(batch_size).reshape(-1, [1] * (noise_pred.ndim - 1))
        alpha_t_prev = alpha[t_prev].repeat(batch_size).reshape(-1, [1] * (noise_pred.ndim - 1))

        x_start = (x_t - (1 - alpha_t).sqrt() * noise_pred / alpha_t.sqrt())
        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - (alpha_t / alpha_t_prev))
        direct_x_t = (1 - alpha_t_prev - sigma ** 2.0).sqrt() * noise_pred
        random_noise = sigma * torch.randn_like(x_t, device=x_t.device)

        x_t_prev = alpha_t_prev.sqrt() * x_start + direct_x_t + random_noise

        return x_t_prev

    def check_kv_empty(self):
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and hasattr(module, 'save_kv'):
                assert module.save_kv.idx == 0, f'{name}\'s save_kv is not empty'

    @torch.no_grad()
    def __call__(self,
                 target_image: torch.Tensor,
                 texture_reference: torch.Tensor,
                 inversion_reference: torch.Tensor = None,
                 num_inference_steps=50):
        pass
    
    def generate(self, prompt, width=512, height=512, num_inference_steps=50, generator=None):
        latents = torch.randn([1, 4, width // self.pipeline.vae_scale_factor, height // self.pipeline.vae_scale_factor], 
                              device=self.device, generator=generator)
        
        return self.sampling(latents, width, height, num_inference_steps, prompt)
    
    def frozen(self):
        to_frozen: list[nn.Module] = [self.vae, self.unet, self.text_encoder]

        for module in to_frozen:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        
        return
    
    def predict_noise(
        self,
        x0: torch.Tensor,
        random_noise: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        **cross_attention_kwargs
    ):
        x_t = self.scheduler.add_noise(x0, random_noise, timestep)

        noise_pred: torch.Tensor = self.unet(x_t, timestep, encoder_hidden_states,
                                             cross_attention_kwargs=cross_attention_kwargs, return_dict=False)

        return noise_pred
    

class InpaintingPipeline(Pipeline):
    def __init__(self, pipeline: Union[StableDiffusionInpaintPipeline, str] = None, device='cpu', **kwargs):
        super().__init__(pipeline, device, **kwargs)

        return
    
    def predict_noise(
        self,
        masked_latents: torch.Tensor,
        mask: torch.Tensor,
        ref_latents: torch.Tensor,
        random_noise: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        **cross_attention_kwargs
    ):
        noisy_latents = self.scheduler.add_noise(masked_latents, random_noise, timestep)

        input_latents = torch.concat([noisy_latents, mask, ref_latents], dim=1)
        noise_pred: torch.Tensor = self.unet(input_latents, timestep, encoder_hidden_states,
                                             cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]

        return noise_pred
    
    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        embedding_matcher: Callable,
        embedder: Callable,
        device='cpu',
        use_cfg=False,
        num_inference_steps=50
    ):  
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        masked_latents = self.image2latents(image, 1 - mask)
        latents_mask = nn.functional.interpolate(mask, masked_latents.shape[-2:])

        mask = nn.functional.interpolate(mask, masked_latents.shape[-2:])
        noisy_latents = torch.randn_like(masked_latents)

        embedding_input = torch.concat([masked_latents, latents_mask], dim=1)
        image_embeds = embedding_matcher.encode(embedding_input, 1 - latents_mask)
        encoder_hidden_states = embedder(['*'], image_embeds)
        # encoder_hidden_states = self.prepare_prompt_embeddings([''])

        for t in tqdm.tqdm(self.scheduler.timesteps):
            t = t.reshape([1,])
            input_latents = torch.concat([noisy_latents, mask, masked_latents], dim=1)
            if use_cfg:
                input_latents = torch.concat([input_latents] * 2, dim=0)

            noise_pred = self.unet(input_latents, t, encoder_hidden_states)[0]
            noisy_latents = self.remove_noise(noisy_latents, t, None, noise_pred)
            
        return self.latents2image(noisy_latents)
    

if __name__ == '__main__':
    InpaintingPipeline.from_pretrained('./pretrained/stable-diffusion-2-inpainting')
