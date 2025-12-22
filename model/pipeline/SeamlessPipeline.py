import torch
import torch.nn as nn

from .SelfRectificationPipeline import SelfRectificationPipeline
import tqdm
from utils.functionals import check_prompt
from textile import Textile as TextileLoss


class SeamlessPipeline(SelfRectificationPipeline):
    def __init__(self, model_path: str, textile_path: str, textile_config={}, device='cpu', **kwargs):
        super().__init__(model_path, device=device, **kwargs)
        self.textile_loss = TextileLoss(textile_path, **textile_config)

    @torch.no_grad()
    def invert(self,
               image: torch.Tensor,
               num_inference_steps,
               prompt='',
               verbose=True,
               desc='DDIM Inverting',
               is_latents=False,
               **cross_attention_kwargs):

        batch_size = image.shape[0]
        device = image.device
        self.scheduler.set_timesteps(num_inference_steps)
        prompt = check_prompt(prompt, batch_size)
        latents = self.image2latents(image) if not is_latents else image
        timesteps = reversed(self.scheduler.timesteps)
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        for i, timestep in enumerate(iteration):
            latents = self.add_noise(latents, timestep, encoder_hidden_states, cross_attention_kwargs)

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
                 lambda_value=0.,
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

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        # encoder_hidden_states, _ = self.pipeline.encode_prompt(prompt, device, 1, True)
        for timestep in iteration:
            latents = self.remove_noise(latents, timestep, encoder_hidden_states, eta,
                                        cross_attention_kwargs=cross_attention_kwargs)
            # calculate gradient
            with torch.enable_grad():
                latents_ = nn.Parameter(latents.detach())
                image = self.latents2image(latents_)
                textile_loss: torch.Tensor = self.textile_loss(image)
                # calculate gradient
                textile_loss.backward()
                # update direction
                latents -= latents_.grad * lambda_value
                for parameter in self.textile_loss.parameters():
                    # clear gradient
                    parameter.grad = None

        if return_latents:
            return latents

        image = self.latents2image(latents)

        return image
