import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import math
import os
import sys
from typing import Optional
from diffusers.models.attention_processor import Attention

sys.path.append(os.getcwd())

from model.Lora import LoraProcessor


class MultiCrossAttention(nn.Module):
    def __init__(self, in_dims: list[int], attn_dims: int):
        super().__init__()
        
        self.proj_in = [nn.Linear(in_dim, attn_dims) for in_dim in in_dims]
        self.proj_in = nn.ModuleList(self.proj_in)
        self.proj_out = nn.Linear(attn_dims, in_dims[0])

        self.attns = list()
        
        depth = 0
        while 2 ** depth < len(in_dims):
            depth += 1
        self.depth = depth

        cnt = len(in_dims)
        attns_cnt = 0
        for _ in range(self.depth):
            cnt = cnt // 2
            attns_cnt += cnt
            cnt += cnt % 2
        
        for _ in range(attns_cnt):
            attn = Attention(attn_dims, cross_attention_dim=attn_dims, out_dim=attn_dims)
            self.attns.append(attn)
        self.attns = nn.ModuleList(self.attns)

        return

    def forward(self, *args):
        x = args
        assert len(x) == len(self.proj_in), 'input doesn\'t match!'

        # x_new = list()
        # for i in range(len(self.proj_in)):
        #     print(i, x[i].shape, self.proj_in[i].weight.shape)
        #     x_new.append(self.proj_in[i](x[i]))
        # x = x_new
        x = [self.proj_in[i](x[i]) for i in range(len(self.proj_in))]
        
        current = 0
        for _ in range(self.depth, 0, -1):
            x_new = list()
            for i in range(0, len(x) - 1, 2):
                a = self.attns[current + i](x[i], x[i + 1])
                x_new.append(a)
            if len(x) % 2:
                x_new.append(x[-1])
            current += int(math.sqrt(len(x)))
            x = x_new
        
        # print(len(x))
        assert len(x) == 1
        return self.proj_out(x[0])


class FeatureAugmentor(nn.Module):
    def __init__(self, in_dims: list[int], attn_dims: int):
        super().__init__()
        assert len(in_dims) == 4

        self.multi_attn = MultiCrossAttention(in_dims, attn_dims)

        return
        
    def forward(self, hidden_states: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor, latents_ref: torch.Tensor):
        batch_size = hidden_states.shape[0]
        residual = hidden_states
        if latents.ndim == 4:
            latents = latents.view(*latents.shape[:2], -1).transpose(-1, -2)
        if mask.ndim == 4:
            mask = mask.view(*mask.shape[:2], -1).transpose(-1, -2)
        if mask.shape[0] != batch_size:
            mask = torch.concat([mask] * batch_size)
        if latents_ref.ndim == 4:
            latents_ref = latents_ref.view(*latents_ref.shape[:2], -1).transpose(-1, -2)
        if latents_ref.ndim != batch_size:
            latents_ref = torch.concat([latents_ref] * batch_size)
        
        hidden_states = self.multi_attn(hidden_states, latents, mask, latents_ref)

        return residual + hidden_states


class FeatureAugmentProcessor(LoraProcessor):
    def __init__(self, augmentor: FeatureAugmentor, attn: Attention, lora_ratio, device='cuda'):
        super().__init__(attn, lora_ratio)
        self.augmentor = augmentor
        self.hidden_states = None
        self.is_invert = True

        self.device = device
        self.module_dict['augmentor'] = self.augmentor
        return

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        attention_mask_: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None,
        latents_ref: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if self.is_invert:
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            self.hidden_states = encoder_hidden_states.cpu()
        else:
            # self.hidden_states = self.hidden_states.to(self.device)
            # assert latents is not None and mask is not None and latents_ref is not None
            encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            # encoder_hidden_states = self.hidden_states.to(hidden_states.device)
            # encoder_hidden_states = self.augmentor(self.hidden_states, latents, mask, latents_ref)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if self.is_enable:
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(encoder_hidden_states)
            value = value + self.lora_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if attention_mask_ is not None:
            attention_mask = attention_mask_[0, 0, :, :]
            attention_mask = attention_mask.view(1, 1, *attention_mask.shape[-2:])
            attention_mask = nn.functional.interpolate(attention_mask, [int(math.sqrt(query.shape[2]))] * 2)
            attention_mask = attention_mask.view(1, 1, 1, -1)

        if self.is_enable and attention_mask_ is not None:
            attn_score = torch.matmul(query, key.transpose(-1, -2)) / (head_dim ** 0.5)
            attn_score = attn_score + (attention_mask * -100)
            attn_score = torch.softmax(attn_score, dim=-1)
            hidden_states = torch.matmul(attn_score, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        if self.is_enable:
            hidden_states = hidden_states + self.lora_out(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def invert(self):
        self.is_invert = True
        return

    def inference(self):
        self.is_invert = False
        return

    def parameters(self):
        param = list()
        modules = [self.lora_q, self.lora_k, self.lora_v, self.lora_out, self.augmentor]
        for module in modules:
            param.extend(list(module.parameters()))

        return param


class FeatureAugmentorRegister:
    block_types = ('down_blocks', 'mid_blocks', 'up_blocks')
    def __init__(self, unet: UNet2DConditionModel, in_dims: list[int], lora_ratio=1.0):
        super().__init__()
        self.lora_ratio = lora_ratio
        self.block_modules: dict[str: nn.Module] = dict()
        self.processors = dict()

        for name, module in unet.named_children():
            if name.endswith(self.block_types):
                self.block_modules[name] = module
        
        for name, module in self.block_modules.items():
            for name_, module_ in module.named_modules():
                if isinstance(module_, Attention) and name_.endswith('attn1'):
                    augmentor = FeatureAugmentor([module_.out_dim] + in_dims, module_.out_dim)
                    processor = FeatureAugmentProcessor(augmentor, module_, lora_ratio)
                    self.processors[name + name_] = processor
                    module_.set_processor(processor)
        
        return

    def save(self, fp: str):
        weights = {name: module.state_dict() for name, module in self.processors.items()}
        config = {'lora_ratio': self.lora_ratio}
        state_dict = {'weights': weights, 'config': config}
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save(state_dict, fp)

        return

    def load(self, fp: str):
        state_dict = torch.load(fp, weights_only=False)
        config = state_dict['config']
        lora_ratio = config['lora_ratio']
        assert lora_ratio == self.lora_ratio, 'lora ratio do not match'
        weights = state_dict['weights']
        
        for name, module in self.processors.items():
            module.load_state_dict(weights[name])
        
        return

    def parameters(self):
        param = list()
        for module in self.processors.values():
            param.extend(list(module.parameters()))

        return param
    
    def invert(self):
        for module in self.processors.values():
            module.invert()
        
        return
    
    def inference(self):
        for module in self.processors.values():
            module.inference()

        return
    
    def to(self, device):
        for module in self.processors.values():
            module.to(device)
        
        return
    
    def enable(self):
        for module in self.processors.values():
            module.enable()
        return
    
    def disable(self):
        for module in self.processors.values():
            module.disable()

        return
    
    def train(self):
        for module in self.processors.values():
            module.train()


if __name__ == '__main__':
    multi_attn = MultiCrossAttention([768, 768, 768, 768], 1024)
    x = torch.rand([1, 5, 768])
    y = torch.rand([1, 6, 768])
    z = torch.rand([1, 7, 768])
    f = torch.rand([1, 7, 768])
    g = multi_attn(x, y, z, f)
    print(g.shape)
