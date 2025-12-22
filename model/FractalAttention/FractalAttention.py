import torch
import torch.nn as nn
import os
import sys
import math
import time
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from typing import Optional, Tuple
import torch.nn.functional as F
import math


sys.path.append(os.getcwd())

from utils.functionals import attention, count_parameters
from model.Lora import Lora


class FractalAttentionSubModule:
    def __init__(self, patch_size: int, patch_num: int):
        self.patch_size = patch_size
        self.patch_num = patch_num

        return
    
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor=None):
        b, d = query.shape[0], query.shape[-1]

        query = self.patchify(query)
        key = self.patchify(key)
        value = self.patchify(value)

        query = self.attention(query, key, value, attention_mask=attention_mask)

        query =query.reshape(b, -1, d, self.patch_size ** 2).transpose(-1, -2)
        key = key.reshape(b, -1, d, self.patch_size ** 2).transpose(-1, -2)
        value = value.reshape(b, -1, d, self.patch_size ** 2).transpose(-1, -2)

        return query, key, value
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor=None):
        b, l1, d = q.shape
        l2 = k.shape[1]
        heads = self.patch_size ** 2
        # print(q.shape, k.shape, v.shape)
        q = q.reshape(b, l1, d // heads, heads).permute(0, 3, 1, 2)
        k = k.reshape(b, l2, d // heads, heads).permute(0, 3, 1, 2)
        v = v.reshape(b, l2, d // heads, heads).permute(0, 3, 1, 2)

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (d // heads) ** 0.5
        if attention_mask is not None:
            attn_score[attention_mask.transpose(-1, -2)] = -10000.

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        # attn_score = torch.softmax(attn_score, dim=-1)
        # out = torch.matmul(attn_score, v)
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(b, l1, d)

        return out
    
    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 3:
            b, l, d = img.shape
            w = int(math.sqrt(img.shape[1]))
            img = img.transpose(-1, -2).reshape(b, d, w, w)
        else:
            b, _, _, d = img.shape
            img = img.transpose(-1, -2)
            w = int(math.sqrt(img.shape[-1]))
            img = img.reshape(-1, d, w, w)

        b, c, h, w = img.shape
        img = img.unfold(-2, self.patch_size, self.patch_size)
        patches = img.unfold(-2, self.patch_size, self.patch_size)
        assert patches.shape[2] == self.patch_num

        # current shape is [b, c, n, n, p, p]
        patches = patches.reshape(b, c, self.patch_num ** 2, self.patch_size ** 2)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.patch_num ** 2, c * self.patch_size ** 2)

        return patches
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, d, h, w = patches.shape

        img = patches.reshape(b, -1, self.patch_num ** 2, d, self.patch_size, self.patch_size)
        img = img.permute(0, 1, 3, 2, 4, 5)
        img = img.reshape(*img.shape[:3], self.patch_num, self.patch_num, self.patch_size, self.patch_size)
        img = img.transpose(-2, -3).reshape(*img.shape[:3], self.patch_num * self.patch_size, self.patch_num * self.patch_size)
        
        return img
        

class FractalAttention(nn.Module):
    def __init__(self, 
                 depth: int,
                 patch_size: list[int],
                 patch_num: list[int],
                 module: Attention=None,
                 in_dim: int=None,
                 inner_dim: int=None,
                 out_dim: int=None):
        super().__init__()

        self.patch_size = patch_size
        self.patch_num = patch_num
        self.depth = depth
        self.heads = 8

        # only top layer perform project
        if module is None:
            assert in_dim is not None and inner_dim is not None and out_dim is not None
            # self.null_token = nn.Parameter(torch.zeros(1, 1, inner_dim))
            # torch.nn.init.normal_(self.null_token, 2e-1)

            self.to_q = nn.Linear(in_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(in_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(in_dim, inner_dim, bias=False)

            self.lora_q = Lora(in_dim, inner_dim, bias=False)
            self.lora_k = Lora(in_dim, inner_dim, bias=False)
            self.lora_v = Lora(in_dim, inner_dim, bias=False)
            # print(in_dim, in_dim // self.scale_factor)
            self.to_out = nn.Linear(inner_dim, out_dim, bias=True)
        else:
            self.init_from_pretrained(module)

        depth = self.depth
        
        self.sub_modules = list([
            FractalAttentionSubModule(patch_size[i], patch_num[i]) for i in range(depth)
        ])

        return

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        attention_mask = kwargs.get('fractal_mask', None)

        residual = hidden_states

        # if attention_mask is not None:
        #     image_size = int(math.sqrt(hidden_states.shape[1]))
        #     attention_mask = torch.nn.functional.interpolate(attention_mask, [image_size, image_size])
        #     attention_mask = attention_mask.view(*attention_mask.shape[:2], -1).transpose(-1, -2)
        #     attention_mask = (attention_mask > 1e-2).bool()

        #     hidden_states[attention_mask.squeeze(-1)] = self.null_token

        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query = self.to_q(hidden_states) + self.lora_q(hidden_states)
        key = self.to_k(encoder_hidden_states) + self.lora_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states) + self.lora_v(encoder_hidden_states)

        for module in self.sub_modules:
            if attention_mask is not None:
                attention_mask = module.patchify(attention_mask)
            query, key, value = module(query, key, value, attention_mask)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1)

        hidden_states = query
        b, n, p, d = hidden_states.shape
        hidden_states = hidden_states.transpose(-1, -2).reshape(b, n, d, 1, 1)

        for module in reversed(self.sub_modules):
            hidden_states = module.unpatchify(hidden_states)

        hidden_states = hidden_states.squeeze(1).reshape(b, d, -1).transpose(-1, -2)

        hidden_states = self.to_out(hidden_states) + self.lora_out(hidden_states)

        return hidden_states + residual
    
    @torch.no_grad()
    def init_from_pretrained(self, module: Attention, lora_ratio=1.):
        self.heads = module.heads

        use_bias = module.to_q.bias is not None
        
        # self.null_token = nn.Parameter(torch.zeros(1, 1, module.inner_dim))
        # torch.nn.init.normal_(self.null_token, 2e-1)

        self.to_q = nn.Linear(module.query_dim, module.inner_dim, bias=use_bias)
        self.to_k = nn.Linear(module.cross_attention_dim, module.inner_kv_dim, bias=use_bias)
        self.to_v = nn.Linear(module.cross_attention_dim, module.inner_kv_dim, bias=use_bias)
        self.to_out = nn.Linear(module.inner_dim, module.out_dim, True)

        self.lora_q = Lora(module.query_dim, module.inner_dim, bias=use_bias, lora_ratio=lora_ratio)
        self.lora_k = Lora(module.cross_attention_dim, module.inner_kv_dim, bias=use_bias, lora_ratio=lora_ratio)
        self.lora_v = Lora(module.cross_attention_dim, module.inner_kv_dim, bias=use_bias, lora_ratio=lora_ratio)
        self.lora_out = Lora(module.inner_dim, module.out_dim, True, lora_ratio=lora_ratio)

        self.to_q.weight.copy_(module.to_q.weight)
        self.to_k.weight.copy_(module.to_k.weight)
        self.to_v.weight.copy_(module.to_v.weight)

        if use_bias:
            self.to_q.bias.copy_(module.to_q.bias)
            self.to_k.bias.copy_(module.to_k.bias)
            self.to_v.bias.copy_(module.to_v.bias)

        self.to_out.weight.copy_(module.to_out[0].weight)
        self.to_out.bias.copy_(module.to_out[0].bias)

        return
    
    def parameters_(self):
        modules = [self.lora_q, self.lora_k, self.lora_v, self.lora_out]

        params = []

        for module in modules:
            params.extend(list(module.parameters()))

        return params
    
    def train_(self):
        trainable_modules = [self.lora_q, self.lora_k, self.lora_v, self.lora_out]
        untrainable_modules = [self.to_q, self.to_k, self.to_v, self.to_out]

        params = []

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

        for module in untrainable_modules:
            for param in module.parameters():
                param.requires_grad = False

        return params


class FractalAttnProcessor:
    def __init__(self, patch_num: int=2, max_depth=4, bias=False):
        self.patch_num = patch_num
        self.fractal_attn = None
        self.proj_in = None
        self.proj_out = None
        self.bias = bias
        self.max_depth = max_depth

        return
    
    def get_depth(self, image_size: int):
        depth = int(math.log(image_size, self.patch_num))
        if depth > self.max_depth:
            self.patch_num *= 2
            depth = self.get_depth(image_size)

        return depth
    
    def to(self, device):
        self.proj_in.to(device)
        self.proj_out.to(device)
        self.fractal_attn.to(device)

        return
    
    def parameters(self):
        params = list()
        params = list(self.proj_in.parameters()) + list(self.proj_out.parameters()) + list(self.fractal_attn.parameters())
        return params
    
    def train(self):
        self.proj_in.train()
        self.proj_out.train()
        self.fractal_attn.train()

        return
    
    def state_dict(self):
        state_dict = {
            'proj_in': self.proj_in.state_dict(),
            'proj_out': self.proj_out.state_dict(),
            'fractal_attn': self.fractal_attn.state_dict()
        }

        return state_dict
    
    def load(self, state_dict):
        self.proj_in.load_state_dict(state_dict['proj_in'])
        self.proj_out.load_state_dict(state_dict['proj_out'])
        self.fractal_attn.load_state_dict(state_dict['fractal_attn'])

        return
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.fractal_attn is None:
            # print(query.shape)
            seq_len = query.shape[-2]
            depth = self.get_depth(int(math.sqrt(seq_len)))
            self.fractal_attn = FractalAttention(depth, attn.inner_dim // 4, self.patch_num)
            self.proj_in = nn.Linear(inner_dim, inner_dim // 4, self.bias)
            self.proj_out = nn.Linear(inner_dim // 4, inner_dim, self.bias)
            # print(hidden_states.shape, encoder_hidden_states.shape)

        fractal_hidden_states = self.proj_in(hidden_states)
        fractal_encoder_hidden_states = self.proj_in(encoder_hidden_states)
        fractal_out = self.fractal_attn(fractal_hidden_states, fractal_encoder_hidden_states, attn.heads)
        fractal_out = self.proj_out(fractal_out)
        
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # add fractall attn values
        hidden_states = hidden_states + fractal_out

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

class FractalAttnRegister:
    sub_modules = ('down_blocks', 'mid_block', 'up_blocks')
    def __init__(self,
                 unet: UNet2DConditionModel,
                 register_modules: str | list[str],
                 register_layers: list[int],
                 attn_type='attn1'):
        if isinstance(register_modules, str):
            register_modules = [register_modules]
        
        self.layer_count = dict()
        self.processors = dict()

        self.registers = {module: list() for module in self.sub_modules}
        for module in register_modules:
            self.registers[module] = register_layers

        self.attn_type = attn_type

        self.count_layers(unet)
        latents = torch.rand([1, 4, 64, 64])
        encoder_hidden_states = torch.rand([1, 77, 768])
        unet(latents, 0, encoder_hidden_states)

        return
    
    def count_layers(self, unet: UNet2DConditionModel, patch_num=2, max_depth=4, bias=False):
        for module_type in self.sub_modules:
            self.layer_count[module_type] = 0
            self.processors[module_type] = list()
            for name, module in unet.named_children():
                if not name.endswith(module_type):
                    continue
                for name_, module_ in module.named_modules():
                    if isinstance(module_, Attention) and name_.endswith(self.attn_type):
                        patch_size = 8
                        if self.layer_count[module_type] % 3 == 0:
                            patch_size *= 2

                        self.layer_count[module_type] += 1
                        if self.layer_count[module_type] in self.registers[module_type]:
                            # 一共有 3*3个 attention
                            depth  = int(math.log(patch_size / 8, 2.)) + 1
                            pass

        return
    
    def count_parameters(self):
        params = 0
        for module_type in self.sub_modules:
            for processor in self.processors[module_type]:
                if isinstance(processor, FractalAttnProcessor):
                    params += count_parameters(processor.fractal_attn)
        
        return params
    
    def parameters(self):
        params = list()
        for module_type in self.sub_modules:
            for processor in self.processors[module_type]:
                if isinstance(processor, FractalAttnProcessor):
                    params.extend(processor.parameters())

        return params
    
    def to(self, device):
        for module_type in self.sub_modules:
            for processor in self.processors[module_type]:
                processor.to(device)

        return
    
    def train(self):
        for module_type in self.sub_modules:
            for processor in self.processors[module_type]:
                processor.train()

        return
    
    def save(self, fp: str):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        state_dict = dict()

        state_dict['registers'] = self.registers
        state_dict['state_dict'] = dict()

        for module_type in self.sub_modules:
            state_dict['state_dict'][module_type] = list()
            for processor in self.processors[module_type]:
                state_dict['state_dict'][module_type].append(processor.state_dict())
            
        torch.save(state_dict, fp)
        return
    
    def load(self, fp: str):
        state_dict = torch.load(fp, weights_only=False)
        
        self.registers = state_dict['registers']
        
        for module_type in self.sub_modules:
            sd_list = state_dict['state_dict'][module_type]
            for i, processor in enumerate(self.processors[module_type]):
                processor.load(sd_list[i])

        return


if __name__ == '__main__':
    from utils.functionals import count_parameters
    from model.pipeline import Pipeline

    device = 'cuda'
    pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4')
    fractal_attn = FractalAttention(3, 768, 2)
    # register = FractalAttnRegister(pipeline.unet, 'up_blocks', list(range(3, 9)))
    # print(register.layer_count)
    # print(register.count_parameters() / (1024 ** 2) * 4)

    # fractal_attn = torch.jit.script(fractal_attn)
    # print(fractal_attn.code)
    fractal_attn.to(device)
    pipeline.to(device)
    a = torch.rand(1, 4096, 768, device=device)
    b = torch.rand(1, 4096, 768, device=device)

    t = time.time()
    for i in range(100):
        fractal_attn(a, b)
        
    print((time.time() - t) / 100)

    # register.save('./register.ckpt')
    # register.load('./register.ckpt')
