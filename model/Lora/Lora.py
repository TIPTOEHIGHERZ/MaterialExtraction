import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from typing import Optional
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP

from model.Attention import LocalAttention, PatchAttention, FractalAttention


class Lora(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool, lora_ratio=1.0, device='cpu', dtype=None):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.mid_channels = int(math.sqrt(in_channels * out_channels) * lora_ratio)
        self.out_channels = out_channels

        factory_kwargs = {"device": device, "dtype": dtype}
        self.A = nn.Parameter(torch.empty([self.mid_channels, self.in_channels], **factory_kwargs))
        self.B = nn.Parameter(torch.empty([self.out_channels, self.mid_channels], **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        return
    
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in_A, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.A)
            fan_in_B, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.B)
            fan_in = (fan_in_A + fan_in_B) / 2
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return

    def forward(self, tensor):
        weights = torch.matmul(self.B, self.A)
        return F.linear(tensor, weights, self.bias)


class LoraProcessor(nn.Module):
    def __init__(self, attn: Attention, lora_ratio, enable_attention_mask=False, replace_attention_score=False, batch_attn=False):
        super().__init__()

        bias = False if attn.to_q.bias is None else True
        q_in = attn.to_q.in_features
        q_out = attn.to_q.out_features
        self.lora_q = Lora(q_in, q_out, bias, lora_ratio)

        bias = False if attn.to_k.bias is None else True
        k_in = attn.to_k.in_features
        k_out = attn.to_k.out_features
        self.lora_k = Lora(k_in, k_out, bias, lora_ratio)

        bias = False if attn.to_v.bias is None else True
        v_in = attn.to_v.in_features
        v_out = attn.to_v.out_features
        self.lora_v = Lora(v_in, v_out, bias, lora_ratio)

        bias = False if attn.to_out[0].bias is None else True
        out_in = attn.to_out[0].in_features
        out_out = attn.to_out[0].out_features
        self.lora_out = Lora(out_in, out_out, bias, lora_ratio)

        self.is_enable = True
        self.enable_attention_mask = enable_attention_mask
        self.replace_attention_score = replace_attention_score
        self.batch_attn = batch_attn

        self.module_dict = {'lora_q': self.lora_q, 'lora_k': self.lora_k, 'lora_v': self.lora_v, 'lora_out': self.lora_out}
        # self.modules = [self.lora_q, self.lora_k, self.lora_v, self.lora_out]

        return
    
    def __call__(
        self,
        *args,
        fractal_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        attention_mask_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # batch_size, sequence_length, _ = (
        #     hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # )
        batch_size = hidden_states.shape[0]

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attention_mask = attention_mask if self.enable_attention_mask else None

        if self.is_enable:
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(encoder_hidden_states)
            value = value + self.lora_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            key_ = attn.to_k(hidden_states)
            value_ = attn.to_v(hidden_states)
            if self.is_enable:
                key_ = key_ + self.lora_k(hidden_states)
                value_ = value_ + self.lora_v(hidden_states)

            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ = value_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key_ = attn.norm_k(key_)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if attention_mask_ is not None:
            attention_mask = attention_mask_[0, 0, :, :]
            attention_mask = attention_mask.view(1, 1, *attention_mask.shape[-2:])
            attention_mask = nn.functional.interpolate(attention_mask, [int(math.sqrt(query.shape[2]))] * 2)
            attention_mask = attention_mask.view(1, 1, -1, 1)

        if self.is_enable and attention_mask_ is not None:
            attn_score = torch.matmul(query, key.transpose(-1, -2)) / (head_dim ** 0.5)
            attn_score = attn_score + (attention_mask * -100)
            attn_score = torch.softmax(attn_score, dim=-1)
            hidden_states = torch.matmul(attn_score, value)
        else:
            if attention_mask is not None:
                attention_mask = attention_mask.to(query.dtype)
                
                if attention_mask.shape[-1] ** 2 != query.shape[-2]:
                    if attention_mask.ndim == 3:
                        attention_mask = attention_mask.unsqueeze(1)
                    elif attention_mask.ndim == 4 and attention_mask.shape[-1] != attention_mask.shape[-2]:
                        raise RuntimeError(f'attention mask should have same resolution in (h, w) instead of {attention_mask.shape}')
                    elif attention_mask.ndim != 4:
                        raise NotImplementedError(f'attention mask has dim of {attention_mask.ndim}')
                    
                    attention_mask = F.interpolate(attention_mask, [int(math.sqrt(query.shape[-2]))] * 2)
                
                # attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-2)
                # attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)

                attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-1)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            
            if attention_mask is not None:
                hidden_states_ = F.scaled_dot_product_attention(
                    query, key_, value_, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                hidden_states = hidden_states * attention_mask + hidden_states_ * (1 - attention_mask)

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

        return hidden_states.contiguous()
    
    def enable(self):
        self.is_enable = True
        return
    
    def disable(self):
        self.is_enable = False
        return


class LocalLoraProcessor(LoraProcessor):
    def __init__(
        self, 
        attn: Attention, 
        lora_ratio, 
        enable_attention_mask=False, 
        replace_attention_score=False,
        patch_size=4,
        step_size=2
    ):
        super().__init__(attn, lora_ratio, enable_attention_mask, replace_attention_score)
        hidden_dim = attn.to_q.out_features
        self.local_attn = LocalAttention(hidden_dim, patch_size, step_size)

        return
    
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
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

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attention_mask = attention_mask if self.enable_attention_mask else None

        if self.is_enable:
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(encoder_hidden_states)
            value = value + self.lora_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            key_ = attn.to_k(hidden_states)
            value_ = attn.to_v(hidden_states)
            if self.is_enable:
                key_ = key_ + self.lora_k(hidden_states)
                value_ = value_ + self.lora_v(hidden_states)

            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ = value_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key_ = attn.norm_k(key_)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        if attention_mask is not None:
            attention_mask = attention_mask.to(query.dtype)
            
            if attention_mask.shape[-1] ** 2 != query.shape[-2]:
                if attention_mask.ndim == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.ndim == 4 and attention_mask.shape[-1] != attention_mask.shape[-2]:
                    raise RuntimeError(f'attention mask should have same resolution in (h, w) instead of {attention_mask.shape}')
                elif attention_mask.ndim != 4:
                    raise NotImplementedError(f'attention mask has dim of {attention_mask.ndim}')
                
                attention_mask = F.interpolate(attention_mask, [int(math.sqrt(query.shape[-2]))] * 2)
            
            # attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-2)
            # attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)

            attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-1)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # perform local-attention
        hidden_states = hidden_states + self.local_attn(hidden_states, hidden_states, hidden_states)
        # hidden_states = self.local_attn(hidden_states, hidden_states, hidden_states)
        
        if attention_mask is not None:
            hidden_states_ = F.scaled_dot_product_attention(
                query, key_, value_, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states * attention_mask + hidden_states_ * (1 - attention_mask)

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
        return hidden_states.contiguous()


class PatchLoraProcessor(LoraProcessor):
    def __init__(
        self, 
        attn: Attention, 
        lora_ratio, 
        enable_attention_mask=False, 
        replace_attention_score=False,
        patch_size=4,
        step_size=2
    ):
        super().__init__(attn, lora_ratio, enable_attention_mask, replace_attention_score)
        hidden_dim = attn.to_q.out_features
        self.patch_attn = PatchAttention(hidden_dim, patch_size, step_size)

        return
    
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
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

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attention_mask = attention_mask if self.enable_attention_mask else None

        if self.is_enable:
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(encoder_hidden_states)
            value = value + self.lora_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            key_ = attn.to_k(hidden_states)
            value_ = attn.to_v(hidden_states)
            if self.is_enable:
                key_ = key_ + self.lora_k(hidden_states)
                value_ = value_ + self.lora_v(hidden_states)

            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ = value_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key_ = attn.norm_k(key_)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        if attention_mask is not None:
            attention_mask = attention_mask.to(query.dtype)
            
            if attention_mask.shape[-1] ** 2 != query.shape[-2]:
                if attention_mask.ndim == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.ndim == 4 and attention_mask.shape[-1] != attention_mask.shape[-2]:
                    raise RuntimeError(f'attention mask should have same resolution in (h, w) instead of {attention_mask.shape}')
                elif attention_mask.ndim != 4:
                    raise NotImplementedError(f'attention mask has dim of {attention_mask.ndim}')
                
                attention_mask = F.interpolate(attention_mask, [int(math.sqrt(query.shape[-2]))] * 2)
            
            # attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-2)
            # attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)

            attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-1)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # perform patch-attention
        # hidden_states = hidden_states + self.patch_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = self.patch_attn(hidden_states, hidden_states, hidden_states)
        
        if attention_mask is not None:
            hidden_states_ = F.scaled_dot_product_attention(
                query, key_, value_, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states * attention_mask + hidden_states_ * (1 - attention_mask)

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
        return hidden_states.contiguous()


class FractalLoraProcessor(LoraProcessor):
    def __init__(
        self, 
        attn: Attention,
        lora_ratio, 
        enable_attention_mask=False,
        replace_attention_score=False,
        patch_num=4,
        step_ratio=0.5
    ):
        super().__init__(attn, lora_ratio, enable_attention_mask, replace_attention_score)
        hidden_dim = attn.to_q.out_features
        self.fractal_attn = FractalAttention(hidden_dim, patch_num, step_ratio)

        return
    
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
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

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attention_mask = attention_mask if self.enable_attention_mask else None

        if self.is_enable:
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(encoder_hidden_states)
            value = value + self.lora_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            key_ = attn.to_k(hidden_states)
            value_ = attn.to_v(hidden_states)
            # if self.is_enable:
            #     key_ = key_ + self.lora_k(hidden_states)
            #     value_ = value_ + self.lora_v(hidden_states)

            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ = value_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key_ = attn.norm_k(key_)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        if attention_mask is not None:
            attention_mask = attention_mask.to(query.dtype)
            
            if attention_mask.shape[-1] ** 2 != query.shape[-2]:
                if attention_mask.ndim == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.ndim == 4 and attention_mask.shape[-1] != attention_mask.shape[-2]:
                    raise RuntimeError(f'attention mask should have same resolution in (h, w) instead of {attention_mask.shape}')
                elif attention_mask.ndim != 4:
                    raise NotImplementedError(f'attention mask has dim of {attention_mask.ndim}')
                
                attention_mask = F.interpolate(attention_mask, [int(math.sqrt(query.shape[-2]))] * 2)
            
            # attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-2)
            # attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)

            attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-1)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # perform patch-attention
        hidden_states = hidden_states + self.fractal_attn(hidden_states, hidden_states, hidden_states)
        # hidden_states = self.fractal_attn(hidden_states, hidden_states, hidden_states)
        
        if attention_mask is not None:
            hidden_states_ = F.scaled_dot_product_attention(
                query, key_, value_, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states * attention_mask + hidden_states_ * (1 - attention_mask)

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
        return hidden_states.contiguous()


class CrossAttnProcessor:
    def __init__(
        self,
        attn: Attention,
        enable_attention_mask=False,
        *args,
        **kwargs
    ):
        self.enable_attention_mask = enable_attention_mask
        return
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        attention_mask = attention_mask if self.enable_attention_mask else None

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            key_ = attn.to_k(hidden_states)
            value_ = attn.to_v(hidden_states)
            # if self.is_enable:
            #     key_ = key_ + self.lora_k(hidden_states)
            #     value_ = value_ + self.lora_v(hidden_states)

            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ = value_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key_ = attn.norm_k(key_)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        if attention_mask is not None:
            attention_mask = attention_mask.to(query.dtype)
            
            if attention_mask.shape[-1] ** 2 != query.shape[-2]:
                if attention_mask.ndim == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                elif attention_mask.ndim == 4 and attention_mask.shape[-1] != attention_mask.shape[-2]:
                    raise RuntimeError(f'attention mask should have same resolution in (h, w) instead of {attention_mask.shape}')
                elif attention_mask.ndim != 4:
                    raise NotImplementedError(f'attention mask has dim of {attention_mask.ndim}')
                
                attention_mask = F.interpolate(attention_mask, [int(math.sqrt(query.shape[-2]))] * 2)
            
            # attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-2)
            # attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)

            attention_mask = attention_mask.reshape(*attention_mask.shape[:2], -1).unsqueeze(-1)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        
        if attention_mask is not None:
            hidden_states_ = F.scaled_dot_product_attention(
                query, key_, value_, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states * attention_mask + hidden_states_ * (1 - attention_mask)

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

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states.contiguous()
    

class LoraRegister(nn.Module):
    attn_type = {
        'vanilla': LoraProcessor, 
        'local': LocalLoraProcessor,
        'patch': PatchLoraProcessor,
        'fractal': FractalLoraProcessor,
        'cross_attn': CrossAttnProcessor
    }

    def __init__(
            self, 
            module: UNet2DConditionModel, 
            name_list=['attn1', 'attn2'], 
            enable_attention_mask=[False, False],
            replace_attention_score=[False, False],
            attn_types=['local', 'vanilla'],
            lora_ratio=1.0,
            register_range=[None, None],
            attn_classes=[Attention, Attention],
            mute=True,
            **processor_kwars
        ):
        super().__init__()

        # self.module = module
        self.lora_ratio = lora_ratio
        self.name_list = name_list
        self.loras: nn.ModuleDict = nn.ModuleDict()
        self.processors = dict()
        self.attn_classes = attn_classes
        self.mute = mute

        self.block_cnt = dict()

        self.register(
            module,
            name_list, 
            enable_attention_mask, 
            replace_attention_score, 
            attn_types, 
            lora_ratio,
            register_range,
            **processor_kwars
        )

        return
    
    def unfrozen(self):
        for param in self.parameters():
            param.requires_grad_(True)
        
        return
    
    def register(
            self, 
            module: UNet2DConditionModel,
            name_list=['attn1', 'attn2'],
            enable_attention_mask=[False, False],
            replace_attention_score=[False, False],
            attn_types=['local', 'vanilla'],
            lora_ratio=1.0,
            register_range=[None, None],
            **processor_kwars,
        ):
        assert len(name_list) == len(enable_attention_mask)

        for name, enable_mask, replace_score, attn_type, regist_range, attn_class in zip(
            name_list,
            enable_attention_mask, 
            replace_attention_score, 
            attn_types, 
            register_range,
            self.attn_classes
        ):
            assert attn_type in self.attn_type.keys(), f'{attn_type} is not supported!'

            cnt = 0
            for name_, module_ in module.named_modules():
                if isinstance(module_, attn_class) and name_.endswith(name):
                    cnt += 1

                    if regist_range is not None and not cnt in range(regist_range[0], regist_range[1]):
                        continue

                    processor_type = self.attn_type[attn_type]
                    processor = processor_type(
                        module_, 
                        lora_ratio=lora_ratio, 
                        enable_attention_mask=enable_mask,
                        replace_attention_score=replace_score,
                        **processor_kwars
                    )

                    block_type = name_.split('.')[0]
                    if block_type in self.block_cnt.keys():
                        self.block_cnt[block_type] += 1
                    else:
                        self.block_cnt[block_type] = 1

                    name_ = name_.replace('.', '_')
                    try:
                        self.loras[name_] = processor
                    except Exception as e:
                        self.processors[name_] = processor
                    
                    module_.set_processor(processor)
        return
    
    def count_attns(self) -> int:
        name = self.name_list[0]

        cnt = 0
        for key in self.loras.keys():
            if key.endswith(name):
                cnt += 1
        
        return cnt

    @classmethod
    def from_pretrained(cls, module: UNet2DConditionModel, pretrained_path: str):
        state_dict = torch.load(pretrained_path, weights_only=False)
        config = state_dict['config']
        lora_ratio = config['lora_ratio']
        name = config['name']
        register = cls(module, name, lora_ratio)
        weights = state_dict['weights']
        
        # load pretrained weights
        for name, module in register.loras.items():
            module.load_state_dict(weights[name])

        return register
    
    def save(self, fp: str):
        if self.is_ddp:
            weights = {name: module.module.state_dict() for name, module in self.loras.items()}
        else:
            weights = {name: module.state_dict() for name, module in self.loras.items()}

        config = {'lora_ratio': self.lora_ratio, 'name': self.name_list}
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
        
        for name, module in self.loras.items():
            module.load_state_dict(weights[name])
        
        return
    
    def enable(self):
        for module in self.loras.values():
            module.enable()

        return
    
    def disable(self):
        for module in self.loras.values():
            module.disable()

        return
    

if __name__ == '__main__':
    unet = UNet2DConditionModel.from_pretrained('./pretrained/stable-diffusion-v1-4/unet')
    lora_register = LoraRegister(unet)
    a = torch.rand([1, 4, 64, 64])
    c = torch.rand([1, 77, 768])
    b = unet(a, 1, c).sample
    print(b.shape)
