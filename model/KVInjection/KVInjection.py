import torch
import torch.nn as nn
from typing import Union, Optional
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import logging
import torch.nn.functional as F
import os
import sys

sys.path.append(os.getcwd())

from utils.functionals import batch_attention, attention
from model.pipeline import Pipeline


class KVInjectionProccessor:
    # 😤😤😤😤😤
    def __init__(self, num_inference_steps, forward_injection=False, ref_num=None, name=None):
        self.name = name
        self.is_enable = False
        self.num_inference_steps = num_inference_steps
        self.key = [None] * num_inference_steps
        self.value = [None] * num_inference_steps
        self.curr_step = -1
        self.step_range = [0, 50]
        self.high = num_inference_steps
        self.low = -1
        self.forward_injection = forward_injection
        self.is_denoising = True
        self.is_save = False
        self.is_load = False
        if not forward_injection and ref_num is not None:
            raise NotImplementedError
        self.ref_num = ref_num
        self.save_atttn_map = False
        self.attn_map = None

        return
    
    @staticmethod
    def between(x, r):
        return r[0] < x < r[1]

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

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.is_enable:
            if self.is_denoising:
                self.curr_step -= 1
            else:
                self.curr_step += 1
            assert -1 < self.curr_step < self.num_inference_steps

        is_cross = (self.low < self.curr_step < self.high)

        if self.forward_injection:
            query_tgt = query[:-self.ref_num]
            key_tgt = key[:-self.ref_num]
            value_tgt = value[:-self.ref_num]

            query_ref = query[-self.ref_num:]
            key_ref = key[-self.ref_num:]
            value_ref = value[-self.ref_num:]

            if self.is_enable and is_cross and self.is_load:
                # print(1, query_tgt.shape, key_ref.shape)
                if self.save_atttn_map:
                    hidden_states_tgt, self.attn_map = batch_attention(query_tgt, key_ref, value_ref, attn.heads, 
                                                                       return_attn_map=True)
                    self.attn_map = self.attn_map.detach().cpu()
                    torch.cuda.synchronize()
                else:
                    hidden_states_tgt = batch_attention(query_tgt, key_ref, value_ref, attn.heads)
                hidden_states_ref = attention(query_ref, key_ref, value_ref, attn.heads)
                hidden_states = torch.concat([hidden_states_tgt, hidden_states_ref], dim=0)
            elif self.is_load:
                # print(2, query_tgt.shape, key_ref.shape)
                hidden_states_tgt = attention(query_tgt, key_tgt, value_tgt, attn.heads)
                hidden_states_ref = attention(query_ref, key_ref, value_ref, attn.heads)
                hidden_states = torch.concat([hidden_states_tgt, hidden_states_ref], dim=0)
            else:
                # print(3, query.shape, key.shape)
                hidden_states = attention(query, key, value, attn.heads)
        else:
            if self.is_enable and self.is_save:
                self.key[self.curr_step] = key.detach().cpu()
                self.value[self.curr_step] = value.detach().cpu()
                torch.cuda.synchronize()
            elif self.is_enable and is_cross and self.is_load:
                key = self.key[self.curr_step].to(hidden_states.device)
                value = self.value[self.curr_step].to(hidden_states.device)
        
            if is_cross and self.is_load and self.is_enable:
                if self.save_atttn_map:
                    hidden_states, self.attn_map = batch_attention(query, key, value, attn.heads, return_attn_map=True)
                    self.attn_map = self.attn_map.detach().cpu()
                    torch.cuda.synchronize()
                else:
                    hidden_states = batch_attention(query, key, value, attn.heads)
            else:
                hidden_states = attention(query, key, value, attn.heads)
                # print(1, hidden_states.shape)


        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
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

    def enable(self):
        self.is_enable = True
        return
    
    def disable(self):
        self.is_enable = False
        return
    
    def clear(self):
        del self.key
        del self.value
        torch.cuda.empty_cache()

        self.key = [None] * self.num_inference_steps
        self.value = [None] * self.num_inference_steps
        return
    
    def set_range(self, low, high):
        self.low = max(-1, low)
        self.high = high
        return

    def denoise(self, state=True):
        self.is_denoising = state
        self.curr_step = self.num_inference_steps if self.is_denoising else -1
        return
    
    def load(self, state=True):
        self.is_load = state
        return
    
    def save(self, state=False):
        self.is_save = state
        return


class KVInjectionRegister:
    injection_modules = ('down_blocks', 'mid_block', 'up_blocks')
    def __init__(self, 
                 pipeline: Pipeline, 
                 num_inference_steps: int, 
                 attn_type='attn1', 
                 forward_injection=False, 
                 ref_num=None):
        self.pipeline = pipeline
        self.attn_type = attn_type
        self.num_inference_steps = num_inference_steps
        self.forward_injection = forward_injection
        self.ref_num = ref_num
        self.inspect_attention_type = None
        self.inspect_attention_layer = None

        self.processors = dict()
        self.blocks_count = self.register_blocks()

        self.register_idx = {n: list(range(cnt)) for n, cnt in self.blocks_count.items()}

    def register_blocks(self):
        unet = self.pipeline.unet
        
        blocks_count = dict()
        for name, module in unet.named_children():
            find_name = None
            for suffix in self.injection_modules:
                if name.endswith(suffix):
                    find_name = suffix
                    break
            
            if find_name is None:
                continue
                
            blocks_count[find_name] = 0
            self.processors[find_name] = list()
            for name_, module_ in module.named_modules():
                if isinstance(module_, Attention) and name_.endswith(self.attn_type):
                    processor = KVInjectionProccessor(self.num_inference_steps, self.forward_injection, self.ref_num, name=(find_name + name_))
                    processor.disable()
                    self.processors[find_name].append(processor)
                    module_.set_processor(processor)
                    blocks_count[find_name] += 1
        
        return blocks_count
    
    def reset_registers(self, name=None, enable_idx=None):
        if name is None:
            name, enable_idx = self.register_idx.keys(), self.register_idx.values()
        else:
            if isinstance(name, str):
                if name not in self.register_idx.keys():
                    raise NotImplementedError
                self.register_idx[name] = enable_idx
            else:
                for i, n in enumerate(name):
                    if n not in self.register_idx.keys():
                        raise NotImplementedError
                    self.register_idx[n] = enable_idx[i]
        
        return
    
    def set_enable(self, name=None, enable_idx=None):
        self.reset_registers(name, enable_idx)

        for n, cnt in self.blocks_count.items():
            for i in range(cnt):
                if i in self.register_idx[n]:
                    self.processors[n][i].enable()
                else:
                    self.processors[n][i].disable()

        return
    
    def set_range(self, low, high):
        for n, cnt in self.blocks_count.items():
            for i in range(cnt):
                self.processors[n][i].set_range(low, high)
        return
    
    def clear(self):
        for processor_list in self.processors.values():
            for proccessor in processor_list:
                proccessor.clear()
        
        return
    
    def set_ref_num(self, ref_num):
        self.ref_num = ref_num
        for processor_list in self.processors.values():
            for processor in processor_list:
                processor.ref_num = ref_num
        
        return
    
    def denoise(self, state=True):
        for processor_list in self.processors.values():
            for processor in processor_list:
                processor.denoise(state)
        
        return
    
    def save(self, state=True):
        for processor_list in self.processors.values():
            for processor in processor_list:
                processor.save(state)
        
        return

    def load(self, state=True):
        for processor_list in self.processors.values():
            for processor in processor_list:
                processor.load(state)
        
        return
    
    def set_inspect_layer(self, inspect_attention_type: str, inspect_attention_layer: int):
        assert inspect_attention_type in self.injection_modules, f'not support inspect type {inspect_attention_type}'
        assert 0 <= inspect_attention_layer < len(self.processors[inspect_attention_type]), \
            f'{inspect_attention_layer} is not in range {[0, len(self.processors[inspect_attention_type])]}'
        self.inspect_attention_layer = inspect_attention_layer
        self.inspect_attention_type = inspect_attention_type
        self.processors[inspect_attention_type][inspect_attention_layer].save_atttn_map = True

        return
    
    def unset_inspect_layer(self):
        self.inspect_attention_layer = None
        self.inspect_attention_type = None

        for processor_list in self.processors.values():
            for processor in processor_list:
                processor.save_attn_map = False
                del processor.attn_map
                processor.attn_map = None

        torch.cuda.empty_cache()
        return
    
    def get_attn_map(self):
        if self.inspect_attention_layer is None:
            return None

        return self.processors[self.inspect_attention_type][self.inspect_attention_layer].attn_map


if __name__ == '__main__':
    pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4')
    register = KVInjectionRegister(pipeline)
    register.set_invert(['up_blocks', 'mid_block', 'down_blocks'],
                        [list(range(2, 9)), [], []])
    # print(pipeline.unet)
