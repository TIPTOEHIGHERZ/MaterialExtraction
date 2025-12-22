import os
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import Transformer2DModel, CrossAttnUpBlock2D

# unet relevants
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, deprecate

import inspect
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Dict, Any
from diffusers import UNet2DConditionModel
import math
import time

from .UNetRef import UNet2DConditionModelReference
from .UNetMain import UNet2DConditionModelMain

from .attn_ref import (
    BasicTransformerBlockReference, 
    CrossAttnDownBlock2DReference, 
    CrossAttnUpBlock2DReference, 
    UNetMidBlock2DCrossAttnReference
)

from utils.functionals import try_load
from utils.functionals import getattr_recursive


class Adapter(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, ratio=4.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj_in = nn.Linear(in_channels, mid_channels)

        self.mlp = nn.Sequential(
            nn.Linear(mid_channels, int(mid_channels * ratio)),
            nn.SiLU(),
            nn.Linear(int(mid_channels * ratio), mid_channels)
        )

        self.attention = Attention(
            query_dim=mid_channels,
            cross_attention_dim=None,
            heads=8,
        )

        self.proj_out = nn.Linear(mid_channels, out_channels)

        return 
    
    def forward(self, hidden_states: torch.Tensor):
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.reshape(b, c, h * w)
        hidden_states = hidden_states.transpose(-1, -2)

        hidden_states = self.proj_in(hidden_states)

        hidden_states = self.mlp(hidden_states) + hidden_states
        hidden_states = self.attention(hidden_states) + hidden_states
        hidden_states = self.proj_out(hidden_states)

        return hidden_states
    

class FeatureAdapter(nn.Module):
    def __init__(
            self, 
            unet: UNet2DConditionModelMain,
            apply_mask=False,
            concat_mask=True
        ):
        super().__init__()
        
        hidden_dim = unet.config['block_out_channels']
        self.apply_mask = apply_mask
        self.concat_mask = concat_mask

        self.hidden_dim = {
            'down_blocks': list(hidden_dim),
            'mid_block': hidden_dim[-1],
            'up_blocks': list(reversed(hidden_dim))
        }

        self.convs = nn.ModuleList()
        self.adapters = nn.ModuleList()
        for j, name in enumerate(self.hidden_dim.keys()):
            blocks: nn.Module = getattr(unet, name)
            
            if isinstance(blocks, nn.ModuleList):
                for i in range(len(blocks) - 1):
                    bias = int(name == 'up_blocks')
                    idx = i + bias
                    self.convs.append(
                        nn.Conv2d(self.hidden_dim[name][idx] + int(self.concat_mask), self.hidden_dim[name][idx], kernel_size=3, stride=1, padding=1)
                    )

                    self.adapters.append(
                        Adapter(self.hidden_dim[name][idx], self.hidden_dim[name][idx], self.hidden_dim[name][idx])
                    )
            else:
                self.convs.append(
                    nn.Conv2d(self.hidden_dim[name] + int(self.concat_mask), self.hidden_dim[name], kernel_size=3, stride=1, padding=1)
                )
                
                self.adapters.append(
                    Adapter(self.hidden_dim[name], self.hidden_dim[name], self.hidden_dim[name])
                )

        return
    
    def forward(self, attn_queue: list, mask: torch.Tensor):
        layer_idx = 0

        def apply_forward(attn_queue: list, depth: int):
            nonlocal layer_idx

            for i in range(len(attn_queue)):
                if isinstance(attn_queue[i], list):
                    apply_forward(attn_queue[i], depth + 1)
                elif isinstance(attn_queue[i], tuple):
                    attn1_states, attn2_states = attn_queue[i]
                    # apply modules
                    image_size = int(math.sqrt(attn1_states.shape[1]))
                    hidden_mask = torch.nn.functional.interpolate(mask, (image_size, image_size))
                    attn1_states = attn1_states.transpose(-1, -2).reshape(attn1_states.shape[0], -1, image_size, image_size)

                    if self.apply_mask:
                        attn1_states = attn1_states * hidden_mask
                    if self.concat_mask:
                        attn1_states = torch.concat([attn1_states, hidden_mask], dim=1)

                    attn1_states = self.convs[layer_idx](attn1_states)
                    # attn1_states = attn1_states.reshape(*attn1_states.shape[:2], -1).transpose(-1, -2)
                    attn1_states = self.adapters[layer_idx](attn1_states)
                    attn_queue[i] = tuple([attn1_states, attn2_states])
                else:
                    raise NotImplementedError(f'not supported type {type(attn_queue)}')
            
                if depth == 0:
                    layer_idx += 1

            return
        
        apply_forward(attn_queue, 0)
        return attn_queue


class SiameseUnet(nn.Module):
    def __init__(self, unet_main: UNet2DConditionModelMain, unet_ref: UNet2DConditionModelReference, changed_keys: dict=None, frozen_ref=True):
        super().__init__()
        self.unet_main: UNet2DConditionModelMain = unet_main
        self.unet_ref: UNet2DConditionModelReference = unet_ref

        self.changed_keys = list()
        if changed_keys is not None:
            builtin_keys = self.state_dict().keys()
            for key, value in changed_keys.items():
                for sub_key in value:
                    changed_key =  '.'.join([key, sub_key])
                    assert changed_key in builtin_keys
                    self.changed_keys.append(changed_key)

        self.frozen_ref = frozen_ref

        return
    
    def change_parameters(self):
        param_dict = dict(self.named_parameters())
        # param_dict_ = self.state_dict()
        # for key in self.changed_keys:
        #     print(key, id(param_dict[key]), id(param_dict_[key]))
        
        # exit()
        params = [param_dict[key] for key in self.changed_keys]

        return params
    
    def unfrozen_ref(self):
        self.frozen_ref = False

        trainable_param = list(self.unet_ref.parameters())
        trainable_param = {id(param): param for param in trainable_param}
        for param in list(self.unet_ref.parameters()):
            param.requires_grad_(True)
        
        nontrainable_param = self.configure_ref()
        for param in nontrainable_param:
            trainable_param.pop(id(param))

        return list(trainable_param.values())
    
    def configure_ref(self):
        none_trainable = (
            'unet_ref.conv_out', 
            'unet_ref.conv_norm_out', 
            'unet_ref.up_blocks.3.attentions.2.proj_out', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.norm3', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k', 
            'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q'
            # 'unet_ref.conv_norm_out',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.ff',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn1',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.norm3',
            # 'unet_ref.up_blocks.3.attentions.2.proj_out',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0',
            # 'unet_ref.up_blocks.3.attentions.2.transformer_blocks.0.norm2'
        )

        nontrainable_param = list()
        for name, param in self.named_parameters():
            if name.startswith(none_trainable):
                param.requires_grad_(False)
                nontrainable_param.append(param)

        return nontrainable_param
    
    def unfrozen_control(self):
        self.frozen_ref = False

        # none_trainable = ('unet_main', 'unet_ref.conv_norm_out', 'unet_ref.conv_out', 'unet_ref.up_blocks.3.attentions.2')
        none_trainable = ('unet_ref', 'unet_main.conv_in')

        for attr in none_trainable:
            assert getattr_recursive(self, attr) is not None, f'{attr} do not exisit!'

        trainable_param = list()
        for name, param in self.named_parameters():
            if name.startswith(none_trainable):
                pass
                # param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                trainable_param.append(param)
        
        return trainable_param
    
    def configure_attn_range(self, r: list):
        order = ['down_blocks', 'mid_block', 'up_blocks']
        keys = ['attentions', 'transformer_blocks']

        def get_list_module(modules: list[nn.Module], keys, depth=0) -> list[nn.Module]:
            module_list = list()

            if depth >= len(keys):
                return modules

            for m in modules:
                sub_module_list = get_list_module(getattr(m, keys[depth]), keys, depth + 1)
                sub_module_list = [sub_module for sub_module in sub_module_list]
                module_list.extend(sub_module_list)
            
            return module_list

        sorted_module_list = list()
        for child in order:
            module_list = getattr(self.unet_ref, child)
            module_list = module_list if isinstance(module_list, nn.ModuleList) else [module_list]
            module_list = [module for module in module_list if isinstance(module, (CrossAttnDownBlock2DReference, CrossAttnUpBlock2DReference, UNetMidBlock2DCrossAttnReference))]

            sorted_module_list.extend(get_list_module(module_list, keys, 0))

        for i in range(len(sorted_module_list)):
            assert isinstance(sorted_module_list[i], BasicTransformerBlockReference)
            if i in r:
                sorted_module_list[i].enable_cross = True
            else:
                sorted_module_list[i].enable_cross = False

        return sorted_module_list

    def unfrozen_main(self):
        for param in list(self.unet_main.parameters()):
            param.requires_grad_(True)

        return list(self.unet_main.parameters())

    def unfrozen_trainable(self):
        param_dict = dict(self.named_parameters())

        for key in self.changed_keys:
            param_dict[key].requires_grad_(True)

        return

    @classmethod
    def from_pretrained(cls, pretrained_path: str, config_main: str='config.json', config_ref: str='config.json'):
        config_main = UNet2DConditionModelMain.load_config(os.path.join(pretrained_path, config_main))
        config_ref = UNet2DConditionModelMain.load_config(os.path.join(pretrained_path, config_ref))

        unet = UNet2DConditionModel.from_pretrained(pretrained_path)

        unet_main = UNet2DConditionModelMain.from_config(config_main)
        # unet_main.load_state_dict(unet.state_dict())
        unet_main, unmatched_keys_main = try_load(unet_main, unet, return_keys=True)

        unet_ref = UNet2DConditionModelReference.from_config(config_ref)
        # unet_ref.load_state_dict(unet.state_dict())
        unet_ref, unmatched_keys_ref = try_load(unet_ref, unet, return_keys=True)

        changed_keys = {
            'unet_main': unmatched_keys_main,
            'unet_ref': unmatched_keys_ref
        }

        return cls(unet_main, unet_ref, changed_keys=changed_keys)
    
    @classmethod
    def from_config(cls, pretrained_path: str, config_main: str='config.json', config_ref: str='config.json', frozen_ref=True):
        config_main = UNet2DConditionModelMain.load_config(os.path.join(pretrained_path, config_main))
        config_ref = UNet2DConditionModelMain.load_config(os.path.join(pretrained_path, config_ref))

        unet_main = UNet2DConditionModelMain.from_config(config_main)
        unet_ref = UNet2DConditionModelReference.from_config(config_ref)

        return cls(unet_main, unet_ref, frozen_ref=frozen_ref)

    def forward(
        self,
        sample: torch.Tensor,
        ref_sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor=None,
        encoder_attention_mask: torch.Tensor=None,
        feature_adapter: FeatureAdapter=None,
        feature_mask: torch.Tensor=None,
        return_dict=False
    ):
        rates = ref_sample.shape[0] // sample.shape[0]

        if rates == 1:
            encoder_hidden_states_ref = encoder_hidden_states
        else:
            encoder_hidden_states_ref = list()
            
            for i in range(sample.shape[0]):
                encoder_hidden_states_ref.extend([encoder_hidden_states[i: i + 1] for _ in range(rates)])
            encoder_hidden_states_ref = torch.concat(encoder_hidden_states_ref, dim=0)
        
        if self.frozen_ref:
            with torch.no_grad():
                ref_sample, attn_queue = self.unet_ref(
                    ref_sample,
                    timestep,
                    encoder_hidden_states_ref
                )
        else:
            ref_sample, attn_queue = self.unet_ref(
                ref_sample,
                timestep,
                encoder_hidden_states_ref
            )

        if feature_adapter is not None:
            assert feature_mask is not None
            attn_queue = feature_adapter(attn_queue, feature_mask)

        sample = self.unet_main(
            sample,
            timestep,
            encoder_hidden_states,
            attn_queue=attn_queue,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )[0]

        return (sample, ref_sample)
    

if __name__ == '__main__':
    unet = SiameseUnet.from_pretrained('./pretrained/stable-diffusion-v1-4/unet', config_ref='config_ref.json')
