import torch
import torch.nn as nn
from typing import Union, Optional
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import logging
from utils.functionals import batch_attention, attention, between


REGISTER_BLOCK_NAMES = ['down_blocks', 'up_blocks', 'mid_block']


class MaskInjectionAgent:
    """
    将mask作用于attention map
    """
    def __init__(self, activated_range: list[int]):
        """
        Args:
            activated_layers: list[int],长度应该为3,分别对应down_blocks、up_blocks、mid_blocks更改attention map的深度
        """
        self.activated_range = activated_range

        self.cur_layer = 0

        # will be set through register function
        self.total_layer = 0

        return

    def step(self, q, k, v, heads):
        self.cur_layer += 1
        return batch_attention(q, k, v, heads)

    def reset(self):
        self.cur_layer = 0
        return

    def __call__(self, 
                 attn: Attention, 
                 hidden_states: torch.Tensor, 
                 encoder_hidden_states: torch.Tensor, 
                 heads: int,
                 attention_mask: torch.Tensor = None):
        b, c = hidden_states.shape[2:]
        hidden_states = hidden_states.view(b, c, -1).transpose(-1, -2)

        if between(self.cur_layer, self.activated_range):
            attention_mask = nn.functional.interpolate(attention_mask, hidden_states.shape[-2:])

            # mask out the region need to be filled
            hidden_states = hidden_states * attention_mask

        hidden_states = hidden_states.view(b, c, -1).permute(0, 2, 1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is not None and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        q = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
        else:
            # cross attention
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)

        out = attention(q, k, v, heads)

        self.cur_layer += 1
        self.cur_layer %= 0

        return out


def register_mask_injection(model: Union[StableDiffusionPipeline, UNet2DConditionModel],
                            injection_agents: list[MaskInjectionAgent],
                            register_name=''):
    assert len(injection_agents) == len(REGISTER_BLOCK_NAMES)
    injection_agents = dict(zip(REGISTER_BLOCK_NAMES, injection_agents))

    if isinstance(model, UNet2DConditionModel):
        unet = model
    else:
        unet = model.unet

    # dict to save register information
    register_dict = dict()
    register_count = dict()

    def register_forward(attn: Attention, attn_type: str, block_type: str, count=0):
        def forward(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    attention_mask=None):
            self: Attention = attn
            
            out = self.injection_agent(self, hidden_states, encoder_hidden_states, self.heads, attention_mask)

            out = self.to_out[0](out)
            out = self.to_out[1](out)

            return out

        def forward_attn1(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    **cross_attention_kwargs):
            self: Attention = attn
            assert hasattr(self, 'registered') and self.registered == True, 'seems not registered'
            assert 'attention_mask' in cross_attention_kwargs.keys()
            assert hidden_states.ndim == 4
            b, c, h, w = hidden_states.shape
            attention_mask = cross_attention_kwargs.pop('attention_mask', None)
            
            residual = hidden_states
            out = forward(hidden_states, encoder_hidden_states, attention_mask)

            out = out.transpose(-1, -2).reshape(b, c, h, w)

            if self.residual_connection:
                out = out + residual

            out /= self.rescale_output_factor
            return out

        def forward_attn2(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    **cross_attention_kwargs):
            pass

        NAME2FORWARD = {'attn1': forward_attn1, 'attn2': forward_attn1}
        # register new forward function and injection
        attn.forward = NAME2FORWARD[attn_type]
        attn.registered = True
        attn.injection_agent = injection_agents[block_type]
        # Attention's child can't have attention, just return
        return count + 1

    def register_blocks(block: nn.Module, block_type: str, count=0):
        for name, child in block.named_modules():
            if isinstance(child, Attention):
                # TODO only register to attn2 not attn1
                name = name.split('.')[-1]
                count = register_forward(child, name, block_type, count)
                register_dict[f'{register_name}.{block_type}.{name}'] = child

        return count

    for name, module in unet.named_children():
        if name in REGISTER_BLOCK_NAMES:
            register_count[name] = register_blocks(module, name)

    # access from the outside
    unet.register_dict = register_dict
    unet.register_count = register_count
    unet.injection_agents = injection_agents
    
    for key in REGISTER_BLOCK_NAMES:
        injection_agents[key].total_layer = register_count[key]

    return register_dict, register_count


def reset_inference_steps(attn: nn.Module, count=0):
    for name, child in attn.named_children():
        if isinstance(child, Attention) and hasattr(child, 'inference_steps'):
            child.inference_steps = 0
            return count + 1
        else:
           count = reset_inference_steps(child, count)

    return count


