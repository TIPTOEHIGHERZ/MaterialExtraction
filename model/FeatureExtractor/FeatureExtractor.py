import torch
import torch.nn as nn
import os
from transformers import CLIPTextModel, CLIPModel, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPVisionEmbeddings
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import math
from typing import Union, Optional

from utils.functionals import attention


def embeddings2image(embeddings: torch.Tensor):
    if embeddings.ndim == 4:
        return embeddings

    assert embeddings.ndim == 3, 'do not have enough dims to convert'
    image_shape = int(math.sqrt(embeddings.shape[-2]))
    embeddings = embeddings.transpose(-1, -2)
    image = embeddings.reshape(*embeddings.shape[:2], image_shape, image_shape)
    
    return image


def image2embeddings(image: torch.Tensor):
    if image.ndim == 3:
        return image

    assert image.ndim == 4, f'do not support {image.ndim} to convert'
    embeddings = image.view(image.shape[:2], -1).transpose(-1, -2)

    return embeddings


def masked_attention(q: torch.Tensor,
                     k: torch.Tensor,
                     v: torch.Tensor,
                     heads: int,
                     mask: torch.Tensor):
    b, l1, d = q.shape
    l2 = k.shape[1]
    # print(q.shape, k.shape, v.shape)
    if mask.ndim == 4:
        # mask will have shape of [b, 1, 1, h * w]
        mask = mask.reshape(*mask.shape[:2], -1).unsqueeze(2)
    else:
        # from shape [b, h * w, 1] to [b, 1, 1, h * w]
        mask = mask.transpose(-1, -2).unsqueeze(2)
    q = q.reshape(b, l1, heads, d // heads).permute(0, 2, 1, 3)
    k = k.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)
    v = v.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)

    attn_score = torch.softmax((torch.matmul(q, k.transpose(-1, -2)) + mask * -100) / (d // heads) ** 0.5, dim=-1)
    # only unmasked values are functional
    # attn_score = attn_score * mask
    out = torch.matmul(attn_score, v)
    out = out.permute(0, 2, 1, 3).reshape(b, l1, d)

    return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 qkv_dims: int,
                 embedding_dims: int,
                 heads: int,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.heads = heads

        self.to_q = nn.Linear(qkv_dims, embedding_dims, bias=False)
        self.to_k = nn.Linear(qkv_dims, embedding_dims, bias=False)
        self.to_v = nn.Linear(qkv_dims, embedding_dims, bias=False)

        # default to input channels
        out_dims = qkv_dims
        self.project_out = nn.Linear(embedding_dims, out_dims)

        self.to(device)
        return
    
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # TODO maybe use masked attention?
        out = masked_attention(q, k, v, self.heads, mask)
        out: torch.Tensor = self.project_out(out)

        return out
    

class ConvolutionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 device='cpu'):
        super().__init__()
        self.device = device

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        return
        
    def forward(self, hidden_states: torch.Tensor):
        # image = image * mask
        if hidden_states.ndim == 3:
            image_size = int(math.sqrt(hidden_states.shape[1]))
            hidden_states = hidden_states.transpose(-1, -2)
            hidden_states = hidden_states.reshape(*hidden_states.shape[:2], image_size, image_size)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states


class VisionExtractor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 heads: int,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.is_enable = True

        out_channels = in_channels
        self.attn = TransformerBlock(in_channels, out_channels, heads, device)
        self.conv = ConvolutionBlock(in_channels, out_channels, device)
        
        self.to(device)
        return

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        # residual = hidden_states
        # masked_hidden_states = hidden_states
        if hidden_states.ndim == 3:
            image_size = int(math.sqrt(hidden_states.shape[1]))
        else:
            image_size = hidden_states.shape[-1]
        
        mask = nn.functional.interpolate(mask, (image_size, image_size))
        # mask = mask.repeat(hidden_states.shape[0], 1, 1, 1)

        if hidden_states.ndim == 3:
            mask = mask.view(*mask.shape[:2], -1).transpose(-1, -2)

        # TODO 这段代码有问题，information flow存在问题，导致被mask掉的部分的attention score一直是0
        attn_out = self.attn(hidden_states, mask)
        conv_out = self.conv(mask * hidden_states)
        conv_out = conv_out.view(*conv_out.shape[:2], -1).transpose(-1, -2)

        out = attn_out + conv_out
        out = hidden_states + out
        return out

    def enable(self):
        self.is_enable = True
        return

    def disable(self):
        self.is_enable = False
        return
    

class FeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 heads: int,
                 device='cpu'):
        super().__init__()
        self.vision_extractor = VisionExtractor(in_channels, heads, device)
        # TODO extract text embeddings for better synthesis
        self.text_extractor = None
        self.is_enable = True

        self.to(device)
        return

    def forward(self, image: torch.Tensor, i):
        hidden_states = self.mlp[i](self.vision_backbone(image).pooler_output)
        hidden_states = hidden_states.reshape(-1, self.feature_len, self.kv_dim_list[i])

        k = self.to_k[i](hidden_states)
        v = self.to_v[i](hidden_states)

        return k, v
    
    def enable(self):
        self.is_enable = True
        return

    def disable(self):
        self.is_enable = False
        return

    def frozen(self):
        pass

    @staticmethod
    def from_pretrained(image_size,
                        in_channels: int,
                        feature_len: int,
                        kv_dims_list: list[int],
                        fp: str,
                        device='cpu'):
        clip_model: CLIPModel = CLIPModel.from_pretrained(fp)
        vision_backbone: CLIPVisionTransformer = clip_model.vision_model

        return FeatureExtractor(image_size, in_channels, feature_len, kv_dims_list, vision_backbone, device)


class ExtractorRegister:
    BLOCK_TYPES = ['down_blocks', 'up_blocks', 'mid_block']
    def __init__(self, module: Union[UNet2DConditionModel, StableDiffusionPipeline] = None):
        self.module_dict: dict[str, VisionExtractor] = dict()
        self.unet: UNet2DConditionModel = None
        if module is not None:
            self.register_unet(module)

        return

    def load(self, fp: str):
        module_dict = torch.load(fp, weights_only=False)
        weights_dict = {k: v.state_dict() for k, v in module_dict.items()}

        for name, module in self.module_dict.items():
            module.load_state_dict(weights_dict[name])
        
        return
    
    def save(self, fp):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save(self.module_dict, fp)
        return
    
    def train(self):
        for module in self.module_dict.values():
            module.train()

        return
    
    def enable(self):
        for module in self.module_dict.values():
            module.enable()
        return

    def disable(self):
        for module in self.module_dict.values():
            module.disable()
        return

    def parameters(self):
        trainable_parameters = list()
        for module in self.module_dict.values():
            trainable_parameters += list(module.parameters())

        return trainable_parameters

    def register_unet(self, model: Union[UNet2DConditionModel, StableDiffusionPipeline]):
        unet = model if isinstance(model, UNet2DConditionModel) else model.unet
        self.unet = unet

        def register_forward(attn: Attention,
                             extractor_config: dict,
                             is_vision_attn: bool):
            attn.extractor = VisionExtractor(**extractor_config) if is_vision_attn else None

            def forward(hidden_states: torch.Tensor,
                        encoder_hidden_states: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        **cross_attention_kwargs):                
                in_dims = hidden_states.ndim
                hidden_states = image2embeddings(hidden_states)

                if encoder_hidden_states is None:
                    mask = cross_attention_kwargs.get('mask', None)
                    if attn.extractor.is_enable:
                        encoder_hidden_states = attn.extractor(hidden_states, mask)
                    else:
                        encoder_hidden_states = hidden_states
                
                residual = hidden_states

                q = attn.to_q(hidden_states)
                k = attn.to_k(encoder_hidden_states)
                v = attn.to_v(encoder_hidden_states)

                out = attention(q, k, v, heads=attn.heads)
                out = attn.to_out[0](out)
                out = attn.to_out[1](out)
                
                if attn.residual_connection:
                    out = out + residual

                out = out / attn.rescale_output_factor

                if in_dims == 4:
                    out = embeddings2image(out)

                return out
            
            attn.forward = forward
            return attn.extractor

        def register_blocks(block: nn.Module, name: str):
            for name_, module in block.named_modules():
                if isinstance(module, Attention):
                    query_dim = module.query_dim
                    heads = module.heads
                    is_vision_attn = name_.endswith('attn1')
                    extractor_config = {'in_channels': query_dim, 'heads': heads}
                    extractor = register_forward(module, extractor_config, is_vision_attn)
                    if extractor is not None:
                        self.module_dict[name + name_] = extractor

            return

        for name, module in unet.named_children():
            if name in self.BLOCK_TYPES:
                register_blocks(module, name)

        return
    