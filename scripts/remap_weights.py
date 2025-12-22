import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())

from utils.functionals import crop_mid
from utils.io import load_image, save_image
from omegaconf import OmegaConf
from utils.functionals import try_load
from model.oned_tokenizer.modeling.titok import TiTok
from utils.datasets import LabelDataLoader
from model.BasicModules.transformer import BasicTransformer
from utils.functionals import remap_weights


state_dict = torch.load('./pretrained/titok.ckpt', weights_only=True)
# model = TiTok.from_pretrained('./pretrained/tokenizer_titok_s128_imagenet')
model = TiTok(OmegaConf.load('./pretrained/tokenizer_titok_s128_imagenet/my_config.json'))
model = try_load(model, state_dict)

torch.save(state_dict, './pretrained/titok_new.ckpt')

for key, value in state_dict.copy().items():
    if 'attn.in_proj_weight' in key:
        new_key = key.split('.')
        
        q_weight, k_weight, v_weight = value.chunk(3, dim=0)

        new_key[-1] = 'q_proj_weight'
        q_key = '.'.join(new_key)
        dims = value.shape[0]
        state_dict[q_key] = q_weight

        new_key[-1] = 'k_proj_weight'
        k_key = '.'.join(new_key)
        dims = value.shape[0]
        state_dict[k_key] = k_weight

        new_key[-1] = 'v_proj_weight'
        v_key = '.'.join(new_key)
        dims = value.shape[0]
        state_dict[v_key] = v_weight

        del state_dict[key]
    elif 'attn.in_proj_bias' in key:
        new_key = key.split('.')
        
        q_weight, k_weight, v_weight = value.chunk(3, dim=0)

        new_key[-1] = 'q_proj_bias'
        q_key = '.'.join(new_key)
        dims = value.shape[0]
        state_dict[q_key] = q_weight

        new_key[-1] = 'k_proj_bias'
        k_key = '.'.join(new_key)
        dims = value.shape[0]
        state_dict[k_key] = k_weight

        new_key[-1] = 'v_proj_bias'
        v_key = '.'.join(new_key)
        dims = value.shape[0]
        state_dict[v_key] = v_weight

        del state_dict[key]

print(len(state_dict), len(model.state_dict()))

nn.MultiheadAttention

mapping = dict()
src_keys = state_dict.keys()

for name, module in model.named_modules():
    if isinstance(module, BasicTransformer):
        mapping[f'{name}.q_proj_weight'] = f'{name}.to_q.weight'
        mapping[f'{name}.k_proj_weight'] = f'{name}.to_k.weight'
        mapping[f'{name}.v_proj_weight'] = f'{name}.to_v.weight'

        mapping[f'{name}.q_proj_bias'] = f'{name}.to_q.bias'
        mapping[f'{name}.k_proj_bias'] = f'{name}.to_k.bias'
        mapping[f'{name}.v_proj_bias'] = f'{name}.to_v.bias'

        mapping[f'{name}.out_proj.weight'] = f'{name}.to_out.weight'
        mapping[f'{name}.out_proj.bias'] = f'{name}.to_out.bias'

unmapped_keys = list()

for key in state_dict.keys():
    if key not in mapping.keys() and '.attn.' in key:
        unmapped_keys.append(key)

print(f'{len(mapping.keys())} keys to remap')
print(f'unmapped keys: {unmapped_keys}')

torch.save(model.state_dict(), './pretrained/titok_modified.ckpt')
