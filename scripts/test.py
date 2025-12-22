import os
import sys
# from diffusers.models.attention_processor import Attention

sys.path.append(os.getcwd())

from model.unet.SiameseUnet import SiameseUnet
from model.unet.attn_ref import AttentionReference, BasicTransformerBlockReference

unet = SiameseUnet.from_config('./configs/unet/render_pretrained', config_main='config_main.json', config_ref='config_ref.json')
module_list = unet.configure_attn_range(list(range(4, 14)))

for name, module in unet.unet_ref.named_modules():
    if isinstance(module, BasicTransformerBlockReference):
        print(name, module.enable_cross)

# for name, module in unet.unet_ref.named_modules():
#     if isinstance(module, BasicTransformerBlockReference):
#         print(name)
