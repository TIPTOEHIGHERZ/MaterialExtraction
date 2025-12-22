import torch
import os
from omegaconf import OmegaConf
from torch.cuda import device


fp = os.path.join(os.getcwd(), 'config/device', 'device.yaml')
config = None
if os.path.exists(fp):
    config = OmegaConf.load(fp)


WORLD_SIZE = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if config is not None:
    # multi gpu enable
    device_ids = config.get('ids', None)
    if device_ids:
        # gpu selected
        device_ids = [str(ids) for ids in device_ids]
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
        WORLD_SIZE = torch.cuda.device_count()
