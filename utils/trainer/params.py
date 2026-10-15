import torch
import torch.nn as nn
import torch.optim


def shadow_parameters(module: nn.Module, device='cpu') -> dict[str, torch.Tensor]:
    return {name: param.detach().to(device) for name, param in module.named_parameters()}


@torch.no_grad()
def apply_shadow_parameters(module: nn.Module, state_dict: dict[str, torch.Tensor]):
    for name, param in module.named_parameters():
        param.copy_(state_dict[name])


def set_group_lr(optimizer: torch.optim.Optimizer, lr_list: list[float]):
    assert len(optimizer.param_groups) == len(lr_list), \
        f'optimizer has {len(optimizer.param_groups)} param groups, got {len(lr_list)} lrs'
    for group, lr in zip(optimizer.param_groups, lr_list):
        group['lr'] = lr
