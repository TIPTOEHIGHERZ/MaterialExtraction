import accelerate.optimizer
import torch
import torch.nn as nn
from typing import Union, Callable
import importlib
import inspect
import tqdm
import os
import numpy as np
import random
import copy
import cv2
import multiprocessing as mp
import accelerate
import math
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def get_mask(image: torch.Tensor, thresh):
    ch = image.shape[1]
    image = torch.sum(image, dim=1, keepdim=True) / ch
    mask = torch.zeros_like(image)
    mask[image > thresh] = 1

    return mask


def between(value, range_: list):
    assert len(range_) == 2
    return range_[0] <= value < range_[1]


class Gaussian2D:
    def __init__(self, m: Union[torch.Tensor, list, tuple], std: Union[torch.Tensor, list, tuple], rotation: Union[float, torch.Tensor]):
        assert len(m) == 2 and len(std) == 2, '2d gaussian should have 2-dimension'
        rotation = torch.tensor([rotation]) if isinstance(rotation, float) else rotation
        self.m = m
        self.std = torch.tensor(std, dtype=torch.float32)
        self.rotation_matrix = torch.tensor([[torch.cos(rotation), -torch.sin(rotation)],
                                             [torch.sin(rotation), torch.cos(rotation)]])

        return

    def create_gaussian(self, mask: torch.Tensor):
        w, h = mask.shape[-2:]
        x = torch.arange(0, w).unsqueeze(0).float()
        y = torch.arange(0, h).unsqueeze(1).float()

        x = x - self.m[0]
        y = y - self.m[1]

        x = x.repeat(h, 1).flatten().unsqueeze(0)
        y = y.repeat(1, w).flatten().unsqueeze(0)

        x, y = (self.rotation_matrix @ torch.cat([x, y], dim=0)).chunk(2, dim=0)
        x = x.reshape(w, h)
        y = y.reshape(w, h)

        ellipse_mask = torch.pow(x, 2) / torch.pow(self.std[0], 2) + torch.pow(y, 2) / torch.pow(self.std[1], 2) < 1

        mask += ellipse_mask.int()

        return mask


def crop(image: torch.Tensor):
    w, h = image.shape[-2:]
    l = min(w, h)
    return image[..., (w - l) // 2: (w + l) // 2, (h - l) // 2: (h + l) // 2]


def super_resolution(image: torch.Tensor, target_res, only_interpolate=False) -> torch.Tensor:
    # TODO apply sr method here, use bilinear interpolate take its place
    if image.shape[-1] < target_res[-1] and not only_interpolate:
        from .RealESRGAN.inference import upsample
        
        current_res = image.shape[-1]
        while current_res < target_res[-1]:
            image = upsample(image, target_res[-1] / image.shape[-1])
            current_res = image.shape[-1]
        
    result = nn.functional.interpolate(image, target_res)
    return result


def image_sr_batch(image_list: list[torch.Tensor], target_res: Union[list[int], tuple[int], int], to_batch=False):
    if len(image_list) == 0:
        raise ValueError('image list should at least have one image')

    for i, image in enumerate(image_list.copy()):
        if image.ndim == 3:
            image.unsqueeze_(0)        
        image_list[i] = super_resolution(crop(image), target_res)

    if to_batch:
        try:
            image_list = torch.concat(image_list, dim=0)
        except Exception as e:
            print(f'{e}\nimage_list has {len(image_list)} objects and image_list has {[image.shape[1] for image in image_list]} channels')
            raise

    return image_list


def init_from_config(config: dict):
    module_name = config.pop('module_name')
    class_name = config.pop('class_name')
    module = importlib.import_module(module_name)
    class_name = getattr(module, class_name)
    input_keys = inspect.signature(class_name.__init__).parameters.keys()
    config = {key: value for key, value in config.items() if key in input_keys}

    return class_name(**config)


def crop_window(image: torch.Tensor, crop_center, crop_size):
    assert len(crop_center) == 2
    crop_size = crop_size if (hasattr(crop_size, '__len__') and len(crop_size) == 2) else (crop_size, crop_size)
    base_point = (crop_center[0] - crop_size[0] // 2, crop_center[1] - crop_size[1] // 2)

    return image[..., base_point[0]:crop_size[0], base_point[1]:crop_size[1]]


def batch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    # perform attention on each batch of them
    b1, l, d = q.shape
    b2 = k.shape[0]

    q = q.reshape(b1, l, heads, d // heads)
    k = k.reshape(b2, l, heads, d // heads)
    v = v.reshape(b2, l, heads, d // heads)

    q = q.permute(2, 1, 0, 3).reshape(heads, l * b1, d // heads)
    k = k.permute(2, 1, 0, 3).reshape(heads, l * b2, d // heads)
    v = v.permute(2, 1, 0, 3).reshape(heads, l * b2, d // heads)

    attn_score = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / ((d // heads) ** 0.5), dim=-1)
    out = torch.matmul(attn_score, v)

    # reset dim
    out = out.reshape(heads, l, b1, d // heads).permute(2, 1, 0, 3).reshape(b1, l, d)

    return out


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    b, l1, d = q.shape
    l2 = k.shape[1]
    q = q.reshape(b, l1, heads, d // heads).permute(0, 2, 1, 3)
    k = k.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)
    v = v.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)

    attn_score = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (d // heads) ** 0.5, dim=-1)
    out = torch.matmul(attn_score, v)
    out = out.permute(0, 2, 1, 3).reshape(b, l1, d)

    return out


def reset_view(tensor: torch.Tensor):
    tensor = tensor.transpose(1, 2)
    tensor = tensor.reshape(*tensor.shape[:2], -1)

    return tensor

def batch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, viewd=False, return_attn_map=False):
    # perform attention on each batch of them
    if viewd:
        # b1, h, l1, d = 
        q = reset_view(q)
        k = reset_view(k)
        v = reset_view(v)
        print(q.shape, k.shape)

    b1, l1, d = q.shape
    b2, l2, _ = k.shape

    q = q.reshape(b1, l1, heads, d // heads)
    k = k.reshape(b2, l2, heads, d // heads)
    v = v.reshape(b2, l2, heads, d // heads)

    q = q.permute(2, 1, 0, 3).reshape(heads, l1 * b1, d // heads)
    k = k.permute(2, 1, 0, 3).reshape(heads, l2 * b2, d // heads)
    v = v.permute(2, 1, 0, 3).reshape(heads, l2 * b2, d // heads)

    attn_score = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / ((d // heads) ** 0.5), dim=-1)
    out = torch.matmul(attn_score, v)

    # reset dim
    if viewd:
        out = out.reshape(heads, b1, -1, d).permute(1, 0, 2, 3)
    else:
        out = out.reshape(heads, l1, b1, d // heads).permute(2, 1, 0, 3).reshape(b1, l1, d)

    if return_attn_map:
        return out, attn_score

    return out


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, attention_mask: torch.Tensor=None):
    b, l1, d = q.shape
    l2 = k.shape[1]
    # print(q.shape, k.shape, v.shape)
    q = q.reshape(b, l1, heads, d // heads).permute(0, 2, 1, 3)
    k = k.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)
    v = v.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)

    attn_score = torch.matmul(q, k.transpose(-1, -2)) / (d // heads) ** 0.5
    if attention_mask is not None:
        attn_score = attn_score + attention_mask.unsqueeze(1)
    attn_score = torch.softmax(attn_score, dim=-1)
    out = torch.matmul(attn_score, v)
    out = out.permute(0, 2, 1, 3).reshape(b, l1, d)

    return out


def masked_attention(q: torch.Tensor,
                     k: torch.Tensor,
                     v: torch.Tensor,
                     heads: int,
                     mask: torch.Tensor,
                     transform_shape=True):
    # print(q.shape, k.shape, v.shape)
    b, l1, d = q.shape
    l2 = k.shape[1]
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


def sample_list(pipeline, num_inference_steps, latents_list: list[torch.Tensor], is_reverse=True):
    device = pipeline.vae.device
    batch_size = latents_list[0].shape[0]

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = reversed(pipeline.scheduler.timesteps) if is_reverse else pipeline.scheduler.timesteps
    iteration = tqdm.tqdm(timesteps, desc='Samping list')

    text_embeddings = pipeline.prepare_prompt_embeddings([''] * batch_size, use_conditional_guidance=False)
    for i, timestep in enumerate(iteration):
        latents = latents_list[-1 - i].to(device)
        pipeline.remove_noise(latents, timestep, text_embeddings, use_conditional_guidance=False)

    return


def check_prompt(prompt, batch_size):
    if isinstance(prompt, str):
        prompt = [prompt] * batch_size
    elif len(prompt) == 1:
        prompt = prompt * batch_size
    elif len(prompt) != batch_size:
        raise ValueError(f'Prompts should have the same number as the sample!,'
                         f'{batch_size} samples accept, but {len(prompt)} are given.')

    return prompt


def count_parameters(module: nn.Module):
    total_params = 0
    for name, param in module.named_parameters():
        if param.requires_grad:
            total_params += param.numel() * param.element_size()

    return total_params / 1024 ** 2


def make_mask(images, resolution, times=30, shuffle_rates=0.5):
    mask, times = torch.ones_like(images[:, :1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
        mask[:, :, y_start:y_start + height, x_start:x_start + width] = 0.

    mask = 1 - mask if random.random() < shuffle_rates else mask
    return mask


@torch.no_grad()
def image2latents(vae, image: torch.Tensor, mask=None):
    # normalize
    if image.ndim == 3:
        image.unsqueeze_(0)
    image = image * 2 - 1

    if mask is not None:
        image = image * mask

    latents = vae.encode(image, return_dict=False)[0].mean
    latents *= vae.config.scaling_factor
    # latents *= 0.18215

    return latents


@torch.no_grad()
def latents2image(vae, latents: torch.Tensor):
    # denormalize
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    # image = self.vae.decode(latents / 0.18215, return_dict=False)[0]
    image = image.clamp(-1, 1)
    image = (image + 1) / 2

    return image


def crop_mid(image: torch.Tensor, crop_size=None):
    l_h, l_w = (min(image.shape[-2:]), min(image.shape[-2:])) if crop_size is None else crop_size
    h_mid, w_mid = image.shape[-2] // 2, image.shape[-1] // 2 
    
    return image[..., h_mid - l_h // 2: h_mid + l_h // 2, w_mid - l_w // 2: w_mid + l_w // 2]


def try_load(son: nn.Module, father: nn.Module | dict, ignore_keys: list=[], return_keys=False):
    father_dict = father if isinstance(father, dict) else father.state_dict()
    son_dict = son.state_dict()
    
    changed_keys = list()
    unmatched_keys = list()
    ignored_keys = list()

    ignore_keys = tuple(ignore_keys)
    for key in list(father_dict.keys()).copy():
        if key.startswith(ignore_keys):
            father_dict.pop(key)
            ignored_keys.append(key)

    for name, param in father_dict.items():
        if name not in son_dict.keys():
            changed_keys.append(name)
            continue

        if son_dict[name].shape == param.shape:
            son_dict[name] = param
        else:
            unmatched_keys.append(name)
    
    son.load_state_dict(son_dict)

    print(f'unmatched keys total {len(unmatched_keys)}: {unmatched_keys}')
    print(f'changed keys total {len(changed_keys)}: {changed_keys}')
    print(f'ignored keys total {len(ignored_keys)}: {ignored_keys}')

    if return_keys:
        return son, unmatched_keys

    return son


@torch.no_grad()
def remap_weights(
    module: nn.Module, 
    mapping: dict,
    state_dict: dict
):
    module_dict = module.state_dict()
    
    for src, dst in mapping.items():
        module_dict[dst] = state_dict[src]

    module.load_state_dict(module_dict)

    return module

def normalize_fft(x: torch.Tensor):
    # x = x / (x.shape[-1] * x.shape[-2])

    x_real = x.real
    x_imag = x.imag

    x_real = (-2 * (x_real < 0) + 1) * torch.log((x_real + 1 + (-2) * (x_real < 0)).abs())
    x_imag = (-2 * (x_imag < 0) + 1) * torch.log((x_imag + 1 + (-2) * (x_imag < 0)).abs())

    return x_real + 1j * x_imag


def denormalize_fft(x: torch.Tensor):
    x_real = x.real
    x_imag = x.imag

    x_real = (-2 * (x_real < 0) + 1) * (torch.exp(x_real.abs()) - 1)
    x_imag = (-2 * (x_imag < 0) + 1) * (torch.exp(x_imag.abs()) - 1)

    return x_real + 1j * x_imag


def create_lowpass_filter(image: torch.Tensor, radius: int):
    device = image.device
    h, w = image.shape[-2:]

    cy, cx = h //2, w // 2

    y = torch.linspace(0, h, h, device=device)
    x = torch.linspace(0, w, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    distance = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_filter = (distance <= radius).float()
    
    low_filter = low_filter.reshape(1, 1, *low_filter.shape)
    low_filter = low_filter.repeat(*image.shape[:2], 1, 1)
    
    return low_filter


def create_highpass_filter(image: torch.Tensor, radius: int):
    device = image.device
    h, w = image.shape[-2:]

    cy, cx = h //2, w // 2

    y = torch.linspace(0, h, h, device=device)
    x = torch.linspace(0, w, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    distance = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high_filter = (distance > radius).float()
    
    high_filter = high_filter.reshape(1, 1, *high_filter.shape)
    high_filter = high_filter.repeat(*image.shape[:2], 1, 1)
    
    return high_filter


def to_heatmap(image: torch.Tensor):
    device = image.device
    image = torch.clamp(image, 0., 1.)

    if image.ndim == 4:
        image_list = [image[i] for i in range(image.shape[0])]
    else:
        if image.ndim != 3:
            raise NotImplementedError(f'should have 3 dims, instead have shape of {image.shape}')
        image_list = [image]

    color_map = cv2.COLORMAP_JET
    heat_maps = list()
    for image in image_list:
        image = image.detach().cpu()
        image = image.mean(dim=0, keepdim=True)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255.).astype(np.uint8)

        heat_map = cv2.applyColorMap(image, colormap=color_map)
        # heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2RGB)
        heat_map = torch.tensor(heat_map, device=device)
        heat_maps.append(heat_map.permute(2, 0, 1).unsqueeze(0))

    heat_maps = torch.concat(heat_maps, dim=0)
    return heat_maps


def freeze_net(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(False)
    
    return


def patchify(image: torch.Tensor, patch_size: int | list[int] | tuple[int], stride=None):
    b, c, h, w = image.shape
    if isinstance(patch_size, int):
        patch_size = [patch_size, patch_size]
    
    assert len(patch_size) == 2, f'patch_size have length of {len(patch_size)}'

    stride = patch_size if stride is None else stride

    image = image.unfold(-2, patch_size[0], stride[0])
    image = image.unfold(-2, patch_size[1], stride[1])

    image = image.reshape(b, c, -1, patch_size[0], patch_size[1])

    return image


def unpatchify(image: torch.Tensor):
    patch_size = image.shape[-2:]

    image = image.permute(0, 1, 2, 4, 3, 5)
    image = image.reshape(*image.shape[:2], image.shape[2] * image.shape[3], image.shape[4] * image.shape[5])

    return image


def patch_shuffle(image: torch.Tensor, patch_size: int | list[int] | tuple[int]):
    b, c, h, w = image.shape
    if isinstance(patch_size, int):
        patch_size = [patch_size, patch_size]
    
    assert len(patch_size) == 2, f'patch_size have length of {len(patch_size)}'
    image = patchify(image, patch_size)
    
    shuffle_idx = torch.randperm(image.shape[2], device=image.device)
    image = image[:, :, shuffle_idx, :, :]
    image = image.reshape(b, c, h // patch_size[0], w // patch_size[1], patch_size[0], patch_size[1])

    image = unpatchify(image)
    
    return image


def gaussian_pdf(point: np.ndarray | torch.Tensor, mean, std, device='cpu'):
    assert len(mean) == len(std) and len(mean) == point.shape[-1]
    dims = len(mean)
    point = point if isinstance(point, torch.Tensor) else torch.tensor(point, device=device)
    mean = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean, device=device)
    std = std if isinstance(std, torch.Tensor) else torch.tensor(std, device=device)
    mean = mean.reshape(1, dims)
    std = std.reshape(1, dims)
    exponent_coefficient = (point - mean) / std
    exponent_coefficient = torch.sum(torch.pow(exponent_coefficient, 2.), dim=-1) * -0.5

    return torch.exp(exponent_coefficient)


def generate_gaussian(mask: torch.Tensor | np.ndarray, ratio=3.0, return_tensor=True, strategy='sum'):
    assert strategy in ('sum', 'max')
    mask = mask[:, :1, ...]

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4:
            assert mask.shape[0] == 1, f'can only process one image each time'
            mask = mask.squeeze(0)
        mask = mask.permute(1, 2, 0)
        mask = mask.cpu().numpy()
        mask = (mask * 255.).astype(np.uint8)
    
    countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = list()
    areas = list()
    for cnt in countours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue  # 避免除以零错误
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((cX, cY))
        areas.append(cv2.contourArea(cnt))

    centers = np.array(centers)
    areas = np.array(areas)
    radius = np.sqrt(areas / np.pi)
    sigmas = radius * ratio / 3
    sigmas = np.expand_dims(sigmas, axis=-1)
    sigmas = np.concat([sigmas, sigmas], axis=-1)

    gaussians = list()
    for mu, sigma in zip(centers, sigmas):
        x = np.arange(0, mask.shape[0], 1)
        y = np.arange(0, mask.shape[1], 1)

        xx, yy = np.meshgrid(x, y)
        
        xy = np.concat([np.expand_dims(xx, axis=-1), np.expand_dims(yy, axis=-1)], axis=-1)
        gaussian = gaussian_pdf(xy, mu, sigma).cpu().numpy()
        gaussians.append(np.expand_dims(gaussian, axis=0))
    
    gaussians = np.concat(gaussians, axis=0)
    if strategy == 'sum':
        gaussians = np.sum(gaussians, axis=0, keepdims=False)
    elif strategy == 'max':
        gaussians = np.max(gaussians, axis=0, keepdims=False)
    else:
        raise NotImplementedError
    gaussians = np.expand_dims(gaussians, axis=-1)
    gaussians = np.clip(gaussians, 0., 1.)

    # gaussians has shape[h, w, 1]
    if return_tensor:
        gaussians = torch.tensor(gaussians)
        # gaussians has shape[1, h, w]
        gaussians = gaussians.permute(2, 0, 1)

    return gaussians


def generate_seperate_mask(masks: torch.Tensor, ratio=3., thresh=0.1):
    masks = masks[:, :1, ...]
    gaussians = [
        generate_gaussian(mask.unsqueeze(0), ratio, True, strategy='max').unsqueeze(0) for mask in masks
    ]
    gaussians = torch.concat(gaussians, dim=0)

    gaussians = (gaussians - masks).clamp(0., 1.)
    high = torch.max(torch.max(gaussians, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    low = torch.min(torch.min(gaussians, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    gaussians = (gaussians - low) / (high - low)

    masks = (2 * masks - masks.sum(dim=0, keepdim=True)).clamp(0., 1.)
    # for i in range(masks.shape[0]):
    #     # 去除重叠的部位
    #     masks[i: i + 1] = (2 * masks[i: i + 1] - masks.sum(dim=0, keepdim=True)).clamp(0., 1.)

    aggregate_mask = masks + (2 * gaussians - torch.sum(gaussians, dim=0, keepdim=True)).clamp(0., 1.)
    aggregate_mask = torch.sum(aggregate_mask, dim=0, keepdim=True).clamp(0., 1.)
    # aggregate_mask = torch.sum(masks, dim=0, keepdim=True)
    # aggregate_mask = torch.max(aggregate_mask, dim=0, keepdim=True)[0]
    # aggregate_mask[aggregate_mask < thresh] = 0.
    # aggregate_mask[aggregate_mask >= thresh] = 1.

    # return torch.max(gaussians, dim=0, keepdim=True)[0].clamp(0., 1.)

    return aggregate_mask


def getattr_recursive(obj, attr: list | str, depth=0):
            if isinstance(attr, str):
                attr = attr.split('.')
            
            attribute = getattr(obj, attr[depth], None)
            if attribute is None or depth == len(attr) - 1:
                return attribute
            attribute = getattr_recursive(attribute, attr, depth=depth + 1)

            return attribute


def sample_exponential_decay(num_samples: int, min_val: int, max_val: int, decay_rate: float = 2e-3):
    # 生成0~max_val的整数
    indices = torch.arange(min_val, max_val)
    
    # 计算权重：w(t) = exp(-decay_rate * t)
    weights = torch.exp(-decay_rate * indices)
    
    # 归一化权重（multinomial会自动归一化，显式归一化更安全）
    probs = weights / weights.sum()
    
    # 按权重采样（返回的是索引值，即t）
    samples = torch.multinomial(probs, num_samples, replacement=True)
    return samples


def barrier_process(proc_id, process_func: Callable, dataloader, barrier, rank, *args, **kwargs):
    """
    处理数据的进程函数
    :param proc_id: 进程ID (0-9)
    :param data_chunk: 分配给该进程的数据块（10条数据）
    :param barrier: 屏障同步对象
    """
    print(f"进程 {proc_id} 准备就绪，等待开始...")
    barrier.wait()  # 等待所有进程初始化完成
    
    # new_gt_fp = texture_loader.gt_fp + new_posfix

    # 按序号同步处理每条数据
    for i, data in enumerate(tqdm.tqdm(dataloader, desc=f'process: {proc_id}') if rank == 0 else dataloader):
        # print(proc_id, i, data)
        data = process_func(data, rank, *args, **kwargs)
        barrier.wait()
    
    print(f"进程 {proc_id} 已完成所有任务")
    return


def multi_process(num_processes, process_func: Callable, dataloader, *args, attr=None, **kwargs):
    files_each_loader = len(dataloader) // num_processes if attr is None else len(getattr(dataloader, attr)) // num_processes
    remain_files = len(dataloader) % num_processes if attr is None else len(getattr(dataloader, attr)) % num_processes

    dataloader_list = list()
    for i in range(num_processes):
        dataloader_files = dataloader[i * files_each_loader: (i + 1) * files_each_loader] if attr is None else \
        getattr(dataloader, attr)[i * files_each_loader: (i + 1) * files_each_loader]

        if attr is not None:
            loader = copy.deepcopy(dataloader)
            setattr(loader, attr, dataloader_files)
        else:
            loader = dataloader_files
        # loader.total_images = len(loader.files)
        dataloader_list.append(loader)

    barrier = mp.Barrier(num_processes)
    processes = []
    for i in range(num_processes):
        iterator = dataloader_list[i] if attr is None else getattr(dataloader_list[i], attr)

        p = mp.Process(
            target=barrier_process,
            args=(i, process_func, iterator, barrier, i, *args),
            kwargs=kwargs
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("all process finished, procceed remaining files...")

    dataloader_files = dataloader[-remain_files:] if attr is None else getattr(dataloader, attr)[-remain_files:]
    if attr is not None:
        loader = copy.deepcopy(dataloader)
        setattr(loader, attr, dataloader_files)
    else:
        loader = dataloader_files

    iterator = loader if attr is None else getattr(loader, attr)
    for i, data in enumerate(tqdm.tqdm(iterator, desc=f'process remain files')):
        data = process_func(data, 0, *args, **kwargs)

    return


def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def random_mask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(4 * coef), s // 2)
        MultiFill(int(2 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(8 * coef), s))  # hole denoted as 0, reserved as 1
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)


class MemoryHalt:
    def __init__(self):
        self.tensor_list = list()

        return
    
    def halt(self, m_gb: int):
        for i in range(m_gb):
            self.tensor_list.append(torch.randn([1, 1024, 1024, 256], device='cuda'))

        return
    
    def release(self):
        del self.tensor_list
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.tensor_list = list()

        return
    

def calculate_ssim(image: torch.Tensor, image_ref: torch.Tensor):
    score = list()

    for image_, image_ref_ in zip(image, image_ref):
        image_ = (image_.cpu().numpy().clip(0., 1.) * 255.).astype(np.uint8)
        image_ref_ = (image_ref_.cpu().numpy().clip(0., 1.) * 255.).astype(np.uint8)
        
        score.append(ssim(image_, image_ref_, channel_axis=0))

    return torch.tensor(score, device=image.device)


@torch.no_grad()
def calculate_lpips(image: torch.Tensor, image_ref: torch.Tensor, device='cuda'):
    if not hasattr(calculate_lpips, 'net'):
        import lpips
        calculate_lpips.net = lpips.LPIPS(net='alex')
        calculate_lpips.net.to(device)
    
    image = image.unsqueeze(0) if image.ndim == 3 else image
    bsz = image.shape[0]

    return calculate_lpips.net(image, image_ref).reshape((bsz,))


@torch.no_grad()
def calculate_fid(image: torch.Tensor, image_ref: torch.Tensor):
    # TODO 这个没有完成，感觉fid的计算不是必要指标
    if not hasattr(calculate_fid, 'net'):
        from pytorch_fid import fid_score
        import torchvision
        import torchvision.transforms as transforms
        calculate_fid.net = torchvision.models.inception_v3(pretrained=True)
        calculate_fid.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(256),
        ])
    
    image = calculate_fid.transform(image)
    image_ref = calculate_fid.transform(image_ref)


@torch.no_grad()
def calculate_clip(image: torch.Tensor, image_ref: torch.Tensor, device='cuda'):
    if not hasattr(calculate_clip, 'net'):
        from transformers.models.clip.modeling_clip import CLIPVisionModel, CLIPModel
        import torchvision.transforms as transforms
        calculate_clip.net = CLIPModel.from_pretrained('./pretrained/clip-vit-large-patch14').vision_model
        calculate_clip.net.to(device)
        calculate_clip.transform = transforms.Compose([
            # transforms.CenterCrop(),
            transforms.Resize(224),
        ])

    image = image.to(device)
    image_ref = image_ref.to(device)
    image = image.unsqueeze(0) if image.ndim == 3 else image
    bsz = image.shape[0]

    image = calculate_clip.transform(image)
    image_ref = calculate_clip.transform(image_ref)

    image_embedding = calculate_clip.net(image)['pooler_output']
    ref_embedding = calculate_clip.net(image_ref)['pooler_output']

    cosine_similarity = nn.functional.cosine_similarity(image_embedding, ref_embedding, dim=1)

    return cosine_similarity
