from PIL import Image
import numpy as np
import torch
import torchvision
import os
from typing import Union, Iterable
import torch
import torch.nn as nn
import cv2


def to_numpy(obj: Union[torch.Tensor, np.ndarray], dtype=np.uint8):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 3:
            obj = obj.unsqueeze(0)

        assert obj.ndim == 4, 'tensor should be 4 dimensional'
        obj = obj.permute(0, 2, 3, 1).detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        pass
    else:
        raise NotImplementedError('unknown instance to convert')

    if not np.issubdtype(obj.dtype, np.integer):
        if dtype == np.uint8:
            obj = (obj * 255.).astype(np.uint8)
        elif dtype == np.uint16:
            obj = (obj * 65535.).astype(np.uint16)
            obj = obj[..., 0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return obj


def load_image(path: Union[str, list[str]], to_batch=False, device='cpu', return_dtype=False) -> torch.Tensor:

    if isinstance(path, str):
        path = os.path.join(os.getcwd(), path)
        image = Image.open(path)

        if image.mode.endswith('16'):
            image = np.array(image)
            if image.ndim == 2:
                image = np.concatenate([image[..., np.newaxis]] * 3, axis=-1)
        else:
            image = image.convert('RGB')
            image = np.array(image)
        
        dtype = image.dtype
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        else:
            raise NotImplementedError(f'unsupported dtype {image.dtype}')
        
        image = torch.tensor(image, device=device)
        image = image.permute(2, 0, 1)
    else:
        raise NotImplementedError('can read one image each time')

    if isinstance(path, str) and to_batch:
        image.unsqueeze_(0)

    if return_dtype:
        return image, dtype

    return image

def load_exr(path: str, to_batch=False) -> torch.Tensor:
    assert path.endswith('.exr'), f'file {path} is not a exr file'
    image = cv2.imread(path, -1)

    if image.ndim == 2:
        image = np.concatenate([image[..., np.newaxis]] * 3, axis=-1)

    # switch to rgb, since cv2.imread using channel order b -> g -> r
    image = np.stack([image[..., 2], image[..., 1], image[..., 0]], axis=-1)

    image = image.transpose(2, 0, 1)
    image = torch.tensor(image)
    image = image.unsqueeze(0) if to_batch else image

    return image


# todo too duplicate, need to rewrite
def save_image(obj: Union[torch.Tensor, Image.Image, np.ndarray], 
               fp: Union[str], 
               save_format='RGB', 
               default_name='result.jpg',
               absolute_path=False,
               dtype=np.uint8):
    # if isinstance(fp, Iterable) and not isinstance(fp, str):
    #     assert len(fp) == len(obj), 'fp should have same length as obj'
    #     for obj_, fp_ in zip(obj, fp):
    #         save_image(
    #             obj_, 
    #             fp_, 
    #             save_format=save_format, 
    #             absolute_path=absolute_path
    #         )
    #     return

    fp = fp if absolute_path else os.path.join(os.getcwd(), fp)

    if isinstance(obj, Image.Image):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        obj.convert(save_format).save(fp)
        return
    elif isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        obj = to_numpy(obj, dtype=dtype)
    else:
        raise NotImplementedError(f'not support type {obj.__class__.__name__}')


    assert len(obj) == 1
    obj = obj[0]

    if save_format == 'L' and obj.ndim == 3:
        if obj.shape[-1] == 1:
            obj = obj[..., 0]
        elif obj.shape[0] == 1:
            obj = obj[0, ...]
        # else:
        #     raise

    os.makedirs(os.path.dirname(fp), exist_ok=True)
    Image.fromarray(obj, mode=save_format).save(fp)
        
    return

def load_from_ddp(fp: str):
    state_dict = torch.load(fp, weights_only=True)

    keys = list(state_dict.keys())

    for key in keys:
        new_key = key.split('.')[1:]
        new_key = '.'.join(new_key)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return state_dict


def load_mask(prefix: str, device='cpu'):
    dir_name = os.path.dirname(prefix)
    prefix = os.path.basename(prefix)

    targets = os.listdir(dir_name)
    for target in targets.copy():
        target_name = os.path.splitext(target)[0]
        if not target_name.startswith(prefix):
            targets.remove(target)

    masks = [load_image(os.path.join(dir_name, target), to_batch=True) for target in targets]
    masks = torch.concat(masks, dim=0)
    masks[masks < 0.5] = 0.
    masks[masks >= 0.5] = 1.

    return masks.to(device)
