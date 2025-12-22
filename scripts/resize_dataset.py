import os
import sys
import torch
import torch.nn as nn
import tqdm
import omegaconf
from typing import Callable
import copy
from PIL import Image
import numpy as np

sys.path.append(os.getcwd())

from utils.datasets.DataLoader import PBRTextureDataLoader
from utils.io import load_image, save_image
from utils.functionals import crop, multi_process


def transform(image: torch.Tensor):
    image = crop(image)
    image = nn.functional.interpolate(image, [512, 512])

    return image


def resize_dataset(file, rank, new_fp):
    base_ratio = [3., 7.]
    base_radius = [0.7, 9.]

    dir_name = os.path.basename(os.path.dirname(file['Color']))
    new_fp = os.path.join(new_fp, dir_name)

    for attr in file.values():
        image, dtype = load_image(attr, to_batch=True, return_dtype=True)
        image = transform(image)
        save_format = 'I;16' if dtype == np.uint16 else 'RGB'
        save_image(image, os.path.join(new_fp, os.path.basename(attr)), save_format=save_format, dtype=dtype)

    # image_path = new_fp + file['image'][len(texture_loader.fp):]
    # os.makedirs(os.path.dirname(image_path), exist_ok=True)
    # save_image(image, image_path)

    # mask_path = new_fp + file['mask'][len(texture_loader.fp):]
    # os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    # save_image(mask, mask_path)

    
    return

    gt_path = os.path.dirname(new_fp + file['mask'][len(texture_loader.fp):])
    gt_path = os.path.join(gt_path, 'gt.png')

    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    config = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(file['image']), 'config.yaml'))
    radius = config['radius']
    ratio = config['ratio']
    # scale_ratio = (radius * ratio) / (base_ratio[1] * base_radius[1])
    scale_ratio = ratio / base_ratio[1]

    scale_ratio = min(scale_ratio, 1.)
    omegaconf.OmegaConf.save({'ratio': ratio, 'radius': radius, 'scale_ratio': scale_ratio}, os.path.join(os.path.dirname(gt_path), 'config.yaml'))

    gt = load_image(file['gt'], to_batch=True)
    # gt = gt[:, :, :int(scale_ratio * gt.shape[-2]), :int(scale_ratio * gt.shape[-1])]
    gt = transform(gt)
    save_image(gt, gt_path)

# for file in tqdm.tqdm(texture_loader.gt_files.values()):
#     image = load_image(file, to_batch=True)

#     image = transform(image)

#     image_path = new_gt_fp + file[len(texture_loader.gt_fp):]
#     os.makedirs(os.path.dirname(image_path), exist_ok=True)
#     save_image(image, image_path)


if __name__ == '__main__':
    PBRTextureDataLoader.file_type = ('Color', 'NormalGL', 'Roughness', 'Height')
    texture_loader = PBRTextureDataLoader(
        fp='./datasets/render_polyhaven_train_50_resized', 
        gt_fp='./datasets/MatSynth/polyhaven_train_render',
        gt_mapping={
            'diff_4k': 'Color',
            'nor_gl_4k': 'NormalGL',
            'rough_4k': 'Roughness',
            'disp_4k': 'Height'
        }
    )

    num_process= 32
    files_each_loader = len(texture_loader) // num_process
    remain_files = len(texture_loader) % num_process

    # texture_loaders = list()
    # for i in range(num_process):
    #     loader = copy.deepcopy(texture_loader)
    #     loader.files = texture_loader.files[i * files_each_loader: (i + 1) * files_each_loader]
    #     texture_loaders.append(loader)

    # loader = copy.deepcopy(texture_loader)

    new_fp = './datasets/MatSynth/polyhaven_train_render_resized'
    multi_process(num_process, resize_dataset, texture_loader, attr='gt_files', new_fp=new_fp)

    # loader.files = texture_loader.files[-remain_files:]
    # loader.total_images = len(loader.files)

    # for file in tqdm.tqdm(loader.files):
    #     new_posfix = '_resized_noscale'
    #     new_fp = loader.fp + new_posfix
    #     resize_dataset(file, new_fp)
