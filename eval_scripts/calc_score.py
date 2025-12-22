from torchvision.io import read_image
from torchvision.transforms.functional import resize, to_tensor
import torchvision
from PIL import Image
import os
import torch
import torch.nn as nn
from functools import partial
import math
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import tqdm
import lpips
import skimage
import cv2
import pandas as pd
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.functionals import count_parameters
from utils.io import load_image, save_image
from utils.functionals import calculate_clip, calculate_ssim, calculate_lpips


torch.cuda.set_device(2)

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
args = parser.parse_args()
data_dir = args.data_dir

score_list = dict()
for folder in tqdm.tqdm(os.listdir(data_dir)):
    color = load_image(os.path.join(data_dir, folder, f'color.png')).unsqueeze(0)
    normal = load_image(os.path.join(data_dir, folder, f'normal.png')).unsqueeze(0)
    height = load_image(os.path.join(data_dir, folder, f'height.png')).unsqueeze(0)
    roughness = load_image(os.path.join(data_dir, folder, f'roughness.png')).unsqueeze(0)

    color_gt = load_image(os.path.join(data_dir, folder, f'color_gt.png')).unsqueeze(0)
    normal_gt = load_image(os.path.join(data_dir, folder, f'normal_gt.png')).unsqueeze(0)
    height_gt = load_image(os.path.join(data_dir, folder, f'height_gt.png')).unsqueeze(0)
    roughness_gt = load_image(os.path.join(data_dir, folder, f'roughness_gt.png')).unsqueeze(0)

    tensors = {
        'color': color, 
        'normal': normal, 
        'height': height, 
        'roughness': roughness,
    }

    tensors_gt = {
        'color': color_gt, 
        'normal': normal_gt, 
        'height': height_gt, 
        'roughness': roughness_gt,
    }

    with torch.no_grad():
        # score = {'name': folder}
        score = dict()

        for key in tensors.keys():
            t = tensors[key]
            t_gt = tensors_gt[key]

            t = t.to('cuda')
            t_gt = t_gt.to('cuda')

            score[key] = {
                'lpips': calculate_lpips(t, t_gt).item(),
                'ssim': calculate_ssim(t, t_gt).item(),
                'clip': calculate_clip(t, t_gt).item(),
            }

        score_list[folder] = score

# sheets = {key: list() for key in tensors.keys()}
sheets = {key: dict() for key in tensors.keys()}

for folder in score_list.keys():
    for key in tensors.keys():
        # sheets[key].append(score_list[folder][key])
        sheets[key].update({folder: score_list[folder][key]})

dfs = dict()
for key in sheets.keys():
    df = pd.DataFrame.from_dict(sheets[key])
    df = df.T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    dfs[key] = df

with pd.ExcelWriter(f'scores/scores_{os.path.basename(data_dir)}.xlsx') as writer:
    for key in dfs.keys():
        dfs[key].to_excel(writer, sheet_name=key)


