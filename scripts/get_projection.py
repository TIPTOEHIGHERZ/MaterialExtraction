import os
import sys
import torch
import tqdm
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.estimate_depth import DepthEstimator
from utils.io import load_image, save_image
from utils.functionals import multi_process
from utils.datasets.DataLoader import PBRTextureDataLoader, PolyLoader
from utils.project import project


def process(data, rank):
    image = load_image(data['image'])
    depth = load_image(data['depth'])[0]
    mask = load_image(data['mask'])

    image_project, mask_project = project(image, depth, mask, shift=0.8)

    image_project = nn.functional.interpolate(image_project.unsqueeze(0), image.shape[-2:])
    mask_project = nn.functional.interpolate(mask_project.unsqueeze(0), image.shape[-2:])

    mask_project = mask_project[:, :1, ...]
    mask_project[mask_project >= 0.9] = 1.
    mask_project[mask_project < 0.9] = 0.

    dir_name = os.path.dirname(data['image'])
    save_image(image_project, os.path.join(dir_name, 'image_project.png'))
    save_image(mask_project, os.path.join(dir_name, 'mask_project.png'), save_format='L')
    return


PolyLoader.base_names = tuple()
PolyLoader.image_names = ('image', 'mask', 'depth')
data_loader = PolyLoader(
    './datasets/polyhaven_examples/',
)

# PBRTextureDataLoader.image_names = ('image', 'mask', 'depth')
# data_loader = PBRTextureDataLoader(
#     fp='./datasets/render_base_10_resized', 
#     gt_fp='./datasets/MatSynth/textures_all_resized', 
#     transforms={'default': lambda x: x, 'no_transform': lambda x: x},
#     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
#     selected_files=None,
#     good_examples=None,
#     transform_group={'default': ['Color', 'NormalGL', 'Height', 'Roughness'], 'no_transform': ['image', 'mask']},
# )                                                                                                                                                                                                                                                                                                                                                                                         


for data in tqdm.tqdm(data_loader.files):
    process(data, 0)
