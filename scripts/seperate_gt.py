import os
import sys
import tqdm
import torch

sys.path.append(os.getcwd())

from utils.datasets.DataLoader import PBRTextureDataLoader
from utils.functionals import crop_mid, multi_process
from utils.io import load_image, save_image


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = x.repeat(1, 1, 2, 2)
    x = torch.nn.functional.interpolate(crop_mid(x), [1024, 1024])

    if in_dim == 3:
        return x.squeeze(0)

    return x


data_loader = PBRTextureDataLoader(fp='./datasets/render_result_matsynth_resized_noscale', gt_fp='./datasets/MatSynth/textures')

target_dir = './datasets/MatSynth/color_1024_tiled/'
os.system(f'rm -r {target_dir}')
os.makedirs(target_dir)

files = list()
for file in tqdm.tqdm(data_loader.files):
    if file['gt'] not in files:
        files.append(file['gt'])


def process(file, target_dir):
    # print('fuck', file)
    image = load_image(file, to_batch=True)
    image = transform(image)
    save_image(image, os.path.join(target_dir, os.path.basename(file)))


num_process = 32
files_per_process = len(files) // num_process
remain_files = len(files) % num_process
file_list = [files[i * files_per_process: (i + 1) * files_per_process] for i in range(num_process)]

multi_process(num_process, process, files, attr=None, target_dir=target_dir)
