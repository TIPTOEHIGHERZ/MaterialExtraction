import os
import sys

import torch
from torch.utils.data import DataLoader
import tqdm
import importlib
from omegaconf import OmegaConf
import argparse

sys.path.append(os.getcwd())
from utils.datasets import DTDLoader
from utils.functionals import image_sr_batch, init_from_config
from utils.io import save_image


parser = argparse.ArgumentParser(description='data process')
parser.add_argument('-c', '--config', help='config file for certain datasets')
parser.add_argument('-sd', '--save_dir', help='save directory after process', default=None)
parser.add_argument('-nw', '--num_workers', help='num workers to retrive data', default=1)
parser.add_argument('-dv', '--device', help='device to process data', default='cpu')


def main():
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    save_dir: str = args.save_dir
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(config)
    data_dir = config['fp']
    data_dir_modified = config['fp'] + '_modified' if save_dir is None else save_dir
    print(f'data will be save to {data_dir_modified}')
    config['device'] = device
    loader = init_from_config(config)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return list(images), list(labels)

    loader = DataLoader(loader, loader.batch_size, args.num_workers, collate_fn=collate_fn)

    for image, file_path in tqdm.tqdm(loader):
        image: list[torch.Tensor] = image_sr_batch(image, 512, True)
        file_path = list(map(lambda fp: os.path.join(data_dir_modified, os.path.relpath(fp, data_dir)), file_path))
        image = image.cpu()
        save_image(image, file_path, absolute_path=True)

    return


if __name__ == '__main__':
    main()
