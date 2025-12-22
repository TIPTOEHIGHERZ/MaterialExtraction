import sys
import os
import tqdm
import re
import omegaconf


sys.path.append(os.getcwd())
from utils.datasets.DataLoader import TextureDataLoader, DTDLoader, PexelLoader, KTHaLoader, KTHbLoader, KTHLoader
from utils.datasets.DataLoader import ManyTexureLoader, AmbientcgLoader
from utils.datasets.Augmentation import ImageAugmentation
from utils.io import save_image


data_path = './datasets/texture'

conifg_path = './configs/augmentation'
config_mapping = {
    AmbientcgLoader: f'{conifg_path}/ambientcg_textures.yaml',
    ManyTexureLoader: f'{conifg_path}/manytextures.yaml'
}
# loader_types = [KTHaLoader, KTHbLoader, KTHLoader, ManyTexureLoader, AmbientcgLoader]
loader_types = [ManyTexureLoader]
loaders, configs = zip(*[(loader_t(data_path, batch_size=1), omegaconf.OmegaConf.load(config_mapping[loader_t])) for loader_t in loader_types])
# data_loader = TextureDataLoader(loaders)

for loader, config in zip(loaders, configs):
    out_shape = config['out_shape']
    stride = config['stride']
    crop_size = config['crop_size']
    augmentation = ImageAugmentation(out_shape, stride, crop_size)
    with tqdm.tqdm(loader) as progress_bar:
        for image, file_path in progress_bar:
            image = image[0]
            file_path: str = file_path[0]
            image_augmented = augmentation(image)
            
            file_path = re.split(r'[/\\]', file_path)
            file_path[2] = 'texture_modified'
            file_path = '/'.join(file_path)
            for key, value in image_augmented.items():
                if value is not None:
                    with tqdm.tqdm(value, desc=f'Saving {key}', leave=False) as saving_bar:
                        for i, img in enumerate(saving_bar):
                            name, ext = os.path.splitext(file_path)
                            save_path = f'{name}_{key}_{i}' + ext
                            save_image(img, save_path)
