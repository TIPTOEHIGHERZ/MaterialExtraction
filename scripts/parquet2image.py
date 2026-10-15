import pandas as pd
from PIL import Image
from io import BytesIO
import os
import sys
import tqdm
import omegaconf
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.functionals import multi_process

mapping = {
    'basecolor': 'Color',
    'displacement': 'Displacement',
    'normal': 'NormalGL',
    'metallic': 'Metallic',
    'roughness': 'Roughness',
    'height': 'Height',
    'specular': 'Specular',
    'opacity': 'Opacity',
    'diffuse': 'Diffuse'
}


def process(file, rank, base_folder):
    if not file.endswith('.parquet'):
        print(f'{file} is not a support file')
        return

    # file_type = file.split('-')[0]

    try:
        df = pd.read_parquet(file, engine='pyarrow')
    except Exception as e:
        print(e)
        print(f'fail to read {file}')
        # fail_file.append([os.path.join(data_folder, file), cnt])
        # return

    for index, row in tqdm.tqdm(list(df.iterrows()), leave=False) if rank == 0 else list(df.iterrows()):
    # for index, row in list(df.iterrows()):
        data = row.to_dict()
        name = data['name']
        m_data = {key: value if not isinstance(value, np.ndarray) else list(value) for key, value in data['metadata'].items()}

        metadata = {
            'name': name,
            'category': data['category'],
            'metadata': m_data
        }
        
        os.makedirs(os.path.join(base_folder, name), exist_ok=True)
        omegaconf.OmegaConf.save(metadata, os.path.join(base_folder, name, 'metadata.yaml'))

        for k in data.keys():
            if k in mapping.keys():
                image = Image.open(BytesIO(data[k]['bytes']))
                image.save(os.path.join(base_folder, name, f'{name}_{mapping[k]}.png'))

    return


if __name__ == '__main__':
    base_folder = '/home/user/data/datasets/MatSynth/textures_all'
    data_folder = '/home/user/data/datasets/MatSynth/data'

    files = [os.path.join(data_folder, file) for file in os.listdir(data_folder)]

    multi_process(32, process, files, base_folder=base_folder)
