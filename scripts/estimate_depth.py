import os
import sys
import torch
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.estimate_depth import DepthEstimator
from utils.io import load_image, save_image
from utils.functionals import multi_process
from utils.datasets.DataLoader import PBRTextureDataLoader, PolyLoader


torch.cuda.set_device(3)

estimator = DepthEstimator('./pretrained/iid')
def process(data, rank):
    image = load_image(data['image'], to_batch=True)
    depth = estimator(image)
    dir_name = os.path.dirname(data['image'])

    save_image(depth, os.path.join(dir_name, 'depth.png'), save_format='L')
    return


PolyLoader.base_names = tuple()
# data_loader = PolyLoader(
#     './datasets/polyhaven_examples/tile_examples'
# )

PolyLoader.file_type = tuple()
data_loader = PolyLoader(
    './datasets/phone_capture'
)
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
# multi_process(num_process, process, data_loader, attr='files')
