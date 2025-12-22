import omegaconf
import tqdm
import time
import os
import sys
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, PNDMScheduler, DDPMScheduler, DiffusionPipeline
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.finetune_render_pretrained import evaluate, prepare_model, transform
from utils.datasets.DataLoader import PBRTextureDataLoader, TestPolyLoader, PolyLoader
from utils.io import load_image, save_image
from utils.project import project


parser = argparse.ArgumentParser()
parser.add_argument('--cond_scale', type=float, default=None)
args = parser.parse_args()
cond_scale = args.cond_scale

torch.cuda.set_device(3)

config = omegaconf.OmegaConf.load('./configs/finetune_render_pretrained.yaml')
train_config = config['train_config']
device = train_config['device']

prepare_config = {
    'device': train_config['device'], 
    'batch_size': train_config['batch_size'],
    'eval_batch_size': config['eval_config']['batch_size'],
    'test_batch_size': config['test_config']['batch_size'],
    'lr': 0.,
    'lr_gain': [1., 1.],
    'ckpt_dir': train_config['ckpt_dir'],
}

module, data_loader, eval_loader, test_loader, optimizer, start_epoch, pipeline = prepare_model(**prepare_config)

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0015,
    beta_end=0.0195,
    beta_schedule="scaled_linear",
    clip_sample=False,
)
pipeline.scheduler = scheduler

PolyLoader.base_names = tuple()
PolyLoader.image_names = ('image_project', 'mask_project', 'image', 'mask')
# PolyLoader.image_names = ('image', 'mask')
loader = PolyLoader(
    './datasets/poly_test/',
)

current_time = time.localtime()
save_dir = os.path.join('eval', f'{time.strftime("%Y-%m-%d %H:%M:%S", current_time)}_{cond_scale: .3f}_cfg')
print(f'saving to {save_dir}')

num_samples = len(loader)
with torch.no_grad():
    for i in tqdm.trange(num_samples):
        data = loader[i]
        for key in data.keys():
            data[key] = data[key].unsqueeze(0)
        # image_ref, mask_ref, color, normal, height, roughness = data['image'], data['mask'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']
        image_ref, mask_ref, color, normal, height, roughness = data['image_project'], data['mask_project'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']

        reference = torch.concat([image_ref, mask_ref], dim=1)
        # reference = transform(reference)
        image_ref = reference[:, :image_ref.shape[1]]
        mask_ref = reference[:, image_ref.shape[1]:]

        image_ref = image_ref.to(device)
        mask_ref = mask_ref.to(device)
        color = color.to(device)
        normal = normal.to(device)
        height = height.to(device)
        roughness = roughness.to(device)

        thresh = 0.5

        mask_ref = mask_ref[:, :1, ...]
        mask_ref[mask_ref > thresh] = 1.
        mask_ref[mask_ref <= thresh] = 0.

        bsz = color.shape[0]
        # image_ref *= mask_ref

        masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
        latents_mask = nn.functional.interpolate(mask_ref, masked_ref_latents.shape[-2:])

        encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)

        eval_result = evaluate(
            pipeline,
            masked_ref_latents,
            # [s // 8 for s in color.shape[-2:]],
            [64, 64],
            verbose=True,
            cond_scale=cond_scale,
            # cond_scale=None,
            encoder_attention_mask=None,
            num_inference_steps=50
        )
        eval_color, eval_normal, eval_height, eval_roughness = eval_result
        
        path = os.path.basename(os.path.dirname(loader.files[i]['Color']))
        os.makedirs(os.path.join(save_dir, path), exist_ok=True)
        save_image(eval_color[0], os.path.join(save_dir, path, 'color.png'))
        save_image(eval_normal[0], os.path.join(save_dir, path, 'normal.png'))
        save_image(eval_height[0], os.path.join(save_dir, path, 'height.png'))
        save_image(eval_roughness[0], os.path.join(save_dir, path, 'roughness.png'))

        save_image(color, os.path.join(save_dir, path, 'color_gt.png'))
        save_image(normal, os.path.join(save_dir, path, 'normal_gt.png'))
        save_image(height, os.path.join(save_dir, path, 'height_gt.png'))
        save_image(roughness, os.path.join(save_dir, path, 'roughness_gt.png'))

        save_image(pipeline.latents2image(masked_ref_latents), os.path.join(save_dir, path, 'reference.png'))

