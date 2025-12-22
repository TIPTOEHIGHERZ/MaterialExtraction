import torch
import torch.nn as nn
import sys
import os
import torch.optim.optimizer
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms.functional
import tqdm
from diffusers import DDIMScheduler, AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import argparse
from omegaconf import OmegaConf
from typing import Callable
import random
import numpy as np
import kornia.augmentation as KAug
from torchvision import transforms
import time
import functools

sys.path.append(os.getcwd())
# from model.FeatureExtractor.FeatureExtractor import VisionExtractor, ExtractorRegister
from utils.datasets.DataLoader import (
    PBRTextureDataLoader,
    TestPolyLoader,
)

from model.pipeline import Pipeline
from model.legacy.encoders import PartialSAEncoder

from utils.functionals import (
    random_mask, 
    crop_mid, 
    try_load, 
    freeze_net,
    patch_shuffle,
    generate_gaussian,
    sample_exponential_decay,
    MemoryHalt
)
from utils.transforms import RandomThinPlateSpline
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
from model.controlnet.controlnet import ControlNetCrossAttnModel, ControlUNet2DConditionModel
import utils.trainer as trainer


def get_cropsize(theta: float, d: list[float], base_angle=np.pi / 4, eps=1e-6):
    theta0 = theta - base_angle
    theta1 = theta + base_angle

    d0 = d[0] / abs(np.sin(theta0 + eps))
    d1 = d[1] / abs(np.sin(theta1 + eps))

    l = min(d0, d1)

    return int(l * np.sin(base_angle)), int(l * np.cos(base_angle))


def collate_fn(batch: list[dict]):
    keys = batch[0].keys()

    result_dict = dict()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            result_dict[key] = torch.concat([batch[i][key].unsqueeze(0) for i in range(len(batch))])
        else:
            meta_data = [OmegaConf.load(batch[i][key]) for i in range(len(batch))]     
            result_dict[key] = {key: [meta_data[i][key] for i in range(len(batch))] for key in meta_data[0].keys()}
        
    return result_dict

def align_attr(color, normal, height, roughness, meta_data: dict, transform_func: Callable):
    # FIXME apply transform base on meta data
    bsz = color.shape[0]

    thetas = meta_data['theta']
    concat_attr = [color, normal, height, roughness]
    splits_size = [attr.shape[1] for attr in concat_attr]
    concat_attr = torch.concat(concat_attr, dim=1)

    concat_attr = [torchvision.transforms.functional.rotate(concat_attr[i:i+1], -(thetas[i] / np.pi * 180. + 90.)) for i in range(bsz)]
    crop_sizes = [get_cropsize(thetas[i], concat_attr[i].shape[-2:]) for i in range(bsz)]
    concat_attr = [crop_mid(concat_attr[i], crop_sizes[i]) for i in range(bsz)]
    # scale transform
    # base_distance = 2.78
    # scale_factors = [base_radius / base_distance for radius in meta_data['radius']]
    # concat_attr = [nn.functional.interpolate(concat_attr[i], scale_factor=scale_factors[i]) for i in range(bsz)]

    concat_attr = torch.concat([transform_func(attr) for attr in concat_attr], dim=0)
    color, normal, height, roughness = concat_attr.split(splits_size, dim=1)

    return color, normal, height, roughness


@torch.no_grad()
def evaluate(
    modules: nn.ModuleDict | dict[nn.Module],
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor,
    latents_mask: torch.Tensor,
    latents_shape,
    device='cuda',
    num_inference_steps=50,
    verbose=False,
    cond_scale=3.,
    adapter: FeatureAdapter=None,
    encoder_attention_mask=None,
):  
    unet = modules['unet']
    pipeline.unet.eval()

    # pipeline.scheduler = scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    bsz = masked_ref_latents.shape[0]
    latents_start = torch.randn(bsz, 16, *latents_shape, device=masked_ref_latents.device)

    encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)
    controlnet_input = torch.concat([masked_ref_latents, latents_mask[:, :1]], dim=1)
    if cond_scale is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states, encoder_hidden_states], dim=0)
        controlnet_input = torch.concat([controlnet_input, torch.zeros_like(controlnet_input)], dim=0)
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(bsz)

        latents_input = latents_start
        # latents_input = torch.concat([latents_start, masked_ref_latents], dim=1)
        latents_input_ref = pipeline.scheduler.add_noise(masked_ref_latents, torch.randn_like(masked_ref_latents), t)

        if cond_scale is not None:
            latents_input_null = latents_input
            latents_input = torch.concat([latents_input, latents_input_null], dim=0)
            t = torch.concat([t, t], dim=0)

        noise_pred = unet(
            latents_input,
            t,
            encoder_hidden_states,
            return_dict=False, 
            controlnet_input=controlnet_input,
        )[0]

        if cond_scale is not None:
            noise_pred_cond, noise_pred_null = noise_pred.chunk(2, dim=0)
            # cfg
            noise_pred = noise_pred_null + cond_scale * (noise_pred_cond - noise_pred_null)

        t = t[:1]
        latents_start = pipeline.scheduler.step(noise_pred, t, latents_start, return_dict=False)[0]
    
    # return latents2image(pipeline, latents_start)
    color = pipeline.latents2image(latents_start[:, :4, ...])
    normal = pipeline.latents2image(latents_start[:, 4:8, ...])
    height = pipeline.latents2image(latents_start[:, 8:12, ...])
    roughness = pipeline.latents2image(latents_start[:, 12:, ...])

    return color, normal, height, roughness


@torch.no_grad()
def log_function(
    logger: SummaryWriter,
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    pipeline: Pipeline,
    # feature_encoder: FeatureEncoder,
    module: dict[nn.Module],
    color: torch.Tensor,
    normal: torch.Tensor,
    height: torch.Tensor,
    roughness: torch.Tensor,
    masked_ref_latents: torch.Tensor,
    latents_mask: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    name_space: str=None
):
    log_period_loss = log_config['log_period_loss']
    log_period_visual = log_config['log_period_visual']

    if logger is not None and global_iter % log_period_loss == 0:
        logger.add_scalars(
            'loss',
            avg_loss,
            global_iter
        )
    
    if global_iter % log_period_visual == 0:
        eval_result = evaluate(
            module,
            pipeline,
            masked_ref_latents,
            latents_mask,
            [s // 8 for s in color.shape[-2:]],
            verbose=logger is not None,
            cond_scale=None,
            encoder_attention_mask=encoder_attention_mask
        )

        eval_color, eval_normal, eval_height, eval_roughness = eval_result
        masked_ref_latents = nn.functional.interpolate(pipeline.latents2image(masked_ref_latents), eval_color.shape[-2:])

        if logger is not None:
            logger.add_image(
                'evaluate' if name_space is None else name_space,
                torch.concat([
                    torch.concat([
                        ev_c, ev_n, ev_m, ev_r,c, n, m, r, m_rf_la, 
                    ], dim=-1) for ev_c, ev_n, ev_m, ev_r, c, n, m, r, m_rf_la in \
                        zip(eval_color, eval_normal, eval_height, eval_roughness, color, normal, height, roughness, masked_ref_latents)
                ], dim=-2),
                # torch.concat([
                #     torch.concat([
                #         e_rs, im, latents2image(pipeline, m_rf_la.unsqueeze(0)).squeeze(0), im_ref, 
                #     ], dim=-1) for e_rs, im, m_rf_la, im_ref in zip(eval_result, image, masked_ref_latents, image_ref)
                # ], dim=-2),
                global_iter
            )
    return


@torch.no_grad()
def val_function(
    logger: SummaryWriter,
    global_iter: int,
    log_config: dict,
    dataloader: torch.utils.data.DataLoader,
    module: nn.ModuleDict | dict[nn.Module],
    pipeline: Pipeline,
    name_space: str=None,
    device='cuda',
    **kwargs
):
    log_period_visual = log_config['log_period_visual']
    log_period_loss = log_config['log_period_loss']
    
    unet = module['unet']

    total_loss = 0.
    num_samples = log_config['num_samples']
    batch_size = log_config['batch_size']
    dataset = dataloader.dataset

    idx = torch.randperm(len(dataset))[:batch_size]
    if global_iter % log_period_loss == 0:
        for i in range(num_samples) if logger is None else tqdm.trange(num_samples, leave=False):
            data = dataset[idx]
            meta_data = [OmegaConf.load(d) for d in data['meta_data']]
            data['meta_data'] = {key: [d[key] for d in meta_data] for key in meta_data[0].keys()}

            image_ref, mask_ref, color, normal, height, roughness = data['image'], data['mask'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']
            
            image_ref = image_ref.to(device)
            mask_ref = mask_ref.to(device)

            # color, normal, height, roughness = align_attr(color, normal, height, roughness, data['meta_data'], transform_eval)

            color = color.to(device)
            normal = normal.to(device)
            height = height.to(device)
            roughness = roughness.to(device)

            thresh = 0.5

            mask_ref[mask_ref > thresh] = 1.
            mask_ref[mask_ref <= thresh] = 0.

            bsz = color.shape[0]
                
            color_latents = pipeline.image2latents(color)
            normal_latents = pipeline.image2latents(normal)
            height_latents = pipeline.image2latents(height)
            roughness_latents = pipeline.image2latents(roughness)

            masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
            attn_mask = nn.functional.interpolate(mask_ref, masked_ref_latents.shape[-2:])
            latents_mask = attn_mask
            
            encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)

            t = torch.randint(0, 1000, (bsz,), device=device)

            predict_latents = torch.concat([
                color_latents, normal_latents, height_latents, roughness_latents
            ], dim=1)

            random_noise = torch.randn_like(predict_latents)
            noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

            # 20 channels
            latents_input = noised_latents
            # latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)
            controlnet_input = torch.concat([masked_ref_latents, latents_mask[:, :1]], dim=1)

            noise_pred = unet(
                latents_input,
                t,
                encoder_hidden_states,
                return_dict=False, 
                controlnet_input=controlnet_input,
            )[0]

            loss = nn.functional.mse_loss(noise_pred, random_noise)
            total_loss += loss.item()

            if i >= (num_samples - 1):
                break

        avg_loss = total_loss / (i + 1)

        if logger is not None:
            logger.add_scalar('test_loss', avg_loss, global_step=global_iter)

    if global_iter % log_period_visual == 0:
        eval_result = evaluate(
            module,
            pipeline,
            masked_ref_latents,
            latents_mask,
            [s // 8 for s in color.shape[-2:]],
            verbose=logger is not None,
            cond_scale=None,
            encoder_attention_mask=attn_mask
        )

        eval_color, eval_normal, eval_height, eval_roughness = eval_result
        masked_ref_latents = nn.functional.interpolate(pipeline.latents2image(masked_ref_latents), eval_color.shape[-2:])

        if logger is not None:
            logger.add_image(
                'test_visualize',
                torch.concat([
                    torch.concat([
                        ev_c, ev_n, ev_m, ev_r,c, n, m, r, m_rf_la, 
                    ], dim=-1) for ev_c, ev_n, ev_m, ev_r, c, n, m, r, m_rf_la in \
                        zip(eval_color, eval_normal, eval_height, eval_roughness, color, normal, height, roughness, masked_ref_latents)
                ], dim=-2),
                global_iter
            )

    return


def training_step(module: dict[nn.Module], data, pipeline: Pipeline, device='cuda'):
    unet: UNet2DConditionModel = module['unet']

    image_ref, mask_ref, color, normal, height, roughness = data['image'], data['mask'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']

    image_ref = image_ref.to(device)
    mask_ref: torch.Tensor = mask_ref.to(device)[:, :1, ...]
    # mask_ref: torch.Tensor = mask_ref.to(device)

    ref_flip_angle = random.randint(0, 3) * 90
    image_ref = torchvision.transforms.functional.rotate(image_ref, ref_flip_angle)
    mask_ref = torchvision.transforms.functional.rotate(mask_ref, ref_flip_angle)

    image_ref, mask_ref = transform_ref(torch.concat([image_ref, mask_ref], dim=1)).split([3, 1], dim=1)

    concat_attr = [color, normal, height, roughness]
    splits_size = [attr.shape[1] for attr in concat_attr]
    concat_attr = torch.concat(concat_attr, dim=1)
    concat_attr = torchvision.transforms.functional.rotate(concat_attr, ref_flip_angle)
    color, normal, height, roughness = concat_attr.split(splits_size, dim=1)
    
    color, normal, height, roughness = align_attr(color, normal, height, roughness, data['meta_data'], transform)

    color = color.to(device)
    normal = normal.to(device)
    height = height.to(device)
    roughness = roughness.to(device)

    tps_transform = RandomThinPlateSpline(scale=random.uniform(0.1, 0.3), p=0.8)
    augment_mask = torch.concat([torch.tensor(random_mask(512), device=device).unsqueeze(0) for _ in range(mask_ref.shape[0])], dim=0)
    augment_mask = tps_transform(augment_mask)
    augment_mask = nn.functional.interpolate(augment_mask, mask_ref.shape[-2:])
    mask_ref *= augment_mask

    thresh = 0.5
    mask_ref[mask_ref > thresh] = 1.
    mask_ref[mask_ref <= thresh] = 0.

    bsz = color.shape[0]

    color_latents = pipeline.image2latents(color)
    normal_latents = pipeline.image2latents(normal)
    height_latents = pipeline.image2latents(height)
    roughness_latents = pipeline.image2latents(roughness)

    masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
    attn_mask = nn.functional.interpolate(mask_ref, masked_ref_latents.shape[-2:])
    latents_mask = attn_mask
    
    color: torch.Tensor = (color - 0.5) * 2
    normal: torch.Tensor = (normal - 0.5) * 2
    height: torch.Tensor = (height - 0.5) * 2
    roughness: torch.Tensor = (roughness - 0.5) * 2
    
    encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)

    t = torch.randint(0, 1000, (bsz,), device=device)

    predict_latents = torch.concat([
        color_latents, normal_latents, height_latents, roughness_latents
    ], dim=1)

    random_noise = torch.randn_like(predict_latents)
    noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

    # 16 channels
    latents_input = noised_latents
    # latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)
    controlnet_input = torch.concat([masked_ref_latents, latents_mask[:, :1]], dim=1)

    noise_pred = unet(
        latents_input,
        t,
        encoder_hidden_states,
        return_dict=False, 
        controlnet_input=controlnet_input,
    )[0]

    loss = nn.functional.mse_loss(noise_pred, random_noise)
    # loss_weights = [0.4, 0.2, 0.2, 0.2]

    # loss = 0.
    # for i, weight in enumerate(loss_weights):
    #     loss += nn.functional.mse_loss(noise_pred[:, i * 4: (i + 1) * 4, ...], random_noise[:, i * 4: (i + 1) * 4, ...]) * weight
    # loss /= sum(loss_weights)

    loss_dict = {'loss': loss.item()}

    return_examples = int(min(bsz, 8))

    log_parameters = {
        'color': ((color + 1) / 2)[:return_examples],
        'normal': ((normal + 1) / 2)[:return_examples],
        'height': ((height + 1) / 2)[:return_examples],
        'roughness': ((roughness + 1) / 2)[:return_examples],
        'pipeline': pipeline,
        'module': module,
        # 'feature_encoder': feature_encoder,
        'masked_ref_latents': masked_ref_latents[:return_examples],
        'encoder_attention_mask': attn_mask[:return_examples],
        'latents_mask': latents_mask[:return_examples]
    }

    return loss, loss_dict, log_parameters


def transform(x: torch.Tensor, res=256):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torchvision.transforms.RandomCrop(res)(x)
    # x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])
    # x_ref, mask = ref_transform(x.squeeze(0))

    if in_dim == 3:
        x = x.squeeze(0)

    return x


def transform_ref(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torchvision.transforms.RandomCrop(384)(x)

    if in_dim == 3:
        x = x.squeeze(0)

    return x


def prepare_model(
    batch_size,
    eval_batch_size,
    test_batch_size,
    lr,
    lr_gain,
    ckpt_dir: str=None,
    device='cuda'
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        # beta_start=0.00085,
        # beta_end=0.012,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
        # set_alpha_to_one=False
    )
    scheduler.set_timesteps(1000)
    pipeline: Pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4', scheduler=scheduler)
    config = OmegaConf.load('./configs/legacy/model_config.yaml')
    # unet = instantiate_from_config(config['unet_config'])
    # unet: UNet2DConditionModel = UNet2DConditionModel.from_config(
    #     UNet2DConditionModel.load_config('./pretrained/stable-diffusion-v1-4/unet/config_render.json')
    # )
    # TODO add config path
    unet: ControlUNet2DConditionModel = ControlUNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config('./configs/unet/controlnet/config_main.json')
    )
    # try_load(unet, pipeline.unet)
    pipeline.unet = unet
    controlnet = ControlNetCrossAttnModel.from_config(ControlNetCrossAttnModel.load_config('./configs/controlnet/config_cross.json'))
    unet.apply_controlnet(controlnet)

    # frozen parameters
    pipeline.frozen()

    for param in unet.parameters():
        param.requires_grad_(True)

    for param in controlnet.parameters():
        param.requires_grad_(True)

    params = [
        {'params': list(unet.parameters()), 'lr': lr * lr_gain[0]},
    ]
    
    for i, param in enumerate(params):
        for p in param['params']:
            assert p.requires_grad is True, f'{i} th param has param don\'t require grad'

    optimizer = torch.optim.AdamW(params, lr=lr)
    # optimizer = torch.optim.Adam(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights')
        try_load(unet, torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        # try_load(controlnet, torch.load(os.path.join(ckpt_dir, 'controlnet.ckpt'), weights_only=False, map_location='cpu'))
        # unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        # controlnet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'controlnet.ckpt'), weights_only=False, map_location='cpu'))
        try:
            start_epoch = int(ckpt_dir.split('_')[-1])
        except ValueError:
            start_epoch = 0

    # move to cuda
    pipeline.to(device)

    unet.eval()
    controlnet.eval()

    memory_halt = MemoryHalt()
    memory_halt.halt(int(24 - torch.cuda.memory_allocated() / 1024 ** 3 - 2))

    examples_split = OmegaConf.load('./datasets/MatSynth/examples_split.yaml')
    good_examples = examples_split['good']
    data_split = OmegaConf.load('./datasets/MatSynth/seperate.yaml')
    
    PBRTextureDataLoader.image_names += ('meta_data',)
    data_loader = PBRTextureDataLoader(
        fp='./datasets/render_base_10_resized', 
        gt_fp='./datasets/MatSynth/textures_all_resized', 
        transforms={'default': transform, 'no_transform': lambda x: x},
        fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
        selected_files=None,
        good_examples=good_examples,
        transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    )

    test_loader = PBRTextureDataLoader(
        fp='./datasets/render_base_10_resized', 
        gt_fp='./datasets/MatSynth/textures_all_resized', 
        transforms={'default': transform, 'no_transform': lambda x: x},
        fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
        selected_files=data_split['test'],
        good_examples=good_examples,
        transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    )

    val_loader = TestPolyLoader(
        './datasets/polyhaven_edit'
    )

    memory_halt.release()
    data_loader.shuffle()

    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=eval_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_loader, batch_size=test_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    module = {
        'unet': unet,
    }

    torch.cuda.empty_cache()
    return module, data_loader, test_loader, val_loader, optimizer, start_epoch, pipeline


def main():
    parser = argparse.ArgumentParser('train feature extractor')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size use to train')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='device to use, choise between [\'cuda\', \'cpu\']')
    parser.add_argument('--device_ids', type=int, choices=list(range(8)), default=0,
                        help='if use cuda, the cuda index to use')
    parser.add_argument('--save_period', type=int, default=None,
                        help='save after x epochs')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='total epochs to train')
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume_training', type=bool, default=False)


    args = parser.parse_args()
    
    config_file = args.config_file
    config = OmegaConf.load(config_file)
    train_config = config['train_config']
    log_config = config['log_config']
    eval_config = config['eval_config']
    test_config = config['test_config']

    ckpt_dir = train_config.get('ckpt_dir', None)
    device = train_config['device']
    batch_size = train_config['batch_size']
    train_config['base_lr'] = train_config['lr']
    train_config['lr'] = train_config['lr'] * len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) * batch_size * train_config['accumulation_steps']

    eval_batch_size = eval_config['batch_size']
    eval_config['base_lr'] = train_config['base_lr']
    eval_config['lr'] = eval_config['base_lr'] * len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) * eval_batch_size * train_config['accumulation_steps']

    test_batch_size = test_config['batch_size']

    max_lr = 3.2e-5
    if 'RANK' not in os.environ.keys() or os.environ['RANK'] == '0':
        print(f'setting training lr to batch_size({batch_size})' + \
            f' * n_gpu({len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))})' + \
            f' * accumulation_steps({train_config["accumulation_steps"]})' + \
                f' * lr({eval_config["base_lr"]}) = min({train_config["lr"]}, {max_lr})')
        
        print(f'setting eval lr to batch_size({eval_batch_size})' + \
            f' * n_gpu({len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))})' + \
            f' * accumulation_steps({train_config["accumulation_steps"]})' + \
                f' * lr({eval_config["base_lr"]}) = min({eval_config["lr"]}, {max_lr})')
    
    train_config['lr'] = min(train_config['lr'], max_lr)
    eval_config['lr'] = min(eval_config['lr'], max_lr)

    lr = train_config['lr']

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    
    world_size = torch.cuda.device_count()

    train_config['lr_gain'] = [1.]
    eval_config['lr_gain'] = train_config['lr_gain']

    prepare_config = {
        'device': device, 
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'test_batch_size': test_batch_size,
        'lr':lr,
        'lr_gain': train_config['lr_gain'],
        'ckpt_dir': ckpt_dir,
    }

    rank = 0
    if device == 'cuda' and world_size > 1:
        try:
            rank = int(os.environ['RANK'])
        except KeyError:
            print('not in multi process')
        torch.cuda.set_device(rank)
        print(f'rank {rank} is initialized')

    module, data_loader, eval_loader, test_loader, optimizer, start_epoch, pipeline = prepare_model(**prepare_config)

    trainer.train(
        module,
        training_step, 
        data_loader,
        # eval_loader,
        # test_loader,
        None,
        eval_loader,
        optimizer, 
        train_config,
        # eval_config,
        None,
        test_config,
        log_function,
        log_config,
        val_function,
        train_args=(),
        train_kwargs={
            'pipeline': pipeline,
            'device': device,
        }
    )

    return


if __name__ == '__main__':
    main()

