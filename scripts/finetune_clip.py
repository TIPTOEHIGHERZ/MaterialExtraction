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
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPModel
import argparse
from omegaconf import OmegaConf
from typing import Callable
import accelerate
import random
import numpy as np
import kornia.augmentation as KAug
from torchvision import transforms
import functools

sys.path.append(os.getcwd())
# from model.FeatureExtractor.FeatureExtractor import VisionExtractor, ExtractorRegister
from utils.datasets.DataLoader import (
    PBRTextureDataLoader,
    TestPolyLoader
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
    MemoryHalt,
)
from utils.transforms import RandomThinPlateSpline
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
import utils.trainer as trainer


class ClipTransformer(nn.Module):
    def __init__(self, vision_transformer: CLIPVisionTransformer):
        super().__init__()
        self.vision_transformer = vision_transformer
        self.proj_out = nn.Linear(1024, 768)

        return
    
    def forward(self, x: torch.Tensor):
        x = self.vision_transformer(x).pooler_output.unsqueeze(1)
        x = self.proj_out(x)

        return x

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
            try:
                result_dict[key] = torch.concat([batch[i][key].unsqueeze(0) for i in range(len(batch))])
            except Exception as e:
                print([batch[i][key].shape for i in range(len(batch))])
                raise
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
    # radius_min, radius_max = 2., 3.
    # base_radius = 3.
    # scale_factors = [base_radius / radius for radius in meta_data['radius']]
    # concat_attr = [nn.functional.interpolate(concat_attr[i], scale_factor=scale_factors[i]) for i in range(bsz)]

    concat_attr = torch.concat([transform_func(attr) for attr in concat_attr], dim=0)
    color, normal, height, roughness = concat_attr.split(splits_size, dim=1)

    return color, normal, height, roughness


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    module: nn.ModuleDict,
    image_ref: torch.Tensor,
    mask_ref: torch.Tensor,
    latents_shape,
    device='cuda',
    num_inference_steps=50,
    verbose=False,
    cond_scale=3.,
    adapter: FeatureAdapter=None,
):    
    pipeline.unet.eval()
    clip_model = module['clip']

    # pipeline.scheduler = scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    bsz = image_ref.shape[0]
    latents_start = torch.randn(bsz, 16, *latents_shape, device=image_ref.device)

    image_ref = (image_ref - 0.5) * 2 
    image_ref *= mask_ref
    encoder_hidden_states = clip_model(nn.functional.interpolate(image_ref, [224, 224]))
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(bsz)

        latents_input = latents_start

        encoder_hidden_states_input = encoder_hidden_states
        if cond_scale is not None:
            latents_input_null = latents_input
            latents_input = torch.concat([latents_input, latents_input_null], dim=0)

            encoder_hidden_states_input = torch.concat([encoder_hidden_states_input, torch.zeros_like(encoder_hidden_states_input)], dim=0)

        noise_pred = pipeline.unet(
            latents_input,
            t, 
            encoder_hidden_states_input,
            return_dict=False, 
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
    module: dict,
    color: torch.Tensor,
    normal: torch.Tensor,
    height: torch.Tensor,
    roughness: torch.Tensor,
    image_ref: torch.Tensor,
    mask_ref: torch.Tensor,
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
            pipeline,
            module,
            image_ref,
            mask_ref,
            [s // 8 for s in color.shape[-2:]],
            verbose=logger is not None,
            cond_scale=None,
        )

        eval_color, eval_normal, eval_height, eval_roughness = eval_result
        image_ref = nn.functional.interpolate(image_ref, color.shape[-2:])

        if logger is not None:
            logger.add_image(
                'evaluate' if name_space is None else name_space,
                torch.concat([
                    torch.concat([
                        ev_c, ev_n, ev_m, ev_r,c, n, m, r, m_rf_la, 
                    ], dim=-1) for ev_c, ev_n, ev_m, ev_r, c, n, m, r, m_rf_la in \
                        zip(eval_color, eval_normal, eval_height, eval_roughness, color, normal, height, roughness, image_ref)
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
    pipeline: Pipeline,
    module: nn.ModuleDict,
    name_space: str=None,
    device='cuda',
    **kwargs
):
    log_period_visual = log_config['log_period_visual']
    log_period_loss = log_config['log_period_loss']
    
    unet = module['unet']
    clip_model = module['clip']

    # total_samples = len(test_bar)
    total_loss = 0.
    num_samples = log_config['num_samples']
    batch_size = log_config['batch_size']
    dataset = dataloader.dataset

    idx = torch.randperm(len(dataset))[:batch_size]
    if global_iter % log_period_loss == 0:
        for i in range(num_samples) if logger is None else tqdm.trange(num_samples, leave=False):
            data = dataset[idx]
            image_ref, mask_ref, color, normal, height, roughness = data['image'], data['mask'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']
            
            image_ref = image_ref.to(device)
            mask_ref = mask_ref.to(device)
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
            
            # color: torch.Tensor = (color - 0.5) * 2
            # normal: torch.Tensor = (normal - 0.5) * 2
            # height: torch.Tensor = (height - 0.5) * 2
            # roughness: torch.Tensor = (roughness - 0.5) * 2

            image_ref: torch.Tensor = (image_ref - 0.5) * 2
            image_ref *= mask_ref
            
            encoder_hidden_states = clip_model(nn.functional.interpolate(image_ref, [224, 224]))

            t = torch.randint(0, 1000, (bsz,), device=device)

            predict_latents = torch.concat([
                color_latents, normal_latents, height_latents, roughness_latents
            ], dim=1)

            random_noise = torch.randn_like(predict_latents)
            noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

            # 16 channels
            latents_input = noised_latents
            # 20 channels
            # latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)
            latents_input_ref = masked_ref_latents

            noise_pred = unet(
                latents_input,
                t, 
                encoder_hidden_states, 
                return_dict=False, 
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
            pipeline,
            module,
            image_ref,
            mask_ref,
            [s // 8 for s in color.shape[-2:]],
            verbose=logger is not None,
            cond_scale=None,
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


def training_step(module: list[nn.Module], data, pipeline: Pipeline, device='cuda'):
    unet = module['unet']
    clip_model: CLIPVisionTransformer = module['clip']

    image_ref, mask_ref, color, normal, height, roughness = data['image'], data['mask'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']
    color = color.to(device)
    normal = normal.to(device)
    height = height.to(device)
    roughness = roughness.to(device)

    image_ref = image_ref.to(device)
    mask_ref: torch.Tensor = mask_ref.to(device)[:, :1, ...]
    # mask_ref: torch.Tensor = mask_ref.to(device)

    ref_flip_angle = random.randint(0, 3) * 90.
    image_ref = torchvision.transforms.functional.rotate(image_ref, ref_flip_angle)
    mask_ref = torchvision.transforms.functional.rotate(mask_ref, ref_flip_angle)
    image_ref = nn.functional.interpolate(image_ref, [256, 256])
    mask_ref = nn.functional.interpolate(mask_ref, [256, 256])

    # rotation augmentation for targets
    targets = [color, normal, height, roughness]
    target_splits = [t.shape[1] for t in targets]
    targets = torch.concat(targets, dim=1)
    targets = nn.functional.interpolate(targets, [256, 256])
    # targets = torchvision.transforms.functional.rotate(targets, ref_flip_angle)
    color, normal, height, roughness = targets.split(target_splits, dim=1)

    # color, normal, height, roughness = align_attr(color, normal, height, roughness, data['meta_data'], functools.partial(transform))

    tps_transform = RandomThinPlateSpline(scale=random.uniform(0.1, 0.3), p=0.8)
    augment_mask = torch.concat([torch.tensor(random_mask(512), device=device).unsqueeze(0) for _ in range(mask_ref.shape[0])], dim=0)
    augment_mask = tps_transform(augment_mask)
    augment_mask = nn.functional.interpolate(augment_mask, mask_ref.shape[-2:])
    mask_ref *= augment_mask

    rotate_angels = [(random.random() * 2 - 1) * 10. for _ in range(image_ref.shape[0])]
    image_ref = torch.concat(
        [torchvision.transforms.functional.rotate(image_ref[i: i + 1, ...], rotate_angels[i]) for i in range(image_ref.shape[0])]
    )

    mask_ref = torch.concat(
        [torchvision.transforms.functional.rotate(mask_ref[i: i + 1, ...], rotate_angels[i]) for i in range(mask_ref.shape[0])]
    )

    thresh = 0.5
    mask_ref[mask_ref > thresh] = 1.
    mask_ref[mask_ref <= thresh] = 0.

    bsz = color.shape[0]
        
    color_latents = pipeline.image2latents(color)
    normal_latents = pipeline.image2latents(normal)
    height_latents = pipeline.image2latents(height)
    roughness_latents = pipeline.image2latents(roughness)
    
    color: torch.Tensor = (color - 0.5) * 2
    normal: torch.Tensor = (normal - 0.5) * 2
    height: torch.Tensor = (height - 0.5) * 2
    roughness: torch.Tensor = (roughness - 0.5) * 2

    image_ref: torch.Tensor = (image_ref - 0.5) * 2
    image_ref *= mask_ref
    
    encoder_hidden_states = clip_model(nn.functional.interpolate(image_ref, [224, 224]))

    t = torch.randint(0, 1000, (bsz,), device=device)

    predict_latents = torch.concat([
        color_latents, normal_latents, height_latents, roughness_latents
    ], dim=1)

    # predict_latents = torch.concat([
    #     color_latents, normal_latents
    # ], dim=1)

    random_noise = torch.randn_like(predict_latents)
    noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

    # 16 channels
    latents_input = noised_latents

    noise_pred = unet(
        latents_input,
        t,
        encoder_hidden_states,
        return_dict=False, 
    )[0]

    loss = nn.functional.mse_loss(noise_pred, random_noise)

    # loss = nn.functional.mse_loss(noise_pred, random_noise)
    loss_dict = {'loss': loss.item()}

    return_examples = int(min(bsz, 8))

    log_parameters = {
        'color': ((color + 1) / 2)[:return_examples],
        'normal': ((normal + 1) / 2)[:return_examples],
        'height': ((height + 1) / 2)[:return_examples],
        'roughness': ((roughness + 1) / 2)[:return_examples],
        'pipeline': pipeline,
        'module': module,
        'image_ref': ((image_ref + 1) / 2)[:return_examples],
        'mask_ref': mask_ref[:return_examples],
    }

    return loss, loss_dict, log_parameters


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torchvision.transforms.RandomCrop(256)(x)
    # x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])
    # x_ref, mask = ref_transform(x.squeeze(0))

    if in_dim == 3:
        x = x.squeeze(0)

    return x


def transform_eval(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    # x = torchvision.transforms.RandomCrop(512)(x)
    # x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])
    # x_ref, mask = ref_transform(x.squeeze(0))

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
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config('configs/unet/render_all/config_main.json'))
    # unet.unet_ref.load_state_dict(pipeline.unet.state_dict())
    pipeline.unet = unet
    # frozen parameters
    pipeline.frozen()
    clip_model = ClipTransformer(CLIPModel.from_pretrained('./pretrained/clip-vit-large-patch14').vision_model)

    for param in unet.parameters():
        param.requires_grad_(True)
    
    for param in clip_model.vision_transformer.parameters():
        param.requires_grad_(False)
    
    for param in clip_model.proj_out.parameters():
        param.requires_grad_(True)
    
    # for param in clip_model.parameters():
    #     param.requires_grad_(True)

    params = [
        {'params': list(unet.parameters()), 'lr': lr},
        {'params': list(clip_model.proj_out.parameters()), 'lr': lr},
    ]
    
    for i, param in enumerate(params):
        for p in param['params']:
            assert p.requires_grad is True, f'{i} th param has param don\'t require grad'

    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights')
        # try_load(unet, torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        # optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=False, map_location='cpu'))
        try:
            start_epoch = int(ckpt_dir.split('_')[-1])
        except ValueError:
            start_epoch = 0

    # move to cuda
    pipeline.to(device)
    clip_model.to(device)
    # feature_encoder.to(device)

    # feature_encoder.eval()
    unet.eval()
    examples_split = OmegaConf.load('./datasets/MatSynth/examples_split.yaml')
    good_examples = examples_split['good']
    # data_split = OmegaConf.load('./datasets/render_base_10_resized/seperate.yaml')

    memory_halt = MemoryHalt()
    # memory_halt.halt(int(20 - torch.cuda.memory_allocated() / 1024 ** 3 - 2))

    PBRTextureDataLoader.image_names += ('meta_data',)
    data_loader = PBRTextureDataLoader(
        fp='./datasets/render_base_10_resized', 
        gt_fp='./datasets/MatSynth/textures_all_resized', 
        transforms={'default': transform, 'no_transform': lambda x: x},
        fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
        selected_files=None,
        good_examples=None,
        # good_examples=good_examples,
        transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    )

    test_loader = PBRTextureDataLoader(
        fp='./datasets/render_base_10_resized', 
        gt_fp='./datasets/MatSynth/textures_all_resized', 
        transforms={'default': transform, 'no_transform': lambda x: x},
        fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
        selected_files=None,
        good_examples=None,
        # good_examples=good_examples,
        transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    )

    # data_loader = FolderLoader(fp='./datasets/MatSynth/color', transform=transform)
    data_loader.shuffle()
    # memory_halt.release()

    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=eval_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_loader, batch_size=test_batch_size, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader()
    val_loader = None

    # return pipeline, data_loader, optimizer, lora_register, lora_register_ref, titok_tokenizer, adapter, start_epoch
    module = {
        'unet': unet,
        'clip': clip_model
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

    max_lr = 5e-5
    if 'RANK' not in os.environ.keys() or os.environ['RANK'] == '0':
        print(f'setting training lr to batch_size({batch_size})' + \
            f' * n_gpu({len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))})' + \
            f' * accumulation_steps({train_config['accumulation_steps']})' + \
                f' * lr({eval_config['base_lr']}) = min({train_config['lr']}, {max_lr})')
        
        print(f'setting eval lr to batch_size({eval_batch_size})' + \
            f' * n_gpu({len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))})' + \
            f' * accumulation_steps({train_config['accumulation_steps']})' + \
                f' * lr({eval_config['base_lr']}) = min({eval_config['lr']}, {max_lr})')
    
    train_config['lr'] = min(train_config['lr'], max_lr)
    eval_config['lr'] = min(eval_config['lr'], max_lr)

    lr = train_config['lr']

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    
    world_size = torch.cuda.device_count()

    train_config['lr_gain'] = [1., 1.]
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

    module, data_loader, eval_loader, test_loader,optimizer, start_epoch, pipeline = prepare_model(**prepare_config)

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

