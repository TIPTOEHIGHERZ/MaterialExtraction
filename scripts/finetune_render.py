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
import accelerate
import random
import numpy as np
import kornia.augmentation as KAug
from torchvision import transforms

sys.path.append(os.getcwd())
# from model.FeatureExtractor.FeatureExtractor import VisionExtractor, ExtractorRegister
from model.FeatureExtractor.EmbeddingMatcher import EmbeddingMatcher
from model.oned_tokenizer.modeling.titok import TiTok
from utils.datasets import VirtualLoader
from utils.datasets import TextureDataLoader
from utils.datasets.DataLoader import (
    PexelLoader, 
    ImageDataLoader,
    PBRTextureDataLoader,
    FolderLoader
)

from model.pipeline import Pipeline
from model.GaussianSampler.GaussianSampler import GaussianSampler
from model.Lora import LoraRegister
from model.FeatureExtractor.FeatureAugmentor import FeatureAugmentorRegister
from model.legacy.encoders import PartialSAEncoder
from model.legacy.diffusionmodules.openaimodel import UNetModel
from model.legacy.diffusionmodules.util import instantiate_from_config

from utils.transforms import RandomThinPlateSpline
from utils.functionals import (
    make_mask, 
    crop_mid, 
    try_load, 
    freeze_net,
    patch_shuffle,
    generate_gaussian,
    sample_exponential_decay,
)
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
from model.unet.attn_ref import AttentionReference
import utils.trainer as trainer
from model.legacy.diffusionmodules.openaimodel import UNetModel
from model.legacy.diffusionmodules.util import instantiate_from_config
from model.legacy.texture import RandomMask
from model.ema import LitEma


class FeatureEncoder(PartialSAEncoder):
    def __init__(
            self, 
            in_channels=3, 
            context_dim=64,
            image_dim=4096,
            cross_attn_dim=768,
            key='c_crossattn'
        ):
        super().__init__(in_channels, context_dim, key)

        self.proj_out = nn.Linear(image_dim, cross_attn_dim)

        return
    
    def forward(self, x, mask):
        x = super().forward(x, mask)
        x = self.proj_out(x)

        return x
    

def image2latents(pipeline, image, mask=None):
    if mask is not None:
        image *= mask
    return pipeline.vae.encode(image)


def latents2image(pipeline, image):
    return (torch.clamp(pipeline.vae.decode(image), -1., 1.) + 1) / 2


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor,
    device='cuda',
    num_inference_steps=50,
    verbose=False,
    cond_scale=3.,
    adapter: FeatureAdapter=None,
):    
    pipeline.unet.eval()

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    bsz = masked_ref_latents.shape[0]
    latents_shape = masked_ref_latents.shape[-2:]
    latents_start = torch.randn(bsz, 4, *latents_shape, device=masked_ref_latents.device)

    # encoder_hidden_states = feature_encoder(image_ref, mask_ref).transpose(-1, -2)
    # if int(os.environ['RANK']) == 0:
    #     print(image_ref.max(), image_ref.min(), mask_ref.max(), mask_ref.min())
    encoder_hidden_states = None
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(latents_start.shape[0])

        latents_input = torch.concat([latents_start, masked_ref_latents], dim=1)
        latents_input_ref = masked_ref_latents

        if cond_scale is not None:
            latents_input_null = torch.concat([latents_start, torch.zeros_like(masked_ref_latents)], dim=1)
            # latents_input_null = torch.concat([latents_start, torch.zeros_like(mask_latents), torch.zeros_like(masked_ref_latents)], dim=1)
            latents_input = torch.concat([latents_input, latents_input_null], dim=0)

        noise_pred = pipeline.unet(
            latents_input,
            latents_input_ref,
            t, 
            encoder_hidden_states, 
            return_dict=False, 
        )[0]

        # noise_pred = pipeline.unet(
        #     latents_input,
        #     t,
        #     encoder_hidden_states
        # )

        if cond_scale is not None:
            noise_pred_cond, noise_pred_null = noise_pred.chunk(2, dim=0)
            # cfg
            noise_pred = noise_pred_null + cond_scale * (noise_pred_cond - noise_pred_null)

        t = t[:1]
        latents_start = pipeline.scheduler.step(noise_pred, t, latents_start, return_dict=False)[0]
    
    # return latents2image(pipeline, latents_start)
    color = pipeline.latents2image(latents_start[:, :4, ...])
    # normal = pipeline.latents2image(latents_start[:, 4:, ...])
    # return color, normal
    return color


@torch.no_grad()
def log_function(
    logger: SummaryWriter,
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    pipeline: Pipeline,
    # feature_encoder: FeatureEncoder,
    color: torch.Tensor,
    normal: torch.Tensor,
    masked_ref_latents: torch.Tensor,
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
            masked_ref_latents,
            verbose=logger is not None,
            cond_scale=None
        )

        eval_color = eval_result

        if logger is not None:
            logger.add_image(
                'evaluate' if name_space is None else name_space,
                torch.concat([
                    torch.concat([
                        ev_c, c, pipeline.latents2image(m_rf_la.unsqueeze(0)).squeeze(0), im_ref, 
                    ], dim=-1) for ev_c, c, m_rf_la, im_ref in zip(eval_color, color, masked_ref_latents, image_ref)
                ], dim=-2),
                # torch.concat([
                #     torch.concat([
                #         e_rs, im, latents2image(pipeline, m_rf_la.unsqueeze(0)).squeeze(0), im_ref, 
                #     ], dim=-1) for e_rs, im, m_rf_la, im_ref in zip(eval_result, image, masked_ref_latents, image_ref)
                # ], dim=-2),
                global_iter
            )
    return


def training_step(module: list[nn.Module], data, pipeline: Pipeline, device='cuda'):
    unet = module[0]

    image_ref, mask_ref, color, normal = data
    color = color.to(device)
    normal = normal.to(device)
    image_ref = image_ref.to(device)
    mask_ref: torch.Tensor = mask_ref.to(device)[:, :1, ...]
    # mask_ref: torch.Tensor = mask_ref.to(device)

    flip_angle = random.randint(0, 3) * 90.
    image_ref = torchvision.transforms.functional.rotate(image_ref, flip_angle)
    mask_ref = torchvision.transforms.functional.rotate(mask_ref, flip_angle)

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
    image_ref *= mask_ref
    
    color_latents = pipeline.image2latents(color)
    # normal_latents = pipeline.image2latents(normal)

    masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
    
    color: torch.Tensor = (color - 0.5) * 2
    normal: torch.Tensor = (normal - 0.5) * 2
    image_ref: torch.Tensor = (image_ref - 0.5) * 2
    
    # image_latents = image2latents(pipeline, image)
    # masked_ref_latents = image2latents(pipeline, image_ref * mask_ref)
    
    encoder_hidden_states = None

    t = torch.randint(0, 1000, (bsz,), device=device)
    # t = sample_exponential_decay(bsz, min_val=0, max_val=1000, decay_rate=3.5e-3).to(device)

    # predict_latents = torch.concat([color_latents, normal_latents], dim=1)
    predict_latents = color_latents

    random_noise = torch.randn_like(predict_latents)
    noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

    # 12 channels
    latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)
    latents_input_ref = masked_ref_latents

    noise_pred = unet(
        latents_input,
        latents_input_ref,
        t, 
        encoder_hidden_states, 
        return_dict=False, 
    )[0]

    # noise_pred = pipeline.unet(
    #     latents_input,
    #     t,
    #     encoder_hidden_states
    # )

    loss = nn.functional.mse_loss(noise_pred, random_noise)
    loss_dict = {'loss': loss.item()}

    return_examples = int(min(bsz, 8))

    log_parameters = {
        'color': ((color + 1) / 2)[:return_examples],
        'normal': ((normal + 1) / 2)[:return_examples],
        'pipeline': pipeline,
        # 'feature_encoder': feature_encoder,
        'masked_ref_latents': masked_ref_latents[:return_examples],
        'image_ref': ((image_ref + 1) / 2)[:return_examples],
        'mask_ref': mask_ref[:return_examples],
    }

    return loss, loss_dict, log_parameters


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    

    x = torch.nn.functional.interpolate(crop_mid(x), [256, 256])
    x = torchvision.transforms.RandomCrop(128)(x)
    # x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])
    # x_ref, mask = ref_transform(x.squeeze(0))

    if in_dim == 3:
        x = x.squeeze(0)

    return x


def transform_eval(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torch.nn.functional.interpolate(crop_mid(x), [256, 256])
    # x = torchvision.transforms.RandomCrop(512)(x)
    # x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])
    # x_ref, mask = ref_transform(x.squeeze(0))

    if in_dim == 3:
        x = x.squeeze(0)

    return x


def prepare_model(
    batch_size,
    eval_batch_size,
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
    unet = SiameseUnet.from_config('./configs/unet/render_color', config_main='config_main.json', config_ref='config_ref.json')
    pipeline.unet = unet

    # vae = instantiate_from_config(config['first_stage_config'])
    # pipeline.vae = vae
    
    unet_config = OmegaConf.load('./pretrained/stable-diffusion-v1-4/unet/config_render.json')

    cross_attention_dim = unet_config['cross_attention_dim']
    # feature_encoder = instantiate_from_config(config['cond_stage_config'])
    # feature_encoder = FeatureEncoder(in_channels=3, context_dim=1024, cross_attn_dim=cross_attention_dim, image_dim=4096)
    # feature_encoder = PartialSAEncoder(4, cross_attention_dim)

    # frozen parameters
    pipeline.frozen()
    unet_main_params = unet.unfrozen_main()
    unet_ref_params = unet.unfrozen_ref()
    unet_params = unet_main_params + unet_ref_params

    params = [
        # {'params': list(feature_encoder.parameters()), 'lr': lr},
        {'params': unet_params, 'lr': lr * lr_gain[0]},
        # {'params': list(pipeline.unet.parameters()), 'lr': lr},
    ]
    
    for i, param in enumerate(params):
        for p in param['params']:
            assert p.requires_grad is True, f'{i} th param has param don\'t require grad'

    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights')
        # unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        # try_load(unet, torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        # feature_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'feature_encoder.ckpt'), weights_only=False, map_location='cpu'))
        # optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=False, map_location='cpu'))
        try:
            start_epoch = int(ckpt_dir.split('_')[-1])
        except ValueError:
            start_epoch = 0

    # move to cuda
    pipeline.to(device)
    # feature_encoder.to(device)

    # feature_encoder.eval()
    unet.eval()

    data_loader = PBRTextureDataLoader(
        fp='./datasets/render_result_matsynth_resized_noscale', 
        gt_fp='./datasets/MatSynth/textures_resized', 
        transform=transform,
        # no_subdir=True
    )

    eval_loader = PBRTextureDataLoader(
        fp='./datasets/render_result_matsynth_resized_noscale', 
        gt_fp='./datasets/MatSynth/textures_resized', 
        transform=transform_eval,
        # no_subdir=True
    )

    # data_loader = FolderLoader(fp='./datasets/MatSynth/color', transform=transform)
    data_loader.shuffle()

    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=eval_batch_size, shuffle=False, num_workers=4)

    # return pipeline, data_loader, optimizer, lora_register, lora_register_ref, titok_tokenizer, adapter, start_epoch
    module = {
        'unet': unet,
        # 'feature_encoder': feature_encoder
    }
    return module, data_loader, eval_loader, optimizer, start_epoch, pipeline


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

    ckpt_dir = train_config.get('ckpt_dir', None)
    device = train_config['device']
    batch_size = train_config['batch_size']
    train_config['base_lr'] = train_config['lr']
    train_config['lr'] = train_config['lr'] * len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) * batch_size * train_config['accumulation_steps']

    eval_batch_size = eval_config['batch_size']
    eval_config['base_lr'] = train_config['base_lr']
    eval_config['lr'] = eval_config['base_lr'] * len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) * eval_batch_size * train_config['accumulation_steps']

    max_lr = 5e-5
    if os.environ['RANK'] == '0':
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

    train_config['lr_gain'] = [1.]
    eval_config['lr_gain'] = train_config['lr_gain']

    prepare_config = {
        'device': device, 
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
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

    module, data_loader, eval_loader, optimizer, start_epoch, pipeline = prepare_model(**prepare_config)

    trainer.train(
        module,
        training_step, 
        data_loader,
        eval_loader,
        optimizer, 
        train_config,
        eval_config,
        log_function,
        log_config,
        train_args=(),
        train_kwargs={
            'pipeline': pipeline,
            'device': device,
        }
    )

    return


if __name__ == '__main__':
    main()

