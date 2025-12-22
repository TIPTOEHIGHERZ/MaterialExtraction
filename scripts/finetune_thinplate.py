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
    sample_exponential_decay
)
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
from model.unet.attn_ref import AttentionReference
from model.oned_tokenizer.modeling.titok import TiTok
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
        x = super().forward(x, mask).transpose(-1, -2)
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
    feature_encoder: PartialSAEncoder,
    masked_ref_latents: torch.Tensor,
    image_ref: torch.Tensor,
    mask_ref: torch.Tensor,
    device='cuda',
    num_inference_steps=200,
    verbose=False,
    cond_scale=3.,
    adapter: FeatureAdapter=None,
):   
    original_scheduler = pipeline.scheduler
    ddim_scheduler = DDIMScheduler(
        1000,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        timestep_spacing='linspace',
        clip_sample=False,
    )
    pipeline.scheduler = ddim_scheduler

    pipeline.unet.eval()
    feature_encoder.eval()

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    latents_start = torch.randn_like(masked_ref_latents)

    # encoder_hidden_states = feature_encoder(image_ref, mask_ref).transpose(-1, -2)
    # if int(os.environ['RANK']) == 0:
    #     print(image_ref.max(), image_ref.min(), mask_ref.max(), mask_ref.min())
    encoder_hidden_states = feature_encoder(image_ref, mask_ref)

    if cond_scale is not None:
        encoder_hidden_states_null = feature_encoder(torch.zeros_like(image_ref), torch.zeros_like(mask_ref)).transpose(-1, -2)
        encoder_hidden_states = torch.concat([encoder_hidden_states, encoder_hidden_states_null], dim=0)
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(latents_start.shape[0])

        latents_input = torch.concat([latents_start, masked_ref_latents], dim=1)
        # latents_input = torch.concat([latents_start, mask_latents, masked_ref_latents], dim=1)

        if cond_scale is not None:
            latents_input_null = torch.concat([latents_start, torch.zeros_like(masked_ref_latents)], dim=1)
            # latents_input_null = torch.concat([latents_start, torch.zeros_like(mask_latents), torch.zeros_like(masked_ref_latents)], dim=1)
            latents_input = torch.concat([latents_input, latents_input_null], dim=0)

        # noise_pred = pipeline.unet(
        #     latents_input, 
        #     t, 
        #     encoder_hidden_states, 
        #     return_dict=False, 
        # )[0]

        noise_pred = pipeline.unet(
            latents_input,
            t,
            encoder_hidden_states
        )

        if cond_scale is not None:
            noise_pred_cond, noise_pred_null = noise_pred.chunk(2, dim=0)
            # cfg
            noise_pred = noise_pred_null + cond_scale * (noise_pred_cond - noise_pred_null)

        t = t[:1]
        latents_start = pipeline.scheduler.step(noise_pred, t, latents_start, eta=1., return_dict=False)[0]
    
    pipeline.scheduler = original_scheduler
    return latents2image(pipeline, latents_start)
    # return pipeline.latents2image(latents_start)


@torch.no_grad()
def log_function(
    logger: SummaryWriter,
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    pipeline: Pipeline,
    feature_encoder: FeatureEncoder,
    image: torch.Tensor,
    masked_ref_latents: torch.Tensor,
    image_ref: torch.Tensor,
    mask_ref: torch.Tensor
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
            feature_encoder,
            masked_ref_latents,
            image_ref * 2 - 1,
            mask_ref,
            verbose=logger is not None,
            cond_scale=None
        )

        if logger is not None:
            logger.add_image(
                'evaluate',
                # torch.concat([
                #     torch.concat([
                #         e_rs, im, pipeline.latents2image(m_rf_la.unsqueeze(0)).squeeze(0), im_ref, 
                #     ], dim=-1) for e_rs, im, m_rf_la, im_ref in zip(eval_result, image, masked_ref_latents, image_ref)
                # ], dim=-2),
                torch.concat([
                    torch.concat([
                        e_rs, im, latents2image(pipeline, m_rf_la.unsqueeze(0)).squeeze(0), im_ref, 
                    ], dim=-1) for e_rs, im, m_rf_la, im_ref in zip(eval_result, image, masked_ref_latents, image_ref)
                ], dim=-2),
                global_iter
            )
    return


def training_step(module: list[nn.Module], data, pipeline: Pipeline, device='cuda'):
    unet, feature_encoder = module

    image, image_ref, mask_ref = data
    image = image.to(device)
    image_ref = image.to(device)
    # mask_ref: torch.Tensor = mask_ref.to(device)[:, :1, ...]
    mask_ref: torch.Tensor = mask_ref.to(device)

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

    bsz = image.shape[0]
    image_ref *= mask_ref
    
    # image_latents = pipeline.image2latents(image)
    # masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
    
    image: torch.Tensor = (image.to(device) - 0.5) * 2
    image_ref: torch.Tensor = (image_ref.to(device) - 0.5) * 2
    
    image_latents = image2latents(pipeline, image)
    masked_ref_latents = image2latents(pipeline, image_ref * mask_ref)
    
    encoder_hidden_states = feature_encoder(image_ref, mask_ref)

    t = torch.randint(0, 1000, (bsz,), device=device)
    # t = sample_exponential_decay(bsz, min_val=0, max_val=1000, decay_rate=3.5e-3).to(device)

    random_noise = torch.randn_like(image_latents)
    noised_latents = pipeline.scheduler.add_noise(image_latents, random_noise, t)

    # 8 channels
    latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)

    # noise_pred = unet(
    #     latents_input, 
    #     t, 
    #     encoder_hidden_states, 
    #     return_dict=False, 
    # )[0]

    noise_pred = pipeline.unet(
        latents_input,
        t,
        encoder_hidden_states
    )

    loss = nn.functional.mse_loss(noise_pred, random_noise)
    loss_dict = {'loss': loss.item()}

    log_parameters = {
        'image': (image + 1) / 2,
        'pipeline': pipeline,
        'feature_encoder': feature_encoder,
        'masked_ref_latents': masked_ref_latents,
        'image_ref': (image_ref + 1) / 2,
        'mask_ref': mask_ref,
    }

    return loss, loss_dict, log_parameters


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torchvision.transforms.RandomCrop(256)(x)
    # x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])
    x_ref, mask = ref_transform(x.squeeze(0))

    if in_dim == 3:
        x = x.squeeze(0)

    return x, x_ref, mask


def ref_transform(ref_image: torch.Tensor):
    mask = RandomMask(ref_image.shape[-1], hole_range=[0, 1])
    mask = torch.from_numpy(mask).float().to(ref_image.device)
    combined = torch.cat([ref_image, mask], dim=0)

    pers_scale = random.uniform(0.3, 0.5)
    tps_scale = random.uniform(0.1, 0.3)
    # pytorch transform
    perspective_transformer = transforms.RandomPerspective(distortion_scale=pers_scale, p=0.8)
    combined = perspective_transformer(combined)

    geo_transform = KAug.AugmentationSequential(
        KAug.RandomThinPlateSpline(scale=tps_scale, p=0.8, align_corners=False),
        data_keys=["input"],
        same_on_batch=False,
    )
    combined = geo_transform(combined).squeeze()

    mask = combined[3:, :, :]
    transformed = combined[:3, :, :]
    transformed *= mask

    mask = mask.repeat(3, 1, 1)

    return transformed, mask


def prepare_model(
    batch_size,
    lr,
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
    unet = instantiate_from_config(config['unet_config'])
    # unet: UNet2DConditionModel = UNet2DConditionModel.from_config(
    #     UNet2DConditionModel.load_config('./pretrained/stable-diffusion-v1-4/unet/config_align.json')
    # )
    # unet, unmatched_keys = try_load(unet, pipeline.unet, return_keys=True)
    pipeline.unet = unet

    vae = instantiate_from_config(config['first_stage_config'])
    pipeline.vae = vae
    
    unet_config = OmegaConf.load('./pretrained/stable-diffusion-v1-4/unet/config_align.json')

    cross_attention_dim = unet_config['cross_attention_dim']
    feature_encoder = instantiate_from_config(config['cond_stage_config'])
    # feature_encoder = FeatureEncoder(in_channels=3, context_dim=64, cross_attn_dim=cross_attention_dim, image_dim=4096)
    # feature_encoder = PartialSAEncoder(4, cross_attention_dim)

    # frozen parameters
    pipeline.frozen()
    for param in unet.parameters():
        param.requires_grad_(True)
    # unet.unfrozen_trainable()

    params = [
        {'params': list(feature_encoder.parameters()), 'lr': lr},
        {'params': list(pipeline.unet.parameters()), 'lr': lr},
    ]
    
    for i, param in enumerate(params):
        for p in param['params']:
            assert p.requires_grad is True, f'{i} th param has param don\'t require grad'

    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights')
        unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        feature_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'feature_encoder.ckpt'), weights_only=False, map_location='cpu'))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=False, map_location='cpu'))
        try:
            start_epoch = int(ckpt_dir.split('_')[-1])
        except ValueError:
            start_epoch = 0

    # move to cuda
    pipeline.to(device)
    feature_encoder.to(device)

    feature_encoder.eval()
    unet.eval()

    # data_loader = PBRTextureDataLoader(fp='./datasets/render_result_matsynth_resized_noscale', gt_fp=None, transform=transform)
    data_loader = FolderLoader(fp='./datasets/MatSynth/color', transform=transform)
    data_loader.shuffle()

    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=4)

    # return pipeline, data_loader, optimizer, lora_register, lora_register_ref, titok_tokenizer, adapter, start_epoch
    module = {
        'unet': unet,
        'feature_encoder': feature_encoder
    }
    return module, data_loader, optimizer, start_epoch, pipeline


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

    ckpt_dir = train_config.get('ckpt_dir', None)
    device = train_config['device']
    batch_size = train_config['batch_size']
    lr = train_config['lr']
    train_config['lr'] = train_config['lr'] * len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) * batch_size
    print(f'setting lr to batch_size({batch_size})' + \
           f' * n_gpu({len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))})' + \
            f' * lr({lr}) = {train_config['lr']}')
    lr = train_config['lr']

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    
    world_size = torch.cuda.device_count()

    prepare_config = {
        'device': device, 
        'batch_size': batch_size,
        'lr':lr, 
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

    module, data_loader, optimizer, start_epoch, pipeline = prepare_model(**prepare_config)
    trainer.train(
        module,
        training_step, 
        data_loader, 
        optimizer, 
        train_config,
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

