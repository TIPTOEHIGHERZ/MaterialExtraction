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
from transformers.models.clip import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
import argparse
from omegaconf import OmegaConf
from typing import Callable
import accelerate
import random
import numpy as np
import kornia.augmentation as KAug
from torchvision import transforms
import copy

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
)
from utils.transforms import RandomThinPlateSpline
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
import utils.trainer as trainer
    

def image2latents(pipeline, image, mask=None):
    if mask is not None:
        image *= mask
    return pipeline.vae.encode(image)


def latents2image(pipeline, image):
    return (torch.clamp(pipeline.vae.decode(image), -1., 1.) + 1) / 2


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    encoder_hidden_states: torch.Tensor,
    latents_shape,
    device='cuda',
    num_inference_steps=50,
    verbose=False,
    cond_scale=3.,
    adapter: FeatureAdapter=None,
):    
    pipeline.unet.eval()

    # pipeline.scheduler = scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    bsz = encoder_hidden_states.shape[0]
    latents_start = torch.randn(bsz, 4, *latents_shape, device=device)
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(latents_start.shape[0])

        latents_input = latents_start

        noise_pred = pipeline.unet(
            latents_input,
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

    return color


@torch.no_grad()
def log_function(
    logger: SummaryWriter,
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    pipeline: Pipeline,
    image_ref: torch.Tensor,
    color: torch.Tensor,
    frozen_vision_transformer: CLIPVisionTransformer,
    trainable_vision_transformer: CLIPVisionTransformer,
    clip_processor: CLIPProcessor,
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
        clip_input = clip_processor(images=image_ref, return_tensors="pt", padding=True, do_rescale=False)['pixel_values'].to(color.device)
        encoder_hidden_states = frozen_vision_transformer(clip_input, return_dict=True).last_hidden_state[:, 1:, ...]

        eval_result = evaluate(
            pipeline,
            encoder_hidden_states,
            [s // 8 for s in color.shape[-2:]],
            color.device,
            verbose=logger is not None,
            cond_scale=None
        )

        eval_color = eval_result

        if logger is not None:
            logger.add_image(
                'evaluate' if name_space is None else name_space,
                torch.concat([
                    torch.concat([
                        ev_c, c, im_ref
                    ], dim=-1) for ev_c, c, im_ref in \
                        zip(eval_color, color, nn.functional.interpolate(image_ref, color.shape[-2:]))
                ], dim=-2),
                global_iter
            )
    return


@torch.no_grad()
def test_function(
    logger: SummaryWriter,
    global_iter: int,
    log_config: dict,
    dataloader: torch.utils.data.DataLoader,
    pipeline: Pipeline,
    frozen_vision_transformer: CLIPVisionTransformer,
    trainable_vision_transformer: CLIPVisionTransformer,
    clip_processor: CLIPProcessor,
    name_space: str=None,
    device='cuda',
    **kwargs
):
    log_period_visual = log_config['log_period_visual']
    log_period_loss = log_config['log_period_loss']
    
    unet = pipeline.unet

    # total_samples = len(test_bar)
    total_loss = 0.
    total_unet_loss = 0.
    total_learning_loss = 0.
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

            flip_angle = random.randint(0, 3) * 90.
            tps_scale = random.uniform(0.1, 0.3)
            tps = KAug.RandomThinPlateSpline(scale=tps_scale, align_corners=False, p=0.8)

            mask_ref = torch.ones_like(color)
            mask_ref = mask_ref * nn.functional.interpolate(torch.tensor(random_mask(512), device=device).unsqueeze(0), color.shape[-2:])
            augment = tps(torch.concat([color, mask_ref], dim=1))
            image_ref = augment[:, :color.shape[1], ...]
            mask_ref = augment[:, color.shape[1]:, ...][:, :1, ...]

            thresh = 0.5

            mask_ref[mask_ref > thresh] = 1.
            mask_ref[mask_ref <= thresh] = 0.

            bsz = color.shape[0]
            image_ref *= mask_ref
                
            color_latents = pipeline.image2latents(color)
            normal_latents = pipeline.image2latents(normal)
            height_latents = pipeline.image2latents(height)
            roughness_latents = pipeline.image2latents(roughness)

            masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
            
            # color: torch.Tensor = (color - 0.5) * 2
            # normal: torch.Tensor = (normal - 0.5) * 2
            # height: torch.Tensor = (height - 0.5) * 2
            # roughness: torch.Tensor = (roughness - 0.5) * 2

            # image_ref: torch.Tensor = (image_ref - 0.5) * 2
            
            clip_input = clip_processor(images=color, return_tensors="pt", padding=True, do_rescale=False)['pixel_values'].to(color.device)
            encoder_hidden_states_teacher = frozen_vision_transformer(clip_input, return_dict=True).last_hidden_state[:, 1:, ...]
            clip_input = clip_processor(images=image_ref, return_tensors="pt", padding=True, do_rescale=False)['pixel_values'].to(color.device)
            encoder_hidden_states = trainable_vision_transformer(clip_input, return_dict=True).last_hidden_state[:, 1:, ...]

            learning_loss = nn.functional.mse_loss(encoder_hidden_states, encoder_hidden_states_teacher)

            t = torch.randint(0, 1000, (bsz,), device=device)

            predict_latents = color_latents

            random_noise = torch.randn_like(predict_latents)
            noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

            # 4 channels
            latents_input = noised_latents

            noise_pred = unet(
                latents_input,
                t, 
                encoder_hidden_states, 
                return_dict=False, 
            )[0]

            unet_loss = nn.functional.mse_loss(noise_pred, random_noise)
            total_unet_loss += unet_loss.item()
            total_learning_loss += learning_loss.item()

            if i >= (num_samples - 1):
                break

        avg_unet_loss = unet_loss / (i + 1)
        avg_learning_loss = learning_loss / (i + 1)

        if logger is not None:
            logger.add_scalars(
                'test_loss', 
                {'unet_loss': avg_unet_loss, 'learning_loss': avg_learning_loss, 'loss': avg_learning_loss + avg_unet_loss}, 
                global_step=global_iter
            )

    if global_iter % log_period_visual == 0:
        clip_input = clip_processor(images=image_ref, return_tensors="pt", padding=True, do_rescale=False)['pixel_values'].to(color.device)
        encoder_hidden_states = trainable_vision_transformer(clip_input, return_dict=True).last_hidden_state[:, 1:, ...]

        eval_result = evaluate(
            pipeline,
            encoder_hidden_states,
            color_latents.shape[-2:],
            color_latents.device,
            verbose=logger is not None,
            cond_scale=None
        )

        eval_color = eval_result

        if logger is not None:
            logger.add_image(
                'test_visualize',
                torch.concat([
                    torch.concat([
                        ev_c, c, im_ref
                    ], dim=-1) for ev_c, c, im_ref in \
                        zip(eval_color, color, nn.functional.interpolate(image_ref, color.shape[-2:]))
                ], dim=-2),
                global_iter
            )

    return


def training_step(module: list[nn.Module], data, pipeline: Pipeline, clip_processor: CLIPProcessor, device='cuda'):
    unet = module['unet']
    frozen_vision_transformer = module['clip_teacher']
    trainable_vision_transformer = module['clip_student']

    image_ref, mask_ref, color, normal, height, roughness = data['image'], data['mask'], data['Color'], data['NormalGL'], data['Height'], data['Roughness']
    color = color.to(device)
    normal = normal.to(device)
    height = height.to(device)
    roughness = roughness.to(device)

    image_ref = image_ref.to(device)
    mask_ref: torch.Tensor = mask_ref.to(device)[:, :1, ...]

    flip_angle = random.randint(0, 3) * 90.
    tps_scale = random.uniform(0.1, 0.3)
    tps = KAug.RandomThinPlateSpline(scale=tps_scale, align_corners=False, p=0.8)

    mask_ref = torch.ones_like(color)
    mask_ref = mask_ref * nn.functional.interpolate(torch.tensor(random_mask(512), device=device).unsqueeze(0), color.shape[-2:])
    augment = tps(torch.concat([color, mask_ref], dim=1))
    image_ref = augment[:, :color.shape[1], ...]
    mask_ref = augment[:, color.shape[1]:, ...][:, :1, ...]

    # image_ref = torchvision.transforms.functional.rotate(image_ref, flip_angle)
    # mask_ref = torchvision.transforms.functional.rotate(mask_ref, flip_angle)

    # rotate_angels = [(random.random() * 2 - 1) * 10. for _ in range(image_ref.shape[0])]
    # color = torch.concat(
    #     [torchvision.transforms.functional.rotate(color[i: i + 1, ...], rotate_angels[i]) for i in range(image_ref.shape[0])]
    # )

    thresh = 0.5
    mask_ref[mask_ref > thresh] = 1.
    mask_ref[mask_ref <= thresh] = 0.

    bsz = color.shape[0]
    image_ref *= mask_ref
        
    color_latents = pipeline.image2latents(color)

    masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)

    clip_input = clip_processor(images=color, return_tensors="pt", padding=True, do_rescale=False)['pixel_values'].to(device)
    encoder_hidden_states_teacher = frozen_vision_transformer(clip_input, return_dict=True).last_hidden_state[:, 1:, ...]
    clip_input = clip_processor(images=image_ref, return_tensors="pt", padding=True, do_rescale=False)['pixel_values'].to(device)
    encoder_hidden_states = trainable_vision_transformer(clip_input, return_dict=True).last_hidden_state[:, 1:, ...]

    learning_loss = nn.functional.mse_loss(encoder_hidden_states, encoder_hidden_states_teacher)

    t = torch.randint(0, 1000, (bsz,), device=device)

    predict_latents = color_latents

    random_noise = torch.randn_like(predict_latents)
    noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

    # 4 channels
    latents_input = noised_latents
    latents_input_ref = masked_ref_latents

    noise_pred = unet(
        latents_input,
        t, 
        encoder_hidden_states, 
        return_dict=False, 
    )[0]

    unet_loss = nn.functional.mse_loss(noise_pred, random_noise)
    loss = unet_loss + learning_loss
    loss_dict = {'loss': loss.item(), 'unet_loss': unet_loss.item(), 'learning_loss': learning_loss.item()}

    return_examples = int(min(bsz, 8))

    log_parameters = {
        'image_ref': image_ref,
        'color': color,
        'pipeline': pipeline,
        'clip_processor': clip_processor,
        'frozen_vision_transformer': frozen_vision_transformer,
        'trainable_vision_transformer': trainable_vision_transformer
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
    clip_model: CLIPModel = CLIPModel.from_pretrained('./pretrained/clip-vit-large-patch14')
    frozen_vision_transformer: CLIPVisionTransformer = clip_model.vision_model
    clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained('./pretrained/clip-vit-large-patch14')
    trainable_vision_transformer: CLIPVisionTransformer = copy.deepcopy(frozen_vision_transformer)

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
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config('./configs/unet/align/config_main.json'))
    pipeline.unet = unet

    for param in frozen_vision_transformer.parameters():
        param.requires_grad_(False)

    vit_params = list()
    for name, param in trainable_vision_transformer.named_parameters():
        if name in ('post_layernorm.bias', 'post_layernorm.weight'):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            vit_params.append(param)

    for param in unet.parameters():
        param.requires_grad_(True)
    
    params = [
        # {'params': list(unet.parameters()), 'lr': lr * lr_gain[0]},
        {'params': list(vit_params), 'lr': lr * lr_gain[0]},
    ]
    
    for i, param in enumerate(params):
        for p in param['params']:
            assert p.requires_grad is True, f'{i} th param has param don\'t require grad'

    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights')
        unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        trainable_vision_transformer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'clip.ckpt'), weights_only=False, map_location='cpu'))
        # optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=False, map_location='cpu'))
        try:
            start_epoch = int(ckpt_dir.split('_')[-1])
        except ValueError:
            start_epoch = 0

    # move to cuda
    frozen_vision_transformer.to(device)
    trainable_vision_transformer.to(device)
    pipeline.to(device)

    # feature_encoder.eval()
    frozen_vision_transformer.eval()
    trainable_vision_transformer.eval()

    data_loader = PBRTextureDataLoader(
        # fp='./datasets/render_result_matsynth_resized_noscale', 
        fp='./datasets/render_result_matsynth_10_resized', 
        gt_fp='./datasets/MatSynth/textures_all_resized', 
        transforms={'default': transform},
        transform_group={'default': ['Color', 'NormalGL', 'Height', 'Roughness']},
        fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness')
        # no_subdir=True
    )

    eval_loader = PBRTextureDataLoader(
        # fp='./datasets/render_result_matsynth_resized_noscale', 
        fp='./datasets/render_result_matsynth_10_resized', 
        gt_fp='./datasets/MatSynth/textures_all_resized', 
        transforms={'default': transform_eval},
        transform_group={'default': ['Color', 'NormalGL', 'Height', 'Roughness']},
        fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness')
        # no_subdir=True
    )

    # test_loader = PBRTextureDataLoader(
    #     fp='./datasets/render_result_matsynth_eval_10_resized', 
    #     gt_fp='./datasets/MatSynth/textures_all_resized', 
    #     transform=transform,
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness')
    # )
    test_loader = TestPolyLoader(
        './datasets/polyhaven_edit'
    )

    # data_loader = FolderLoader(fp='./datasets/MatSynth/color', transform=transform)
    data_loader.shuffle()

    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=eval_batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=test_batch_size, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader()

    # return pipeline, data_loader, optimizer, lora_register, lora_register_ref, titok_tokenizer, adapter, start_epoch
    module = {
        'unet': unet,
        'clip_teacher': frozen_vision_transformer,
        'clip_student': trainable_vision_transformer
        # 'feature_encoder': feature_encoder
    }

    torch.cuda.empty_cache()
    return module, data_loader, eval_loader, test_loader, optimizer, start_epoch, pipeline, clip_processor


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

    module, data_loader, eval_loader, test_loader,optimizer, start_epoch, pipeline, clip_processor = prepare_model(**prepare_config)

    trainer.train(
        module,
        training_step, 
        data_loader,
        eval_loader,
        test_loader,
        optimizer, 
        train_config,
        eval_config,
        test_config,
        log_function,
        log_config,
        test_function,
        train_args=(),
        train_kwargs={
            'pipeline': pipeline,
            'clip_processor': clip_processor,
            'device': device,
        }
    )

    return


if __name__ == '__main__':
    main()

