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
from diffusers import DDIMScheduler, AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import Attention
import argparse
from omegaconf import OmegaConf
from typing import Callable
import accelerate
import random

sys.path.append(os.getcwd())
# from model.FeatureExtractor.FeatureExtractor import VisionExtractor, ExtractorRegister
from model.FeatureExtractor.EmbeddingMatcher import EmbeddingMatcher
from model.oned_tokenizer.modeling.titok import TiTok
from utils.datasets import VirtualLoader
from utils.datasets import TextureDataLoader
from utils.datasets.DataLoader import DTDLoader, PexelLoader, KTHaLoader, KTHbLoader, KTHLoader, ImageDataLoader
from utils.datasets.DataLoader import ManyTexureLoader, AmbientcgLoader, LabelDataLoader
from model.pipeline import Pipeline
from model.GaussianSampler.GaussianSampler import GaussianSampler
from model.Lora import LoraRegister
from model.FeatureExtractor.FeatureAugmentor import FeatureAugmentorRegister
from utils.functionals import (
    image2latents, 
    latents2image, 
    make_mask, 
    crop_mid, 
    try_load, 
    freeze_net,
    patch_shuffle,
    generate_gaussian
)
from utils.io import save_image, load_image
from model.unet.SiameseUnet import SiameseUnet, Adapter
from model.oned_tokenizer.modeling.titok import TiTok


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    source_latents: torch.Tensor,
    source_latents_ref: torch.Tensor,
    latents_mask: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    device='cuda',
    num_inference_steps=50,
    verbose=False
):    
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    latents = torch.randn_like(source_latents)
    latents_ref = torch.randn_like(source_latents)
    
    for t in timesteps:
        t = t.reshape([1,])

        latents_input = torch.concat([latents, latents_mask, source_latents], dim=1)
        latents_ref_input = torch.concat([latents_ref, torch.zeros_like(latents_ref)[:, :1, ...], source_latents_ref], dim=1)

        noise_pred, noise_pred_ref = pipeline.unet(latents_input, latents_ref_input, t, encoder_hidden_states, return_dict=False)
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latents_ref = pipeline.scheduler.step(noise_pred_ref, t, latents_ref, return_dict=False)[0]
        # latents = pipeline.remove_noise(latents, t, encoder_hidden_states, noise=noise_pred)
    
    return pipeline.latents2image(latents), pipeline.latents2image(latents_ref)


def train(
    pipeline: Pipeline,
    datasets: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lora_register: LoraRegister,
    titok_tokenizer: TiTok,
    adapter: Adapter,
    epochs: int, 
    save_period=1000, 
    start_epoch: int = 0,
    device='cpu',
    log_dir='siamese'
):
    accelerator = accelerate.Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)
    
    finetune_single = isinstance(datasets.dataset, ImageDataLoader)

    lora_register.enable()
    pipeline.unet, optimizer, datasets = accelerator.prepare(
        pipeline.unet, optimizer, datasets
    )
    lora_register = accelerator.prepare_model(lora_register)
    adapter = accelerator.prepare_model(adapter)
    
    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join('./runs', log_dir))

    epoch = 0
    total_loss = 0.

    torch.cuda.empty_cache()

    total_epochs = tqdm.trange(epochs) if accelerator.is_main_process and finetune_single else range(epochs)

    global_iter = -1
    rotate = torchvision.transforms.transforms.RandomRotation(10.)
    with accelerator.accumulate():
        for epoch in total_epochs:
            progress_bar = tqdm.tqdm(datasets) if accelerator.is_main_process and not finetune_single else datasets

            for (image, label) in progress_bar:
                global_iter += 1
                image: torch.Tensor = image.to(device)
                
                image_ref = list()
                image_input = list()
                for i in range(image.shape[0]):
                    image_input.extend([torchvision.transforms.functional.rotate(image[i: i + 1], j * 90) for j in range(4)])
                    for j in range(4):
                        select_idx = list(range(4))
                        del select_idx[j]
                        select_idx = select_idx[random.randint(0, 2)]
                        image_ref.append(torchvision.transforms.functional.rotate(image[i: i + 1], select_idx * 90))

                image_ref = torch.concat(image_ref, dim=0)
                image_ref = patch_shuffle(image_ref, 128)
                image_ref = rotate(image_ref)

                image = torch.concat(image_input, dim=0)

                bsz = image.shape[0]

                encoder_hidden_states = pipeline.prompt2embeddings([''] * bsz)
                # encoder_hidden_states = titok_tokenizer.encode(nn.functional.interpolate(image_ref, [256, 256]))[0]
                # encoder_hidden_states = adapter(encoder_hidden_states)

                # rotate_degrees = [random.random() * 10. for _ in range(bsz)]
                # image = [torchvision.transforms.functional.rotate(image[i:i + 1], rotate_degrees[i]) for i in range(image.shape[0])]
                # image = torch.concat(image, dim=0)

                loss_mask = torch.ones_like(image)[:, :1, ...]
                # loss_mask = [torchvision.transforms.functional.rotate(loss_mask[i:i + 1], rotate_degrees[i]) for i in range(loss_mask.shape[0])]
                # loss_mask = torch.concat(loss_mask, dim=0)

                optimizer.zero_grad()

                mask = make_mask(image, image.shape[-1], shuffle_rates=0.05)[:, :1, ...]
                mask = 1 - mask
                # for i in range(mask.shape[0]):
                #     if random.random() < 0.5:
                #         mask_ = mask[i: i + 1]
                #         gaussians = generate_gaussian(mask_, ratio=3. + random.random() * 12., return_tensor=True, strategy='max').to(device).unsqueeze(1)
                #         mask_ = torch.clamp(gaussians, 0., 1.)
                #         low = torch.min(torch.min(mask_, dim=-1)[0], dim=-1)[0]
                #         high = torch.max(torch.max(mask_, dim=-1)[0], dim=-1)[0]
                #         mask_ = (mask_ - low) / (high - low)
                #         mask_ = mask_.to(image.dtype)
                #         mask[i: i + 1] = mask_

                mask = rotate(mask)
                mask = 1 - mask

                latents = pipeline.image2latents(image)
                latents_masked = pipeline.image2latents(image, mask=1 - mask)
                latents_ref = pipeline.image2latents(image_ref)
                latents_mask = nn.functional.interpolate(mask, latents.shape[-2:], mode='nearest')
                loss_mask = nn.functional.interpolate(loss_mask, latents.shape[-2:], mode='nearest')

                t = torch.randint(0, 1000, (bsz,), device=image.device)

                random_noise = torch.randn_like(latents)
                noised_latents = pipeline.scheduler.add_noise(latents, random_noise, t)
                noised_latents_ref = pipeline.scheduler.add_noise(latents_ref, random_noise, t)

                latents_input = torch.concat([noised_latents, latents_mask, latents_masked], dim=1)
                latents_ref_input = torch.concat([noised_latents_ref, torch.zeros_like(noised_latents_ref)[:, :1, ...], latents_ref], dim=1)

                noise_pred = pipeline.unet(latents_input, latents_ref_input, t, encoder_hidden_states, return_dict=False)[0]

                loss = nn.functional.mse_loss(noise_pred * loss_mask, random_noise * loss_mask)

                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    total_loss += loss.item()

                    avg_loss = total_loss / (global_iter % 20 + 1)

                    if finetune_single:
                        total_epochs.set_description(f'{epoch} / {epochs}:')
                        total_epochs.set_postfix({'loss': avg_loss})
                    else:
                        progress_bar.set_description(f'{epoch} / {epochs}:')
                        progress_bar.set_postfix({'loss': avg_loss})

                    if (global_iter + 1) % 20 == 0:
                        writer.add_scalars('loss', {
                            'avg': avg_loss,
                        }, global_iter)

                        total_loss = 0.
                
                if global_iter % 100 == 0:
                    with torch.no_grad():
                        origin_scheduler = pipeline.scheduler
                        pipeline.scheduler = DDIMScheduler(
                            num_train_timesteps=1000,
                            beta_start=0.00085,
                            beta_end=0.012,
                            beta_schedule="scaled_linear",
                            clip_sample=False,
                            set_alpha_to_one=False
                        )

                        result, result_ref = evaluate(
                            pipeline,
                            latents_masked[:1],
                            latents_ref[:1],
                            latents_mask[:1],
                            encoder_hidden_states[:1],
                            verbose=accelerator.is_main_process
                        )

                        result = torch.concat([result, result_ref, image[:1], image[:1] * (1 - mask[:1])], dim=0)

                        pipeline.scheduler = origin_scheduler

                        if accelerator.is_main_process:
                            writer.add_images(
                                'eval_result',
                                result,
                                global_iter
                            )                    
        
            if accelerator.is_main_process and (epoch + 1) % save_period == 0:
                save_dir = os.path.join('./checkpoints', log_dir, f'train_{(epoch + 1) % 10}')
                # save_dir = f'./checkpoints/siamese/train_{(epoch + 1) % 10}'
                os.makedirs(save_dir, exist_ok=True)

                if isinstance(lora_register, DDP):
                    torch.save(lora_register.module.state_dict(), os.path.join(save_dir, f'lora_register.ckpt'))
                    torch.save(adapter.module.state_dict(), os.path.join(save_dir, f'adapter.ckpt'))
                else:
                    torch.save(lora_register.state_dict(), os.path.join(save_dir, f'lora_register.ckpt'))
                    torch.save(adapter.state_dict(), os.path.join(save_dir, f'adapter.ckpt'))

                torch.save(optimizer.state_dict(), os.path.join(save_dir, f'optimizer.ckpt'))
        
    return


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])

    if in_dim == 3:
        return x.squeeze(0)

    return x


def prepare_model(
    batch_size,
    lr,
    ckpt_dir: str=None,
    finetune_image: str=None,
    device='cuda'
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        # set_alpha_to_one=False
    )
    scheduler.set_timesteps(1000)
    pipeline: Pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-2-inpainting', scheduler=scheduler)
    unet: SiameseUnet = SiameseUnet.from_pretrained('./pretrained/stable-diffusion-2-inpainting/unet')
    unet_config = OmegaConf.load('./pretrained/stable-diffusion-2-inpainting/unet/config.json')
    titok_tokenizer = TiTok.from_pretrained('./pretrained/tokenizer_titok_s128_imagenet')
    titok_config = titok_tokenizer.config

    cross_attention_dim = unet_config['cross_attention_dim']
    titok_dim = titok_config.model.vq_model.token_size
    hidden_dim = cross_attention_dim
    adapter = Adapter(titok_dim, hidden_dim, cross_attention_dim)

    pipeline.unet = unet
    pipeline.frozen()

    lora_register = LoraRegister(
        unet.unet_main, 
        ['attn1', 'attn2'], 
        lora_ratio=1.0,
        attn_types=['vanilla', 'vanilla'],
        register_range=[None, None]
    )
    # only apply lora on main unet

    if finetune_image is None:
        data_loader = PexelLoader('./datasets/finetune', batch_size=batch_size, relative_dir='images', transforms=transform)
        data_loader.shuffle()
    else:
        data_loader = ImageDataLoader(finetune_image, transforms=transform)

    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=1)
    
    params = [
        {'params': lora_register.parameters()},
        # {'params': adapter.parameters()}
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        lora_register.load_state_dict(torch.load(os.path.join(ckpt_dir, 'lora_register.ckpt'), weights_only=True))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=True))
        start_epoch = int(ckpt_dir.split('_')[-1])

    lora_register.train()
    lora_register.to(device)

    adapter.train()
    adapter.to(device)

    titok_tokenizer.eval()
    freeze_net(titok_tokenizer)
    titok_tokenizer.to(device)

    pipeline.to(device)

    return pipeline, data_loader, optimizer, lora_register, titok_tokenizer, adapter, start_epoch


def train_distribute(rank: int,
                     config_prepare: dict):
    world_size = torch.cuda.device_count()

    device = config_prepare.get('device')
    epochs = config_prepare.pop('epochs')
    save_period = config_prepare.pop('save_period')
    log_dir = config_prepare.pop('log_dir', 'siamese')

    pipeline, data_loader, optimizer, lora_register, titok_tokenizer, adapter, start_epoch = prepare_model(**config_prepare)
    
    train(pipeline, data_loader, optimizer, lora_register, titok_tokenizer, adapter, epochs, 
          device=device, save_period=save_period, start_epoch=start_epoch, log_dir=log_dir)
    
    if world_size > 1:
        destroy_process_group()

    return


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
    if config_file is None:
        epochs = args.epochs
        device = args.device
        batch_size = args.batch_size
        save_period = args.save_period
        lr = args.lr
        resume_training = args.resume_training
    else:
        config = OmegaConf.load(config_file)
        epochs = config['epochs']
        device = config['device']
        batch_size = config['batch_size']
        save_period = config['save_period']
        lr = config['lr']
        ckpt_dir = config.get('ckpt_dir', None)
        finetune_image = config.get('finetune_image', None)
        log_dir = config.get('log_dir', 'siamese')

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    
    world_size = torch.cuda.device_count()

    prepare_config = {
        'epochs': epochs, 
        'device': device, 
        'batch_size': batch_size,
        'save_period': save_period, 
        'lr':lr, 
        'ckpt_dir': ckpt_dir,
        'finetune_image': finetune_image,
        'log_dir': log_dir
    }

    rank = 0
    if device == 'cuda' and world_size > 1:
        try:
            rank = int(os.environ['RANK'])
        except KeyError:
            print('not in multi process')
        torch.cuda.set_device(rank)
        print(f'rank {rank} is initialized')

    train_distribute(rank, prepare_config)

    return


if __name__ == '__main__':
    main()

