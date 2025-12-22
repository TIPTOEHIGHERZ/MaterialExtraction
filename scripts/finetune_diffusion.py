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
from utils.datasets import VirtualLoader
from utils.datasets import TextureDataLoader
from utils.datasets.DataLoader import DTDLoader, PexelLoader, KTHaLoader, KTHbLoader, KTHLoader
from utils.datasets.DataLoader import ManyTexureLoader, AmbientcgLoader, LabelDataLoader
from model.pipeline import Pipeline
from model.GaussianSampler.GaussianSampler import GaussianSampler
from model.Lora import LoraRegister
from model.FeatureExtractor.FeatureAugmentor import FeatureAugmentorRegister
from utils.functionals import image2latents, latents2image, make_mask, crop_mid, try_load
from utils.io import save_image, load_image
from model.oned_tokenizer.modeling.titok import TiTok, PretrainedTokenizer
from model.oned_tokenizer.modeling.modules.losses import PerceptualLoss


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    prompt: str,
    device='cuda',
    num_inference_steps=50,
    verbose=False
):
    latents = torch.randn([1, 4, 64, 64], device=device)
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps
    encoder_hidden_states = pipeline.prepare_prompt_embeddings(prompt)

    for t in timesteps:
        t = t.reshape([1,])
        noise_pred = pipeline.unet(latents, t, encoder_hidden_states,return_dict=False)[0]
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # latents = pipeline.remove_noise(latents, t, encoder_hidden_states, noise=noise_pred)
    
    return pipeline.latents2image(latents)


def train(
    pipeline: Pipeline,
    datasets: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lora_register: LoraRegister,
    epochs: int, 
    save_period=1, 
    start_epoch: int = 0,
    device='cpu',
):
    accelerator = accelerate.Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)

    lora_register.enable()
    pipeline.unet, optimizer, datasets = accelerator.prepare(
        pipeline.unet, optimizer, datasets
    )
    lora_register = accelerator.prepare_model(lora_register)
    
    if accelerator.is_main_process:
        writer = SummaryWriter('./runs/lora')


    global_iter = 0
    total_loss = 0.

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        datasets.dataset.shuffle()
        progress_bar = tqdm.tqdm(datasets) if accelerator.is_main_process else datasets

        with accelerator.accumulate():
            for i, (image, label) in enumerate(progress_bar):
                global_iter = i + epoch * len(progress_bar)

                bsz = image.shape[0]
                image: torch.Tensor = image.to(device)
                label = [l.item() for l in label]
                
                optimizer.zero_grad()

                latents = pipeline.image2latents(image)
                t = torch.randint(0, 1000, (bsz,), device=image.device)

                prompt = [f'{datasets.dataset.label2name[l]} style texture.' for l in label]
                encoder_hidden_states = pipeline.prompt2embeddings(prompt)
                random_noise = torch.randn_like(latents)
                noised_latents = pipeline.scheduler.add_noise(latents, random_noise, t)
                noise_pred = pipeline.unet(noised_latents, t, encoder_hidden_states, return_dict=False)[0]

                loss = nn.functional.mse_loss(noise_pred, random_noise)

                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    total_loss += loss.item()

                    avg_loss = total_loss / (global_iter % 20 + 1)
                    progress_bar.set_description(f'{epoch} / {epochs} {global_iter}: ')
                    progress_bar.set_postfix({'loss': avg_loss})

                    if (global_iter + 1) % 20 == 0:
                        writer.add_scalars('loss', {
                            'avg': avg_loss,
                        }, global_iter)

                        total_loss = 0.
                
                if global_iter % 200 == 0:
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

                        result = evaluate(
                            pipeline,
                            prompt[0],
                            verbose=accelerator.is_main_process
                        )

                        result = torch.concat([result, image[:1]], dim=0)

                        pipeline.scheduler = origin_scheduler

                        if accelerator.is_main_process:
                            writer.add_images(
                                'eval_result',
                                result,
                                global_iter
                            )                    
        
        if accelerator.is_main_process and (epoch + 1) % save_period == 0:
            save_dir = f'./checkpoints/lora/train_{(epoch + 1) % 50}'
            os.makedirs(save_dir, exist_ok=True)

            if isinstance(lora_register, DDP):
                torch.save(lora_register.module.state_dict(), os.path.join(save_dir, f'lora_register.ckpt'))
            else:
                torch.save(lora_register.state_dict(), os.path.join(save_dir, f'lora_register.ckpt'))

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
    ckpt_dir:str=None,
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
    pipeline: Pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4', scheduler=scheduler)
    pipeline.frozen()
    lora_register = LoraRegister(pipeline.unet, ['attn1', 'attn2'], lora_ratio=1.0)

    data_loader = LabelDataLoader('./datasets/finetune_dataset/images', transforms=transform, shuffle=True)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=False, num_workers=8)
    
    params = [
        {'params': lora_register.parameters()},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        lora_register.load_state_dict(torch.load(os.path.join(ckpt_dir, 'lora_register.ckpt'), weights_only=True))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=True))
        start_epoch = int(ckpt_dir.split('_')[-1])

    lora_register.train()
    lora_register.to(device)

    pipeline.to(device)

    return pipeline, data_loader, optimizer, lora_register, start_epoch


def train_distribute(rank: int,
                     prepare_model_: Callable,
                     config_prepare: dict):
    world_size = torch.cuda.device_count()

    device = config_prepare.get('device')
    epochs = config_prepare.pop('epochs')
    save_period = config_prepare.pop('save_period')
    embedding_matcher, data_loader, optimizer, lora_register, start_epoch = prepare_model_(**config_prepare)
    
    train(embedding_matcher, data_loader, optimizer, lora_register, epochs, 
          device=device, save_period=save_period, start_epoch=start_epoch)
    
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
        ckpt_idx = config.get('ckpt_idx', None)

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
        'ckpt_dir': ckpt_dir
    }

    rank = 0
    if device == 'cuda' and world_size > 1:
        try:
            rank = int(os.environ['RANK'])
        except KeyError:
            print('not in multi process')
        torch.cuda.set_device(rank)
        print(f'rank {rank} is initialized')

    train_distribute(rank, prepare_model, prepare_config)

    return


if __name__ == '__main__':
    main()

