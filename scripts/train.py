import torch
import torchvision
import torch.nn as nn
import sys
import os
import torch.optim.optimizer
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional
import tqdm
from diffusers import DDIMScheduler
from diffusers.models.attention_processor import Attention
import argparse
import omegaconf
from typing import Callable
import logging
import random
import accelerate

sys.path.append(os.getcwd())

from utils.datasets import TextureDataLoader, DTDLoader, KTHaLoader, KTHbLoader, KTHLoader, PexelLoader
from utils.datasets import ManyTexureLoader, AmbientcgLoader

from model.FeatureExtractor.EmbeddingMatcher import EmbeddingMatcher
from model.pipeline import Pipeline, InpaintingPipeline
from model.GaussianSampler.GaussianSampler import GaussianSampler
from model.Lora import LoraRegister
from model.FeatureExtractor.FeatureAugmentor import FeatureAugmentorRegister
from model.AttentionReplacer import replace_attn_processor
from model.EmbeddingManager import VisionTextEmbedding
from model.FractalAttention import FractalAttention
from model.unet import UNetModel
from utils.io import load_image
from utils.functionals import make_mask

from scripts.eval import evaluate


logging.basicConfig(level=logging.INFO)


class EmbeddingWrapper(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

        return
    
    def forward(self, *args, **kwargs):
        result = [self.m(arg, **kwargs) for arg in args]

        return result


def train(pipeline: InpaintingPipeline,
          data_loader,
          embedding_matcher: EmbeddingMatcher,
          lora_register: LoraRegister,
          embedder: VisionTextEmbedding,
          time_steps: int,
          optimizer: torch.optim.Optimizer, 
          epochs: int, 
          device='cpu',
          rank=None,
          start_epoch=0,
          save_period=None,
          percision=None):
    accelerator = accelerate.Accelerator(mixed_precision=percision)
    datasets = TextureDataLoader([DTDLoader('./datasets/texture_modified')], shuffle=False)

    # embedding_wrapper = EmbeddingWrapper(embedding_matcher)

    pipeline.unet, data_loader, embedding_matcher, optimizer = accelerator.prepare(
        pipeline.unet,
        data_loader,
        embedding_matcher,
        optimizer
    )
    if torch.cuda.device_count() > 1:
        lora_register.ddp()

    pipeline.scheduler.set_timesteps(time_steps, device=device)
    pipeline.scheduler.timesteps = pipeline.scheduler.timesteps - 1
    pipeline.scheduler.alphas = pipeline.scheduler.alphas.to(device)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)
    # define loss function
    loss_fn = nn.MSELoss()
    gaussian_sampler = GaussianSampler([512, 512], [8, 8], 1, device=device)

    text_embeddings = pipeline.prepare_prompt_embeddings('', 1, use_conditional_guidance=False).to(device)

    # place_holders = ['*', '(', ')', '-']
    place_holders = ['*']
    # source_prompt = ['a $ style texture', 'texture of $ style']
    # source_prompt = '*'

    embedder.add_tokens(place_holders)

    lora_register.enable()

    if rank == 0:
        total_loss = 0.
        e_loss = 0.
        g_loss = 0.
        writer = SummaryWriter('runs/fratal_diffusion_mae')

    torch.cuda.empty_cache()
    weight = 0.7

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(180)
    ])

    for epoch in range(epochs):
        if rank == 0 and sys.stdout.isatty():
            progress_bar = tqdm.tqdm(data_loader, disable=(rank != 0))
        else:
            progress_bar = data_loader

        for i, data in enumerate(progress_bar):
            global_iter = i + (start_epoch + epoch) * len(progress_bar)

            data = data.to(device)
            data = transform(data)

            optimizer.zero_grad()
            batch_size = data.shape[0]

            # mask: list[torch.Tensor] = [gaussian_sampler.generate_gaussian(1, [10., 10.],
            #                                                         gt_radius=torch.randint(1, 3, [1,], device=device),
            #                                                         device=device).unsqueeze(0) for _ in range(batch_size)]
            mask = make_mask(data, data.shape[-1], shuffle_rates=0.1)

            latents_mask = torch.nn.functional.interpolate(mask, [64, 64])

            t = torch.randint(0, time_steps, [batch_size,], device=device, requires_grad=False)
            
            source_latents = pipeline.image2latents(data)
            # latents = pipeline.image2latents(
            #     torchvision.transforms.functional.rotate(data, 90. * random.randint(1, 3))
            # )
            masked_latents = pipeline.image2latents(data, (1 - mask))
            random_noise = torch.randn_like(source_latents)

            # masked_latents = pipeline.image2latents(data) * (1 - latents_mask)

            embedding_input = torch.concat([masked_latents, latents_mask], dim=1)
            embedding_input_gt = torch.concat([source_latents, 
                                               torch.ones(source_latents.shape[0], 1, *source_latents.shape[-2:], device=device)],
                                               dim=1)

            image_embeds, embedding_loss = embedding_matcher(embedding_input, embedding_input_gt, latents_mask)

            prompt = place_holders * batch_size
            encoder_hidden_states = embedder(prompt, image_embeds)
            
            # noise_pred = predict_noise(pipeline, latents, random_noise, t, encoder_hidden_states, fractal_mask=torch.zeros_like(latents_mask))
            # masked_noise_pred = predict_noise(pipeline, masked_latents, random_noise, t, encoder_hidden_states, fractal_mask=latents_mask)
            noise_pred = pipeline.predict_noise(source_latents, latents_mask, masked_latents, random_noise, t, encoder_hidden_states, fractal_mask=None)
            
            # 单纯噪声估计的loss
            global_loss: torch.Tensor = loss_fn(noise_pred * latents_mask, random_noise * latents_mask)
            # 被mask掉部分重新填入的loss
            # filling_loss: torch.Tensor = loss_fn(noise_pred * latents_mask, random_noise * latents_mask)

            # noised_mask_latents = pipeline.scheduler.add_noise(masked_latents, random_noise, t)
            # masked_latents = pipeline.remove_noise(noised_mask_latents, t, None, noise=noise_pred)
            # filling_loss: torch.Tensor = loss_fn(masked_latents, source_latents)

            # loss = filling_loss * weight + global_loss * (1 - weight)
            loss = global_loss + embedding_loss
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()

            if global_iter % 100 == 0:
                image, mask, gt = evaluate(
                    pipeline,
                    gaussian_sampler,
                    datasets,
                    embedding_matcher,
                    embedder,
                    device,
                    verbose=accelerator.is_main_process,
                    leave=False
                )
                image = pipeline.latents2image(image)

                if accelerator.is_main_process:
                    writer.add_images(
                        'result',
                        torch.concat([image, torch.concat([mask] * 3, dim=1), gt], dim=0),
                        global_iter
                    )
            
            if rank == 0:
                total_loss += loss.item()
                e_loss += embedding_loss.item()
                g_loss += global_loss.item()
                avg_loss = total_loss / (global_iter % 20 + 1)

                if (global_iter + 1) % 20 == 0:
                    writer.add_scalars(
                        'loss',
                        {
                            'avg_loss': avg_loss,
                            'embed_loss': e_loss / (global_iter % 20 + 1),
                            'global_loss': g_loss / (global_iter % 20 + 1)
                        },
                        global_iter
                    )
                    
                    total_loss = 0.
                    e_loss = 0.
                    g_loss = 0.

                if sys.stdout.isatty():
                    progress_bar.set_description(f'{epoch} / {epochs}')
                    progress_bar.set_postfix({'loss': avg_loss})
                elif i % 10 == 0:
                    logging.info(f'{epoch} / {epochs} loss: {loss.item(): .3f}')

        if (rank == 0 or rank is None) and (epoch + 1) % save_period == 0:
            save_dir = f'./checkpoints/sd_mae/train_{epoch + start_epoch + 1}'
            os.makedirs(save_dir, exist_ok=True)
            lora_register.save(os.path.join(save_dir, 'lora.ckpt'))
            if isinstance(embedding_matcher, DDP):
                torch.save(embedding_matcher.module.state_dict(), os.path.join(save_dir, 'embedding_matcher.ckpt'))
            else:
                torch.save(embedding_matcher.state_dict(), os.path.join(save_dir, 'embedding_matcher.ckpt'))

            # if isinstance(pipeline.unet, DDP):
            #     pipeline.unet.module.save(os.path.join(save_dir, 'fractals.ckpt'))
            # else:
            #     pipeline.unet.module.save(os.path.join(save_dir, 'fractals.ckpt'))

    return


def prepare_model(batch_size,
                  lr,
                  ckpt_dir: str=None,
                  device='cpu'):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False)
    pipeline: InpaintingPipeline = InpaintingPipeline.from_pretrained('./pretrained/stable-diffusion-2-inpainting/', scheduler=scheduler)
    # replace_attn_processor(pipeline.unet)
    # unet = UNetModel.from_pretrained('./pretrained/stable-diffusion-2-inpainting/unet')
    pipeline.frozen()
    # unet.train_()
    # pipeline.unet = unet
    
    lora_register = LoraRegister(pipeline.unet, name_list=['attn2', 'attn1'])
    lora_register.train()
    
    embedding_matcher = EmbeddingMatcher(5, 2, 1024, './pretrained/clip-vit-large-patch14')
    embedding_matcher.train_()

    embedder = VisionTextEmbedding('pretrained/stable-diffusion-2-inpainting/tokenizer', 'pretrained/stable-diffusion-2-inpainting/text_encoder')
    embedder.train_()
    
    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights...')
        start_epoch = int(ckpt_dir.split('_')[-1])
        embedding_matcher.load_state_dict(torch.load(os.path.join(ckpt_dir, 'embedding_matcher.ckpt'), weights_only=True))
        lora_register.load(os.path.join(ckpt_dir, 'lora.ckpt'))
        # unet.load(os.path.join(ckpt_dir, 'fractals.ckpt'))
        print(f'starting with epoch: {start_epoch}')

    embedding_matcher.to(device)
    embedder.to(device)
    lora_register.to(device)
    pipeline.to(device)

    # data_types = [DTDLoader, PexelLoader, KTHaLoader, KTHbLoader, KTHLoader, ManyTexureLoader, AmbientcgLoader]
    data_types = [DTDLoader]

    data_loader = list()
    for t in data_types:
        data_loader.append(t('./datasets/texture_modified')) 
    data_loader = TextureDataLoader(data_loader, shuffle=True)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size)

    params = [
        # {'params': embedding_matcher.raw_parameters()},
        # {'params': embedding_matcher.fine_parameters(), 'lr': lr * 0.1},
        {'params': embedding_matcher.parameters()},
        {'params': lora_register.parameters()},
        # {'params': unet.fractal_parameters()}
    ]

    optimizer = torch.optim.AdamW(params, lr=lr)

    return pipeline, data_loader, embedding_matcher, optimizer, lora_register, embedder, start_epoch


def ddp_setup(rank: int, world_size: int, port='40172'):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        port: master port
    """
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return


def train_distribute(rank: int,
                     prepare_model_: Callable,
                     config_prepare: dict):
    world_size = torch.cuda.device_count()

    device = config_prepare.get('device')
    epochs = config_prepare.pop('epochs')
    time_steps = config_prepare.pop('time_steps')
    save_period = config_prepare.pop('save_period', None)
    pipeline, data_loader, embedding_matcher, optimizer, lora_register, embedder, start_epoch = prepare_model_(**config_prepare)

    # if world_size > 1:
    #     pipeline.unet = DDP(pipeline.unet)
    #     embedding_matcher = DDP(embedding_matcher, find_unused_parameters=True)
    #     lora_register.ddp()

    # if world_size > 1:
    #     data_loader = torch.utils.data.DataLoader(data_loader.dataset, data_loader.batch_size,
    #                                             shuffle=False,
    #                                             sampler=DistributedSampler(data_loader.dataset, 
    #                                                                         rank=rank, num_replicas=world_size),
    #                                             num_workers=8)
    # else:
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, data_loader.batch_size,
                                                shuffle=False,
                                                num_workers=8)
    
    # accelerator = accelerate.Accelerator()
    # pipeline.unet, data_loader, embedding_matcher, optimizer = accelerator.prepare(pipeline.unet, data_loader, embedding_matcher, optimizer)

    train(
        pipeline=pipeline,
        data_loader=data_loader,
        lora_register=lora_register,
        embedding_matcher=embedding_matcher,
        embedder=embedder,
        time_steps=time_steps,
        optimizer=optimizer,
        epochs=epochs,
        rank=rank,
        start_epoch=start_epoch,
        save_period=save_period,
        device=device
    )
    
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


    args = parser.parse_args()
    
    config_file = args.config_file
    if config_file is None:
        epochs = args.epochs
        device = args.device
        device_ids = args.device_ids
        batch_size = args.batch_size
        save_period = args.save_period
        lr = args.lr
    else:
        config = omegaconf.OmegaConf.load(config_file)
        epochs = config['epochs']
        device = config['device']
        device_ids = config['device_ids']
        device_ids = [str(i) for i in device_ids]
        batch_size = config['batch_size']
        save_period = config['save_period']
        lr = config['lr']
        ckpt_dir = config.get('ckpt_dir', None)
        ckpt_idx = config.get('ckpt_idx', None)

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

    total_steps = 1000
    
    world_size = torch.cuda.device_count()

    prepare_config = {
        'epochs': epochs, 
        'device': device,
        'save_period': save_period,
        'batch_size': batch_size,
        'lr': lr,
        'time_steps': total_steps,
        'ckpt_dir': ckpt_dir,
    }

    rank = 0
    if device == 'cuda' and world_size > 1:
        try:
            rank = int(os.environ['RANK'])
        except KeyError:
            print('not in multi process')
        torch.cuda.set_device(rank)
    #     ddp_setup(rank, world_size)
        print(f'rank {rank} is initialized')

    train_distribute(rank, prepare_model, prepare_config)

    return


if __name__ == '__main__':
    main()
