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
from diffusers import DDIMScheduler, AutoencoderKL
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
    embedding_matcher: EmbeddingMatcher,
    datasets,
    vae: AutoencoderKL,
    device='cuda'
):
    image = datasets.dataset[random.randint(0, len(datasets) - 1)][0].unsqueeze(0)
    # image = datasets.dataset[random.randint(0, len(datasets) - 1)].unsqueeze(0)
    image = image.to(device)

    # mask = make_mask(image, image.shape[-1], shuffle_rates=0.1)
    # masked_latents = image2latents(vae, image, 1 - mask)
    # latents_mask = nn.functional.interpolate(mask, masked_latents.shape[-2:])
    # embedding_input = torch.concat([masked_latents, latents_mask], dim=1)

    mask = torch.zeros(image.shape[0], 1, *image.shape[-2:], device=image.device)
    # latents = image2latents(vae, image, 1 - mask)
    # latents_mask = nn.functional.interpolate(mask, latents.shape[-2:])
    # embedding_input = torch.concat([latents, latents_mask], dim=1)
    image_input = torch.concat([image, mask], dim=1)

    if isinstance(embedding_matcher, DDP):
        embedding_matcher = embedding_matcher.module
    
    embedding_matcher.eval()
    # embeds, _ = embedding_matcher.encode(image_input, mask)
    # result = embedding_matcher.decode(embeds)
    result, _ = embedding_matcher(image_input, mask)
    embedding_matcher.train_encoder()

    # return latents2image(vae, result), mask, image
    return result, mask, image


def train(
    embedding_matcher: EmbeddingMatcher,
    datasets: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    vae: AutoencoderKL,
    epochs: int, 
    save_period=1, 
    start_epoch: int = 0,
    perceptual_loss_fn: PerceptualLoss=None,
    device='cpu',
):
    accelerator = accelerate.Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)

    embedding_matcher, optimizer, datasets = accelerator.prepare(
        embedding_matcher, optimizer, datasets
    )
    
    if accelerator.is_main_process:
        writer = SummaryWriter('./runs/mae')

    global_iter = 0

    total_loss = 0.
    quant_loss = 0.
    recon_loss = 0.
    percept_loss = 0.

    torch.cuda.empty_cache()
    for epoch in range(epochs):
        datasets.dataset.shuffle()
        progress_bar = tqdm.tqdm(datasets) if accelerator.is_main_process else datasets

        with accelerator.accumulate():
            for i, data in enumerate(progress_bar):
                global_iter = i + epoch * len(progress_bar)

                image = data[0]
                bsz = image.shape[0]
                # image = data
                image = image.to(device)
                
                optimizer.zero_grad()

                mask = make_mask(image, image.shape[-1], shuffle_rates=0.1)
                image_masked_input = torch.concat([image * (1 - mask), mask], dim=1)
                image_input = image

                sample_result, loss_dict = embedding_matcher(image_masked_input, mask, image_input)

                quantizer_loss = loss_dict['quantizer_loss']
                reconstruct_loss = loss_dict['reconstruct_loss']

                perceptual_loss = torch.tensor(0., device=image.device)
                if perceptual_loss_fn is not None:
                    perceptual_loss = perceptual_loss_fn(sample_result, image)

                loss = loss_dict['total_loss'] + perceptual_loss
                
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    total_loss += loss.item()
                    recon_loss += reconstruct_loss.item()
                    percept_loss += perceptual_loss.item()
                    quant_loss += quantizer_loss.item()

                    avg_loss = total_loss / (global_iter % 20 + 1)
                    progress_bar.set_description(f'{epoch} / {epochs} {global_iter}: ')
                    progress_bar.set_postfix({'loss': avg_loss})

                    if (global_iter + 1) % 20 == 0:
                        writer.add_scalars('loss', {
                            'avg': avg_loss,
                            'quant': quant_loss / (global_iter % 20 + 1),
                            'recon': recon_loss / (global_iter % 20 + 1),
                            'percept': percept_loss / (global_iter % 20 + 1)
                        }, global_iter)

                        total_loss = 0.
                        quant_loss = 0.
                        recon_loss = 0.
                        percept_loss = 0.
                
                if global_iter % 200 == 0:
                    with torch.no_grad():
                        matcher = embedding_matcher.module if isinstance(embedding_matcher, DDP) else embedding_matcher

                        sample_result = sample_result[:1]
                        quantized_states = torch.einsum(
                            'nchw,cd->ndhw', sample_result.softmax(1),
                            matcher.backbone.pixel_quantize.embedding.weight)
                        sample_result = matcher.backbone.pixel_decoder(quantized_states)
                        image = image[:1]

                        result = torch.concat([sample_result[:1], image[:1], image[:1] *(1 - mask[:1])], dim=0)
                        if accelerator.is_main_process:
                            writer.add_images(
                                'eval_result',
                                result,
                                global_iter
                            )              
        
        if accelerator.is_main_process and (epoch + 1) % save_period == 0:
            save_dir = f'./checkpoints/mae/train_{(epoch + 1) % 500}'
            os.makedirs(save_dir, exist_ok=True)

            if isinstance(embedding_matcher, DDP):
                module = embedding_matcher.module
            else:
                module = embedding_matcher

            torch.save(optimizer.state_dict(), os.path.join(save_dir, f'optimizer.ckpt'))
            torch.save(module.state_dict(), os.path.join(save_dir, f'embedding_matcher.ckpt'))
        
    return


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torch.nn.functional.interpolate(crop_mid(x), [256, 256])

    if in_dim == 3:
        return x.squeeze(0)

    return x


def prepare_model(
    batch_size,
    lr,
    ckpt_dir:str=None,
    device='cuda'
):
    backbone = TiTok(OmegaConf.load('./pretrained/tokenizer_titok_s128_imagenet/my_config.json'))
    embedding_matcher = EmbeddingMatcher(backbone)

    vae = AutoencoderKL.from_pretrained('./pretrained/stable-diffusion-v1-4/vae')
    # perceptual_loss = PerceptualLoss(["convnext_s"])
    perceptual_loss = None

    data_loader = LabelDataLoader('./imagenet', transforms=transform, shuffle=True)
    # data_loader = TextureDataLoader([DTDLoader('./datasets/texture_modified')])
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, shuffle=True)
    
    params = [
        {'params': embedding_matcher.raw_parameters()},
        # {'params': embedding_matcher.parameters()},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)

    start_epoch = 0
    if ckpt_dir is not None:
        with torch.no_grad():
            print(f'loading weights {ckpt_dir}')

            embedding_matcher: EmbeddingMatcher = try_load(
                embedding_matcher, 
                torch.load(os.path.join(ckpt_dir, 'embedding_matcher.ckpt'), weights_only=True), 
                ignore_keys=('pixel_encoder', 'pixel_decoder', 'pixel_quantize')
            )

            if os.path.exists(os.path.join(ckpt_dir, 'optimizer.ckpt')):
                optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.ckpt'), weights_only=True))
                
            start_epoch = int(ckpt_dir.split('_')[-1])

    embedding_matcher.train_encoder()
    embedding_matcher.to(device)

    vae.to(device)

    return embedding_matcher, data_loader, optimizer, vae, perceptual_loss, start_epoch


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
    save_period = config_prepare.pop('save_period')
    embedding_matcher, data_loader, optimizer, vae, perceptual_loss, start_epoch = prepare_model_(**config_prepare)
    
    if world_size > 1:
        data_loader = torch.utils.data.DataLoader(data_loader.dataset, data_loader.batch_size,
                                                shuffle=False,
                                                sampler=DistributedSampler(data_loader.dataset, 
                                                                            rank=rank, num_replicas=world_size),
                                                num_workers=8)
    else:
        data_loader = torch.utils.data.DataLoader(data_loader.dataset, data_loader.batch_size,
                                                shuffle=False,
                                                num_workers=8)
    train(embedding_matcher, data_loader, optimizer, vae, epochs, 
          device=device, save_period=save_period, start_epoch=start_epoch, perceptual_loss_fn=perceptual_loss)
    
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
