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
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.getcwd())
# from model.FeatureExtractor.FeatureExtractor import VisionExtractor, ExtractorRegister
from model.FeatureExtractor.EmbeddingMatcher import EmbeddingMatcher
from model.oned_tokenizer.modeling.titok import TiTok
from utils.datasets import VirtualLoader
from utils.datasets import TextureDataLoader
from utils.datasets.DataLoader import (
    PexelLoader, 
    ImageDataLoader,
    TextureDataLoader,
    PBRTextureDataLoader
)

from model.pipeline import Pipeline
from model.GaussianSampler.GaussianSampler import GaussianSampler
from model.Lora import LoraRegister
from model.FeatureExtractor.FeatureAugmentor import FeatureAugmentorRegister
from model.legacy.encoders import PartialSAEncoder
from utils.transforms import RandomThinPlateSpline
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
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
from model.unet.attn_ref import AttentionReference
from model.oned_tokenizer.modeling.titok import TiTok


def transform(x: torch.Tensor):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    x = torch.nn.functional.interpolate(crop_mid(x), [512, 512])

    if in_dim == 3:
        return x.squeeze(0)

    return x


def prepare_model(
    ckpt_dir: str=None,
    finetune_image: str=None,
    device='cuda'
):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        # set_alpha_to_one=False
    )
    scheduler.set_timesteps(1000)
    pipeline: Pipeline = Pipeline.from_pretrained('./pretrained/stable-diffusion-v1-4', scheduler=scheduler)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config('./pretrained/stable-diffusion-v1-4/unet/config_main.json')
    )
    unet, unmatched_keys = try_load(unet, pipeline.unet, return_keys=True)
    pipeline.unet = unet
    
    unet_config = OmegaConf.load('./pretrained/stable-diffusion-v1-4/unet/config.json')

    cross_attention_dim = unet_config['cross_attention_dim']
    feature_encoder = PartialSAEncoder(4, cross_attention_dim)

    pipeline.unet = unet

    # frozen parameters
    pipeline.frozen()

    start_epoch = 0
    if ckpt_dir is not None:
        print('loading weights')
        unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        feature_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, 'feature_encoder.ckpt'), weights_only=False, map_location='cpu'))
        start_epoch = int(ckpt_dir.split('_')[-1])

    # move to cuda
    pipeline.to(device)
    feature_encoder.to(device)

    # only apply lora on main unet

    if finetune_image is None:
        data_loader = PBRTextureDataLoader(fp='./datasets/render_result_resized', gt_fp=None, transform=transform)
        data_loader.shuffle()
    else:
        data_loader = ImageDataLoader(finetune_image, transforms=transform)

    # return pipeline, data_loader, optimizer, lora_register, lora_register_ref, titok_tokenizer, adapter, start_epoch
    return pipeline, data_loader, feature_encoder


if __name__ == '__main__':
    device = 'cuda'
    torch.cuda.set_device(7)
    sd_path = './pretrained/stable-diffusion-v1-4'
    ckpt = './checkpoints/render_v2/base_0'
    train_timesteps = 1000
    batch_size = 8

    writer = SummaryWriter(os.path.join('./runs', 'test_render'))

    pipeline, data_loader, feature_encoder = prepare_model(
        ckpt_dir=ckpt,
        device='cuda'
    )

    pipeline.scheduler.set_timesteps(train_timesteps)

    concat = lambda x: torch.concat([x_.unsqueeze(0) for x_ in x], dim=0)
    loss_list = list()
    with torch.no_grad():
        progress_bar = tqdm.tqdm(reversed(pipeline.scheduler.timesteps - 1))
        for t in progress_bar:
            choice = np.random.choice(np.arange(len(data_loader)), (batch_size,), replace=False)
            # files = data_loader[choice]
            data = [data_loader[c] for c in choice]
            data = [concat([d[i] for d in data]).to(device) for i in range(len(data[0]))]
            
            image, ref, mask = data
            mask = mask[:, :1, ...]
            mask[mask > 0.3] = 1.
            mask[mask <= 0.3] = 0.

            latents = pipeline.image2latents(image)
            ref_latents = pipeline.image2latents(ref)
            masked_ref_latents = pipeline.image2latents(ref, mask=mask)
            mask_latents = nn.functional.interpolate(mask, latents.shape[-2:], mode='nearest')[:, :1, ...]
            
            random_noise = torch.randn_like(latents)

            noised_latents = pipeline.scheduler.add_noise(latents, random_noise, t)
            latents_input = torch.concat([noised_latents, mask_latents, masked_ref_latents], dim=1)
            latents_input_null = torch.concat(
                [noised_latents, torch.zeros_like(mask_latents), torch.zeros_like(masked_ref_latents)], dim=1
            )

            # encoder_hidden_states = pipeline.prompt2embeddings([''] * latents.shape[0])
            encoder_hidden_states = feature_encoder(ref_latents, mask_latents)
            encoder_hidden_states_null = feature_encoder(torch.zeros_like(ref_latents), torch.zeros_like(mask_latents))

            noise_pred = pipeline.unet(
                latents_input, 
                t, 
                encoder_hidden_states, 
                return_dict=False, 
            )[0]

            noise_pred_null = pipeline.unet(
                latents_input_null, 
                t, 
                encoder_hidden_states_null, 
                return_dict=False, 
            )[0]

            latents_pred = pipeline.get_x0(noised_latents, t.reshape([1,]), None, noise=noise_pred)

            loss = nn.functional.mse_loss(noise_pred, random_noise)
            loss_null = nn.functional.mse_loss(noise_pred_null, random_noise)

            loss_list.append([f'{t.item()}', loss.item()])
            progress_bar.set_postfix_str(f'loss:{loss.item(): .3f}')

            writer.add_images(
                'input',
                torch.concat([
                    pipeline.latents2image(latents_pred)[:1],
                    pipeline.latents2image(noised_latents)[:1],
                    image[:1],
                    ref[:1]
                ], dim=0),
                int(t.item())
            )

            difference = nn.functional.mse_loss(noise_pred, noise_pred_null)

            writer.add_scalars(
                'loss', 
                {
                    'cond_loss': loss.item(),
                    'null_loss': loss_null.item(),
                    'difference': difference.item()
                },
                int(t.item())
            )
            # writer.add_scalar('loss', loss.item(), int(t.item()))

    import pandas as pd

    df = pd.DataFrame(loss_list)
    df.to_csv('./loss.csv')
