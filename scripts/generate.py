import torch
import os
import sys
import tqdm
from diffusers import DDIMScheduler
import torch.nn as nn
import random

sys.path.append(os.getcwd())

from model.unet import UNetModel
from model.pipeline import Pipeline
from model.Lora import LoraRegister
from model.FeatureExtractor.EmbeddingMatcher import EmbeddingMatcher
from model.EmbeddingManager import VisionTextEmbedding
from utils.datasets import TextureDataLoader, DTDLoader, KTHaLoader, KTHbLoader, KTHLoader, PexelLoader
from model.GaussianSampler.GaussianSampler import GaussianSampler
from utils.io import load_image, save_image, load_from_ddp
from utils.functionals import make_mask
from model.pipeline import InpaintingPipeline


if __name__ == '__main__':
    device = 'cuda'
    device_id = 0

    ckpt_dir = './checkpoints/sd_mae/train_55'

    torch.cuda.set_device(device_id)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                            clip_sample=False, set_alpha_to_one=False)
    pipeline: InpaintingPipeline = InpaintingPipeline.from_pretrained('./pretrained/stable-diffusion-2-inpainting', scheduler=scheduler)
    # unet: UNetModel = UNetModel.from_pretrained('./pretrained/stable-diffusion-2-inpainting/unet')
    # unet.load(os.path.join(ckpt_dir, 'fractals.ckpt'))
    # pipeline.unet = unet
    pipeline.to(device)

    lora_register = LoraRegister(pipeline.unet, ['attn2', 'attn1'])
    lora_register.load(os.path.join(ckpt_dir, 'lora.ckpt'))
    lora_register.to(device)

    # 特征提取
    embedding_matcher = EmbeddingMatcher(5, 2, 1024, './pretrained/clip-vit-large-patch14')
    embedding_matcher.load_state_dict(torch.load(os.path.join(ckpt_dir, 'embedding_matcher.ckpt')))
    # embedding_matcher.load_state_dict(load_from_ddp(os.path.join(ckpt_dir, 'embedding_matcher.ckpt')))
    embedding_matcher.to(device)

    embedder = VisionTextEmbedding('pretrained/stable-diffusion-2-inpainting/tokenizer', 'pretrained/stable-diffusion-2-inpainting/text_encoder')
    embedder.to(device)

    # 数据
    image = load_image('./test_files/myplan/6.jpg', to_batch=True, device=device)
    # image = load_image('./test_files/myplan/6_modified.png', to_batch=True, device=device)
    mask = load_image('./test_files/myplan/6_mask.png', to_batch=True, device=device)

    h, w  = image.shape[-2:]
    shift = 200
    import copy
    source = copy.deepcopy(image[:, :, :h // 2, :w - shift])
    image[:, :, :h // 2, shift:] = source
    image[:, :, :h // 2, :shift] = 0.

    if mask.shape[1] != 1:
        mask = mask[:, :1, ...]

    source = copy.deepcopy(mask[:, :, :h // 2, :w - shift])
    mask[:, :, :h // 2, shift:] = source
    mask[:, :, :h // 2, :shift] = 0.

    mask = 1 - mask

    # mask = make_mask(image, image.shape[-1], shuffle_rates=0.)

    # mask生成器
    result = pipeline(
        image * (1 - mask),
        mask,
        embedding_matcher,
        embedder,
        device,
        use_cfg=False,
        num_inference_steps=200
    )

    # 文件夹📁
    save_dir = './test_files/myplan'
    idx = 0
    while os.path.exists(os.path.join(save_dir, f'sample_{idx}')):
        idx += 1

    save_dir = os.path.join(save_dir, f'sample_{idx}')
    os.makedirs(save_dir, exist_ok=True)

    save_image(
        torch.concat([mask] * 3, dim=1),
        os.path.join(save_dir, 'mask.png')
    )

    save_image(
        result,
        os.path.join(save_dir, 'result.png')
    )

    save_image(
        image * (1 - mask),
        os.path.join(save_dir, 'input.png')
    )

    save_image(
        image,
        os.path.join(save_dir, 'gt.png')
    )

