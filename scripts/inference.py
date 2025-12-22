import sys
import os
import argparse
import torch
import torchvision
from diffusers import DDIMScheduler
import omegaconf
import tqdm
import torchvision


sys.path.append(os.getcwd())
from model.Lora import LoraRegister
from model.FeatureExtractor import EmbeddingMatcher
from model.AttentionReplacer import replace_attn_processor
from model.FeatureExtractor.FeatureAugmentor import FeatureAugmentorRegister
from model.EmbeddingManager import VisionTextEmbedding
from model.pipeline import Pipeline
from utils.io import load_image, save_image


def inference(image: torch.Tensor,
              image_ref: torch.Tensor,
              mask: torch.Tensor, 
              pipeline: Pipeline,
              lora_register: LoraRegister,
              augment_register: FeatureAugmentorRegister,
              embedding_matcher: EmbeddingMatcher,
              embedder: VisionTextEmbedding,
              num_inference_steps: int):
    device = image.device
    pipeline.scheduler.set_timesteps(num_inference_steps, device)
    pipeline.scheduler.alphas = pipeline.scheduler.alphas.to(device)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)

    latents = pipeline.image2latents(image)
    mask = torch.nn.functional.interpolate(mask, latents.shape[-2:])
    source_latents = latents

    # blending
    latents_ref = pipeline.image2latents(image_ref)
    image_ref_resized = torch.nn.functional.interpolate(image_ref, [224, 224])
    image_ref_resized = torch.concat([
        torchvision.transforms.functional.rotate(image_ref_resized, i * 90) for i in range(0, 4)
    ], dim=0)
    null_embeddings = pipeline.prepare_prompt_embeddings([''], 1)
    
    lora_register.disable()
    augment_register.disable()
    augment_register.invert()
    # lora_register.enable()

    prompt = ['*', '(', ')', '-']
    embedder.add_tokens(prompt)
    # prompt = '* * * *'
    prompt = '*'
    null_embeddings = pipeline.prepare_prompt_embeddings('', use_conditional_guidance=False)
    image_embeds = embedding_matcher(image_ref_resized)
    # image_embeds = torch.mean(image_embeds, dim=0, keepdim=True)
    encoder_hidden_states = embedder(prompt, image_embeds)

    noised_source_latents_list = list()
    noised_source_latents = source_latents
    # 😤
    with tqdm.tqdm(reversed(pipeline.scheduler.timesteps)) as progress_bar:
        for i, t in enumerate(progress_bar):
            latents = pipeline.add_noise(latents, t, null_embeddings, use_conditional_guidance=False)
            noised_source_latents = pipeline.add_noise(noised_source_latents, t, null_embeddings, use_conditional_guidance=False)
            noised_source_latents_list.append(noised_source_latents)

    latents = torch.randn_like(source_latents)
    with tqdm.tqdm(pipeline.scheduler.timesteps) as progress_bar:
        for i, t in enumerate(progress_bar):
            latents = pipeline.remove_noise(latents, t, encoder_hidden_states, use_conditional_guidance=False)
            if num_inference_steps * .8 > i > num_inference_steps * 1.0:
                noised_source_latents = noised_source_latents_list[num_inference_steps - i - 1].to(device)
                # latents = latents * (1 - mask) + noised_source_latents * mask
                latents = latents * mask * 0.9 + latents * (1 - mask) + noised_source_latents * mask * 0.1

    # with tqdm.tqdm(reversed(pipeline.scheduler.timesteps)) as progress_bar:
    #     for i, t in enumerate(progress_bar):
    #         # if i >= num_inference_steps * 0:
    #         #     break
    #         latents = pipeline.add_noise(latents, t, null_embeddings, use_conditional_guidance=False)
    #         # latents = pipeline.remove_noise(latents, t, null_embeddings, use_conditional_guidance=False)

    # with tqdm.tqdm(pipeline.scheduler.timesteps) as progress_bar:
    #     for i, t in enumerate(progress_bar):
    #         latents = pipeline.remove_noise(latents, t, encoder_hidden_states, use_conditional_guidance=False)
            
    #         if num_inference_steps * .8 > i > num_inference_steps * .0:
    #             noised_source_latents = noised_source_latents_list[num_inference_steps - i - 1].to(device)
    #             # latents = latents * (1 - mask) + noised_source_latents * mask
    #             latents = latents * mask * 0.9 + latents * (1 - mask) + noised_source_latents * mask * 0.1

    result = pipeline.latents2image(latents)
    return result


def main():
    parser = argparse.ArgumentParser('inference')
    parser.add_argument('image_path', type=str, help='path for image to inference')
    parser.add_argument('image_ref_path', type=str, help='path for image to inference')
    parser.add_argument('--mask_path', type=str, default=None, help='mask for the image')
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args()
    image_path = args.image_path
    image_ref_path = args.image_ref_path

    if args.mask_path is None:
        name, ext = os.path.splitext(image_path)
        mask_path = name + '_mask' + ext
    else:
        mask_path = args.mask_path
    
    if args.config is None:
        raise ValueError('config file not specified!')
    config = omegaconf.OmegaConf.load(args.config)
    device = config.get('device', 'cpu')
    device_id = int(config.get('device_id', 0))
    num_inference_steps = config['num_inference_steps']
    stable_diffusion_ckpt = config['stable_diffusion_ckpt']
    embedding_matcher_ckpt = config['embedding_matcher_ckpt']
    clip_vision_transformer_ckpt = config['clip_vision_transformer_ckpt']
    lora_ckpt = config['lora_ckpt']
    augment_ckpt = config['augment_ckpt']
    
    if device not in ['cuda', 'cpu']:
        raise NotImplementedError(f'not support device: {device}')
    if device == 'cuda':
        torch.cuda.set_device(device_id)
    
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False)
    pipeline = Pipeline.from_pretrained(stable_diffusion_ckpt, scheduler=scheduler)
    pipeline.to(device)
    replace_attn_processor(pipeline.unet)
    # lora_register: LoraRegister = LoraRegister.from_pretrained(pipeline.unet, lora_ckpt)
    lora_register = LoraRegister(pipeline.unet, name_list=['attn2'])
    lora_register.load(lora_ckpt)
    lora_register.to(device)
    # lora_register = None
    augment_register: FeatureAugmentorRegister = FeatureAugmentorRegister(pipeline.unet, [4, 1, 4], 1.0)
    augment_register.load(augment_ckpt)
    augment_register.to(device)
    embedding_matcher = EmbeddingMatcher(3, 224, 768, clip_vision_transformer_ckpt)
    embedding_matcher.load_state_dict(torch.load(embedding_matcher_ckpt, weights_only=False))
    embedding_matcher.to(device)
    embedder = VisionTextEmbedding('pretrained/stable-diffusion-v1-4/tokenizer', 
                                   'pretrained/stable-diffusion-v1-4/text_encoder')
    embedder.to(device)
    # embedding_matcher = None
    
    # load files for inference
    image = load_image(image_path, to_batch=True, device=device)
    image_ref = load_image(image_ref_path, to_batch=True, device=device)
    mask = load_image(mask_path, to_batch=True, device=device)[:, 0, :, :].unsqueeze(1)

    result = inference(image, image_ref, mask, pipeline, lora_register, augment_register, embedding_matcher, embedder, num_inference_steps)
    name, ext = os.path.splitext(image_path)
    result_name = name + '_result' + ext
    print(f'Saving result to {result_name}')
    save_image(result, result_name)


if __name__ == '__main__':
    main()
