from __future__ import annotations

import os
from pathlib import Path
import sys

from diffusers import DDPMScheduler
import torch
import torch.nn as nn
import tqdm

sys.path.append(os.getcwd())

from model.pipeline import Pipeline
from model.unet.SiameseUnet import SiameseUnet
from scripts._training.render_pretrained import (
    DEFAULT_RENDER_AUGMENT_CONFIG,
    _build_data_loaders,
    _validate_parquet_training_config,
)
from utils.augment import RenderAugmentor, build_render_augment
from utils.condition_projection import build_condition_projector, prepare_condition
from utils.functionals import MemoryHalt, try_load
from utils.trainer import TrainSpec


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor,
    latents_shape,
    device="cuda",
    num_inference_steps=50,
    verbose=False,
    encoder_attention_mask=None,
    initial_latents: torch.Tensor | None = None,
):
    pipeline.unet.eval()
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = (
        tqdm.tqdm(pipeline.scheduler.timesteps, desc="evaluating", leave=False)
        if verbose
        else pipeline.scheduler.timesteps
    )

    batch_size = masked_ref_latents.shape[0]
    expected_shape = (batch_size, 16, *latents_shape)
    if initial_latents is None:
        latents = torch.randn(
            expected_shape,
            device=masked_ref_latents.device,
        )
    else:
        if tuple(initial_latents.shape) != expected_shape:
            raise ValueError(
                "initial_latents shape must be "
                f"{expected_shape}, got {tuple(initial_latents.shape)}"
            )
        if initial_latents.device != masked_ref_latents.device:
            raise ValueError(
                "initial_latents and masked_ref_latents must be on the same device"
            )
        latents = initial_latents.clone()

    encoder_hidden_states = pipeline.prepare_prompt_embeddings([""] * batch_size)
    for timestep in timesteps:
        timestep = timestep.reshape(1).repeat(batch_size)
        noise_prediction = pipeline.unet(
            latents,
            masked_ref_latents,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        latents = pipeline.scheduler.step(
            noise_prediction,
            timestep[:1],
            latents,
            return_dict=False,
        )[0]

    albedo_latents, normal_latents, height_latents, roughness_latents = latents.chunk(
        4, dim=1
    )
    return (
        pipeline.latents2image(albedo_latents),
        pipeline.latents2image(normal_latents),
        pipeline.latents2image(height_latents),
        pipeline.latents2image(roughness_latents),
    )


def _visual_grid(
    generated_albedo: torch.Tensor,
    color: torch.Tensor,
    generated_normal: torch.Tensor,
    normal: torch.Tensor,
    generated_height: torch.Tensor,
    height: torch.Tensor,
    generated_roughness: torch.Tensor,
    roughness: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    return torch.concat(
        [
            torch.concat(items, dim=-1)
            for items in zip(
                generated_albedo,
                color,
                generated_normal,
                normal,
                generated_height,
                height,
                generated_roughness,
                roughness,
                condition,
            )
        ],
        dim=-2,
    )


@torch.no_grad()
def log_function(
    logger,
    avg_loss,
    global_iter,
    log_config,
    pipeline: Pipeline,
    color: torch.Tensor,
    normal: torch.Tensor,
    height: torch.Tensor,
    roughness: torch.Tensor,
    masked_ref_latents: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    name_space=None,
):
    if logger is not None and global_iter % log_config["log_period_loss"] == 0:
        logger.add_scalars("loss", avg_loss, global_iter)

    if global_iter % log_config["log_period_visual"] != 0:
        return

    (
        generated_albedo,
        generated_normal,
        generated_height,
        generated_roughness,
    ) = evaluate(
        pipeline,
        masked_ref_latents,
        tuple(size // 8 for size in color.shape[-2:]),
        device=masked_ref_latents.device,
        verbose=logger is not None,
        encoder_attention_mask=encoder_attention_mask,
    )
    decoded_reference = nn.functional.interpolate(
        pipeline.latents2image(masked_ref_latents),
        generated_albedo.shape[-2:],
    )
    if logger is not None:
        logger.add_image(
            "evaluate" if name_space is None else name_space,
            _visual_grid(
                generated_albedo,
                color,
                generated_normal,
                normal,
                generated_height,
                height,
                generated_roughness,
                roughness,
                decoded_reference,
            ),
            global_iter,
        )


@torch.no_grad()
def val_function(
    logger,
    global_iter,
    log_config,
    dataloader,
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor | None = None,
    name_space=None,
    device="cuda",
    **kwargs,
):
    should_log_loss = global_iter % log_config["log_period_loss"] == 0
    should_log_visual = global_iter % log_config["log_period_visual"] == 0
    if not should_log_loss and not should_log_visual:
        return

    dataset = dataloader.dataset
    batch_size = log_config["batch_size"]
    indices = torch.randperm(len(dataset))[:batch_size]
    iterations = log_config["num_samples"] if should_log_loss else 1
    total_loss = 0.0
    color = None
    normal = None
    height = None
    roughness = None
    attention_mask = None

    for _ in range(iterations):
        data = dataset[indices]
        image_ref, mask_ref = prepare_condition(data, pipeline)
        image_ref = image_ref.to(device)
        mask_ref = (mask_ref.to(device) > 0.5).to(mask_ref.dtype)
        color = data["Color"].to(device)
        normal = data["NormalGL"].to(device)
        height = data["Height"].to(device)
        roughness = data["Roughness"].to(device)

        color_latents = pipeline.image2latents(color)
        normal_latents = pipeline.image2latents(normal)
        height_latents = pipeline.image2latents(height)
        roughness_latents = pipeline.image2latents(roughness)
        predict_latents = torch.concat(
            [
                color_latents,
                normal_latents,
                height_latents,
                roughness_latents,
            ],
            dim=1,
        )
        masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
        attention_mask = nn.functional.interpolate(
            mask_ref,
            masked_ref_latents.shape[-2:],
        )
        current_batch_size = color.shape[0]
        encoder_hidden_states = pipeline.prepare_prompt_embeddings(
            [""] * current_batch_size
        )
        timesteps = torch.randint(
            0,
            1000,
            (current_batch_size,),
            device=device,
        )
        random_noise = torch.randn_like(predict_latents)
        noised_latents = pipeline.scheduler.add_noise(
            predict_latents,
            random_noise,
            timesteps,
        )
        noise_prediction = pipeline.unet(
            noised_latents,
            masked_ref_latents,
            timesteps,
            encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False,
        )[0]
        if noise_prediction.shape != random_noise.shape:
            raise ValueError(
                "full-attribute noise prediction shape must match target: "
                f"{tuple(noise_prediction.shape)} != {tuple(random_noise.shape)}"
            )
        if should_log_loss:
            total_loss += nn.functional.mse_loss(
                noise_prediction,
                random_noise,
            ).item()

    if should_log_loss and logger is not None:
        logger.add_scalar(
            "test_loss",
            total_loss / iterations,
            global_step=global_iter,
        )

    if should_log_visual:
        (
            generated_albedo,
            generated_normal,
            generated_height,
            generated_roughness,
        ) = evaluate(
            pipeline,
            masked_ref_latents,
            tuple(size // 8 for size in color.shape[-2:]),
            device=masked_ref_latents.device,
            verbose=logger is not None,
            encoder_attention_mask=attention_mask,
        )
        decoded_reference = nn.functional.interpolate(
            pipeline.latents2image(masked_ref_latents),
            generated_albedo.shape[-2:],
        )
        if logger is not None:
            logger.add_image(
                "test_visualize",
                _visual_grid(
                    generated_albedo,
                    color,
                    generated_normal,
                    normal,
                    generated_height,
                    height,
                    generated_roughness,
                    roughness,
                    decoded_reference,
                ),
                global_iter,
            )


def build_joint_optimizer(main_params, ref_params, lr):
    parameters = list(main_params) + list(ref_params)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    return optimizer, [1.0]


ALBEDO_NORMAL_INIT_SHAPE_TRANSITIONS = {
    "unet_main.conv_in.weight": ((320, 8, 3, 3), (320, 16, 3, 3)),
    "unet_main.conv_out.weight": ((8, 320, 3, 3), (16, 320, 3, 3)),
    "unet_main.conv_out.bias": ((8,), (16,)),
}


def try_load_albedo_normal_initialization(unet, checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir) / "unet.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"albedo+normal initialization checkpoint is missing: {checkpoint_path}"
        )

    source_state = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    if not isinstance(source_state, dict):
        raise ValueError(f"{checkpoint_path} does not contain a model state dict")

    target_state = unet.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    source_only = sorted(source_keys - target_keys)
    target_only = sorted(target_keys - source_keys)
    if source_only or target_only:
        raise ValueError(
            "albedo+normal initialization architecture mismatch: "
            f"source_only={source_only}, target_only={target_only}"
        )

    shape_transitions = {
        key: (tuple(source_state[key].shape), tuple(target_state[key].shape))
        for key in sorted(source_keys)
        if source_state[key].shape != target_state[key].shape
    }
    if shape_transitions != ALBEDO_NORMAL_INIT_SHAPE_TRANSITIONS:
        raise ValueError(
            "unexpected albedo+normal initialization shape transitions: "
            f"{shape_transitions}"
        )

    try_load(unet, source_state)
    loaded_key_count = len(source_keys) - len(shape_transitions)
    print(
        "loaded albedo+normal initialization: "
        f"{loaded_key_count} compatible tensors; "
        f"kept {len(shape_transitions)} expanded tensors initialized"
    )
    return {
        "loaded_key_count": loaded_key_count,
        "shape_transitions": shape_transitions,
    }


def load_full_attr_checkpoint(unet, optimizer, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    try:
        unet.load_state_dict(
            torch.load(
                checkpoint_dir / "unet.ckpt",
                weights_only=False,
                map_location="cpu",
            )
        )
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            f"{checkpoint_dir} is not a compatible full-attribute checkpoint"
        ) from error

    optimizer_path = checkpoint_dir / "optimizer.ckpt"
    if optimizer_path.exists():
        optimizer_state = torch.load(
            optimizer_path,
            weights_only=False,
            map_location="cpu",
        )
        saved_group_count = len(optimizer_state.get("param_groups", ()))
        if saved_group_count != 1:
            raise ValueError(
                "full-attribute optimizer checkpoint must contain one joint "
                f"parameter group, got {saved_group_count}"
            )
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as error:
            raise ValueError(
                f"{checkpoint_dir} has an incompatible full-attribute optimizer state"
            ) from error

    try:
        return int(checkpoint_dir.name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def resolve_checkpoint_paths(train_config):
    checkpoint_dir = train_config.get("ckpt_dir", None)
    init_checkpoint_dir = train_config.get("init_ckpt_dir", None)
    if checkpoint_dir and init_checkpoint_dir:
        raise ValueError("ckpt_dir and init_ckpt_dir cannot be set together")
    if init_checkpoint_dir and train_config.get("resume_training", False):
        raise ValueError("init_ckpt_dir requires resume_training=false")
    return checkpoint_dir, init_checkpoint_dir


def prepare_model(cfg, accelerator):
    train_config = cfg["train_config"]
    test_config = cfg["test_config"]
    data_config = cfg["data_config"]
    _validate_parquet_training_config(data_config, train_config)
    condition_projector = build_condition_projector(
        cfg.get("condition_projection"), device=train_config["device"]
    )
    checkpoint_dir, init_checkpoint_dir = resolve_checkpoint_paths(train_config)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )
    scheduler.set_timesteps(1000)
    pipeline = Pipeline.from_pretrained(
        "./pretrained/stable-diffusion-v1-4",
        scheduler=scheduler,
    )
    pipeline.render_augment = build_render_augment(
        cfg.get("augment_config", DEFAULT_RENDER_AUGMENT_CONFIG)
    )
    pipeline.condition_projector = condition_projector

    unet_config = cfg["unet_config"]
    unet = SiameseUnet.from_config(
        unet_config["root"],
        config_main=unet_config["config_main"],
        config_ref=unet_config["config_ref"],
    )
    try_load(unet.unet_main, pipeline.unet.state_dict())
    if init_checkpoint_dir:
        print("loading albedo+normal initialization weights")
        try_load_albedo_normal_initialization(unet, init_checkpoint_dir)
    pipeline.unet = unet
    pipeline.frozen()
    main_params = unet.unfrozen_main()
    ref_params = unet.unfrozen_ref()
    optimizer, lr_gains = build_joint_optimizer(
        main_params,
        ref_params,
        lr=train_config["lr"],
    )
    train_config["lr_gain"] = lr_gains

    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            assert parameter.requires_grad is True

    start_epoch = 0
    if checkpoint_dir:
        print("loading full-attribute resume weights")
        start_epoch = load_full_attr_checkpoint(
            unet,
            optimizer,
            checkpoint_dir,
        )

    device = train_config["device"]
    pipeline.to(device)
    unet.eval()

    memory_halt = MemoryHalt()
    try:
        train_loader, test_loader = _build_data_loaders(
            data_config,
            train_config,
            test_config,
            accelerator,
        )
    finally:
        memory_halt.release()

    modules = {"unet": unet}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        modules,
        train_loader,
        test_loader,
        None,
        optimizer,
        start_epoch,
        pipeline,
    )


def build(cfg, accelerator):
    train_config = cfg["train_config"]
    _validate_parquet_training_config(cfg["data_config"], train_config)
    (
        modules,
        train_loader,
        test_loader,
        _val_loader,
        optimizer,
        start_epoch,
        pipeline,
    ) = prepare_model(cfg, accelerator)

    return TrainSpec(
        modules=modules,
        training_step=training_step,
        train_loader=train_loader,
        eval_loader=None,
        test_loader=test_loader,
        optimizer=optimizer,
        before_optimizer_step=None,
        log_train=log_function,
        log_test=val_function,
        train_kwargs={
            "pipeline": pipeline,
            "device": train_config["device"],
            "augment": pipeline.render_augment,
        },
        start_epoch=start_epoch,
    )


def training_step(
    module: dict[str, nn.Module],
    data,
    pipeline: Pipeline,
    device="cuda",
    augment: RenderAugmentor | None = None,
):
    if augment is None:
        raise ValueError("training_step requires an augment callable")

    unet = module["unet"]
    image_ref, mask_ref = prepare_condition(data, pipeline)
    image_ref, mask_ref, color, normal, height, roughness = augment(
        image_ref,
        mask_ref,
        data["Color"],
        data["NormalGL"],
        data["Height"],
        data["Roughness"],
        data["meta_data"],
        device,
    )

    batch_size = color.shape[0]
    color_latents = pipeline.image2latents(color)
    normal_latents = pipeline.image2latents(normal)
    height_latents = pipeline.image2latents(height)
    roughness_latents = pipeline.image2latents(roughness)
    predict_latents = torch.concat(
        [
            color_latents,
            normal_latents,
            height_latents,
            roughness_latents,
        ],
        dim=1,
    )
    masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
    attention_mask = nn.functional.interpolate(
        mask_ref,
        masked_ref_latents.shape[-2:],
    )
    encoder_hidden_states = pipeline.prepare_prompt_embeddings([""] * batch_size)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    random_noise = torch.randn_like(predict_latents)
    noised_latents = pipeline.scheduler.add_noise(
        predict_latents,
        random_noise,
        timesteps,
    )
    noise_prediction = unet(
        noised_latents,
        masked_ref_latents,
        timesteps,
        encoder_hidden_states,
        encoder_attention_mask=attention_mask,
        return_dict=False,
    )[0]
    if noise_prediction.shape != random_noise.shape:
        raise ValueError(
            "full-attribute noise prediction shape must match target: "
            f"{tuple(noise_prediction.shape)} != {tuple(random_noise.shape)}"
        )

    loss = nn.functional.mse_loss(noise_prediction, random_noise)
    return_examples = min(batch_size, 8)
    log_parameters = {
        "color": color[:return_examples],
        "normal": normal[:return_examples],
        "height": height[:return_examples],
        "roughness": roughness[:return_examples],
        "pipeline": pipeline,
        "masked_ref_latents": masked_ref_latents[:return_examples],
        "encoder_attention_mask": attention_mask[:return_examples],
    }
    return loss, {"loss": loss.item()}, log_parameters
