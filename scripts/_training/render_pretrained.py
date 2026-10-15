import torch
import torch.nn as nn
import sys
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
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
from omegaconf import OmegaConf
from typing import Callable
import accelerate
import numpy as np
from torchvision import transforms

sys.path.append(os.getcwd())

from utils.datasets.DataLoader import (
    PBRTextureDataLoader,
    TestPolyLoader,
    PolyLoader
)
from utils.datasets.pbr_parquet import (
    PBRParquetBatchSampler,
    PBRParquetDataset,
)

from model.pipeline import Pipeline

from utils.functionals import (
    crop_mid, 
    try_load, 
    freeze_net,
    patch_shuffle,
    generate_gaussian,
    sample_exponential_decay,
    MemoryHalt
)
from utils.transforms import remap_normal_axes
from utils.io import save_image, load_image
from utils.augment import RenderAugmentor, build_render_augment
from model.unet.SiameseUnet import SiameseUnet, Adapter, FeatureAdapter
from model.unet.attn_main import BasicTransformerBlockMain
from utils.trainer import TrainSpec

from utils.trainer.constants import LOCAL_BATCH_SIZE_KEY
DEFAULT_RENDER_AUGMENT_CONFIG = {
    'rotate90': {
        'enabled': True,
        'angles': [0., 90., 180., 270.],
    },
    'jitter': {
        'enabled': True,
        'range_deg': 10.,
    },
    'align_rotation': {
        'enabled': True,
    },
    'crop': {
        'size': 352,
    },
    'mask_aug': {
        'enabled': True,
        'tps_scale': [0.1, 0.3],
        'tps_p': 0.8,
        'occ_size': 512,
        'binarize_thresh': 0.5,
    },
    'photometric': {
        'enabled': False,
    },
    'R': 512,
}


@dataclass(frozen=True)
class ConditionPathUpdatePhase:
    until_step: int
    condition_only_prob: float


@dataclass(frozen=True)
class ConditionPathUpdateConfig:
    enabled: bool = False
    seed: int = 0
    main_backbone_lr_gain: float = 1.0
    main_bridge_lr_gain: float = 1.0
    ref_lr_gain: float = 1.0
    phases: tuple[ConditionPathUpdatePhase, ...] = ()


def _validate_exact_keys(config, allowed, context, *, required=()):
    keys = set(config)
    unknown = keys - set(allowed)
    missing = set(required) - keys
    if unknown:
        raise ValueError(f'{context} has unknown keys: {sorted(unknown)}')
    if missing:
        raise ValueError(f'{context} is missing keys: {sorted(missing)}')


def _positive_float(value, context):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{context} must be a positive number')
    value = float(value)
    if value <= 0:
        raise ValueError(f'{context} must be > 0, got {value}')
    return value


def parse_condition_path_update_config(raw_config, max_steps):
    """Validate the optional condition-path optimizer/update schedule."""
    if raw_config is None:
        return ConditionPathUpdateConfig()
    if not isinstance(raw_config, Mapping):
        raise ValueError('condition_path_update must be a mapping')

    enabled = raw_config.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('condition_path_update.enabled must be a boolean')
    if not enabled:
        _validate_exact_keys(
            raw_config,
            {'enabled'},
            'condition_path_update',
        )
        return ConditionPathUpdateConfig()

    _validate_exact_keys(
        raw_config,
        {'enabled', 'seed', 'lr_gain', 'phases'},
        'condition_path_update',
        required={'lr_gain', 'phases'},
    )
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps <= 0:
        raise ValueError('train_config.max_steps must be a positive integer')

    seed = raw_config.get('seed', 0)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('condition_path_update.seed must be a non-negative integer')

    lr_gain = raw_config['lr_gain']
    if not isinstance(lr_gain, Mapping):
        raise ValueError('condition_path_update.lr_gain must be a mapping')
    _validate_exact_keys(
        lr_gain,
        {'main_backbone', 'main_bridge', 'ref'},
        'condition_path_update.lr_gain',
        required={'main_backbone', 'main_bridge', 'ref'},
    )
    main_backbone_lr_gain = _positive_float(
        lr_gain['main_backbone'],
        'condition_path_update.lr_gain.main_backbone',
    )
    main_bridge_lr_gain = _positive_float(
        lr_gain['main_bridge'],
        'condition_path_update.lr_gain.main_bridge',
    )
    ref_lr_gain = _positive_float(
        lr_gain['ref'],
        'condition_path_update.lr_gain.ref',
    )

    raw_phases = raw_config['phases']
    if (
        not isinstance(raw_phases, Sequence)
        or isinstance(raw_phases, (str, bytes))
        or not raw_phases
    ):
        raise ValueError('condition_path_update.phases must be a non-empty sequence')

    phases = []
    previous_until_step = 0
    for index, raw_phase in enumerate(raw_phases):
        context = f'condition_path_update.phases[{index}]'
        if not isinstance(raw_phase, Mapping):
            raise ValueError(f'{context} must be a mapping')
        _validate_exact_keys(
            raw_phase,
            {'until_step', 'condition_only_prob'},
            context,
            required={'until_step', 'condition_only_prob'},
        )
        until_step = raw_phase['until_step']
        if isinstance(until_step, bool) or not isinstance(until_step, Integral):
            raise ValueError(f'{context}.until_step must be an integer')
        until_step = int(until_step)
        if until_step <= previous_until_step:
            raise ValueError(
                'condition_path_update phase until_step values must be '
                'strictly increasing'
            )
        probability = raw_phase['condition_only_prob']
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise ValueError(f'{context}.condition_only_prob must be numeric')
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f'{context}.condition_only_prob must be within [0, 1], '
                f'got {probability}'
            )
        phases.append(ConditionPathUpdatePhase(until_step, probability))
        previous_until_step = until_step

    if phases[-1].until_step != int(max_steps):
        raise ValueError(
            'condition_path_update phases must end exactly at '
            f'train_config.max_steps={max_steps}'
        )

    return ConditionPathUpdateConfig(
        enabled=True,
        seed=int(seed),
        main_backbone_lr_gain=main_backbone_lr_gain,
        main_bridge_lr_gain=main_bridge_lr_gain,
        ref_lr_gain=ref_lr_gain,
        phases=tuple(phases),
    )


def split_condition_path_parameters(unet, unet_main_params, unet_ref_params):
    """Partition existing trainable parameters without changing the model."""
    main_params = list(unet_main_params)
    ref_params = list(unet_ref_params)
    main_ids = [id(param) for param in main_params]
    ref_ids = [id(param) for param in ref_params]
    if len(main_ids) != len(set(main_ids)):
        raise ValueError('unet_main_params contains duplicate parameters')
    if len(ref_ids) != len(set(ref_ids)):
        raise ValueError('unet_ref_params contains duplicate parameters')
    if set(main_ids) & set(ref_ids):
        raise ValueError('Main and Ref parameter sets overlap')

    bridge_ids = set()
    for module in unet.unet_main.modules():
        if not isinstance(module, BasicTransformerBlockMain):
            continue
        for bridge_name in ('norm2', 'attn2'):
            bridge_module = getattr(module, bridge_name, None)
            if bridge_module is None:
                continue
            bridge_ids.update(
                id(param)
                for param in bridge_module.parameters()
                if param.requires_grad
            )

    if not bridge_ids:
        raise ValueError('no trainable BasicTransformerBlockMain norm2/attn2 parameters found')
    unexpected_bridge_ids = bridge_ids - set(main_ids)
    if unexpected_bridge_ids:
        raise ValueError('bridge parameters are missing from unet_main_params')

    groups = {
        'main_backbone': [param for param in main_params if id(param) not in bridge_ids],
        'main_bridge': [param for param in main_params if id(param) in bridge_ids],
        'ref': ref_params,
    }
    grouped_ids = [id(param) for params in groups.values() for param in params]
    expected_ids = main_ids + ref_ids
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(expected_ids):
        raise ValueError('condition-path parameter groups are not disjoint and complete')
    if {id(param) for param in groups['main_bridge']} != bridge_ids:
        raise ValueError('condition-path bridge grouping is incomplete')
    return groups


def build_condition_path_optimizer(
    unet,
    unet_main_params,
    unet_ref_params,
    *,
    lr,
    config,
):
    """Build the legacy optimizer or the enabled three-group optimizer."""
    if not config.enabled:
        params = list(unet_main_params) + list(unet_ref_params)
        optimizer = torch.optim.AdamW([{'params': params, 'lr': lr}], lr=lr)
        return optimizer, [1.0]

    groups = split_condition_path_parameters(
        unet,
        unet_main_params,
        unet_ref_params,
    )
    gains = [
        config.main_backbone_lr_gain,
        config.main_bridge_lr_gain,
        config.ref_lr_gain,
    ]
    names = ['main_backbone', 'main_bridge', 'ref']
    optimizer_groups = [
        {
            'name': name,
            'params': groups[name],
            'lr': lr * gain,
        }
        for name, gain in zip(names, gains)
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr)
    return optimizer, gains


def _mix_condition_path_seed(seed, step):
    mask = (1 << 64) - 1
    value = (int(seed) + int(step) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


class ConditionPathUpdatePolicy:
    def __init__(self, config):
        if not config.enabled:
            raise ValueError('ConditionPathUpdatePolicy requires an enabled config')
        self.config = config
        self.condition_only_count = 0
        self.total_count = 0

    def probability_for_step(self, step):
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError('condition-path optimizer step must be non-negative')
        for phase in self.config.phases:
            if step < phase.until_step:
                return phase.condition_only_prob
        raise ValueError(
            f'condition-path optimizer step {step} exceeds configured phases'
        )

    def is_condition_only_step(self, step):
        probability = self.probability_for_step(step)
        if probability <= 0:
            return False
        if probability >= 1:
            return True
        mixed_seed = _mix_condition_path_seed(self.config.seed, step)
        return random.Random(mixed_seed).random() < probability

    def __call__(self, *, optimizer, step):
        groups = {
            group.get('name'): group
            for group in optimizer.param_groups
            if group.get('name') is not None
        }
        required_groups = {'main_backbone', 'main_bridge', 'ref'}
        if set(groups) != required_groups:
            raise ValueError(
                'condition-path optimizer requires named groups '
                'main_backbone/main_bridge/ref'
            )

        condition_only = self.is_condition_only_step(step)
        if condition_only:
            for parameter in groups['main_backbone']['params']:
                parameter.grad = None

        self.total_count += 1
        self.condition_only_count += int(condition_only)
        return {
            'condition_only_update': float(condition_only),
            'condition_only_update_rate': self.condition_only_count / self.total_count,
            'main_backbone_update': float(not condition_only),
            'lr_main_backbone': float(groups['main_backbone']['lr']),
            'lr_main_bridge': float(groups['main_bridge']['lr']),
            'lr_ref': float(groups['ref']['lr']),
        }

def get_cropsize(theta: float, d: list[float], base_angle=np.pi / 4, eps=1e-6):
    theta0 = theta - base_angle
    theta1 = theta + base_angle

    d0 = d[0] / abs(np.sin(theta0 + eps))
    d1 = d[1] / abs(np.sin(theta1 + eps))

    l = min(d0, d1)

    return int(l * np.sin(base_angle)), int(l * np.cos(base_angle))


_META_DATA_KEYS = (
    'theta',
    'phi',
    'radius',
    'camera_direction',
    'head_direction',
)


def _load_metadata(value):
    if isinstance(value, Mapping):
        metadata = value
    elif isinstance(value, (str, os.PathLike)):
        metadata = OmegaConf.load(value)
    else:
        raise TypeError(
            'meta_data must be a Mapping or YAML path, '
            f'got {type(value).__name__}'
        )
    missing = [key for key in _META_DATA_KEYS if key not in metadata]
    if missing:
        raise KeyError(f'meta_data is missing required fields: {missing}')
    return metadata


def collate_fn(batch: list[dict]):
    if not batch:
        raise ValueError('cannot collate an empty batch')
    keys = tuple(batch[0])
    if any(tuple(sample) != keys for sample in batch[1:]):
        raise ValueError('batch samples must have identical top-level keys')

    result_dict = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            try:
                result_dict[key] = torch.concat(
                    [sample[key].unsqueeze(0) for sample in batch]
                )
            except Exception:
                print([sample[key].shape for sample in batch])
                raise
        else:
            if key != 'meta_data':
                raise TypeError(f'unsupported non-Tensor batch field: {key}')
            metadata = [_load_metadata(sample[key]) for sample in batch]
            result_dict[key] = {
                metadata_key: [
                    sample_metadata[metadata_key]
                    for sample_metadata in metadata
                ]
                for metadata_key in _META_DATA_KEYS
            }
    return result_dict


def parquet_collate_fn(batch: list[dict]):
    result = collate_fn(batch)
    result[LOCAL_BATCH_SIZE_KEY] = torch.tensor(len(batch), dtype=torch.int64)
    return result


def _identity_transform(value):
    return value


def _config_int(config, key, default, *, minimum=0):
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f'{key} must be an integer >= {minimum}')
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{key} must be an integer >= {minimum}') from exc
    if value < minimum:
        raise ValueError(f'{key} must be an integer >= {minimum}')
    return value


def _config_bool(config, key, default):
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f'{key} must be a boolean')
    return value


def _plain_mapping(value):
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError('expected a mapping config')
    return dict(value)


def _validate_parquet_training_config(data_config, train_config):
    if 'parquet_dir' not in data_config:
        return
    if train_config['accumulation_steps'] != 1:
        raise ValueError(
            'PBR Parquet v1 requires train_config.accumulation_steps=1'
        )
    if train_config.get('even_batches', True) is not False:
        raise ValueError(
            'PBR Parquet v1 requires train_config.even_batches=false'
        )
    if _config_bool(data_config, 'persistent_workers', False):
        raise ValueError('PBR Parquet v1 requires persistent_workers=false')


def _build_data_loaders(data_config, train_config, test_config, accelerator):
    _validate_parquet_training_config(data_config, train_config)
    num_workers = _config_int(data_config, 'num_workers', 4, minimum=0)
    persistent_workers = _config_bool(
        data_config,
        'persistent_workers',
        False,
    )
    pin_memory = _config_bool(data_config, 'pin_memory', False)
    if persistent_workers and num_workers == 0:
        raise ValueError('persistent_workers requires num_workers > 0')

    worker_kwargs = {
        'num_workers': num_workers,
        'persistent_workers': persistent_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        worker_kwargs['prefetch_factor'] = _config_int(
            data_config,
            'prefetch_factor',
            2,
            minimum=1,
        )

    image_names = tuple(data_config['image_names'])
    fetch_attr = tuple(data_config['fetch_attr'])
    transform_group = _plain_mapping(data_config.get('transform_group', {}))
    transforms_by_name = {
        transform_name: _identity_transform
        for transform_name in transform_group
    }
    train_batch_size = _config_int(
        train_config,
        'batch_size',
        None,
        minimum=1,
    )
    test_batch_size = _config_int(
        test_config,
        'batch_size',
        None,
        minimum=1,
    )

    if 'parquet_dir' in data_config:
        if image_names != ('image', 'mask', 'meta_data'):
            raise ValueError(
                'PBR Parquet v1 image_names must be '
                '[image, mask, meta_data]'
            )
        if fetch_attr != ('Color', 'NormalGL', 'Height', 'Roughness'):
            raise ValueError(
                'PBR Parquet v1 fetch_attr must be '
                '[Color, NormalGL, Height, Roughness]'
            )
        shuffle_seed = _config_int(
            data_config,
            'shuffle_seed',
            0,
            minimum=0,
        )
        train_dataset = PBRParquetDataset(
            data_config['parquet_dir'],
            image_names=image_names,
            fetch_attr=fetch_attr,
            transforms=transforms_by_name,
            transform_group=transform_group,
        )
        test_dataset = PBRParquetDataset(
            data_config['parquet_dir'],
            image_names=image_names,
            fetch_attr=fetch_attr,
            transforms=transforms_by_name,
            transform_group=transform_group,
        )
        batch_sampler = PBRParquetBatchSampler(
            train_dataset,
            configured_batch_size=train_batch_size,
            num_processes=int(accelerator.num_processes),
            num_workers=num_workers,
            seed=shuffle_seed,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=parquet_collate_fn,
            **worker_kwargs,
        )
        batch_sampler.validate_dataloader(train_loader)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **worker_kwargs,
        )
        return train_loader, test_loader

    PBRTextureDataLoader.image_names = image_names
    gt_mapping = _plain_mapping(data_config.get('gt_mapping', {})) or None
    train_dataset = PBRTextureDataLoader(
        fp=data_config['train_fp'],
        gt_fp=data_config['gt_fp'],
        transforms=transforms_by_name,
        fetch_attr=fetch_attr,
        selected_files=None,
        good_examples=None,
        transform_group=transform_group,
        gt_mapping=gt_mapping,
    )
    test_dataset = PBRTextureDataLoader(
        fp=data_config['train_fp'],
        gt_fp=data_config['gt_fp'],
        transforms=transforms_by_name,
        fetch_attr=fetch_attr,
        selected_files=None,
        good_examples=None,
        transform_group=transform_group,
        gt_mapping=gt_mapping,
    )
    train_dataset.shuffle()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **worker_kwargs,
    )
    return train_loader, test_loader


def align_attr(color, normal, height, roughness, meta_data: dict, transform_func: Callable):
    # FIXME apply transform base on meta data
    bsz = color.shape[0]

    thetas = meta_data['theta']
    concat_attr = [color, normal, height, roughness]
    splits_size = [attr.shape[1] for attr in concat_attr]
    normal_offset = sum(splits_size[:1])
    concat_attr = torch.concat(concat_attr, dim=1)

    angles = [-(thetas[i] / np.pi * 180. + 90.) for i in range(bsz)]
    concat_attr = [torchvision.transforms.functional.rotate(concat_attr[i:i+1], angles[i]) for i in range(bsz)]
    for i in range(bsz):
        concat_attr[i][:, normal_offset:normal_offset + 3] = remap_normal_axes(
            concat_attr[i][:, normal_offset:normal_offset + 3], angles[i]
        )
    crop_sizes = [get_cropsize(thetas[i], concat_attr[i].shape[-2:]) for i in range(bsz)]
    concat_attr = [crop_mid(concat_attr[i], crop_sizes[i]) for i in range(bsz)]
    # scale transform
    # radius_min, radius_max = 2., 3.
    # base_radius = 3.
    # scale_factors = [base_radius / radius for radius in meta_data['radius']]
    # concat_attr = [nn.functional.interpolate(concat_attr[i], scale_factor=scale_factors[i]) for i in range(bsz)]

    concat_attr = torch.concat([transform_func(attr) for attr in concat_attr], dim=0)
    color, normal, height, roughness = concat_attr.split(splits_size, dim=1)

    return color, normal, height, roughness


@torch.no_grad()
def evaluate(
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor,
    latents_shape,
    device='cuda',
    num_inference_steps=50,
    verbose=False,
    cond_scale=3.,
    adapter: FeatureAdapter=None,
    encoder_attention_mask=None,
    initial_latents: torch.Tensor=None,
):    
    pipeline.unet.eval()

    # pipeline.scheduler = scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    pipeline.scheduler_to(device)
    timesteps = tqdm.tqdm(pipeline.scheduler.timesteps, desc='evaluating', leave=False) if verbose else pipeline.scheduler.timesteps

    bsz = masked_ref_latents.shape[0]
    expected_shape = (bsz, 16, *latents_shape)
    if initial_latents is None:
        latents_start = torch.randn(
            expected_shape,
            device=masked_ref_latents.device,
        )
    else:
        if tuple(initial_latents.shape) != expected_shape:
            raise ValueError(
                'initial_latents shape must be '
                f'{expected_shape}, got {tuple(initial_latents.shape)}'
            )
        if initial_latents.device != masked_ref_latents.device:
            raise ValueError(
                'initial_latents and masked_ref_latents must be on the same device'
            )
        latents_start = initial_latents.clone()

    # encoder_hidden_states = feature_encoder(image_ref, mask_ref).transpose(-1, -2)
    # if int(os.environ['RANK']) == 0:
    #     print(image_ref.max(), image_ref.min(), mask_ref.max(), mask_ref.min())
    encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)
    
    for t in timesteps:
        t = t.reshape([1,]).repeat(bsz)

        latents_input = latents_start
        # latents_input = torch.concat([latents_start, masked_ref_latents], dim=1)
        latents_input_ref = masked_ref_latents

        if cond_scale is not None:
            latents_input_null = latents_input
            latents_input = torch.concat([latents_input, latents_input_null], dim=0)

            latents_input_ref = torch.concat([latents_input_ref, torch.zeros_like(latents_input_ref)])

        noise_pred = pipeline.unet(
            latents_input,
            latents_input_ref,
            t, 
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

        if cond_scale is not None:
            noise_pred_cond, noise_pred_null = noise_pred.chunk(2, dim=0)
            # cfg
            noise_pred = noise_pred_null + cond_scale * (noise_pred_cond - noise_pred_null)

        t = t[:1]
        latents_start = pipeline.scheduler.step(noise_pred, t, latents_start, return_dict=False)[0]
    
    # return latents2image(pipeline, latents_start)
    color = pipeline.latents2image(latents_start[:, :4, ...])
    normal = pipeline.latents2image(latents_start[:, 4:8, ...])
    height = pipeline.latents2image(latents_start[:, 8:12, ...])
    roughness = pipeline.latents2image(latents_start[:, 12:, ...])

    return color, normal, height, roughness


@torch.no_grad()
def log_function(
    logger: SummaryWriter,
    avg_loss: dict,
    global_iter: int,
    log_config: dict,
    pipeline: Pipeline,
    # feature_encoder: FeatureEncoder,
    color: torch.Tensor,
    normal: torch.Tensor,
    height: torch.Tensor,
    roughness: torch.Tensor,
    masked_ref_latents: torch.Tensor,
    image_ref: torch.Tensor,
    mask_ref: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
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
        eval_result = evaluate(
            pipeline,
            masked_ref_latents,
            [s // 8 for s in color.shape[-2:]],
            verbose=logger is not None,
            cond_scale=None,
            # encoder_attention_mask=encoder_attention_mask
            encoder_attention_mask=None
        )

        eval_color, eval_normal, eval_height, eval_roughness = eval_result
        masked_ref_latents = nn.functional.interpolate(pipeline.latents2image(masked_ref_latents), eval_color.shape[-2:])

        if logger is not None:
            logger.add_image(
                'evaluate' if name_space is None else name_space,
                torch.concat([
                    torch.concat([
                        ev_c, ev_n, ev_m, ev_r,c, n, m, r, m_rf_la, 
                    ], dim=-1) for ev_c, ev_n, ev_m, ev_r, c, n, m, r, m_rf_la in \
                        zip(eval_color, eval_normal, eval_height, eval_roughness, color, normal, height, roughness, masked_ref_latents)
                ], dim=-2),
                # torch.concat([
                #     torch.concat([
                #         e_rs, im, latents2image(pipeline, m_rf_la.unsqueeze(0)).squeeze(0), im_ref, 
                #     ], dim=-1) for e_rs, im, m_rf_la, im_ref in zip(eval_result, image, masked_ref_latents, image_ref)
                # ], dim=-2),
                global_iter
            )
    return


@torch.no_grad()
def val_function(
    logger: SummaryWriter,
    global_iter: int,
    log_config: dict,
    dataloader: torch.utils.data.DataLoader,
    pipeline: Pipeline,
    masked_ref_latents: torch.Tensor,
    name_space: str=None,
    device='cuda',
    **kwargs
):
    log_period_visual = log_config['log_period_visual']
    log_period_loss = log_config['log_period_loss']
    
    unet = pipeline.unet

    # total_samples = len(test_bar)
    total_loss = 0.
    num_samples = log_config['num_samples']
    batch_size = log_config['batch_size']
    dataset = dataloader.dataset

    idx = torch.randperm(len(dataset))[:batch_size]
    if global_iter % log_period_loss == 0:
        for i in range(num_samples) if logger is None else tqdm.trange(num_samples, leave=False):
            data = dataset[idx]
            image_key = 'image_project' if 'image_project' in data else 'image'
            mask_key = 'mask_project' if 'mask_project' in data else 'mask'
            image_ref, mask_ref, color, normal, height, roughness = data[image_key], data[mask_key], data['Color'], data['NormalGL'], data['Height'], data['Roughness']

            image_ref = image_ref.to(device)
            mask_ref = mask_ref.to(device)
            color = color.to(device)
            normal = normal.to(device)
            height = height.to(device)
            roughness = roughness.to(device)

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
            attn_mask = nn.functional.interpolate(mask_ref, masked_ref_latents.shape[-2:])
            
            # color: torch.Tensor = (color - 0.5) * 2
            # normal: torch.Tensor = (normal - 0.5) * 2
            # height: torch.Tensor = (height - 0.5) * 2
            # roughness: torch.Tensor = (roughness - 0.5) * 2

            image_ref: torch.Tensor = (image_ref - 0.5) * 2
            
            encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)

            t = torch.randint(0, 1000, (bsz,), device=device)

            predict_latents = torch.concat([
                color_latents, normal_latents, height_latents, roughness_latents
            ], dim=1)

            random_noise = torch.randn_like(predict_latents)
            noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

            # 16 channels
            latents_input = noised_latents
            # 20 channels
            # latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)
            latents_input_ref = masked_ref_latents

            noise_pred = unet(
                latents_input,
                latents_input_ref,
                t, 
                encoder_hidden_states, 
                # encoder_attention_mask=attn_mask,
                encoder_attention_mask=None,
                return_dict=False, 
            )[0]

            loss = nn.functional.mse_loss(noise_pred, random_noise)
            total_loss += loss.item()

            if i >= (num_samples - 1):
                break

        avg_loss = total_loss / (i + 1)

        if logger is not None:
            logger.add_scalar('test_loss', avg_loss, global_step=global_iter)

    if global_iter % log_period_visual == 0:
        eval_result = evaluate(
            pipeline,
            masked_ref_latents,
            [s // 8 for s in color.shape[-2:]],
            verbose=logger is not None,
            cond_scale=None,
            # encoder_attention_mask=attn_mask,
            encoder_attention_mask=None,
        )

        eval_color, eval_normal, eval_height, eval_roughness = eval_result
        masked_ref_latents = nn.functional.interpolate(pipeline.latents2image(masked_ref_latents), eval_color.shape[-2:])

        if logger is not None:
            logger.add_image(
                'test_visualize',
                torch.concat([
                    torch.concat([
                        ev_c, ev_n, ev_m, ev_r,c, n, m, r, m_rf_la, 
                    ], dim=-1) for ev_c, ev_n, ev_m, ev_r, c, n, m, r, m_rf_la in \
                        zip(eval_color, eval_normal, eval_height, eval_roughness, color, normal, height, roughness, masked_ref_latents)
                ], dim=-2),
                global_iter
            )

    return


PBR_DENOISING_LOSS_WEIGHTS = (0.4, 0.2, 0.2, 0.2)
CONDITION_NULL_METRIC_NAMES = (
    'condition_cond_loss',
    'condition_null_loss',
    'condition_null_rank_loss',
    'condition_null_rank_active_rate',
)


def pbr_denoising_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, ...] = PBR_DENOISING_LOSS_WEIGHTS,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim < 2:
        raise ValueError("prediction and target must include batch and channel dimensions")
    if not weights or prediction.shape[1] != 4 * len(weights):
        raise ValueError(
            f"expected {4 * len(weights)} channels for {len(weights)} PBR blocks, "
            f"got {prediction.shape[1]}"
        )

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("PBR denoising loss weights must sum to a positive value")

    per_sample_loss = prediction.new_zeros(prediction.shape[0])
    for index, weight in enumerate(weights):
        block_error = (
            prediction[:, index * 4 : (index + 1) * 4]
            - target[:, index * 4 : (index + 1) * 4]
        ).square()
        per_sample_loss = per_sample_loss + block_error.flatten(1).mean(1) * weight
    return per_sample_loss / weight_sum


def validate_condition_null_contrastive_config(
    weight: float,
    relative_margin: float,
) -> tuple[float, float]:
    try:
        weight = float(weight)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"condition_null_contrastive_weight must be numeric, got {weight!r}"
        ) from error
    try:
        relative_margin = float(relative_margin)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"condition_null_contrastive_margin must be numeric, got {relative_margin!r}"
        ) from error

    if weight < 0:
        raise ValueError(
            f"condition_null_contrastive_weight must be >= 0, got {weight}"
        )
    if not 0.0 <= relative_margin < 1.0:
        raise ValueError(
            "condition_null_contrastive_margin must be within [0, 1), "
            f"got {relative_margin}"
        )
    return weight, relative_margin


def compute_condition_null_contrastive_loss(
    conditional_prediction: torch.Tensor,
    target: torch.Tensor,
    eligible_mask: torch.Tensor,
    null_prediction_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    weight: float,
    relative_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight, relative_margin = validate_condition_null_contrastive_config(
        weight,
        relative_margin,
    )
    if conditional_prediction.shape != target.shape:
        raise ValueError("conditional prediction and target must have the same shape")
    if eligible_mask.dtype != torch.bool or eligible_mask.shape != (
        conditional_prediction.shape[0],
    ):
        raise ValueError("eligible_mask must be a boolean tensor with one value per sample")

    zero = conditional_prediction.new_zeros(())
    zero_metrics = {name: zero for name in CONDITION_NULL_METRIC_NAMES}
    if weight == 0.0 or not torch.any(eligible_mask):
        return zero, zero_metrics

    conditional_loss = pbr_denoising_mse_per_sample(
        conditional_prediction,
        target,
    )[eligible_mask]
    with torch.no_grad():
        null_prediction = null_prediction_fn(eligible_mask)
        eligible_target = target[eligible_mask]
        if null_prediction.shape != eligible_target.shape:
            raise ValueError(
                "null prediction must match the eligible target shape, got "
                f"{tuple(null_prediction.shape)} and {tuple(eligible_target.shape)}"
            )
        null_loss = pbr_denoising_mse_per_sample(
            null_prediction,
            eligible_target,
        ).detach()

    rank_loss = torch.relu(
        conditional_loss - (1.0 - relative_margin) * null_loss
    )
    metrics = {
        'condition_cond_loss': conditional_loss.detach().mean(),
        'condition_null_loss': null_loss.mean(),
        'condition_null_rank_loss': rank_loss.detach().mean(),
        'condition_null_rank_active_rate': (rank_loss.detach() > 0).float().mean(),
    }
    return weight * rank_loss.mean(), metrics


def apply_condition_dropout(
    condition: torch.Tensor,
    attention_mask: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"condition_dropout_prob must be within [0, 1], got {probability}"
        )

    dropped = torch.rand(condition.shape[0], device=condition.device) < probability
    if not torch.any(dropped):
        return condition, attention_mask, dropped

    condition = condition.clone()
    attention_mask = attention_mask.clone()
    condition[dropped] = 0
    attention_mask[dropped] = 1
    return condition, attention_mask, dropped


def training_step(
    module: list[nn.Module],
    data,
    pipeline: Pipeline,
    device='cuda',
    augment: RenderAugmentor=None,
    condition_dropout_prob: float = 0.0,
    condition_null_contrastive_weight: float = 0.0,
    condition_null_contrastive_margin: float = 0.05,
):
    unet = module['unet']

    # Current MatSynth render exports use image/mask; older projected renders use image_project/mask_project.
    image_key = 'image_project' if 'image_project' in data else 'image'
    mask_key = 'mask_project' if 'mask_project' in data else 'mask'
    image_ref, mask_ref, color, normal, height, roughness = data[image_key], data[mask_key], data['Color'], data['NormalGL'], data['Height'], data['Roughness']
    bsz = color.shape[0]
    if augment is None:
        raise ValueError("training_step requires an augment callable")

    image_ref, mask_ref, color, normal, height, roughness = augment(
        image_ref, mask_ref, color, normal, height, roughness, data['meta_data'], device
    )

    # augment_idx = torch.rand((bsz,), device=device) < 0.8
    # augment_idx = augment_idx.to(torch.long)
    # image_ref += (0.025 * (torch.randn_like(image_ref) + 0.5) + 0.025 * torch.rand_like(image_ref)) * augment_idx.reshape(bsz, 1, 1, 1)
    # image_ref = image_ref.clip(0., 1.)
    # original_shape = copy.deepcopy(image_ref.shape[-2:])
    # image_ref_degrade = nn.functional.interpolate(image_ref, [image_ref.shape[-2] // 4, image_ref.shape[-1] // 4])
    # image_ref_degrade = nn.functional.interpolate(image_ref_degrade, original_shape)

    # image_ref = nn.functional.interpolate(image_ref, [image_ref.shape[-2] // 2, image_ref.shape[-1] // 2])
    # image_ref = nn.functional.interpolate(image_ref, original_shape)
    # image_ref[augment_idx] = image_ref_degrade[augment_idx]

    color_latents = pipeline.image2latents(color)
    normal_latents = pipeline.image2latents(normal)
    height_latents = pipeline.image2latents(height)
    roughness_latents = pipeline.image2latents(roughness)

    masked_ref_latents = pipeline.image2latents(image_ref, mask_ref)
    attn_mask = nn.functional.interpolate(mask_ref, masked_ref_latents.shape[-2:])
    masked_ref_latents, attn_mask, condition_dropped = apply_condition_dropout(
        masked_ref_latents,
        attn_mask,
        condition_dropout_prob,
    )
    
    color: torch.Tensor = (color - 0.5) * 2
    normal: torch.Tensor = (normal - 0.5) * 2
    height: torch.Tensor = (height - 0.5) * 2
    roughness: torch.Tensor = (roughness - 0.5) * 2

    image_ref: torch.Tensor = (image_ref - 0.5) * 2
    
    # image_latents = image2latents(pipeline, image)
    # masked_ref_latents = image2latents(pipeline, image_ref * mask_ref)
    
    encoder_hidden_states = pipeline.prepare_prompt_embeddings([''] * bsz)

    t = torch.randint(0, 1000, (bsz,), device=device)

    predict_latents = torch.concat([
        color_latents, normal_latents, height_latents, roughness_latents
    ], dim=1)

    # predict_latents = torch.concat([
    #     color_latents, normal_latents
    # ], dim=1)

    random_noise = torch.randn_like(predict_latents)
    noised_latents = pipeline.scheduler.add_noise(predict_latents, random_noise, t)

    # 16 channels
    latents_input = noised_latents
    # 20 channels
    # latents_input = torch.concat([noised_latents, masked_ref_latents], dim=1)
    latents_input_ref = masked_ref_latents

    noise_pred = unet(
        latents_input,
        latents_input_ref,
        t,
        encoder_hidden_states,
        encoder_attention_mask=attn_mask,
        # encoder_attention_mask=None,
        return_dict=False, 
    )[0]

    denoising_loss = pbr_denoising_mse_per_sample(
        noise_pred,
        random_noise,
    ).mean()

    def predict_with_null_condition(eligible_mask: torch.Tensor) -> torch.Tensor:
        # This optional rank-local no-grad forward bypasses the DDP wrapper. Ranks can
        # have different eligible counts without introducing mismatched DDP forwards.
        null_unet = unet.module if hasattr(unet, 'module') else unet
        return null_unet(
            latents_input[eligible_mask],
            torch.zeros_like(latents_input_ref[eligible_mask]),
            t[eligible_mask],
            encoder_hidden_states[eligible_mask],
            encoder_attention_mask=torch.ones_like(attn_mask[eligible_mask]),
            return_dict=False,
        )[0]

    condition_null_loss, condition_null_metrics = (
        compute_condition_null_contrastive_loss(
            noise_pred,
            random_noise,
            ~condition_dropped,
            predict_with_null_condition,
            weight=condition_null_contrastive_weight,
            relative_margin=condition_null_contrastive_margin,
        )
    )
    loss = denoising_loss + condition_null_loss

    loss_dict = {
        'loss': loss.item(),
        'condition_dropout_rate': condition_dropped.float().mean().item(),
    }
    if condition_null_contrastive_weight > 0:
        loss_dict.update({
            name: value.item()
            for name, value in condition_null_metrics.items()
        })

    return_examples = int(min(bsz, 8))

    log_parameters = {
        'color': ((color + 1) / 2)[:return_examples],
        'normal': ((normal + 1) / 2)[:return_examples],
        'height': ((height + 1) / 2)[:return_examples],
        'roughness': ((roughness + 1) / 2)[:return_examples],
        'pipeline': pipeline,
        # 'feature_encoder': feature_encoder,
        'masked_ref_latents': masked_ref_latents[:return_examples],
        # 'encoder_attention_mask': attn_mask[:return_examples],
        'encoder_attention_mask': None,
        'image_ref': ((image_ref + 1) / 2)[:return_examples],
        'mask_ref': mask_ref[:return_examples],
    }

    return loss, loss_dict, log_parameters


def transform(x: torch.Tensor, res=256):
    in_dim = x.ndim
    if in_dim == 3:
        x = x.unsqueeze(0)
    
    h, w = x.shape[-2:]
    if h < res or w < res:
        raise
    x = torchvision.transforms.RandomCrop(res)(x)

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


def prepare_model(cfg, accelerator):
    train_config = cfg['train_config']
    test_config = cfg['test_config']
    data_config = cfg['data_config']
    _validate_parquet_training_config(data_config, train_config)

    batch_size = train_config['batch_size']
    eval_batch_size = test_config['batch_size']
    test_batch_size = test_config['batch_size']
    lr = train_config['lr']
    ckpt_dir = train_config.get('ckpt_dir', None)
    device = train_config['device']
    condition_path_update_config = parse_condition_path_update_config(
        train_config.get('condition_path_update', None),
        train_config.get('max_steps', None),
    )

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
    pipeline.render_augment = build_render_augment(cfg.get('augment_config', DEFAULT_RENDER_AUGMENT_CONFIG))
    # unet: UNet2DConditionModel = UNet2DConditionModel.from_config(
    #     UNet2DConditionModel.load_config('./pretrained/stable-diffusion-v1-4/unet/config_render.json')
    # )
    unet_config = cfg['unet_config']
    unet = SiameseUnet.from_config(
        unet_config['root'],
        config_main=unet_config['config_main'],
        config_ref=unet_config['config_ref'],
    )
    try_load(unet.unet_main, pipeline.unet.state_dict())
    pipeline.unet = unet
    # frozen parameters
    pipeline.frozen()
    unet_main_params = unet.unfrozen_main()
    unet_ref_params = unet.unfrozen_ref()

    optimizer, lr_gains = build_condition_path_optimizer(
        unet,
        unet_main_params,
        unet_ref_params,
        lr=lr,
        config=condition_path_update_config,
    )
    train_config['lr_gain'] = lr_gains

    for i, param_group in enumerate(optimizer.param_groups):
        for p in param_group['params']:
            assert p.requires_grad is True, f'{i} th param has param don\'t require grad'

    start_epoch = 0
    if ckpt_dir:
        print('loading weights')
        # try_load(unet, torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        unet.load_state_dict(torch.load(os.path.join(ckpt_dir, 'unet.ckpt'), weights_only=False, map_location='cpu'))
        if os.path.exists(os.path.join(ckpt_dir, 'optimizer.ckpt')):
            optimizer_state = torch.load(
                os.path.join(ckpt_dir, 'optimizer.ckpt'),
                weights_only=False,
                map_location='cpu',
            )
            saved_group_count = len(optimizer_state.get('param_groups', ()))
            current_group_count = len(optimizer.param_groups)
            if saved_group_count != current_group_count:
                raise ValueError(
                    'optimizer checkpoint parameter-group count is incompatible: '
                    f'saved={saved_group_count}, current={current_group_count}. '
                    'Condition-path update training requires a fresh optimizer '
                    'or a checkpoint created with the same grouping.'
                )
            try:
                optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                raise ValueError(
                    'optimizer checkpoint is incompatible with the current '
                    f'condition-path groups (saved={saved_group_count}, '
                    f'current={current_group_count}): {exc}'
                ) from exc
        try:
            start_epoch = int(ckpt_dir.split('_')[-1])
        except ValueError:
            start_epoch = 0

    # move to cuda
    pipeline.to(device)
    # feature_encoder.to(device)

    # feature_encoder.eval()
    unet.eval()
    # data_split = OmegaConf.load('./datasets/render_base_10_resized/seperate.yaml')

    memory_halt = MemoryHalt()
    # memory_halt.halt(int(45 - torch.cuda.memory_allocated() / 1024 ** 3 - 2))

    data_loader, test_loader = _build_data_loaders(
        data_config,
        train_config,
        test_config,
        accelerator,
    )

    # data_loader_poly = PBRTextureDataLoader(
    #     fp='./datasets/render_polyhaven_train_50_resized', 
    #     gt_fp='./datasets/MatSynth/polyhaven_ground_truth_resized', 
    #     transforms={'default': transform, 'no_transform': lambda x: x},
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
    #     selected_files=None,
    #     good_examples=good_examples,
    #     transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    # )

    # data_loader = PBRTextureDataLoader(
    #     fp='./datasets/render_base_10_resized', 
    #     gt_fp='./datasets/MatSynth/textures_all_resized', 
    #     transforms={'default': transform, 'no_transform': lambda x: x},
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
    #     selected_files=None,
    #     good_examples=good_examples,
    #     transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    # )
    
    # data_loader.files += data_loader_poly.files

    # test_loader = PBRTextureDataLoader(
    #     fp='./datasets/render_base_10_resized', 
    #     gt_fp='./datasets/MatSynth/textures_all_resized', 
    #     transforms={'default': transform, 'no_transform': lambda x: x},
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
    #     selected_files=None,
    #     good_examples=good_examples,
    #     transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    # )
    # test_loader.files += data_loader_poly.files
    # PBRTextureDataLoader.image_names = ('image', 'mask', 'meta_data')
    # PBRTextureDataLoader.file_type = ('Color', 'NormalGL', 'Roughness', 'Height')
    # data_loader = PBRTextureDataLoader(
    #     fp='./datasets/render_polyhaven_train_50_resized', 
    #     gt_fp='./datasets/MatSynth/polyhaven_train',
    #     fetch_attr=('Color', 'NormalGL', 'Roughness', 'Height'),
    #     gt_mapping={
    #         'diff_4k': 'Color',
    #         'nor_gl_4k': 'NormalGL',
    #         'rough_4k': 'Roughness',
    #         'disp_4k': 'Height'
    #     }
    # )

    # test_loader = PBRTextureDataLoader(
    #     fp='./datasets/render_polyhaven_train_50_resized', 
    #     gt_fp='./datasets/MatSynth/polyhaven_train',
    #     fetch_attr=('Color', 'NormalGL', 'Roughness', 'Height'),
    #     gt_mapping={
    #         'diff_4k': 'Color',
    #         'nor_gl_4k': 'NormalGL',
    #         'rough_4k': 'Roughness',
    #         'disp_4k': 'Height'
    #     }
    # )

    # data_loader = PBRTextureDataLoader(
    #     # fp='./datasets/render_result_matsynth_resized_noscale', 
    #     fp='./datasets/render_result_matsynth_10_resized', 
    #     gt_fp='./datasets/MatSynth/textures_all_resized', 
    #     transforms={'default': transform, 'no_transform': lambda x: x},
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
    #     selected_files=data_split['train'],
    #     good_examples=good_examples,
    #     transform_group={'default': ['Color', 'NormalGL', 'Height', 'Roughness'], 'no_transform': ['image', 'mask']}
    # )

    # test_loader = PBRTextureDataLoader(
    #     # fp='./datasets/render_result_matsynth_resized_noscale', 
    #     fp='./datasets/render_result_matsynth_10_resized', 
    #     gt_fp='./datasets/MatSynth/textures_all_resized', 
    #     transforms={'default': transform_eval, 'no_transform': lambda x: x},
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness'),
    #     selected_files=data_split['test'],
    #     good_examples=good_examples,
    #     transform_group={'default': [], 'no_transform': ['image', 'mask', 'Color', 'NormalGL', 'Height', 'Roughness']}
    # )

    # test_loader = PBRTextureDataLoader(
    #     fp='./datasets/render_result_matsynth_eval_10_resized', 
    #     gt_fp='./datasets/MatSynth/textures_all_resized', 
    #     transform=transform,
    #     fetch_attr=('Color', 'NormalGL', 'Height', 'Roughness')
    # )
    # val_loader = TestPolyLoader(
    #     './datasets/polyhaven_edit'
    # )

    # data_loader = FolderLoader(fp='./datasets/MatSynth/color', transform=transform)
    memory_halt.release()

    # val_loader = torch.utils.data.DataLoader(val_loader, batch_size=test_batch_size, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader()
    val_loader = None

    module = {
        'unet': unet,
        # 'feature_encoder': feature_encoder
    }

    torch.cuda.empty_cache()
    return module, data_loader, test_loader, val_loader, optimizer, start_epoch, pipeline


def build(cfg, accelerator):
    train_config = cfg['train_config']
    _validate_parquet_training_config(cfg['data_config'], train_config)
    condition_path_update_config = parse_condition_path_update_config(
        train_config.get('condition_path_update', None),
        train_config.get('max_steps', None),
    )
    condition_null_contrastive_weight, condition_null_contrastive_margin = (
        validate_condition_null_contrastive_config(
            train_config.get('condition_null_contrastive_weight', 0.0),
            train_config.get('condition_null_contrastive_margin', 0.05),
        )
    )
    device = train_config['device']

    modules, train_loader, eval_loader, test_loader, optimizer, start_epoch, pipeline = \
        prepare_model(cfg, accelerator)


    return TrainSpec(
        modules=modules,
        training_step=training_step,
        train_loader=train_loader,
        eval_loader=None,
        test_loader=eval_loader,
        optimizer=optimizer,
        before_optimizer_step=(
            ConditionPathUpdatePolicy(condition_path_update_config)
            if condition_path_update_config.enabled
            else None
        ),
        log_train=log_function,
        log_test=val_function,
        train_kwargs={
            'pipeline': pipeline,
            'device': device,
            'augment': pipeline.render_augment,
            'condition_dropout_prob': train_config.get('condition_dropout_prob', 0.0),
            'condition_null_contrastive_weight': condition_null_contrastive_weight,
            'condition_null_contrastive_margin': condition_null_contrastive_margin,
        },
        start_epoch=start_epoch,
    )
