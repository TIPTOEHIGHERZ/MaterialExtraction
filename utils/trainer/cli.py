"""CLI helpers that absorb the boilerplate `main()` shared by every finetune
script: argparse, YAML loading, lr scaling, NCCL env vars, accelerator setup,
and the final `Trainer(...).fit()` call.

A migrated script becomes::

    from utils.trainer import run_from_config, TrainSpec

    def build(cfg, accelerator):
        ...
        return TrainSpec(...)

    if __name__ == '__main__':
        run_from_config(build)
"""
import argparse
import os
from typing import Callable, Optional

import accelerate
from omegaconf import DictConfig, OmegaConf

from .core import TrainResult, Trainer, _build_dataloader_config, _resolve_precision
from .spec import TrainSpec


_NCCL_DEFAULTS = {
    'NCCL_DEBUG': 'INFO',
    'NCCL_P2P_DISABLE': '0',
    'NCCL_IB_DISABLE': '0',
    'TORCH_DISTRIBUTED_DEBUG': 'INFO',
}


def _n_visible_gpus() -> int:
    cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not cuda:
        return 1
    return len([x for x in cuda.split(',') if x.strip() != ''])


def scale_lr(
    base_lr: float,
    batch_size: int,
    accumulation_steps: int,
    n_gpu: int,
    max_lr: float = 5e-5,
) -> tuple[float, float]:
    """Compute (scaled_lr, base_lr).

    The convention across all finetune scripts is:
        effective_lr = min(max_lr, base_lr * n_gpu * batch_size * accumulation_steps)
    """
    scaled = base_lr * n_gpu * batch_size * accumulation_steps
    return min(scaled, max_lr), base_lr


def _build_argparser(extra_args: Optional[Callable[[argparse.ArgumentParser], None]]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--resume_training', action='store_true', default=None)
    if extra_args is not None:
        extra_args(parser)
    return parser


def _resolve_resume_training(cli_resume_training: Optional[bool], train_config: dict) -> bool:
    """CLI flag wins when present; otherwise fall back to the config default."""
    if cli_resume_training is not None:
        return bool(cli_resume_training)
    return bool(train_config.get('resume_training', False))


def _apply_nccl_defaults():
    for k, v in _NCCL_DEFAULTS.items():
        os.environ.setdefault(k, v)


def _scale_section(
    section: dict,
    base_lr: float,
    accumulation_steps: int,
    n_gpu: int,
    max_lr: float,
    label: str,
    is_main: bool,
):
    """Apply the standard lr scaling + clamp to one config section in place."""
    bsz = section['batch_size']
    scaled, base = scale_lr(base_lr, bsz, accumulation_steps, n_gpu, max_lr)
    section['base_lr'] = base
    section['lr'] = scaled
    section.setdefault('lr_gain', [1.0])
    if is_main:
        formula = (
            f'{label} lr: batch_size({bsz}) * n_gpu({n_gpu}) * '
            f'accumulation_steps({accumulation_steps}) * base_lr({base}) '
            f'-> min({base * bsz * n_gpu * accumulation_steps}, {max_lr}) = {scaled}'
        )
        print(formula)


def create_accelerator(
    cfg: DictConfig,
    *,
    set_nccl_env: bool = True,
) -> accelerate.Accelerator:
    """Create the process-wide Accelerator used by one or more train configs."""
    train_config = cfg['train_config']
    if set_nccl_env:
        _apply_nccl_defaults()
    return accelerate.Accelerator(
        mixed_precision=_resolve_precision(train_config),
        gradient_accumulation_steps=train_config['accumulation_steps'],
        dataloader_config=_build_dataloader_config(train_config),
    )


def run_config(
    build_fn: Callable[[DictConfig, accelerate.Accelerator], TrainSpec],
    cfg: DictConfig,
    accelerator: accelerate.Accelerator,
    *,
    resume_training: Optional[bool] = None,
    max_lr: float = 5e-5,
) -> TrainResult:
    """Run one loaded config without owning the shared Accelerator lifecycle."""
    run_config_snapshot = OmegaConf.to_container(cfg, resolve=True)

    train_config = cfg['train_config']
    eval_config = cfg.get('eval_config', None)
    accumulation_steps = train_config['accumulation_steps']
    base_lr = train_config['lr']
    n_gpu = int(accelerator.num_processes)
    if n_gpu < 1:
        raise ValueError('accelerator.num_processes must be positive')
    is_main = accelerator.is_main_process

    _scale_section(train_config, base_lr, accumulation_steps, n_gpu, max_lr, 'train', is_main)
    if eval_config is not None:
        _scale_section(eval_config, base_lr, accumulation_steps, n_gpu, max_lr, 'eval', is_main)

    train_config['resume_training'] = _resolve_resume_training(
        resume_training,
        train_config,
    )
    snapshot_train_config = run_config_snapshot.get('train_config')
    if isinstance(snapshot_train_config, dict):
        snapshot_train_config['resume_training'] = train_config['resume_training']

    spec = build_fn(cfg, accelerator)
    if spec.run_config is None:
        spec.run_config = run_config_snapshot

    # build_fn may intentionally remove or mutate optional config sections.
    train_config = cfg['train_config']
    eval_config = cfg.get('eval_config', None)
    test_config = cfg.get('test_config', None)
    log_config = cfg.get('log_config', {})
    trainer = Trainer(
        spec=spec,
        train_config=train_config,
        eval_config=eval_config,
        test_config=test_config,
        log_config=log_config,
        accelerator=accelerator,
        run_config=spec.run_config,
    )
    return trainer.fit()


def run_from_config(
    build_fn: Callable[[DictConfig, accelerate.Accelerator], TrainSpec],
    *,
    extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    max_lr: float = 5e-5,
    set_nccl_env: bool = True,
) -> None:
    """Standard finetune entry point.

    `build_fn` constructs a TrainSpec given the loaded OmegaConf and the live
    Accelerator. Trainer takes over from there.
    """
    args = _build_argparser(extra_args).parse_args()
    cfg = OmegaConf.load(args.config_file)

    accelerator = create_accelerator(cfg, set_nccl_env=set_nccl_env)
    run_config(
        build_fn,
        cfg,
        accelerator,
        resume_training=args.resume_training,
        max_lr=max_lr,
    )
