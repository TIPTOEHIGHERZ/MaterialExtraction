import os
import re
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import accelerate
import omegaconf
import torch
import tqdm
from accelerate.utils import DataLoaderConfiguration, broadcast_object_list

from .constants import LOCAL_BATCH_SIZE_KEY
from .checkpoint import load_run_metadata, rotate_checkpoints, save_modules
from .logging import LogWrapper
from .params import set_group_lr
from .spec import TrainSpec


def _resolve_precision(train_config: dict) -> Optional[str]:
    """Accepts both `precision` (new) and `percision` (legacy typo) keys."""
    if 'precision' in train_config:
        return train_config['precision']
    if 'percision' in train_config:
        warnings.warn(
            "config key 'percision' is deprecated; rename to 'precision'",
            DeprecationWarning,
            stacklevel=3,
        )
        return train_config['percision']
    return None


def _build_dataloader_config(train_config: dict) -> DataLoaderConfiguration:
    even_batches = train_config.get('even_batches', True)
    if not isinstance(even_batches, bool):
        raise ValueError('train_config.even_batches must be a boolean')
    return DataLoaderConfiguration(even_batches=even_batches)


def _compile_prepared_modules(
    modules: Mapping[str, torch.nn.Module],
    train_config: Mapping,
) -> None:
    """Compile selected prepared modules in place after DDP wrapping."""

    config = train_config.get('torch_compile')
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ValueError('train_config.torch_compile must be a mapping')

    enabled = config.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError(
            'train_config.torch_compile.enabled must be a boolean'
        )
    if not enabled:
        return

    module_names = config.get('modules')
    if isinstance(module_names, (str, bytes)) or not isinstance(
        module_names,
        Sequence,
    ):
        raise ValueError(
            'train_config.torch_compile.modules must be a sequence'
        )
    module_names = list(module_names)
    if not module_names:
        raise ValueError(
            'train_config.torch_compile.modules must be non-empty'
        )
    if not all(isinstance(name, str) for name in module_names):
        raise ValueError('torch_compile module names must be strings')
    if any(not name.strip() for name in module_names):
        raise ValueError('torch_compile module names must be non-empty strings')

    duplicate_names = sorted(
        name for name in set(module_names) if module_names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError(
            'train_config.torch_compile.modules contains duplicate names: '
            + ', '.join(duplicate_names)
        )
    unknown_names = sorted(set(module_names) - set(modules))
    if unknown_names:
        raise ValueError(
            'train_config.torch_compile contains unknown modules: '
            + ', '.join(unknown_names)
        )

    backend = config.get('backend', 'inductor')
    mode = config.get('mode', 'default')
    fullgraph = config.get('fullgraph', False)
    dynamic = config.get('dynamic', None)
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError('train_config.torch_compile.backend must be non-empty')
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError('train_config.torch_compile.mode must be non-empty')
    if not isinstance(fullgraph, bool):
        raise ValueError('train_config.torch_compile.fullgraph must be a boolean')
    if dynamic is not None and not isinstance(dynamic, bool):
        raise ValueError(
            'train_config.torch_compile.dynamic must be a boolean or null'
        )

    compile_kwargs = {
        'backend': backend,
        'mode': mode,
        'fullgraph': fullgraph,
        'dynamic': dynamic,
    }
    compile_functions = {}
    for name in module_names:
        compile_fn = getattr(modules[name], 'compile', None)
        if not callable(compile_fn):
            raise RuntimeError(f'module {name!r} does not support compile()')
        compile_functions[name] = compile_fn

    for name in module_names:
        compile_functions[name](**compile_kwargs)


def _set_train_pass(train_loader, pass_idx: int) -> None:
    """Set the single shuffle authority without rebuilding the raw loader."""

    batch_sampler = getattr(train_loader, 'batch_sampler', None)
    sampler_set_epoch = getattr(batch_sampler, 'set_epoch', None)
    if callable(sampler_set_epoch):
        validate = getattr(batch_sampler, 'validate_dataloader', None)
        if callable(validate):
            validate(train_loader)
        sampler_workers = getattr(batch_sampler, 'num_workers', None)
        if (
            sampler_workers is not None
            and sampler_workers != train_loader.num_workers
        ):
            raise ValueError(
                'batch sampler num_workers does not match DataLoader num_workers'
            )
        sampler_set_epoch(pass_idx)
        return

    dataset = train_loader.dataset
    dataset_set_epoch = getattr(dataset, 'set_epoch', None)
    if callable(dataset_set_epoch):
        dataset_set_epoch(pass_idx)
        return
    dataset_shuffle = getattr(dataset, 'shuffle', None)
    if callable(dataset_shuffle):
        dataset_shuffle()


def _validate_reusable_loader(train_loader) -> None:
    """Reject pass mutations that persistent worker copies cannot observe."""
    if not getattr(train_loader, 'persistent_workers', False):
        return
    batch_sampler = getattr(train_loader, 'batch_sampler', None)
    if callable(getattr(batch_sampler, 'set_epoch', None)):
        return
    dataset = train_loader.dataset
    if callable(getattr(dataset, 'set_epoch', None)) or callable(
        getattr(dataset, 'shuffle', None)
    ):
        raise ValueError(
            'persistent_workers=True is incompatible with dataset-level '
            'set_epoch/shuffle; use a batch sampler with set_epoch or disable '
            'persistent workers'
        )


def _prepare_training_batch(
    data,
    accelerator: accelerate.Accelerator,
    accumulation_steps: int,
):
    """Remove the reserved size field and compute sample-mean DDP scaling."""

    if not isinstance(data, Mapping) or LOCAL_BATCH_SIZE_KEY not in data:
        return data, None
    if accumulation_steps != 1:
        raise ValueError(
            'variable-size PBR Parquet batches require accumulation_steps=1'
        )

    training_data = dict(data)
    local_value = training_data.pop(LOCAL_BATCH_SIZE_KEY)
    local_size = torch.as_tensor(local_value)
    if local_size.numel() != 1:
        raise ValueError(f'{LOCAL_BATCH_SIZE_KEY} must be a scalar')
    device = getattr(accelerator, 'device', None)
    if device is None:
        local_size = local_size.detach().to(dtype=torch.float64)
    else:
        local_size = local_size.detach().to(
            device=device,
            dtype=torch.float64,
        )
    if not torch.isfinite(local_size).item() or local_size.item() <= 0:
        raise ValueError(f'{LOCAL_BATCH_SIZE_KEY} must be positive and finite')

    global_size = accelerator.reduce(local_size, reduction='sum')
    global_size = torch.as_tensor(
        global_size,
        device=local_size.device,
        dtype=torch.float64,
    )
    if global_size.numel() != 1:
        raise ValueError('reduced global batch size must be a scalar')
    if not torch.isfinite(global_size).item() or global_size.item() <= 0:
        raise ValueError('reduced global batch size must be positive and finite')
    num_processes = int(getattr(accelerator, 'num_processes', 1))
    if num_processes < 1:
        raise ValueError('accelerator.num_processes must be positive')
    loss_scale = num_processes * local_size / global_size
    return training_data, loss_scale


def _end_prepared_dataloader(data_loader) -> None:
    """End a partially consumed Accelerate loader on versions using strong refs."""
    if data_loader is None:
        return
    gradient_state = getattr(data_loader, 'gradient_state', None)
    references = getattr(gradient_state, 'dataloader_references', ())
    if not any(reference is data_loader for reference in references):
        return
    end = getattr(data_loader, 'end', None)
    if callable(end):
        end()


_LOG_DIR_TIMESTAMP_FORMAT = '%Y-%m-%d-%H-%M-%S'
_LOG_DIR_TIMESTAMP_RE = re.compile(r'.*_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')


def _looks_timestamped_log_dir(log_dir: Optional[str]) -> bool:
    return bool(log_dir) and _LOG_DIR_TIMESTAMP_RE.fullmatch(log_dir) is not None


def _checkpoint_parent_log_dir(ckpt_dir: Optional[str]) -> Optional[str]:
    if not ckpt_dir:
        return None
    normalized = os.path.normpath(ckpt_dir)
    parent = os.path.dirname(normalized)
    if not parent or parent == normalized:
        return None
    return os.path.basename(parent) or None


def _resolve_log_context(
    log_config: dict,
    train_config: dict,
) -> tuple[dict, Optional[dict]]:
    """Resolve the effective run directory for fresh runs vs resume launches."""
    resolved = dict(log_config)
    requested_log_dir = resolved.get('requested_log_dir') or resolved.get('log_dir')
    if requested_log_dir is not None:
        resolved['requested_log_dir'] = requested_log_dir

    ckpt_dir = train_config.get('ckpt_dir', None)
    resume_meta = load_run_metadata(ckpt_dir)

    # Only true resume launches should keep writing into the historical run
    # directory recorded by the checkpoint metadata. When ckpt_dir is used just
    # for weight initialization, a fresh run still needs its own timestamped
    # output root.
    if train_config.get('resume_training'):
        if resume_meta is not None and resume_meta.get('log_dir'):
            resolved['log_dir'] = resume_meta['log_dir']
            return resolved, resume_meta

        # Legacy resume checkpoints predate run.json; fall back to the
        # checkpoint root instead of opening a fresh run.
        inferred_log_dir = _checkpoint_parent_log_dir(ckpt_dir)
        if inferred_log_dir:
            resolved['log_dir'] = inferred_log_dir
            return resolved, resume_meta

    if requested_log_dir and not _looks_timestamped_log_dir(requested_log_dir):
        launch_suffix = datetime.now().strftime(_LOG_DIR_TIMESTAMP_FORMAT)
        resolved['log_dir'] = f'{requested_log_dir}_{launch_suffix}'
    else:
        resolved['log_dir'] = requested_log_dir

    return resolved, resume_meta


@dataclass(frozen=True)
class TrainResult:
    """Scalar/path-only result that cannot retain a completed training stage."""

    checkpoint_dir: Optional[str]
    step: int
    global_iter: int
    log_dir: Optional[str]


@dataclass
class Trainer:
    spec: TrainSpec
    train_config: dict
    eval_config: Optional[dict] = None
    test_config: Optional[dict] = None
    log_config: dict = field(default_factory=dict)
    accelerator: Optional[accelerate.Accelerator] = None
    run_config: Optional[dict] = None

    def _run_metadata(self, step: int, epoch: int, global_iter: int) -> dict:
        """Assemble the run.json payload persisted alongside the weights."""
        return {
            'step': step,
            'epoch': epoch,
            'global_iter': global_iter,
            'accumulation_steps': self.train_config.get('accumulation_steps'),
            'log_dir': self.log_config.get('log_dir'),
            'requested_log_dir': self.log_config.get('requested_log_dir'),
            'config': self.run_config,
        }

    def _build_accelerator(self) -> accelerate.Accelerator:
        if self.accelerator is not None:
            return self.accelerator
        precision = _resolve_precision(self.train_config)
        accumulation_steps = self.train_config['accumulation_steps']
        return accelerate.Accelerator(
            mixed_precision=precision,
            gradient_accumulation_steps=accumulation_steps,
            dataloader_config=_build_dataloader_config(self.train_config),
        )

    def fit(self) -> TrainResult:
        spec = self.spec
        train_config = self.train_config
        test_config = self.test_config
        log_config, resume_meta = _resolve_log_context(self.log_config, train_config)
        self.log_config = log_config

        summon_ckpt = train_config.get('summon_ckpt', False)
        # Training is bounded by optimizer steps (`max_steps`). `epochs` is kept
        # only as a deprecated fallback: when max_steps is absent it is converted
        # to a step budget at runtime (epochs * steps_per_pass).
        max_steps = train_config.get('max_steps', None)
        epochs = train_config.get('epochs', None)
        if max_steps is None and epochs is None:
            raise ValueError(
                "train_config needs 'max_steps' (preferred) or 'epochs' (deprecated)"
            )
        accumulation_steps = train_config['accumulation_steps']
        save_period = train_config['save_period']
        max_saves = train_config['max_saves']
        precision = _resolve_precision(train_config)

        save_folder = os.path.join('./checkpoints', log_config.get('log_dir'))

        accelerator = self._build_accelerator()
        self.accelerator = accelerator

        # Resume from run.json metadata when present. This overrides the
        # per-script `start_epoch` (parsed from the folder name as `train_{step}`,
        # which conflated step with epoch and broke `range(start_epoch, epochs)`).
        # No run.json -> keep spec.start_epoch, preserving legacy behavior.
        start_epoch = spec.start_epoch
        resume_global_iter = None
        ckpt_dir = train_config.get('ckpt_dir', None)
        if resume_meta is not None:
            if resume_meta.get('epoch') is not None:
                start_epoch = int(resume_meta['epoch'])
            if resume_meta.get('global_iter') is not None:
                resume_global_iter = int(resume_meta['global_iter'])
            if accelerator.is_main_process:
                print(
                    f'[resume] run.json in {ckpt_dir}: '
                    f'start_epoch={start_epoch}, global_iter={resume_global_iter}'
                )

        optimizer = accelerator.prepare_optimizer(spec.optimizer)
        train_loader = spec.train_loader

        modules = {n: accelerator.prepare_model(m) for n, m in spec.modules.items()}
        _compile_prepared_modules(modules, train_config)

        global_iter = resume_global_iter if resume_global_iter is not None else -1
        save_queue: list[int] = []
        train_args = spec.train_args or ()
        train_kwargs = spec.train_kwargs or {}
        test_loader = spec.test_loader

        # `pass_idx` counts full (reshuffled) traversals of the dataset; it only
        # drives logging/reshuffle. Termination is governed by `max_micro`.
        pass_idx = start_epoch
        max_micro = max_steps * accumulation_steps if max_steps is not None else None
        done = False
        last_checkpoint_dir = None
        progress_bar = None
        data_loader = None
        log_func = None

        try:
            log_func = LogWrapper(
                spec.log_train,
                spec.log_test,
                log_config,
                test_config,
                accelerator.is_main_process,
            )
            if accelerator.is_main_process:
                print(f'using training configs: {train_config}')
                time.sleep(3)

            cnt = 0
            optimizer.zero_grad(set_to_none=True)
            _validate_reusable_loader(train_loader)
            data_loader = accelerator.prepare_data_loader(train_loader)

            # Deprecated fallback: resolve the step budget once from the
            # prepared loader. Reusing this wrapper across passes avoids
            # accumulating one Accelerator registry entry per dataset pass.
            if max_micro is None:
                max_micro = int(epochs) * len(data_loader)
                warnings.warn(
                    "train_config.max_steps missing; deriving budget from "
                    f"epochs*steps_per_pass = {epochs}*{len(data_loader)} micro-steps. "
                    "Set max_steps explicitly ('epochs' is deprecated).",
                    DeprecationWarning,
                    stacklevel=2,
                )

            while not done:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                _set_train_pass(train_loader, pass_idx)
                lr_list = [train_config['lr'] * gain for gain in train_config['lr_gain']]

                set_group_lr(optimizer, lr_list)

                if (global_iter + 1) >= max_micro:
                    break

                total_steps = max_micro // accumulation_steps
                progress_bar = tqdm.tqdm(data_loader) if accelerator.is_main_process else data_loader

                for data in progress_bar:
                    training_data, loss_scale = _prepare_training_batch(
                        data,
                        accelerator,
                        accumulation_steps,
                    )
                    if precision is not None:
                        with accelerator.autocast():
                            raw_loss, loss_dict, log_parameters = spec.training_step(
                                modules,
                                training_data,
                                *train_args,
                                **train_kwargs,
                            )
                    else:
                        raw_loss, loss_dict, log_parameters = spec.training_step(
                            modules,
                            training_data,
                            *train_args,
                            **train_kwargs,
                        )

                    backward_loss = raw_loss / accumulation_steps
                    if loss_scale is not None:
                        backward_loss = backward_loss * loss_scale.to(
                            device=raw_loss.device,
                            dtype=raw_loss.dtype,
                        )
                    accelerator.backward(backward_loss)

                    cnt = (cnt + 1) % accumulation_steps
                    if cnt == 0:
                        optimizer_step = (global_iter + 1) // accumulation_steps
                        if spec.before_optimizer_step is not None:
                            callback_metrics = spec.before_optimizer_step(
                                optimizer=optimizer,
                                step=optimizer_step,
                            )
                            if callback_metrics:
                                loss_dict.update(callback_metrics)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    global_iter += 1
                    global_iter_normalized = global_iter // accumulation_steps
                    global_iter_mods = global_iter % accumulation_steps

                    if log_func is not None:
                        with accelerator.autocast():
                            log_func(
                                loss_dict,
                                progress_bar,
                                global_iter_mods,
                                test_dataloader=test_loader,
                                **log_parameters,
                                global_iter=global_iter_normalized,
                                accumulation_steps=accumulation_steps,
                                name_space='evaluate',
                            )

                    if accelerator.is_main_process and isinstance(progress_bar, tqdm.tqdm):
                        progress_bar.set_description(
                            f'step {global_iter_normalized + 1} / {total_steps}'
                        )

                    should_save = (
                        (global_iter_normalized + 1) % save_period == 0
                        and global_iter_mods == 0
                    )
                    if should_save:
                        checkpoint_context = f'train_{global_iter_normalized + 1}'
                        last_checkpoint_dir = os.path.join(
                            save_folder,
                            checkpoint_context,
                        )
                        save_error = [None]
                        if accelerator.is_main_process:
                            try:
                                save_queue.append(global_iter_normalized + 1)
                                rotate_checkpoints(save_queue, max_saves, save_folder)
                                save_modules(
                                    modules,
                                    optimizer,
                                    base_folder=save_folder,
                                    context=checkpoint_context,
                                    metadata=self._run_metadata(
                                        step=global_iter_normalized + 1,
                                        epoch=pass_idx,
                                        global_iter=global_iter,
                                    ),
                                )
                            except Exception as error:
                                save_error[0] = (
                                    f'{type(error).__name__}: {error}'
                                )
                        if accelerator.num_processes > 1:
                            broadcast_object_list(save_error)
                        if save_error[0] is not None:
                            raise RuntimeError(
                                f'checkpoint save failed at {checkpoint_context}: '
                                f'{save_error[0]}'
                            )

                    if (global_iter + 1) >= max_micro:
                        done = True
                        break

                pass_idx += 1

        except KeyboardInterrupt:
            if summon_ckpt and accelerator.is_main_process:
                print('summoning ckpt...')
                global_iter_normalized = global_iter // accumulation_steps
                save_modules(
                    modules,
                    optimizer,
                    base_folder=save_folder,
                    context='last',
                    metadata=self._run_metadata(
                        step=global_iter_normalized + 1,
                        epoch=pass_idx,
                        global_iter=global_iter,
                    ),
                )
                train_legacy = {'global_iter': global_iter}
                omegaconf.OmegaConf.save(
                    train_legacy, os.path.join(save_folder, 'last/train_legacy.yaml'),
                )
            raise
        finally:
            if isinstance(progress_bar, tqdm.tqdm):
                progress_bar.close()
            _end_prepared_dataloader(data_loader)
            if log_func is not None:
                log_func.close()

        completed_steps = max(0, (global_iter + 1) // accumulation_steps)
        return TrainResult(
            checkpoint_dir=last_checkpoint_dir,
            step=completed_steps,
            global_iter=global_iter,
            log_dir=log_config.get('log_dir'),
        )
