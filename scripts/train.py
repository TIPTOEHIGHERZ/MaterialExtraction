from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Callable
import weakref

import accelerate
from accelerate.utils import broadcast_object_list, gather_object
from omegaconf import DictConfig, OmegaConf
import torch

sys.path.append(os.getcwd())

from scripts._training.albedo_normal import (
    build as build_albedo_normal,
)
from scripts._training.full_attr import build as build_full_attr
from utils.trainer import (
    TrainResult,
    TrainSpec,
    create_accelerator,
    run_config,
)

_TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"
_CHECKPOINT_FILES = ("unet.ckpt", "optimizer.ckpt", "run.json")
_REGISTRY_NAMES = ("_models", "_optimizers", "_schedulers", "_dataloaders")


@dataclass(frozen=True)
class ProcessSnapshot:
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    cuda_peak_allocated_mb: float
    rss_mb: float
    child_pids: tuple[int, ...]
    registries: dict[str, int]
    accelerator_step: int


class StageObjectTracker:
    """Keep weak references only, so diagnostics cannot keep a stage alive."""

    def __init__(self) -> None:
        self._refs: dict[str, weakref.ReferenceType] = {}

    def capture(self, spec: TrainSpec) -> None:
        objects = {
            "spec": spec,
            "optimizer": spec.optimizer,
            "train_loader": spec.train_loader,
            "train_dataset": getattr(spec.train_loader, "dataset", None),
            "test_loader": spec.test_loader,
            "test_dataset": getattr(spec.test_loader, "dataset", None),
            "pipeline": (spec.train_kwargs or {}).get("pipeline"),
            "augment": (spec.train_kwargs or {}).get("augment"),
        }
        objects.update(
            (f"module:{name}", module) for name, module in spec.modules.items()
        )
        projector = getattr(objects["pipeline"], "condition_projector", None)
        objects["condition_projector"] = projector
        estimator = getattr(projector, "depth_estimator", None)
        objects["depth_model"] = getattr(estimator, "model", None)
        for name, value in objects.items():
            if value is not None:
                self._refs[name] = weakref.ref(value)

    def live_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, ref in self._refs.items() if ref() is not None)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("two-stage render-pretrained training")
    parser.add_argument("--config_file", type=str, required=True)
    return parser.parse_args()


def _load_stage_config(stage_config: DictConfig) -> DictConfig:
    config_file = Path(stage_config["config_file"])
    if not config_file.is_file():
        raise FileNotFoundError(f"stage config does not exist: {config_file}")
    base = OmegaConf.load(config_file)
    return OmegaConf.merge(base, stage_config.get("overrides", {}))


def _accelerator_contract(cfg: DictConfig) -> dict[str, object]:
    train_config = cfg["train_config"]
    precision = train_config.get(
        "precision",
        train_config.get("percision", None),
    )
    return {
        "precision": precision,
        "accumulation_steps": int(train_config["accumulation_steps"]),
        "even_batches": bool(train_config.get("even_batches", True)),
        "device": str(train_config.get("device", "cuda")),
    }


def _validate_stage_configs(
    albedo_normal_cfg: DictConfig,
    full_attr_cfg: DictConfig,
) -> None:
    first_contract = _accelerator_contract(albedo_normal_cfg)
    second_contract = _accelerator_contract(full_attr_cfg)
    if first_contract != second_contract:
        raise ValueError(
            "two stages must share one Accelerator contract: "
            f"albedo_normal={first_contract}, full_attr={second_contract}"
        )

    if first_contract["precision"] == "fp16":
        raise ValueError(
            "two-stage fp16 is unsupported because Accelerator.free_memory() "
            "does not reset the GradScaler; use bf16 or no mixed precision"
        )

    for stage_name, cfg in (
        ("albedo_normal", albedo_normal_cfg),
        ("full_attr", full_attr_cfg),
    ):
        train_config = cfg["train_config"]
        max_steps = int(train_config.get("max_steps", 0))
        save_period = int(train_config.get("save_period", 0))
        if max_steps <= 0 or save_period <= 0:
            raise ValueError(
                f"{stage_name} requires positive max_steps and save_period"
            )
        if max_steps % save_period != 0:
            raise ValueError(
                f"{stage_name}.max_steps must be divisible by save_period so "
                "the terminal checkpoint is available to the next stage"
            )

    first_log_dir = str(albedo_normal_cfg["log_config"].get("log_dir", ""))
    second_log_dir = str(full_attr_cfg["log_config"].get("log_dir", ""))
    if not first_log_dir or not second_log_dir:
        raise ValueError("both stages require a non-empty log_config.log_dir")
    first_checkpoint_root = Path("./checkpoints", first_log_dir).resolve()
    second_checkpoint_root = Path("./checkpoints", second_log_dir).resolve()
    if first_checkpoint_root == second_checkpoint_root:
        raise ValueError("two stages must use different log_config.log_dir values")

    second_train = full_attr_cfg["train_config"]
    if second_train.get("ckpt_dir"):
        raise ValueError("full_attr.ckpt_dir must be null in a fresh two-stage run")
    if second_train.get("resume_training", False):
        raise ValueError("full_attr.resume_training must be false in a two-stage run")


def _validate_runtime_precision(accelerator: accelerate.Accelerator) -> None:
    if accelerator.mixed_precision == "fp16":
        raise ValueError(
            "two-stage fp16 is unsupported because Accelerator.free_memory() "
            "does not reset the GradScaler; use bf16 or no mixed precision"
        )


def _shared_timestamp(accelerator: accelerate.Accelerator) -> str:
    timestamp = (
        datetime.now().strftime(_TIMESTAMP_FORMAT)
        if accelerator.is_main_process
        else None
    )
    values = [timestamp]
    broadcast_object_list(values)
    if not isinstance(values[0], str):
        raise RuntimeError("could not broadcast the two-stage run timestamp")
    return values[0]


def _set_run_log_dir(cfg: DictConfig, timestamp: str) -> None:
    base_log_dir = str(cfg["log_config"]["log_dir"])
    cfg["log_config"]["log_dir"] = f"{base_log_dir}_{timestamp}"


def _child_pids() -> tuple[int, ...]:
    children_path = Path(f"/proc/self/task/{os.getpid()}/children")
    try:
        contents = children_path.read_text(encoding="ascii").strip()
    except OSError:
        return ()
    if not contents:
        return ()
    return tuple(sorted(int(pid) for pid in contents.split()))


def _rss_mb() -> float:
    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
    except (OSError, IndexError, ValueError):
        return 0.0
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024**2)


def _registry_counts(accelerator: accelerate.Accelerator) -> dict[str, int]:
    return {
        name.removeprefix("_"): len(getattr(accelerator, name, ()))
        for name in _REGISTRY_NAMES
    }


def _snapshot(accelerator: accelerate.Accelerator) -> ProcessSnapshot:
    device = accelerator.device
    if torch.cuda.is_available() and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        allocated = reserved = peak = 0.0
    return ProcessSnapshot(
        cuda_allocated_mb=allocated,
        cuda_reserved_mb=reserved,
        cuda_peak_allocated_mb=peak,
        rss_mb=_rss_mb(),
        child_pids=_child_pids(),
        registries=_registry_counts(accelerator),
        accelerator_step=int(accelerator.step),
    )


def _collect_stage_memory(accelerator: accelerate.Accelerator) -> None:
    device = accelerator.device
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
    accelerator.free_memory()
    # Accelerate clears its registries after its own GC/cache pass. Run a
    # second pass so the objects released by those registries are collectible.
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _wait_for_worker_exit(
    baseline_children: tuple[int, ...],
    timeout_seconds: float,
) -> tuple[int, ...]:
    baseline = set(baseline_children)
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = tuple(sorted(set(_child_pids()) - baseline))
        if not remaining or time.monotonic() >= deadline:
            return remaining
        gc.collect()
        time.sleep(0.1)


def _checkpoint_errors(result: TrainResult) -> list[str]:
    if result.checkpoint_dir is None:
        return ["stage did not save a checkpoint"]
    checkpoint_dir = Path(result.checkpoint_dir)
    expected_name = f"train_{result.step}"
    errors = []
    if checkpoint_dir.name != expected_name:
        errors.append(
            f"last checkpoint {checkpoint_dir.name!r} is not terminal {expected_name!r}"
        )
    for filename in _CHECKPOINT_FILES:
        if not (checkpoint_dir / filename).is_file():
            errors.append(f"missing checkpoint file: {checkpoint_dir / filename}")
    return errors


def _release_stage(
    *,
    stage_name: str,
    accelerator: accelerate.Accelerator,
    tracker: StageObjectTracker,
    baseline: ProcessSnapshot,
    max_allocated_delta_mb: float,
    worker_exit_timeout_seconds: float,
    checkpoint_errors: list[str],
    enforce: bool,
) -> dict:
    before = _snapshot(accelerator)
    _collect_stage_memory(accelerator)
    remaining_children = _wait_for_worker_exit(
        baseline.child_pids,
        worker_exit_timeout_seconds,
    )
    after = _snapshot(accelerator)

    errors = list(checkpoint_errors)
    live_objects = tracker.live_names()
    if live_objects:
        errors.append(f"live stage objects: {list(live_objects)}")
    nonempty_registries = {
        name: count for name, count in after.registries.items() if count != 0
    }
    if nonempty_registries:
        errors.append(f"non-empty Accelerator registries: {nonempty_registries}")
    if after.accelerator_step != 0:
        errors.append(f"Accelerator.step was not reset: {after.accelerator_step}")
    allocated_delta = after.cuda_allocated_mb - baseline.cuda_allocated_mb
    if allocated_delta > max_allocated_delta_mb:
        errors.append(
            "CUDA allocated memory stayed above baseline: "
            f"delta={allocated_delta:.2f} MiB, limit={max_allocated_delta_mb:.2f} MiB"
        )
    # Host RSS is diagnostic only; allocator caches need not return to baseline.
    rss_delta = after.rss_mb - baseline.rss_mb
    if remaining_children:
        errors.append(f"stage worker processes still alive: {list(remaining_children)}")

    return {
        "stage": stage_name,
        "rank": int(accelerator.process_index),
        "enforced": enforce,
        "passed": not errors if enforce else None,
        "errors": errors,
        "live_objects": list(live_objects),
        "cuda_allocated_delta_mb": allocated_delta,
        "rss_delta_mb": rss_delta,
        "before": asdict(before),
        "after": asdict(after),
    }


def _wrapped_build(
    build_fn: Callable[[DictConfig, accelerate.Accelerator], TrainSpec],
    tracker: StageObjectTracker | None,
    disable_logging_callbacks: bool,
):
    def build(cfg: DictConfig, accelerator: accelerate.Accelerator) -> TrainSpec:
        spec = build_fn(cfg, accelerator)
        if tracker is not None:
            tracker.capture(spec)
        if disable_logging_callbacks:
            spec.log_train = None
            spec.log_test = None
        return spec

    return build


def _write_cleanup_report(result: TrainResult, reports: list[dict]) -> None:
    if result.checkpoint_dir is None:
        return
    report_path = Path(result.checkpoint_dir) / "stage_cleanup.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=True)


def _run_stage(
    *,
    stage_name: str,
    build_fn: Callable[[DictConfig, accelerate.Accelerator], TrainSpec],
    cfg: DictConfig,
    accelerator: accelerate.Accelerator,
    baseline: ProcessSnapshot | None,
    runtime_config: DictConfig,
) -> tuple[TrainResult, list[dict]]:
    verify_cleanup = bool(runtime_config.get("verify_cleanup", False))
    tracker = StageObjectTracker() if verify_cleanup else None
    disable_callbacks = bool(runtime_config.get("disable_logging_callbacks", False))

    device = accelerator.device
    if verify_cleanup:
        tolerance = float(runtime_config.get("max_cuda_allocated_delta_mb", 256.0))
        worker_timeout = float(runtime_config.get("worker_exit_timeout_seconds", 10.0))
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    result = run_config(
        _wrapped_build(build_fn, tracker, disable_callbacks),
        cfg,
        accelerator,
    )

    accelerator.wait_for_everyone()
    checkpoint_errors = _checkpoint_errors(result)

    if not verify_cleanup:
        _collect_stage_memory(accelerator)
        reports = gather_object(
            [
                {
                    "stage": stage_name,
                    "rank": int(accelerator.process_index),
                    "errors": checkpoint_errors,
                }
            ]
        )
        failures = [item for item in reports if item["errors"]]
        if failures:
            raise RuntimeError(
                f"{stage_name} checkpoint validation failed: "
                + json.dumps(failures, ensure_ascii=True)
            )
        accelerator.wait_for_everyone()
        return result, []

    if tracker is None or baseline is None:
        raise RuntimeError("cleanup verification requires a tracker and baseline")
    report = _release_stage(
        stage_name=stage_name,
        accelerator=accelerator,
        tracker=tracker,
        baseline=baseline,
        max_allocated_delta_mb=tolerance,
        worker_exit_timeout_seconds=worker_timeout,
        checkpoint_errors=checkpoint_errors,
        enforce=True,
    )
    reports = gather_object([report])
    if accelerator.is_main_process:
        print(
            f"[two-stage cleanup] {json.dumps(reports, ensure_ascii=True)}",
            flush=True,
        )
    write_status = [None]
    if accelerator.is_main_process:
        try:
            _write_cleanup_report(result, reports)
        except OSError as error:
            write_status[0] = f"could not write cleanup report: {error}"
    broadcast_object_list(write_status)
    if write_status[0] is not None:
        raise RuntimeError(write_status[0])
    failures = [item for item in reports if not item["passed"]]
    if failures:
        raise RuntimeError(
            f"{stage_name} cleanup gate failed: "
            + json.dumps(failures, ensure_ascii=True)
        )
    accelerator.wait_for_everyone()
    return result, reports


def main() -> None:
    args = _parse_args()
    workflow_cfg = OmegaConf.load(args.config_file)
    stages = workflow_cfg["stages"]
    albedo_normal_cfg = _load_stage_config(stages["albedo_normal"])
    full_attr_cfg = _load_stage_config(stages["full_attr"])
    _validate_stage_configs(albedo_normal_cfg, full_attr_cfg)

    set_nccl_env = os.environ.get("TEX_EDIT_SET_NCCL_ENV", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    accelerator = create_accelerator(
        albedo_normal_cfg,
        set_nccl_env=set_nccl_env,
    )
    try:
        _validate_runtime_precision(accelerator)
        timestamp = _shared_timestamp(accelerator)
        _set_run_log_dir(albedo_normal_cfg, timestamp)
        _set_run_log_dir(full_attr_cfg, timestamp)

        runtime_config = workflow_cfg.get("runtime", OmegaConf.create({}))
        verify_cleanup = bool(runtime_config.get("verify_cleanup", False))
        _collect_stage_memory(accelerator)
        baseline = _snapshot(accelerator) if verify_cleanup else None

        first, _first_reports = _run_stage(
            stage_name="albedo_normal",
            build_fn=build_albedo_normal,
            cfg=albedo_normal_cfg,
            accelerator=accelerator,
            baseline=baseline,
            runtime_config=runtime_config,
        )
        full_attr_cfg["train_config"]["init_ckpt_dir"] = first.checkpoint_dir

        second, _second_reports = _run_stage(
            stage_name="full_attr",
            build_fn=build_full_attr,
            cfg=full_attr_cfg,
            accelerator=accelerator,
            baseline=baseline,
            runtime_config=runtime_config,
        )
        if accelerator.is_main_process:
            print(
                "[two-stage] completed: "
                f"albedo_normal={first.checkpoint_dir}, "
                f"full_attr={second.checkpoint_dir}",
                flush=True,
            )
    finally:
        # This is the only process-group teardown in the two-stage workflow.
        accelerator.end_training()


if __name__ == "__main__":
    main()
