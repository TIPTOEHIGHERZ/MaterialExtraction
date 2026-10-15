#!/usr/bin/env python3
"""Benchmark the directory and direct-Parquet PBR training data paths.

The ``run`` command is intentionally launched once with one process and once
with eight processes.  Each invocation alternates the two backends for five
fresh-loader trials and writes a self-describing JSON result.  The ``evaluate``
command combines those two files and is the only command that can emit a GO
decision.

Examples (from the repository root)::

    /root/envs/blender/bin/python scripts/benchmark_pbr_parquet.py run \
      --expected-world-size 1 --render-root datasets/render_result_matsynth_50_resized \
      --gt-root datasets/Matsynth_standard/unpacked_512 \
      --parquet-dir datasets/parquet/render_result_matsynth_50_resized \
      --cold-cache linux-drop-caches --output-json /tmp/pbr-bench-1.json

    /root/envs/blender/bin/torchrun --standalone --nproc-per-node=8 \
      scripts/benchmark_pbr_parquet.py run --expected-world-size 8 \
      --render-root datasets/render_result_matsynth_50_resized \
      --gt-root datasets/Matsynth_standard/unpacked_512 \
      --parquet-dir datasets/parquet/render_result_matsynth_50_resized \
      --cold-cache linux-drop-caches --output-json /tmp/pbr-bench-8.json

Linux page-cache dropping is opt-in because it is host-wide and normally needs
root.  If it is unavailable, the cold-first-batch gate is recorded as
``indeterminate`` rather than being silently treated as a warm-cache result.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import socket
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from utils.datasets.DataLoader import PBRTextureDataLoader
from utils.datasets.pbr_parquet import (
    SAMPLES_TABLE,
    PBRParquetBatchSampler,
    PBRParquetDataset,
    inspect_bundle,
)

RESULT_VERSION = 1
RESULT_KIND = "tex_edit_pbr_parquet_benchmark"
COMBINED_RESULT_KIND = "tex_edit_pbr_parquet_benchmark_evaluation"
BACKENDS = ("directory", "parquet")
CONTENTION_STATUSES = ("unknown", "uncontended", "contended")
DEFAULT_CONTENTION_NOTE = "contention was not declared by the benchmark operator"
IMAGE_NAMES = ("image", "mask", "meta_data")
FETCH_ATTR = ("Color", "NormalGL", "Height", "Roughness")
METADATA_KEYS = (
    "theta",
    "phi",
    "radius",
    "camera_direction",
    "head_direction",
)
GT_MAPPING = {
    "basecolor": "Color",
    "normal": "NormalGL",
    "height": "Height",
    "roughness": "Roughness",
}
GATE_THRESHOLDS = {
    "minimum_runs": 5,
    "minimum_init_speedup": 5.0,
    "minimum_cold_first_batch_reduction": 0.20,
    "minimum_steady_throughput_ratio": 0.95,
    "maximum_sample_read_amplification": 1.25,
    "maximum_worker_rss_increase_bytes": 512 * 1024 * 1024,
    "maximum_parent_fd_delta": 2,
    "maximum_worker_fd_p95_increase": 8,
    "maximum_worker_fd_growth_during_trial": 8,
    "maximum_parent_thread_delta": 2,
    "maximum_worker_thread_increase": 1.0,
}


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _median(values: Sequence[float | int]) -> float:
    return _percentile(values, 50.0)


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _gate(
    status: str,
    *,
    actual: Any,
    requirement: str,
    reason: str | None = None,
) -> dict[str, Any]:
    if status not in {"pass", "fail", "indeterminate"}:
        raise ValueError(f"invalid gate status: {status}")
    result = {
        "status": status,
        "actual": actual,
        "requirement": requirement,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def _overall_status(checks: Mapping[str, Mapping[str, Any]]) -> str:
    statuses = {str(check["status"]) for check in checks.values()}
    if "fail" in statuses:
        return "fail"
    if "indeterminate" in statuses:
        return "indeterminate"
    return "pass"


def _validate_contention(
    value: Mapping[str, Any] | Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} contention must be an object")
    status = value.get("status")
    note = value.get("note")
    if status not in CONTENTION_STATUSES:
        raise ValueError(
            f"{label} contention status must be one of {CONTENTION_STATUSES}"
        )
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{label} contention note must be a non-empty string")
    return {"status": str(status), "note": note.strip()}


def benchmark_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Match the pretrained entry's business batch without importing models."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    keys = tuple(batch[0])
    if any(tuple(sample) != keys for sample in batch[1:]):
        raise ValueError("batch samples must have identical top-level keys")

    result: dict[str, Any] = {}
    for key in keys:
        first = batch[0][key]
        if isinstance(first, torch.Tensor):
            result[key] = torch.cat(
                [sample[key].unsqueeze(0) for sample in batch], dim=0
            )
            continue
        if key != "meta_data":
            raise TypeError(f"unsupported non-Tensor batch field: {key}")
        metadata: list[Mapping[str, Any]] = []
        for sample in batch:
            value = sample[key]
            if isinstance(value, Mapping):
                loaded = value
            elif isinstance(value, (str, os.PathLike)):
                loaded = OmegaConf.load(value)
            else:
                raise TypeError(
                    "meta_data must be a Mapping or YAML path, "
                    f"got {type(value).__name__}"
                )
            missing = [name for name in METADATA_KEYS if name not in loaded]
            if missing:
                raise KeyError(f"meta_data is missing required fields: {missing}")
            metadata.append(loaded)
        result[key] = {
            name: [sample_metadata[name] for sample_metadata in metadata]
            for name in METADATA_KEYS
        }
    return result


def _read_key_value_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        token = raw_value.strip().split(maxsplit=1)[0]
        try:
            values[key] = int(token)
        except ValueError:
            continue
    return values


def _proc_snapshot(pid: int) -> dict[str, int] | None:
    status = _read_key_value_file(Path("/proc") / str(pid) / "status")
    if not status:
        return None
    try:
        fd_count = len(list((Path("/proc") / str(pid) / "fd").iterdir()))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "rss_bytes": status.get("VmRSS", 0) * 1024,
        "threads": status.get("Threads", 0),
        "fds": fd_count,
    }


def _direct_child_pids(pid: int) -> list[int]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    if not raw:
        return []
    result = []
    for value in raw.split():
        try:
            result.append(int(value))
        except ValueError:
            continue
    return result


def _sample_worker_processes(num_workers: int) -> dict[str, Any]:
    child_pids = _direct_child_pids(os.getpid())
    measured_pids = child_pids if num_workers > 0 else [os.getpid()]
    snapshots = [snapshot for pid in measured_pids if (snapshot := _proc_snapshot(pid))]
    return {
        "rss_bytes": [snapshot["rss_bytes"] for snapshot in snapshots],
        "threads": [snapshot["threads"] for snapshot in snapshots],
        "fds": [snapshot["fds"] for snapshot in snapshots],
        "fd_snapshots": (
            [{str(snapshot["pid"]): snapshot["fds"] for snapshot in snapshots}]
            if snapshots
            else []
        ),
        "worker_counts": [len(child_pids)],
    }


def _read_proc_io(pid: int | None = None) -> dict[str, int]:
    selected_pid = os.getpid() if pid is None else pid
    return _read_key_value_file(Path("/proc") / str(selected_pid) / "io")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_cold_first_batch(
    prepared: Any,
    device: torch.device,
    backend: str,
) -> tuple[Any, Mapping[str, Any], float]:
    # All ranks enter one timing barrier, then iterator construction, worker
    # spawn/prefetch, first next(), and device synchronization are all measured.
    _barrier()
    first_started = time.perf_counter()
    iterator = iter(prepared)
    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"{backend} loader produced no batches") from exc
    _sync_device(device)
    return iterator, first_batch, time.perf_counter() - first_started


def _batch_size(batch: Mapping[str, Any]) -> int:
    image = batch.get("image")
    if not isinstance(image, torch.Tensor) or image.ndim == 0:
        raise RuntimeError("benchmark batch must contain a batched image Tensor")
    return int(image.shape[0])


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(value: Any, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return value
    payload = [value if rank == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _cold_cache_outcome(mode: str) -> dict[str, Any]:
    if mode == "none":
        return {
            "mode": mode,
            "reliable": False,
            "scope": "uncontrolled-existing-cache-state",
            "limitation": "Linux page cache was not cleared",
        }
    if mode != "linux-drop-caches":
        raise ValueError(f"unsupported cold-cache mode: {mode}")
    try:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")
    except OSError as exc:
        return {
            "mode": mode,
            "reliable": False,
            "scope": "attempted-local-linux-page-cache",
            "limitation": f"drop_caches failed: {type(exc).__name__}: {exc}",
        }
    return {
        "mode": mode,
        "reliable": True,
        "scope": "local-linux-page-cache",
        "limitation": "remote storage/server caches are outside this measurement",
    }


def _coordinated_cold_cache(mode: str, rank: int, world_size: int) -> dict[str, Any]:
    outcome = _cold_cache_outcome(mode) if rank == 0 else None
    outcome = _broadcast_object(outcome, rank, world_size)
    _barrier()
    return outcome


def _worker_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_workers": args.num_workers,
        "persistent_workers": False,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        result["prefetch_factor"] = args.prefetch_factor
    return result


def _build_backend(
    backend: str,
    args: argparse.Namespace,
    accelerator: Any,
) -> tuple[Any, torch.utils.data.DataLoader, Any]:
    worker_kwargs = _worker_kwargs(args)
    if backend == "directory":
        PBRTextureDataLoader.image_names = IMAGE_NAMES
        dataset = PBRTextureDataLoader(
            fp=args.render_root,
            gt_fp=args.gt_root,
            transforms={},
            fetch_attr=FETCH_ATTR,
            transform_group={},
            gt_mapping=GT_MAPPING,
        )
        raw_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=benchmark_collate_fn,
            **worker_kwargs,
        )
    elif backend == "parquet":
        dataset = PBRParquetDataset(
            args.parquet_dir,
            image_names=IMAGE_NAMES,
            fetch_attr=FETCH_ATTR,
            transforms={},
            transform_group={},
            use_threads=False,
        )
        batch_sampler = PBRParquetBatchSampler(
            dataset,
            configured_batch_size=args.batch_size,
            num_processes=int(accelerator.num_processes),
            num_workers=args.num_workers,
            seed=args.seed,
        )
        raw_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=benchmark_collate_fn,
            **worker_kwargs,
        )
        batch_sampler.validate_dataloader(raw_loader)
    else:
        raise ValueError(f"unknown benchmark backend: {backend}")
    prepared = accelerator.prepare_data_loader(raw_loader)
    return dataset, raw_loader, prepared


def _extend_monitor(target: dict[str, list[Any]], sample: Mapping[str, Any]) -> None:
    for key in ("rss_bytes", "threads", "fds", "worker_counts"):
        target[key].extend(sample[key])
    target["fd_snapshots"].extend(sample["fd_snapshots"])


def _wait_for_workers_to_exit(timeout_s: float = 2.0) -> list[int]:
    deadline = time.monotonic() + timeout_s
    children = _direct_child_pids(os.getpid())
    while children and time.monotonic() < deadline:
        time.sleep(0.05)
        children = _direct_child_pids(os.getpid())
    return children


def _dispose_loader(
    iterator: Any,
    prepared: Any,
    raw_loader: torch.utils.data.DataLoader,
    dataset: Any,
) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _run_local_trial(
    backend: str,
    run_index: int,
    order_index: int,
    args: argparse.Namespace,
    accelerator: Any,
) -> dict[str, Any]:
    pre = _proc_snapshot(os.getpid())
    _barrier()
    init_started = time.perf_counter()
    dataset, raw_loader, prepared = _build_backend(backend, args, accelerator)
    init_s = time.perf_counter() - init_started
    expected_samples = len(dataset)
    _barrier()

    # Cold first-batch and hot steady are deliberately separate.  The cold
    # loader completes an untimed pass to warm the storage/cache hierarchy,
    # then it and its workers are destroyed before a fresh measured loader is
    # created.  Runtime Dataset caches therefore cannot survive into hot timing.
    cold_cache = _coordinated_cold_cache(
        args.cold_cache,
        int(accelerator.process_index),
        int(accelerator.num_processes),
    )
    iterator, first_batch, first_batch_s = _measure_cold_first_batch(
        prepared, accelerator.device, backend
    )
    first_samples = _batch_size(first_batch)
    warmup_samples = first_samples
    warmup_batches = 1
    while args.max_batches == 0 or warmup_batches < args.max_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        _sync_device(accelerator.device)
        warmup_samples += _batch_size(batch)
        warmup_batches += 1
    del first_batch
    _dispose_loader(iterator, prepared, raw_loader, dataset)
    iterator = prepared = raw_loader = dataset = None
    gc.collect()
    cold_remaining_children = _wait_for_workers_to_exit()

    _barrier()
    hot_dataset, hot_raw_loader, hot_prepared = _build_backend(
        backend, args, accelerator
    )
    if len(hot_dataset) != expected_samples:
        raise RuntimeError(
            f"{backend} sample count changed between cold and hot passes"
        )
    hot_iterator = iter(hot_prepared)
    monitor = {
        "rss_bytes": [],
        "threads": [],
        "fds": [],
        "fd_snapshots": [],
        "worker_counts": [],
    }
    wait_times: list[float] = []
    steady_samples = 0
    batches = 0
    monitor_overhead_s = 0.0
    _barrier()
    steady_started = time.perf_counter()
    while args.max_batches == 0 or batches < args.max_batches:
        wait_started = time.perf_counter()
        try:
            batch = next(hot_iterator)
        except StopIteration:
            break
        _sync_device(accelerator.device)
        wait_times.append(time.perf_counter() - wait_started)
        steady_samples += _batch_size(batch)
        batches += 1
        if batches == 1 or batches % args.proc_sample_every == 0:
            monitor_started = time.perf_counter()
            _extend_monitor(monitor, _sample_worker_processes(args.num_workers))
            monitor_overhead_s += time.perf_counter() - monitor_started
    steady_wall_s = time.perf_counter() - steady_started
    steady_s = max(steady_wall_s - monitor_overhead_s, 0.0)
    _barrier()

    _dispose_loader(hot_iterator, hot_prepared, hot_raw_loader, hot_dataset)
    hot_iterator = hot_prepared = hot_raw_loader = hot_dataset = None
    gc.collect()
    remaining_children = _wait_for_workers_to_exit()
    post = _proc_snapshot(os.getpid())
    full_pass = args.max_batches == 0
    if full_pass and (warmup_samples <= 0 or steady_samples <= 0):
        raise RuntimeError(f"{backend} cold or hot full pass consumed no samples")
    return {
        "backend": backend,
        "run_index": run_index,
        "order_index": order_index,
        "rank": int(accelerator.process_index),
        "init_s": init_s,
        "first_batch_s": first_batch_s,
        "first_batch_samples": first_samples,
        "warmup_samples": warmup_samples,
        "warmup_batches": warmup_batches,
        "steady_s": steady_s,
        "steady_samples": steady_samples,
        "wait_times_s": wait_times,
        "batches": batches,
        "local_samples": steady_samples,
        "expected_global_samples": expected_samples,
        "cold_cache": cold_cache,
        "hot_cache": {
            "method": "complete-untimed-pass-then-fresh-loader",
            "scope": "storage/page cache only; Dataset and worker caches destroyed",
            "full_warmup": full_pass,
        },
        "monitor": monitor,
        "pre": pre,
        "post": post,
        "cold_remaining_children": cold_remaining_children,
        "remaining_children": remaining_children,
        "full_pass": full_pass,
    }


def _flatten(records: Sequence[Mapping[str, Any]], *keys: str) -> list[Any]:
    result: list[Any] = []
    for record in records:
        value: Any = record
        for key in keys:
            value = value[key]
        result.extend(value)
    return result


def _stepwise_rank_max_waits(
    records: Sequence[Mapping[str, Any]],
) -> list[float]:
    """Align rank waits by batch ordinal and retain unmatched tail samples."""

    sequences = [
        [float(value) for value in record["wait_times_s"]] for record in records
    ]
    step_count = max((len(sequence) for sequence in sequences), default=0)
    return [
        max(sequence[step] for sequence in sequences if step < len(sequence))
        for step in range(step_count)
    ]


def _worker_fd_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sequences: dict[tuple[int, str], list[int]] = {}
    available = bool(records)
    observations = 0
    for rank_index, record in enumerate(records):
        monitor = record["monitor"]
        snapshots = list(monitor.get("fd_snapshots", []))
        worker_counts = [int(value) for value in monitor.get("worker_counts", [])]
        if not snapshots or len(snapshots) != len(worker_counts):
            available = False
            continue
        expected_pids: set[str] | None = None
        rank = int(record.get("rank", rank_index))
        for snapshot, worker_count in zip(snapshots, worker_counts):
            if not isinstance(snapshot, Mapping):
                available = False
                continue
            normalized = {str(pid): int(value) for pid, value in snapshot.items()}
            pids = set(normalized)
            if len(pids) != max(worker_count, 1):
                available = False
            if expected_pids is None:
                expected_pids = pids
            elif pids != expected_pids:
                available = False
            observations += len(normalized)
            for pid, value in normalized.items():
                sequences.setdefault((rank, pid), []).append(value)

    values = list(sequences.values())
    maximum_growth = (
        max(max(sequence) - min(sequence) for sequence in values) if values else None
    )
    sustained_growth = any(
        len(sequence) >= 3
        and all(right > left for left, right in zip(sequence, sequence[1:]))
        for sequence in values
    )
    return {
        "available": available and bool(values),
        "observations": observations,
        "worker_sequences": len(values),
        "maximum_growth": maximum_growth,
        "sustained_growth": sustained_growth,
    }


def _aggregate_trial(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty distributed trial")
    first = records[0]
    backend = str(first["backend"])
    run_index = int(first["run_index"])
    order_index = int(first["order_index"])
    if any(
        record["backend"] != backend
        or int(record["run_index"]) != run_index
        or int(record["order_index"]) != order_index
        for record in records
    ):
        raise RuntimeError("distributed trial identities do not match")
    expected_values = {int(record["expected_global_samples"]) for record in records}
    if len(expected_values) != 1:
        raise RuntimeError("backend sample counts differ across ranks")
    expected_samples = expected_values.pop()
    total_samples = sum(int(record["local_samples"]) for record in records)
    warmup_samples = sum(int(record["warmup_samples"]) for record in records)
    full_pass = all(bool(record["full_pass"]) for record in records)
    if full_pass and (
        total_samples != expected_samples or warmup_samples != expected_samples
    ):
        raise RuntimeError(
            f"{backend} cold/hot full-pass coverage mismatch: "
            f"warmup={warmup_samples}, hot={total_samples}, "
            f"expected={expected_samples}"
        )

    wait_times = _flatten(records, "wait_times_s")
    stepwise_rank_max_waits = _stepwise_rank_max_waits(records)
    rss_values = _flatten(records, "monitor", "rss_bytes")
    thread_values = _flatten(records, "monitor", "threads")
    fd_values = _flatten(records, "monitor", "fds")
    worker_counts = _flatten(records, "monitor", "worker_counts")
    fd_diagnostics = _worker_fd_diagnostics(records)
    steady_elapsed = max(float(record["steady_s"]) for record in records)
    steady_samples = sum(int(record["steady_samples"]) for record in records)
    cold_reliable = all(bool(record["cold_cache"]["reliable"]) for record in records)

    parent_fd_deltas = []
    parent_thread_deltas = []
    parent_post_fds = []
    parent_post_threads = []
    proc_available = True
    for record in records:
        pre = record["pre"]
        post = record["post"]
        if pre is None or post is None:
            proc_available = False
            continue
        parent_fd_deltas.append(int(post["fds"]) - int(pre["fds"]))
        parent_thread_deltas.append(int(post["threads"]) - int(pre["threads"]))
        parent_post_fds.append(int(post["fds"]))
        parent_post_threads.append(int(post["threads"]))

    public = {
        "backend": backend,
        "run_index": run_index,
        "order_index": order_index,
        "rank_count": len(records),
        "init_s": max(float(record["init_s"]) for record in records),
        "first_batch_s": max(float(record["first_batch_s"]) for record in records),
        "first_batch_samples": sum(
            int(record["first_batch_samples"]) for record in records
        ),
        "warmup_samples": warmup_samples,
        "hot_cache": dict(first["hot_cache"]),
        "steady_elapsed_s": steady_elapsed,
        "steady_samples": steady_samples,
        "steady_samples_per_s": (
            steady_samples / steady_elapsed if steady_elapsed > 0.0 else None
        ),
        "gpu_data_wait_p50_s": (
            _median(stepwise_rank_max_waits) if stepwise_rank_max_waits else None
        ),
        "gpu_data_wait_aggregation": "p50-of-stepwise-rank-max",
        "batches_per_rank": [int(record["batches"]) for record in records],
        "samples": total_samples,
        "expected_samples": expected_samples,
        "full_pass": full_pass,
        "cold_cache": dict(first["cold_cache"]),
        "cold_cache_reliable": cold_reliable,
        "worker_rss_p95_bytes": _percentile(rss_values, 95.0) if rss_values else None,
        "worker_threads_p95": (
            _percentile(thread_values, 95.0) if thread_values else None
        ),
        "worker_fds_p95": _percentile(fd_values, 95.0) if fd_values else None,
        "worker_fd_peak": max(fd_values) if fd_values else None,
        "worker_fd_monitor_available": fd_diagnostics["available"],
        "worker_fd_observations": fd_diagnostics["observations"],
        "worker_fd_sequences": fd_diagnostics["worker_sequences"],
        "maximum_worker_fd_growth": fd_diagnostics["maximum_growth"],
        "sustained_worker_fd_growth": fd_diagnostics["sustained_growth"],
        "worker_count_min": min(worker_counts) if worker_counts else None,
        "worker_count_max": max(worker_counts) if worker_counts else None,
        "proc_monitor_available": proc_available and bool(rss_values),
        "maximum_parent_fd_delta": max(parent_fd_deltas, default=None),
        "maximum_parent_thread_delta": max(parent_thread_deltas, default=None),
        "parent_post_fds": parent_post_fds,
        "parent_post_threads": parent_post_threads,
        "remaining_child_processes": sum(
            len(record["cold_remaining_children"]) + len(record["remaining_children"])
            for record in records
        ),
        "fresh_dataset_and_workers": True,
    }
    public["_raw_monitor"] = {
        "rss_bytes": rss_values,
        "threads": thread_values,
        "fds": fd_values,
        "wait_times_s": wait_times,
    }
    return public


def _process_lifecycle(
    initial: Sequence[Mapping[str, Any] | None],
    final: Sequence[Mapping[str, Any] | None],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    available = all(snapshot is not None for snapshot in (*initial, *final))
    if not available:
        return {"available": False}
    fd_deltas = [
        int(end["fds"]) - int(start["fds"]) for start, end in zip(initial, final)
    ]
    thread_deltas = [
        int(end["threads"]) - int(start["threads"])
        for start, end in zip(initial, final)
    ]
    fd_sequences = [[] for _ in initial]
    thread_sequences = [[] for _ in initial]
    for trial in trials:
        for rank, value in enumerate(trial["parent_post_fds"]):
            fd_sequences[rank].append(int(value))
        for rank, value in enumerate(trial["parent_post_threads"]):
            thread_sequences[rank].append(int(value))

    def monotonic_growth(sequences: Sequence[Sequence[int]]) -> bool:
        return any(
            len(sequence) >= 2
            and all(left <= right for left, right in zip(sequence, sequence[1:]))
            and sequence[-1] > sequence[0]
            for sequence in sequences
        )

    return {
        "available": True,
        "fd_deltas": fd_deltas,
        "thread_deltas": thread_deltas,
        "post_fd_sequences": fd_sequences,
        "post_thread_sequences": thread_sequences,
        "monotonic_fd_growth": monotonic_growth(fd_sequences),
        "monotonic_thread_growth": monotonic_growth(thread_sequences),
    }


def _public_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trial.items() if not key.startswith("_")}


def summarize_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for backend in BACKENDS:
        selected = [trial for trial in trials if trial["backend"] == backend]
        if not selected:
            continue
        steady_rates = [
            float(trial["steady_samples_per_s"])
            for trial in selected
            if trial["steady_samples_per_s"] is not None
        ]
        wait_p50s = [
            float(trial["gpu_data_wait_p50_s"])
            for trial in selected
            if trial["gpu_data_wait_p50_s"] is not None
        ]
        rss_values = _flatten(selected, "_raw_monitor", "rss_bytes")
        thread_values = _flatten(selected, "_raw_monitor", "threads")
        fd_values = _flatten(selected, "_raw_monitor", "fds")
        summary[backend] = {
            "trials": len(selected),
            "init_p50_s": _median([trial["init_s"] for trial in selected]),
            "first_batch_p50_s": _median(
                [trial["first_batch_s"] for trial in selected]
            ),
            "steady_samples_per_s_p50": _median(steady_rates) if steady_rates else None,
            "gpu_data_wait_p50_s": _median(wait_p50s) if wait_p50s else None,
            "worker_rss_p95_bytes": (
                _percentile(rss_values, 95.0) if rss_values else None
            ),
            "worker_threads_p95": (
                _percentile(thread_values, 95.0) if thread_values else None
            ),
            "worker_fds_p95": (_percentile(fd_values, 95.0) if fd_values else None),
            "worker_fd_peak": max(fd_values) if fd_values else None,
            "worker_fd_observations": sum(
                int(trial["worker_fd_observations"]) for trial in selected
            ),
            "worker_fd_sequences": sum(
                int(trial["worker_fd_sequences"]) for trial in selected
            ),
            "maximum_worker_fd_growth": max(
                (
                    trial["maximum_worker_fd_growth"]
                    for trial in selected
                    if trial["maximum_worker_fd_growth"] is not None
                ),
                default=None,
            ),
            "sustained_worker_fd_growth": any(
                bool(trial["sustained_worker_fd_growth"]) for trial in selected
            ),
            "maximum_parent_fd_delta": max(
                (
                    trial["maximum_parent_fd_delta"]
                    for trial in selected
                    if trial["maximum_parent_fd_delta"] is not None
                ),
                default=None,
            ),
            "maximum_parent_thread_delta": max(
                (
                    trial["maximum_parent_thread_delta"]
                    for trial in selected
                    if trial["maximum_parent_thread_delta"] is not None
                ),
                default=None,
            ),
            "remaining_child_processes": sum(
                int(trial["remaining_child_processes"]) for trial in selected
            ),
            "all_full_pass": all(bool(trial["full_pass"]) for trial in selected),
            "all_cold_cache_reliable": all(
                bool(trial["cold_cache_reliable"]) for trial in selected
            ),
            "all_proc_monitor_available": all(
                bool(trial["proc_monitor_available"]) for trial in selected
            ),
            "all_worker_fd_monitor_available": all(
                bool(trial["worker_fd_monitor_available"]) for trial in selected
            ),
        }
    return summary


def _column_root(path_in_schema: str) -> str:
    return path_in_schema.split(".", 1)[0]


def _scan_sample_shards(paths: Sequence[Path]) -> dict[str, Any]:
    """Upper-bound payload-phase row-group reads via ``/proc``.

    Every row group reopens its Parquet file, rereads the footer, and reads all
    sample columns.  Runtime reads each assigned row group at most once per
    pass and retains bounded file handles, so this deliberately pessimistic
    access pattern covers rank/worker shard duplication rather than understating
    it with one sequential handle per shard.
    """

    before = _read_proc_io()
    logical_payload_bytes = 0
    footer_binary_column_bytes = 0
    rows = 0
    row_groups = 0
    for path in paths:
        with path.open("rb") as handle:
            row_group_count = pq.ParquetFile(handle).metadata.num_row_groups
        for row_group_index in range(row_group_count):
            with path.open("rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                row_group = parquet_file.metadata.row_group(row_group_index)
                for column_index in range(row_group.num_columns):
                    column = row_group.column(column_index)
                    if _column_root(column.path_in_schema) in {
                        "image_png",
                        "mask_png",
                    }:
                        footer_binary_column_bytes += int(column.total_compressed_size)
                table = parquet_file.read_row_group(
                    row_group_index,
                    use_threads=False,
                )
            for name in ("image_png", "mask_png"):
                lengths = pc.binary_length(table[name])
                total = pc.sum(lengths).as_py()
                logical_payload_bytes += int(total or 0)
            rows += table.num_rows
            row_groups += 1
    after = _read_proc_io()
    rchar_available = "rchar" in before and "rchar" in after
    storage_read_bytes_available = "read_bytes" in before and "read_bytes" in after
    return {
        "logical_payload_bytes": logical_payload_bytes,
        "footer_binary_column_bytes": footer_binary_column_bytes,
        "process_rchar_available": rchar_available,
        "process_read_chars": (
            max(int(after["rchar"]) - int(before["rchar"]), 0)
            if rchar_available
            else None
        ),
        "storage_read_bytes_available": storage_read_bytes_available,
        "storage_read_bytes": (
            max(int(after["read_bytes"]) - int(before["read_bytes"]), 0)
            if storage_read_bytes_available
            else None
        ),
        "rows": rows,
        "row_groups": row_groups,
        "shards": len(paths),
    }


def measure_sample_io_amplification(
    parquet_dir: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    index = inspect_bundle(parquet_dir)
    sample_shards = [
        shard for shard in index.shards if shard.table_type == SAMPLES_TABLE
    ]
    selected = [
        index.root / shard.relative_path
        for shard in sample_shards
        if shard.ordinal % world_size == rank
    ]
    result = _scan_sample_shards(selected)
    result.update(
        {
            "rank": rank,
            "world_size": world_size,
            "method": "per-row-group-reopen-all-sample-columns-proc-rchar-upper-bound",
        }
    )
    return result


def _aggregate_io(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        key: sum(int(record[key]) for record in records)
        for key in (
            "logical_payload_bytes",
            "footer_binary_column_bytes",
            "rows",
            "row_groups",
            "shards",
        )
    }
    rchar_available = bool(records) and all(
        record.get("process_rchar_available") is True
        and record.get("process_read_chars") is not None
        for record in records
    )
    storage_read_bytes_available = bool(records) and all(
        record.get("storage_read_bytes_available") is True
        and record.get("storage_read_bytes") is not None
        for record in records
    )
    process_read_chars = (
        sum(int(record["process_read_chars"]) for record in records)
        if rchar_available
        else None
    )
    storage_read_bytes = (
        sum(int(record["storage_read_bytes"]) for record in records)
        if storage_read_bytes_available
        else None
    )
    logical = totals["logical_payload_bytes"]
    footer_ratio = _finite_ratio(totals["footer_binary_column_bytes"], logical)
    observed_ratio = (
        _finite_ratio(process_read_chars, logical)
        if process_read_chars is not None
        else None
    )
    measured = logical > 0 and rchar_available and observed_ratio is not None
    return {
        "status": "measured" if measured else "unavailable",
        "method": "per-row-group-reopen-all-sample-columns-proc-rchar-upper-bound",
        **totals,
        "process_rchar_available": rchar_available,
        "process_read_chars": process_read_chars,
        "storage_read_bytes_available": storage_read_bytes_available,
        "storage_read_bytes": storage_read_bytes,
        "footer_binary_column_ratio": footer_ratio,
        "footer_binary_column_ratio_role": "diagnostic-only-not-a-measured-read-ratio",
        "observed_process_read_ratio": observed_ratio,
        "gate_ratio": observed_ratio if measured else None,
        "limitation": (
            None
            if measured
            else "/proc/io rchar was unavailable on at least one benchmark rank"
        ),
        "upper_bound_reason": (
            "payload phase only: each row group reopens its shard and reads every "
            "sample column; runtime uses bounded handles and reads each assigned row "
            "group at most once; initialization/inspect footer reads are excluded and "
            "covered by the separate initialization gate"
        ),
        "storage_read_bytes_note": (
            "storage_read_bytes can be zero for page-cache/PFS hits and is reported "
            "but not used as the amplification denominator"
        ),
    }


def evaluate_run(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["config"]
    summary = result["summary"]
    directory = summary.get("directory")
    parquet = summary.get("parquet")
    checks: dict[str, dict[str, Any]] = {}

    runtime_matches = (
        result["environment"].get("accelerate") == "1.0.0"
        and result["environment"].get("pyarrow") == "24.0.0"
        and int(result["environment"]["world_size"]) in {1, 8}
    )
    checks["runtime_contract"] = _gate(
        "pass" if runtime_matches else "fail",
        actual={
            "accelerate": result["environment"].get("accelerate"),
            "pyarrow": result["environment"].get("pyarrow"),
            "world_size": result["environment"].get("world_size"),
        },
        requirement=("accelerate==1.0.0, pyarrow==24.0.0, and world size is 1 or 8"),
    )

    dataset = result.get("dataset", {})
    dataset_matches = (
        int(dataset.get("materials", -1)) == 5789
        and int(dataset.get("samples", -1)) == 289450
        and int(dataset.get("views_per_material", -1)) == 50
    )
    checks["dataset_contract"] = _gate(
        "pass" if dataset_matches else "fail",
        actual={
            "materials": dataset.get("materials"),
            "samples": dataset.get("samples"),
            "views_per_material": dataset.get("views_per_material"),
        },
        requirement=(
            "full benchmark bundle contains 5,789 materials, 289,450 samples, "
            "and 50 views per material"
        ),
    )

    expected_trials = int(config["runs"])
    trial_counts_ok = (
        directory is not None
        and parquet is not None
        and int(directory["trials"]) == expected_trials
        and int(parquet["trials"]) == expected_trials
    )
    trial_orders: dict[int, list[tuple[int, str]]] = {}
    for trial in result.get("trials", []):
        trial_orders.setdefault(int(trial["run_index"]), []).append(
            (int(trial["order_index"]), str(trial["backend"]))
        )
    alternating = len(trial_orders) == expected_trials and all(
        tuple(backend for _order, backend in sorted(trial_orders[run_index]))
        == _alternating_order(run_index)
        for run_index in range(expected_trials)
    )
    trial_expected_counts = {
        int(trial["expected_samples"]) for trial in result.get("trials", [])
    }
    coverage_ok = bool(result.get("trials")) and all(
        int(trial["samples"]) == int(trial["expected_samples"])
        for trial in result["trials"]
    )
    coverage_ok = coverage_ok and len(trial_expected_counts) == 1
    hot_warmup_valid = bool(result.get("trials")) and all(
        trial.get("hot_cache", {}).get("method")
        == "complete-untimed-pass-then-fresh-loader"
        and bool(trial.get("hot_cache", {}).get("full_warmup"))
        for trial in result["trials"]
    )
    full_pass = bool(
        directory
        and parquet
        and directory["all_full_pass"]
        and parquet["all_full_pass"]
        and int(config["max_batches"]) == 0
        and hot_warmup_valid
    )
    enough_runs = expected_trials >= GATE_THRESHOLDS["minimum_runs"]
    prerequisite_status = (
        "pass"
        if trial_counts_ok and alternating and full_pass and enough_runs and coverage_ok
        else "indeterminate"
    )
    checks["five_alternating_full_pass_runs"] = _gate(
        prerequisite_status,
        actual={
            "configured_runs": expected_trials,
            "trial_counts_ok": trial_counts_ok,
            "alternating": alternating,
            "full_pass": full_pass,
            "independent_hot_warmup": hot_warmup_valid,
            "exact_same_source_coverage": coverage_ok,
        },
        requirement="at least five alternating, fresh-loader, full-pass trials",
        reason=(
            None
            if prerequisite_status == "pass"
            else "short/partial runs are diagnostic only and cannot produce GO"
        ),
    )

    if directory is None or parquet is None:
        for name in (
            "initialization_speedup",
            "cold_first_batch_reduction",
            "steady_throughput",
            "gpu_data_wait",
            "worker_rss",
            "fd_cleanup",
            "thread_cleanup",
            "pass_cache_isolation",
        ):
            checks[name] = _gate(
                "indeterminate",
                actual=None,
                requirement="both backend summaries must be present",
                reason="missing backend benchmark results",
            )
    else:
        init_speedup = _finite_ratio(
            float(directory["init_p50_s"]), float(parquet["init_p50_s"])
        )
        if not enough_runs:
            init_status = "indeterminate"
            init_reason = "initialization p50 requires at least five trials"
        else:
            init_status = (
                "pass"
                if init_speedup is not None
                and init_speedup >= GATE_THRESHOLDS["minimum_init_speedup"]
                else "fail"
            )
            init_reason = None
        checks["initialization_speedup"] = _gate(
            init_status,
            actual=init_speedup,
            requirement=(
                f">= {GATE_THRESHOLDS['minimum_init_speedup']:.2f}x "
                "directory/Parquet init p50"
            ),
            reason=init_reason,
        )

        cold_reduction = None
        if float(directory["first_batch_p50_s"]) > 0.0:
            cold_reduction = 1.0 - (
                float(parquet["first_batch_p50_s"])
                / float(directory["first_batch_p50_s"])
            )
        cold_reliable = bool(
            directory["all_cold_cache_reliable"] and parquet["all_cold_cache_reliable"]
        )
        if not cold_reliable or not full_pass or not enough_runs:
            cold_status = "indeterminate"
            cold_reason = "cold cache was not reliably cleared for every full trial"
        else:
            cold_status = (
                "pass"
                if cold_reduction is not None
                and cold_reduction
                >= GATE_THRESHOLDS["minimum_cold_first_batch_reduction"]
                else "fail"
            )
            cold_reason = None
        checks["cold_first_batch_reduction"] = _gate(
            cold_status,
            actual=cold_reduction,
            requirement=(
                f">= {GATE_THRESHOLDS['minimum_cold_first_batch_reduction']:.0%} "
                "reduction in cold first-batch p50"
            ),
            reason=cold_reason,
        )

        directory_rate = directory["steady_samples_per_s_p50"]
        parquet_rate = parquet["steady_samples_per_s_p50"]
        throughput_ratio = (
            None
            if directory_rate is None or parquet_rate is None
            else _finite_ratio(float(parquet_rate), float(directory_rate))
        )
        if not full_pass or not enough_runs:
            throughput_status = "indeterminate"
        else:
            throughput_status = (
                "pass"
                if throughput_ratio is not None
                and throughput_ratio
                >= GATE_THRESHOLDS["minimum_steady_throughput_ratio"]
                else "fail"
            )
        checks["steady_throughput"] = _gate(
            throughput_status,
            actual=throughput_ratio,
            requirement=(
                f">= {GATE_THRESHOLDS['minimum_steady_throughput_ratio']:.0%} "
                "of directory steady samples/s p50"
            ),
            reason=(
                "requires five full passes"
                if throughput_status == "indeterminate"
                else None
            ),
        )

        directory_wait = directory["gpu_data_wait_p50_s"]
        parquet_wait = parquet["gpu_data_wait_p50_s"]
        wait_ratio = (
            None
            if directory_wait is None or parquet_wait is None
            else _finite_ratio(float(parquet_wait), float(directory_wait))
        )
        gpu_valid = str(result["environment"]["device_type"]) == "cuda"
        if not gpu_valid or not full_pass or not enough_runs:
            wait_status = "indeterminate"
            wait_reason = "GPU data-wait requires CUDA and five full passes"
        else:
            wait_status = (
                "pass" if wait_ratio is not None and wait_ratio <= 1.0 else "fail"
            )
            wait_reason = None
        checks["gpu_data_wait"] = _gate(
            wait_status,
            actual=wait_ratio,
            requirement="Parquet synchronized GPU data-wait p50 <= directory baseline",
            reason=wait_reason,
        )

        directory_rss = directory["worker_rss_p95_bytes"]
        parquet_rss = parquet["worker_rss_p95_bytes"]
        proc_available = bool(
            directory["all_proc_monitor_available"]
            and parquet["all_proc_monitor_available"]
        )
        rss_increase = (
            None
            if directory_rss is None or parquet_rss is None
            else float(parquet_rss) - float(directory_rss)
        )
        if not proc_available or not full_pass:
            rss_status = "indeterminate"
        else:
            rss_status = (
                "pass"
                if rss_increase is not None
                and rss_increase <= GATE_THRESHOLDS["maximum_worker_rss_increase_bytes"]
                else "fail"
            )
        checks["worker_rss"] = _gate(
            rss_status,
            actual=rss_increase,
            requirement=(
                "Parquet worker RSS p95 <= directory + "
                f"{GATE_THRESHOLDS['maximum_worker_rss_increase_bytes']} bytes"
            ),
            reason=(
                "/proc worker monitoring unavailable" if not proc_available else None
            ),
        )

        fd_values = [
            value
            for value in (
                directory["maximum_parent_fd_delta"],
                parquet["maximum_parent_fd_delta"],
            )
            if value is not None
        ]
        directory_worker_fds = directory.get("worker_fds_p95")
        parquet_worker_fds = parquet.get("worker_fds_p95")
        worker_fd_p95_increase = (
            None
            if directory_worker_fds is None or parquet_worker_fds is None
            else float(parquet_worker_fds) - float(directory_worker_fds)
        )
        worker_fd_growth_values = [
            int(value)
            for value in (
                directory.get("maximum_worker_fd_growth"),
                parquet.get("maximum_worker_fd_growth"),
            )
            if value is not None
        ]
        maximum_worker_fd_growth = (
            max(worker_fd_growth_values) if worker_fd_growth_values else None
        )
        worker_fd_available = bool(
            directory.get("all_worker_fd_monitor_available")
            and parquet.get("all_worker_fd_monitor_available")
            and worker_fd_p95_increase is not None
            and maximum_worker_fd_growth is not None
        )
        lifecycle = result.get("process_lifecycle", {})
        lifecycle_available = bool(lifecycle.get("available"))
        lifecycle_fd_deltas = lifecycle.get("fd_deltas", [])
        fd_evidence_available = bool(
            proc_available
            and worker_fd_available
            and lifecycle_available
            and lifecycle_fd_deltas
        )
        fd_clean = (
            fd_evidence_available
            and worker_fd_p95_increase
            <= GATE_THRESHOLDS["maximum_worker_fd_p95_increase"]
            and maximum_worker_fd_growth
            <= GATE_THRESHOLDS["maximum_worker_fd_growth_during_trial"]
            and max(float(value) for value in lifecycle_fd_deltas)
            <= GATE_THRESHOLDS["maximum_parent_fd_delta"]
            and not bool(lifecycle.get("monotonic_fd_growth"))
            and int(directory["remaining_child_processes"]) == 0
            and int(parquet["remaining_child_processes"]) == 0
        )
        checks["fd_cleanup"] = _gate(
            (
                "pass"
                if fd_clean
                else ("fail" if fd_evidence_available else "indeterminate")
            ),
            actual={
                "worker_fd_p95": {
                    "directory": directory_worker_fds,
                    "parquet": parquet_worker_fds,
                    "increase": worker_fd_p95_increase,
                },
                "maximum_worker_fd_growth": maximum_worker_fd_growth,
                "sustained_worker_fd_growth_observed": {
                    "directory": directory.get("sustained_worker_fd_growth"),
                    "parquet": parquet.get("sustained_worker_fd_growth"),
                },
                "maximum_parent_fd_deltas": fd_values,
                "benchmark_lifecycle_fd_deltas": lifecycle_fd_deltas,
                "monotonic_parent_growth": lifecycle.get("monotonic_fd_growth"),
                "remaining_children": {
                    "directory": directory["remaining_child_processes"],
                    "parquet": parquet["remaining_child_processes"],
                },
            },
            requirement=(
                "Parquet worker FD p95 <= directory + "
                f"{GATE_THRESHOLDS['maximum_worker_fd_p95_increase']}; each worker "
                "PID max(fd)-min(fd) per trial <= "
                f"{GATE_THRESHOLDS['maximum_worker_fd_growth_during_trial']}; "
                "whole-benchmark parent FD delta <= "
                f"{GATE_THRESHOLDS['maximum_parent_fd_delta']}; no monotonic parent "
                "growth; and no DataLoader child survives a trial"
            ),
            reason=(
                None
                if fd_evidence_available
                else "/proc worker/parent FD time-series monitoring unavailable"
            ),
        )

        directory_threads = directory["worker_threads_p95"]
        parquet_threads = parquet["worker_threads_p95"]
        worker_thread_increase = (
            None
            if directory_threads is None or parquet_threads is None
            else float(parquet_threads) - float(directory_threads)
        )
        parent_thread_deltas = [
            value
            for value in (
                directory["maximum_parent_thread_delta"],
                parquet["maximum_parent_thread_delta"],
            )
            if value is not None
        ]
        lifecycle_thread_deltas = lifecycle.get("thread_deltas", [])
        thread_clean = (
            proc_available
            and lifecycle_available
            and worker_thread_increase is not None
            and worker_thread_increase
            <= GATE_THRESHOLDS["maximum_worker_thread_increase"]
            and lifecycle_thread_deltas
            and max(float(value) for value in lifecycle_thread_deltas)
            <= GATE_THRESHOLDS["maximum_parent_thread_delta"]
            and not bool(lifecycle.get("monotonic_thread_growth"))
        )
        checks["thread_cleanup"] = _gate(
            "pass" if thread_clean else ("fail" if proc_available else "indeterminate"),
            actual={
                "worker_thread_p95_increase": worker_thread_increase,
                "maximum_parent_thread_deltas": parent_thread_deltas,
                "benchmark_lifecycle_thread_deltas": lifecycle_thread_deltas,
                "monotonic_growth": lifecycle.get("monotonic_thread_growth"),
            },
            requirement=(
                "no Arrow worker-thread inflation, whole-benchmark thread delta <= 2, "
                "and no monotonic growth"
            ),
            reason=("/proc monitoring unavailable" if not proc_available else None),
        )

        cache_clean = (
            int(config["persistent_workers"]) == 0
            and int(directory["remaining_child_processes"]) == 0
            and int(parquet["remaining_child_processes"]) == 0
            and all(
                bool(trial.get("fresh_dataset_and_workers"))
                for trial in result["trials"]
            )
        )
        checks["pass_cache_isolation"] = _gate(
            "pass" if cache_clean else "fail",
            actual=cache_clean,
            requirement=(
                "persistent_workers=false, fresh loader per trial, and no worker "
                "survives into the next pass"
            ),
        )

    worker_count_available = bool(result.get("trials")) and all(
        trial.get("worker_count_min") is not None
        and trial.get("worker_count_max") is not None
        for trial in result["trials"]
    )
    worker_count_exact = worker_count_available and all(
        int(trial["worker_count_min"]) == int(config["num_workers"])
        and int(trial["worker_count_max"]) == int(config["num_workers"])
        for trial in result["trials"]
    )
    checks["worker_process_count"] = _gate(
        (
            "pass"
            if worker_count_exact
            else ("fail" if worker_count_available else "indeterminate")
        ),
        actual={
            "configured": config["num_workers"],
            "exact_every_trial": worker_count_exact,
        },
        requirement="every rank observes exactly configured num_workers",
        reason=None if worker_count_available else "/proc worker count unavailable",
    )

    io_result = result.get("sample_io_amplification", {})
    io_ratio = io_result.get("gate_ratio")
    io_method_valid = io_result.get("method") == (
        "per-row-group-reopen-all-sample-columns-proc-rchar-upper-bound"
    )
    io_coverage_valid = int(io_result.get("rows", -1)) == int(
        dataset.get("samples", -2)
    ) and int(io_result.get("row_groups", -1)) == int(dataset.get("materials", -2))
    if (
        io_result.get("status") != "measured"
        or io_result.get("process_rchar_available") is not True
        or io_ratio is None
    ):
        io_status = "indeterminate"
        io_reason = (
            "conservative sample I/O upper-bound scan was skipped or unavailable"
        )
    elif not io_method_valid or not io_coverage_valid:
        io_status = "fail"
        io_reason = "I/O scan method or full row-group coverage is invalid"
    else:
        io_status = (
            "pass"
            if float(io_ratio) <= GATE_THRESHOLDS["maximum_sample_read_amplification"]
            else "fail"
        )
        io_reason = None
    checks["sample_read_amplification"] = _gate(
        io_status,
        actual={
            "ratio": io_ratio,
            "process_rchar_available": io_result.get("process_rchar_available"),
            "footer_ratio_diagnostic_only": io_result.get("footer_binary_column_ratio"),
            "method_valid": io_method_valid,
            "coverage_valid": io_coverage_valid,
        },
        requirement=(
            "actual sample binary read bytes / logical encoded payload <= "
            f"{GATE_THRESHOLDS['maximum_sample_read_amplification']:.2f}x"
        ),
        reason=io_reason,
    )

    status = _overall_status(checks)
    return {
        "status": status,
        "decision": {
            "pass": "world-size-pass",
            "fail": "no-go",
            "indeterminate": "review-required",
        }[status],
        "thresholds": dict(GATE_THRESHOLDS),
        "checks": checks,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fsynced result atomically without an overwrite path."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable result already exists: {path}") from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _alternating_order(run_index: int) -> tuple[str, str]:
    return BACKENDS if run_index % 2 == 0 else tuple(reversed(BACKENDS))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any] | None:
    from accelerate import Accelerator, __version__ as accelerate_version
    from accelerate.utils import DataLoaderConfiguration

    if accelerate_version != "1.0.0":
        raise RuntimeError(
            f"benchmark requires accelerate==1.0.0, got {accelerate_version}"
        )
    if pa.__version__ != "24.0.0":
        raise RuntimeError(f"benchmark requires pyarrow==24.0.0, got {pa.__version__}")
    contention = _validate_contention(
        {
            "status": args.contention_status,
            "note": args.contention_note,
        },
        label="run",
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    use_cpu = args.device == "cpu" or (
        args.device == "auto" and not torch.cuda.is_available()
    )
    accelerator = Accelerator(
        cpu=use_cpu,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"launched world size {world_size} != --expected-world-size "
            f"{args.expected_world_size}"
        )
    hostnames = _all_gather_object(socket.gethostname(), world_size)
    if len(set(hostnames)) != 1:
        raise RuntimeError("benchmark gate requires all ranks on the same machine")

    output_path = Path(args.output_json)
    output_error = None
    if rank == 0 and output_path.exists():
        output_error = f"immutable result already exists: {output_path.resolve()}"
    output_error = _broadcast_object(output_error, rank, world_size)
    if output_error is not None:
        raise FileExistsError(output_error)
    _barrier()

    bundle_index = inspect_bundle(args.parquet_dir)
    base_result: dict[str, Any] | None = None
    if rank == 0:
        base_result = {
            "kind": RESULT_KIND,
            "version": RESULT_VERSION,
            "complete": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "bundle_id": bundle_index.marker["bundle_id"],
            "contention": contention,
            "dataset": {
                "materials": bundle_index.material_count,
                "samples": bundle_index.sample_count,
                "views_per_material": bundle_index.views_per_material,
            },
            "alternating_backend_order": True,
            "config": {
                "render_root": str(Path(args.render_root).resolve()),
                "gt_root": str(Path(args.gt_root).resolve()),
                "parquet_dir": str(Path(args.parquet_dir).resolve()),
                "runs": args.runs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": 0,
                "pin_memory": args.pin_memory,
                "seed": args.seed,
                "max_batches": args.max_batches,
                "proc_sample_every": args.proc_sample_every,
                "cold_cache": args.cold_cache,
                "io_scan": args.io_scan,
                "expected_world_size": args.expected_world_size,
            },
            "environment": {
                "hostname": hostnames[0],
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "pyarrow": pa.__version__,
                "accelerate": accelerate_version,
                "world_size": world_size,
                "device_type": accelerator.device.type,
                "device_names": _all_gather_object(
                    (
                        torch.cuda.get_device_name(accelerator.device)
                        if accelerator.device.type == "cuda"
                        else "cpu"
                    ),
                    world_size,
                ),
            },
            "trials": [],
        }
    else:
        # Rank zero enters the collective above while constructing environment.
        _all_gather_object(
            (
                torch.cuda.get_device_name(accelerator.device)
                if accelerator.device.type == "cuda"
                else "cpu"
            ),
            world_size,
        )

    lifecycle_initial = _all_gather_object(_proc_snapshot(os.getpid()), world_size)
    internal_trials: list[dict[str, Any]] = []
    for run_index in range(args.runs):
        for order_index, backend in enumerate(_alternating_order(run_index)):
            local = _run_local_trial(
                backend,
                run_index,
                order_index,
                args,
                accelerator,
            )
            distributed = _all_gather_object(local, world_size)
            if rank == 0:
                trial = _aggregate_trial(distributed)
                internal_trials.append(trial)
                print(
                    json.dumps(
                        {
                            "run": run_index,
                            "backend": backend,
                            "init_s": trial["init_s"],
                            "first_batch_s": trial["first_batch_s"],
                            "steady_samples_per_s": trial["steady_samples_per_s"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if args.io_scan == "exact":
        local_io = measure_sample_io_amplification(
            args.parquet_dir,
            rank=rank,
            world_size=world_size,
        )
        io_records = _all_gather_object(local_io, world_size)
        io_result = _aggregate_io(io_records) if rank == 0 else None
    else:
        io_result = (
            {
                "status": "skipped",
                "method": None,
                "gate_ratio": None,
                "limitation": "--io-scan=skip disables the sample read gate",
            }
            if rank == 0
            else None
        )

    lifecycle_final = _all_gather_object(_proc_snapshot(os.getpid()), world_size)

    if rank != 0:
        return None
    assert base_result is not None
    base_result["trials"] = [_public_trial(trial) for trial in internal_trials]
    base_result["summary"] = summarize_trials(internal_trials)
    base_result["sample_io_amplification"] = io_result
    base_result["process_lifecycle"] = _process_lifecycle(
        lifecycle_initial, lifecycle_final, internal_trials
    )
    base_result["complete"] = True
    base_result["gate"] = evaluate_run(base_result)
    _write_json(output_path, base_result)
    return base_result


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"benchmark result must be a JSON object: {path}")
    return value


def evaluate_pair(
    one_rank: Mapping[str, Any], eight_rank: Mapping[str, Any]
) -> dict[str, Any]:
    contention_records: dict[str, dict[str, str]] = {}
    for label, result, expected_world_size in (
        ("one_rank", one_rank, 1),
        ("eight_rank", eight_rank, 8),
    ):
        if result.get("kind") != RESULT_KIND or result.get("version") != RESULT_VERSION:
            raise ValueError(f"{label} is not a supported benchmark result")
        if not result.get("complete"):
            raise ValueError(f"{label} benchmark result is incomplete")
        if int(result["environment"]["world_size"]) != expected_world_size:
            raise ValueError(
                f"{label} world size must be {expected_world_size}, got "
                f"{result['environment']['world_size']}"
            )
        contention_records[label] = _validate_contention(
            result.get("contention"),
            label=f"{label} result",
        )

    recomputed_one_rank = evaluate_run(one_rank)
    recomputed_eight_rank = evaluate_run(eight_rank)
    if one_rank.get("gate") != recomputed_one_rank:
        raise ValueError("one_rank stored gate differs from recomputed gate")
    if eight_rank.get("gate") != recomputed_eight_rank:
        raise ValueError("eight_rank stored gate differs from recomputed gate")

    compatibility_fields = (
        "render_root",
        "gt_root",
        "parquet_dir",
        "runs",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "seed",
        "max_batches",
        "proc_sample_every",
        "cold_cache",
        "io_scan",
    )
    mismatches = {
        field: {
            "one_rank": one_rank["config"].get(field),
            "eight_rank": eight_rank["config"].get(field),
        }
        for field in compatibility_fields
        if one_rank["config"].get(field) != eight_rank["config"].get(field)
    }
    if one_rank["bundle_id"] != eight_rank["bundle_id"]:
        mismatches["bundle_id"] = {
            "one_rank": one_rank["bundle_id"],
            "eight_rank": eight_rank["bundle_id"],
        }
    for field in (
        "hostname",
        "platform",
        "python",
        "torch",
        "pyarrow",
        "accelerate",
        "device_type",
    ):
        if one_rank["environment"].get(field) != eight_rank["environment"].get(field):
            mismatches[f"environment.{field}"] = {
                "one_rank": one_rank["environment"].get(field),
                "eight_rank": eight_rank["environment"].get(field),
            }
    one_device_names = set(one_rank["environment"].get("device_names", []))
    eight_device_names = set(eight_rank["environment"].get("device_names", []))
    if one_device_names != eight_device_names:
        mismatches["environment.device_names"] = {
            "one_rank": sorted(one_device_names),
            "eight_rank": sorted(eight_device_names),
        }

    checks = {
        "matching_inputs_and_settings": _gate(
            "pass" if not mismatches else "fail",
            actual=mismatches,
            requirement="1-rank and 8-rank results use the same bundle and settings",
        ),
        "one_rank_gate": _gate(
            str(one_rank["gate"]["status"]),
            actual=one_rank["gate"]["decision"],
            requirement="1-rank performance gate passes",
        ),
        "eight_rank_gate": _gate(
            str(eight_rank["gate"]["status"]),
            actual=eight_rank["gate"]["decision"],
            requirement="8-rank performance gate passes; single-rank alone cannot GO",
        ),
    }
    status = _overall_status(checks)
    return {
        "kind": COMBINED_RESULT_KIND,
        "version": RESULT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": one_rank["bundle_id"],
        "contention": contention_records,
        "status": status,
        "decision": {
            "pass": "go",
            "fail": "no-go",
            "indeterminate": "review-required",
        }[status],
        "checks": checks,
        "inputs": {
            "one_rank_world_size": one_rank["environment"]["world_size"],
            "eight_rank_world_size": eight_rank["environment"]["world_size"],
            "one_rank_gate": one_rank["gate"],
            "eight_rank_gate": eight_rank["gate"],
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="run one distributed world-size benchmark"
    )
    run_parser.add_argument("--render-root", required=True)
    run_parser.add_argument("--gt-root", required=True)
    run_parser.add_argument("--parquet-dir", required=True)
    run_parser.add_argument("--output-json", required=True)
    run_parser.add_argument(
        "--expected-world-size", type=int, choices=(1, 8), required=True
    )
    run_parser.add_argument(
        "--contention-status",
        choices=CONTENTION_STATUSES,
        default="unknown",
        help="operator-declared resource contention during this benchmark",
    )
    run_parser.add_argument(
        "--contention-note",
        default=DEFAULT_CONTENTION_NOTE,
        help="non-empty explanation preserved in run and combined result JSON",
    )
    run_parser.add_argument("--runs", type=_positive_int, default=5)
    run_parser.add_argument("--batch-size", type=_positive_int, default=16)
    run_parser.add_argument("--num-workers", type=_nonnegative_int, default=4)
    run_parser.add_argument("--prefetch-factor", type=_positive_int, default=2)
    run_parser.add_argument("--seed", type=_nonnegative_int, default=0)
    run_parser.add_argument(
        "--max-batches",
        type=_nonnegative_int,
        default=0,
        help="diagnostic cap; 0 is the required full pass",
    )
    run_parser.add_argument("--proc-sample-every", type=_positive_int, default=16)
    run_parser.add_argument(
        "--cold-cache",
        choices=("none", "linux-drop-caches"),
        default="none",
    )
    run_parser.add_argument("--io-scan", choices=("exact", "skip"), default="exact")
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--pin-memory", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="combine the required 1-rank and 8-rank JSON results"
    )
    evaluate_parser.add_argument("--one-rank-json", required=True)
    evaluate_parser.add_argument("--eight-rank-json", required=True)
    evaluate_parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "run":
        result = run_benchmark(args)
        if result is None:
            return 0
        print(json.dumps(result["gate"], indent=2, sort_keys=True), flush=True)
        return 0 if result["gate"]["status"] == "pass" else 2
    if args.command == "evaluate":
        evaluation = evaluate_pair(
            _load_json(args.one_rank_json),
            _load_json(args.eight_rank_json),
        )
        _write_json(Path(args.output_json), evaluation)
        print(json.dumps(evaluation, indent=2, sort_keys=True), flush=True)
        return 0 if evaluation["decision"] == "go" else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
