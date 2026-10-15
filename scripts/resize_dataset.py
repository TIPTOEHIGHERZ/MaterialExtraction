#!/usr/bin/env python3
r"""Resize texture datasets while preserving their directory layout.

Examples:
    # MatSynth_standard unpacked layout.
    python scripts/resize_dataset.py \
        --src datasets/Matsynth_standard/unpacked \
        --out datasets/Matsynth_standard/unpacked_512 \
        --size 512 \
        --layout sample-dirs \
        --include basecolor normal roughness height input \
        --copy-extra metadata.yaml label.json

    # Generic recursive image tree.
    python scripts/resize_dataset.py --src datasets/raw --out datasets/raw_512 --size 512

    # Legacy PBRTextureDataLoader ground-truth resizing.
    python scripts/resize_dataset.py \
        --layout pbr-loader \
        --render-fp datasets/render_polyhaven_train_50_resized \
        --src datasets/MatSynth/polyhaven_train_render \
        --out datasets/MatSynth/polyhaven_train_render_resized \
        --gt-mapping diff_4k=Color nor_gl_4k=NormalGL rough_4k=Roughness disp_4k=Height
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.append(os.getcwd())


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_FILE_TYPES = ("Color", "NormalGL", "Roughness", "Height")


@dataclass(frozen=True)
class ResizeConfig:
    size: tuple[int, int]
    crop_mode: str
    interpolation: str
    overwrite: bool
    dry_run: bool
    jpeg_quality: int


@dataclass(frozen=True)
class ResizeTask:
    src: Path
    dst: Path
    task_type: str


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = [part for part in normalized.split("x") if part]

    if len(parts) == 1:
        width = height = int(parts[0])
    elif len(parts) == 2:
        width, height = (int(part) for part in parts)
    else:
        raise argparse.ArgumentTypeError("size must be N or WIDTHxHEIGHT")

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def parse_mapping(values: Iterable[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"mapping must be KEY=VALUE, got {value}")
        key, mapped = value.split("=", 1)
        key = key.strip()
        mapped = mapped.strip()
        if not key or not mapped:
            raise argparse.ArgumentTypeError(f"mapping must be KEY=VALUE, got {value}")
        mapping[key] = mapped
    return mapping


def normalize_names(values: Iterable[str] | None) -> set[str]:
    return {value.lower() for value in values or ()}


def normalize_extensions(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value.lower() if value.startswith(".") else f".{value.lower()}" for value in values)


def normalize_copy_extra(values: Iterable[str] | None) -> set[str]:
    names = set()
    for value in values or ():
        normalized = value.lower()
        names.add(normalized)
        if "." not in normalized:
            names.add(f".{normalized}")
    return names


def name_selected(path: Path, include: set[str], exclude: set[str]) -> bool:
    candidates = {path.name.lower(), path.stem.lower()}
    if include and candidates.isdisjoint(include):
        return False
    return candidates.isdisjoint(exclude)


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def resample_filter(name: str) -> int:
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "hamming": Image.Resampling.HAMMING,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return filters[name]


def resize_image(src: Path, dst: Path, config: ResizeConfig) -> str:
    if dst.exists() and not config.overwrite:
        return "skipped"
    if config.dry_run:
        return "would_write"

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as image:
            image.load()
            if config.crop_mode == "center-square":
                image = center_crop_square(image)
            image = image.resize(config.size, resample=resample_filter(config.interpolation))

            save_kwargs = {}
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                save_kwargs.update({"quality": config.jpeg_quality, "subsampling": 0})

            image.save(dst, **save_kwargs)
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"failed to read image: {src}") from exc
    return "written"


def copy_file(src: Path, dst: Path, config: ResizeConfig) -> str:
    if dst.exists() and not config.overwrite:
        return "skipped"
    if config.dry_run:
        return "would_copy"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def run_task(args: tuple[ResizeTask, ResizeConfig]) -> tuple[str, str, str]:
    task, config = args
    if task.task_type == "image":
        status = resize_image(task.src, task.dst, config)
    elif task.task_type == "copy":
        status = copy_file(task.src, task.dst, config)
    else:
        raise ValueError(f"unknown task type: {task.task_type}")
    return status, str(task.src), str(task.dst)


def relative_to_src(path: Path, src_root: Path) -> Path:
    try:
        return path.relative_to(src_root)
    except ValueError:
        return Path(path.parent.name) / path.name


def iter_recursive_tasks(
    src_root: Path,
    out_root: Path,
    include: set[str],
    exclude: set[str],
    image_extensions: tuple[str, ...],
    copy_extra: set[str],
) -> list[ResizeTask]:
    tasks = []
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = relative_to_src(path, src_root)
        suffix = path.suffix.lower()
        if suffix in image_extensions and name_selected(path, include, exclude):
            tasks.append(ResizeTask(path, out_root / rel_path, "image"))
        elif path.name.lower() in copy_extra or suffix in copy_extra:
            tasks.append(ResizeTask(path, out_root / rel_path, "copy"))
    return tasks


def iter_sample_dir_tasks(
    src_root: Path,
    out_root: Path,
    include: set[str],
    exclude: set[str],
    image_extensions: tuple[str, ...],
    copy_extra: set[str],
) -> list[ResizeTask]:
    tasks = []
    sample_dirs = [path for path in sorted(src_root.iterdir()) if path.is_dir()]
    for sample_dir in sample_dirs:
        for path in sorted(sample_dir.iterdir()):
            if not path.is_file():
                continue
            rel_path = Path(sample_dir.name) / path.name
            suffix = path.suffix.lower()
            if suffix in image_extensions and name_selected(path, include, exclude):
                tasks.append(ResizeTask(path, out_root / rel_path, "image"))
            elif path.name.lower() in copy_extra or suffix in copy_extra:
                tasks.append(ResizeTask(path, out_root / rel_path, "copy"))
    return tasks


def iter_pbr_loader_tasks(args: argparse.Namespace, out_root: Path) -> list[ResizeTask]:
    from utils.datasets.DataLoader import PBRTextureDataLoader

    file_types = tuple(args.file_types)
    PBRTextureDataLoader.file_type = file_types
    loader = PBRTextureDataLoader(
        fp=str(args.render_fp),
        gt_fp=str(args.src),
        gt_mapping=parse_mapping(args.gt_mapping),
    )

    tasks = []
    seen = set()
    for file_dict in loader.gt_files:
        for attr_name in file_types:
            path_value = file_dict.get(attr_name)
            if path_value is None:
                continue
            src_path = Path(path_value)
            rel_path = relative_to_src(src_path, args.src)
            task_key = (src_path, out_root / rel_path)
            if task_key in seen:
                continue
            seen.add(task_key)
            tasks.append(ResizeTask(src_path, out_root / rel_path, "image"))
    return tasks


def build_tasks(args: argparse.Namespace) -> list[ResizeTask]:
    src_root = args.src.resolve()
    out_root = args.out.resolve()
    include = normalize_names(args.include)
    exclude = normalize_names(args.exclude)
    image_extensions = normalize_extensions(args.extensions)
    copy_extra = normalize_copy_extra(args.copy_extra)

    if args.layout == "recursive":
        tasks = iter_recursive_tasks(src_root, out_root, include, exclude, image_extensions, copy_extra)
    elif args.layout == "sample-dirs":
        tasks = iter_sample_dir_tasks(src_root, out_root, include, exclude, image_extensions, copy_extra)
    elif args.layout == "pbr-loader":
        if args.render_fp is None:
            raise ValueError("--render-fp is required when --layout pbr-loader")
        tasks = iter_pbr_loader_tasks(args, out_root)
    else:
        raise ValueError(f"unknown layout: {args.layout}")

    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def process_tasks(tasks: list[ResizeTask], config: ResizeConfig, num_workers: int) -> dict[str, int]:
    summary: dict[str, int] = {}
    work_items = [(task, config) for task in tasks]

    if num_workers <= 1:
        results = (run_task(item) for item in work_items)
        iterator = tqdm(results, total=len(tasks), desc="resize")
        for status, _, _ in iterator:
            summary[status] = summary.get(status, 0) + 1
        return summary

    with Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(run_task, work_items)
        for status, _, _ in tqdm(iterator, total=len(tasks), desc="resize"):
            summary[status] = summary.get(status, 0) + 1
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--layout",
        choices=("recursive", "sample-dirs", "pbr-loader"),
        default="recursive",
        help="Input layout. Default: recursive.",
    )
    parser.add_argument("--size", type=parse_size, default=parse_size("512"), help="Output size, N or WIDTHxHEIGHT.")
    parser.add_argument(
        "--crop-mode",
        choices=("center-square", "none"),
        default="center-square",
        help="Crop before resizing. Default matches the previous script behavior.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"),
        default="lanczos",
        help="Pillow resize filter.",
    )
    parser.add_argument("--extensions", nargs="+", default=IMAGE_EXTENSIONS, help="Image extensions to process.")
    parser.add_argument("--include", nargs="*", default=(), help="Only process matching filenames or stems.")
    parser.add_argument("--exclude", nargs="*", default=(), help="Skip matching filenames or stems.")
    parser.add_argument("--copy-extra", nargs="*", default=(), help="Extra filenames or extensions to copy unchanged.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tasks.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes. Default: 1.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for resized JPEG outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Build tasks and print the summary without writing files.")

    parser.add_argument("--render-fp", type=Path, default=None, help="Render dataset root for --layout pbr-loader.")
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=DEFAULT_FILE_TYPES,
        help="PBR attributes to process for --layout pbr-loader.",
    )
    parser.add_argument(
        "--gt-mapping",
        nargs="*",
        default=(),
        help="Ground-truth filename mapping for --layout pbr-loader, e.g. diff_4k=Color.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if not args.src.exists():
        raise FileNotFoundError(f"source path does not exist: {args.src}")
    if args.src.resolve() == args.out.resolve():
        raise ValueError("--src and --out must be different directories")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    tasks = build_tasks(args)
    config = ResizeConfig(
        size=args.size,
        crop_mode=args.crop_mode,
        interpolation=args.interpolation,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"layout: {args.layout}")
    print(f"source: {args.src}")
    print(f"output: {args.out}")
    print(f"size: {args.size[0]}x{args.size[1]}")
    print(f"tasks: {len(tasks)}")

    summary = process_tasks(tasks, config, args.num_workers)
    print("summary:")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
