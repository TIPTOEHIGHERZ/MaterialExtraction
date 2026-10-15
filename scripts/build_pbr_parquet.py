#!/usr/bin/env python3
"""Build and strictly verify self-contained PBR Parquet bundles.

The CLI intentionally exposes only source roots, an immutable output directory,
and a locked validation profile. Physical row-group/shard defaults are owned by
the profile and cannot drift between launches.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, UnidentifiedImageError

sys.path.append(os.getcwd())

from utils.datasets.pbr_parquet import (  # noqa: E402
    FORMAT_VERSION,
    HARD_SHARD_BYTES,
    JPEG_MAGIC,
    MARKER_NAME,
    MATERIALS_TABLE,
    PNG_MAGIC,
    REQUIRED_PYARROW_VERSION,
    SAMPLES_TABLE,
    SORT_ORDERS,
    TABLE_TYPES,
    TARGET_SHARD_BYTES,
    WRITER_NAME,
    WRITER_VERSION,
    IntegrityError,
    MarkerError,
    PBRParquetError,
    PublicationError,
    ShardError,
    SourceDataError,
    _validate_column_chunks,
    canonical_json_bytes,
    inspect_bundle,
    parquet_schema,
    schema_fingerprint,
    sha256_file,
    validate_parquet_schema,
    validate_success_marker,
)

_GIB = 1024 * 1024 * 1024
_REQUIRED_VIEW_FILES = ("image.png", "mask.png", "meta_data.yaml")
_GT_FILES = {
    "color_jpeg": "basecolor.jpg",
    "normal_gl_png": "normal.png",
    "height_png": "height.png",
    "roughness_png": "roughness.png",
}
_SAMPLE_FIELDS = (
    "sample_id",
    "material_id",
    "view_id",
    "image_png",
    "mask_png",
    "theta",
    "phi",
    "radius",
    "camera_direction",
    "head_direction",
)
_MATERIAL_FIELDS = (
    "material_id",
    "color_jpeg",
    "normal_gl_png",
    "height_png",
    "roughness_png",
)
_METADATA_FIELDS = (
    "theta",
    "phi",
    "radius",
    "camera_direction",
    "head_direction",
)
_VALIDATION_CHECKS = [
    "source_scan",
    "schema",
    "footer",
    "hashes",
    "logical_digest",
    "codec_decode",
]


@dataclass(frozen=True)
class BuildProfile:
    name: str
    logical_dataset_id: str
    expected_materials: int
    views_per_material: int
    target_shard_bytes: int
    hard_shard_bytes: int

    def __post_init__(self) -> None:
        if self.expected_materials < 1:
            raise ValueError("expected_materials must be positive")
        if self.views_per_material < 1:
            raise ValueError("views_per_material must be positive")
        if self.target_shard_bytes < 1:
            raise ValueError("target_shard_bytes must be positive")
        if self.hard_shard_bytes < self.target_shard_bytes:
            raise ValueError("hard_shard_bytes must be >= target_shard_bytes")


PROFILES: dict[str, BuildProfile] = {
    "matsynth-50": BuildProfile(
        name="matsynth-50",
        logical_dataset_id="matsynth_render_result_50_resized_v1",
        expected_materials=5_789,
        views_per_material=50,
        target_shard_bytes=TARGET_SHARD_BYTES,
        hard_shard_bytes=HARD_SHARD_BYTES,
    ),
    "matsynth-50-subset": BuildProfile(
        name="matsynth-50-subset",
        logical_dataset_id="matsynth_render_result_50_resized_subset_v1",
        expected_materials=1,
        views_per_material=50,
        target_shard_bytes=TARGET_SHARD_BYTES,
        hard_shard_bytes=HARD_SHARD_BYTES,
    ),
    "tiny": BuildProfile(
        name="tiny",
        logical_dataset_id="tex_edit_pbr_tiny_v1",
        expected_materials=3,
        views_per_material=3,
        target_shard_bytes=64 * 1024,
        hard_shard_bytes=4 * 1024 * 1024,
    ),
}


@dataclass(frozen=True)
class SourceView:
    view_id: int
    image_path: Path
    mask_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class SourceMaterial:
    material_id: str
    views: tuple[SourceView, ...]
    gt_paths: Mapping[str, Path]


@dataclass
class SourceScan:
    render_root: Path
    gt_root: Path
    materials: tuple[SourceMaterial, ...]
    directory_tokens: dict[Path, tuple[int, int, int, int, int]]
    ignored_render_root_files: tuple[str, ...]
    ignored_gt_root_files: tuple[str, ...]
    extra_gt_materials: tuple[str, ...]
    ignored_gt_attachments: int

    def assert_unchanged(self) -> None:
        for path, expected in self.directory_tokens.items():
            actual = _directory_token(path)
            if actual != expected:
                raise SourceDataError(f"source directory changed after scan: {path}")


class LogicalContentDigest:
    """Stable v1 row encoder, independent of Arrow/Parquet serialization."""

    def __init__(self, table_type: str):
        if table_type not in TABLE_TYPES:
            raise ValueError(f"unknown table type: {table_type!r}")
        self.table_type = table_type
        self._digest = hashlib.sha256()
        self._update_variable(table_type.encode("utf-8"))

    def _update_variable(self, value: bytes) -> None:
        self._digest.update(struct.pack("<Q", len(value)))
        self._digest.update(value)

    def update(self, row: Mapping[str, Any]) -> None:
        fields = (
            _SAMPLE_FIELDS if self.table_type == SAMPLES_TABLE else _MATERIAL_FIELDS
        )
        for field in fields:
            value = row[field]
            if field in {"sample_id", "material_id"}:
                if not isinstance(value, str):
                    raise IntegrityError(f"{field} must be a string")
                self._update_variable(value.encode("utf-8"))
            elif field in {
                "image_png",
                "mask_png",
                "color_jpeg",
                "normal_gl_png",
                "height_png",
                "roughness_png",
            }:
                if not isinstance(value, (bytes, bytearray, memoryview)):
                    raise IntegrityError(f"{field} must be binary")
                self._update_variable(bytes(value))
            elif field == "view_id":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise IntegrityError("view_id must be an integer")
                self._digest.update(struct.pack("<h", value))
            elif field in {"theta", "phi", "radius"}:
                number = _finite_float(value, field, IntegrityError)
                self._digest.update(struct.pack("<d", number))
            elif field in {"camera_direction", "head_direction"}:
                vector = _finite_vector(value, field, IntegrityError)
                for number in vector:
                    self._digest.update(struct.pack("<d", number))
            else:  # pragma: no cover - guarded by the fixed v1 field list.
                raise AssertionError(field)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class SourceDigest:
    """Digest exact consumed source bytes under logical, non-absolute names."""

    def __init__(self, logical_dataset_id: str):
        self._digest = hashlib.sha256()
        self._update_variable(logical_dataset_id.encode("utf-8"))

    def _update_variable(self, value: bytes) -> None:
        self._digest.update(struct.pack("<Q", len(value)))
        self._digest.update(value)

    def update(self, logical_path: str, payload: bytes) -> None:
        self._update_variable(logical_path.encode("utf-8"))
        self._update_variable(payload)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _profile(profile: str | BuildProfile) -> BuildProfile:
    if isinstance(profile, BuildProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(PROFILES)}"
        ) from exc


def _require_pyarrow_version() -> None:
    if pa.__version__ != REQUIRED_PYARROW_VERSION:
        raise PBRParquetError(
            f"PyArrow {REQUIRED_PYARROW_VERSION} is required, "
            f"found {pa.__version__}"
        )


def _path_token(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_token(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SourceDataError(f"cannot stat source directory: {path}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise SourceDataError(f"source path is not a regular directory: {path}")
    return _path_token(info)


def _scan_directory(
    path: Path,
    tokens: dict[Path, tuple[int, int, int, int, int]],
) -> list[os.DirEntry[str]]:
    before = _directory_token(path)
    try:
        with os.scandir(path) as iterator:
            entries = list(iterator)
    except OSError as exc:
        raise SourceDataError(f"cannot scan source directory: {path}") from exc
    after = _directory_token(path)
    if before != after:
        raise SourceDataError(f"source directory changed while scanning: {path}")
    tokens[path] = after
    return entries


def _validate_material_id(material_id: str) -> None:
    if (
        not material_id
        or material_id in {".", ".."}
        or "/" in material_id
        or "\x00" in material_id
    ):
        raise SourceDataError(f"invalid material directory name: {material_id!r}")


def scan_sources(
    render_root: str | Path,
    gt_root: str | Path,
    *,
    profile: str | BuildProfile,
) -> SourceScan:
    selected_profile = _profile(profile)
    render_root = Path(render_root)
    gt_root = Path(gt_root)
    tokens: dict[Path, tuple[int, int, int, int, int]] = {}

    render_entries = _scan_directory(render_root, tokens)
    render_dirs: dict[str, Path] = {}
    ignored_render_files: list[str] = []
    for entry in render_entries:
        if entry.is_symlink():
            raise SourceDataError(f"source symlink is not allowed: {entry.path}")
        if entry.is_dir(follow_symlinks=False):
            _validate_material_id(entry.name)
            render_dirs[entry.name] = Path(entry.path)
        elif entry.is_file(follow_symlinks=False):
            ignored_render_files.append(entry.name)
        else:
            raise SourceDataError(f"unsupported render root entry: {entry.path}")

    if len(render_dirs) != selected_profile.expected_materials:
        raise SourceDataError(
            f"profile {selected_profile.name} expects "
            f"{selected_profile.expected_materials} materials, "
            f"found {len(render_dirs)}"
        )

    gt_entries = _scan_directory(gt_root, tokens)
    gt_dirs: dict[str, Path] = {}
    ignored_gt_files: list[str] = []
    for entry in gt_entries:
        if entry.is_symlink():
            raise SourceDataError(f"source symlink is not allowed: {entry.path}")
        if entry.is_dir(follow_symlinks=False):
            _validate_material_id(entry.name)
            gt_dirs[entry.name] = Path(entry.path)
        elif entry.is_file(follow_symlinks=False):
            ignored_gt_files.append(entry.name)
        else:
            raise SourceDataError(f"unsupported GT root entry: {entry.path}")

    missing_gt = sorted(set(render_dirs) - set(gt_dirs))
    if missing_gt:
        raise SourceDataError(
            f"render materials missing GT directories: {missing_gt[:10]}"
        )
    extra_gt = tuple(sorted(set(gt_dirs) - set(render_dirs)))

    expected_view_names = {
        str(view_id) for view_id in range(selected_profile.views_per_material)
    }
    materials: list[SourceMaterial] = []
    ignored_gt_attachments = 0
    for material_id in sorted(render_dirs):
        render_material = render_dirs[material_id]
        view_entries = _scan_directory(render_material, tokens)
        actual_view_names: set[str] = set()
        views_by_id: dict[int, SourceView] = {}
        for entry in view_entries:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise SourceDataError(
                    f"render material entries must be view directories: {entry.path}"
                )
            try:
                view_id = int(entry.name)
            except ValueError as exc:
                raise SourceDataError(
                    f"view directory is not numeric: {entry.path}"
                ) from exc
            if entry.name != str(view_id):
                raise SourceDataError(
                    f"view directory is not canonical decimal: {entry.path}"
                )
            actual_view_names.add(entry.name)
            view_path = Path(entry.path)
            files = _scan_directory(view_path, tokens)
            files_by_name: dict[str, Path] = {}
            for file_entry in files:
                if file_entry.is_symlink() or not file_entry.is_file(
                    follow_symlinks=False
                ):
                    raise SourceDataError(
                        f"view entries must be regular files: {file_entry.path}"
                    )
                files_by_name[file_entry.name] = Path(file_entry.path)
            if set(files_by_name) != set(_REQUIRED_VIEW_FILES):
                raise SourceDataError(
                    f"{view_path} files mismatch: "
                    f"missing={sorted(set(_REQUIRED_VIEW_FILES) - set(files_by_name))}, "
                    f"extra={sorted(set(files_by_name) - set(_REQUIRED_VIEW_FILES))}"
                )
            views_by_id[view_id] = SourceView(
                view_id=view_id,
                image_path=files_by_name["image.png"],
                mask_path=files_by_name["mask.png"],
                metadata_path=files_by_name["meta_data.yaml"],
            )
        if actual_view_names != expected_view_names:
            raise SourceDataError(
                f"{render_material} view set mismatch: "
                f"missing={sorted(expected_view_names - actual_view_names)}, "
                f"extra={sorted(actual_view_names - expected_view_names)}"
            )

        gt_material = gt_dirs[material_id]
        gt_entries_for_material = _scan_directory(gt_material, tokens)
        gt_by_name: dict[str, Path] = {}
        for entry in gt_entries_for_material:
            if entry.is_symlink():
                raise SourceDataError(f"source symlink is not allowed: {entry.path}")
            if entry.is_file(follow_symlinks=False):
                gt_by_name[entry.name] = Path(entry.path)
            elif entry.is_dir(follow_symlinks=False):
                ignored_gt_attachments += 1
            else:
                raise SourceDataError(f"unsupported GT entry: {entry.path}")
        missing_files = sorted(set(_GT_FILES.values()) - set(gt_by_name))
        if missing_files:
            raise SourceDataError(f"{gt_material} is missing GT files: {missing_files}")
        ignored_gt_attachments += len(set(gt_by_name) - set(_GT_FILES.values()))
        materials.append(
            SourceMaterial(
                material_id=material_id,
                views=tuple(
                    views_by_id[view_id]
                    for view_id in range(selected_profile.views_per_material)
                ),
                gt_paths={
                    field: gt_by_name[filename] for field, filename in _GT_FILES.items()
                },
            )
        )

    return SourceScan(
        render_root=render_root,
        gt_root=gt_root,
        materials=tuple(materials),
        directory_tokens=tokens,
        ignored_render_root_files=tuple(sorted(ignored_render_files)),
        ignored_gt_root_files=tuple(sorted(ignored_gt_files)),
        extra_gt_materials=extra_gt,
        ignored_gt_attachments=ignored_gt_attachments,
    )


def _read_stable_file(path: Path) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SourceDataError(f"cannot stat source file: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise SourceDataError(f"source payload is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            payload = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SourceDataError(f"cannot read source file: {path}") from exc
    expected = _path_token(before)
    if (
        _path_token(opened_before) != expected
        or _path_token(opened_after) != expected
        or _path_token(after) != expected
    ):
        raise SourceDataError(f"source file changed while reading: {path}")
    if not payload:
        raise SourceDataError(f"source file is empty: {path}")
    return payload


def _validate_magic(payload: bytes, codec: str, label: str) -> None:
    if codec == "png":
        if not payload.startswith(PNG_MAGIC):
            raise SourceDataError(f"{label} is not PNG encoded")
    elif codec == "jpeg":
        if not payload.startswith(JPEG_MAGIC) or not payload.endswith(b"\xff\xd9"):
            raise SourceDataError(f"{label} is not JPEG encoded")
    else:  # pragma: no cover - fixed format contract.
        raise AssertionError(codec)


def _finite_float(
    value: Any,
    label: str,
    error_type: type[Exception] = SourceDataError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise error_type(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise error_type(f"{label} must be finite")
    return result


def _finite_vector(
    value: Any,
    label: str,
    error_type: type[Exception] = SourceDataError,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise error_type(f"{label} must contain exactly three numbers")
    return [
        _finite_float(number, f"{label}[{index}]", error_type)
        for index, number in enumerate(value)
    ]


def _parse_metadata(payload: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
        raw = yaml.safe_load(decoded)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SourceDataError(f"{label} is not valid UTF-8 YAML") from exc
    if not isinstance(raw, Mapping):
        raise SourceDataError(f"{label} must contain a YAML mapping")
    if set(raw) != set(_METADATA_FIELDS):
        raise SourceDataError(
            f"{label} metadata keys mismatch: "
            f"missing={sorted(set(_METADATA_FIELDS) - set(raw))}, "
            f"extra={sorted(set(raw) - set(_METADATA_FIELDS))}"
        )
    return {
        "theta": _finite_float(raw["theta"], f"{label}.theta"),
        "phi": _finite_float(raw["phi"], f"{label}.phi"),
        "radius": _finite_float(raw["radius"], f"{label}.radius"),
        "camera_direction": _finite_vector(
            raw["camera_direction"], f"{label}.camera_direction"
        ),
        "head_direction": _finite_vector(
            raw["head_direction"], f"{label}.head_direction"
        ),
    }


def _read_material(
    source: SourceMaterial,
    source_digest: SourceDigest,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    material_id = source.material_id
    material_row: dict[str, Any] = {"material_id": material_id}
    material_payload_bytes = 0
    for field in _MATERIAL_FIELDS[1:]:
        payload = _read_stable_file(source.gt_paths[field])
        codec = "jpeg" if field == "color_jpeg" else "png"
        _validate_magic(payload, codec, f"{material_id}.{field}")
        material_row[field] = payload
        material_payload_bytes += len(payload)
        source_digest.update(f"materials/{material_id}/{_GT_FILES[field]}", payload)

    sample_rows: list[dict[str, Any]] = []
    sample_payload_bytes = 0
    for view in source.views:
        image = _read_stable_file(view.image_path)
        mask = _read_stable_file(view.mask_path)
        metadata_payload = _read_stable_file(view.metadata_path)
        _validate_magic(image, "png", f"{material_id}/{view.view_id}.image")
        _validate_magic(mask, "png", f"{material_id}/{view.view_id}.mask")
        metadata = _parse_metadata(
            metadata_payload, f"{material_id}/{view.view_id}.meta_data"
        )
        source_digest.update(f"samples/{material_id}/{view.view_id}/image.png", image)
        source_digest.update(f"samples/{material_id}/{view.view_id}/mask.png", mask)
        source_digest.update(
            f"samples/{material_id}/{view.view_id}/meta_data.yaml",
            metadata_payload,
        )
        sample_rows.append(
            {
                "sample_id": f"{material_id}/{view.view_id}",
                "material_id": material_id,
                "view_id": view.view_id,
                "image_png": image,
                "mask_png": mask,
                **metadata,
            }
        )
        sample_payload_bytes += len(image) + len(mask)
    return (
        material_row,
        sample_rows,
        material_payload_bytes,
        sample_payload_bytes,
    )


def _compression(table_type: str) -> dict[str, str]:
    if table_type == SAMPLES_TABLE:
        return {
            "sample_id": "ZSTD",
            "material_id": "ZSTD",
            "view_id": "ZSTD",
            "image_png": "NONE",
            "mask_png": "NONE",
            "theta": "ZSTD",
            "phi": "ZSTD",
            "radius": "ZSTD",
            "camera_direction.list.element": "ZSTD",
            "head_direction.list.element": "ZSTD",
        }
    return {
        "material_id": "ZSTD",
        "color_jpeg": "NONE",
        "normal_gl_png": "NONE",
        "height_png": "NONE",
        "roughness_png": "NONE",
    }


def _statistics(table_type: str) -> list[str]:
    return (
        ["material_id", "view_id"] if table_type == SAMPLES_TABLE else ["material_id"]
    )


class StreamingShardWriter:
    def __init__(
        self,
        root: Path,
        table_type: str,
        bundle_id: str,
        profile: BuildProfile,
    ):
        self.root = root
        self.table_type = table_type
        self.schema = parquet_schema(table_type, bundle_id)
        self.profile = profile
        self.records: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self._path: Path | None = None
        self._payload_bytes = 0
        self._rows = 0
        self._row_groups = 0
        self._first_material_id: str | None = None
        self._last_material_id: str | None = None

    def _open(self) -> None:
        ordinal = len(self.records)
        self._path = self.root / self.table_type / f"part-{ordinal:05d}.parquet"
        self._writer = pq.ParquetWriter(
            self._path,
            self.schema,
            version="2.6",
            compression=_compression(self.table_type),
            use_dictionary=(
                ["material_id"] if self.table_type == SAMPLES_TABLE else False
            ),
            write_statistics=_statistics(self.table_type),
        )

    def write_material(
        self,
        material_id: str,
        rows: list[dict[str, Any]],
        payload_bytes: int,
    ) -> None:
        if (
            self._writer is not None
            and self._payload_bytes > 0
            and self._payload_bytes + payload_bytes > self.profile.target_shard_bytes
        ):
            self._close_current()
        if self._writer is None:
            self._open()
        assert self._writer is not None
        table = pa.Table.from_pylist(rows, schema=self.schema)
        self._writer.write_table(table, row_group_size=len(rows))
        self._payload_bytes += payload_bytes
        self._rows += len(rows)
        self._row_groups += 1
        self._first_material_id = self._first_material_id or material_id
        self._last_material_id = material_id

    def _close_current(self) -> None:
        if self._writer is None:
            return
        writer = self._writer
        path = self._path
        self._writer = None
        self._path = None
        writer.close()
        assert path is not None
        size = path.stat().st_size
        if size > self.profile.hard_shard_bytes:
            raise ShardError(
                f"{path} is {size} bytes, above profile hard limit "
                f"{self.profile.hard_shard_bytes}"
            )
        _fsync_file(path)
        ordinal = len(self.records)
        assert self._first_material_id is not None
        assert self._last_material_id is not None
        self.records.append(
            {
                "path": path.relative_to(self.root).as_posix(),
                "table_type": self.table_type,
                "ordinal": ordinal,
                "rows": self._rows,
                "row_groups": self._row_groups,
                "bytes": size,
                "sha256": sha256_file(path),
                "first_material_id": self._first_material_id,
                "last_material_id": self._last_material_id,
            }
        )
        self._payload_bytes = 0
        self._rows = 0
        self._row_groups = 0
        self._first_material_id = None
        self._last_material_id = None

    def finish(self) -> list[dict[str, Any]]:
        self._close_current()
        if not self.records:
            raise ShardError(f"no {self.table_type} shards were written")
        return list(self.records)

    def abort(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
                self._path = None


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_payload(payload: bytes, field: str, label: str) -> None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            array = np.asarray(image)
            if image.width < 1 or image.height < 1:
                raise IntegrityError(f"{label} has an empty image")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise IntegrityError(f"{label} cannot be decoded") from exc
    expected_dtype = np.uint16 if field == "height_png" else np.uint8
    if array.dtype != expected_dtype:
        raise IntegrityError(f"{label} decoded dtype {array.dtype} != {expected_dtype}")


def _row_from_table(table: pa.Table, row_index: int) -> dict[str, Any]:
    return {name: table[name][row_index].as_py() for name in table.schema.names}


def _validate_sample_row(
    row: Mapping[str, Any], material_id: str, view_id: int
) -> None:
    if row["material_id"] != material_id:
        raise IntegrityError("sample row group spans materials")
    if row["view_id"] != view_id:
        raise IntegrityError(f"{material_id} views are not exactly ordered 0..N-1")
    if row["sample_id"] != f"{material_id}/{view_id}":
        raise IntegrityError(f"invalid sample_id: {row['sample_id']!r}")
    _validate_magic(row["image_png"], "png", f"{row['sample_id']}.image_png")
    _validate_magic(row["mask_png"], "png", f"{row['sample_id']}.mask_png")
    for field in ("theta", "phi", "radius"):
        _finite_float(row[field], field, IntegrityError)
    for field in ("camera_direction", "head_direction"):
        _finite_vector(row[field], field, IntegrityError)


def _validate_material_row(row: Mapping[str, Any], material_id: str) -> None:
    if row["material_id"] != material_id:
        raise IntegrityError("material row content conflicts with ordering")
    _validate_magic(row["color_jpeg"], "jpeg", f"{material_id}.color_jpeg")
    for field in ("normal_gl_png", "height_png", "roughness_png"):
        _validate_magic(row[field], "png", f"{material_id}.{field}")


def _decode_footer_material(statistic: Any, label: str) -> str:
    value = statistic
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ShardError(f"{label} is not UTF-8") from exc
    if not isinstance(value, str) or not value:
        raise ShardError(f"{label} is not a material string")
    return value


def _verify_content(
    bundle_dir: Path,
    marker: Mapping[str, Any],
    shards: Iterable[Mapping[str, Any]],
    profile: BuildProfile,
    *,
    require_marker: bool,
    verify_hashes: bool,
) -> dict[str, Any]:
    expected_root_entries = set(TABLE_TYPES)
    if require_marker:
        expected_root_entries.add(MARKER_NAME)
    actual_root_entries = {entry.name for entry in bundle_dir.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ShardError(
            "bundle root layout mismatch during full verification: "
            f"missing={sorted(expected_root_entries - actual_root_entries)}, "
            f"extra={sorted(actual_root_entries - expected_root_entries)}"
        )

    shard_records = [dict(record) for record in shards]
    listed_paths = {record["path"] for record in shard_records}
    actual_paths: set[str] = set()
    for table_type in TABLE_TYPES:
        table_dir = bundle_dir / table_type
        if table_dir.is_symlink() or not table_dir.is_dir():
            raise ShardError(f"missing shard directory: {table_dir}")
        for entry in table_dir.iterdir():
            relative = entry.relative_to(bundle_dir).as_posix()
            if entry.is_symlink() or not entry.is_file():
                raise ShardError(f"non-regular shard entry: {relative}")
            actual_paths.add(relative)
    if listed_paths != actual_paths:
        raise ShardError(
            "shard manifest mismatch during full verification: "
            f"missing={sorted(listed_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - listed_paths)}"
        )

    digests = {
        table_type: LogicalContentDigest(table_type) for table_type in TABLE_TYPES
    }
    material_orders: dict[str, list[str]] = {
        table_type: [] for table_type in TABLE_TYPES
    }
    row_totals = {table_type: 0 for table_type in TABLE_TYPES}
    row_group_totals = {table_type: 0 for table_type in TABLE_TYPES}
    decoded_rows = 0

    for record in shard_records:
        table_type = record["table_type"]
        path = bundle_dir / record["path"]
        if path.stat().st_size != record["bytes"]:
            raise ShardError(f"{record['path']} size mismatch")
        if record["bytes"] > profile.hard_shard_bytes:
            raise ShardError(f"{record['path']} exceeds profile hard limit")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise IntegrityError(f"{record['path']} SHA256 mismatch")

        with path.open("rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            validate_parquet_schema(
                parquet_file.schema_arrow, table_type, marker["bundle_id"]
            )
            metadata = parquet_file.metadata
            if metadata.num_rows != record["rows"]:
                raise ShardError(f"{record['path']} row count mismatch")
            if metadata.num_row_groups != record["row_groups"]:
                raise ShardError(f"{record['path']} row-group count mismatch")
            names = parquet_file.schema_arrow.names
            material_column = names.index("material_id")
            view_column = (
                names.index("view_id") if table_type == SAMPLES_TABLE else None
            )
            shard_materials: list[str] = []

            for row_group_index in range(metadata.num_row_groups):
                row_group_meta = metadata.row_group(row_group_index)
                label = f"{record['path']}:row_group[{row_group_index}]"
                expected_rows = (
                    profile.views_per_material if table_type == SAMPLES_TABLE else 1
                )
                if row_group_meta.num_rows != expected_rows:
                    raise ShardError(
                        f"{label} rows {row_group_meta.num_rows} != {expected_rows}"
                    )
                _validate_column_chunks(row_group_meta, table_type, label)
                material_stats = row_group_meta.column(material_column).statistics
                if material_stats is None:
                    raise ShardError(f"{label} material statistics missing")
                minimum = _decode_footer_material(
                    material_stats.min, f"{label}.material_id.min"
                )
                maximum = _decode_footer_material(
                    material_stats.max, f"{label}.material_id.max"
                )
                if minimum != maximum:
                    raise ShardError(f"{label} spans multiple materials")
                material_id = minimum
                if view_column is not None:
                    view_stats = row_group_meta.column(view_column).statistics
                    if (
                        view_stats is None
                        or view_stats.min != 0
                        or view_stats.max != profile.views_per_material - 1
                        or view_stats.null_count != 0
                    ):
                        raise ShardError(f"{label} view statistics mismatch")

                table = parquet_file.read_row_group(row_group_index, use_threads=False)
                if table.num_rows != expected_rows:
                    raise IntegrityError(f"{label} decoded row count mismatch")
                if table_type == SAMPLES_TABLE:
                    for view_id in range(profile.views_per_material):
                        row = _row_from_table(table, view_id)
                        _validate_sample_row(row, material_id, view_id)
                        digests[table_type].update(row)
                        if row_group_index == 0:
                            _decode_payload(
                                row["image_png"],
                                "image_png",
                                f"{row['sample_id']}.image_png",
                            )
                            _decode_payload(
                                row["mask_png"],
                                "mask_png",
                                f"{row['sample_id']}.mask_png",
                            )
                    if row_group_index == 0:
                        decoded_rows += table.num_rows
                else:
                    row = _row_from_table(table, 0)
                    _validate_material_row(row, material_id)
                    digests[table_type].update(row)
                    if row_group_index == 0:
                        for field in _MATERIAL_FIELDS[1:]:
                            _decode_payload(row[field], field, f"{material_id}.{field}")
                        decoded_rows += 1

                shard_materials.append(material_id)
                material_orders[table_type].append(material_id)
                row_totals[table_type] += expected_rows
                row_group_totals[table_type] += 1

        if shard_materials[0] != record["first_material_id"]:
            raise ShardError(f"{record['path']} first material mismatch")
        if shard_materials[-1] != record["last_material_id"]:
            raise ShardError(f"{record['path']} last material mismatch")

    for table_type in TABLE_TYPES:
        order = material_orders[table_type]
        if order != sorted(order) or len(order) != len(set(order)):
            raise IntegrityError(
                f"{table_type} material order is not strictly sorted/unique"
            )
        summary = marker["tables"][table_type]
        if row_totals[table_type] != summary["rows"]:
            raise IntegrityError(f"{table_type} row total differs from marker")
        if row_group_totals[table_type] != summary["row_groups"]:
            raise IntegrityError(f"{table_type} row-group total differs from marker")
        if digests[table_type].hexdigest() != summary["logical_content_digest"]:
            raise IntegrityError(f"{table_type} logical content digest mismatch")

    if material_orders[SAMPLES_TABLE] != material_orders[MATERIALS_TABLE]:
        raise IntegrityError("sample/material material order differs")
    return {
        "status": "passed",
        "decoded_rows": decoded_rows,
        "materials": len(material_orders[MATERIALS_TABLE]),
        "samples": row_totals[SAMPLES_TABLE],
        "logical_content_digests": {
            table_type: digests[table_type].hexdigest() for table_type in TABLE_TYPES
        },
    }


def _source_contract(
    scan: SourceScan,
    profile: BuildProfile,
) -> dict[str, Any]:
    source_digest = SourceDigest(profile.logical_dataset_id)
    logical = {
        table_type: LogicalContentDigest(table_type) for table_type in TABLE_TYPES
    }
    sample_rows = 0
    for source in scan.materials:
        material_row, samples, _material_bytes, _sample_bytes = _read_material(
            source, source_digest
        )
        logical[MATERIALS_TABLE].update(material_row)
        for row in samples:
            logical[SAMPLES_TABLE].update(row)
            sample_rows += 1
    scan.assert_unchanged()
    return {
        "source_digest": source_digest.hexdigest(),
        "logical_content_digests": {
            table_type: logical[table_type].hexdigest() for table_type in TABLE_TYPES
        },
        "materials": len(scan.materials),
        "samples": sample_rows,
    }


def _build_marker(
    *,
    bundle_id: str,
    profile: BuildProfile,
    sample_records: list[dict[str, Any]],
    material_records: list[dict[str, Any]],
    logical_digests: Mapping[str, str],
    source_digest: str,
    extra_gt_materials: int,
) -> dict[str, Any]:
    decoded_rows = len(sample_records) * profile.views_per_material + len(
        material_records
    )
    return {
        "tex_edit_pbr_parquet_version": FORMAT_VERSION,
        "bundle_id": bundle_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "writer": {
            "name": WRITER_NAME,
            "version": WRITER_VERSION,
            "pyarrow_version": pa.__version__,
        },
        "sort_orders": dict(SORT_ORDERS),
        "schema_fingerprints": {
            table_type: schema_fingerprint(table_type) for table_type in TABLE_TYPES
        },
        "tables": {
            SAMPLES_TABLE: {
                "rows": profile.expected_materials * profile.views_per_material,
                "row_groups": profile.expected_materials,
                "logical_content_digest": logical_digests[SAMPLES_TABLE],
            },
            MATERIALS_TABLE: {
                "rows": profile.expected_materials,
                "row_groups": profile.expected_materials,
                "logical_content_digest": logical_digests[MATERIALS_TABLE],
            },
        },
        "source": {
            "logical_dataset_id": profile.logical_dataset_id,
            "materials": profile.expected_materials,
            "samples": profile.expected_materials * profile.views_per_material,
            "digest": source_digest,
            "extra_gt_materials": extra_gt_materials,
        },
        "shards": sample_records + material_records,
        "validation": {
            "profile": profile.name,
            "status": "passed",
            "checks": list(_VALIDATION_CHECKS),
            "decoded_rows": decoded_rows,
        },
        "expected": {
            "materials": profile.expected_materials,
            "views_per_material": profile.views_per_material,
        },
    }


def _write_marker(root: Path, marker: Mapping[str, Any]) -> None:
    marker_path = root / MARKER_NAME
    payload = canonical_json_bytes(marker) + b"\n"
    try:
        with marker_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PublicationError(f"cannot write release marker: {marker_path}") from exc
    _fsync_directory(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic no-replace rename, with a guarded portable fallback."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise PublicationError(f"immutable output already exists: {destination}")
        if error_number not in {errno.ENOSYS, errno.EINVAL}:
            raise PublicationError(
                f"atomic publication failed: {os.strerror(error_number)}"
            )

    if os.path.lexists(destination):
        raise PublicationError(f"immutable output already exists: {destination}")
    try:
        os.rename(source, destination)
    except OSError as exc:
        raise PublicationError(
            f"cannot publish staging directory {source} -> {destination}"
        ) from exc


def build_bundle(
    render_root: str | Path,
    gt_root: str | Path,
    output_dir: str | Path,
    *,
    profile: str | BuildProfile = "matsynth-50",
) -> dict[str, Any]:
    selected_profile = _profile(profile)
    _require_pyarrow_version()
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_dir):
        raise PublicationError(f"immutable output already exists: {output_dir}")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    sample_writer: StreamingShardWriter | None = None
    material_writer: StreamingShardWriter | None = None
    try:
        (staging / SAMPLES_TABLE).mkdir()
        (staging / MATERIALS_TABLE).mkdir()
        scan = scan_sources(render_root, gt_root, profile=selected_profile)
        bundle_id = str(uuid.uuid4())
        sample_writer = StreamingShardWriter(
            staging, SAMPLES_TABLE, bundle_id, selected_profile
        )
        material_writer = StreamingShardWriter(
            staging, MATERIALS_TABLE, bundle_id, selected_profile
        )
        source_digest = SourceDigest(selected_profile.logical_dataset_id)
        logical = {
            table_type: LogicalContentDigest(table_type) for table_type in TABLE_TYPES
        }

        for source in scan.materials:
            (
                material_row,
                sample_rows,
                material_payload_bytes,
                sample_payload_bytes,
            ) = _read_material(source, source_digest)
            logical[MATERIALS_TABLE].update(material_row)
            for row in sample_rows:
                logical[SAMPLES_TABLE].update(row)
            sample_writer.write_material(
                source.material_id, sample_rows, sample_payload_bytes
            )
            material_writer.write_material(
                source.material_id, [material_row], material_payload_bytes
            )

        sample_records = sample_writer.finish()
        material_records = material_writer.finish()
        scan.assert_unchanged()
        marker = _build_marker(
            bundle_id=bundle_id,
            profile=selected_profile,
            sample_records=sample_records,
            material_records=material_records,
            logical_digests={
                table_type: logical[table_type].hexdigest()
                for table_type in TABLE_TYPES
            },
            source_digest=source_digest.hexdigest(),
            extra_gt_materials=len(scan.extra_gt_materials),
        )
        validated_marker, _validated_shards = validate_success_marker(marker)
        report = _verify_content(
            staging,
            validated_marker,
            marker["shards"],
            selected_profile,
            require_marker=False,
            verify_hashes=True,
        )
        if report["decoded_rows"] != marker["validation"]["decoded_rows"]:
            raise IntegrityError("decode sampling count differs from marker")
        _write_marker(staging, marker)
        inspect_bundle(staging, verify_file_hashes=False)
        _fsync_directory(staging / SAMPLES_TABLE)
        _fsync_directory(staging / MATERIALS_TABLE)
        _fsync_directory(staging)
        _rename_noreplace(staging, output_dir)
        _fsync_directory(output_dir.parent)
        return marker
    except BaseException:
        if sample_writer is not None:
            sample_writer.abort()
        if material_writer is not None:
            material_writer.abort()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_bundle(
    bundle_dir: str | Path,
    *,
    profile: str | BuildProfile = "matsynth-50",
    render_root: str | Path | None = None,
    gt_root: str | Path | None = None,
) -> dict[str, Any]:
    selected_profile = _profile(profile)
    _require_pyarrow_version()
    bundle_dir = Path(bundle_dir)
    index = inspect_bundle(bundle_dir, verify_file_hashes=True)
    marker = index.marker
    if marker["validation"]["profile"] != selected_profile.name:
        raise MarkerError(
            f"marker profile {marker['validation']['profile']!r} "
            f"!= requested {selected_profile.name!r}"
        )
    if marker["expected"] != {
        "materials": selected_profile.expected_materials,
        "views_per_material": selected_profile.views_per_material,
    }:
        raise MarkerError("marker expected counts conflict with profile")
    if marker["source"]["logical_dataset_id"] != selected_profile.logical_dataset_id:
        raise MarkerError("marker logical_dataset_id conflicts with profile")

    report = _verify_content(
        bundle_dir,
        marker,
        marker["shards"],
        selected_profile,
        require_marker=True,
        verify_hashes=False,
    )
    if report["decoded_rows"] != marker["validation"]["decoded_rows"]:
        raise IntegrityError("decode sampling count differs from marker")

    if (render_root is None) != (gt_root is None):
        raise ValueError("render_root and gt_root must be provided together")
    if render_root is not None and gt_root is not None:
        scan = scan_sources(render_root, gt_root, profile=selected_profile)
        source = _source_contract(scan, selected_profile)
        if source["source_digest"] != marker["source"]["digest"]:
            raise IntegrityError("source scan digest differs from marker")
        if source["logical_content_digests"] != report["logical_content_digests"]:
            raise IntegrityError("source payload/metadata differs from Parquet")
        report["source_equivalent"] = True
        report["extra_gt_materials"] = len(scan.extra_gt_materials)
        report["ignored_render_root_files"] = list(scan.ignored_render_root_files)
        report["ignored_gt_root_files"] = list(scan.ignored_gt_root_files)
        report["ignored_gt_attachments"] = scan.ignored_gt_attachments
    else:
        report["source_equivalent"] = None
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify tex_edit_pbr_parquet_version=1 bundles"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--render-root", type=Path, required=True)
    build_parser.add_argument("--gt-root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="matsynth-50"
    )

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument("--render-root", type=Path)
    verify_parser.add_argument("--gt-root", type=Path)
    verify_parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="matsynth-50"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_bundle(
                args.render_root,
                args.gt_root,
                args.output_dir,
                profile=args.profile,
            )
            summary = {
                "status": "published",
                "output_dir": str(args.output_dir),
                "bundle_id": result["bundle_id"],
                "materials": result["tables"][MATERIALS_TABLE]["rows"],
                "samples": result["tables"][SAMPLES_TABLE]["rows"],
            }
        else:
            result = verify_bundle(
                args.output_dir,
                profile=args.profile,
                render_root=args.render_root,
                gt_root=args.gt_root,
            )
            summary = {**result, "status": "verified"}
    except (PBRParquetError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
