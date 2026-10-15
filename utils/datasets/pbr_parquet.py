"""Self-contained MatSynth PBR Parquet format primitives.

Format version 1 deliberately keeps every encoded image payload in Parquet while
separating per-view samples from per-material ground truth.  This module owns
the strict schemas, marker validation, and footer-only bundle indexing used by
both the converter and runtime dataset.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import operator
import os
import random
import re
import uuid
from collections import Counter, OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import torch as _torch
    from torch.utils.data import BatchSampler as _TorchBatchSampler
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import get_worker_info as _get_worker_info
except ImportError:  # Format/converter utilities remain usable without PyTorch.
    _torch = None

    class _TorchDataset:
        pass

    class _TorchBatchSampler:
        pass

    def _get_worker_info():
        return None


FORMAT_VERSION = 1
REQUIRED_PYARROW_VERSION = "24.0.0"
MARKER_NAME = "_SUCCESS.json"
WRITER_NAME = "tex-edit-pbr-parquet"
WRITER_VERSION = 1

SAMPLES_TABLE = "samples"
MATERIALS_TABLE = "materials"
TABLE_TYPES = (SAMPLES_TABLE, MATERIALS_TABLE)

SAMPLE_SORT_ORDER = "material_id_asc,view_id_asc"
MATERIAL_SORT_ORDER = "material_id_asc"
SORT_ORDERS = {
    SAMPLES_TABLE: SAMPLE_SORT_ORDER,
    MATERIALS_TABLE: MATERIAL_SORT_ORDER,
}

TARGET_SHARD_BYTES = 512 * 1024 * 1024
HARD_SHARD_BYTES = 640 * 1024 * 1024

FORMAT_METADATA_KEY = b"tex_edit_pbr_parquet_version"
TABLE_METADATA_KEY = b"tex_edit_table_type"
BUNDLE_METADATA_KEY = b"bundle_id"
SORT_METADATA_KEY = b"sort_order"
FINGERPRINT_METADATA_KEY = b"schema_fingerprint"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

_BINARY_COLUMNS = {
    SAMPLES_TABLE: {"image_png", "mask_png"},
    MATERIALS_TABLE: {
        "color_jpeg",
        "normal_gl_png",
        "height_png",
        "roughness_png",
    },
}
_STATISTIC_COLUMNS = {
    SAMPLES_TABLE: {"material_id", "view_id"},
    MATERIALS_TABLE: {"material_id"},
}

# name, Arrow type factory, stable v1 type spelling used by the fingerprint.
_SAMPLE_FIELD_SPECS = (
    ("sample_id", pa.string, "string"),
    ("material_id", pa.string, "string"),
    ("view_id", pa.int16, "int16"),
    ("image_png", pa.binary, "binary"),
    ("mask_png", pa.binary, "binary"),
    ("theta", pa.float64, "float64"),
    ("phi", pa.float64, "float64"),
    ("radius", pa.float64, "float64"),
    (
        "camera_direction",
        lambda: pa.list_(pa.field("item", pa.float64(), nullable=False), 3),
        "fixed_size_list<item:float64:not_null>[3]",
    ),
    (
        "head_direction",
        lambda: pa.list_(pa.field("item", pa.float64(), nullable=False), 3),
        "fixed_size_list<item:float64:not_null>[3]",
    ),
)
_MATERIAL_FIELD_SPECS = (
    ("material_id", pa.string, "string"),
    ("color_jpeg", pa.binary, "binary"),
    ("normal_gl_png", pa.binary, "binary"),
    ("height_png", pa.binary, "binary"),
    ("roughness_png", pa.binary, "binary"),
)
_FIELD_SPECS = {
    SAMPLES_TABLE: _SAMPLE_FIELD_SPECS,
    MATERIALS_TABLE: _MATERIAL_FIELD_SPECS,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_PATH_RE = re.compile(
    r"^(samples|materials)/part-(?P<ordinal>[0-9]{5})\.parquet$"
)


class PBRParquetError(RuntimeError):
    """Base class for all v1 bundle contract failures."""


class MarkerError(PBRParquetError):
    """The release marker is missing, malformed, or internally inconsistent."""


class SchemaError(PBRParquetError):
    """A shard schema or required footer metadata does not match v1."""


class ShardError(PBRParquetError):
    """A shard manifest, file, or row-group layout is invalid."""


class IntegrityError(PBRParquetError):
    """Logical content, hashes, ordering, or cross-table relations are invalid."""


class SourceDataError(PBRParquetError):
    """Source render/GT files do not satisfy the converter input contract."""


class PublicationError(PBRParquetError):
    """The immutable staging/rename publication contract could not be met."""


class SamplerContractError(PBRParquetError):
    """Rank/worker lane planning cannot satisfy the v1 sampler contract."""


@dataclass(frozen=True)
class ShardManifest:
    relative_path: str
    table_type: str
    ordinal: int
    rows: int
    row_groups: int
    bytes: int
    sha256: str
    first_material_id: str
    last_material_id: str


@dataclass(frozen=True)
class RowGroupLocation:
    relative_path: str
    shard_ordinal: int
    row_group: int
    material_id: str
    rows: int
    global_start: int


@dataclass(frozen=True)
class BundleIndex:
    """Footer-derived logical coordinates; no Parquet handles are retained."""

    root: Path
    marker: Mapping[str, Any]
    shards: tuple[ShardManifest, ...]
    sample_row_groups: tuple[RowGroupLocation, ...]
    material_row_groups: Mapping[str, RowGroupLocation]

    @property
    def sample_count(self) -> int:
        return int(self.marker["tables"][SAMPLES_TABLE]["rows"])

    @property
    def material_count(self) -> int:
        return int(self.marker["tables"][MATERIALS_TABLE]["rows"])

    @property
    def views_per_material(self) -> int:
        return int(self.marker["expected"]["views_per_material"])


def _field_specs(table_type: str):
    try:
        return _FIELD_SPECS[table_type]
    except KeyError as exc:
        raise ValueError(f"unknown PBR Parquet table type: {table_type!r}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode contract JSON deterministically without whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def schema_contract(table_type: str) -> dict[str, Any]:
    """Return the versioned logical schema input used for its fingerprint."""

    return {
        "tex_edit_pbr_parquet_version": FORMAT_VERSION,
        "table_type": table_type,
        "sort_order": SORT_ORDERS[table_type],
        "fields": [
            {"name": name, "type": stable_type, "nullable": False}
            for name, _factory, stable_type in _field_specs(table_type)
        ],
    }


def schema_fingerprint(table_type: str) -> str:
    return hashlib.sha256(canonical_json_bytes(schema_contract(table_type))).hexdigest()


def parquet_schema(table_type: str, bundle_id: str) -> pa.Schema:
    """Build the exact non-null Arrow schema plus required shard metadata."""

    _validate_uuid(bundle_id, "bundle_id", SchemaError)
    fields = [
        pa.field(name, factory(), nullable=False)
        for name, factory, _stable_type in _field_specs(table_type)
    ]
    metadata = {
        FORMAT_METADATA_KEY: str(FORMAT_VERSION).encode("ascii"),
        TABLE_METADATA_KEY: table_type.encode("ascii"),
        BUNDLE_METADATA_KEY: bundle_id.encode("ascii"),
        SORT_METADATA_KEY: SORT_ORDERS[table_type].encode("ascii"),
        FINGERPRINT_METADATA_KEY: schema_fingerprint(table_type).encode("ascii"),
    }
    return pa.schema(fields, metadata=metadata)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_uuid(
    value: Any,
    label: str,
    error_type: type[PBRParquetError] = MarkerError,
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise error_type(f"{label} is not a valid UUID: {value!r}") from exc
    canonical = str(parsed)
    if value != canonical:
        raise error_type(f"{label} must use canonical lowercase UUID form")
    return value


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarkerError(f"{label} must be a JSON object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MarkerError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _expect_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MarkerError(f"{label} must be an integer >= {minimum}")
    return value


def _expect_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MarkerError(f"{label} must be a non-empty string")
    return value


def _expect_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MarkerError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _validate_created_at(value: Any) -> str:
    value = _expect_nonempty_string(value, "marker.created_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MarkerError("marker.created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise MarkerError("marker.created_at must include a UTC offset")
    return value


def _parse_shard_manifest(raw: Any, index: int) -> ShardManifest:
    raw = _expect_mapping(raw, f"marker.shards[{index}]")
    _expect_exact_keys(
        raw,
        {
            "path",
            "table_type",
            "ordinal",
            "rows",
            "row_groups",
            "bytes",
            "sha256",
            "first_material_id",
            "last_material_id",
        },
        f"marker.shards[{index}]",
    )
    relative_path = _expect_nonempty_string(raw["path"], f"marker.shards[{index}].path")
    match = _SHARD_PATH_RE.fullmatch(relative_path)
    if match is None:
        raise MarkerError(
            f"marker.shards[{index}].path is not a canonical shard path: "
            f"{relative_path!r}"
        )
    table_type = _expect_nonempty_string(
        raw["table_type"], f"marker.shards[{index}].table_type"
    )
    if table_type not in TABLE_TYPES or match.group(1) != table_type:
        raise MarkerError(f"marker.shards[{index}] table type conflicts with its path")
    ordinal = _expect_int(raw["ordinal"], f"marker.shards[{index}].ordinal")
    if int(match.group("ordinal")) != ordinal:
        raise MarkerError(f"marker.shards[{index}] ordinal conflicts with its filename")
    return ShardManifest(
        relative_path=relative_path,
        table_type=table_type,
        ordinal=ordinal,
        rows=_expect_int(raw["rows"], f"marker.shards[{index}].rows", 1),
        row_groups=_expect_int(
            raw["row_groups"], f"marker.shards[{index}].row_groups", 1
        ),
        bytes=_expect_int(raw["bytes"], f"marker.shards[{index}].bytes", 1),
        sha256=_expect_sha256(raw["sha256"], f"marker.shards[{index}].sha256"),
        first_material_id=_expect_nonempty_string(
            raw["first_material_id"],
            f"marker.shards[{index}].first_material_id",
        ),
        last_material_id=_expect_nonempty_string(
            raw["last_material_id"],
            f"marker.shards[{index}].last_material_id",
        ),
    )


def validate_success_marker(
    raw: Any,
) -> tuple[dict[str, Any], tuple[ShardManifest, ...]]:
    """Validate the exact v1 marker shape and all marker-only invariants."""

    marker = dict(_expect_mapping(raw, "marker"))
    _expect_exact_keys(
        marker,
        {
            "tex_edit_pbr_parquet_version",
            "bundle_id",
            "created_at",
            "writer",
            "sort_orders",
            "schema_fingerprints",
            "tables",
            "source",
            "shards",
            "validation",
            "expected",
        },
        "marker",
    )
    if marker["tex_edit_pbr_parquet_version"] != FORMAT_VERSION:
        raise MarkerError(
            "unsupported tex_edit_pbr_parquet_version: "
            f"{marker['tex_edit_pbr_parquet_version']!r}"
        )
    _validate_uuid(marker["bundle_id"], "marker.bundle_id")
    _validate_created_at(marker["created_at"])

    writer = _expect_mapping(marker["writer"], "marker.writer")
    _expect_exact_keys(writer, {"name", "version", "pyarrow_version"}, "marker.writer")
    if writer["name"] != WRITER_NAME or writer["version"] != WRITER_VERSION:
        raise MarkerError("marker writer identity is not supported")
    if writer["pyarrow_version"] != REQUIRED_PYARROW_VERSION:
        raise MarkerError(
            "marker was not written by the required PyArrow version "
            f"{REQUIRED_PYARROW_VERSION}"
        )

    sort_orders = _expect_mapping(marker["sort_orders"], "marker.sort_orders")
    _expect_exact_keys(sort_orders, set(TABLE_TYPES), "marker.sort_orders")
    if dict(sort_orders) != SORT_ORDERS:
        raise MarkerError("marker sort orders do not match format v1")

    fingerprints = _expect_mapping(
        marker["schema_fingerprints"], "marker.schema_fingerprints"
    )
    _expect_exact_keys(fingerprints, set(TABLE_TYPES), "marker.schema_fingerprints")
    expected_fingerprints = {
        table_type: schema_fingerprint(table_type) for table_type in TABLE_TYPES
    }
    if dict(fingerprints) != expected_fingerprints:
        raise MarkerError("marker schema fingerprints do not match format v1")

    expected = _expect_mapping(marker["expected"], "marker.expected")
    _expect_exact_keys(expected, {"materials", "views_per_material"}, "marker.expected")
    expected_materials = _expect_int(
        expected["materials"], "marker.expected.materials", 1
    )
    views_per_material = _expect_int(
        expected["views_per_material"],
        "marker.expected.views_per_material",
        1,
    )

    tables = _expect_mapping(marker["tables"], "marker.tables")
    _expect_exact_keys(tables, set(TABLE_TYPES), "marker.tables")
    normalized_tables: dict[str, dict[str, Any]] = {}
    for table_type in TABLE_TYPES:
        summary = dict(
            _expect_mapping(tables[table_type], f"marker.tables.{table_type}")
        )
        _expect_exact_keys(
            summary,
            {"rows", "row_groups", "logical_content_digest"},
            f"marker.tables.{table_type}",
        )
        summary["rows"] = _expect_int(
            summary["rows"], f"marker.tables.{table_type}.rows", 1
        )
        summary["row_groups"] = _expect_int(
            summary["row_groups"],
            f"marker.tables.{table_type}.row_groups",
            1,
        )
        summary["logical_content_digest"] = _expect_sha256(
            summary["logical_content_digest"],
            f"marker.tables.{table_type}.logical_content_digest",
        )
        normalized_tables[table_type] = summary
    marker["tables"] = normalized_tables

    if normalized_tables[MATERIALS_TABLE]["rows"] != expected_materials:
        raise MarkerError("materials row count conflicts with marker.expected")
    if (
        normalized_tables[SAMPLES_TABLE]["rows"]
        != expected_materials * views_per_material
    ):
        raise MarkerError("samples row count conflicts with marker.expected")
    if normalized_tables[MATERIALS_TABLE]["row_groups"] != expected_materials:
        raise MarkerError("materials must have exactly one row group per material")
    if normalized_tables[SAMPLES_TABLE]["row_groups"] != expected_materials:
        raise MarkerError("samples must have exactly one row group per material")

    source = dict(_expect_mapping(marker["source"], "marker.source"))
    _expect_exact_keys(
        source,
        {
            "logical_dataset_id",
            "materials",
            "samples",
            "digest",
            "extra_gt_materials",
        },
        "marker.source",
    )
    _expect_nonempty_string(
        source["logical_dataset_id"], "marker.source.logical_dataset_id"
    )
    if (
        _expect_int(source["materials"], "marker.source.materials", 1)
        != expected_materials
    ):
        raise MarkerError("source material count conflicts with marker.expected")
    if (
        _expect_int(source["samples"], "marker.source.samples", 1)
        != expected_materials * views_per_material
    ):
        raise MarkerError("source sample count conflicts with marker.expected")
    _expect_sha256(source["digest"], "marker.source.digest")
    _expect_int(source["extra_gt_materials"], "marker.source.extra_gt_materials")
    marker["source"] = source

    validation = dict(_expect_mapping(marker["validation"], "marker.validation"))
    _expect_exact_keys(
        validation,
        {"profile", "status", "checks", "decoded_rows"},
        "marker.validation",
    )
    _expect_nonempty_string(validation["profile"], "marker.validation.profile")
    if validation["status"] != "passed":
        raise MarkerError("marker.validation.status must be 'passed'")
    checks = validation["checks"]
    if (
        not isinstance(checks, list)
        or not checks
        or any(not isinstance(check, str) or not check for check in checks)
    ):
        raise MarkerError("marker.validation.checks must be non-empty strings")
    _expect_int(validation["decoded_rows"], "marker.validation.decoded_rows")
    marker["validation"] = validation

    raw_shards = marker["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise MarkerError("marker.shards must be a non-empty list")
    shards = tuple(
        _parse_shard_manifest(raw_shard, index)
        for index, raw_shard in enumerate(raw_shards)
    )
    by_table = {
        table_type: [shard for shard in shards if shard.table_type == table_type]
        for table_type in TABLE_TYPES
    }
    for table_type, table_shards in by_table.items():
        if not table_shards:
            raise MarkerError(f"marker contains no {table_type} shards")
        ordinals = [shard.ordinal for shard in table_shards]
        if ordinals != list(range(len(table_shards))):
            raise MarkerError(
                f"{table_type} shard ordinals must be contiguous from zero"
            )
        if (
            sum(shard.rows for shard in table_shards)
            != normalized_tables[table_type]["rows"]
        ):
            raise MarkerError(f"{table_type} shard rows do not match table summary")
        if (
            sum(shard.row_groups for shard in table_shards)
            != normalized_tables[table_type]["row_groups"]
        ):
            raise MarkerError(
                f"{table_type} shard row groups do not match table summary"
            )
    marker["shards"] = [dict(item) for item in raw_shards]
    marker["expected"] = {
        "materials": expected_materials,
        "views_per_material": views_per_material,
    }
    return marker, shards


def load_success_marker(
    bundle_dir: str | Path,
) -> tuple[dict[str, Any], tuple[ShardManifest, ...]]:
    root = Path(bundle_dir)
    marker_path = root / MARKER_NAME
    if not root.is_dir():
        raise MarkerError(f"bundle directory does not exist: {root}")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise MarkerError(
            f"bundle marker is missing or not a regular file: {marker_path}"
        )
    try:
        marker_bytes = marker_path.read_bytes()
    except OSError as exc:
        raise MarkerError(f"cannot read bundle marker: {marker_path}") from exc
    if len(marker_bytes) > 16 * 1024 * 1024:
        raise MarkerError("bundle marker is unexpectedly large")
    try:
        raw = json.loads(marker_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarkerError("bundle marker is not valid UTF-8 JSON") from exc
    return validate_success_marker(raw)


def _decode_schema_metadata(schema: pa.Schema, key: bytes, label: str) -> str:
    metadata = schema.metadata or {}
    value = metadata.get(key)
    if value is None:
        raise SchemaError(f"{label} metadata is missing")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{label} metadata is not UTF-8") from exc


def validate_parquet_schema(
    schema: pa.Schema,
    table_type: str,
    bundle_id: str,
) -> None:
    expected = parquet_schema(table_type, bundle_id)
    if not schema.remove_metadata().equals(expected.remove_metadata()):
        raise SchemaError(f"{table_type} Arrow schema does not match format v1")
    expected_metadata = {
        FORMAT_METADATA_KEY: str(FORMAT_VERSION),
        TABLE_METADATA_KEY: table_type,
        BUNDLE_METADATA_KEY: bundle_id,
        SORT_METADATA_KEY: SORT_ORDERS[table_type],
        FINGERPRINT_METADATA_KEY: schema_fingerprint(table_type),
    }
    for key, expected_value in expected_metadata.items():
        actual_value = _decode_schema_metadata(
            schema, key, key.decode("ascii", errors="replace")
        )
        if actual_value != expected_value:
            raise SchemaError(
                f"{table_type} metadata {key!r} mismatch: "
                f"{actual_value!r} != {expected_value!r}"
            )


def _decode_material_stat(value: Any, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ShardError(f"{label} statistics are not UTF-8") from exc
    if not isinstance(value, str) or not value:
        raise ShardError(f"{label} statistics must be a non-empty string")
    return value


def _column_root_name(path_in_schema: str) -> str:
    return path_in_schema.split(".", 1)[0]


def _validate_column_chunks(
    row_group: pq.RowGroupMetaData,
    table_type: str,
    row_group_label: str,
) -> None:
    binary_columns = _BINARY_COLUMNS[table_type]
    statistic_columns = _STATISTIC_COLUMNS[table_type]
    for column_index in range(row_group.num_columns):
        column = row_group.column(column_index)
        root_name = _column_root_name(column.path_in_schema)
        compression = str(column.compression).upper()
        if root_name in binary_columns:
            if compression not in {"NONE", "UNCOMPRESSED"}:
                raise ShardError(
                    f"{row_group_label}.{root_name} binary compression must be NONE"
                )
        elif compression != "ZSTD":
            raise ShardError(f"{row_group_label}.{root_name} compression must be ZSTD")

        encodings = {str(encoding).upper() for encoding in column.encodings}
        has_dictionary = bool(
            encodings.intersection({"RLE_DICTIONARY", "PLAIN_DICTIONARY"})
        )
        expects_dictionary = table_type == SAMPLES_TABLE and root_name == "material_id"
        if expects_dictionary:
            if not has_dictionary:
                raise ShardError(
                    f"{row_group_label}.material_id must use dictionary encoding"
                )
        elif has_dictionary:
            raise ShardError(
                f"{row_group_label}.{root_name} must not use dictionary encoding"
            )

        if root_name in statistic_columns:
            if column.statistics is None:
                raise ShardError(
                    f"{row_group_label}.{root_name} statistics are required"
                )
        elif column.statistics is not None:
            raise ShardError(
                f"{row_group_label}.{root_name} statistics must be disabled"
            )


def _validate_actual_shard_set(
    root: Path,
    shards: tuple[ShardManifest, ...],
) -> None:
    allowed_root_entries = {MARKER_NAME, *TABLE_TYPES}
    actual_root_entries = {entry.name for entry in root.iterdir()}
    if actual_root_entries != allowed_root_entries:
        raise ShardError(
            "bundle root layout mismatch: "
            f"missing={sorted(allowed_root_entries - actual_root_entries)}, "
            f"extra={sorted(actual_root_entries - allowed_root_entries)}"
        )
    listed = {shard.relative_path for shard in shards}
    actual: set[str] = set()
    for table_type in TABLE_TYPES:
        table_dir = root / table_type
        if table_dir.is_symlink() or not table_dir.is_dir():
            raise ShardError(f"missing regular shard directory: {table_dir}")
        for entry in table_dir.iterdir():
            relative_path = entry.relative_to(root).as_posix()
            if entry.is_symlink() or not entry.is_file():
                raise ShardError(f"unexpected non-file shard entry: {relative_path}")
            actual.add(relative_path)
    if actual != listed:
        raise ShardError(
            "marker shard list differs from bundle files: "
            f"missing={sorted(listed - actual)}, extra={sorted(actual - listed)}"
        )


def inspect_bundle(
    bundle_dir: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> BundleIndex:
    """Validate marker/manifest/schema/footer invariants without reading payloads."""

    root = Path(bundle_dir)
    marker, shards = load_success_marker(root)
    _validate_actual_shard_set(root, shards)

    bundle_id = marker["bundle_id"]
    views_per_material = int(marker["expected"]["views_per_material"])
    sample_groups: list[RowGroupLocation] = []
    material_groups: dict[str, RowGroupLocation] = {}
    material_order: dict[str, list[str]] = {
        SAMPLES_TABLE: [],
        MATERIALS_TABLE: [],
    }
    global_starts = {SAMPLES_TABLE: 0, MATERIALS_TABLE: 0}

    for shard in shards:
        path = root / shard.relative_path
        if path.is_symlink() or not path.is_file():
            raise ShardError(f"shard is missing or not a regular file: {path}")
        stat = path.stat()
        if stat.st_size != shard.bytes:
            raise ShardError(
                f"{shard.relative_path} size mismatch: "
                f"{stat.st_size} != {shard.bytes}"
            )
        if stat.st_size > HARD_SHARD_BYTES:
            raise ShardError(
                f"{shard.relative_path} exceeds the {HARD_SHARD_BYTES} byte hard limit"
            )
        if verify_file_hashes and sha256_file(path) != shard.sha256:
            raise IntegrityError(f"{shard.relative_path} SHA256 mismatch")

        try:
            with path.open("rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                validate_parquet_schema(
                    parquet_file.schema_arrow, shard.table_type, bundle_id
                )
                file_metadata = parquet_file.metadata
                if file_metadata.num_rows != shard.rows:
                    raise ShardError(f"{shard.relative_path} footer row count mismatch")
                if file_metadata.num_row_groups != shard.row_groups:
                    raise ShardError(
                        f"{shard.relative_path} footer row-group count mismatch"
                    )

                arrow_names = parquet_file.schema_arrow.names
                material_column_index = arrow_names.index("material_id")
                view_column_index = (
                    arrow_names.index("view_id")
                    if shard.table_type == SAMPLES_TABLE
                    else None
                )
                shard_materials: list[str] = []
                for row_group_index in range(file_metadata.num_row_groups):
                    row_group = file_metadata.row_group(row_group_index)
                    label = f"{shard.relative_path}:row_group[{row_group_index}]"
                    expected_rows = (
                        views_per_material if shard.table_type == SAMPLES_TABLE else 1
                    )
                    if row_group.num_rows != expected_rows:
                        raise ShardError(
                            f"{label} must contain exactly {expected_rows} rows"
                        )
                    _validate_column_chunks(row_group, shard.table_type, label)

                    material_statistics = row_group.column(
                        material_column_index
                    ).statistics
                    if material_statistics is None:
                        raise ShardError(f"{label}.material_id statistics are missing")
                    material_min = _decode_material_stat(
                        material_statistics.min, f"{label}.material_id.min"
                    )
                    material_max = _decode_material_stat(
                        material_statistics.max, f"{label}.material_id.max"
                    )
                    if material_min != material_max:
                        raise ShardError(f"{label} spans more than one material")
                    material_id = material_min
                    shard_materials.append(material_id)
                    material_order[shard.table_type].append(material_id)

                    if view_column_index is not None:
                        view_statistics = row_group.column(view_column_index).statistics
                        if view_statistics is None:
                            raise ShardError(f"{label}.view_id statistics are missing")
                        if (
                            view_statistics.min != 0
                            or view_statistics.max != views_per_material - 1
                            or view_statistics.null_count != 0
                        ):
                            raise ShardError(
                                f"{label}.view_id statistics do not cover "
                                f"0..{views_per_material - 1}"
                            )

                    location = RowGroupLocation(
                        relative_path=shard.relative_path,
                        shard_ordinal=shard.ordinal,
                        row_group=row_group_index,
                        material_id=material_id,
                        rows=expected_rows,
                        global_start=global_starts[shard.table_type],
                    )
                    global_starts[shard.table_type] += expected_rows
                    if shard.table_type == SAMPLES_TABLE:
                        sample_groups.append(location)
                    else:
                        if material_id in material_groups:
                            raise IntegrityError(
                                f"duplicate material row group: {material_id}"
                            )
                        material_groups[material_id] = location
        except PBRParquetError:
            raise
        except Exception as exc:
            raise ShardError(f"cannot read Parquet footer: {path}") from exc

        if not shard_materials:
            raise ShardError(f"{shard.relative_path} has no row groups")
        if shard_materials[0] != shard.first_material_id:
            raise ShardError(
                f"{shard.relative_path} first material conflicts with marker"
            )
        if shard_materials[-1] != shard.last_material_id:
            raise ShardError(
                f"{shard.relative_path} last material conflicts with marker"
            )

    for table_type, materials in material_order.items():
        if materials != sorted(materials) or len(materials) != len(set(materials)):
            raise IntegrityError(
                f"{table_type} material row groups are not strictly sorted and unique"
            )
    if material_order[SAMPLES_TABLE] != material_order[MATERIALS_TABLE]:
        raise IntegrityError("samples/materials material_id sets or ordering differ")

    sample_summary = marker["tables"][SAMPLES_TABLE]
    material_summary = marker["tables"][MATERIALS_TABLE]
    if global_starts[SAMPLES_TABLE] != sample_summary["rows"]:
        raise IntegrityError("samples footer totals differ from marker")
    if len(sample_groups) != sample_summary["row_groups"]:
        raise IntegrityError("samples row-group totals differ from marker")
    if global_starts[MATERIALS_TABLE] != material_summary["rows"]:
        raise IntegrityError("materials footer totals differ from marker")
    if len(material_groups) != material_summary["row_groups"]:
        raise IntegrityError("materials row-group totals differ from marker")

    return BundleIndex(
        root=root,
        marker=marker,
        shards=shards,
        sample_row_groups=tuple(sample_groups),
        material_row_groups=material_groups,
    )


_SAMPLE_IMAGE_COLUMNS = {
    "image": "image_png",
    "mask": "mask_png",
}
_METADATA_COLUMNS = (
    "theta",
    "phi",
    "radius",
    "camera_direction",
    "head_direction",
)
_MATERIAL_IMAGE_COLUMNS = {
    "Color": "color_jpeg",
    "NormalGL": "normal_gl_png",
    "Height": "height_png",
    "Roughness": "roughness_png",
}
_RUNTIME_CONTEXT_UNSET = object()


class _PBRFetchKey(NamedTuple):
    """Sampler-only key carrying the worker/pass contract to the Dataset."""

    sample_index: int
    pass_idx: int
    expected_worker_id: int


class _CachedParquetFile:
    def __init__(self, handle: Any, parquet_file: pq.ParquetFile):
        self.handle = handle
        self.parquet_file = parquet_file

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass


class _WorkerRuntime:
    """Mutable state that must never be shared across process boundaries."""

    def __init__(self, pid: int):
        self.pid = pid
        self.context: object | int | None = _RUNTIME_CONTEXT_UNSET
        self.handles: OrderedDict[str, _CachedParquetFile] = OrderedDict()
        self.sample_row_groups: OrderedDict[tuple[str, int], pa.Table] = OrderedDict()
        self.materials: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # Private instrumentation used by the runtime contract tests.  It is
        # reset with each pass/context, so it cannot grow across training.
        self.sample_row_group_reads: Counter[tuple[str, int]] = Counter()
        self.material_row_group_reads: Counter[tuple[str, int]] = Counter()
        self.handle_opens: Counter[str] = Counter()

    def clear_payload_caches(self) -> None:
        self.sample_row_groups.clear()
        self.materials.clear()
        self.sample_row_group_reads.clear()
        self.material_row_group_reads.clear()

    def close(self) -> None:
        for cached in self.handles.values():
            cached.close()
        self.handles.clear()
        self.clear_payload_caches()


def _contract_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    error_type: type[Exception] = ValueError,
) -> int:
    if isinstance(value, bool):
        raise error_type(f"{label} must be an integer >= {minimum}")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise error_type(f"{label} must be an integer >= {minimum}") from exc
    if normalized < minimum:
        raise error_type(f"{label} must be an integer >= {minimum}")
    return int(normalized)


def _name_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an iterable of names, not a string")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of names") from exc
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must not contain duplicate names")
    return names


def _require_torch():
    if _torch is None:
        raise RuntimeError("PyTorch is required to decode PBR Parquet samples")
    return _torch


def _decode_image_bytes(payload: Any, label: str):
    """Mirror utils.io.load_image, but read an encoded in-memory payload."""

    torch = _require_torch()
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow and NumPy are required for image decoding") from exc

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise IntegrityError(f"{label} is not an encoded byte payload")
    try:
        with Image.open(io.BytesIO(bytes(payload))) as image:
            if image.mode.endswith("16"):
                array = np.array(image)
                if array.ndim == 2:
                    array = np.concatenate([array[..., np.newaxis]] * 3, axis=-1)
            else:
                array = np.array(image.convert("RGB"))
    except Exception as exc:
        raise IntegrityError(f"cannot decode {label}") from exc

    if array.dtype == np.uint16:
        array = array.astype(np.float32) / 65535.0
    elif array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        raise IntegrityError(f"{label} decoded to unsupported dtype {array.dtype}")
    if array.ndim != 3:
        raise IntegrityError(f"{label} decoded to a non-image shape {array.shape}")
    return torch.tensor(array).permute(2, 0, 1)


class PBRParquetDataset(_TorchDataset):
    """Map-style reader for a validated self-contained PBR Parquet bundle."""

    def __init__(
        self,
        parquet_dir: str | Path,
        *,
        image_names: Iterable[str] = ("image", "mask", "meta_data"),
        fetch_attr: Iterable[str] = (
            "Color",
            "NormalGL",
            "Height",
            "Roughness",
        ),
        transforms: Mapping[str, Any] | None = None,
        transform_group: Mapping[str, Iterable[str]] | None = None,
        sample_cache_size: int = 2,
        material_cache_size: int = 2,
        handle_cache_size: int = 4,
        use_threads: bool = False,
    ):
        self.index = inspect_bundle(parquet_dir)
        self.image_names = _name_tuple(image_names, "image_names")
        self.fetch_attr = _name_tuple(fetch_attr, "fetch_attr")

        allowed_image_names = set(_SAMPLE_IMAGE_COLUMNS) | {"meta_data"}
        unknown_images = sorted(set(self.image_names) - allowed_image_names)
        if unknown_images:
            raise ValueError(f"unsupported image_names: {unknown_images}")
        unknown_attrs = sorted(set(self.fetch_attr) - set(_MATERIAL_IMAGE_COLUMNS))
        if unknown_attrs:
            raise ValueError(f"unsupported fetch_attr: {unknown_attrs}")
        if not self.image_names and not self.fetch_attr:
            raise ValueError("image_names and fetch_attr cannot both be empty")

        if transforms is None:
            self.transforms: dict[str, Any] = {}
        elif not isinstance(transforms, Mapping):
            raise ValueError("transforms must be a Mapping or None")
        else:
            self.transforms = dict(transforms)
        if transform_group is None:
            transform_group = {}
        if not isinstance(transform_group, Mapping):
            raise ValueError("transform_group must be a Mapping or None")

        normalized_groups: dict[str, tuple[str, ...]] = {}
        for transform_name, raw_names in transform_group.items():
            if not isinstance(transform_name, str) or not transform_name:
                raise ValueError("transform_group keys must be non-empty strings")
            if transform_name not in self.transforms:
                raise ValueError(
                    f"transform_group references missing transform {transform_name!r}"
                )
            names = _name_tuple(raw_names, f"transform_group[{transform_name!r}]")
            known_outputs = (
                set(_SAMPLE_IMAGE_COLUMNS)
                | set(_MATERIAL_IMAGE_COLUMNS)
                | {"meta_data"}
            )
            unknown = sorted(set(names) - known_outputs)
            if unknown:
                raise ValueError(
                    f"transform_group[{transform_name!r}] references unsupported "
                    f"outputs: {unknown}"
                )
            if "meta_data" in names:
                raise ValueError("meta_data cannot be passed to an image transform")
            normalized_groups[transform_name] = names
        self.transform_group = normalized_groups

        self.sample_cache_size = _contract_int(
            sample_cache_size, "sample_cache_size", minimum=2
        )
        self.material_cache_size = _contract_int(
            material_cache_size, "material_cache_size", minimum=2
        )
        self.handle_cache_size = _contract_int(
            handle_cache_size, "handle_cache_size", minimum=1
        )
        if not isinstance(use_threads, bool):
            raise ValueError("use_threads must be a boolean")
        self.use_threads = use_threads

        sample_columns = ["material_id"]
        for name in self.image_names:
            if name in _SAMPLE_IMAGE_COLUMNS:
                sample_columns.append(_SAMPLE_IMAGE_COLUMNS[name])
            elif name == "meta_data":
                sample_columns.extend(_METADATA_COLUMNS)
        self._sample_columns = tuple(dict.fromkeys(sample_columns))
        self._material_columns = (
            "material_id",
            *(_MATERIAL_IMAGE_COLUMNS[name] for name in self.fetch_attr),
        )
        self._sample_group_starts = tuple(
            location.global_start for location in self.index.sample_row_groups
        )
        self._runtime: _WorkerRuntime | None = None

    @property
    def sample_row_groups(self) -> tuple[RowGroupLocation, ...]:
        return self.index.sample_row_groups

    @property
    def views_per_material(self) -> int:
        return self.index.views_per_material

    def __len__(self) -> int:
        return self.index.sample_count

    def __getitem__(self, item: Any):
        if isinstance(item, _PBRFetchKey):
            return self.__getitems__((item,))[0]
        if isinstance(item, slice):
            indices = range(*item.indices(len(self)))
            return self._collate_public_batch(self.__getitems__(indices))
        if _torch is not None and isinstance(item, _torch.Tensor):
            if item.ndim == 0:
                return self.__getitems__((item.item(),))[0]
            if item.ndim != 1:
                raise IndexError("tensor indices must be scalar or one-dimensional")
            return self._collate_public_batch(
                self.__getitems__(item.detach().cpu().tolist())
            )
        if isinstance(item, Iterable) and not isinstance(
            item, (str, bytes, bytearray, Mapping)
        ):
            return self._collate_public_batch(self.__getitems__(item))
        return self.__getitems__((item,))[0]

    def __getitems__(self, indices: Iterable[Any]) -> list[dict[str, Any]]:
        """Read each touched row group once and preserve requested order."""

        raw_indices = list(indices)
        if not raw_indices:
            return []
        normalized, context = self._normalize_fetch_indices(raw_indices)
        runtime = self._runtime_for_context(context)

        grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
        for output_position, sample_index in enumerate(normalized):
            group_index, row_offset = self._locate_sample(sample_index)
            grouped.setdefault(group_index, []).append((output_position, row_offset))

        output: list[dict[str, Any] | None] = [None] * len(normalized)
        for group_index, requests in grouped.items():
            location = self.index.sample_row_groups[group_index]
            table = self._sample_row_group(runtime, location)
            for output_position, row_offset in requests:
                output[output_position] = self._decode_sample(
                    runtime, table, location, row_offset
                )
        if any(sample is None for sample in output):
            raise RuntimeError("internal PBR Parquet batch assembly failure")
        return [sample for sample in output if sample is not None]

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Spawned workers must never deserialize a parent process file handle.
        state["_runtime"] = None
        return state

    def __del__(self):
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.close()

    def _normalize_fetch_indices(
        self, raw_indices: list[Any]
    ) -> tuple[list[int], int | None]:
        private_flags = [isinstance(item, _PBRFetchKey) for item in raw_indices]
        if any(private_flags) and not all(private_flags):
            raise SamplerContractError(
                "a fetch batch cannot mix sampler keys with public indices"
            )

        context: int | None = None
        if all(private_flags):
            keys = raw_indices
            context = _contract_int(
                keys[0].pass_idx,
                "fetch key pass_idx",
                minimum=0,
                error_type=SamplerContractError,
            )
            worker_info = _get_worker_info()
            actual_worker_id = 0 if worker_info is None else worker_info.id
            for key in keys:
                key_pass = _contract_int(
                    key.pass_idx,
                    "fetch key pass_idx",
                    minimum=0,
                    error_type=SamplerContractError,
                )
                if key_pass != context:
                    raise SamplerContractError(
                        "all sampler keys in a batch must use the same pass_idx"
                    )
                expected_worker_id = _contract_int(
                    key.expected_worker_id,
                    "fetch key expected_worker_id",
                    minimum=0,
                    error_type=SamplerContractError,
                )
                if expected_worker_id != actual_worker_id:
                    raise SamplerContractError(
                        "lane batch reached worker "
                        f"{actual_worker_id}, expected worker {expected_worker_id}"
                    )
            raw_indices = [key.sample_index for key in keys]

        normalized = [self._normalize_sample_index(item) for item in raw_indices]
        return normalized, context

    def _normalize_sample_index(self, value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError("sample index must be an integer")
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise TypeError("sample index must be an integer") from exc
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"sample index out of range: {index}")
        return int(index)

    def _locate_sample(self, sample_index: int) -> tuple[int, int]:
        group_index = bisect.bisect_right(self._sample_group_starts, sample_index) - 1
        if group_index < 0:
            raise IndexError(f"sample index out of range: {sample_index}")
        location = self.index.sample_row_groups[group_index]
        row_offset = sample_index - location.global_start
        if row_offset < 0 or row_offset >= location.rows:
            raise IndexError(f"sample index out of range: {sample_index}")
        return group_index, row_offset

    def _runtime_for_context(self, context: int | None) -> _WorkerRuntime:
        pid = os.getpid()
        runtime = self._runtime
        if runtime is None or runtime.pid != pid:
            if runtime is not None:
                runtime.close()
            runtime = _WorkerRuntime(pid)
            self._runtime = runtime
        if runtime.context is _RUNTIME_CONTEXT_UNSET or runtime.context != context:
            runtime.clear_payload_caches()
            runtime.context = context
        return runtime

    def _parquet_file(
        self, runtime: _WorkerRuntime, relative_path: str
    ) -> pq.ParquetFile:
        cached = runtime.handles.pop(relative_path, None)
        if cached is not None:
            runtime.handles[relative_path] = cached
            return cached.parquet_file

        path = self.index.root / relative_path
        try:
            handle = path.open("rb")
            parquet_file = pq.ParquetFile(handle)
        except Exception as exc:
            try:
                handle.close()
            except (Exception, UnboundLocalError):
                pass
            raise ShardError(f"cannot open runtime Parquet shard: {path}") from exc
        runtime.handles[relative_path] = _CachedParquetFile(handle, parquet_file)
        runtime.handle_opens[relative_path] += 1
        while len(runtime.handles) > self.handle_cache_size:
            _evicted_path, evicted = runtime.handles.popitem(last=False)
            evicted.close()
        return parquet_file

    def _sample_row_group(
        self,
        runtime: _WorkerRuntime,
        location: RowGroupLocation,
    ) -> pa.Table:
        cache_key = (location.relative_path, location.row_group)
        table = runtime.sample_row_groups.pop(cache_key, None)
        if table is not None:
            runtime.sample_row_groups[cache_key] = table
            return table
        try:
            table = self._parquet_file(runtime, location.relative_path).read_row_group(
                location.row_group,
                columns=list(self._sample_columns),
                use_threads=self.use_threads,
            )
        except Exception as exc:
            raise ShardError(
                "cannot read sample row group "
                f"{location.relative_path}:{location.row_group}"
            ) from exc
        if table.num_rows != location.rows:
            raise ShardError(
                "sample row group changed after bundle inspection: "
                f"{location.relative_path}:{location.row_group}"
            )
        runtime.sample_row_group_reads[cache_key] += 1
        runtime.sample_row_groups[cache_key] = table
        while len(runtime.sample_row_groups) > self.sample_cache_size:
            runtime.sample_row_groups.popitem(last=False)
        return table

    def _material_tensors(
        self, runtime: _WorkerRuntime, material_id: str
    ) -> dict[str, Any]:
        if not self.fetch_attr:
            return {}
        tensors = runtime.materials.pop(material_id, None)
        if tensors is not None:
            runtime.materials[material_id] = tensors
            return tensors

        try:
            location = self.index.material_row_groups[material_id]
        except KeyError as exc:
            raise IntegrityError(f"material row is missing: {material_id}") from exc
        try:
            table = self._parquet_file(runtime, location.relative_path).read_row_group(
                location.row_group,
                columns=list(self._material_columns),
                use_threads=self.use_threads,
            )
        except Exception as exc:
            raise ShardError(
                "cannot read material row group "
                f"{location.relative_path}:{location.row_group}"
            ) from exc
        if table.num_rows != 1:
            raise ShardError(
                "material row group changed after bundle inspection: "
                f"{location.relative_path}:{location.row_group}"
            )
        stored_material_id = table.column("material_id")[0].as_py()
        if stored_material_id != material_id:
            raise IntegrityError(
                f"material row-group payload mismatch: {stored_material_id!r} "
                f"!= {material_id!r}"
            )

        tensors = {}
        for attr in self.fetch_attr:
            column = _MATERIAL_IMAGE_COLUMNS[attr]
            tensors[attr] = _decode_image_bytes(
                table.column(column)[0].as_py(), f"{material_id}.{column}"
            )
        cache_key = (location.relative_path, location.row_group)
        runtime.material_row_group_reads[cache_key] += 1
        runtime.materials[material_id] = tensors
        while len(runtime.materials) > self.material_cache_size:
            runtime.materials.popitem(last=False)
        return tensors

    def _decode_sample(
        self,
        runtime: _WorkerRuntime,
        table: pa.Table,
        location: RowGroupLocation,
        row_offset: int,
    ) -> dict[str, Any]:
        material_id = table.column("material_id")[row_offset].as_py()
        if material_id != location.material_id:
            raise IntegrityError(
                f"sample row-group payload mismatch: {material_id!r} "
                f"!= {location.material_id!r}"
            )

        sample: dict[str, Any] = {}
        for name in self.image_names:
            if name in _SAMPLE_IMAGE_COLUMNS:
                column = _SAMPLE_IMAGE_COLUMNS[name]
                sample[name] = _decode_image_bytes(
                    table.column(column)[row_offset].as_py(),
                    f"{material_id}.{column}[{row_offset}]",
                ).clone()
            else:
                sample["meta_data"] = {
                    "theta": float(table.column("theta")[row_offset].as_py()),
                    "phi": float(table.column("phi")[row_offset].as_py()),
                    "radius": float(table.column("radius")[row_offset].as_py()),
                    "camera_direction": [
                        float(value)
                        for value in table.column("camera_direction")[
                            row_offset
                        ].as_py()
                    ],
                    "head_direction": [
                        float(value)
                        for value in table.column("head_direction")[row_offset].as_py()
                    ],
                }

        for attr, tensor in self._material_tensors(runtime, material_id).items():
            # Cached decoded material tensors are immutable templates.  Every
            # returned sample owns a clone so transforms cannot leak state.
            sample[attr] = tensor.clone()
        self._apply_transforms(sample)
        return sample

    def _apply_transforms(self, sample: dict[str, Any]) -> None:
        if not self.transform_group:
            return
        torch = _require_torch()
        for transform_name, configured_names in self.transform_group.items():
            names = tuple(name for name in sample if name in configured_names)
            if not names:
                continue
            values = [sample[name] for name in names]
            if any(not isinstance(value, torch.Tensor) for value in values):
                raise TypeError(
                    f"transform {transform_name!r} received a non-tensor output"
                )
            channel_sizes = [value.shape[0] for value in values]
            concatenated = torch.cat(values, dim=0)
            transformed = self.transforms[transform_name](concatenated)
            if not isinstance(transformed, torch.Tensor):
                raise TypeError(f"transform {transform_name!r} must return a Tensor")
            if transformed.ndim == 0 or transformed.shape[0] != sum(channel_sizes):
                raise ValueError(
                    f"transform {transform_name!r} changed the channel count"
                )
            for name, value in zip(
                names, torch.split(transformed, channel_sizes, dim=0)
            ):
                sample[name] = value

    @staticmethod
    def _collate_public_batch(
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not samples:
            raise IndexError("cannot batch an empty PBR Parquet index selection")
        torch = _require_torch()
        keys = tuple(samples[0])
        if any(tuple(sample) != keys for sample in samples[1:]):
            raise RuntimeError("PBR Parquet samples have inconsistent keys")
        return {
            key: (
                torch.stack([sample[key] for sample in samples], dim=0)
                if isinstance(samples[0][key], torch.Tensor)
                else [sample[key] for sample in samples]
            )
            for key in keys
        }


class PBRParquetBatchSampler(_TorchBatchSampler):
    """Globally shuffled sample plan consumed by rank striding and workers."""

    batch_size = None
    drop_last = False

    def __init__(
        self,
        dataset: PBRParquetDataset,
        *,
        configured_batch_size: int,
        num_processes: int,
        num_workers: int,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.configured_batch_size = _contract_int(
            configured_batch_size,
            "configured_batch_size",
            minimum=1,
            error_type=SamplerContractError,
        )
        self.num_processes = _contract_int(
            num_processes,
            "num_processes",
            minimum=1,
            error_type=SamplerContractError,
        )
        self.num_workers = _contract_int(
            num_workers,
            "num_workers",
            minimum=0,
            error_type=SamplerContractError,
        )
        if isinstance(seed, bool):
            raise SamplerContractError("seed must be an integer")
        try:
            self.seed = int(operator.index(seed))
        except TypeError as exc:
            raise SamplerContractError("seed must be an integer") from exc

        try:
            row_groups = tuple(dataset.index.sample_row_groups)
            views_per_material = _contract_int(
                dataset.index.views_per_material,
                "dataset views_per_material",
                minimum=1,
                error_type=SamplerContractError,
            )
        except AttributeError as exc:
            raise SamplerContractError(
                "dataset must expose a validated BundleIndex as .index"
            ) from exc
        if self.configured_batch_size > views_per_material:
            raise SamplerContractError(
                "configured_batch_size cannot exceed views_per_material"
            )

        self.lane_count = self.num_processes * max(self.num_workers, 1)
        if len(row_groups) < self.lane_count:
            raise SamplerContractError(
                "sample row-group count must be at least the rank/worker lane count"
            )
        expected_start = 0
        for location in row_groups:
            if location.rows != views_per_material:
                raise SamplerContractError(
                    "all sample row groups must match views_per_material"
                )
            if location.global_start != expected_start:
                raise SamplerContractError(
                    "sample row groups must form a contiguous logical index"
                )
            expected_start += location.rows

        quotient, remainder = divmod(len(row_groups), self.lane_count)
        lane_group_counts = tuple(
            quotient + (lane_id < remainder) for lane_id in range(self.lane_count)
        )
        self.lane_rows = tuple(
            group_count * views_per_material for group_count in lane_group_counts
        )
        max_lane_rows = max(self.lane_rows)
        self.rounds = (
            max_lane_rows + self.configured_batch_size - 1
        ) // self.configured_batch_size
        for lane_rows in self.lane_rows:
            quotient, remainder = divmod(lane_rows, self.rounds)
            if quotient == 0:
                raise SamplerContractError("lane geometry would produce an empty batch")
            largest_batch = quotient + (remainder > 0)
            if largest_batch > self.configured_batch_size:
                raise SamplerContractError(
                    "lane geometry exceeds configured_batch_size"
                )

        self._row_groups = row_groups
        self._pass_idx = 0

    @property
    def pass_idx(self) -> int:
        return self._pass_idx

    def set_epoch(self, pass_idx: int) -> None:
        self._pass_idx = _contract_int(
            pass_idx,
            "pass_idx",
            minimum=0,
            error_type=SamplerContractError,
        )

    def __len__(self) -> int:
        return self.rounds * self.lane_count

    def __iter__(self):
        pass_idx = self._pass_idx
        rng = random.Random(self.seed + pass_idx)
        shuffled_indices = list(range(sum(self.lane_rows)))
        rng.shuffle(shuffled_indices)

        lanes: list[list[int]] = []
        sample_offset = 0
        for lane_rows in self.lane_rows:
            sample_stop = sample_offset + lane_rows
            lanes.append(shuffled_indices[sample_offset:sample_stop])
            sample_offset = sample_stop

        lane_offsets = [0] * self.lane_count
        lane_batch_sizes: list[tuple[int, int]] = []
        for lane_rows in self.lane_rows:
            quotient, remainder = divmod(lane_rows, self.rounds)
            lane_batch_sizes.append((quotient, remainder))

        for round_index in range(self.rounds):
            for lane_id, lane in enumerate(lanes):
                quotient, remainder = lane_batch_sizes[lane_id]
                size = quotient + (round_index < remainder)
                start = lane_offsets[lane_id]
                stop = start + size
                lane_offsets[lane_id] = stop
                expected_worker_id = lane_id // self.num_processes
                yield [
                    _PBRFetchKey(
                        sample_index=sample_index,
                        pass_idx=pass_idx,
                        expected_worker_id=expected_worker_id,
                    )
                    for sample_index in lane[start:stop]
                ]
        if any(offset != len(lane) for offset, lane in zip(lane_offsets, lanes)):
            raise RuntimeError("internal PBR Parquet lane planning failure")

    def validate_dataloader(self, dataloader: Any) -> None:
        if getattr(dataloader, "dataset", None) is not self.dataset:
            raise SamplerContractError(
                "DataLoader dataset does not match the batch sampler dataset"
            )
        if getattr(dataloader, "batch_sampler", None) is not self:
            raise SamplerContractError(
                "DataLoader must retain the original PBRParquetBatchSampler"
            )
        if getattr(dataloader, "batch_size", None) is not None:
            raise SamplerContractError(
                "DataLoader batch_size must be None with the lane sampler"
            )
        if getattr(dataloader, "drop_last", False) is not False:
            raise SamplerContractError("DataLoader drop_last must be False")
        if getattr(dataloader, "num_workers", None) != self.num_workers:
            raise SamplerContractError(
                "DataLoader num_workers does not match the lane sampler"
            )
        if getattr(dataloader, "persistent_workers", False):
            raise SamplerContractError(
                "PBR Parquet v1 requires persistent_workers=False"
            )


__all__ = [
    "FORMAT_VERSION",
    "REQUIRED_PYARROW_VERSION",
    "MARKER_NAME",
    "WRITER_NAME",
    "WRITER_VERSION",
    "SAMPLES_TABLE",
    "MATERIALS_TABLE",
    "TABLE_TYPES",
    "SORT_ORDERS",
    "TARGET_SHARD_BYTES",
    "HARD_SHARD_BYTES",
    "PNG_MAGIC",
    "JPEG_MAGIC",
    "PBRParquetError",
    "MarkerError",
    "SchemaError",
    "ShardError",
    "IntegrityError",
    "SourceDataError",
    "PublicationError",
    "SamplerContractError",
    "ShardManifest",
    "RowGroupLocation",
    "BundleIndex",
    "PBRParquetDataset",
    "PBRParquetBatchSampler",
    "canonical_json_bytes",
    "schema_contract",
    "schema_fingerprint",
    "parquet_schema",
    "sha256_file",
    "validate_success_marker",
    "load_success_marker",
    "validate_parquet_schema",
    "inspect_bundle",
]
