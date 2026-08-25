#!/usr/bin/env python3
"""Prepare provenance-bound resident-bitmap workloads for cuVS exact search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
from gpu_memory_preflight import estimate_gpu_memory, validate_estimate

MATRIX_HEADER = struct.Struct("<II")
BITMAP_HEADER = struct.Struct("<8sIIQQQ")
BITMAP_MAGIC = b"CUVSBMAP"
DEFAULT_K = 10
CONVERSION_SCHEMA_VERSION = 1
TIMING_CONTRACT = (
    "resident bitmap; timed search includes bitmap counting, query norms, optional "
    "bitmap-to-CSR and CSR-to-COO construction, masked/tiled distance evaluation, "
    "distance/top-k epilogues, invalid-sentinel normalization, and temporary "
    "allocation/deallocation"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    path = path.resolve()
    stat = path.stat()
    cache_name = os.environ.get("RETRIEVE_HASH_CACHE", "").strip()
    cache_path = Path(cache_name).resolve() if cache_name else None
    cache: dict[str, object] = {}
    key = str(path)
    if cache_path is not None and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
        row = cache.get(key, {})
        if (
            int(row.get("bytes", -1)) == stat.st_size
            and int(row.get("mtime_ns", -1)) == stat.st_mtime_ns
            and isinstance(row.get("sha256"), str)
        ):
            return {"path": key, "bytes": stat.st_size, "sha256": row["sha256"]}
    digest = sha256(path)
    if cache_path is not None:
        cache[key] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        atomic_write_json(cache_path, cache)
    return {
        "path": key,
        "bytes": stat.st_size,
        "sha256": digest,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def matrix_info(path: Path, dtype: np.dtype) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(MATRIX_HEADER.size)
    if len(header) != MATRIX_HEADER.size:
        raise ValueError(f"truncated matrix header: {path}")
    rows, cols = MATRIX_HEADER.unpack(header)
    expected = MATRIX_HEADER.size + rows * cols * dtype.itemsize
    if rows == 0 or cols == 0 or path.stat().st_size != expected:
        raise ValueError(f"matrix size does not match header: {path}")
    return rows, cols


def bitmap_info(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as stream:
        header = stream.read(BITMAP_HEADER.size)
    if len(header) != BITMAP_HEADER.size:
        raise ValueError(f"truncated bitmap header: {path}")
    magic, version, word_bits, rows, cols, words = BITMAP_HEADER.unpack(header)
    if magic != BITMAP_MAGIC or version != 1 or word_bits != 32:
        raise ValueError(f"unsupported bitmap format: {path}")
    expected_words = (rows * cols + word_bits - 1) // word_bits
    expected_bytes = BITMAP_HEADER.size + expected_words * (word_bits // 8)
    if words != expected_words or path.stat().st_size != expected_bytes:
        raise ValueError(f"bitmap size does not match header: {path}")
    return rows, cols, words


def bitmap_passing_count(path: Path, words: int, logical_bits: int) -> int:
    """Count logical set bits in bounded host chunks without materializing the bitmap."""
    payload = np.memmap(
        path,
        dtype="<u4",
        mode="r",
        offset=BITMAP_HEADER.size,
        shape=(words,),
    )
    byte_popcount = np.asarray(
        [value.bit_count() for value in range(256)], dtype=np.uint8
    )
    total = 0
    chunk_words = 1 << 20
    for first in range(0, words, chunk_words):
        values = np.asarray(payload[first : min(words, first + chunk_words)])
        total += int(byte_popcount[values.view(np.uint8)].sum(dtype=np.uint64))
    tail = logical_bits % 32
    if tail and words:
        invalid = int(payload[-1]) & ~((1 << tail) - 1)
        if invalid:
            raise ValueError(f"bitmap has nonzero padding bits: {path}")
    return total


def conversion_sidecar(target: Path) -> Path:
    return target.with_name(f"{target.name}.conversion.json")


def validate_conversion_record(
    record: dict[str, object],
    source_record: dict[str, object],
    target: Path,
    rows: int,
    cols: int,
) -> dict[str, object]:
    if (
        int(record.get("schema_version", -1)) != CONVERSION_SCHEMA_VERSION
        or record.get("conversion") != "uint8_to_float32_coordinate_exact"
        or record.get("source") != source_record
        or int(record.get("rows", -1)) != rows
        or int(record.get("cols", -1)) != cols
    ):
        raise ValueError(
            "conversion provenance does not match its current source"
        )
    target_record = record.get("target")
    if not isinstance(target_record, dict) or target_record != file_record(
        target
    ):
        raise ValueError(
            "converted target content does not match recorded provenance"
        )
    if matrix_info(target, np.dtype("<f4")) != (rows, cols):
        raise ValueError("converted target matrix shape is invalid")
    return record


def convert_uint8_matrix(
    source: Path, target: Path, chunk_rows: int, force: bool
) -> dict[str, object]:
    rows, cols = matrix_info(source, np.dtype("uint8"))
    source_provenance = file_record(source)
    sidecar = conversion_sidecar(target)
    if target.exists() and sidecar.exists():
        try:
            return validate_conversion_record(
                json.loads(sidecar.read_text()),
                source_provenance,
                target,
                rows,
                cols,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            if not force:
                raise ValueError(
                    f"converted matrix cache does not match its source: {target}; "
                    "rerun with --force"
                )
    elif (target.exists() or sidecar.exists()) and not force:
        raise ValueError(
            f"incomplete converted matrix cache: {target}; rerun with --force"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    expected_size = (
        MATRIX_HEADER.size + rows * cols * np.dtype("float32").itemsize
    )
    source_values = np.memmap(
        source,
        dtype="uint8",
        mode="r",
        offset=MATRIX_HEADER.size,
        shape=(rows, cols),
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(MATRIX_HEADER.pack(rows, cols))
            stream.truncate(expected_size)
        target_values = np.memmap(
            temporary,
            dtype="<f4",
            mode="r+",
            offset=MATRIX_HEADER.size,
            shape=(rows, cols),
        )
        for first in range(0, rows, chunk_rows):
            end = min(rows, first + chunk_rows)
            target_values[first:end] = source_values[first:end]
        target_values.flush()
        del target_values
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    record: dict[str, object] = {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "conversion": "uint8_to_float32_coordinate_exact",
        "source": source_provenance,
        "target": file_record(target),
        "rows": rows,
        "cols": cols,
    }
    atomic_write_json(sidecar, record)
    return record


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return (
        path if path.is_absolute() else manifest_path.parent / path
    ).resolve()


def validate_cached_manifest(
    cached: dict[str, object],
    *,
    preparation: dict[str, object],
    source_dtype: str,
    base_rows: int,
    dim: int,
    query_total: int,
    search_base: Path,
    source_shards: list[dict[str, object]],
) -> None:
    if (
        cached.get("method") != "cuvs_brute_force_bitmap"
        or cached.get("timing_contract") != TIMING_CONTRACT
        or cached.get("timed_invalid_sentinel_normalization") is not True
        or cached.get("preparation") != preparation
        or cached.get("source_dtype") != source_dtype
        or cached.get("search_dtype") != "float32"
        or int(cached.get("base_rows", -1)) != base_rows
        or int(cached.get("dim", -1)) != dim
        or int(cached.get("query_rows", -1)) != query_total
        or Path(str(cached.get("base_file", ""))).resolve()
        != search_base.resolve()
    ):
        raise ValueError(
            "prepared manifest provenance/shape does not match current inputs"
        )
    if matrix_info(search_base, np.dtype("<f4")) != (base_rows, dim):
        raise ValueError("cached search base is missing or malformed")
    if source_dtype == "uint8":
        base_conversion = cached.get("base_conversion")
        if not isinstance(base_conversion, dict):
            raise ValueError("cached uint8 base lacks conversion provenance")
        source_base = preparation.get("source_base")
        if not isinstance(source_base, dict):
            raise ValueError("malformed source-base provenance")
        validate_conversion_record(
            base_conversion, source_base, search_base, base_rows, dim
        )
    cached_shards = cached.get("shards")
    if not isinstance(cached_shards, list) or len(cached_shards) != len(
        source_shards
    ):
        raise ValueError(
            "prepared shard set does not match the source manifest"
        )
    for cached_shard, source_shard in zip(
        cached_shards, source_shards, strict=True
    ):
        if not isinstance(cached_shard, dict):
            raise TypeError("malformed cached shard")
        for key in (
            "shard_number",
            "first_query",
            "query_count",
            "groundtruth_file",
            "bitmap_file",
        ):
            if cached_shard.get(key) != source_shard.get(key):
                raise ValueError(
                    f"cached shard field {key} no longer matches its source"
                )
        query_file = Path(str(cached_shard["query_file"]))
        if matrix_info(query_file, np.dtype("<f4")) != (
            int(source_shard["query_count"]),
            dim,
        ):
            raise ValueError("cached query conversion is missing or malformed")
        if source_dtype == "uint8":
            query_conversion = cached_shard.get("query_conversion")
            if not isinstance(query_conversion, dict):
                raise ValueError(
                    "cached uint8 query lacks conversion provenance"
                )
            validate_conversion_record(
                query_conversion,
                file_record(Path(str(source_shard["source_query"]))),
                query_file,
                int(source_shard["query_count"]),
                dim,
            )
        estimate = cached_shard.get("gpu_memory_estimate")
        if not isinstance(estimate, dict):
            raise TypeError("cached shard lacks GPU-memory estimate")
        validate_estimate(estimate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-manifest", type=Path, required=True)
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument(
        "--source-dtype", choices=("float32", "uint8"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--converted-base",
        type=Path,
        help="shared float32 base output; required for a uint8 workload",
    )
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace stale prepared metadata/conversions after provenance validation fails",
    )
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.source_dtype == "uint8" and args.converted_base is None:
        parser.error("--converted-base is required for uint8 workloads")

    args.base_file = args.base_file.resolve()
    args.bitmap_manifest = args.bitmap_manifest.resolve()
    args.output = args.output.resolve()
    source_dtype = np.dtype("uint8" if args.source_dtype == "uint8" else "<f4")
    base_rows, dim = matrix_info(args.base_file, source_dtype)
    source_manifest = json.loads(args.bitmap_manifest.read_text())
    if source_manifest.get("bitmap_schema") != "CUVSBMAP/v1/u32/row-major":
        raise ValueError("input does not describe a CUVSBMAP/v1 bitmap")
    if int(source_manifest["base_rows"]) != base_rows:
        raise ValueError("bitmap width does not match base-vector rows")

    preparation = {
        "schema_version": 1,
        "source_base": file_record(args.base_file),
        "source_bitmap_manifest": file_record(args.bitmap_manifest),
        "ground_truth_k": args.k,
    }
    source_shards: list[dict[str, object]] = []
    query_total = 0
    for shard_number, shard in enumerate(source_manifest["shards"]):
        if int(shard["first_query"]) != query_total:
            raise ValueError(
                f"source shard {shard_number} starts at {shard['first_query']}, "
                f"expected contiguous offset {query_total}"
            )
        source_directory = resolve_manifest_path(
            args.bitmap_manifest, shard["directory"]
        )
        source_query = source_directory / "query.bin"
        source_gt = source_directory / "groundtruth.ibin"
        source_bitmap = resolve_manifest_path(
            args.bitmap_manifest, shard["bitmap"]
        )
        query_rows, query_dim = matrix_info(source_query, source_dtype)
        gt_rows, gt_width = matrix_info(source_gt, np.dtype("<u4"))
        bitmap_rows, bitmap_cols, bitmap_words = bitmap_info(source_bitmap)
        expected_rows = int(shard["query_count"])
        if (
            query_rows != expected_rows
            or gt_rows != expected_rows
            or bitmap_rows != expected_rows
            or query_dim != dim
            or bitmap_cols != base_rows
            or gt_width != args.k
        ):
            raise ValueError(
                f"inconsistent source shard {shard_number}; "
                f"ground truth must have exactly k={args.k}"
            )
        source_shards.append(
            {
                "shard_number": shard_number,
                "first_query": int(shard["first_query"]),
                "query_count": expected_rows,
                "source_query": str(source_query.resolve()),
                "groundtruth_file": str(source_gt.resolve()),
                "bitmap_file": str(source_bitmap.resolve()),
                "bitmap_words": bitmap_words,
                "min_passing": int(shard["min_passing"]),
                "max_passing": int(shard["max_passing"]),
                "mean_selectivity": float(shard["mean_selectivity"]),
                "empty_queries": int(shard["empty_queries"]),
            }
        )
        query_total += expected_rows
    if query_total != int(source_manifest["query_rows"]):
        raise ValueError(
            "source shard query counts do not match manifest total"
        )

    if args.source_dtype == "uint8":
        assert args.converted_base is not None
        search_base = args.converted_base.resolve()
        conversion = "uint8_to_float32_coordinate_exact"
    else:
        search_base = args.base_file
        conversion = "none"

    cached_path = args.output / "manifest.json"
    if cached_path.exists() and not args.force:
        try:
            cached = json.loads(cached_path.read_text())
            validate_cached_manifest(
                cached,
                preparation=preparation,
                source_dtype=args.source_dtype,
                base_rows=base_rows,
                dim=dim,
                query_total=query_total,
                search_base=search_base,
                source_shards=source_shards,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                f"prepared cache is stale or unproven: {cached_path}; rerun with --force"
            ) from error
        print(json.dumps(cached, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    base_conversion: dict[str, object] | None = None
    if args.source_dtype == "uint8":
        base_conversion = convert_uint8_matrix(
            args.base_file, search_base, args.chunk_rows, args.force
        )

    output_shards: list[dict[str, object]] = []
    for source_shard in source_shards:
        shard_number = int(source_shard["shard_number"])
        expected_rows = int(source_shard["query_count"])
        source_query = Path(str(source_shard["source_query"]))
        source_bitmap = Path(str(source_shard["bitmap_file"]))
        if args.source_dtype == "uint8":
            target_query = (
                args.output / f"shard_{shard_number:02d}" / "query.fbin"
            )
            query_conversion = convert_uint8_matrix(
                source_query, target_query, args.chunk_rows, args.force
            )
            search_query = target_query.resolve()
        else:
            query_conversion = None
            search_query = source_query

        passing_count = bitmap_passing_count(
            source_bitmap,
            int(source_shard["bitmap_words"]),
            expected_rows * base_rows,
        )
        memory_estimate = estimate_gpu_memory(
            base_rows=base_rows,
            dim=dim,
            query_rows=expected_rows,
            k=args.k,
            bitmap_storage_bytes=source_bitmap.stat().st_size
            - BITMAP_HEADER.size,
            passing_count=passing_count,
        )
        output_shards.append(
            {
                "shard_number": shard_number,
                "first_query": int(source_shard["first_query"]),
                "query_count": expected_rows,
                "query_file": str(search_query.resolve()),
                "query_conversion": query_conversion,
                "groundtruth_file": str(source_shard["groundtruth_file"]),
                "bitmap_file": str(source_bitmap.resolve()),
                "min_passing": int(source_shard["min_passing"]),
                "max_passing": int(source_shard["max_passing"]),
                "mean_selectivity": float(source_shard["mean_selectivity"]),
                "empty_queries": int(source_shard["empty_queries"]),
                "passing_count": passing_count,
                "gpu_memory_estimate": memory_estimate,
            }
        )

    output_manifest: dict[str, object] = {
        "schema_version": 1,
        "method": "cuvs_brute_force_bitmap",
        "timing_contract": TIMING_CONTRACT,
        "timed_invalid_sentinel_normalization": True,
        "source_bitmap_manifest": str(args.bitmap_manifest),
        "preparation": preparation,
        "source_dtype": args.source_dtype,
        "search_dtype": "float32",
        "conversion": conversion,
        "base_conversion": base_conversion,
        "base_file": str(search_base.resolve()),
        "base_rows": base_rows,
        "dim": dim,
        "query_rows": query_total,
        "k": args.k,
        "gpu_memory_preflight_contract": (
            "each shard records a modeled peak plus max(2 GiB,20%) safety; the runner "
            "requires at least required_free_device_bytes before launching the benchmark"
        ),
        "shards": output_shards,
    }
    atomic_write_json(cached_path, output_manifest)
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
