#!/usr/bin/env python3
"""Prepare and validate masked cuVS Brute Force KNN ground truth for YFCC Recall@100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXACT_DIR = SCRIPT_DIR.parent / "exact_bitmap"
sys.path.insert(0, str(EXACT_DIR))

from prepare_exact_workload import (
    BITMAP_HEADER,
    bitmap_info,
    convert_uint8_matrix,
    matrix_info,
    resolve_manifest_path,
)

K = 100
MATRIX_HEADER = struct.Struct("<II")
INVALID = int(np.iinfo(np.uint32).max)
BYTE_POPCOUNT = np.asarray(
    [value.bit_count() for value in range(256)], dtype=np.uint8
)


def file_fingerprint(path: Path, *, hash_content: bool) -> dict[str, object]:
    path = path.resolve()
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def read_matrix(path: Path, dtype: np.dtype) -> np.memmap:
    rows, cols = matrix_info(path, dtype)
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=MATRIX_HEADER.size,
        shape=(rows, cols),
    )


def read_yfcc_official_gt(path: Path, query_count: int) -> np.memmap:
    with path.open("rb") as source:
        raw = source.read(MATRIX_HEADER.size)
    if len(raw) != MATRIX_HEADER.size:
        raise ValueError(f"truncated YFCC ground truth: {path}")
    rows, cols = MATRIX_HEADER.unpack(raw)
    id_bytes = rows * cols * np.dtype("<u4").itemsize
    actual = path.stat().st_size
    if (
        rows < query_count
        or cols != 10
        or actual
        not in {
            MATRIX_HEADER.size + id_bytes,
            MATRIX_HEADER.size + 2 * id_bytes,
        }
    ):
        raise ValueError(f"unsupported YFCC ground-truth layout: {path}")
    return np.memmap(
        path,
        dtype="<u4",
        mode="r",
        offset=MATRIX_HEADER.size,
        shape=(rows, cols),
    )


def config_payload(
    *,
    name: str,
    base: Path,
    query: Path,
    bitmap: Path,
    marker: Path,
    output: Path,
    query_count: int,
) -> dict[str, object]:
    return {
        "dataset": {
            "name": name,
            "base_file": str(base.resolve()),
            "query_file": str(query.resolve()),
            "distance": "euclidean",
            "dtype": "float",
            "filter": {"kind": "bitmap", "file": str(bitmap.resolve())},
        },
        "search_basic_param": {"batch_size": query_count, "k": K},
        "index": [
            {
                "name": "cuvs-yfcc-gt100",
                "algo": "cuvs_brute_force",
                "file": str(marker.resolve()),
                "build_param": {},
                "search_params": [
                    {
                        "exact_control": "bitmap_count_csr_search",
                        "resident_bitmap": True,
                        "benchmark_output_neighbors_file": str(
                            output.resolve()
                        ),
                    }
                ],
            }
        ],
    }


def prepare(args: argparse.Namespace) -> None:
    source_manifest = args.bitmap_manifest.resolve()
    source = json.loads(source_manifest.read_text())
    if source.get("bitmap_schema") != "CUVSBMAP/v1/u32/row-major":
        raise ValueError("source manifest is not a CUVSBMAP/v1 workload")
    base = args.base.resolve()
    base_rows, dim = matrix_info(base, np.dtype("uint8"))
    if (
        int(source["base_rows"]) != base_rows
        or int(source["query_rows"]) != 10_000
    ):
        raise ValueError("YFCC source manifest has unexpected geometry")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    converted_base = args.converted_base.resolve()
    base_conversion = convert_uint8_matrix(
        base, converted_base, args.chunk_rows, args.force
    )
    marker = output_root / "cuvs_brute_force.index"
    marker.touch(exist_ok=True)

    records: list[dict[str, object]] = []
    input_shards: list[dict[str, object]] = []
    cursor = 0
    for shard_number, shard in enumerate(source["shards"]):
        first = int(shard["first_query"])
        count = int(shard["query_count"])
        if first != cursor:
            raise ValueError("YFCC source shards are not contiguous")
        source_directory = resolve_manifest_path(
            source_manifest, shard["directory"]
        )
        source_query = source_directory / "query.bin"
        bitmap = resolve_manifest_path(source_manifest, shard["bitmap"])
        if matrix_info(source_query, np.dtype("uint8")) != (count, dim):
            raise ValueError(f"YFCC query geometry mismatch: {source_query}")
        bitmap_rows, bitmap_cols, _ = bitmap_info(bitmap)
        if (bitmap_rows, bitmap_cols) != (count, base_rows):
            raise ValueError(f"YFCC bitmap geometry mismatch: {bitmap}")
        shard_root = (
            output_root
            / f"shard_{shard_number:02d}_{first:05d}_{first + count:05d}"
        )
        query = shard_root / "query.fbin"
        gt = shard_root / "groundtruth.ibin"
        config = output_root / "configs" / f"shard_{shard_number:02d}.json"
        conversion = convert_uint8_matrix(
            source_query, query, args.chunk_rows, args.force
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps(
                config_payload(
                    name=f"retrieve-yfcc-gt100-q{first:05d}-{first + count:05d}",
                    base=converted_base,
                    query=query,
                    bitmap=bitmap,
                    marker=marker,
                    output=gt,
                    query_count=count,
                ),
                indent=2,
            )
            + "\n"
        )
        records.append(
            {
                "shard_number": shard_number,
                "first_query": first,
                "query_count": count,
                "config": str(config.resolve()),
                "query_file": str(query.resolve()),
                "query_conversion": conversion,
                "bitmap_file": str(bitmap.resolve()),
                "groundtruth_file": str(gt.resolve()),
            }
        )
        input_shards.append(
            {
                "shard_number": shard_number,
                "query_source": conversion["source"],
                "bitmap": file_fingerprint(bitmap, hash_content=False),
            }
        )
        cursor += count
    if cursor != 10_000:
        raise ValueError(
            f"YFCC source contains {cursor} queries, expected 10000"
        )
    input_contract = {
        "source_bitmap_manifest": file_fingerprint(
            source_manifest, hash_content=True
        ),
        "source_official_gt10": file_fingerprint(
            args.official_gt, hash_content=True
        ),
        "base_conversion_source": base_conversion["source"],
        "shards": input_shards,
    }
    manifest = {
        "schema_version": 1,
        "method": "cuvs_brute_force_knn_masked_gt_generation",
        "k": K,
        "base_rows": base_rows,
        "dim": dim,
        "query_rows": cursor,
        "source_bitmap_manifest": str(source_manifest),
        "source_official_gt10": str(args.official_gt.resolve()),
        "base_file": str(converted_base),
        "base_conversion": base_conversion,
        "input_contract": input_contract,
        "timing_use": "untimed preprocessing only; never reported as search throughput",
        "complete": False,
        "shards": records,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("input_contract") != input_contract:
            raise ValueError(
                f"YFCC GT-generation inputs changed; move or remove the stale output: {output_root}"
            )
        if previous.get("complete") is True:
            for row in previous.get("shards", []):
                path = Path(row["groundtruth_file"])
                count = int(row["query_count"])
                if matrix_info(path, np.dtype("<u4")) != (count, K):
                    raise ValueError(
                        f"completed YFCC GT cache is malformed: {path}"
                    )
            print(f"reuse {manifest_path}")
            return
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def bitmap_words(path: Path) -> tuple[np.memmap, int, int]:
    rows, cols, words = bitmap_info(path)
    return (
        np.memmap(
            path,
            dtype="<u4",
            mode="r",
            offset=BITMAP_HEADER.size,
            shape=(words,),
        ),
        rows,
        cols,
    )


def count_bitmap_row(words: np.memmap, row: int, cols: int) -> int:
    first_bit = row * cols
    last_bit = first_bit + cols
    first_word = first_bit // 32
    last_word = (last_bit - 1) // 32
    values = np.asarray(words[first_word : last_word + 1]).copy()
    values[0] &= np.uint32((0xFFFFFFFF << (first_bit & 31)) & 0xFFFFFFFF)
    tail = last_bit & 31
    if tail:
        values[-1] &= np.uint32((1 << tail) - 1)
    return int(BYTE_POPCOUNT[values.view(np.uint8)].sum(dtype=np.uint64))


def validate_shard(
    *,
    row: dict[str, object],
    official: np.ndarray,
    base_rows: int,
) -> dict[str, object]:
    first = int(row["first_query"])
    count = int(row["query_count"])
    gt_path = Path(str(row["groundtruth_file"]))
    gt = read_matrix(gt_path, np.dtype("<u4"))
    if gt.shape != (count, K):
        raise ValueError(f"generated GT has wrong shape: {gt_path}")
    words, bitmap_rows, cols = bitmap_words(Path(str(row["bitmap_file"])))
    if (bitmap_rows, cols) != (count, base_rows):
        raise ValueError("generated-GT bitmap geometry changed")

    underfilled = 0
    for local in range(count):
        ids = np.asarray(gt[local])
        invalid_positions = np.flatnonzero(ids >= base_rows)
        valid_count = (
            int(invalid_positions[0]) if invalid_positions.size else K
        )
        if np.any(ids[valid_count:] != INVALID):
            raise ValueError(
                f"invalid sentinel ordering at YFCC query {first + local}"
            )
        legal = ids[:valid_count]
        if np.unique(legal).size != valid_count:
            raise ValueError(
                f"duplicate generated GT ID at YFCC query {first + local}"
            )
        flat = np.uint64(local * cols) + legal.astype(np.uint64)
        passing = (
            words[flat >> np.uint64(5)]
            & np.left_shift(
                np.uint32(1), (flat & np.uint64(31)).astype(np.uint32)
            )
        ) != 0
        if not np.all(passing):
            raise ValueError(
                f"generated GT violates predicate at YFCC query {first + local}"
            )
        official_valid = official[first + local]
        official_valid = official_valid[official_valid < base_rows]
        if not set(map(int, official_valid)).issubset(set(map(int, legal))):
            raise ValueError(
                f"generated GT omits official top-10 ID at YFCC query {first + local}"
            )
        if valid_count < K:
            underfilled += 1
            exact_passing = count_bitmap_row(words, local, cols)
            if valid_count != exact_passing:
                raise ValueError(
                    f"underfilled exact result mismatch at YFCC query {first + local}: "
                    f"returned={valid_count}, bitmap={exact_passing}"
                )
    return {
        "query_count": count,
        "underfilled_queries": underfilled,
        "groundtruth_bytes": gt_path.stat().st_size,
    }


def finalize(args: argparse.Namespace) -> None:
    manifest_path = args.output.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        int(manifest.get("k", -1)) != K
        or int(manifest.get("query_rows", -1)) != 10_000
    ):
        raise ValueError("stale YFCC GT-generation manifest")
    official = read_yfcc_official_gt(
        Path(manifest["source_official_gt10"]), 10_000
    )
    validation = [
        validate_shard(
            row=row,
            official=official,
            base_rows=int(manifest["base_rows"]),
        )
        for row in manifest["shards"]
    ]
    validation_summary = {
        "queries": sum(int(row["query_count"]) for row in validation),
        "underfilled_queries": sum(
            int(row["underfilled_queries"]) for row in validation
        ),
        "requirements": [
            "all valid IDs are distinct, in range, and predicate-passing",
            "invalid IDs form a UINT32_MAX suffix",
            "every official YFCC top-10 ID occurs in the generated top-100",
            "underfilled rows return every bitmap-passing ID",
        ],
    }
    if (
        manifest.get("complete") is True
        and manifest.get("validation") == validation_summary
    ):
        print(f"reuse {manifest_path}")
        return
    manifest["complete"] = True
    manifest["validation"] = validation_summary
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest["validation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--bitmap-manifest", type=Path, required=True)
    prepare_parser.add_argument("--base", type=Path, required=True)
    prepare_parser.add_argument("--official-gt", type=Path, required=True)
    prepare_parser.add_argument("--converted-base", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--chunk-rows", type=int, default=65_536)
    prepare_parser.add_argument("--force", action="store_true")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.chunk_rows <= 0:
            parser.error("--chunk-rows must be positive")
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
