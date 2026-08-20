#!/usr/bin/env python3
"""Derive exact per-query selectivity distributions from resident bitmap rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

BITMAP_HEADER = struct.Struct("<8sIIQQQ")
MATRIX_HEADER = struct.Struct("<II")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_workload(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected WORKLOAD=/path/to/manifest.json")
    return name, Path(path).resolve()


def resolve_source(manifest_path: Path, value: object) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def shard_source(
    manifest_path: Path, shard: dict[str, object], key: str, default_name: str
) -> Path:
    value = shard.get(key)
    if value is None:
        value = Path(str(shard["directory"])) / default_name
    return resolve_source(manifest_path, value)


def count_valid_ground_truth(
    path: Path, expected_rows: int, expected_k: int, base_rows: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(MATRIX_HEADER.size)
        digest.update(header)
        if len(header) != MATRIX_HEADER.size:
            raise ValueError(f"truncated ground-truth header: {path}")
        rows, columns = MATRIX_HEADER.unpack(header)
        expected_bytes = MATRIX_HEADER.size + rows * columns * 4
        if (
            rows != expected_rows
            or columns != expected_k
            or path.stat().st_size != expected_bytes
        ):
            raise ValueError(f"ground-truth matrix violates manifest contract: {path}")
        payload = stream.read()
        digest.update(payload)
    values = np.frombuffer(payload, dtype="<u4")
    return int(np.count_nonzero(values < base_rows)), digest.hexdigest()


def count_bitmap_rows(
    path: Path, expected_rows: int, expected_cols: int
) -> tuple[np.ndarray, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(BITMAP_HEADER.size)
        digest.update(header)
        if len(header) != BITMAP_HEADER.size:
            raise ValueError(f"truncated bitmap header: {path}")
        magic, version, word_bits, rows, cols, words = BITMAP_HEADER.unpack(header)
        expected_words = (rows * cols + word_bits - 1) // word_bits
        if (
            magic != b"CUVSBMAP"
            or version != 1
            or word_bits != 32
            or rows != expected_rows
            or cols != expected_cols
            or words != expected_words
            or path.stat().st_size != BITMAP_HEADER.size + words * 4
        ):
            raise ValueError(f"bitmap header violates manifest contract: {path}")

        # CUVSBMAP packs the complete row-major bit matrix continuously.  The common aligned case
        # can therefore stream whole rows; an unaligned row begins inside the prior row's last word.
        if cols % word_bits:
            payload = np.memmap(
                path,
                dtype="<u4",
                mode="r",
                offset=BITMAP_HEADER.size,
                shape=(words,),
            )
            counts = np.empty(rows, dtype=np.uint64)
            for row in range(rows):
                first_bit = row * cols
                end_bit = first_bit + cols
                first_word = first_bit // word_bits
                end_word = (end_bit + word_bits - 1) // word_bits
                values = np.asarray(payload[first_word:end_word])
                count = int(np.bitwise_count(values).sum(dtype=np.uint64))
                first_offset = first_bit % word_bits
                if first_offset:
                    count -= int(
                        np.uint32(values[0])
                        & np.uint32((1 << first_offset) - 1)
                    ).bit_count()
                end_offset = end_bit % word_bits
                if end_offset:
                    count -= int(
                        np.uint32(values[-1])
                        & np.uint32(~((1 << end_offset) - 1) & 0xFFFFFFFF)
                    ).bit_count()
                counts[row] = count
            used_bits = (rows * cols) % word_bits
            if used_bits and (
                np.uint32(payload[-1])
                & np.uint32(~((1 << used_bits) - 1) & 0xFFFFFFFF)
            ):
                raise ValueError(f"nonzero padding bits in bitmap: {path}")
            del payload
            return counts, sha256(path)

        words_per_row = cols // word_bits
        row_bytes = words_per_row * 4
        counts = np.empty(rows, dtype=np.uint64)
        cursor = 0
        rows_per_chunk = max(1, min(64, (64 * 1024 * 1024) // row_bytes))
        while cursor < rows:
            chunk_rows = min(rows_per_chunk, rows - cursor)
            block = stream.read(chunk_rows * row_bytes)
            if len(block) != chunk_rows * row_bytes:
                raise ValueError(f"truncated bitmap payload: {path}")
            digest.update(block)
            values = np.frombuffer(block, dtype="<u4").reshape(
                chunk_rows, words_per_row
            )
            counts[cursor : cursor + chunk_rows] = np.bitwise_count(values).sum(
                axis=1, dtype=np.uint64
            )
            cursor += chunk_rows
        if stream.read(1):
            raise ValueError(f"bitmap has trailing bytes: {path}")
    return counts, digest.hexdigest()


def summarize(name: str, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("bitmap_schema") != "CUVSBMAP/v1/u32/row-major"
    ):
        raise ValueError(f"unsupported bitmap schema: {manifest_path}")
    base_rows = int(manifest["base_rows"])
    query_rows = int(manifest["query_rows"])
    all_counts: list[np.ndarray] = []
    bitmap_sources: list[dict[str, object]] = []
    ground_truth_sources: list[dict[str, object]] = []
    valid_gt_slots = 0
    cursor = 0
    for shard_number, shard in enumerate(manifest["shards"]):
        first_query = int(shard["first_query"])
        shard_queries = int(shard["query_count"])
        if first_query != cursor or shard_queries <= 0:
            raise ValueError(f"noncontiguous query shards in {manifest_path}")
        bitmap_path = shard_source(manifest_path, shard, "bitmap", "filter.bitmap")
        counts, bitmap_hash = count_bitmap_rows(
            bitmap_path, shard_queries, base_rows
        )
        ground_truth_path = shard_source(
            manifest_path, shard, "groundtruth", "groundtruth.ibin"
        )
        shard_valid_gt, ground_truth_hash = count_valid_ground_truth(
            ground_truth_path, shard_queries, 10, base_rows
        )
        valid_gt_slots += shard_valid_gt
        if int(counts.min()) != int(shard["min_passing"]):
            raise ValueError(f"min-passing mismatch for {bitmap_path}")
        if int(counts.max()) != int(shard["max_passing"]):
            raise ValueError(f"max-passing mismatch for {bitmap_path}")
        if not np.isclose(
            counts.mean() / base_rows,
            float(shard["mean_selectivity"]),
            rtol=0.0,
            atol=5e-16,
        ):
            raise ValueError(f"mean-selectivity mismatch for {bitmap_path}")
        all_counts.append(counts)
        bitmap_sources.append(
            {
                "shard_number": shard_number,
                "first_query": first_query,
                "query_count": shard_queries,
                "path": str(bitmap_path),
                "bytes": bitmap_path.stat().st_size,
                "sha256": bitmap_hash,
            }
        )
        ground_truth_sources.append(
            {
                "shard_number": shard_number,
                "first_query": first_query,
                "query_count": shard_queries,
                "path": str(ground_truth_path),
                "bytes": ground_truth_path.stat().st_size,
                "sha256": ground_truth_hash,
            }
        )
        cursor += shard_queries
    if cursor != query_rows:
        raise ValueError(f"query coverage mismatch in {manifest_path}")

    counts = np.concatenate(all_counts)
    selectivity = counts.astype(np.float64) / base_rows
    return {
        "workload": name,
        "base_rows": base_rows,
        "query_rows": query_rows,
        "passing_count": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "p90": float(np.percentile(counts, 90)),
            "max": int(counts.max()),
        },
        "selectivity": {
            "mean": float(selectivity.mean()),
            "median": float(np.median(selectivity)),
            "p90": float(np.percentile(selectivity, 90)),
            "fraction_below_1_percent": float(np.mean(selectivity < 0.01)),
            "fraction_at_most_1_percent": float(np.mean(selectivity <= 0.01)),
        },
        "valid_ground_truth": {
            "k": 10,
            "slots": valid_gt_slots,
            "fraction": valid_gt_slots / (query_rows * 10),
        },
        "source_manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "source_bitmaps": bitmap_sources,
        "source_ground_truth": ground_truth_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        action="append",
        type=parse_workload,
        required=True,
        metavar="NAME=MANIFEST",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.workload]
    if len(names) != len(set(names)):
        raise ValueError("workload names must be unique")
    result = {
        "schema_version": 1,
        "definition": "passing bitmap population divided by base_rows, per query",
        "workloads": [summarize(name, path) for name, path in args.workload],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
