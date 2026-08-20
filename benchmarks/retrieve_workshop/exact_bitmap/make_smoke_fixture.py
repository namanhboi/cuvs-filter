#!/usr/bin/env python3
"""Create deterministic fixtures for both exact-bitmap execution paths."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from gpu_memory_preflight import estimate_gpu_memory

MATRIX_HEADER = struct.Struct("<II")
BITMAP_HEADER = struct.Struct("<8sIIQQQ")


def write_matrix(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(MATRIX_HEADER.pack(*values.shape))
        values.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("sparse", "dense"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, dim, queries, k = 256, 8, 4, 10
    ids = np.arange(rows, dtype=np.float32)
    base = np.stack(
        [
            ids,
            ids % 17,
            ids % 13,
            ids % 11,
            ids % 7,
            ids % 5,
            ids % 3,
            ids % 2,
        ],
        axis=1,
    ).astype("<f4")
    query_ids = np.asarray([44, 117, 203, 9], dtype=np.int64)
    query_vectors = base[query_ids].copy()
    if args.mode == "sparse":
        passing = [
            np.asarray([1, 8, 21, 44, 57, 89, 110, 143, 201, 250]),
            np.asarray([3, 17, 68, 90, 117, 190, 240]),
            np.asarray([5, 203]),
            np.asarray([0, 2, 9, 31, 77, 101, 155, 199, 241]),
        ]
    else:
        # 130/1024 bits pass: dense/tiled path (>10%), one underfilled row, one empty row.
        passing = [
            np.arange(0, 128, 2, dtype=np.int64),
            np.arange(1, 128, 2, dtype=np.int64),
            np.asarray([5, 203]),
            np.asarray([], dtype=np.int64),
        ]

    gt = np.empty((queries, k), dtype="<u4")
    for query, candidates in enumerate(passing):
        squared_l2 = np.sum(
            (base[candidates] - query_vectors[query]) ** 2,
            axis=1,
            dtype=np.float32,
        )
        order = np.lexsort((candidates, squared_l2))
        exact = candidates[order][:k].astype("<u4")
        gt[query, : exact.size] = exact
        for rank in range(exact.size, k):
            gt[query, rank] = np.uint32(
                np.iinfo(np.uint32).max - (rank - exact.size)
            )

    base_path = args.output / "base.fbin"
    query_path = args.output / "query.fbin"
    gt_path = args.output / "groundtruth.ibin"
    bitmap_path = args.output / "filter.bitmap"
    write_matrix(base_path, base)
    write_matrix(query_path, query_vectors)
    write_matrix(gt_path, gt)

    words = (queries * rows + 31) // 32
    payload = np.zeros(words, dtype="<u4")
    for query, candidates in enumerate(passing):
        flat = query * rows + candidates
        np.bitwise_or.at(
            payload,
            flat >> 5,
            np.left_shift(np.uint32(1), (flat & 31).astype(np.uint32)),
        )
    with bitmap_path.open("wb") as stream:
        stream.write(
            BITMAP_HEADER.pack(b"CUVSBMAP", 1, 32, queries, rows, words)
        )
        payload.tofile(stream)

    passing_count = sum(map(len, passing))
    memory_estimate = estimate_gpu_memory(
        base_rows=rows,
        dim=dim,
        query_rows=queries,
        k=k,
        bitmap_storage_bytes=payload.nbytes,
        passing_count=passing_count,
    )

    manifest = {
        "schema_version": 1,
        "method": "cuvs_brute_force_bitmap",
        "timing_contract": (
            "resident bitmap; timed search includes bitmap counting, query norms, optional "
            "bitmap-to-CSR and CSR-to-COO construction, masked/tiled distance evaluation, "
            "distance/top-k epilogues, invalid-sentinel normalization, and temporary "
            "allocation/deallocation"
        ),
        "source_bitmap_manifest": None,
        "source_dtype": "float32",
        "search_dtype": "float32",
        "conversion": "none",
        "base_file": str(base_path.resolve()),
        "base_rows": rows,
        "dim": dim,
        "query_rows": queries,
        "shards": [
            {
                "shard_number": 0,
                "first_query": 0,
                "query_count": queries,
                "query_file": str(query_path.resolve()),
                "groundtruth_file": str(gt_path.resolve()),
                "bitmap_file": str(bitmap_path.resolve()),
                "min_passing": min(map(len, passing)),
                "max_passing": max(map(len, passing)),
                "mean_selectivity": sum(map(len, passing)) / (queries * rows),
                "empty_queries": sum(len(row) == 0 for row in passing),
                "passing_count": passing_count,
                "gpu_memory_estimate": memory_estimate,
            }
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
