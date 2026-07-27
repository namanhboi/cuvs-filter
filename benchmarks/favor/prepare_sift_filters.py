#!/usr/bin/env python3
"""Create deterministic exact-selectivity bitsets and filtered ground truth."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cupy as cp
import numpy as np


DEFAULT_SELECTIVITIES = (0.01, 0.10, 0.50, 0.90)
DTYPES = {
    "float": np.float32,
    "float32": np.float32,
    "uint8": np.uint8,
    "int8": np.int8,
}


def read_matrix(path: Path, dtype: np.dtype) -> np.memmap:
    with path.open("rb") as stream:
        rows, cols = struct.unpack("<II", stream.read(8))
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, cols))


def write_matrix(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", values.shape[0], values.shape[1]))
        values.tofile(stream)


def write_bitset(path: Path, passing: np.ndarray, rows: int) -> None:
    words = np.zeros((rows + 31) // 32, dtype=np.uint32)
    np.bitwise_or.at(words, passing // 32, np.left_shift(np.uint32(1), passing % 32))
    write_matrix(path, words.reshape(-1, 1))


def exact_ground_truth(
    base: np.memmap,
    queries: np.memmap,
    passing: np.ndarray,
    k: int,
    query_batch: int,
    base_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    result_ids = np.empty((queries.shape[0], k), dtype=np.int32)
    result_distances = np.empty((queries.shape[0], k), dtype=np.float32)

    for query_begin in range(0, queries.shape[0], query_batch):
        query_end = min(query_begin + query_batch, queries.shape[0])
        q = cp.asarray(queries[query_begin:query_end], dtype=cp.float32)
        q_norm = cp.sum(q * q, axis=1, keepdims=True)
        best_d = cp.full((q.shape[0], k), cp.inf, dtype=cp.float32)
        best_i = cp.full((q.shape[0], k), -1, dtype=cp.int32)

        for base_begin in range(0, passing.size, base_chunk):
            ids_host = passing[base_begin : base_begin + base_chunk]
            x = cp.asarray(np.asarray(base[ids_host]), dtype=cp.float32)
            distances = cp.maximum(
                q_norm + cp.sum(x * x, axis=1)[None, :] - 2.0 * (q @ x.T), 0.0
            )
            ids = cp.asarray(ids_host.astype(np.int32, copy=False))
            merged_d = cp.concatenate((best_d, distances), axis=1)
            merged_i = cp.concatenate(
                (best_i, cp.broadcast_to(ids[None, :], distances.shape)), axis=1
            )
            positions = cp.argpartition(merged_d, k - 1, axis=1)[:, :k]
            best_d = cp.take_along_axis(merged_d, positions, axis=1)
            best_i = cp.take_along_axis(merged_i, positions, axis=1)

        order = cp.argsort(best_d, axis=1)
        result_distances[query_begin:query_end] = cp.asnumpy(
            cp.take_along_axis(best_d, order, axis=1)
        )
        result_ids[query_begin:query_end] = cp.asnumpy(
            cp.take_along_axis(best_i, order, axis=1)
        )
        if query_begin == 0 or query_end == queries.shape[0] or query_end % 1024 == 0:
            print(
                f"  queries {query_begin:5d}:{query_end:5d} / {queries.shape[0]}",
                flush=True,
            )

    return result_ids, result_distances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-file", default="base.fbin")
    parser.add_argument("--query-file", default="query.fbin")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--selectivities",
        type=float,
        nargs="+",
        default=DEFAULT_SELECTIVITIES,
        help="fractions in (0, 1], for example: 0.01 0.02 0.03",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-batch", type=int, default=64)
    parser.add_argument("--base-chunk", type=int, default=65536)
    parser.add_argument("--benchmark-query-count", type=int)
    parser.add_argument("--benchmark-query-file", default="query_10000.fbin")
    parser.add_argument(
        "--unfiltered-only",
        action="store_true",
        help="write exact unfiltered ground truth into the dataset directory and exit",
    )
    args = parser.parse_args()
    selectivities = tuple(dict.fromkeys(args.selectivities))
    if not selectivities or any(not 0.0 < value <= 1.0 for value in selectivities):
        raise ValueError("selectivities must contain fractions in (0, 1]")
    percents = [int(round(100 * value)) for value in selectivities]
    if any(abs(percent / 100 - value) > 1e-9 for percent, value in zip(percents, selectivities)):
        raise ValueError("selectivities must be whole percentages")
    if len(set(percents)) != len(percents):
        raise ValueError("selectivities round to duplicate percentages")

    dtype = DTYPES[args.dtype]
    base = read_matrix(args.dataset_dir / args.base_file, dtype)
    if args.subset_size is not None:
        if not 0 < args.subset_size <= base.shape[0]:
            raise ValueError("subset size must be in the base dataset row range")
        base = base[: args.subset_size]
    queries = read_matrix(args.dataset_dir / args.query_file, dtype)
    if args.unfiltered_only:
        passing = np.arange(base.shape[0], dtype=np.int64)
        print(f"unfiltered ground truth: {passing.size} base rows", flush=True)
        neighbors, distances = exact_ground_truth(
            base, queries, passing, args.k, args.query_batch, args.base_chunk
        )
        write_matrix(args.dataset_dir / "groundtruth.neighbors.ibin", neighbors)
        write_matrix(args.dataset_dir / "groundtruth.distances.fbin", distances)
        return

    output_query_count = queries.shape[0]
    if args.benchmark_query_count is not None:
        if args.benchmark_query_count < queries.shape[0]:
            raise ValueError("benchmark query count cannot be smaller than the source query count")
        query_rows = np.arange(args.benchmark_query_count) % queries.shape[0]
        write_matrix(
            args.dataset_dir / args.benchmark_query_file,
            np.asarray(queries[query_rows], dtype=dtype),
        )
        output_query_count = args.benchmark_query_count
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(base.shape[0]).astype(np.int64)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "seed": args.seed,
        "rows": int(base.shape[0]),
        "dtype": args.dtype,
        "base_file": args.base_file,
        "query_file": args.query_file,
        "source_queries": int(queries.shape[0]),
        "queries": int(output_query_count),
        "k": args.k,
        "selectivities": {},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        compatibility_keys = ("seed", "rows", "dtype", "base_file", "query_file", "k")
        if any(
            key in previous and previous[key] != manifest[key]
            for key in compatibility_keys
        ):
            raise ValueError(f"existing manifest is incompatible: {manifest_path}")
        manifest["selectivities"].update(previous.get("selectivities", {}))

    for selectivity in selectivities:
        percent = int(round(100 * selectivity))
        passing_count = int(round(selectivity * base.shape[0]))
        passing = np.sort(order[:passing_count])
        bitset_path = args.output_dir / f"filter_s{percent:02d}.bin"
        neighbors_path = args.output_dir / f"groundtruth_s{percent:02d}.ibin"
        distances_path = args.output_dir / f"groundtruth_s{percent:02d}.fbin"
        write_bitset(bitset_path, passing, base.shape[0])
        print(f"selectivity {selectivity:.0%}: {passing_count} passing rows", flush=True)
        neighbors, distances = exact_ground_truth(
            base, queries, passing, args.k, args.query_batch, args.base_chunk
        )
        if output_query_count != queries.shape[0]:
            query_rows = np.arange(output_query_count) % queries.shape[0]
            neighbors = neighbors[query_rows]
            distances = distances[query_rows]
        write_matrix(neighbors_path, neighbors)
        write_matrix(distances_path, distances)
        manifest["selectivities"][f"{selectivity:.2f}"] = {
            "passing_count": passing_count,
            "bitset": bitset_path.name,
            "neighbors": neighbors_path.name,
            "distances": distances_path.name,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
