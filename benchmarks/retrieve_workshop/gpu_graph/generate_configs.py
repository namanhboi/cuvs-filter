#!/usr/bin/env python3
"""Generate the frozen GPU graph-search configurations for the RETRIEVE study.

The generator deliberately keeps experimental policy out of the benchmark runner.  Every
generated manifest records the exact shards and search points expected by the analyzer, so a
partial run cannot silently become a paper figure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dataset_profile import load_profile, profile_record, workload_spec

WORKLOADS = ("yfcc", "em", "emis", "r")
DEFAULT_K = 10
MAX_QUERIES = int(load_profile()["max_queries"])
THROUGHPUT_REPETITIONS = 3
CORRECTNESS_REPETITIONS = 1
B0_CELLS = ((64, 1), (128, 1), (128, 2), (256, 1), (256, 2), (512, 2))
DEEP_CELLS = ((64, 1), (512, 2))
DEEP_ITERATIONS = (522, 1044, 2088, 4176, 7569)

PRIMARY_METHODS = (
    "default_cagra",
    "default_cagra_accumulator",
    "navix_reference",
)
SEED_CONTROLS = ("default_cagra_seeded", "default_cagra_accumulator_seeded")
METHODS = PRIMARY_METHODS + SEED_CONTROLS


@dataclass(frozen=True)
class DatasetPaths:
    manifest: Path
    base_file: str
    index_file: str
    dtype: str
    graph_degree: int
    intermediate_graph_degree: int
    dataset_size: int
    dimension: int


def dataset_paths(
    data_root: Path,
    workload: str,
    phase: str,
    yfcc_graph_degree: int,
) -> DatasetPaths:
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}")
    sample_count = "1000" if phase == "correctness" else "10000"
    profile = load_profile()
    spec = workload_spec(workload, profile)
    # Retain the historical command-line sensitivity switch only for the built-in profile.  An
    # explicit profile is the single source of truth and must not be silently overridden.
    if workload == "yfcc" and profile["source"] == "built-in":
        if yfcc_graph_degree not in (32, 64):
            raise ValueError(
                "YFCC graph degree must be 32 (sensitivity) or 64 (primary)"
            )
        intermediate_graph_degree = 2 * yfcc_graph_degree
        return DatasetPaths(
            data_root
            / "navix_bitmap"
            / "yfcc"
            / f"{phase}_{sample_count}"
            / "manifest.json",
            "yfcc-10M/base.10M.u8bin",
            (
                "yfcc-10M/"
                f"cagra_g{yfcc_graph_degree}_ig{intermediate_graph_degree}.index"
            ),
            "uint8",
            yfcc_graph_degree,
            intermediate_graph_degree,
            10_000_000,
            192,
        )
    return DatasetPaths(
        data_root / str(spec["bitmap_directory"]) / f"{phase}_{sample_count}" / "manifest.json",
        str(spec["base_file"]),
        str(spec["index_file"]),
        str(spec["dtype"]),
        int(spec["graph_degree"]),
        int(spec["intermediate_graph_degree"]),
        int(spec["dataset_size"]),
        int(spec["dimension"]),
    )


def search_point(
    method: str,
    itopk: int,
    width: int,
    max_iterations: int,
    *,
    k: int = DEFAULT_K,
    max_queries: int | None = None,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    effective_max_queries = MAX_QUERIES if max_queries is None else int(max_queries)
    if effective_max_queries <= 0:
        raise ValueError("max_queries must be positive")
    row: dict[str, object] = {
        "algo": "single_cta",
        "filter_mode": "default",
        "max_queries": effective_max_queries,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": max_iterations,
        "favor_udf_passing_accumulator": "accumulator" in method,
        # The resident bitmap and GT use source-order IDs.  Reject a remapped index rather than
        # silently comparing incompatible ID spaces.
        "require_identity_source_indices": True,
        # String-valued, benchmark-private provenance tag.  cuVS-bench emits it in the label.
        "bitmap_method": method,
    }
    # The benchmark-private NaviX adapter obtains its bitmap-seed stride from the per-search
    # parameter object rather than search_basic_param.  Preserve historical config output at the
    # default width, but make non-default result widths explicit for every method.
    if k != DEFAULT_K:
        row["k"] = k
    if method == "navix_reference":
        row.update(
            {
                "navix_mode": "adaptive_kuzu",
                "navix_scheduler": "tiled",
                "navix_bitmap_seeds": True,
                "navix_kernel_variant": "reference",
            }
        )
    elif method.endswith("_seeded"):
        row["cagra_bitmap_seeds"] = True
    return row


def point_identity(row: dict) -> dict:
    return {
        "method": row["bitmap_method"],
        "itopk": int(row["itopk"]),
        "search_width": int(row["search_width"]),
        "max_iterations": int(row["max_iterations"]),
    }


def config_payload(
    *,
    workload: str,
    phase: str,
    shard: dict,
    paths: DatasetPaths,
    searches: list[dict],
    k: int = DEFAULT_K,
) -> dict:
    shard_directory = Path(shard["directory"])
    query_count = int(shard["query_count"])
    return {
        "dataset": {
            "name": (
                f"retrieve-{workload}-{phase}-q{int(shard['first_query']):05d}-"
                f"{query_count:05d}"
            ),
            "base_file": paths.base_file,
            "query_file": str(shard_directory / "query.bin"),
            "groundtruth_neighbors_file": str(
                shard_directory / "groundtruth.ibin"
            ),
            "distance": "euclidean",
            "dtype": paths.dtype,
            "filter": {
                "kind": "bitmap",
                "file": str(shard_directory / "filter.bitmap"),
            },
        },
        "search_basic_param": {"batch_size": query_count, "k": k},
        "index": [
            {
                "name": (
                    f"cagra-g{paths.graph_degree}-"
                    f"ig{paths.intermediate_graph_degree}"
                ),
                "algo": "cuvs_cagra",
                "file": paths.index_file,
                "build_param": {
                    "graph_build_algo": "NN_DESCENT",
                    "graph_degree": paths.graph_degree,
                    "intermediate_graph_degree": (
                        paths.intermediate_graph_degree
                    ),
                },
                "search_params": searches,
            }
        ],
    }


def write_group(
    output: Path,
    data_root: Path,
    *,
    group: str,
    workload: str,
    phase: str,
    searches: list[dict],
    yfcc_graph_degree: int,
    k: int = DEFAULT_K,
) -> None:
    max_query_values = {int(row["max_queries"]) for row in searches}
    if len(max_query_values) != 1:
        raise ValueError(
            f"{group}/{workload} mixes max_queries values: {sorted(max_query_values)}"
        )
    max_queries = next(iter(max_query_values))
    paths = dataset_paths(
        data_root, workload, phase, yfcc_graph_degree
    )
    if not paths.manifest.is_file():
        raise FileNotFoundError(paths.manifest)
    source_manifest = json.loads(paths.manifest.read_text())
    shards = source_manifest.get("shards", [])
    if not shards:
        raise ValueError(f"no shards in {paths.manifest}")
    cursor = 0
    for shard_number, shard in enumerate(shards):
        first_query = int(shard["first_query"])
        query_count = int(shard["query_count"])
        if first_query != cursor or query_count <= 0:
            raise ValueError(
                f"non-contiguous/invalid shard {shard_number} in {paths.manifest}: "
                f"first={first_query}, expected={cursor}, count={query_count}"
            )
        cursor += query_count

    group_dir = output / group / workload
    group_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict] = []
    for shard_number, shard in enumerate(shards):
        config_path = group_dir / f"shard_{shard_number:02d}.json"
        config_path.write_text(
            json.dumps(
                config_payload(
                    workload=workload,
                    phase=phase,
                    shard=shard,
                    paths=paths,
                    searches=searches,
                    k=k,
                ),
                indent=2,
            )
            + "\n"
        )
        generated.append(
            {
                "config": str(config_path.resolve()),
                "shard_index": shard_number,
                "first_query": int(shard["first_query"]),
                "query_count": int(shard["query_count"]),
            }
        )

    expected_queries = 1_000 if phase == "correctness" else 10_000
    if cursor != expected_queries:
        raise ValueError(
            f"source shards in {paths.manifest} cover {cursor} queries, "
            f"expected {expected_queries}"
        )
    if sum(row["query_count"] for row in generated) != expected_queries:
        raise ValueError(
            f"{group}/{workload} covers "
            f"{sum(row['query_count'] for row in generated)} queries, expected {expected_queries}"
        )
    manifest = {
        "schema_version": 1,
        "experiment": "retrieve_workshop_gpu_graph",
        "group": group,
        "phase": phase,
        "workload": workload,
        "k": k,
        "max_queries": max_queries,
        "graph_degree": paths.graph_degree,
        "intermediate_graph_degree": paths.intermediate_graph_degree,
        "dataset_size": paths.dataset_size,
        "dimension": paths.dimension,
        "dataset_profile": profile_record(),
        "repetitions": (
            CORRECTNESS_REPETITIONS
            if phase == "correctness"
            else THROUGHPUT_REPETITIONS
        ),
        "expected_queries": expected_queries,
        "expected_shards": len(generated),
        "source_bitmap_manifest": str(paths.manifest.resolve()),
        "search_points": [point_identity(row) for row in searches],
        "configs": generated,
    }
    (group_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def parse_deep_pairs(args: argparse.Namespace) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    raw_pairs = list(args.deep_pair)
    if args.deep_selection:
        selection = json.loads(args.deep_selection.read_text())
        raw_pairs.extend(
            f"{item['workload']}:{item['method']}"
            for item in selection.get("pairs", [])
        )
    for value in raw_pairs:
        try:
            workload, method = value.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"deep pair must be WORKLOAD:METHOD, got {value!r}"
            ) from exc
        if workload not in WORKLOADS or method not in METHODS:
            raise ValueError(f"invalid deep pair {value!r}")
        pairs.add((workload, method))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="result width; SINGLE_CTA requires k <= L <= 512",
    )
    parser.add_argument(
        "--primary-methods-only",
        action="store_true",
        help="generate Base, Retain, and NaviX without matched-seed controls",
    )
    parser.add_argument(
        "--cartesian-b0",
        action="store_true",
        help="use every L/W pair from L={64,128,256,512}, W={1,2}",
    )
    parser.add_argument(
        "--yfcc-graph-degree",
        type=int,
        choices=(32, 64),
        default=64,
        help=(
            "YFCC CAGRA degree: 64 is the primary paper configuration; "
            "32 is retained only for graph-degree sensitivity"
        ),
    )
    parser.add_argument(
        "--deep-pair",
        action="append",
        default=[],
        metavar="WORKLOAD:METHOD",
        help="generate optional deep points for this pair; repeat as needed",
    )
    parser.add_argument(
        "--deep-selection",
        type=Path,
        help="consume the analyzer's deep_candidates.json (it may be edited first)",
    )
    args = parser.parse_args()

    if not 1 <= args.k <= 512:
        parser.error("--k must be in [1,512] for the SINGLE_CTA study")
    candidate_b0_cells = (
        tuple((itopk, width) for itopk in (64, 128, 256, 512) for width in (1, 2))
        if args.cartesian_b0
        else B0_CELLS
    )
    b0_cells = tuple(cell for cell in candidate_b0_cells if cell[0] >= args.k)
    deep_cells = tuple(cell for cell in DEEP_CELLS if cell[0] >= args.k)
    if not b0_cells:
        parser.error("no configured B0 cell has L >= k")
    methods = PRIMARY_METHODS if args.primary_methods_only else METHODS

    b0_searches = [
        search_point(method, itopk, width, 0, k=args.k)
        for method in methods
        for itopk, width in b0_cells
    ]
    correctness_itopk = min(itopk for itopk, _ in b0_cells)
    correctness_searches = [
        search_point(method, correctness_itopk, 1, 0, k=args.k)
        for method in methods
    ]
    for workload in WORKLOADS:
        write_group(
            args.output,
            args.data_root,
            group="correctness",
            workload=workload,
            phase="correctness",
            searches=correctness_searches,
            yfcc_graph_degree=args.yfcc_graph_degree,
            k=args.k,
        )
        write_group(
            args.output,
            args.data_root,
            group="b0",
            workload=workload,
            phase="throughput",
            searches=b0_searches,
            yfcc_graph_degree=args.yfcc_graph_degree,
            k=args.k,
        )

    deep_pairs = parse_deep_pairs(args)
    (args.output / "deep_plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_recall": 0.90,
                "cells": [list(cell) for cell in deep_cells],
                "iterations": list(DEEP_ITERATIONS),
                "pairs": [
                    {"workload": workload, "method": method}
                    for workload, method in sorted(deep_pairs)
                ],
            },
            indent=2,
        )
        + "\n"
    )
    for iterations in DEEP_ITERATIONS:
        for workload in WORKLOADS:
            selected_methods = [
                method
                for method in methods
                if (workload, method) in deep_pairs
            ]
            if not selected_methods:
                continue
            # One method per deep group lets the staged runner stop that workload/method series
            # immediately after either retained cell reaches the target recall.
            for method in selected_methods:
                write_group(
                    args.output,
                    args.data_root,
                    group=f"deep_i{iterations}_{workload}_{method}",
                    workload=workload,
                    phase="throughput",
                    searches=[
                        search_point(method, itopk, width, iterations, k=args.k)
                        for itopk, width in deep_cells
                    ],
                    yfcc_graph_degree=args.yfcc_graph_degree,
                    k=args.k,
                )


if __name__ == "__main__":
    main()
