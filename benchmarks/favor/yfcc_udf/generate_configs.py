#!/usr/bin/env python3
"""Generate reproducible cuVS-bench configs for the YFCC SINGLE_CTA UDF experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SWEEP_CELLS = [(l, w) for l in (64, 128, 256, 512) for w in (1, 2, 4)]
DEEP_ITERATIONS = (522, 1044, 2088, 4176, 7569)
MAX_ITERATIONS = (0, *DEEP_ITERATIONS)
MAX_QUERIES = 512
GRAPH_DEGREE = 32
MAX_HASH_SLOTS = 1 << 20
HASHMAP_MAX_FILL_RATE = 0.5
SAMPLING_ESTIMATE_CELLS = (
    (64, 1, 0),
    (64, 2, 0),
    (128, 4, 0),
    (256, 2, 0),
    (512, 2, 0),
    (512, 2, 522),
)
DATA_ROOT = Path(".")


def dataset(workload: str) -> dict:
    root = f"yfcc-10M/workloads/{workload}"
    return {
        "name": f"yfcc-10M-{workload}",
        "base_file": "yfcc-10M/base.10M.u8bin",
        "query_file": f"{root}/query.u8bin",
        "groundtruth_neighbors_file": f"{root}/groundtruth.ibin",
        "distance": "euclidean",
        "dtype": "uint8",
        "filter": {
            "kind": "udf",
            "adapter": "spmat_contains_all",
            "base_metadata_file": "yfcc-10M/base.metadata.10M.spmat",
            "query_metadata_file": f"{root}/query.metadata.spmat",
        },
    }


def default_search(
    itopk: int,
    width: int,
    *,
    accumulator: bool = False,
    max_iterations: int = 0,
) -> dict:
    out = {
        "algo": "single_cta",
        "filter_mode": "default",
        "max_queries": MAX_QUERIES,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": max_iterations,
        "favor_udf_passing_accumulator": accumulator,
    }
    return out


def favor_search(
    itopk: int,
    width: int,
    *,
    accumulator: bool = True,
    include_sampling: bool = False,
    sample_offset: int = 0,
    max_iterations: int = 0,
    diagnostic_output: Path | None = None,
    diagnostic_gt: Path | None = None,
    diagnostic_variant: str | None = None,
) -> dict:
    out = {
        "algo": "single_cta",
        "filter_mode": "favor",
        "max_queries": MAX_QUERIES,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": max_iterations,
        "favor_delta_d_file": str(
            (DATA_ROOT / "yfcc-10M/cagra_g32_ig64.index.delta_d").resolve()
        ),
        "favor_delta_d_alpha": 10,
        "favor_delta_d_beta": 64,
        "favor_delta_d_bfs_depth": 2,
        "favor_penalty_mode": "cagra_retention_safe",
        "favor_penalty_lambda": 1.0,
        "favor_retention_fraction": 0.0,
        "favor_udf_include_sampling": include_sampling,
        "favor_udf_passing_accumulator": accumulator,
        "favor_udf_sample_offset": sample_offset,
    }
    if diagnostic_output is not None:
        out.update(
            {
                "favor_diagnostics_output": str(diagnostic_output),
                "favor_diagnostics_groundtruth": str(diagnostic_gt),
                "favor_diagnostics_dataset": "yfcc10m",
                "favor_diagnostics_variant": diagnostic_variant
                or f"auto_accum_L{itopk}_W{width}_i{max_iterations}",
            }
        )
    return out


def config(workload: str, batch_size: int, searches: list[dict]) -> dict:
    return {
        "dataset": dataset(workload),
        "search_basic_param": {"batch_size": batch_size, "k": 10},
        "index": [
            {
                "name": "cagra-g32-ig64",
                "algo": "cuvs_cagra",
                "file": "yfcc-10M/cagra_g32_ig64.index",
                "build_param": {
                    "graph_build_algo": "NN_DESCENT",
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                },
                "search_params": searches,
            }
        ],
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def supported_search(itopk: int, width: int, max_iterations: int) -> bool:
    if max_iterations == 0:
        return True
    max_visited_nodes = itopk + width * GRAPH_DEGREE * max_iterations
    return max_visited_nodes <= MAX_HASH_SLOTS * HASHMAP_MAX_FILL_RATE


def main() -> None:
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    DATA_ROOT = args.data_root

    # One-query launch/debug gate.  This must pass before any long YFCC sweep starts.
    write(
        args.output / "smoke.json",
        config(
            "latency_a1_d10",
            1,
            [
                default_search(64, 1),
                default_search(64, 1, accumulator=True),
                favor_search(64, 1, include_sampling=True),
            ],
        ),
    )
    write(
        args.output / "debug_favor.json",
        config(
            "correctness_1000",
            1000,
            [favor_search(64, 1, accumulator=False), favor_search(64, 1)],
        ),
    )

    correctness = []
    for l, w in SWEEP_CELLS:
        for max_iterations in MAX_ITERATIONS:
            if not supported_search(l, w, max_iterations):
                continue
            correctness.append(default_search(l, w, accumulator=False, max_iterations=max_iterations))
            correctness.append(default_search(l, w, accumulator=True, max_iterations=max_iterations))
            correctness.append(
                favor_search(
                    l,
                    w,
                    accumulator=True,
                    max_iterations=max_iterations,
                )
            )
    write(args.output / "correctness.json", config("correctness_1000", 1000, correctness))

    throughput = []
    for l, w in SWEEP_CELLS:
        for max_iterations in MAX_ITERATIONS:
            if not supported_search(l, w, max_iterations):
                continue
            throughput.append(default_search(l, w, accumulator=False, max_iterations=max_iterations))
            throughput.append(default_search(l, w, accumulator=True, max_iterations=max_iterations))
            throughput.append(
                favor_search(
                    l,
                    w,
                    accumulator=True,
                    include_sampling=True,
                    max_iterations=max_iterations,
                )
            )
    write(args.output / "throughput.json", config("throughput_10000", 10000, throughput))

    sampling = []
    for l, w, i in SAMPLING_ESTIMATE_CELLS:
        sampling.extend(
            [
                favor_search(
                    l,
                    w,
                    accumulator=False,
                    include_sampling=False,
                    max_iterations=i,
                ),
                favor_search(
                    l,
                    w,
                    accumulator=False,
                    include_sampling=True,
                    max_iterations=i,
                ),
                favor_search(
                    l,
                    w,
                    accumulator=True,
                    include_sampling=False,
                    max_iterations=i,
                ),
                favor_search(
                    l,
                    w,
                    accumulator=True,
                    include_sampling=True,
                    max_iterations=i,
                ),
            ]
        )
    write(
        args.output / "throughput_sampling.json",
        config("throughput_10000", 10000, sampling),
    )

    for arity in (1, 2):
        for decile in range(1, 11):
            workload = f"latency_a{arity}_d{decile}"
            searches = [
                default_search(512, 4),
                favor_search(512, 2),
                favor_search(512, 2, max_iterations=7569),
            ]
            write(args.output / f"{workload}.json", config(workload, 10, searches))

            group_diagnostics = []
            for iterations, suffix in ((0, "b0"), (7569, "deep")):
                variant = f"a{arity}_d{decile}_{suffix}"
                group_diagnostics.append(
                    favor_search(
                        512,
                        2,
                        max_iterations=iterations,
                        diagnostic_output=(
                            args.result_root / "diagnostics" / "groups" / variant
                        ).resolve(),
                        diagnostic_gt=(
                            args.data_root / f"yfcc-10M/workloads/{workload}/groundtruth.ibin"
                        ).resolve(),
                        diagnostic_variant=variant,
                    )
                )
            write(
                args.output / f"diagnostic_{workload}.json",
                config(workload, 10, group_diagnostics),
            )

    diagnostic_gt = (
        args.data_root / "yfcc-10M/workloads/correctness_1000/groundtruth.ibin"
    ).resolve()
    diagnostic_searches = []
    for accumulator, iterations, variant in [
        (False, 0, "legacy_b0"),
        (True, 0, "accumulator_b0"),
        *((True, value, f"accumulator_i{value}") for value in DEEP_ITERATIONS),
        (False, 7569, "legacy_i7569"),
    ]:
        diagnostic_searches.append(
            favor_search(
                512,
                2,
                accumulator=accumulator,
                max_iterations=iterations,
                diagnostic_output=(args.result_root / "diagnostics" / variant).resolve(),
                diagnostic_gt=diagnostic_gt,
                diagnostic_variant=variant,
            )
        )
    write(
        args.output / "diagnostic.json",
        config("correctness_1000", 1000, diagnostic_searches),
    )


if __name__ == "__main__":
    main()
