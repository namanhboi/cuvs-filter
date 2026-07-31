#!/usr/bin/env python3
"""Generate matched default/FAVOR cuvs-bench configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SELECTIVITIES = (1, 10, 50, 90)
DEFAULT_ITOPK_VALUES = (32, 64, 128, 256, 512)
DEFAULT_SEARCH_WIDTHS = (1, 2, 4)
DEFAULT_MAX_ITERATIONS = (0,)
DEFAULT_THREAD_BLOCK_SIZES = (0,)
DEFAULT_BATCH_SIZES = (10, 10000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="sift-128-euclidean")
    parser.add_argument("--result-prefix", default="sift")
    parser.add_argument("--base-file", default="base.fbin")
    parser.add_argument("--query-file", default="query.fbin")
    parser.add_argument("--dtype", default="float")
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--graph-file", default="cagra_g32_ig64.index")
    parser.add_argument("--delta-d-file")
    parser.add_argument("--favor-delta-d", type=float)
    parser.add_argument("--delta-d-beta", type=int, default=64)
    parser.add_argument("--delta-d-bfs-depth", type=int, default=2)
    parser.add_argument("--penalty-lambdas", type=float, nargs="+", default=(1.0,))
    parser.add_argument(
        "--retention-fractions", type=float, nargs="+", default=(0.5,)
    )
    parser.add_argument(
        "--selectivities", type=int, nargs="+", default=DEFAULT_SELECTIVITIES
    )
    parser.add_argument(
        "--itopk-values", type=int, nargs="+", default=DEFAULT_ITOPK_VALUES
    )
    parser.add_argument(
        "--search-widths", type=int, nargs="+", default=DEFAULT_SEARCH_WIDTHS
    )
    parser.add_argument(
        "--max-iterations", type=int, nargs="+", default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument(
        "--thread-block-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_THREAD_BLOCK_SIZES,
    )
    parser.add_argument(
        "--search-cells",
        nargs="+",
        help=(
            "exact ITOPK:WIDTH:MAX_ITERATIONS[:THREAD_BLOCK_SIZE] cells; "
            "when present, replaces the Cartesian search grid"
        ),
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument(
        "--algo", choices=("single_cta", "multi_cta"), default="single_cta"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "default",
            "favor",
            "favor_query_local",
            "favor_retention_safe",
        ),
        default=("default", "favor", "favor_query_local", "favor_retention_safe"),
        help="filtering methods to include",
    )
    parser.add_argument(
        "--deduplicate-multi-cta",
        action="store_true",
        help="omit search widths that produce an already-covered CTA count for the same itopk",
    )
    args = parser.parse_args()
    selectivities = tuple(dict.fromkeys(args.selectivities))
    if not selectivities or any(not 1 <= value <= 100 for value in selectivities):
        raise ValueError("selectivities must be whole percentages in [1, 100]")
    batch_sizes = tuple(dict.fromkeys(args.batch_sizes))
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delta_d_file = args.delta_d_file or f"{args.dataset_name}/{args.graph_file}.delta_d"
    if any(value <= 0 for value in args.penalty_lambdas):
        raise ValueError("penalty lambdas must be positive")
    if any(not 0 <= value < 1 for value in args.retention_fractions):
        raise ValueError("retention fractions must be in [0, 1); zero selects automatic retention")
    if any(value < 0 for value in args.max_iterations):
        raise ValueError("max iterations must be nonnegative")
    if any(value < 0 for value in args.thread_block_sizes):
        raise ValueError("thread block sizes must be nonnegative")
    if args.search_cells:
        tuning_cells = []
        for encoded in args.search_cells:
            fields = tuple(int(value) for value in encoded.split(":"))
            if len(fields) not in (3, 4):
                raise ValueError(
                    "search cells must use ITOPK:WIDTH:MAX_ITERATIONS[:THREAD_BLOCK_SIZE]"
                )
            cell = fields if len(fields) == 4 else (*fields, 0)
            if cell[0] <= 0 or cell[1] <= 0 or cell[2] < 0 or cell[3] < 0:
                raise ValueError("search-cell values must be positive except optional zeros")
            tuning_cells.append(cell)
        tuning_cells = list(dict.fromkeys(tuning_cells))
    else:
        tuning_cells = [
            (itopk, width, max_iterations, thread_block_size)
            for itopk in args.itopk_values
            for width in args.search_widths
            for max_iterations in args.max_iterations
            for thread_block_size in args.thread_block_sizes
        ]

    for selectivity in selectivities:
        for batch_size in batch_sizes:
            search_params = []
            for mode in tuple(dict.fromkeys(args.modes)):
                covered_cta_counts: dict[int, set[int]] = {}
                lambdas = (
                    args.penalty_lambdas
                    if mode
                    in (
                        "favor_query_local",
                        "favor_retention_safe",
                    )
                    else (1.0,)
                )
                retention_fractions = (
                    args.retention_fractions
                    if mode == "favor_retention_safe"
                    else (0.5,)
                )
                for itopk, width, max_iterations, thread_block_size in tuning_cells:
                    covered_cta_counts.setdefault(itopk, set())
                    effective_cta_count = max(width, (itopk + 31) // 32)
                    if (
                        args.algo == "multi_cta"
                        and args.deduplicate_multi_cta
                        and effective_cta_count in covered_cta_counts[itopk]
                    ):
                        continue
                    covered_cta_counts[itopk].add(effective_cta_count)
                    for penalty_lambda in lambdas:
                        for retention_fraction in retention_fractions:
                            param = {
                                "algo": args.algo,
                                "filter_mode": (
                                    "default" if mode == "default" else "favor"
                                ),
                                "itopk": itopk,
                                "search_width": width,
                            }
                            if max_iterations > 0:
                                param["max_iterations"] = max_iterations
                            if thread_block_size > 0:
                                param["thread_block_size"] = thread_block_size
                            if mode != "default":
                                if args.favor_delta_d is not None:
                                    param["favor_delta_d"] = args.favor_delta_d
                                else:
                                    param.update(
                                        {
                                            "favor_delta_d_file": delta_d_file,
                                            "favor_delta_d_alpha": 10,
                                            "favor_delta_d_beta": args.delta_d_beta,
                                            "favor_delta_d_bfs_depth": args.delta_d_bfs_depth,
                                        }
                                    )
                            if mode == "favor_query_local":
                                param["favor_penalty_mode"] = "cagra_query_local"
                                param["favor_penalty_lambda"] = penalty_lambda
                            elif mode == "favor_retention_safe":
                                param["favor_penalty_mode"] = "cagra_retention_safe"
                                param["favor_penalty_lambda"] = penalty_lambda
                                param["favor_retention_fraction"] = retention_fraction
                            search_params.append(param)

            dataset = {
                "name": f"{args.dataset_name}-s{selectivity:02d}",
                "base_file": f"{args.dataset_name}/{args.base_file}",
                "query_file": f"{args.dataset_name}/{args.query_file}",
                "groundtruth_neighbors_file": (
                    f"{args.dataset_name}/favor/groundtruth_s{selectivity:02d}.ibin"
                ),
                "filter_bitset_file": (
                    f"{args.dataset_name}/favor/filter_s{selectivity:02d}.bin"
                ),
                "distance": "euclidean",
                "dtype": args.dtype,
            }
            if args.subset_size is not None:
                dataset["subset_size"] = args.subset_size
            config = {
                "dataset": dataset,
                "search_basic_param": {"batch_size": batch_size, "k": 10},
                "index": [
                    {
                        "name": "cagra-g32-ig64",
                        "algo": "cuvs_cagra",
                        "file": f"{args.dataset_name}/{args.graph_file}",
                        "build_param": {
                            "graph_build_algo": "NN_DESCENT",
                            "graph_degree": 32,
                            "intermediate_graph_degree": 64,
                        },
                        "search_params": search_params,
                    }
                ],
            }
            path = args.output_dir / (
                f"{args.result_prefix}_s{selectivity:02d}_nq{batch_size}.json"
            )
            path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
