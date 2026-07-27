#!/usr/bin/env python3
"""Generate matched default/FAVOR cuvs-bench configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SELECTIVITIES = (1, 10, 50, 90)
DEFAULT_ITOPK_VALUES = (32, 64, 128, 256, 512)
DEFAULT_SEARCH_WIDTHS = (1, 2, 4)


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
    parser.add_argument(
        "--selectivities", type=int, nargs="+", default=DEFAULT_SELECTIVITIES
    )
    parser.add_argument(
        "--itopk-values", type=int, nargs="+", default=DEFAULT_ITOPK_VALUES
    )
    parser.add_argument(
        "--search-widths", type=int, nargs="+", default=DEFAULT_SEARCH_WIDTHS
    )
    args = parser.parse_args()
    selectivities = tuple(dict.fromkeys(args.selectivities))
    if not selectivities or any(not 1 <= value <= 100 for value in selectivities):
        raise ValueError("selectivities must be whole percentages in [1, 100]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delta_d_file = args.delta_d_file or f"{args.dataset_name}/{args.graph_file}.delta_d"

    for selectivity in selectivities:
        for batch_size in (10, 10000):
            search_params = []
            for mode in ("default", "favor"):
                for itopk in args.itopk_values:
                    for width in args.search_widths:
                        param = {
                            "algo": "single_cta",
                            "filter_mode": mode,
                            "itopk": itopk,
                            "search_width": width,
                        }
                        if mode == "favor":
                            if args.favor_delta_d is not None:
                                param["favor_delta_d"] = args.favor_delta_d
                            else:
                                param.update({
                                    "favor_delta_d_file": delta_d_file,
                                    "favor_delta_d_alpha": 10,
                                    "favor_delta_d_beta": args.delta_d_beta,
                                    "favor_delta_d_bfs_depth": args.delta_d_bfs_depth,
                                })
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
