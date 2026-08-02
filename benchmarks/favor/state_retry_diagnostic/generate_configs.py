#!/usr/bin/env python3
"""Generate fixed configs for the FAVOR saved-state retry diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STRATEGIES = ("independent", "passing", "frontier", "combined", "oracle")
DATASETS = {
    "gist": {
        "directory": "gist-960-euclidean",
        "name": "GIST-1M",
        "query": "query_10000.fbin",
        "search_width": 4,
        "b0": 133,
        "rounds": 2,
    },
    "msturing1m": {
        "directory": "msturing-1M",
        "name": "MSTuring-1M",
        "query": "query.fbin",
        "search_width": 1,
        "b0": 517,
        "rounds": 4,
    },
    "msturing10m": {
        "directory": "msturing-10M",
        "name": "MSTuring-10M",
        "query": "query.fbin",
        "search_width": 1,
        "b0": 518,
        "rounds": 7,
    },
}


def config(
    slug: str,
    dataset: dict[str, object],
    strategy: str,
    data_root: Path,
    capture_root: Path,
) -> dict[str, object]:
    directory = str(dataset["directory"])
    diagnostic = capture_root / slug / strategy
    search_param = {
        "experiment_variant": f"state_retry_{strategy}",
        "algo": "single_cta",
        "filter_mode": "favor",
        "itopk": 512,
        "search_width": dataset["search_width"],
        "max_iterations": 0,
        "max_queries": 2048,
        "rand_xor_mask": 0x128394,
        "favor_penalty_mode": "cagra_retention_safe",
        "favor_penalty_lambda": 1.0,
        "favor_retention_fraction": 0.0,
        "favor_delta_d_file": str(data_root / directory / "cagra_g32_ig64.index.delta_d"),
        "favor_delta_d_alpha": 10,
        "favor_delta_d_beta": 64,
        "favor_delta_d_bfs_depth": 2,
        "favor_retry_diagnostics_output": str(diagnostic),
        "favor_retry_diagnostics_groundtruth": str(
            data_root / directory / "favor/groundtruth_s01.ibin"
        ),
        "favor_retry_diagnostics_dataset": slug,
        "favor_retry_strategy": strategy,
        "favor_retry_rounds": dataset["rounds"],
        "favor_retry_b0": dataset["b0"],
    }
    return {
        "dataset": {
            "name": f"{dataset['name']}-s01-state-retry",
            "base_file": f"{directory}/base.fbin",
            "query_file": f"{directory}/{dataset['query']}",
            "groundtruth_neighbors_file": f"{directory}/favor/groundtruth_s01.ibin",
            "filter_bitset_file": f"{directory}/favor/filter_s01.bin",
            "distance": "euclidean",
            "dtype": "float",
        },
        "search_basic_param": {"batch_size": 10000, "k": 10},
        "index": [
            {
                "name": "cagra-g32-ig64",
                "algo": "cuvs_cagra",
                "file": f"{directory}/cagra_g32_ig64.index",
                "build_param": {
                    "graph_build_algo": "NN_DESCENT",
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                },
                "search_params": [search_param],
            }
        ],
    }


def write_configs(
    output_dir: Path, data_root: Path, capture_root: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for slug, dataset in DATASETS.items():
        for strategy in STRATEGIES:
            path = output_dir / f"{slug}_{strategy}.json"
            path.write_text(
                json.dumps(config(slug, dataset, strategy, data_root, capture_root), indent=2)
                + "\n"
            )
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    written = write_configs(
        args.output_dir.resolve(), args.data_root.resolve(), args.capture_root.resolve()
    )
    print(f"wrote {len(written)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
