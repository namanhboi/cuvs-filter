#!/usr/bin/env python3
"""Generate fixed-cell configs for the independent multi-seed B0 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEED_MASKS = (
    0x0000000000128394,
    0x9E3779B97F58FF81,
    0xD1B54A32D1806E97,
)

DATASETS = {
    "gist": {
        "directory": "gist-960-euclidean",
        "name": "GIST-1M",
        "query": "query_10000.fbin",
        "search_width": 4,
    },
    "msturing1m": {
        "directory": "msturing-1M",
        "name": "MSTuring-1M",
        "query": "query.fbin",
        "search_width": 1,
    },
    "msturing10m": {
        "directory": "msturing-10M",
        "name": "MSTuring-10M",
        "query": "query.fbin",
        "search_width": 1,
    },
}


def favor_param(
    dataset: dict[str, object], variant: str, delta_root: Path
) -> dict[str, object]:
    directory = str(dataset["directory"])
    return {
        "experiment_variant": variant,
        "algo": "single_cta",
        "filter_mode": "favor",
        "itopk": 512,
        "search_width": dataset["search_width"],
        "max_iterations": 0,
        "max_queries": 2048,
        "rand_xor_mask": SEED_MASKS[0],
        "favor_penalty_mode": "cagra_retention_safe",
        "favor_penalty_lambda": 1.0,
        "favor_retention_fraction": 0.0,
        "favor_delta_d_file": str(delta_root / directory / "cagra_g32_ig64.index.delta_d"),
        "favor_delta_d_alpha": 10,
        "favor_delta_d_beta": 64,
        "favor_delta_d_bfs_depth": 2,
    }


def default_param(dataset: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_variant": "default_cagra",
        "algo": "single_cta",
        "filter_mode": "default",
        "itopk": 512,
        "search_width": dataset["search_width"],
        "max_iterations": 0,
        "max_queries": 2048,
    }


def config(
    dataset: dict[str, object], batch_size: int, search_params: list[dict[str, object]]
) -> dict[str, object]:
    directory = str(dataset["directory"])
    return {
        "dataset": {
            "name": f"{dataset['name']}-s01",
            "base_file": f"{directory}/base.fbin",
            "query_file": f"{directory}/{dataset['query']}",
            "groundtruth_neighbors_file": f"{directory}/favor/groundtruth_s01.ibin",
            "filter_bitset_file": f"{directory}/favor/filter_s01.bin",
            "distance": "euclidean",
            "dtype": "float",
        },
        "search_basic_param": {"batch_size": batch_size, "k": 10},
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
                "search_params": search_params,
            }
        ],
    }


def build_config(dataset: dict[str, object]) -> dict[str, object]:
    directory = str(dataset["directory"])
    return {
        "dataset": {
            "name": dataset["name"],
            "base_file": f"{directory}/base.fbin",
            "query_file": f"{directory}/{dataset['query']}",
            "distance": "euclidean",
            "dtype": "float",
        },
        "search_basic_param": {"batch_size": 10, "k": 10},
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
                "search_params": [
                    {
                        "algo": "single_cta",
                        "itopk": 64,
                        "search_width": 1,
                    }
                ],
            }
        ],
    }


def write_configs(output_dir: Path, delta_root: Path = Path("datasets")) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug, dataset in DATASETS.items():
        path = output_dir / f"{slug}_build.json"
        path.write_text(json.dumps(build_config(dataset), indent=2) + "\n")
        written.append(path)
        for batch_size in (10, 10000):
            controls = [
                default_param(dataset),
                favor_param(dataset, "automatic_retention", delta_root),
            ]
            adaptive = [favor_param(dataset, "adaptive_termination", delta_root)]
            multi_seed = []
            for rounds in (1, 2, 3):
                param = favor_param(dataset, f"multi_seed_{rounds}", delta_root)
                param["favor_seed_masks"] = list(SEED_MASKS[:rounds])
                multi_seed.append(param)
            groups = {
                "controls": controls,
                "adaptive": adaptive,
                "multiseed": multi_seed,
            }
            for group, params in groups.items():
                path = output_dir / f"{slug}_nq{batch_size}_{group}.json"
                path.write_text(json.dumps(config(dataset, batch_size, params), indent=2) + "\n")
                written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "configs",
    )
    parser.add_argument(
        "--delta-root",
        type=Path,
        default=Path("datasets"),
        help="dataset root used to resolve FAVOR delta-d sidecars",
    )
    args = parser.parse_args()
    written = write_configs(args.output_dir, args.delta_root)
    print(f"wrote {len(written)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
