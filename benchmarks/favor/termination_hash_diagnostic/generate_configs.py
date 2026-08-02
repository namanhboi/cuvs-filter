#!/usr/bin/env python3
"""Generate deep exact/forgetful-hash trajectory configs for FAVOR termination analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HASHES = ("exact", "forgetful")
DATASETS = {
    "gist": {
        "directory": "gist-960-euclidean",
        "name": "GIST-1M",
        "query": "query_10000.fbin",
        "search_width": 4,
        "b0": 133,
        "cap": 266,
    },
    "msturing1m": {
        "directory": "msturing-1M",
        "name": "MSTuring-1M",
        "query": "query.fbin",
        "search_width": 1,
        "b0": 517,
        "cap": 2068,
    },
    "msturing10m": {
        "directory": "msturing-10M",
        "name": "MSTuring-10M",
        "query": "query.fbin",
        "search_width": 1,
        "b0": 518,
        "cap": 3626,
    },
}


def make_config(
    slug: str,
    dataset: dict[str, object],
    hash_variant: str,
    data_root: Path,
    capture_root: Path,
) -> dict[str, object]:
    directory = str(dataset["directory"])
    diagnostic = capture_root / slug / hash_variant
    search_param = {
        "experiment_variant": f"termination_shadow_{hash_variant}",
        "algo": "single_cta",
        "filter_mode": "favor",
        "itopk": 512,
        "search_width": dataset["search_width"],
        "max_iterations": dataset["cap"],
        "max_queries": 2048,
        "rand_xor_mask": 0x128394,
        "favor_penalty_mode": "cagra_retention_safe",
        "favor_penalty_lambda": 1.0,
        "favor_retention_fraction": 0.0,
        "favor_delta_d_file": str(data_root / directory / "cagra_g32_ig64.index.delta_d"),
        "favor_delta_d_alpha": 10,
        "favor_delta_d_beta": 64,
        "favor_delta_d_bfs_depth": 2,
        "favor_diagnostics_output": str(diagnostic),
        "favor_diagnostics_groundtruth": str(
            data_root / directory / "favor/groundtruth_s01.ibin"
        ),
        "favor_diagnostics_dataset": slug,
        "favor_diagnostics_variant": hash_variant,
        "favor_termination_shadow_start_iteration": dataset["b0"],
        "favor_termination_shadow_parent_interval": 32,
    }
    return {
        "dataset": {
            "name": f"{dataset['name']}-s01-termination-shadow-{hash_variant}",
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


def make_current_config(
    slug: str,
    dataset: dict[str, object],
    data_root: Path,
) -> dict[str, object]:
    config = make_config(slug, dataset, "exact", data_root, Path("unused"))
    config["dataset"]["name"] = f"{dataset['name']}-s01-current-adaptive"
    search = config["index"][0]["search_params"][0]
    search["experiment_variant"] = "current_adaptive"
    search["max_iterations"] = 0
    for key in tuple(search):
        if key.startswith("favor_diagnostics_") or key.startswith("favor_termination_shadow_"):
            del search[key]
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for slug, dataset in DATASETS.items():
        for hash_variant in HASHES:
            path = args.output_dir / f"{slug}_{hash_variant}.json"
            path.write_text(
                json.dumps(
                    make_config(
                        slug,
                        dataset,
                        hash_variant,
                        args.data_root.resolve(),
                        args.capture_root.resolve(),
                    ),
                    indent=2,
                )
                + "\n"
            )
        path = args.output_dir / f"{slug}_current.json"
        path.write_text(
            json.dumps(make_current_config(slug, dataset, args.data_root.resolve()), indent=2)
            + "\n"
        )
    print(f"wrote {len(DATASETS) * (len(HASHES) + 1)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
