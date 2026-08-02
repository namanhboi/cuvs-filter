#!/usr/bin/env python3
"""Generate exact-state FAVOR progress-shadow configs.

The six development cells all use the deterministic 1% filters already used by the
automatic-retention report.  SIFT/GIST are separate families; the two BIGANN and two MSTuring
scales are grouped for leave-one-family-out validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS: dict[str, dict[str, object]] = {
    "sift": {
        "family": "sift",
        "directory": "sift-128-euclidean",
        "name": "SIFT-1M",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "search_width": 4,
        "b0": 133,
        "cap": 266,
    },
    "gist": {
        "family": "gist",
        "directory": "gist-960-euclidean",
        "name": "GIST-1M",
        "base": "base.fbin",
        "query": "query_10000.fbin",
        "dtype": "float",
        "search_width": 4,
        "b0": 133,
        "cap": 266,
    },
    "bigann1m": {
        "family": "bigann",
        "directory": "bigann-1M",
        "name": "BIGANN-1M",
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
        "subset_size": 1_000_000,
        "search_width": 1,
        "b0": 517,
        "cap": 1034,
    },
    "bigann10m": {
        "family": "bigann",
        "directory": "bigann-10M",
        "name": "BIGANN-10M",
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
        "search_width": 1,
        "b0": 518,
        "cap": 1036,
    },
    "msturing1m": {
        "family": "msturing",
        "directory": "msturing-1M",
        "name": "MSTuring-1M",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "search_width": 1,
        "b0": 517,
        "cap": 2068,
    },
    "msturing10m": {
        "family": "msturing",
        "directory": "msturing-10M",
        "name": "MSTuring-10M",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "search_width": 1,
        "b0": 518,
        "cap": 3626,
    },
}

SELECTIVITY = 0.01
CHECKPOINT_PARENTS = 32


def make_config(
    slug: str,
    spec: dict[str, object],
    data_root: Path,
    capture_root: Path,
) -> dict[str, object]:
    directory = str(spec["directory"])
    dataset: dict[str, object] = {
        "name": f"{spec['name']}-s01-progress-shadow",
        "base_file": f"{directory}/{spec['base']}",
        "query_file": f"{directory}/{spec['query']}",
        "groundtruth_neighbors_file": f"{directory}/favor/groundtruth_s01.ibin",
        "filter_bitset_file": f"{directory}/favor/filter_s01.bin",
        "distance": "euclidean",
        "dtype": spec["dtype"],
    }
    if "subset_size" in spec:
        dataset["subset_size"] = spec["subset_size"]

    search = {
        "experiment_variant": "progress_termination_shadow",
        "algo": "single_cta",
        "filter_mode": "favor",
        "itopk": 512,
        "search_width": spec["search_width"],
        "max_iterations": spec["cap"],
        "max_queries": 2048,
        "rand_xor_mask": 0x128394,
        "hashmap_max_fill_rate": 0.89,
        "filtering_rate": 1.0 - SELECTIVITY,
        "favor_penalty_mode": "cagra_retention_safe",
        "favor_penalty_lambda": 1.0,
        "favor_retention_fraction": 0.0,
        "favor_delta_d_file": str(data_root / directory / "cagra_g32_ig64.index.delta_d"),
        "favor_delta_d_alpha": 10,
        "favor_delta_d_beta": 64,
        "favor_delta_d_bfs_depth": 2,
        "favor_diagnostics_output": str(capture_root / slug),
        "favor_diagnostics_groundtruth": str(
            data_root / directory / "favor/groundtruth_s01.ibin"
        ),
        "favor_diagnostics_dataset": slug,
        "favor_diagnostics_variant": "exact_progress_v4",
        # Record evidence from the first iteration, but never let an offline rule stop before B0.
        "favor_termination_shadow_record_start_iteration": 1,
        "favor_termination_shadow_start_iteration": spec["b0"],
        "favor_termination_shadow_parent_interval": CHECKPOINT_PARENTS,
    }
    return {
        "dataset": dataset,
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
                "search_params": [search],
            }
        ],
    }


def make_build_config(spec: dict[str, object]) -> dict[str, object]:
    directory = str(spec["directory"])
    dataset: dict[str, object] = {
        "name": spec["name"],
        "base_file": f"{directory}/{spec['base']}",
        "query_file": f"{directory}/{spec['query']}",
        "distance": "euclidean",
        "dtype": spec["dtype"],
    }
    if "subset_size" in spec:
        dataset["subset_size"] = spec["subset_size"]
    return {
        "dataset": dataset,
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
                    {"algo": "single_cta", "itopk": 64, "search_width": 1}
                ],
            }
        ],
    }


def write_configs(
    output_dir: Path, data_root: Path, capture_root: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug, spec in DATASETS.items():
        for suffix, payload in (
            ("shadow", make_config(slug, spec, data_root, capture_root)),
            ("build", make_build_config(spec)),
        ):
            path = output_dir / f"{slug}_{suffix}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
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
