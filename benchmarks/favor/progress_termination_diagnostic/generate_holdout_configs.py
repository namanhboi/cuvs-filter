#!/usr/bin/env python3
"""Generate the frozen DEEP-image1M holdout build and shadow configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (20260802, 20260803)
DIRECTORY = "deep-image-1M"
B0 = 517
CAP = 2068


def dataset(seed: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "name": "DEEP-image1M" if seed is None else f"DEEP-image1M-s01-seed{seed}",
        "base_file": f"{DIRECTORY}/base.fbin",
        "query_file": f"{DIRECTORY}/query.fbin",
        "distance": "euclidean",
        "dtype": "float",
    }
    if seed is not None:
        result.update(
            {
                "groundtruth_neighbors_file": (
                    f"{DIRECTORY}/favor_seed{seed}/groundtruth_s01.ibin"
                ),
                "filter_bitset_file": f"{DIRECTORY}/favor_seed{seed}/filter_s01.bin",
            }
        )
    return result


def build_config() -> dict[str, object]:
    return {
        "dataset": dataset(),
        "search_basic_param": {"batch_size": 10, "k": 10},
        "index": [
            {
                "name": "cagra-g32-ig64",
                "algo": "cuvs_cagra",
                "file": f"{DIRECTORY}/cagra_g32_ig64.index",
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


def shadow_config(seed: int, data_root: Path, capture_root: Path) -> dict[str, object]:
    return {
        "dataset": dataset(seed),
        "search_basic_param": {"batch_size": 10000, "k": 10},
        "index": [
            {
                "name": "cagra-g32-ig64",
                "algo": "cuvs_cagra",
                "file": f"{DIRECTORY}/cagra_g32_ig64.index",
                "build_param": {
                    "graph_build_algo": "NN_DESCENT",
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                },
                "search_params": [
                    {
                        "experiment_variant": "frozen_progress_holdout",
                        "algo": "single_cta",
                        "filter_mode": "favor",
                        "itopk": 512,
                        "search_width": 1,
                        "max_iterations": CAP,
                        "max_queries": 2048,
                        "rand_xor_mask": 0x128394,
                        "hashmap_max_fill_rate": 0.89,
                        "filtering_rate": 0.99,
                        "favor_penalty_mode": "cagra_retention_safe",
                        "favor_penalty_lambda": 1.0,
                        "favor_retention_fraction": 0.0,
                        "favor_delta_d_file": str(
                            data_root / DIRECTORY / "cagra_g32_ig64.index.delta_d"
                        ),
                        "favor_delta_d_alpha": 10,
                        "favor_delta_d_beta": 64,
                        "favor_delta_d_bfs_depth": 2,
                        "favor_diagnostics_output": str(capture_root / f"seed{seed}"),
                        "favor_diagnostics_groundtruth": str(
                            data_root
                            / DIRECTORY
                            / f"favor_seed{seed}/groundtruth_s01.ibin"
                        ),
                        "favor_diagnostics_dataset": f"deep_image1m_seed{seed}",
                        "favor_diagnostics_variant": "frozen_progress_v4",
                        "favor_termination_shadow_record_start_iteration": 1,
                        "favor_termination_shadow_start_iteration": B0,
                        "favor_termination_shadow_parent_interval": 32,
                    }
                ],
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "deep_build.json").write_text(
        json.dumps(build_config(), indent=2) + "\n"
    )
    for seed in SEEDS:
        (args.output_dir / f"deep_seed{seed}_shadow.json").write_text(
            json.dumps(
                shadow_config(
                    seed, args.data_root.resolve(), args.capture_root.resolve()
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
