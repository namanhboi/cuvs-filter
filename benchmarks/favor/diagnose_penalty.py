#!/usr/bin/env python3
"""Generate and summarize focused FAVOR penalty diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


MULTIPLIERS = (
    0.0,
    0.00000001,
    0.000001,
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    11.0,
    1_000_000.0,
)
DATASETS = {
    "sift": {
        "name": "sift-128-euclidean",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "title": "SIFT-1M",
    },
    "gist": {
        "name": "gist-960-euclidean",
        "base": "base.fbin",
        "query": "query_10000.fbin",
        "dtype": "float",
        "title": "GIST-1M",
    },
    "bigann1m": {
        "name": "bigann-1M",
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
        "title": "BIGANN-1M",
        "subset_size": 1_000_000,
    },
    "bigann10m": {
        "name": "bigann-10M",
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
        "title": "BIGANN-10M",
    },
}


def read_delta(path: Path) -> float:
    data = path.read_bytes()
    if len(data) != 80 or data[:8] != b"CUVSDD\r\n":
        raise ValueError(f"invalid delta-d sidecar: {path}")
    return struct.unpack_from("<f", data, 52)[0]


def search_params(delta: float, selectivity: int) -> list[dict]:
    params = [
        {
            "algo": "single_cta",
            "filter_mode": "default",
            "itopk": 512,
            "search_width": 1,
        }
    ]
    multipliers = (1.0,) if selectivity == 1 else MULTIPLIERS
    for multiplier in multipliers:
        params.append(
            {
                "algo": "single_cta",
                "filter_mode": "favor",
                "favor_delta_d": delta * multiplier,
                "itopk": 512,
                "search_width": 1,
            }
        )
    return params


def generate(data_dir: Path, result_dir: Path) -> None:
    config_dir = result_dir / "configs"
    raw_dir = result_dir / "raw"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for prefix, metadata in DATASETS.items():
        dataset_name = metadata["name"]
        delta = read_delta(data_dir / dataset_name / "cagra_g32_ig64.index.delta_d")
        manifest[prefix] = {"delta_d": delta, **metadata}
        for selectivity in (1, 10):
            dataset = {
                "name": f"{dataset_name}-diagnostic-s{selectivity:02d}",
                "base_file": f"{dataset_name}/{metadata['base']}",
                "query_file": f"{dataset_name}/{metadata['query']}",
                "groundtruth_neighbors_file": (
                    f"{dataset_name}/favor/groundtruth_s{selectivity:02d}.ibin"
                ),
                "filter_bitset_file": f"{dataset_name}/favor/filter_s{selectivity:02d}.bin",
                "distance": "euclidean",
                "dtype": metadata["dtype"],
            }
            if "subset_size" in metadata:
                dataset["subset_size"] = metadata["subset_size"]
            config = {
                "dataset": dataset,
                "search_basic_param": {"batch_size": 10_000, "k": 10},
                "index": [
                    {
                        "name": "cagra-g32-ig64",
                        "algo": "cuvs_cagra",
                        "file": f"{dataset_name}/cagra_g32_ig64.index",
                        "build_param": {
                            "graph_build_algo": "NN_DESCENT",
                            "graph_degree": 32,
                            "intermediate_graph_degree": 64,
                        },
                        "search_params": search_params(delta, selectivity),
                    }
                ],
            }
            path = config_dir / f"{prefix}_s{selectivity:02d}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")

    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summarize(result_dir: Path) -> None:
    output_rows = []
    for config_path in sorted((result_dir / "configs").glob("*.json")):
        result_path = result_dir / "raw" / config_path.name
        config = json.loads(config_path.read_text())
        params = config["index"][0]["search_params"]
        benchmarks = json.loads(result_path.read_text())["benchmarks"]
        iterations = [row for row in benchmarks if row.get("run_type") == "iteration"]
        if len(iterations) != len(params):
            raise ValueError(
                f"{result_path}: expected {len(params)} iterations, found {len(iterations)}"
            )
        prefix, selectivity_text = config_path.stem.rsplit("_s", 1)
        base_delta = json.loads((result_dir / "manifest.json").read_text())[prefix][
            "delta_d"
        ]
        for row, param in zip(iterations, params):
            delta = float(param.get("favor_delta_d", 0.0))
            output_rows.append(
                {
                    "dataset": prefix,
                    "selectivity": int(selectivity_text),
                    "mode": param["filter_mode"],
                    "penalty_multiplier": (
                        "" if param["filter_mode"] == "default" else delta / base_delta
                    ),
                    "delta_d": delta,
                    "recall": row["Recall"],
                    "qps": row["items_per_second"],
                    "itopk": param["itopk"],
                    "search_width": param["search_width"],
                }
            )

    output = result_dir / "penalty_diagnostic_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            f"{row['dataset']:10s} s{row['selectivity']:02d} {row['mode']:7s} "
            f"mult={str(row['penalty_multiplier']):>9s} "
            f"recall={float(row['recall']):.5f} qps={float(row['qps']):.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("benchmarks/favor/results_penalty_diagnostic"),
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.result_dir)
    else:
        generate(args.data_dir, args.result_dir)


if __name__ == "__main__":
    main()
