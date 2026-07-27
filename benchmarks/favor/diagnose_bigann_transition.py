#!/usr/bin/env python3
"""Generate and summarize focused BIGANN FAVOR diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


DATASETS = {
    "bigann1m": {
        "name": "bigann-1M",
        "title": "BIGANN-1M",
        "subset_size": 1_000_000,
    },
    "bigann10m": {
        "name": "bigann-10M",
        "title": "BIGANN-10M",
    },
}
TRANSITION_MULTIPLIERS = (
    0.0001,
    0.00015,
    0.0002,
    0.0003,
    0.0004,
    0.0005,
    0.00075,
    0.001,
)


def read_delta(path: Path) -> float:
    data = path.read_bytes()
    if len(data) != 80 or data[:8] != b"CUVSDD\r\n":
        raise ValueError(f"invalid delta-d sidecar: {path}")
    return struct.unpack_from("<f", data, 52)[0]


def make_dataset(metadata: dict) -> dict:
    name = metadata["name"]
    result = {
        "name": f"{name}-diagnostic-s10",
        "base_file": f"{name}/base.10M.u8bin",
        "query_file": f"{name}/query.public.10K.u8bin",
        "groundtruth_neighbors_file": f"{name}/favor/groundtruth_s10.ibin",
        "filter_bitset_file": f"{name}/favor/filter_s10.bin",
        "distance": "euclidean",
        "dtype": "uint8",
    }
    if "subset_size" in metadata:
        result["subset_size"] = metadata["subset_size"]
    return result


def favor(delta: float, multiplier: float, **kwargs) -> dict:
    return {
        "algo": "single_cta",
        "filter_mode": "favor",
        "favor_delta_d": delta * multiplier,
        "itopk": 512,
        "search_width": 1,
        **kwargs,
    }


def generate(data_dir: Path, result_dir: Path) -> None:
    config_dir = result_dir / "configs"
    raw_dir = result_dir / "raw"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for prefix, metadata in DATASETS.items():
        name = metadata["name"]
        delta = read_delta(data_dir / name / "cagra_g32_ig64.index.delta_d")
        manifest[prefix] = {"delta_d": delta, **metadata}
        experiments = {
            "transition": [
                favor(delta, multiplier) for multiplier in TRANSITION_MULTIPLIERS
            ],
            "budget": [
                {
                    "algo": "single_cta",
                    "filter_mode": "default",
                    "itopk": 512,
                    "search_width": 1,
                    "max_iterations": iterations,
                }
                for iterations in (0, 1024)
            ]
            + [
                favor(delta, multiplier, max_iterations=iterations)
                for multiplier in (0.0, 1.0)
                for iterations in (0, 1024)
            ],
            "width": [
                {
                    "algo": "single_cta",
                    "filter_mode": mode,
                    **({"favor_delta_d": delta} if mode == "favor" else {}),
                    "itopk": 512,
                    "search_width": width,
                }
                for width in (1, 2, 4)
                for mode in ("default", "favor")
            ],
        }
        for experiment, search_params in experiments.items():
            config = {
                "dataset": make_dataset(metadata),
                "search_basic_param": {"batch_size": 10_000, "k": 10},
                "index": [
                    {
                        "name": "cagra-g32-ig64",
                        "algo": "cuvs_cagra",
                        "file": f"{name}/cagra_g32_ig64.index",
                        "build_param": {
                            "graph_build_algo": "NN_DESCENT",
                            "graph_degree": 32,
                            "intermediate_graph_degree": 64,
                        },
                        "search_params": search_params,
                    }
                ],
            }
            path = config_dir / f"{prefix}_{experiment}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")

    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summarize(result_dir: Path) -> None:
    manifest = json.loads((result_dir / "manifest.json").read_text())
    rows = []
    for config_path in sorted((result_dir / "configs").glob("*.json")):
        prefix, experiment = config_path.stem.split("_", 1)
        config = json.loads(config_path.read_text())
        params = config["index"][0]["search_params"]
        result_path = result_dir / "raw" / config_path.name
        iterations = [
            row
            for row in json.loads(result_path.read_text())["benchmarks"]
            if row.get("run_type") == "iteration"
        ]
        if len(iterations) != len(params):
            raise ValueError(
                f"{result_path}: expected {len(params)} iterations, found {len(iterations)}"
            )
        delta = manifest[prefix]["delta_d"]
        for result, param in zip(iterations, params):
            favor_delta = float(param.get("favor_delta_d", 0.0))
            rows.append(
                {
                    "dataset": prefix,
                    "experiment": experiment,
                    "mode": param["filter_mode"],
                    "penalty_multiplier": (
                        favor_delta / delta if param["filter_mode"] == "favor" else ""
                    ),
                    "effective_penalty": (
                        favor_delta * 0.9 * (512 - 0.1) / (2 * 0.1 * 512)
                        if param["filter_mode"] == "favor"
                        else ""
                    ),
                    "max_iterations": param.get("max_iterations", 0),
                    "search_width": param["search_width"],
                    "recall": result["Recall"],
                    "qps": result["items_per_second"],
                }
            )

    output = result_dir / "bigann_diagnostic_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['dataset']:10s} {row['experiment']:10s} {row['mode']:7s} "
            f"mult={str(row['penalty_multiplier']):>9s} "
            f"iters={int(row['max_iterations']):4d} width={int(row['search_width'])} "
            f"recall={float(row['recall']):.5f} qps={float(row['qps']):.0f}"
        )
    plot_transition(rows, result_dir)


def plot_transition(rows: list[dict], result_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    labels = {"bigann1m": "BIGANN-1M", "bigann10m": "BIGANN-10M"}
    for dataset, label in labels.items():
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset and row["experiment"] == "transition"
        ]
        selected.sort(key=lambda row: float(row["effective_penalty"]))
        axis.plot(
            [float(row["effective_penalty"]) for row in selected],
            [float(row["recall"]) for row in selected],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.axhline(0.9, color="black", linestyle="--", linewidth=1, label="0.90 recall")
    axis.set_xscale("log")
    axis.set_xlabel("Effective penalty D added to rejected candidates")
    axis.set_ylabel("Recall@10")
    axis.set_title("BIGANN FAVOR penalty transition at 10% selectivity\n"
                   "10,000 queries, SINGLE_CTA, itopk=512, search width=1")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "bigann_penalty_transition.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("benchmarks/favor/results_bigann_diagnostic"),
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.result_dir)
    else:
        generate(args.data_dir, args.result_dir)


if __name__ == "__main__":
    main()
