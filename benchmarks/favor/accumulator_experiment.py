#!/usr/bin/env python3
"""Generate and analyze CAGRA passing-result accumulator experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from static_penalty_experiment import (
    DATASETS,
    dataset_config,
    pareto,
    read_delta_d,
    result_index,
)


DEFAULT_SELECTIVITIES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 50, 90)
DEFAULT_ITOPK_VALUES = (32, 64, 128, 256, 512)
DEFAULT_SEARCH_WIDTHS = (1, 2, 4)
DEFAULT_BATCH_SIZES = (10, 10_000)
METHODS = {
    "default": {
        "label": "Default CAGRA filtering",
        "filter_mode": "default",
        "color": "#1f77b4",
        "marker": "o",
    },
    "favor": {
        "label": "Current FAVOR",
        "filter_mode": "favor",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "favor_accumulator": {
        "label": "FAVOR + passing-result accumulator",
        "filter_mode": "favor_accumulator",
        "color": "#2ca02c",
        "marker": "^",
    },
}


def generate(args: argparse.Namespace) -> None:
    config_dir = args.result_dir / "configs"
    (args.result_dir / "raw").mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "description": (
            "Matched Pareto comparison of default CAGRA, current FAVOR, and "
            "FAVOR with an in-kernel passing-result accumulator."
        ),
        "methods": {key: value["label"] for key, value in METHODS.items()},
        "configs": {},
    }

    for dataset_key in args.datasets:
        metadata = DATASETS[dataset_key]
        delta_path = (
            args.data_dir / metadata["name"] / "cagra_g32_ig64.index.delta_d"
        )
        delta_d = read_delta_d(delta_path)
        for selectivity in args.selectivities:
            if not 1 <= selectivity <= 100:
                raise ValueError("selectivities must be percentages in [1, 100]")
            for batch_size in args.batch_sizes:
                params = []
                entries = []
                for itopk in args.itopk_values:
                    for search_width in args.search_widths:
                        for method, method_info in METHODS.items():
                            param = {
                                "algo": "single_cta",
                                "filter_mode": method_info["filter_mode"],
                                "itopk": itopk,
                                "search_width": search_width,
                            }
                            if method != "default":
                                param["favor_delta_d"] = delta_d
                            params.append(param)
                            entries.append(
                                {
                                    "param_index": len(params) - 1,
                                    "method": method,
                                    "method_label": method_info["label"],
                                    "itopk": itopk,
                                    "search_width": search_width,
                                }
                            )

                config = {
                    "dataset": dataset_config(metadata, selectivity),
                    "search_basic_param": {"batch_size": batch_size, "k": args.k},
                    "index": [
                        {
                            "name": "cagra-g32-ig64",
                            "algo": "cuvs_cagra",
                            "file": f"{metadata['name']}/cagra_g32_ig64.index",
                            "build_param": {
                                "graph_build_algo": "NN_DESCENT",
                                "graph_degree": 32,
                                "intermediate_graph_degree": 64,
                            },
                            "search_params": params,
                        }
                    ],
                }
                filename = f"{dataset_key}_s{selectivity:02d}_nq{batch_size}.json"
                (config_dir / filename).write_text(
                    json.dumps(config, indent=2) + "\n"
                )
                manifest["configs"][filename] = {
                    "dataset": dataset_key,
                    "dataset_title": metadata["title"],
                    "selectivity_percent": selectivity,
                    "batch_size": batch_size,
                    "k": args.k,
                    "delta_d": delta_d,
                    "entries": entries,
                }

    (args.result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"generated {len(manifest['configs'])} configurations in {config_dir}")


def summarize(args: argparse.Namespace) -> None:
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    rows = []
    missing = []
    for filename, record in manifest["configs"].items():
        result_path = args.result_dir / "raw" / filename
        if not result_path.exists():
            missing.append(filename)
            continue
        payload = json.loads(result_path.read_text())
        grouped = defaultdict(list)
        for row in payload["benchmarks"]:
            if (
                row.get("run_type", "iteration") == "iteration"
                and not row.get("error_occurred", False)
            ):
                grouped[result_index(row)].append(row)
        if len(grouped) != len(record["entries"]):
            raise ValueError(
                f"{result_path}: expected {len(record['entries'])} parameter "
                f"results, found {len(grouped)}"
            )

        throughput = record["batch_size"] == 10_000
        metric = "items_per_second" if throughput else "Latency"
        for entry in record["entries"]:
            repetitions = grouped[entry["param_index"]]
            values = sorted(float(row[metric]) for row in repetitions)
            recalls = sorted(float(row["Recall"]) for row in repetitions)
            rows.append(
                {
                    "dataset": record["dataset"],
                    "dataset_title": record["dataset_title"],
                    "selectivity_percent": record["selectivity_percent"],
                    "batch_size": record["batch_size"],
                    "k": record["k"],
                    "workload": "throughput" if throughput else "latency",
                    "method": entry["method"],
                    "method_label": entry["method_label"],
                    "itopk": entry["itopk"],
                    "search_width": entry["search_width"],
                    "recall": median(recalls),
                    "value": median(values),
                }
            )
    if not rows:
        raise ValueError("no complete benchmark results found")
    output = args.result_dir / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if missing:
        print(f"skipped {len(missing)} missing result files")


def plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    with (args.result_dir / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["dataset_title"],
                int(row["selectivity_percent"]),
                row["workload"],
                int(row["batch_size"]),
            )
        ].append(row)

    plot_dir = args.result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, title, selectivity, workload, batch_size), group in groups.items():
        by_method = defaultdict(list)
        for row in group:
            by_method[row["method"]].append(row)
        required = {"default", "favor_accumulator"}
        missing = required - set(by_method)
        if missing:
            raise ValueError(
                f"{dataset} s{selectivity:02d} {workload}: missing required "
                f"methods {sorted(missing)}"
            )

        figure, axis = plt.subplots(figsize=(8.5, 6.0))
        for method in METHODS:
            if method not in by_method:
                continue
            style = METHODS[method]
            frontier = pareto(
                by_method[method], maximize=(workload == "throughput")
            )
            scale = 1.0 if workload == "throughput" else 1000.0
            axis.plot(
                [float(point["recall"]) for point in frontier],
                [scale * float(point["value"]) for point in frontier],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5,
                label=style["label"],
            )
        axis.set_xlabel(f"Recall@{group[0].get('k', 10)}")
        if workload == "throughput":
            axis.set_ylabel("Throughput (queries/second; higher is better)")
            workload_label = f"large batch ({batch_size:,} queries)"
        else:
            axis.set_ylabel("Batch latency (milliseconds; lower is better)")
            workload_label = f"low batch ({batch_size} queries)"
        axis.set_title(
            f"{title}: {selectivity}% selectivity\n"
            f"{workload_label}; graph degree 32, intermediate degree 64"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            plot_dir / f"{dataset}_s{selectivity:02d}_{workload}.png", dpi=180
        )
        plt.close(figure)
    print(f"wrote {len(groups)} Pareto plots to {plot_dir}")


def compare(args: argparse.Namespace) -> None:
    with (args.result_dir / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                int(row["selectivity_percent"]),
                row["workload"],
                row["method"],
            )
        ].append(row)

    output = []
    for (dataset, selectivity, workload, method), points in sorted(groups.items()):
        record = {
            "dataset": dataset,
            "selectivity_percent": selectivity,
            "workload": workload,
            "method": method,
            "max_recall": max(float(point["recall"]) for point in points),
        }
        for target in (0.90, 0.95):
            eligible = [
                float(point["value"])
                for point in points
                if float(point["recall"]) >= target
            ]
            if not eligible:
                value = ""
            elif workload == "throughput":
                value = max(eligible)
            else:
                value = min(eligible)
            record[f"best_value_recall_{int(target * 100)}"] = value
        output.append(record)
    path = args.result_dir / "comparison.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {path}")


def self_check(_: argparse.Namespace) -> None:
    required = {"default", "favor", "favor_accumulator"}
    if set(METHODS) != required:
        raise AssertionError("method registry is incomplete")
    if not all(math.isfinite(float(value)) for value in (32, 64, 128, 256, 512)):
        raise AssertionError("invalid itopk sweep")
    print("accumulator experiment checks passed")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("generate")
    command.add_argument("--result-dir", type=Path, required=True)
    command.add_argument("--data-dir", type=Path, default=Path("datasets"))
    command.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    command.add_argument(
        "--selectivities", type=int, nargs="+", default=DEFAULT_SELECTIVITIES
    )
    command.add_argument(
        "--itopk-values", type=int, nargs="+", default=DEFAULT_ITOPK_VALUES
    )
    command.add_argument(
        "--search-widths", type=int, nargs="+", default=DEFAULT_SEARCH_WIDTHS
    )
    command.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES
    )
    command.add_argument("--k", type=int, default=10)
    command.set_defaults(function=generate)
    for name, function in (
        ("summarize", summarize),
        ("plot", plot),
        ("compare", compare),
        ("check", self_check),
    ):
        command = commands.add_parser(name)
        if name != "check":
            command.add_argument("--result-dir", type=Path, required=True)
        command.set_defaults(function=function)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
