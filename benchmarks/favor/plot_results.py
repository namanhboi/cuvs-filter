#!/usr/bin/env python3
"""Plot one clearly labelled large-batch/low-batch Pareto figure per selectivity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_points(path: Path, metric: str) -> dict[str, list[dict[str, float]]]:
    payload = json.loads(path.read_text())
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in payload["benchmarks"]:
        if row.get("run_type", "iteration") != "iteration":
            continue
        label = row.get("label", "")
        mode = "favor" if 'filter_mode="favor"' in label else "default"
        key = (mode, int(row["itopk"]), int(row["search_width"]))
        grouped[key].append(row)

    result: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (mode, itopk, width), repetitions in grouped.items():
        recall = float(np.median([r["Recall"] for r in repetitions]))
        value = float(np.median([r[metric] for r in repetitions]))
        result[mode].append(
            {"recall": recall, "value": value, "itopk": itopk, "search_width": width}
        )
    return result


def pareto(points: list[dict[str, float]], maximize: bool) -> list[dict[str, float]]:
    ordered = sorted(points, key=lambda point: point["recall"], reverse=True)
    frontier = []
    best = -np.inf if maximize else np.inf
    for point in ordered:
        value = point["value"]
        if (maximize and value > best) or (not maximize and value < best):
            frontier.append(point)
            best = value
    return sorted(frontier, key=lambda point: point["recall"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--result-prefix", default="sift")
    parser.add_argument("--plot-title", default="SIFT-1M")
    parser.add_argument(
        "--selectivities", type=int, nargs="+", default=(1, 10, 50, 90)
    )
    args = parser.parse_args()
    plot_dir = args.result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    colors = {"default": "#1f77b4", "favor": "#d62728"}
    labels = {"default": "Default CAGRA filtering", "favor": "FAVOR filtering"}
    for selectivity in args.selectivities:
        throughput = load_points(
            args.result_dir
            / "raw"
            / f"{args.result_prefix}_s{selectivity:02d}_nq10000.json",
            "items_per_second",
        )
        latency = load_points(
            args.result_dir / "raw" / f"{args.result_prefix}_s{selectivity:02d}_nq10.json",
            "Latency",
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for mode in ("default", "favor"):
            qps_frontier = pareto(throughput[mode], maximize=True)
            latency_frontier = pareto(latency[mode], maximize=False)
            axes[0].plot(
                [p["recall"] for p in qps_frontier],
                [p["value"] for p in qps_frontier],
                marker="o",
                color=colors[mode],
                label=labels[mode],
            )
            axes[1].plot(
                [p["recall"] for p in latency_frontier],
                [1000.0 * p["value"] for p in latency_frontier],
                marker="o",
                color=colors[mode],
                label=labels[mode],
            )
            for workload, points in (("throughput", throughput[mode]), ("latency", latency[mode])):
                for point in points:
                    csv_rows.append(
                        {
                            "selectivity": selectivity / 100,
                            "workload": workload,
                            "mode": mode,
                            **point,
                        }
                    )

        axes[0].set_title("Large batch (10,000 queries)")
        axes[0].set_xlabel("Recall@10")
        axes[0].set_ylabel("Throughput (queries/second, higher is better)")
        axes[1].set_title("Low batch (10 queries)")
        axes[1].set_xlabel("Recall@10")
        axes[1].set_ylabel("Batch latency (milliseconds, lower is better)")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle(
            f"{args.plot_title} filtered CAGRA: {selectivity}% selectivity\n"
            "Graph degree 32, intermediate graph degree 64; one measured run per configuration"
        )
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"{args.result_prefix}_selectivity_{selectivity:02d}.png", dpi=180
        )
        plt.close(fig)

    with (args.result_dir / "favor_benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "selectivity",
                "workload",
                "mode",
                "recall",
                "value",
                "itopk",
                "search_width",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
