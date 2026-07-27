#!/usr/bin/env python3
"""Summarize the matched 1%-8% FAVOR/default crossover sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = {
    "sift": ("SIFT-1M", Path("benchmarks/favor/results")),
    "gist": ("GIST-1M", Path("benchmarks/favor/results_gist")),
    "bigann1m": ("BIGANN-1M", Path("benchmarks/favor/results_bigann_1m")),
    "bigann10m": ("BIGANN-10M", Path("benchmarks/favor/results_bigann_10m")),
    "msturing1m": (
        "MSTuring-1M",
        Path("benchmarks/favor/results_msturing_1m_threshold"),
    ),
    "msturing10m": (
        "MSTuring-10M",
        Path("benchmarks/favor/results_msturing_10m_threshold"),
    ),
}


def load_rows(path: Path, mode: str) -> list[dict]:
    rows = []
    for row in json.loads(path.read_text())["benchmarks"]:
        if row.get("run_type", "iteration") != "iteration" or int(row["itopk"]) != 512:
            continue
        label = row.get("label", "")
        row_mode = "favor" if 'filter_mode="favor"' in label else "default"
        if row_mode == mode:
            rows.append(row)
    return rows


def best_at_recall(rows: list[dict], metric: str, recall_target: float, maximize: bool):
    eligible = [row for row in rows if float(row["Recall"]) >= recall_target]
    if not eligible:
        return None
    key = (lambda row: float(row[metric]))
    return (max if maximize else min)(eligible, key=key)


def summarize(output_dir: Path, recall_target: float) -> list[dict]:
    summary = []
    for prefix, (title, result_dir) in DATASETS.items():
        for selectivity in range(1, 9):
            paths = {
                "throughput": result_dir / "raw" / f"{prefix}_s{selectivity:02d}_nq10000.json",
                "latency": result_dir / "raw" / f"{prefix}_s{selectivity:02d}_nq10.json",
            }
            for mode in ("default", "favor"):
                throughput = load_rows(paths["throughput"], mode)
                latency = load_rows(paths["latency"], mode)
                qps_row = best_at_recall(
                    throughput, "items_per_second", recall_target, maximize=True
                )
                latency_row = best_at_recall(
                    latency, "Latency", recall_target, maximize=False
                )
                max_recall_row = max(throughput, key=lambda row: float(row["Recall"]))
                summary.append(
                    {
                        "dataset": prefix,
                        "dataset_title": title,
                        "selectivity_percent": selectivity,
                        "mode": mode,
                        "max_throughput_recall": float(max_recall_row["Recall"]),
                        "max_recall_qps": float(max_recall_row["items_per_second"]),
                        "max_recall_width": int(max_recall_row["search_width"]),
                        "qps_at_recall_target": (
                            "" if qps_row is None else float(qps_row["items_per_second"])
                        ),
                        "qps_width": "" if qps_row is None else int(qps_row["search_width"]),
                        "latency_ms_at_recall_target": (
                            "" if latency_row is None else 1000.0 * float(latency_row["Latency"])
                        ),
                        "latency_width": (
                            "" if latency_row is None else int(latency_row["search_width"])
                        ),
                        "recall_target": recall_target,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "crossover_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot(summary: list[dict], output_dir: Path, recall_target: float) -> None:
    colors = {"default": "#1f77b4", "favor": "#d62728"}
    labels = {"default": "Default CAGRA", "favor": "FAVOR"}
    for prefix, (title, _) in DATASETS.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for mode in ("default", "favor"):
            rows = [
                row
                for row in summary
                if row["dataset"] == prefix and row["mode"] == mode
            ]
            x = [row["selectivity_percent"] for row in rows]
            axes[0].plot(
                x,
                [row["max_throughput_recall"] for row in rows],
                marker="o",
                color=colors[mode],
                label=labels[mode],
            )
            axes[1].plot(
                x,
                [
                    float("nan")
                    if row["qps_at_recall_target"] == ""
                    else row["qps_at_recall_target"]
                    for row in rows
                ],
                marker="o",
                color=colors[mode],
                label=labels[mode],
            )
            axes[2].plot(
                x,
                [
                    float("nan")
                    if row["latency_ms_at_recall_target"] == ""
                    else row["latency_ms_at_recall_target"]
                    for row in rows
                ],
                marker="o",
                color=colors[mode],
                label=labels[mode],
            )
        axes[0].axhline(recall_target, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Maximum recall@10")
        axes[1].set_ylabel(f"Best QPS at recall ≥ {recall_target:.2f}")
        axes[2].set_ylabel(f"Best 10-query batch latency (ms)\nat recall ≥ {recall_target:.2f}")
        for axis in axes:
            axis.set_xlabel("Selectivity (%)")
            axis.set_xticks(range(1, 9))
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.suptitle(
            f"{title}: default CAGRA vs FAVOR crossover\n"
            "itopk=512, search widths 1/2/4, one run per configuration"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_crossover_01_08.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/favor/results_crossover")
    )
    parser.add_argument("--recall-target", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 < args.recall_target <= 1.0:
        raise ValueError("recall target must be in (0, 1]")
    summary = summarize(args.output_dir, args.recall_target)
    plot(summary, args.output_dir, args.recall_target)


if __name__ == "__main__":
    main()
