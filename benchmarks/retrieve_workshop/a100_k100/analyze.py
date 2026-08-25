#!/usr/bin/env python3
"""Combine k=100 CAGRA and cuVS Brute Force KNN A100 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

K = 100
WORKLOADS = ("yfcc", "em", "emis", "r")
WORKLOAD_LABELS = {
    "yfcc": "YFCC-10M",
    "em": "ArXiv-large EM",
    "emis": "ArXiv-large EMIS",
    "r": "ArXiv-large R",
}
METHODS = {
    "default_cagra": ("CAGRA-Base", "#1f77b4", "o"),
    "default_cagra_accumulator": ("CAGRA-Retain", "#d62728", "s"),
    "navix_reference": ("CAGRA-NaviX", "#2ca02c", "^"),
    "cuvs_brute_force_knn": ("cuVS Brute Force KNN", "#111111", "x"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    best_qps = -math.inf
    for row in sorted(
        rows,
        key=lambda item: (float(item["recall"]), float(item["qps"])),
        reverse=True,
    ):
        if float(row["qps"]) > best_qps:
            frontier.append(row)
            best_qps = float(row["qps"])
    return sorted(frontier, key=lambda item: float(item["recall"]))


def combined_rows(run_root: Path) -> list[dict[str, object]]:
    graph_provenance = json.loads(
        (run_root / "gpu_graph/analysis/provenance.json").read_text()
    )
    exact = json.loads(
        (run_root / "exact_bitmap/analysis/exact_results.json").read_text()
    )
    if int(graph_provenance.get("k", -1)) != K or int(exact.get("k", -1)) != K:
        raise ValueError("graph/exact analyses are not both k=100")

    result: list[dict[str, object]] = []
    for row in read_csv(run_root / "gpu_graph/analysis/summary_points.csv"):
        if (
            row["phase"] != "throughput"
            or row["method"] not in METHODS
            or row["method"] == "cuvs_brute_force_knn"
            or int(row["max_iterations"]) != 0
            or not truth(row["paper_included"])
        ):
            continue
        result.append(
            {
                "workload": row["workload"],
                "method": row["method"],
                "itopk": int(row["itopk"]),
                "search_width": int(row["search_width"]),
                "max_iterations": 0,
                "recall": float(row["recall_median"]),
                "recall_min": float(row["recall_min"]),
                "recall_max": float(row["recall_max"]),
                "qps": float(row["qps_median"]),
                "qps_min": float(row["qps_min"]),
                "qps_max": float(row["qps_max"]),
                "repetitions": int(row["repetitions"]),
            }
        )
    for row in exact["summaries"]:
        if row["phase"] != "throughput":
            continue
        if not bool(row["correct"]):
            raise ValueError(f"incorrect exact result: {row['workload']}")
        result.append(
            {
                "workload": row["workload"],
                "method": "cuvs_brute_force_knn",
                "itopk": "",
                "search_width": "",
                "max_iterations": "",
                "recall": float(row["native_l2_cutoff_recall"]),
                "recall_min": float(row["native_l2_cutoff_recall"]),
                "recall_max": float(row["native_l2_cutoff_recall"]),
                "qps": float(row["median_qps"]),
                "qps_min": float(row["min_qps"]),
                "qps_max": float(row["max_qps"]),
                "repetitions": int(row["repetitions"]),
            }
        )
    expected_graph = 3 * 3 * 2 * len(WORKLOADS)
    expected_exact = len(WORKLOADS)
    if len(result) != expected_graph + expected_exact:
        raise ValueError(
            f"incomplete k=100 combined set: {len(result)} != {expected_graph + expected_exact}"
        )
    return result


def plot_axis(
    axis, workload: str, rows: list[dict[str, object]]
) -> tuple[list, list]:
    handles: list = []
    labels: list[str] = []
    x_values: list[float] = []
    for method, (label, color, marker) in METHODS.items():
        local = [
            row
            for row in rows
            if row["workload"] == workload and row["method"] == method
        ]
        if not local:
            raise ValueError(f"missing {workload}/{method}")
        if method == "cuvs_brute_force_knn":
            row = local[0]
            artist = axis.scatter(
                float(row["recall"]),
                float(row["qps"]),
                color=color,
                marker=marker,
                s=48,
                label=label,
            )
            x_values.append(float(row["recall"]))
        else:
            front = pareto(local)
            xs = [float(row["recall"]) for row in front]
            ys = [float(row["qps"]) for row in front]
            artist = axis.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4.5,
                label=label,
            )[0]
            x_values.extend(xs)
        handles.append(artist)
        labels.append(label)
    low = min(x_values)
    axis.set_xlim(max(0.0, low - max(0.01, 0.04 * (1.0 - low))), 1.005)
    axis.set_ylim(bottom=0)
    axis.set_xlabel(f"Recall@{K}")
    axis.set_title(WORKLOAD_LABELS[workload])
    axis.grid(alpha=0.22)
    return handles, labels


def write_plots(output: Path, rows: list[dict[str, object]]) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for workload in WORKLOADS:
        fig, axis = plt.subplots(figsize=(8.4, 5.2))
        plot_axis(axis, workload, rows)
        axis.set_ylabel("Queries per second")
        axis.legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(
                plots / f"{workload}_qps_recall_k100.{extension}", dpi=220
            )
        plt.close(fig)

    fig, axes = plt.subplots(
        1, 4, figsize=(16.0, 3.7), constrained_layout=True
    )
    legend_handles: list = []
    legend_labels: list[str] = []
    for axis, workload in zip(axes, WORKLOADS, strict=True):
        handles, labels = plot_axis(axis, workload, rows)
        if not legend_handles:
            legend_handles, legend_labels = handles, labels
    axes[0].set_ylabel("Queries per second")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            plots / f"gpu_qps_recall_k100.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = (args.output or run_root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = combined_rows(run_root)
    write_csv(output / "combined_points.csv", rows)
    write_plots(output, rows)
    summary = {
        "schema_version": 1,
        "k": K,
        "max_queries": 2048,
        "workloads": list(WORKLOADS),
        "methods": list(METHODS),
        "graph_points": sum(
            row["method"] != "cuvs_brute_force_knn" for row in rows
        ),
        "exact_points": sum(
            row["method"] == "cuvs_brute_force_knn" for row in rows
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
