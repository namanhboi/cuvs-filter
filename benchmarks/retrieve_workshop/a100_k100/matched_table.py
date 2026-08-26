#!/usr/bin/env python3
"""Validate and render the A100 Recall@100 matched-recall result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

K = 100
MAX_QUERIES = 2048
TARGETS = {"yfcc": 0.80, "em": 0.95, "emis": 0.95, "r": 0.95}
TARGET_WINDOW = 0.002
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
}
TABLE_FIELDS = (
    "workload",
    "workload_label",
    "target_low",
    "target_high",
    "method",
    "status",
    "recall",
    "qps",
    "itopk",
    "search_width",
    "max_iterations",
    "resolved_iterations",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def qps_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    return f"{value:.0f}"


def validate(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    provenance = json.loads((root / "analysis/provenance.json").read_text())
    if int(provenance.get("k", -1)) != K:
        raise ValueError("matched-recall provenance is not k=100")
    if int(provenance.get("max_queries", -1)) != MAX_QUERIES:
        raise ValueError("matched-recall provenance is not max_queries=2048")
    if provenance.get("targets") != TARGETS or float(
        provenance.get("target_window", -1)
    ) != TARGET_WINDOW:
        raise ValueError("matched-recall target contract changed")

    selected_source = read_csv(root / "analysis/selected_points.csv")
    if len(selected_source) != len(WORKLOADS) * len(METHODS):
        raise ValueError("k=100 matched table requires twelve selected points")
    selected: list[dict[str, object]] = []
    for source in selected_source:
        workload = source["workload"]
        method = source["method"]
        if workload not in WORKLOADS or method not in METHODS:
            raise ValueError(f"unexpected selected cell: {workload}/{method}")
        goal = TARGETS[workload]
        recall = float(source["recall_median"])
        within = truth(source["within_target_window"])
        reached = truth(source["target_reached"])
        if within != (float(source["recall_min"]) >= goal and recall <= goal + TARGET_WINDOW):
            raise ValueError(f"target-window flag disagrees with recall: {workload}/{method}")
        if float(source["filter_violations"]) != 0 or float(source["sentinel_errors"]) != 0:
            raise ValueError(f"correctness failure: {workload}/{method}")
        if method != "default_cagra" and float(source["duplicate_output_query_rate_max"]) != 0:
            raise ValueError(f"duplicate Retain/NaviX output: {workload}/{method}")
        status = "matched" if within else ("nearest_overshoot" if reached else "unreached")
        selected.append(
            {
                "workload": workload,
                "workload_label": WORKLOAD_LABELS[workload],
                "target_low": goal,
                "target_high": goal + TARGET_WINDOW,
                "method": method,
                "status": status,
                "recall": recall,
                "qps": float(source["qps_median"]),
                "itopk": int(source["itopk"]),
                "search_width": int(source["search_width"]),
                "max_iterations": int(source["max_iterations"]),
                "resolved_iterations": int(source["resolved_iterations"]),
            }
        )
    identities = {(row["workload"], row["method"]) for row in selected}
    expected = {(workload, method) for workload in WORKLOADS for method in METHODS}
    if identities != expected:
        raise ValueError("selected k=100 table has duplicate or missing cells")

    measurements: list[dict[str, object]] = []
    for source in read_csv(root / "analysis/measurements.csv"):
        if source["workload"] not in WORKLOADS or source["method"] not in METHODS:
            continue
        measurements.append(
            {
                "workload": source["workload"],
                "method": source["method"],
                "recall": float(source["recall"]),
                "qps": float(source["qps"]),
            }
        )
    if not measurements:
        raise ValueError("matched-recall measurement set is empty")
    return selected, measurements


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, selected: list[dict[str, object]]) -> None:
    lines = [
        "% Generated by benchmarks/retrieve_workshop/a100_k100/matched_table.py; do not edit.",
        r"\newcommand{\FixedRecallKOneHundredRows}{%",
    ]
    for workload in WORKLOADS:
        local = {row["method"]: row for row in selected if row["workload"] == workload}
        goal = TARGETS[workload]
        cells: list[str] = []
        for method in METHODS:
            row = local[method]
            if row["status"] == "matched":
                cells.append(f"{qps_label(float(row['qps']))} ({float(row['recall']):.4f})")
            elif row["status"] == "nearest_overshoot":
                cells.append(f"-- [nearest {float(row['recall']):.4f}]")
            else:
                cells.append(f"-- [max {float(row['recall']):.4f}]")
        lines.append(
            f"{WORKLOAD_LABELS[workload]} & $[{goal:.3f},{goal + TARGET_WINDOW:.3f}]$ & "
            + " & ".join(cells)
            + r" \\%"
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, measurements: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 3.7), constrained_layout=True)
    handles = []
    labels = []
    for workload_index, (axis, workload) in enumerate(zip(axes, WORKLOADS, strict=True)):
        for method, (label, color, marker) in METHODS.items():
            local = [
                row
                for row in measurements
                if row["workload"] == workload and row["method"] == method
            ]
            front = pareto(local)
            artist = axis.plot(
                [float(row["recall"]) for row in front],
                [float(row["qps"]) for row in front],
                color=color,
                marker=marker,
                linewidth=1.4,
                markersize=3.8,
                label=label,
            )[0]
            axis.scatter(
                [float(row["recall"]) for row in local],
                [float(row["qps"]) for row in local],
                color=color,
                marker=marker,
                s=11,
                alpha=0.35,
            )
            if workload_index == 0:
                handles.append(artist)
                labels.append(label)
        axis.set_title(WORKLOAD_LABELS[workload])
        axis.set_xlabel("Recall@100")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Queries per second")
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    for extension in ("pdf", "png"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    output = (args.output or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected, measurements = validate(root)
    selected.sort(key=lambda row: (WORKLOADS.index(str(row["workload"])), list(METHODS).index(str(row["method"]))))
    write_csv(output / "matched_recall_k100_table.csv", selected)
    (output / "matched_recall_k100_table.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "k": K,
                "max_queries": MAX_QUERIES,
                "targets": TARGETS,
                "target_window": TARGET_WINDOW,
                "selected": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_latex(output / "fixed_recall_k100_results.tex", selected)
    write_plot(output / "gpu_matched_recall_k100", measurements)
    print(json.dumps({"selected": len(selected), "measurements": len(measurements)}))


if __name__ == "__main__":
    main()
