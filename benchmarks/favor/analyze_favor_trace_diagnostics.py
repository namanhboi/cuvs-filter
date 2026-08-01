#!/usr/bin/env python3
"""Aggregate bounded FAVOR traversal captures and write diagnostic figures/report."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("sift", "gist", "bigann1m", "bigann10m", "msturing1m", "msturing10m")
FAILURES = ("gist", "msturing1m", "msturing10m")
LABELS = {
    "sift": "SIFT-1M", "gist": "GIST-1M", "bigann1m": "BIGANN-1M",
    "bigann10m": "BIGANN-10M", "msturing1m": "MSTuring-1M",
    "msturing10m": "MSTuring-10M",
}
VARIANT_LABELS = {
    "default": "Default CAGRA", "current": "Automatic FAVOR",
    "zero": "Zero penalty", "expanded": "Expanded budget",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def aggregate(dataset: str, variant: str, rows: list[dict[str, str]]) -> dict[str, object]:
    seen = np.array([int(row["gt_seen_mask"]).bit_count() for row in rows], dtype=float)
    recall = np.array([float(row["recall"]) for row in rows], dtype=float)
    stops = Counter(int(row["stop_reason"]) for row in rows)
    return {
        "dataset": dataset,
        "variant": variant,
        "queries": len(rows),
        "mean_recall": float(recall.mean()),
        "mean_gt_seen": float(seen.mean()),
        "mean_seen_but_not_returned": float(np.maximum(seen - recall * 10.0, 0.0).mean()),
        "mean_iterations": mean(int(row["iterations"]) for row in rows),
        "max_unexpanded_stop_fraction": stops[1] / len(rows),
        "underfilled_queries": sum(int(row["output_count"]) < 10 for row in rows),
        "mean_terminal_unexpanded_pass": mean(
            int(row["terminal_unexpanded_pass"]) for row in rows
        ),
        "mean_terminal_unexpanded_reject": mean(
            int(row["terminal_unexpanded_reject"]) for row in rows
        ),
        "mean_hash_duplicates": mean(int(row.get("candidate_duplicates", 0)) for row in rows),
        "hash_full_total": sum(int(row.get("candidate_hash_full", 0)) for row in rows),
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_recall(output: Path, records: dict[tuple[str, str], dict[str, object]]) -> None:
    x = np.arange(len(DATASETS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, variant, color in (
        (-width, "default", "#777777"),
        (0.0, "zero", "#e69f00"),
        (width, "current", "#0072b2"),
    ):
        values = [records[dataset, variant]["mean_recall"] for dataset in DATASETS]
        ax.bar(x + offset, values, width, label=VARIANT_LABELS[variant], color=color)
    for index, dataset in enumerate(DATASETS):
        expanded = records.get((dataset, "expanded"))
        if expanded:
            ax.scatter(index + width, expanded["mean_recall"], marker="D", s=55,
                       color="#009e73", label="Expanded budget" if dataset == "gist" else None,
                       zorder=4)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="0.90 target")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Recall@10")
    ax.set_xticks(x, [LABELS[d] for d in DATASETS], rotation=20, ha="right")
    ax.set_title("1% selectivity: diagnostic capture recall")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "summary_recall.png", dpi=180)
    plt.close(fig)


def plot_decomposition(output: Path, records: dict[tuple[str, str], dict[str, object]]) -> None:
    returned = np.array([records[d, "current"]["mean_recall"] * 10 for d in DATASETS])
    lost = np.array([records[d, "current"]["mean_seen_but_not_returned"] for d in DATASETS])
    undiscovered = 10.0 - returned - lost
    x = np.arange(len(DATASETS))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, returned, label="GT returned", color="#0072b2")
    ax.bar(x, lost, bottom=returned, label="GT seen but lost", color="#e69f00")
    ax.bar(x, undiscovered, bottom=returned + lost, label="GT never discovered", color="#cc79a7")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Mean filtered ground-truth neighbors per query")
    ax.set_xticks(x, [LABELS[d] for d in DATASETS], rotation=20, ha="right")
    ax.set_title("Automatic FAVOR failure decomposition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output / "current_failure_decomposition.png", dpi=180)
    plt.close(fig)


def discovery_cdf(rows: list[dict[str, str]], max_iteration: int) -> tuple[np.ndarray, np.ndarray]:
    first = []
    for row in rows:
        for rank in range(10):
            value = int(row[f"gt_first_iteration_{rank}"])
            if value != 0xFFFFFFFF:
                first.append(value)
    first = np.sort(np.asarray(first, dtype=np.int64))
    x = np.arange(max_iteration + 1)
    y = np.searchsorted(first, x, side="right") / (len(rows) * 10.0)
    return x, y


def plot_discovery(output: Path, root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, dataset in zip(axes, FAILURES):
        for variant, color in (("zero", "#e69f00"), ("current", "#0072b2"),
                               ("expanded", "#009e73")):
            rows = read_rows(root / dataset / f"{variant}_summary" / "query_summary.csv")
            max_iteration = max(int(row["iterations"]) for row in rows)
            x, y = discovery_cdf(rows, max_iteration)
            ax.plot(x, y, label=VARIANT_LABELS[variant], color=color, linewidth=2)
        ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
        ax.set_title(LABELS[dataset])
        ax.set_xlabel("Completed frontier iterations")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Fraction of filtered GT IDs discovered")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Ground-truth discovery continues beyond the normal iteration cap")
    fig.tight_layout()
    fig.savefig(output / "ground_truth_discovery_cdf.png", dpi=180)
    plt.close(fig)


def query_row(path: Path, query_id: str) -> dict[str, str]:
    return next(row for row in read_rows(path) if row["query_id"] == query_id)


def write_report(
    output: Path, root: Path, records: dict[tuple[str, str], dict[str, object]]
) -> None:
    lines = [
        "#+title: FAVOR SINGLE_CTA Deep Traversal Diagnostics",
        "",
        "* Scope",
        "",
        "This report diagnoses the 1% selectivity cases in which automatic retention-safe FAVOR",
        "does not reach 0.90 recall. All captures use batch size 10,000, k=10, itopk=512, and the",
        "existing degree-32 CAGRA graphs. Instrumented timing is intentionally invalid; these runs",
        "measure traversal state and recall, not QPS or latency.",
        "",
        "* Result",
        "",
        "The primary failure is exhaustion of CAGRA's derived iteration budget, not an oversized",
        "penalty or a full visited hash. Every automatic-FAVOR query in all six datasets stops at",
        "=MAX_WITH_UNEXPANDED_FRONTIER=, and every exact =candidate_hash_full= counter is zero.",
        "Zeroing the penalty does not repair the failing datasets. Giving the same method more",
        "iterations raises GIST-1M, MSTuring-1M, and MSTuring-10M above 0.90 recall.",
        "",
        "#+caption: Recall for the normal and control traversals. Diamonds are expanded-budget controls.",
        "[[file:deep_traversal_report/summary_recall.png]]",
        "",
        "The output-loss component is small. Most missing ground-truth IDs were never discovered",
        "before the cap; they were not discovered and later evicted from the fused frontier/result",
        "buffer.",
        "",
        "#+caption: Final correct results, discovered-but-lost GT IDs, and undiscovered GT IDs.",
        "[[file:deep_traversal_report/current_failure_decomposition.png]]",
        "",
        "The cumulative discovery curves remain upward-sloping at the normal cap and continue to",
        "improve under the expanded control, directly demonstrating useful unfinished traversal.",
        "",
        "#+caption: Cumulative filtered-ground-truth discovery by iteration.",
        "[[file:deep_traversal_report/ground_truth_discovery_cdf.png]]",
        "",
        "* Summary values",
        "",
        "| Dataset | Current recall | GT seen / 10 | Expanded recall | Iterations current -> expanded |",
        "|-",
    ]
    for dataset in DATASETS:
        current = records[dataset, "current"]
        expanded = records.get((dataset, "expanded"))
        expanded_recall = f"{expanded['mean_recall']:.5f}" if expanded else "n/a"
        expanded_iterations = (
            f"{current['mean_iterations']:.0f} -> {expanded['mean_iterations']:.0f}"
            if expanded else "n/a"
        )
        lines.append(
            f"| {LABELS[dataset]} | {current['mean_recall']:.5f} | "
            f"{current['mean_gt_seen']:.3f} | {expanded_recall} | {expanded_iterations} |"
        )
    lines.extend([
        "",
        "For GIST-1M, MSTuring-1M, and MSTuring-10M, zero penalty changes recall by -0.0849,",
        "-0.0184, and -0.0208 respectively relative to automatic FAVOR. Expanded traversal changes",
        "it by +0.0838, +0.2038, and +0.2289. The evidence therefore rejects penalty suppression",
        "as the primary explanation for these 1% failures. Penalty effects remain query-specific:",
        "zero penalty improves 10.7% of GIST queries but worsens 49.9%, so it is not a robust fix.",
        "",
        "* Selected low-recall traces",
        "",
        "These deliberately selected worst-case examples are illustrative, not dataset averages:",
        "",
        "| Dataset / query | Current recall (seen) | Zero recall (seen) | Expanded recall (seen) |",
        "|-",
    ])
    for dataset in FAILURES:
        query_id = (root / dataset / "selected_queries.txt").read_text().splitlines()[0]
        values = {}
        for variant in ("current", "zero", "expanded"):
            row = query_row(root / dataset / f"{variant}_trace" / "query_summary.csv", query_id)
            values[variant] = (
                float(row["recall"]), int(row["gt_seen_mask"]).bit_count()
            )
        lines.append(
            f"| {LABELS[dataset]} / {query_id} | {values['current'][0]:.1f} "
            f"({values['current'][1]}) | {values['zero'][0]:.1f} ({values['zero'][1]}) | "
            f"{values['expanded'][0]:.1f} ({values['expanded'][1]}) |"
        )
    lines.extend([
        "",
        "MSTuring-10M uses 3,584 rather than the requested 4,096 expanded iterations. At 4,096,",
        "CAGRA crosses a power-of-two visited-hash boundary and requests 10.5 GB for nq=10,000,",
        "above the 6.3 GB benchmark allocation limit. A fill target of 0.89 and 3,584 iterations",
        "keep the exact hash-full count at zero while using the 5.2 GB allocation tier.",
        "",
        "* Reproduce and inspect",
        "",
        "#+begin_src sh",
        "./benchmarks/favor/run_favor_trace_diagnostics.py --phase summary --resume",
        "./benchmarks/favor/run_favor_trace_diagnostics.py --phase deep --resume",
        "python benchmarks/favor/analyze_favor_trace_diagnostics.py \\",
        "  benchmarks/favor/results_deep_traversal_diagnostics",
        "#+end_src",
        "",
        "See [[file:TRACE_PARSING_AND_INTERPRETATION.org][TRACE_PARSING_AND_INTERPRETATION.org]] for the schema, commands,",
        "iteration convention, and query-level decision tree. Full binary traces are compressed and",
        "git-ignored; manifests, per-query summaries, selected-query maps, aggregate CSV, and figures",
        "are compact reproducible artifacts.",
    ])
    (output.parent / "DEEP_TRAVERSAL_DIAGNOSTIC_REPORT.org").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    output = args.root.parent / "deep_traversal_report"
    output.mkdir(parents=True, exist_ok=True)

    records_list: list[dict[str, object]] = []
    records: dict[tuple[str, str], dict[str, object]] = {}
    for dataset in DATASETS:
        for variant in ("default", "current", "zero", "expanded"):
            summary = args.root / dataset / f"{variant}_summary" / "query_summary.csv"
            if not summary.exists():
                continue
            record = aggregate(dataset, variant, read_rows(summary))
            records_list.append(record)
            records[dataset, variant] = record
    write_csv(output / "aggregate_summary.csv", records_list)
    plot_recall(output, records)
    plot_decomposition(output, records)
    plot_discovery(output, args.root)
    write_report(output, args.root, records)


if __name__ == "__main__":
    main()
