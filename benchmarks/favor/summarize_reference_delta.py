#!/usr/bin/env python3
"""Summarize and plot throughput sweeps using FAVOR-reference delta-d values."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATASETS = (
    ("BIGANN-1M", "bigann1m", ROOT / "results_reference_delta_qps_bigann_1m"),
    ("BIGANN-10M", "bigann10m", ROOT / "results_reference_delta_qps_bigann_10m"),
    ("MSTuring-1M", "msturing1m", ROOT / "results_reference_delta_qps_msturing_1m"),
    ("MSTuring-10M", "msturing10m", ROOT / "results_reference_delta_qps_msturing_10m"),
)
SELECTIVITIES = (1, 10, 50, 90)


def load(path: Path) -> dict[str, list[dict]]:
    rows = [
        row
        for row in json.loads(path.read_text())["benchmarks"]
        if row.get("run_type") == "iteration"
    ]
    return {
        mode: [
            row for row in rows if f'filter_mode="{mode}"' in row.get("label", "")
        ]
        for mode in ("default", "favor")
    }


def frontier(rows: list[dict]) -> list[dict]:
    result = []
    best_qps = -1.0
    for row in sorted(rows, key=lambda item: item["Recall"], reverse=True):
        if row["items_per_second"] > best_qps:
            result.append(row)
            best_qps = row["items_per_second"]
    return sorted(result, key=lambda item: item["Recall"])


def main() -> None:
    plot_dir = ROOT / "results_reference_delta_qps_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for title, prefix, result_dir in DATASETS:
        for selectivity in SELECTIVITIES:
            grouped = load(
                result_dir / "raw" / f"{prefix}_s{selectivity:02d}_nq10000.json"
            )
            values = {}
            for mode, rows in grouped.items():
                max_recall = max(row["Recall"] for row in rows)
                eligible = [row for row in rows if row["Recall"] >= 0.9]
                qps_at_90 = max(
                    (row["items_per_second"] for row in eligible), default=None
                )
                values[mode] = (max_recall, qps_at_90)
                summary.append(
                    {
                        "dataset": title,
                        "selectivity_percent": selectivity,
                        "mode": mode,
                        "max_recall": max_recall,
                        "best_qps_at_recall_ge_0.90": (
                            "" if qps_at_90 is None else qps_at_90
                        ),
                    }
                )

            fig, axis = plt.subplots(figsize=(7.5, 5.0))
            for mode, color, label in (
                ("default", "#1f77b4", "Default CAGRA filtering"),
                ("favor", "#d62728", "FAVOR with reference delta-d"),
            ):
                points = frontier(grouped[mode])
                axis.plot(
                    [point["Recall"] for point in points],
                    [point["items_per_second"] for point in points],
                    marker="o",
                    color=color,
                    label=label,
                )
            axis.axvline(0.9, color="black", linestyle="--", linewidth=1)
            axis.set_title(
                f"{title}: {selectivity}% selectivity\n"
                "10,000 queries; itopk 32–512; search width 1–4"
            )
            axis.set_xlabel("Recall@10")
            axis.set_ylabel("Throughput (queries/second)")
            axis.grid(True, alpha=0.3)
            axis.legend()
            fig.tight_layout()
            fig.savefig(
                plot_dir / f"{prefix}_selectivity_{selectivity:02d}.png", dpi=180
            )
            plt.close(fig)

    with (ROOT / "results_reference_delta_qps_summary.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
