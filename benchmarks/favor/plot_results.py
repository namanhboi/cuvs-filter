#!/usr/bin/env python3
"""Plot clearly labelled default/FAVOR Pareto figures per selectivity."""

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write plots and the summary CSV here instead of modifying result-dir",
    )
    parser.add_argument("--result-prefix", default="sift")
    parser.add_argument("--plot-title", default="SIFT-1M")
    parser.add_argument(
        "--selectivities", type=int, nargs="+", default=(1, 10, 50, 90)
    )
    parser.add_argument("--latency-batch-size", type=int, default=10)
    parser.add_argument("--throughput-batch-size", type=int, default=10000)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--latency-only", action="store_true")
    layout.add_argument(
        "--qps-only",
        action="store_true",
        help="render only the QPS-versus-recall panel",
    )
    parser.add_argument(
        "--latency-derived-qps",
        action="store_true",
        help="plot items_per_second from the latency-mode file as batch query rate",
    )
    parser.add_argument("--latency-unit", choices=("ms", "us"), default="ms")
    parser.add_argument("--algo-label", default="")
    parser.add_argument(
        "--cta-mode",
        choices=("SINGLE_CTA", "MULTI_CTA"),
        help="use concise report titles and labels for this CAGRA search mode",
    )
    parser.add_argument(
        "--zero-y",
        action="store_true",
        help="start every rendered y-axis at zero",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.result_dir
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    colors = {"default": "#1f77b4", "favor": "#d62728"}
    algo_suffix = f" ({args.algo_label})" if args.algo_label else ""
    if args.cta_mode:
        labels = {
            "default": f"Default CAGRA{algo_suffix}",
            "favor": f"FAVOR{algo_suffix}",
        }
    else:
        labels = {
            "default": f"Default CAGRA filtering{algo_suffix}",
            "favor": f"FAVOR filtering{algo_suffix}",
        }
    latency_scale = 1_000_000.0 if args.latency_unit == "us" else 1_000.0
    latency_unit_label = "microseconds" if args.latency_unit == "us" else "milliseconds"
    for selectivity in args.selectivities:
        throughput = None
        if not args.latency_only:
            throughput_batch_size = (
                args.latency_batch_size
                if args.latency_derived_qps
                else args.throughput_batch_size
            )
            throughput = load_points(
                args.result_dir
                / "raw"
                / (
                    f"{args.result_prefix}_s{selectivity:02d}"
                    f"_nq{throughput_batch_size}.json"
                ),
                "items_per_second",
            )
        latency = load_points(
            args.result_dir
            / "raw"
            / (
                f"{args.result_prefix}_s{selectivity:02d}"
                f"_nq{args.latency_batch_size}.json"
            ),
            "Latency",
        )
        if args.latency_only:
            fig, axis = plt.subplots(1, 1, figsize=(7.4, 5.4))
            axes = [axis]
        elif args.qps_only:
            fig, axis = plt.subplots(1, 1, figsize=(8.5, 4.6))
            axes = [axis]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for mode in ("default", "favor"):
            latency_frontier = pareto(latency[mode], maximize=False)
            latency_axis = (
                axes[0] if args.latency_only else (None if args.qps_only else axes[1])
            )
            if latency_axis is not None:
                latency_axis.plot(
                    [p["recall"] for p in latency_frontier],
                    [latency_scale * p["value"] for p in latency_frontier],
                    marker="o",
                    color=colors[mode],
                    label=labels[mode],
                )
            if throughput is not None:
                qps_frontier = pareto(throughput[mode], maximize=True)
                axes[0].plot(
                    [p["recall"] for p in qps_frontier],
                    [p["value"] for p in qps_frontier],
                    marker="o",
                    color=colors[mode],
                    label=labels[mode],
                )
            workloads = [("latency", latency[mode])]
            if throughput is not None:
                workloads.insert(0, ("throughput", throughput[mode]))
            for workload, points in workloads:
                for point in points:
                    csv_rows.append(
                        {
                            "selectivity": selectivity / 100,
                            "workload": workload,
                            "batch_size": (
                                args.latency_batch_size
                                if workload == "latency"
                                else (
                                    args.latency_batch_size
                                    if args.latency_derived_qps
                                    else args.throughput_batch_size
                                )
                            ),
                            "mode": mode,
                            **point,
                        }
                    )

        latency_axis = (
            axes[0] if args.latency_only else (None if args.qps_only else axes[1])
        )
        if latency_axis is not None:
            latency_axis.set_title(
                f"Latency vs Recall@10 (batch size {args.latency_batch_size})"
                if args.cta_mode
                else f"Latency mode (batch size {args.latency_batch_size})"
            )
            latency_axis.set_xlabel("Recall@10")
            latency_axis.set_ylabel(
                f"Batch latency ({latency_unit_label}, lower is better)"
            )
        if not args.latency_only:
            if args.cta_mode:
                qps_batch_size = (
                    args.latency_batch_size
                    if args.latency_derived_qps
                    else args.throughput_batch_size
                )
                axes[0].set_title(
                    f"QPS vs Recall@10 (batch size {qps_batch_size:,})"
                )
                axes[0].set_ylabel("Throughput (QPS, higher is better)")
            elif args.latency_derived_qps:
                axes[0].set_title(
                    f"Latency mode query rate (batch size {args.latency_batch_size})"
                )
                axes[0].set_ylabel(
                    "Query rate (queries/second, higher is better)"
                )
            else:
                axes[0].set_title(
                    f"Throughput mode (batch size {args.throughput_batch_size:,})"
                )
                axes[0].set_ylabel(
                    "Throughput (queries/second, higher is better)"
                )
            axes[0].set_xlabel("Recall@10")
        for axis in axes:
            if args.zero_y:
                axis.set_ylim(bottom=0)
            axis.grid(True, alpha=0.3)
            axis.legend()
        if args.cta_mode:
            fig.suptitle(
                f"{args.plot_title} — {args.cta_mode}, {selectivity}% selectivity\n"
                "Graph degree 32, intermediate graph degree 64; "
                "one measured run per configuration"
            )
        else:
            fig.suptitle(
                f"{args.plot_title} filtered CAGRA: {selectivity}% selectivity\n"
                "Graph degree 32, intermediate graph degree 64; "
                "one measured run per configuration"
            )
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"{args.result_prefix}_selectivity_{selectivity:02d}.png", dpi=180
        )
        plt.close(fig)

    with (output_dir / "favor_benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "selectivity",
                "workload",
                "batch_size",
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
