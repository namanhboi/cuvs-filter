#!/usr/bin/env python3
"""Plot clearly labelled default/FAVOR Pareto figures per selectivity."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class OverlaySeries:
    key: str
    mode: str
    label: str
    result_dir: Path


def load_points(
    path: Path, metric: str, selected_lambdas: set[float] | None = None
) -> dict[str, list[dict[str, float]]]:
    payload = json.loads(path.read_text())
    grouped: dict[tuple[str, float | None, int, int, int, int], list[dict]] = defaultdict(list)
    for row in payload["benchmarks"]:
        if row.get("run_type", "iteration") != "iteration":
            continue
        label = row.get("label", "")
        if 'favor_penalty_mode="cagra_retention_safe"' in label:
            mode = "favor_retention_safe"
        elif 'favor_penalty_mode="cagra_query_local"' in label:
            mode = "favor_query_local"
        elif 'filter_mode="favor"' in label:
            mode = "favor"
        else:
            mode = "default"
        penalty_lambda = row.get("favor_penalty_lambda")
        if penalty_lambda is None:
            match = re.search(r"favor_penalty_lambda=([0-9.eE+-]+)", label)
            if match:
                penalty_lambda = float(match.group(1))
        if penalty_lambda is not None:
            penalty_lambda = float(penalty_lambda)
            if (
                selected_lambdas is not None
                and penalty_lambda not in selected_lambdas
            ):
                continue
        key = (
            mode,
            penalty_lambda,
            int(row["itopk"]),
            int(row["search_width"]),
            int(row.get("max_iterations", 0)),
            int(row.get("thread_block_size", 0)),
        )
        grouped[key].append(row)

    result: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (
        mode,
        penalty_lambda,
        itopk,
        width,
        max_iterations,
        thread_block_size,
    ), repetitions in grouped.items():
        recall = float(np.median([r["Recall"] for r in repetitions]))
        value = float(np.median([r[metric] for r in repetitions]))
        series = (
            mode
            if penalty_lambda is None
            else f"{mode}:lambda={penalty_lambda:g}"
        )
        result[series].append(
            {
                "recall": recall,
                "value": value,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": max_iterations,
                "thread_block_size": thread_block_size,
                "penalty_lambda": penalty_lambda,
            }
        )
    return result


def load_overlay_points(
    path: Path,
    metric: str,
    overlay: OverlaySeries,
    selected_lambdas: set[float] | None = None,
) -> list[dict[str, float]]:
    """Load exactly one requested mode without merging it with the primary run."""
    points = load_points(path, metric, selected_lambdas)
    matches = [
        series
        for series in points
        if series == overlay.mode or series.startswith(f"{overlay.mode}:")
    ]
    if not matches:
        raise ValueError(
            f"{path} does not contain requested overlay mode {overlay.mode!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"{path} contains multiple series for overlay mode {overlay.mode!r}: "
            f"{matches}; select a single mode/lambda"
        )
    return points[matches[0]]


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


def interpolate_at_recall(
    frontier: list[dict[str, float]], target_recall: float, maximize: bool
) -> dict[str, float | str] | None:
    """Estimate a target only from a bracket or a measured target-feasible point."""
    if not frontier:
        return None
    ordered = sorted(frontier, key=lambda point: point["recall"])
    exact = next(
        (point for point in ordered if point["recall"] == target_recall), None
    )
    if exact is not None:
        return {
            "target_recall": target_recall,
            "value": exact["value"],
            "lower_recall": exact["recall"],
            "upper_recall": exact["recall"],
            "point_recall": exact["recall"],
            "target_method": "exact",
        }

    lower = next(
        (point for point in reversed(ordered) if point["recall"] < target_recall),
        None,
    )
    upper = next(
        (point for point in ordered if point["recall"] > target_recall), None
    )
    if upper is None:
        return None
    if lower is None:
        feasible = [point for point in ordered if point["recall"] >= target_recall]
        chosen = (max if maximize else min)(
            feasible, key=lambda point: point["value"]
        )
        return {
            "target_recall": target_recall,
            "value": chosen["value"],
            "lower_recall": chosen["recall"],
            "upper_recall": chosen["recall"],
            "point_recall": chosen["recall"],
            "target_method": "measured_feasible",
        }

    fraction = (target_recall - lower["recall"]) / (
        upper["recall"] - lower["recall"]
    )
    return {
        "target_recall": target_recall,
        "value": lower["value"] + fraction * (upper["value"] - lower["value"]),
        "lower_recall": lower["recall"],
        "upper_recall": upper["recall"],
        "point_recall": target_recall,
        "target_method": "interpolated",
    }


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
        "--penalty-lambdas",
        type=float,
        nargs="+",
        help="plot only these query-local/retention-safe lambda values",
    )
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
    parser.add_argument(
        "--target-recall",
        type=float,
        help=(
            "mark and summarize a bracketed interpolation or measured feasible "
            "Pareto point at this recall; no extrapolation is performed"
        ),
    )
    parser.add_argument(
        "--overlay-series",
        action="append",
        nargs=4,
        metavar=("KEY", "MODE", "LABEL", "RESULT_DIR"),
        default=[],
        help=(
            "add a separately namespaced series from another result directory; "
            "repeat for multiple overlays"
        ),
    )
    args = parser.parse_args()
    overlays = [
        OverlaySeries(key, mode, label, Path(result_dir))
        for key, mode, label, result_dir in args.overlay_series
    ]
    overlay_keys = [overlay.key for overlay in overlays]
    if len(overlay_keys) != len(set(overlay_keys)):
        raise ValueError("overlay-series keys must be unique")
    reserved_keys = {
        "default",
        "favor",
        "favor_query_local",
        "favor_retention_safe",
    }
    if reserved_keys.intersection(overlay_keys):
        raise ValueError(
            "overlay-series keys must not reuse primary algorithm mode names"
        )
    output_dir = args.output_dir or args.result_dir
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    target_rows = []
    colors = {
        "default": "#1f77b4",
        "favor": "#d62728",
        "favor_query_local": "#2ca02c",
        "favor_retention_safe": "#9467bd",
        "current_static": "#d62728",
        "committed_static": "#ff7f0e",
    }
    line_styles = defaultdict(lambda: "-")
    markers = defaultdict(lambda: "o")
    marker_faces = defaultdict(lambda: None)
    line_styles["committed_static"] = "--"
    markers["committed_static"] = "^"
    marker_faces["committed_static"] = "none"
    algo_suffix = f" ({args.algo_label})" if args.algo_label else ""
    if args.cta_mode:
        labels = {
            "default": f"Default CAGRA{algo_suffix}",
            "favor": f"Current static FAVOR{algo_suffix}",
            "favor_query_local": f"Query-local FAVOR{algo_suffix}",
            "favor_retention_safe": f"Retention-safe FAVOR{algo_suffix}",
        }
    else:
        labels = {
            "default": f"Default CAGRA filtering{algo_suffix}",
            "favor": f"Current static FAVOR filtering{algo_suffix}",
            "favor_query_local": f"Query-local FAVOR filtering{algo_suffix}",
            "favor_retention_safe": f"Retention-safe FAVOR filtering{algo_suffix}",
        }
    overlay_by_key = {overlay.key: overlay for overlay in overlays}
    for overlay in overlays:
        labels[overlay.key] = overlay.label
        colors.setdefault(overlay.key, "#7f7f7f")
    latency_scale = 1_000_000.0 if args.latency_unit == "us" else 1_000.0
    latency_unit_label = "microseconds" if args.latency_unit == "us" else "milliseconds"
    selected_lambdas = (
        set(args.penalty_lambdas) if args.penalty_lambdas is not None else None
    )
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
                selected_lambdas,
            )
            for overlay in overlays:
                overlay_path = (
                    overlay.result_dir
                    / "raw"
                    / (
                        f"{args.result_prefix}_s{selectivity:02d}"
                        f"_nq{throughput_batch_size}.json"
                    )
                )
                if not overlay_path.is_file():
                    raise FileNotFoundError(
                        f"missing overlay throughput result: {overlay_path}"
                    )
                throughput[overlay.key] = load_overlay_points(
                    overlay_path,
                    "items_per_second",
                    overlay,
                    selected_lambdas,
                )
        latency = load_points(
            args.result_dir
            / "raw"
            / (
                f"{args.result_prefix}_s{selectivity:02d}"
                f"_nq{args.latency_batch_size}.json"
            ),
            "Latency",
            selected_lambdas,
        )
        for overlay in overlays:
            overlay_path = (
                overlay.result_dir
                / "raw"
                / (
                    f"{args.result_prefix}_s{selectivity:02d}"
                    f"_nq{args.latency_batch_size}.json"
                )
            )
            if not overlay_path.is_file():
                raise FileNotFoundError(
                    f"missing overlay latency result: {overlay_path}"
                )
            latency[overlay.key] = load_overlay_points(
                overlay_path,
                "Latency",
                overlay,
                selected_lambdas,
            )
        if args.latency_only:
            fig, axis = plt.subplots(1, 1, figsize=(7.4, 5.4))
            axes = [axis]
        elif args.qps_only:
            fig, axis = plt.subplots(1, 1, figsize=(8.5, 4.6))
            axes = [axis]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        mode_order = {
            "default": 0,
            "favor": 1,
            "favor_query_local": 2,
            "favor_retention_safe": 3,
            "current_static": 1,
            "committed_static": 4,
        }
        available_series = sorted(
            set(latency) | (set(throughput) if throughput is not None else set()),
            key=lambda series: (
                mode_order.get(series.split(":", 1)[0], 10),
                (
                    float(series.rsplit("=", 1)[1])
                    if ":lambda=" in series
                    else -1.0
                ),
                series,
            ),
        )
        for series in available_series:
            series_key = series.split(":", 1)[0]
            overlay = overlay_by_key.get(series_key)
            mode = overlay.mode if overlay is not None else series_key
            source_run = "current" if overlay is None else series_key
            penalty_lambda = (
                float(series.rsplit("=", 1)[1]) if ":lambda=" in series else None
            )
            series_label = labels[series_key]
            if penalty_lambda is not None:
                series_label += f" (lambda={penalty_lambda:g})"
            series_color = colors[series_key]
            series_line_style = line_styles[series_key]
            series_marker = markers[series_key]
            latency_frontier = pareto(latency[series], maximize=False)
            latency_axis = (
                axes[0] if args.latency_only else (None if args.qps_only else axes[1])
            )
            if latency_axis is not None:
                latency_axis.plot(
                    [p["recall"] for p in latency_frontier],
                    [latency_scale * p["value"] for p in latency_frontier],
                    marker=series_marker,
                    markerfacecolor=marker_faces[series_key],
                    linestyle=series_line_style,
                    color=series_color,
                    label=series_label,
                )
                if args.target_recall is not None:
                    target = interpolate_at_recall(
                        latency_frontier, args.target_recall, maximize=False
                    )
                    if target is not None:
                        latency_axis.scatter(
                            [target["point_recall"]],
                            [latency_scale * target["value"]],
                            color=series_color,
                            marker="X",
                            s=70,
                            zorder=4,
                        )
                        target_rows.append(
                            {
                                "selectivity": selectivity / 100,
                                "workload": "latency",
                                "batch_size": args.latency_batch_size,
                                "series": series_key,
                                "mode": mode,
                                "source_run": source_run,
                                "penalty_lambda": penalty_lambda,
                                **target,
                            }
                        )
            if throughput is not None:
                qps_frontier = pareto(throughput[series], maximize=True)
                axes[0].plot(
                    [p["recall"] for p in qps_frontier],
                    [p["value"] for p in qps_frontier],
                    marker=series_marker,
                    markerfacecolor=marker_faces[series_key],
                    linestyle=series_line_style,
                    color=series_color,
                    label=series_label,
                )
                if args.target_recall is not None:
                    target = interpolate_at_recall(
                        qps_frontier, args.target_recall, maximize=True
                    )
                    if target is not None:
                        axes[0].scatter(
                            [target["point_recall"]],
                            [target["value"]],
                            color=series_color,
                            marker="X",
                            s=70,
                            zorder=4,
                        )
                        target_rows.append(
                            {
                                "selectivity": selectivity / 100,
                                "workload": "throughput",
                                "batch_size": (
                                    args.latency_batch_size
                                    if args.latency_derived_qps
                                    else args.throughput_batch_size
                                ),
                                "series": series_key,
                                "mode": mode,
                                "source_run": source_run,
                                "penalty_lambda": penalty_lambda,
                                **target,
                            }
                        )
            workloads = [("latency", latency[series])]
            if throughput is not None:
                workloads.insert(0, ("throughput", throughput[series]))
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
                            "series": series_key,
                            "mode": mode,
                            "source_run": source_run,
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
            if args.target_recall is not None:
                axis.axvline(
                    args.target_recall,
                    color="#555555",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )
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
                "series",
                "mode",
                "source_run",
                "penalty_lambda",
                "recall",
                "value",
                "itopk",
                "search_width",
                "max_iterations",
                "thread_block_size",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    if args.target_recall is not None:
        with (output_dir / "target_recall_summary.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "selectivity",
                    "workload",
                    "batch_size",
                    "series",
                    "mode",
                    "source_run",
                    "penalty_lambda",
                    "target_recall",
                    "value",
                    "lower_recall",
                    "upper_recall",
                    "point_recall",
                    "target_method",
                ],
            )
            writer.writeheader()
            writer.writerows(target_rows)


if __name__ == "__main__":
    main()
