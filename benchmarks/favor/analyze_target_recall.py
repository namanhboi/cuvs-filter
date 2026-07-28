#!/usr/bin/env python3
"""Analyze default/FAVOR QPS at a fixed recall from multiple result trees."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_results import pareto


RAW_NAME = re.compile(r"(?P<dataset>.+)_s(?P<selectivity>\d+)_nq1\.json$")
TITLES = {
    "sift": "SIFT-1M",
    "gist": "GIST-1M",
    "bigann1m": "BIGANN-1M",
    "bigann10m": "BIGANN-10M",
    "msturing1m": "MSTuring-1M",
    "msturing10m": "MSTuring-10M",
}


@dataclass(frozen=True)
class Estimate:
    status: str
    qps: float | None
    low: dict[str, float] | None
    high: dict[str, float] | None


def load_result_trees(roots: list[Path]) -> dict[tuple[str, int, str], list[dict[str, float]]]:
    grouped_rows: dict[
        tuple[str, int, str, int, int], list[tuple[float, float]]
    ] = defaultdict(list)
    for root in roots:
        for path in sorted(root.rglob("*_nq1.json")):
            match = RAW_NAME.fullmatch(path.name)
            if match is None or "/raw/" not in path.as_posix():
                continue
            dataset = match.group("dataset")
            selectivity = int(match.group("selectivity"))
            payload = json.loads(path.read_text())
            for row in payload.get("benchmarks", []):
                if row.get("run_type", "iteration") != "iteration":
                    continue
                label = row.get("label", "")
                mode = "favor" if 'filter_mode="favor"' in label else "default"
                key = (
                    dataset,
                    selectivity,
                    mode,
                    int(row["itopk"]),
                    int(row["search_width"]),
                )
                grouped_rows[key].append(
                    (float(row["Recall"]), float(row["items_per_second"]))
                )

    points: dict[tuple[str, int, str], list[dict[str, float]]] = defaultdict(list)
    for (dataset, selectivity, mode, itopk, width), values in grouped_rows.items():
        points[(dataset, selectivity, mode)].append(
            {
                "recall": float(np.median([value[0] for value in values])),
                "value": float(np.median([value[1] for value in values])),
                "itopk": itopk,
                "search_width": width,
            }
        )
    return points


def estimate_at_target(
    points: list[dict[str, float]],
    target: float,
    direct_overshoot: float,
    max_extrapolation_distance: float,
) -> Estimate:
    frontier = pareto(points, maximize=True)
    direct = [
        point
        for point in frontier
        if target <= point["recall"] <= target + direct_overshoot
    ]
    if direct:
        point = max(direct, key=lambda item: item["value"])
        return Estimate("measured", point["value"], point, point)

    below = [point for point in frontier if point["recall"] < target]
    above = [point for point in frontier if point["recall"] > target]
    if not below and len(above) >= 2:
        nearest, adjacent = sorted(above, key=lambda point: point["recall"])[:2]
        if nearest["recall"] - target <= max_extrapolation_distance:
            weight = (target - nearest["recall"]) / (
                adjacent["recall"] - nearest["recall"]
            )
            qps = nearest["value"] + weight * (
                adjacent["value"] - nearest["value"]
            )
            return Estimate("extrapolated", qps, nearest, adjacent)
    if not above and len(below) >= 2:
        nearest, adjacent = sorted(
            below, key=lambda point: point["recall"], reverse=True
        )[:2]
        if target - nearest["recall"] <= max_extrapolation_distance:
            weight = (target - adjacent["recall"]) / (
                nearest["recall"] - adjacent["recall"]
            )
            qps = adjacent["value"] + weight * (
                nearest["value"] - adjacent["value"]
            )
            return Estimate("extrapolated", qps, adjacent, nearest)
    if not below and above:
        minimum_itopk = min(point["itopk"] for point in points)
        if minimum_itopk == 10:
            point = max(
                (point for point in frontier if point["itopk"] == minimum_itopk),
                key=lambda item: item["value"],
            )
            return Estimate("nearest_feasible", point["value"], point, point)
    if not below or not above:
        return Estimate("unbracketed", None, max(below, default=None, key=lambda p: p["recall"]),
                        min(above, default=None, key=lambda p: p["recall"]))

    low = max(below, key=lambda point: point["recall"])
    high = min(above, key=lambda point: point["recall"])
    weight = (target - low["recall"]) / (high["recall"] - low["recall"])
    qps = low["value"] + weight * (high["value"] - low["value"])
    return Estimate("interpolated", qps, low, high)


def endpoint_value(estimate: Estimate, endpoint: str, field: str) -> str:
    point = estimate.low if endpoint == "low" else estimate.high
    return "" if point is None else str(point[field])


def analyze(
    roots: list[Path],
    output_dir: Path,
    target: float,
    direct_overshoot: float,
    max_extrapolation_distance: float,
) -> None:
    points = load_result_trees(roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "selectivity",
        "target_recall",
        "default_status",
        "default_qps",
        "default_low_recall",
        "default_low_qps",
        "default_low_itopk",
        "default_high_recall",
        "default_high_qps",
        "default_high_itopk",
        "favor_status",
        "favor_qps",
        "favor_low_recall",
        "favor_low_qps",
        "favor_low_itopk",
        "favor_high_recall",
        "favor_high_qps",
        "favor_high_itopk",
        "improvement_percent",
    ]
    output_rows = []
    colors = {"default": "#1f77b4", "favor": "#d62728"}

    for dataset in TITLES:
        for selectivity in (1, 10, 50, 90):
            estimates = {
                mode: estimate_at_target(
                    points.get((dataset, selectivity, mode), []),
                    target,
                    direct_overshoot,
                    max_extrapolation_distance,
                )
                for mode in ("default", "favor")
            }
            default_qps = estimates["default"].qps
            favor_qps = estimates["favor"].qps
            improvement = (
                None
                if default_qps is None or favor_qps is None
                else 100.0 * (favor_qps / default_qps - 1.0)
            )
            row: dict[str, str | float] = {
                "dataset": dataset,
                "selectivity": selectivity,
                "target_recall": target,
                "improvement_percent": "" if improvement is None else improvement,
            }
            for mode in ("default", "favor"):
                estimate = estimates[mode]
                row[f"{mode}_status"] = estimate.status
                row[f"{mode}_qps"] = "" if estimate.qps is None else estimate.qps
                for endpoint in ("low", "high"):
                    for field in ("recall", "value", "itopk"):
                        output_field = "qps" if field == "value" else field
                        row[f"{mode}_{endpoint}_{output_field}"] = endpoint_value(
                            estimate, endpoint, field
                        )
            output_rows.append(row)

            fig, axis = plt.subplots(figsize=(8.5, 4.6))
            for mode in ("default", "favor"):
                mode_points = points.get((dataset, selectivity, mode), [])
                frontier = pareto(mode_points, maximize=True)
                axis.scatter(
                    [point["recall"] for point in mode_points],
                    [point["value"] for point in mode_points],
                    color=colors[mode],
                    alpha=0.25,
                    s=25,
                )
                axis.plot(
                    [point["recall"] for point in frontier],
                    [point["value"] for point in frontier],
                    marker="o",
                    color=colors[mode],
                    label="Default CAGRA" if mode == "default" else "FAVOR",
                )
                estimate = estimates[mode]
                if estimate.qps is not None:
                    axis.scatter(
                        [target],
                        [estimate.qps],
                        color=colors[mode],
                        marker="X",
                        s=100,
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=5,
                    )
            axis.axvline(target, color="#555555", linestyle="--", linewidth=1)
            axis.set_ylim(bottom=0)
            axis.set_xlabel("Recall@10")
            axis.set_ylabel("Throughput (QPS, higher is better)")
            axis.set_title(
                f"{TITLES[dataset]} — MULTI_CTA, {selectivity}% selectivity\n"
                f"QPS at target recall {target:.2f} is measured, interpolated, or extrapolated"
            )
            axis.grid(True, alpha=0.3)
            axis.legend()
            fig.tight_layout()
            fig.savefig(
                plot_dir / f"{dataset}_selectivity_{selectivity:02d}.png", dpi=180
            )
            plt.close(fig)

    with (output_dir / "qps_at_target_recall.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def self_test() -> None:
    points = [
        {"recall": 0.98, "value": 2000.0, "itopk": 32, "search_width": 1},
        {"recall": 1.00, "value": 1000.0, "itopk": 64, "search_width": 1},
    ]
    estimate = estimate_at_target(points, 0.99, 0.001, 0.005)
    assert estimate.status == "interpolated"
    assert estimate.qps is not None and math.isclose(estimate.qps, 1500.0)
    measured = estimate_at_target(
        [{"recall": 0.9905, "value": 1700.0, "itopk": 48, "search_width": 1}],
        0.99,
        0.001,
        0.005,
    )
    assert measured.status == "measured"
    assert measured.qps == 1700.0
    unbracketed = estimate_at_target(points[:1], 0.99, 0.001, 0.005)
    assert unbracketed.status == "unbracketed"
    nearest = estimate_at_target(
        [{"recall": 0.992, "value": 1600.0, "itopk": 10, "search_width": 1}],
        0.99,
        0.001,
        0.005,
    )
    assert nearest.status == "nearest_feasible"
    assert nearest.qps == 1600.0
    extrapolated = estimate_at_target(
        [
            {"recall": 0.97, "value": 1800.0, "itopk": 32, "search_width": 1},
            {"recall": 0.988, "value": 1400.0, "itopk": 64, "search_width": 1},
        ],
        0.99,
        0.001,
        0.005,
    )
    assert extrapolated.status == "extrapolated"
    assert extrapolated.qps is not None
    assert math.isclose(extrapolated.qps, 1355.5555555555557)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--direct-overshoot", type=float, default=0.001)
    parser.add_argument("--max-extrapolation-distance", type=float, default=0.005)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.result_root or args.output_dir is None:
        parser.error("--result-root and --output-dir are required unless --self-test is used")
    analyze(
        args.result_root,
        args.output_dir,
        args.target_recall,
        args.direct_overshoot,
        args.max_extrapolation_distance,
    )


if __name__ == "__main__":
    main()
