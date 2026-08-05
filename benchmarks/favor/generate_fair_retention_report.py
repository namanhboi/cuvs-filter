#!/usr/bin/env python3
"""Validate and render the fair SINGLE/MULTI CTA retention plot report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from generate_automatic_retention_report import (
    DATASETS as ALL_DATASETS,
)
from generate_automatic_retention_report import (
    METHOD_ORDER,
    SELECTIVITIES,
    validate_evaluation_sources,
)

DATASETS = tuple(
    dataset
    for dataset in ALL_DATASETS
    if dataset.key in {"gist", "bigann1m", "msturing1m"}
)


@dataclass(frozen=True)
class ModeSpec:
    cta_mode: str
    raw_root: Path
    plot_root: Path
    latency_batch_size: int
    throughput_batch_size: int


def raw_method(row: dict) -> str:
    """Classify one raw benchmark row without relying on plot_results.py."""
    label = row.get("label", "")
    if 'filter_mode="default"' in label:
        if "favor_penalty_mode=" in label:
            raise ValueError(
                f"default row unexpectedly has a FAVOR penalty: {label}"
            )
        return "default"
    if 'filter_mode="favor"' not in label:
        raise ValueError(
            f"benchmark row has no recognized filter mode: {label}"
        )
    if 'favor_penalty_mode="cagra_retention_safe"' not in label:
        raise ValueError(
            f"benchmark row has the wrong FAVOR penalty mode: {label}"
        )
    if float(row.get("favor_penalty_lambda", math.nan)) != 1.0:
        raise ValueError(f"benchmark row has the wrong FAVOR lambda: {row}")
    retention = float(row.get("favor_retention_fraction", math.nan))
    if retention == 0.0:
        return "automatic_retention"
    if retention == 0.5:
        return "favor_retention_safe"
    raise ValueError(
        f"benchmark row has an unsupported retention fraction: {retention}"
    )


def point_key(
    selectivity: float,
    workload: str,
    batch_size: int,
    method: str,
    row: dict,
) -> tuple[float, str, int, str, int, int, int, int]:
    return (
        selectivity,
        workload,
        batch_size,
        method,
        int(float(row["itopk"])),
        int(float(row["search_width"])),
        int(float(row.get("max_iterations", 0))),
        int(float(row.get("thread_block_size", 0))),
    )


def load_expected_points(
    spec: ModeSpec, dataset_key: str, dataset_prefix: str
) -> dict[tuple, tuple[float, float]]:
    """Load the raw recall/metric values expected in one plotted summary."""
    expected: dict[tuple, tuple[float, float]] = {}
    workload_sources = (
        ("throughput", spec.throughput_batch_size, "items_per_second"),
        ("latency", spec.latency_batch_size, "Latency"),
    )
    for selectivity in SELECTIVITIES:
        for workload, batch_size, metric in workload_sources:
            path = (
                spec.raw_root
                / dataset_key
                / "raw"
                / f"{dataset_prefix}_s{selectivity:02d}_nq{batch_size}.json"
            )
            payload = json.loads(path.read_text())
            rows = [
                row
                for row in payload["benchmarks"]
                if row.get("run_type", "iteration") == "iteration"
            ]
            methods = {raw_method(row) for row in rows}
            if methods != set(METHOD_ORDER):
                raise ValueError(
                    f"{path} has methods {sorted(methods)}; expected {list(METHOD_ORDER)}"
                )
            for row in rows:
                recall = float(row["Recall"])
                value = float(row[metric])
                if (
                    not 0.0 <= recall <= 1.0
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"invalid plotted value in {path}: {row}")
                key = point_key(
                    selectivity / 100,
                    workload,
                    batch_size,
                    raw_method(row),
                    row,
                )
                if key in expected:
                    raise ValueError(
                        f"duplicate raw plot point {key} in {path}"
                    )
                expected[key] = (recall, value)
    return expected


def validate_plot_summary(
    spec: ModeSpec, dataset_key: str, dataset_prefix: str
) -> int:
    """Verify every plotted CSV value against the corresponding raw benchmark row."""
    summary_path = spec.plot_root / dataset_key / "favor_benchmark_summary.csv"
    with summary_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = load_expected_points(spec, dataset_key, dataset_prefix)
    actual: dict[tuple, tuple[float, float]] = {}
    for row in rows:
        method = row["series"]
        if method not in METHOD_ORDER:
            raise ValueError(
                f"historical or unknown series {method!r} in {summary_path}"
            )
        if row["mode"] != method or row["source_run"] != "current":
            raise ValueError(
                f"non-current series metadata in {summary_path}: {row}"
            )
        if method == "default":
            if row["penalty_lambda"] or row["retention_fraction"]:
                raise ValueError(
                    f"default point has FAVOR parameters in {summary_path}: {row}"
                )
        else:
            expected_retention = (
                0.0 if method == "automatic_retention" else 0.5
            )
            if (
                float(row["penalty_lambda"]) != 1.0
                or float(row["retention_fraction"]) != expected_retention
            ):
                raise ValueError(
                    f"wrong FAVOR parameters in {summary_path}: {row}"
                )
        key = point_key(
            float(row["selectivity"]),
            row["workload"],
            int(row["batch_size"]),
            method,
            row,
        )
        if key in actual:
            raise ValueError(
                f"duplicate plotted point {key} in {summary_path}"
            )
        actual[key] = (float(row["recall"]), float(row["value"]))
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(
            f"plot/raw point mismatch in {summary_path}: missing={missing}, extra={extra}"
        )
    mismatches = [
        (key, actual[key], expected[key])
        for key in actual
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            f"plot/raw value mismatch in {summary_path}: {mismatches}"
        )
    return len(rows)


def validate_filtering_rates(
    spec: ModeSpec, dataset_key: str, dataset_prefix: str
) -> None:
    """Require one explicit, exact rejection rate for every plotted method."""
    batches = {spec.latency_batch_size, spec.throughput_batch_size}
    for selectivity in SELECTIVITIES:
        expected_rate = round(1.0 - selectivity / 100.0, 2)
        for batch_size in batches:
            path = (
                spec.raw_root
                / dataset_key
                / "configs"
                / f"{dataset_prefix}_s{selectivity:02d}_nq{batch_size}.json"
            )
            rows = json.loads(path.read_text())["index"][0]["search_params"]
            for row in rows:
                if "filtering_rate" not in row:
                    raise ValueError(
                        f"missing filtering_rate in {path}: {row}"
                    )
                if float(row["filtering_rate"]) != expected_rate:
                    raise ValueError(
                        f"wrong filtering_rate in {path}: expected {expected_rate}, found {row}"
                    )


def validate_mode(
    spec: ModeSpec, expected_gpu: str
) -> tuple[int, list[datetime]]:
    """Validate source grids, raw-to-plot values, images, and provenance for one mode."""
    validate_evaluation_sources(
        spec.raw_root,
        automatic_overlay_root=None,
        cta_mode=spec.cta_mode,
        batch_size=spec.latency_batch_size,
        config_batch_size=spec.throughput_batch_size,
        expected_gpu=expected_gpu,
        datasets=DATASETS,
    )
    point_count = 0
    dates: list[datetime] = []
    batches = {spec.latency_batch_size, spec.throughput_batch_size}
    for dataset in DATASETS:
        validate_filtering_rates(spec, dataset.key, dataset.prefix)
        point_count += validate_plot_summary(spec, dataset.key, dataset.prefix)
        for selectivity in SELECTIVITIES:
            image_path = (
                spec.plot_root
                / dataset.key
                / "plots"
                / f"{dataset.prefix}_selectivity_{selectivity:02d}.png"
            )
            data = image_path.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"invalid PNG plot: {image_path}")
            for batch_size in batches:
                raw_path = (
                    spec.raw_root
                    / dataset.key
                    / "raw"
                    / f"{dataset.prefix}_s{selectivity:02d}_nq{batch_size}.json"
                )
                context = json.loads(raw_path.read_text())["context"]
                dates.append(datetime.fromisoformat(context["date"]))
    return point_count, dates


def build_report(
    single_plot_root: Path,
    multi_plot_root: Path,
    *,
    gpu_name: str,
    result_date: str,
) -> str:
    """Build a plot-only report with no archived result sections or tables."""
    lines = [
        "#+TITLE: Fair FAVOR Retention Results: SINGLE_CTA and MULTI_CTA",
        f"#+DATE: {result_date}",
        "#+OPTIONS: toc:2 num:nil",
        "",
        (
            f"All figures are fair search-only measurements on {gpu_name}. They compare "
            "Default CAGRA, fixed retention (lambda=1, rho=0.5), and automatic retention "
            "(lambda=1, rho=auto)."
        ),
        "",
    ]
    for heading, plot_root in (
        ("SINGLE_CTA", single_plot_root),
        ("MULTI_CTA", multi_plot_root),
    ):
        lines.extend((f"* {heading}", ""))
        for dataset in DATASETS:
            lines.extend((f"** {dataset.title}", ""))
            for selectivity in SELECTIVITIES:
                image_path = (
                    plot_root
                    / dataset.key
                    / "plots"
                    / f"{dataset.prefix}_selectivity_{selectivity:02d}.png"
                )
                lines.extend(
                    (
                        f"#+CAPTION: {heading}, {dataset.title}, {selectivity}% selectivity",
                        "#+ATTR_ORG: :width 1100",
                        f"[[file:{image_path.as_posix()}]]",
                        "",
                    )
                )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-raw-root",
        type=Path,
        default=root / "results_fair_retention_single_full",
    )
    parser.add_argument(
        "--single-plot-root",
        type=Path,
        default=root / "results_fair_retention_single_comparison",
    )
    parser.add_argument(
        "--multi-raw-root",
        type=Path,
        default=root / "results_fair_retention_multi_full",
    )
    parser.add_argument(
        "--multi-plot-root",
        type=Path,
        default=root / "results_fair_retention_multi_comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "FAIR_RETENTION_FAVOR_SINGLE_MULTI_CTA_REPORT.org",
    )
    parser.add_argument("--expected-gpu", default="NVIDIA L4")
    args = parser.parse_args()

    single = ModeSpec(
        "SINGLE_CTA",
        args.single_raw_root.resolve(),
        args.single_plot_root.resolve(),
        10,
        10000,
    )
    multi = ModeSpec(
        "MULTI_CTA",
        args.multi_raw_root.resolve(),
        args.multi_plot_root.resolve(),
        1,
        1,
    )
    single_points, single_dates = validate_mode(single, args.expected_gpu)
    multi_points, multi_dates = validate_mode(multi, args.expected_gpu)

    output = args.output.resolve()
    report = build_report(
        Path(single.plot_root).relative_to(output.parent),
        Path(multi.plot_root).relative_to(output.parent),
        gpu_name=args.expected_gpu,
        result_date=max(single_dates + multi_dates).date().isoformat(),
    )
    output.write_text(report)
    print(
        f"wrote {output} with 24 fair plots "
        f"({single_points} SINGLE_CTA and {multi_points} MULTI_CTA source points)"
    )


if __name__ == "__main__":
    main()
