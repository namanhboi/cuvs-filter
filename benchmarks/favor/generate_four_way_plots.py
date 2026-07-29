#!/usr/bin/env python3
"""Generate matched four-way SINGLE_CTA and MULTI_CTA FAVOR plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    key: str
    prefix: str
    title: str
    historical_single_dir: str


DATASETS = (
    Dataset("sift", "sift", "SIFT-1M", "results"),
    Dataset("gist", "gist", "GIST-1M", "results_gist"),
    Dataset("bigann1m", "bigann1m", "BIGANN-1M", "results_bigann_1m"),
    Dataset("bigann10m", "bigann10m", "BIGANN-10M", "results_bigann_10m"),
    Dataset("msturing1m", "msturing1m", "MSTuring-1M", "results_msturing_1m"),
    Dataset(
        "msturing10m",
        "msturing10m",
        "MSTuring-10M",
        "results_msturing_10m",
    ),
)


def run_plot(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="default: BENCHMARK_ROOT/results_four_way_comparison",
    )
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else root / "results_four_way_comparison"
    )
    plot_script = root / "plot_results.py"

    for dataset in DATASETS:
        common = [
            sys.executable,
            str(plot_script),
            "--result-prefix",
            dataset.prefix,
            "--plot-title",
            dataset.title,
            "--selectivities",
            "1",
            "10",
            "50",
            "90",
            "--penalty-lambdas",
            "1",
        ]
        run_plot(
            common
            + [
                "--result-dir",
                str(root / "results_retention_safe_single_full" / dataset.key),
                "--output-dir",
                str(output_root / "single" / dataset.key),
                "--cta-mode",
                "SINGLE_CTA",
                "--target-recall",
                "0.90",
                "--overlay-series",
                "current_static",
                "favor",
                "Current static FAVOR",
                str(root / "results_current_static_single_full" / dataset.key),
                "--overlay-series",
                "committed_static",
                "favor",
                "Committed static FAVOR (old report)",
                str(root / dataset.historical_single_dir),
            ]
        )
        run_plot(
            common
            + [
                "--result-dir",
                str(root / "results_multi_cta_packed_full" / dataset.key),
                "--output-dir",
                str(output_root / "multi" / dataset.key),
                "--cta-mode",
                "MULTI_CTA",
                "--target-recall",
                "0.99",
                "--latency-derived-qps",
                "--latency-batch-size",
                "1",
                "--latency-unit",
                "us",
                "--zero-y",
                "--overlay-series",
                "committed_static",
                "favor",
                "Committed static FAVOR (old report)",
                str(root / "results_multi_cta_adjusted_bs1" / dataset.key),
            ]
        )


if __name__ == "__main__":
    main()
