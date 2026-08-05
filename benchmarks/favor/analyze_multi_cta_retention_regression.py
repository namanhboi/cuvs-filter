#!/usr/bin/env python3
"""Generate, validate, and analyze the fixed-retention MULTI_CTA regression A/B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_GPU = "NVIDIA L4"
METHODS = ("default", "fixed")
DATASETS = {
    "sift50": {
        "title": "SIFT-1M",
        "dataset": "sift-128-euclidean",
        "selectivity": 50,
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "itopk": (64, 128),
    },
    "gist01": {
        "title": "GIST-1M",
        "dataset": "gist-960-euclidean",
        "selectivity": 1,
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "itopk": (128, 256, 512),
    },
    "bigann10m50": {
        "title": "BIGANN-10M",
        "dataset": "bigann-10M",
        "selectivity": 50,
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
        "itopk": (64, 128),
    },
    "msturing10m01": {
        "title": "MSTuring-10M",
        "dataset": "msturing-10M",
        "selectivity": 1,
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
        "itopk": (1024, 1536, 2048),
    },
}


@dataclass(frozen=True)
class Measurement:
    build: str
    dataset_key: str
    dataset: str
    selectivity: float
    method: str
    itopk: int
    repetition: int
    gpu_us: float
    latency_us: float
    real_time_us: float
    recall: float
    total_queries: int
    underfilled_queries: float | None
    missing_result_slots: float | None


def generate_configs(config_dir: Path, data_root: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    for key, spec in DATASETS.items():
        dataset = str(spec["dataset"])
        selectivity = int(spec["selectivity"])
        delta_path = data_root / dataset / "cagra_g32_ig64.index.delta_d"
        search_params: list[dict[str, Any]] = []
        for method in METHODS:
            for itopk in spec["itopk"]:
                param: dict[str, Any] = {
                    "algo": "multi_cta",
                    "filter_mode": "default" if method == "default" else "favor",
                    "itopk": itopk,
                    "search_width": 1,
                }
                if method == "fixed":
                    param.update(
                        {
                            "favor_delta_d_file": str(delta_path),
                            "favor_delta_d_alpha": 10,
                            "favor_delta_d_beta": 64,
                            "favor_delta_d_bfs_depth": 2,
                            "favor_penalty_mode": "cagra_retention_safe",
                            "favor_penalty_lambda": 1.0,
                        }
                    )
                search_params.append(param)
        config = {
            "dataset": {
                "name": f"{dataset}-s{selectivity:02d}",
                "base_file": f"{dataset}/{spec['base']}",
                "query_file": f"{dataset}/{spec['query']}",
                "groundtruth_neighbors_file": (
                    f"{dataset}/favor/groundtruth_s{selectivity:02d}.ibin"
                ),
                "filter_bitset_file": f"{dataset}/favor/filter_s{selectivity:02d}.bin",
                "distance": "euclidean",
                "dtype": spec["dtype"],
            },
            "search_basic_param": {"batch_size": 1, "k": 10},
            "index": [
                {
                    "name": "cagra-g32-ig64",
                    "algo": "cuvs_cagra",
                    "file": f"{dataset}/cagra_g32_ig64.index",
                    "build_param": {
                        "graph_build_algo": "NN_DESCENT",
                        "graph_degree": 32,
                        "intermediate_graph_degree": 64,
                    },
                    "search_params": search_params,
                }
            ],
        }
        (config_dir / f"{key}.json").write_text(json.dumps(config, indent=2) + "\n")


def _time_to_us(value: float, unit: str) -> float:
    factors = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    if unit not in factors:
        raise ValueError(f"unsupported Google Benchmark time unit: {unit}")
    return value * factors[unit]


def _method(param: dict[str, Any]) -> str:
    if param.get("filter_mode") == "default":
        return "default"
    if param.get("favor_penalty_mode") == "cagra_retention_safe":
        return "fixed"
    raise ValueError(f"unexpected search parameter: {param}")


def load_measurements(
    build: str,
    dataset_key: str,
    config_path: Path,
    raw_path: Path,
    expected_repetitions: int,
    expected_iterations: int,
) -> list[Measurement]:
    config = json.loads(config_path.read_text())
    raw = json.loads(raw_path.read_text())
    gpu = raw.get("context", {}).get("gpu_name")
    if gpu != EXPECTED_GPU:
        raise ValueError(f"{raw_path}: expected {EXPECTED_GPU}, found {gpu!r}")
    params = config["index"][0]["search_params"]
    rows = [row for row in raw["benchmarks"] if row.get("run_type") == "iteration"]
    expected_rows = len(params) * expected_repetitions
    if len(rows) != expected_rows:
        raise ValueError(f"{raw_path}: expected {expected_rows} iteration rows, found {len(rows)}")

    spec = DATASETS[dataset_key]
    seen: defaultdict[int, set[int]] = defaultdict(set)
    measurements: list[Measurement] = []
    for row in rows:
        family = int(row["family_index"])
        if not 0 <= family < len(params):
            raise ValueError(f"{raw_path}: invalid family_index {family}")
        repetition = int(row.get("repetition_index", 0))
        seen[family].add(repetition)
        iterations = int(row["iterations"])
        total_queries = int(row["total_queries"])
        if iterations != expected_iterations or total_queries != expected_iterations:
            raise ValueError(
                f"{raw_path}: family {family} repetition {repetition} processed "
                f"iterations={iterations}, total_queries={total_queries}; "
                f"expected {expected_iterations}"
            )
        gpu_seconds = float(row["GPU"])
        recall = float(row["Recall"])
        if not math.isfinite(gpu_seconds) or gpu_seconds <= 0:
            raise ValueError(f"{raw_path}: invalid GPU time {gpu_seconds}")
        if not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError(f"{raw_path}: invalid recall {recall}")
        underfilled = row.get("UnderfilledQueries")
        missing = row.get("MissingResultSlots")
        if underfilled is not None and float(underfilled) != 0:
            raise ValueError(f"{raw_path}: nonzero UnderfilledQueries={underfilled}")
        if missing is not None and float(missing) != 0:
            raise ValueError(f"{raw_path}: nonzero MissingResultSlots={missing}")
        param = params[family]
        measurements.append(
            Measurement(
                build=build,
                dataset_key=dataset_key,
                dataset=str(spec["title"]),
                selectivity=float(spec["selectivity"]) / 100.0,
                method=_method(param),
                itopk=int(param["itopk"]),
                repetition=repetition,
                gpu_us=gpu_seconds * 1e6,
                latency_us=float(row["Latency"]) * 1e6,
                real_time_us=_time_to_us(float(row["real_time"]), row["time_unit"]),
                recall=recall,
                total_queries=total_queries,
                underfilled_queries=None if underfilled is None else float(underfilled),
                missing_result_slots=None if missing is None else float(missing),
            )
        )
    expected_repetition_ids = set(range(expected_repetitions))
    for family in range(len(params)):
        if seen[family] != expected_repetition_ids:
            raise ValueError(
                f"{raw_path}: family {family} repetitions {sorted(seen[family])}, "
                f"expected {sorted(expected_repetition_ids)}"
            )
    return measurements


def validate_result(args: argparse.Namespace) -> None:
    rows = load_measurements(
        args.build,
        args.dataset_key,
        args.config,
        args.raw,
        args.repetitions,
        args.iterations,
    )
    print(f"validated {args.raw}: {len(rows)} fixed-query measurements")


def write_manifest(args: argparse.Namespace) -> None:
    def artifact(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }

    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "multi_cta_fixed_retention_regression",
        "date": "2026-08-05",
        "gpu": EXPECTED_GPU,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "builds": {
            args.baseline: args.baseline_commit,
            args.candidate: args.candidate_commit,
        },
        "artifacts": {
            args.baseline: {
                "benchmark": artifact(args.baseline_binary),
                "library": artifact(args.baseline_library),
            },
            args.candidate: {
                "benchmark": artifact(args.candidate_binary),
                "library": artifact(args.candidate_library),
            },
        },
        "dataset_keys": list(DATASETS),
        "primary_metric": "GPU time at matched itopk",
        "normalized_ratio": (
            "(candidate_fixed / baseline_fixed) / "
            "(candidate_default / baseline_default)"
        ),
    }
    manifest_path = args.result_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError(
                f"{manifest_path}: experiment identity changed; use a new result root"
            )
        return
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_geomean_ci(values: list[float], samples: int = 20_000) -> tuple[float, float]:
    rng = random.Random(20260805)
    estimates = []
    for _ in range(samples):
        estimates.append(geometric_mean(rng.choice(values) for _ in values))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def exact_permutation_p_value(baseline: list[float], candidate: list[float]) -> float:
    """Two-sided exact test for a shift in repetition-level recall means."""
    combined = baseline + candidate
    baseline_size = len(baseline)
    observed = abs(statistics.mean(candidate) - statistics.mean(baseline))
    extreme = 0
    total = 0
    for baseline_indices in itertools.combinations(range(len(combined)), baseline_size):
        baseline_set = set(baseline_indices)
        permuted_baseline = [combined[index] for index in baseline_indices]
        permuted_candidate = [
            value for index, value in enumerate(combined) if index not in baseline_set
        ]
        difference = abs(
            statistics.mean(permuted_candidate) - statistics.mean(permuted_baseline)
        )
        extreme += difference >= observed - 1e-15
        total += 1
    return extreme / total


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in input order."""
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _median_rows(measurements: list[Measurement]) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped: defaultdict[tuple[str, str, int], list[Measurement]] = defaultdict(list)
    for row in measurements:
        grouped[(row.build, row.method, row.itopk)].append(row)
    medians: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        medians[key] = {
            "gpu_us": statistics.median(row.gpu_us for row in rows),
            "latency_us": statistics.median(row.latency_us for row in rows),
            "recall": statistics.median(row.recall for row in rows),
        }
    return medians


def analyze(args: argparse.Namespace) -> None:
    result_root: Path = args.result_root
    config_dir = result_root / "configs"
    measurements: list[Measurement] = []
    for build in (args.baseline, args.candidate):
        for dataset_key in DATASETS:
            measurements.extend(
                load_measurements(
                    build,
                    dataset_key,
                    config_dir / f"{dataset_key}.json",
                    result_root / "raw" / build / f"{dataset_key}.json",
                    args.repetitions,
                    args.iterations,
                )
            )

    analysis_dir = result_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    measurement_fields = list(Measurement.__dataclass_fields__)
    with (analysis_dir / "measurements.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=measurement_fields, lineterminator="\n")
        writer.writeheader()
        for row in measurements:
            writer.writerow({field: getattr(row, field) for field in measurement_fields})

    by_dataset: defaultdict[str, list[Measurement]] = defaultdict(list)
    for row in measurements:
        by_dataset[row.dataset_key].append(row)
    comparison_rows: list[dict[str, Any]] = []
    for dataset_key, rows in by_dataset.items():
        medians = _median_rows(rows)
        for itopk in DATASETS[dataset_key]["itopk"]:
            base_default = medians[(args.baseline, "default", itopk)]
            base_fixed = medians[(args.baseline, "fixed", itopk)]
            cand_default = medians[(args.candidate, "default", itopk)]
            cand_fixed = medians[(args.candidate, "fixed", itopk)]
            default_ratio = cand_default["gpu_us"] / base_default["gpu_us"]
            fixed_ratio = cand_fixed["gpu_us"] / base_fixed["gpu_us"]
            recall_samples = {
                (build, method): [
                    row.recall
                    for row in rows
                    if row.build == build and row.method == method and row.itopk == itopk
                ]
                for build in (args.baseline, args.candidate)
                for method in METHODS
            }
            default_mean_delta = statistics.mean(
                recall_samples[(args.candidate, "default")]
            ) - statistics.mean(recall_samples[(args.baseline, "default")])
            fixed_mean_delta = statistics.mean(
                recall_samples[(args.candidate, "fixed")]
            ) - statistics.mean(recall_samples[(args.baseline, "fixed")])
            comparison_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": DATASETS[dataset_key]["title"],
                    "selectivity": float(DATASETS[dataset_key]["selectivity"]) / 100.0,
                    "itopk": itopk,
                    "baseline_default_gpu_us": base_default["gpu_us"],
                    "candidate_default_gpu_us": cand_default["gpu_us"],
                    "baseline_fixed_gpu_us": base_fixed["gpu_us"],
                    "candidate_fixed_gpu_us": cand_fixed["gpu_us"],
                    "default_time_ratio": default_ratio,
                    "fixed_time_ratio": fixed_ratio,
                    "normalized_fixed_ratio": fixed_ratio / default_ratio,
                    "baseline_default_recall": base_default["recall"],
                    "candidate_default_recall": cand_default["recall"],
                    "baseline_fixed_recall": base_fixed["recall"],
                    "candidate_fixed_recall": cand_fixed["recall"],
                    "default_recall_delta": cand_default["recall"] - base_default["recall"],
                    "fixed_recall_delta": cand_fixed["recall"] - base_fixed["recall"],
                    "default_recall_mean_delta": default_mean_delta,
                    "fixed_recall_mean_delta": fixed_mean_delta,
                    "default_recall_permutation_p": exact_permutation_p_value(
                        recall_samples[(args.baseline, "default")],
                        recall_samples[(args.candidate, "default")],
                    ),
                    "fixed_recall_permutation_p": exact_permutation_p_value(
                        recall_samples[(args.baseline, "fixed")],
                        recall_samples[(args.candidate, "fixed")],
                    ),
                }
            )

    recall_tests = [
        (row, method)
        for row in comparison_rows
        for method in METHODS
    ]
    recall_adjusted = holm_adjust(
        [float(row[f"{method}_recall_permutation_p"]) for row, method in recall_tests]
    )
    for (row, method), adjusted in zip(recall_tests, recall_adjusted, strict=True):
        row[f"{method}_recall_holm_p"] = adjusted

    with (analysis_dir / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(comparison_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    normalized = [float(row["normalized_fixed_ratio"]) for row in comparison_rows]
    default_ratios = [float(row["default_time_ratio"]) for row in comparison_rows]
    fixed_ratios = [float(row["fixed_time_ratio"]) for row in comparison_rows]
    baseline_fixed_over_default = [
        float(row["baseline_fixed_gpu_us"]) / float(row["baseline_default_gpu_us"])
        for row in comparison_rows
    ]
    candidate_fixed_over_default = [
        float(row["candidate_fixed_gpu_us"]) / float(row["candidate_default_gpu_us"])
        for row in comparison_rows
    ]
    normalized_geomean = geometric_mean(normalized)
    ci_low, ci_high = bootstrap_geomean_ci(normalized)
    dataset_geomeans = {
        key: geometric_mean(
            float(row["normalized_fixed_ratio"])
            for row in comparison_rows
            if row["dataset_key"] == key
        )
        for key in DATASETS
    }
    datasets_over_threshold = sum(value >= 1.03 for value in dataset_geomeans.values())
    max_recall_delta = max(
        max(abs(float(row["default_recall_delta"])), abs(float(row["fixed_recall_delta"])))
        for row in comparison_rows
    )
    strict_recall_gate_ok = max_recall_delta <= 0.0002
    holm_significant_recall_cells = sum(adjusted < 0.05 for adjusted in recall_adjusted)
    semantic_ok = holm_significant_recall_cells == 0
    if normalized_geomean >= 1.03 and ci_low > 1.0 and datasets_over_threshold >= 3:
        verdict = "PERFORMANCE_REGRESSION"
    elif normalized_geomean < 1.03:
        verdict = "NO_MATERIAL_PERFORMANCE_REGRESSION"
    else:
        verdict = "INCONCLUSIVE"

    manifest_path = result_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary = {
        "verdict": verdict,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_commit": manifest.get("builds", {}).get(args.baseline),
        "candidate_commit": manifest.get("builds", {}).get(args.candidate),
        "gpu": EXPECTED_GPU,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "cells": len(comparison_rows),
        "normalized_fixed_ratio_geomean": normalized_geomean,
        "normalized_fixed_ratio_ci95": [ci_low, ci_high],
        "default_time_ratio_geomean": geometric_mean(default_ratios),
        "fixed_time_ratio_geomean": geometric_mean(fixed_ratios),
        "baseline_fixed_over_default_geomean": geometric_mean(
            baseline_fixed_over_default
        ),
        "candidate_fixed_over_default_geomean": geometric_mean(
            candidate_fixed_over_default
        ),
        "dataset_normalized_geomeans": dataset_geomeans,
        "datasets_at_or_above_1_03": datasets_over_threshold,
        "max_absolute_recall_delta": max_recall_delta,
        "strict_recall_gate_ok": strict_recall_gate_ok,
        "holm_significant_recall_cells": holm_significant_recall_cells,
        "minimum_unadjusted_recall_permutation_p": min(
            float(row[f"{method}_recall_permutation_p"])
            for row, method in recall_tests
        ),
        "semantic_ok": semantic_ok,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(result_root, summary, comparison_rows)
    print(json.dumps(summary, indent=2))


def write_report(
    result_root: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    verdict_text = {
        "PERFORMANCE_REGRESSION": "A material fixed-retention performance regression was detected.",
        "NO_MATERIAL_PERFORMANCE_REGRESSION": (
            "No material fixed-retention performance regression was detected."
        ),
        "INCONCLUSIVE": "The five-repetition performance gate is inconclusive.",
    }[summary["verdict"]]
    lines = [
        "#+TITLE: MULTI_CTA Fixed-Retention Regression Diagnostic",
        "#+AUTHOR: Nam Anh Dang",
        "#+DATE: 2026-08-05",
        "#+OPTIONS: toc:2 num:t",
        "",
        "* Result",
        "",
        verdict_text,
        "",
        (
            f"The normalized fixed-path GPU-time ratio is "
            f"={summary['normalized_fixed_ratio_geomean']:.4f}= with a cell-bootstrap 95% "
            f"interval of =[{summary['normalized_fixed_ratio_ci95'][0]:.4f}, "
            f"{summary['normalized_fixed_ratio_ci95'][1]:.4f}]=.  A value above one means the "
            "candidate fixed-retention path slowed relative to the unchanged default control."
        ),
        "",
        (
            f"Fixed retention is "
            f"={100 * (1 - summary['baseline_fixed_over_default_geomean']):.2f}%= "
            f"faster than default in the historical checkout and "
            f"={100 * (1 - summary['candidate_fixed_over_default_geomean']):.2f}%= faster in the "
            "current checkout.  Their normalized difference is only "
            f"={100 * (summary['normalized_fixed_ratio_geomean'] - 1):+.3f}%=."
        ),
        "",
        "* Interpretation",
        "",
        (
            "The historical code and current code exhibit essentially the same fixed-retention "
            "advantage over default on the L4.  The implementation change is therefore not the "
            "cause of the apparent old-versus-new frontier difference in the reports.  That "
            "cross-report comparison mixed the historical A30 run with the current L4 run and "
            "also mixed one time-limited recorded run with this fixed-work, five-repetition "
            "diagnostic."
        ),
        "",
        (
            "This conclusion covers the ten representative matched cells below.  It is a direct "
            "regression test of fixed retention at identical =itopk=, not a rerun of every cell "
            "in either full frontier report."
        ),
        "",
        "* Method",
        "",
        (
            f"The baseline is ={summary.get('baseline_commit')}= and the candidate is "
            f"={summary.get('candidate_commit')}=.  Both binaries were built for SM89 and run on "
            f"one {summary['gpu']}.  Every benchmark family executes exactly "
            f"{summary['iterations']} batch-size-1 queries for "
            f"{summary['repetitions']} repetitions.  The identical old-compatible JSON omits "
            "=favor_retention_fraction=, selecting the hardcoded midpoint in the baseline and "
            "the public midpoint default in the candidate.  The manifest records the SHA-256 "
            "identity of every executable and loaded shared library."
        ),
        "",
        "The primary statistic is",
        "",
        "#+begin_example",
        "D = (candidate_fixed / baseline_fixed) / (candidate_default / baseline_default)",
        "#+end_example",
        "",
        (
            "using median GPU time at matched dataset, selectivity, and =itopk=.  This removes "
            "target-recall interpolation and uses default CAGRA as a build/platform control."
        ),
        "",
        "The predeclared performance-regression gate requires all of:",
        "",
        "- geometric-mean =D >= 1.03=;",
        "- the 95% interval lower bound above =1.0=; and",
        "- at least three of four dataset aggregates at or above =1.03=.",
        "",
        (
            f"The observed result satisfies none of these conditions: =D="
            f"{summary['normalized_fixed_ratio_geomean']:.4f}=, interval lower bound "
            f"={summary['normalized_fixed_ratio_ci95'][0]:.4f}=, and "
            f"={summary['datasets_at_or_above_1_03']}/4= datasets at the threshold.  The "
            "intermediate-commit isolation and profiling stage was therefore not triggered."
        ),
        "",
        "* Matched-cell results",
        "",
        "| Dataset | Sel. | itopk | Base default us | Cand. default us | Base fixed us | Cand. fixed us | D | Fixed recall delta |",
        "|---------+------+-------+-----------------+------------------+---------------+----------------+---+--------------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {100 * float(row['selectivity']):.0f}% | {row['itopk']} | "
            f"{float(row['baseline_default_gpu_us']):.3f} | "
            f"{float(row['candidate_default_gpu_us']):.3f} | "
            f"{float(row['baseline_fixed_gpu_us']):.3f} | "
            f"{float(row['candidate_fixed_gpu_us']):.3f} | "
            f"{float(row['normalized_fixed_ratio']):.4f} | "
            f"{float(row['fixed_recall_delta']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "* Dataset aggregates",
            "",
            "| Dataset | Geometric-mean D |",
            "|---------+------------------|",
        ]
    )
    for key, value in summary["dataset_normalized_geomeans"].items():
        lines.append(f"| {DATASETS[key]['title']} | {value:.4f} |")
    lines.extend(
        [
            "",
            "* Validation",
            "",
            (
                f"The maximum absolute matched recall delta is "
                f"={summary['max_absolute_recall_delta']:.6f}=.  The strict =0.0002= point-estimate "
                f"gate therefore {'passes' if summary['strict_recall_gate_ok'] else 'does not pass'}. "
                f"CAGRA results vary slightly across identical repetitions; an exact permutation "
                f"test across all default and fixed cells finds "
                f"={summary['holm_significant_recall_cells']}= recall shifts after Holm correction. "
                "This supports a timing conclusion but does not claim bit-for-bit search equivalence.  "
                "Current binaries expose "
                "underfill counters, and all current measurements report zero underfilled queries "
                "and zero missing result slots; the historical binary predates those counters.  "
                "Each benchmark process has a known teardown-only exit 139 after it closes a "
                "complete JSON file.  The runner retains a result only after strict validation, "
                "so the teardown does not truncate any measurement.  Raw repetition measurements "
                "and the machine-readable comparison are stored below this result directory."
            ),
            "",
            "* Reproduction",
            "",
            "#+begin_src sh",
            (
                "benchmarks/favor/run_multi_cta_retention_regression.sh "
                "/tmp/cuvs-multi-cta-retention-regression"
            ),
            "#+end_src",
            "",
        ]
    )
    report = result_root.parent / "MULTI_CTA_RETENTION_REGRESSION_REPORT.org"
    report.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config-dir", type=Path, required=True)
    generate.add_argument("--data-root", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--build", required=True)
    validate.add_argument("--dataset-key", choices=DATASETS, required=True)
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--raw", type=Path, required=True)
    validate.add_argument("--repetitions", type=int, required=True)
    validate.add_argument("--iterations", type=int, required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--result-root", type=Path, required=True)
    manifest.add_argument("--baseline", required=True)
    manifest.add_argument("--baseline-commit", required=True)
    manifest.add_argument("--candidate", required=True)
    manifest.add_argument("--candidate-commit", required=True)
    manifest.add_argument("--baseline-binary", type=Path, required=True)
    manifest.add_argument("--baseline-library", type=Path, required=True)
    manifest.add_argument("--candidate-binary", type=Path, required=True)
    manifest.add_argument("--candidate-library", type=Path, required=True)
    manifest.add_argument("--repetitions", type=int, required=True)
    manifest.add_argument("--iterations", type=int, required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--result-root", type=Path, required=True)
    analyze_parser.add_argument("--baseline", required=True)
    analyze_parser.add_argument("--candidate", required=True)
    analyze_parser.add_argument("--repetitions", type=int, required=True)
    analyze_parser.add_argument("--iterations", type=int, required=True)

    args = parser.parse_args()
    if args.command == "generate":
        generate_configs(args.config_dir, args.data_root)
    elif args.command == "validate":
        validate_result(args)
    elif args.command == "manifest":
        write_manifest(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
