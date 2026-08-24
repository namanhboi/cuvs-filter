#!/usr/bin/env python3
"""Generate and summarize the A100 YFCC retention/NaviX mechanism diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_DIR = SCRIPT_DIR.parent / "gpu_graph"
sys.path.insert(0, str(GPU_DIR))

from generate_configs import (
    MAX_QUERIES,
    config_payload,
    dataset_paths,
    search_point,
)

VARIANTS = (
    ("base_l512_w2_b0", "default_cagra", 512, 2, 0),
    ("retain_l512_w2_b0", "default_cagra_accumulator", 512, 2, 0),
    ("navix_l64_w1_b0", "navix_reference", 64, 1, 0),
    ("navix_l64_w1_i1044", "navix_reference", 64, 1, 1044),
    ("navix_l512_w2_i1044", "navix_reference", 512, 2, 1044),
)


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def validate_raw_results(path: Path) -> None:
    payload = json.loads(path.read_text())
    expected = {row[0] for row in VARIANTS}
    observed: set[str] = set()
    for record in payload.get("benchmarks", []):
        if record.get("run_type") != "iteration":
            continue
        variant = label_value(str(record.get("label", "")), "favor_diagnostics_variant")
        if variant not in expected:
            continue
        if variant in observed:
            raise ValueError(f"duplicate mechanism result for {variant}: {path}")
        if round(float(record.get("max_queries", -1))) != MAX_QUERIES:
            raise ValueError(
                f"mechanism result does not use max_queries={MAX_QUERIES}: {variant}"
            )
        if round(float(record.get("n_queries", -1))) != 1_000:
            raise ValueError(f"mechanism result does not cover 1,000 queries: {variant}")
        observed.add(variant)
    if observed != expected:
        raise ValueError(f"mechanism raw results are incomplete: missing {sorted(expected - observed)}")


def generate(data_root: Path, output: Path, diagnostics: Path) -> None:
    paths = dataset_paths(data_root, "yfcc", "correctness", 64)
    source = json.loads(paths.manifest.read_text())
    shards = source.get("shards", [])
    if len(shards) != 1 or int(shards[0]["query_count"]) != 1_000:
        raise ValueError(
            "YFCC correctness input must be one 1,000-query shard"
        )
    shard = shards[0]
    ground_truth = Path(shard["directory"]) / "groundtruth.ibin"
    searches = []
    for variant, method, itopk, width, iterations in VARIANTS:
        row = search_point(method, itopk, width, iterations)
        row.update(
            {
                "max_queries": MAX_QUERIES,
                "favor_diagnostics_output": str(
                    (diagnostics / variant).resolve()
                ),
                "favor_diagnostics_groundtruth": str(ground_truth.resolve()),
                "favor_diagnostics_dataset": "yfcc10m-a100",
                "favor_diagnostics_variant": variant,
            }
        )
        searches.append(row)
    payload = config_payload(
        workload="yfcc",
        phase="correctness",
        shard=shard,
        paths=paths,
        searches=searches,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1_000 or [int(row["query_id"]) for row in rows] != list(
        range(1_000)
    ):
        raise ValueError(f"incomplete diagnostic capture: {path}")
    return rows


def mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"invalid {field}")
    return sum(values) / len(values)


def popcount_mean(rows: list[dict[str, str]], field: str) -> float:
    return sum(int(row[field]).bit_count() for row in rows) / (10 * len(rows))


def validate_base_retain_traversal(
    base: list[dict[str, str]], retain: list[dict[str, str]]
) -> None:
    """Prove that Retain changes result state, not graph traversal.

    Predicate probes are deliberately excluded.  Retain evaluates the predicate
    while offering distance-eligible candidates to its passing-result
    accumulator, whereas Base evaluates it for parent retirement and terminal
    output filtering.  Their probe counts are therefore useful measured work,
    but they are not a traversal-equivalence invariant.
    """
    for query, (left, right) in enumerate(zip(base, retain, strict=True)):
        for field in (
            "iterations",
            "graph_rows_read",
            "distance_evaluations",
            "passing_admissions",
            "gt_seen_mask",
        ):
            if left[field] != right[field]:
                raise ValueError(
                    f"Base/Retain {field} differs at query {query}"
                )


def summarize(diagnostics: Path, raw_results: Path, output: Path) -> None:
    validate_raw_results(raw_results)
    captures: dict[str, tuple[list[dict[str, str]], dict]] = {}
    for variant, method, itopk, width, iterations in VARIANTS:
        directory = diagnostics / variant
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            int(manifest.get("schema_version", -1)) != 9
            or int(manifest.get("num_queries", -1)) != 1_000
            or int(manifest.get("graph_degree", -1)) != 64
            or int(manifest.get("itopk", -1)) != itopk
            or int(manifest.get("search_width", -1)) != width
            or int(manifest.get("configured_max_iterations", -1)) != iterations
            or bool(manifest.get("timing_valid", True))
        ):
            raise ValueError(
                f"diagnostic manifest contract failed: {directory}"
            )
        captures[variant] = (
            read_rows(directory / "query_summary.csv"),
            manifest,
        )

    base = captures["base_l512_w2_b0"][0]
    retain = captures["retain_l512_w2_b0"][0]
    validate_base_retain_traversal(base, retain)
    base_predicate_probes = mean(base, "predicate_probes")
    retain_predicate_probes = mean(retain, "predicate_probes")
    retention = {
        "configuration": {
            "itopk": 512,
            "search_width": 2,
            "max_iterations": 0,
        },
        "base_recall": mean(base, "recall"),
        "retain_recall": mean(retain, "recall"),
        "gt_seen_rate": popcount_mean(base, "gt_seen_mask"),
        "base_retention_gap": popcount_mean(base, "gt_seen_mask")
        - mean(base, "recall"),
        "retain_retention_gap": popcount_mean(retain, "gt_seen_mask")
        - mean(retain, "recall"),
        "distance_evaluations_per_query": mean(base, "distance_evaluations"),
        "graph_rows_per_query": mean(base, "graph_rows_read"),
        "base_predicate_probes_per_query": base_predicate_probes,
        "retain_predicate_probes_per_query": retain_predicate_probes,
        "retain_additional_predicate_probes_per_query": (
            retain_predicate_probes - base_predicate_probes
        ),
    }

    navix: dict[str, dict[str, object]] = {}
    for variant, method, itopk, width, iterations in VARIANTS[2:]:
        rows = captures[variant][0]
        first_checks = sum(int(row["navix_first_hop_checks"]) for row in rows)
        second_checks = sum(
            int(row["navix_second_hop_checks"]) for row in rows
        )
        seeded = [row for row in rows if int(row["navix_seed_found"]) != 0]
        navix[variant] = {
            "configuration": {
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
            },
            "recall": mean(rows, "recall"),
            "underfilled_fraction": sum(
                int(row["output_count"]) < 10 for row in rows
            )
            / len(rows),
            "seedless_fraction": (len(rows) - len(seeded)) / len(rows),
            "seeded_underfilled_fraction": (
                sum(int(row["output_count"]) < 10 for row in seeded)
                / len(rows)
            ),
            "frontier_exhausted_fraction": sum(
                int(row["stop_reason"]) == 3 for row in rows
            )
            / len(rows),
            "hard_cap_fraction": sum(
                int(row["stop_reason"]) in (1, 2) for row in rows
            )
            / len(rows),
            "first_hop_yield": (
                sum(int(row["navix_first_hop_passing"]) for row in rows)
                / first_checks
                if first_checks
                else 0.0
            ),
            "second_hop_yield": (
                sum(int(row["navix_second_hop_passing"]) for row in rows)
                / second_checks
                if second_checks
                else 0.0
            ),
            "gt_checked_rate": popcount_mean(rows, "gt_seen_mask"),
            "gt_admitted_rate": popcount_mean(rows, "navix_gt_admitted_mask"),
            "gt_retained_rate": popcount_mean(rows, "navix_gt_retained_mask"),
            "gt_output_rate": popcount_mean(rows, "navix_gt_output_mask"),
        }
    payload = {
        "schema_version": 2,
        "status": "PASS",
        "dataset": "YFCC-10M",
        "graph_degree": 64,
        "queries": 1_000,
        "max_queries": MAX_QUERIES,
        "timing_valid": False,
        "retention": retention,
        "navix": navix,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--data-root", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--diagnostics", type=Path, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--diagnostics", type=Path, required=True)
    summarize_parser.add_argument("--raw-results", type=Path, required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(
            args.data_root.resolve(),
            args.output.resolve(),
            args.diagnostics.resolve(),
        )
    else:
        summarize(
            args.diagnostics.resolve(),
            args.raw_results.resolve(),
            args.output.resolve(),
        )


if __name__ == "__main__":
    main()
