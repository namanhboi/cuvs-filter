#!/usr/bin/env python3
"""Generate and validate the A100 max_queries=2048 memory/scheduling gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_DIR = SCRIPT_DIR.parent / "gpu_graph"
sys.path.insert(0, str(GPU_DIR))

from analyze_gpu_graph import (
    finite,
    label_value,
    validate_runtime_flags,
)
from generate_configs import (
    config_payload,
    dataset_paths,
    point_identity,
    search_point,
)

CAPS = (512, 1024, 2048)
WORKLOADS = ("yfcc", "emis")
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")
BASELINE_POINT = (64, 1, 0)
DEEP_POINTS = {
    "yfcc": (
        ("default_cagra", 512, 1, 7569),
        ("default_cagra_accumulator", 64, 2, 337),
        ("navix_reference", 129, 2, 0),
    ),
    "emis": (
        ("default_cagra", 512, 1, 7569),
        ("default_cagra_accumulator", 64, 2, 2201),
        ("navix_reference", 56, 2, 0),
    ),
}


def generate(data_root: Path, output: Path) -> None:
    records: list[dict[str, object]] = []
    for cap in CAPS:
        for workload in WORKLOADS:
            paths = dataset_paths(data_root, workload, "throughput", 64)
            source = json.loads(paths.manifest.read_text())
            shards = source.get("shards", [])
            if not shards or int(shards[0].get("query_count", -1)) != 2_048:
                raise ValueError(f"{paths.manifest} must begin with a 2,048-query shard")
            shard = shards[0]
            searches = [
                search_point(
                    method,
                    *BASELINE_POINT,
                    max_queries=cap,
                )
                for method in METHODS
            ]
            if cap == 2_048:
                searches.extend(
                    search_point(method, itopk, width, iterations, max_queries=cap)
                    for method, itopk, width, iterations in DEEP_POINTS[workload]
                )
            path = output / "configs" / f"maxq_{cap}" / f"{workload}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    config_payload(
                        workload=workload,
                        phase="throughput",
                        shard=shard,
                        paths=paths,
                        searches=searches,
                    ),
                    indent=2,
                )
                + "\n"
            )
            records.append(
                {
                    "max_queries": cap,
                    "workload": workload,
                    "query_count": 2_048,
                    "config": str(path.resolve()),
                    "search_points": [point_identity(row) for row in searches],
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment": "retrieve_a100_max_queries_gate",
        "caps": list(CAPS),
        "production_cap": 2_048,
        "workloads": list(WORKLOADS),
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def analyze(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("experiment") != "retrieve_a100_max_queries_gate"
        or int(manifest.get("production_cap", -1)) != 2_048
    ):
        raise ValueError("invalid max-query gate manifest")
    rows: list[dict[str, object]] = []
    for entry in manifest["records"]:
        cap = int(entry["max_queries"])
        workload = str(entry["workload"])
        path = root / "raw" / f"maxq_{cap}" / f"{workload}.json"
        payload = json.loads(path.read_text())
        expected = {
            (
                str(point["method"]),
                int(point["itopk"]),
                int(point["search_width"]),
                int(point["max_iterations"]),
            )
            for point in entry["search_points"]
        }
        observed: set[tuple[str, int, int, int]] = set()
        for record in payload.get("benchmarks", []):
            if record.get("run_type") != "iteration":
                continue
            label = str(record.get("label", ""))
            method = label_value(label, "bitmap_method")
            if method not in METHODS:
                continue
            key = (
                method,
                round(finite(record, "itopk", path)),
                round(finite(record, "search_width", path)),
                round(finite(record, "max_iterations", path)),
            )
            if key not in expected or key in observed:
                raise ValueError(f"unexpected or duplicate gate point {key} in {path}")
            validate_runtime_flags(record, label, method, path, cap)
            if round(finite(record, "n_queries", path)) != 2_048:
                raise ValueError(f"gate point does not cover 2,048 queries: {key}")
            # Match the graph-paper analyzer's ID-set correctness contract.  An
            # underfilled native CAGRA result can report a non-zero
            # InvalidSentinelDistanceErrors rate even when its invalid IDs,
            # sentinel ordering, filter membership, and set-valued recall are
            # correct.  The exact-scan control has a stricter distance-sentinel
            # contract, but applying that contract here incorrectly rejects
            # native Base and Retain rows.
            hard_errors = sum(
                finite(record, field, path)
                for field in (
                    "FilterViolations",
                    "InvalidSentinelErrors",
                    "SentinelOrderErrors",
                )
            )
            invalid_sentinel_distance_errors = finite(
                record, "InvalidSentinelDistanceErrors", path
            )
            duplicates = finite(record, "DuplicateOutputQueries", path)
            underfilled = finite(record, "UnderfilledQueries", path)
            missing_slots = finite(record, "MissingResultSlots", path)
            output_set_semantics = finite(
                record, "OutputSetSemanticsVersion", path
            )
            valid_gt_fraction = finite(record, "ValidGTFraction", path)
            if not all(
                0.0 <= value <= 1.0
                for value in (
                    invalid_sentinel_distance_errors,
                    duplicates,
                    underfilled,
                    missing_slots,
                )
            ):
                raise ValueError(
                    f"invalid diagnostic rate for maxq={cap}/{workload}/{key}: "
                    f"sentinel_distance={invalid_sentinel_distance_errors}, "
                    f"duplicates={duplicates}, underfilled={underfilled}, "
                    f"missing_slots={missing_slots}"
                )
            if duplicates > underfilled + 1e-12:
                raise ValueError(
                    f"duplicate-query rate exceeds unique-underfill rate for "
                    f"maxq={cap}/{workload}/{key}"
                )
            if not math.isclose(output_set_semantics, 1.0, abs_tol=1e-12):
                raise ValueError(
                    f"unsupported output-set semantics for maxq={cap}/{workload}/{key}: "
                    f"{output_set_semantics}"
                )
            if not math.isclose(valid_gt_fraction, 1.0, abs_tol=1e-12):
                raise ValueError(
                    f"incomplete valid ground truth for maxq={cap}/{workload}/{key}: "
                    f"{valid_gt_fraction}"
                )
            if hard_errors != 0 or (method != "default_cagra" and duplicates != 0):
                raise ValueError(
                    f"correctness failure for maxq={cap}/{workload}/{key}: "
                    f"hard_errors={hard_errors}, duplicates={duplicates}, "
                    f"invalid_sentinel_distance_errors="
                    f"{invalid_sentinel_distance_errors}"
                )
            qps = finite(record, "items_per_second", path)
            recall = finite(record, "ValidGTRecall", path)
            if qps <= 0 or not math.isfinite(qps) or not 0 <= recall <= 1:
                raise ValueError(f"invalid QPS/recall for {key} in {path}")
            rows.append(
                {
                    "max_queries": cap,
                    "workload": workload,
                    "method": method,
                    "itopk": key[1],
                    "search_width": key[2],
                    "max_iterations": key[3],
                    "queries": 2_048,
                    "recall": recall,
                    "qps": qps,
                    "duplicate_output_queries": duplicates,
                    "invalid_sentinel_distance_errors": (
                        invalid_sentinel_distance_errors
                    ),
                    "underfilled_queries": underfilled,
                    "missing_result_slots": missing_slots,
                }
            )
            observed.add(key)
        if observed != expected:
            raise ValueError(f"missing gate points in {path}: {sorted(expected - observed)}")

    baseline = [
        row
        for row in rows
        if (int(row["itopk"]), int(row["search_width"]), int(row["max_iterations"]))
        == BASELINE_POINT
    ]
    sensitivity: list[dict[str, object]] = []
    for workload in WORKLOADS:
        for method in METHODS:
            local = [
                row
                for row in baseline
                if row["workload"] == workload and row["method"] == method
            ]
            if {int(row["max_queries"]) for row in local} != set(CAPS):
                raise ValueError(f"incomplete cap sensitivity for {workload}/{method}")
            best = max(float(row["qps"]) for row in local)
            production = next(row for row in local if int(row["max_queries"]) == 2_048)
            sensitivity.append(
                {
                    "workload": workload,
                    "method": method,
                    "best_qps": best,
                    "maxq_2048_qps": float(production["qps"]),
                    "maxq_2048_fraction_of_best": float(production["qps"]) / best,
                }
            )

    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "max_queries_gate_points.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "production_max_queries": 2_048,
        "queries_per_gate_call": 2_048,
        "sensitivity": sensitivity,
    }
    (output / "max_queries_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--data-root", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.data_root.resolve(), args.output.resolve())
    else:
        analyze(args.root.resolve())


if __name__ == "__main__":
    main()
