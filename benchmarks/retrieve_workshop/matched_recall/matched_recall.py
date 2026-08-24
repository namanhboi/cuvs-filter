#!/usr/bin/env python3
"""Generate, validate, tune, and summarize fixed-recall CAGRA measurements.

The experiment is deliberately separate from the broad GPU graph sweep.  Calibration uses one
complete 10,000-query repetition and may guide subsequent configurations, but only frozen finalist
configurations rerun for three repetitions are eligible for paper tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_GRAPH_DIR = SCRIPT_DIR.parent / "gpu_graph"
sys.path.insert(0, str(GPU_GRAPH_DIR))

from analyze_gpu_graph import aggregate_repetitions, load_group
from dataset_profile import (
    load_profile,
    profile_record,
    workload_spec,
)
from generate_configs import (
    MAX_QUERIES,
    config_payload,
    dataset_paths,
    point_identity,
    search_point,
)

WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")
PROFILE = load_profile()
WIDTHS = tuple(int(value) for value in PROFILE["matched_widths"])
TUNING_WIDTHS = tuple(value for value in WIDTHS if value in (1, 2))
ANCHOR_L = (32, 64, 128, 256, 512)
MIN_L = 10
MAX_L = 512
L_QUANTUM = 32
MAX_ITERATIONS = 7569
HASHMAP_MAX_FILL_RATE = 0.5
MAX_NORMAL_HASH_BITLEN = 20
MAX_NORMAL_HASH_ENTRIES = (1 << MAX_NORMAL_HASH_BITLEN) * HASHMAP_MAX_FILL_RATE
TARGETS = {"yfcc": 0.80, "em": 0.95, "emis": 0.95, "r": 0.95}
TARGET_WINDOW = 0.002
EXPECTED_QUERIES = 10_000

# Only these cells need tighter B0 calibration for the paper.  The other four cells already have
# valid, intentionally deep endpoints and are rerun unchanged in the final 12-cell measurement.
TIGHT_PAIRS = (
    ("yfcc", "navix_reference"),
    ("em", "default_cagra"),
    ("em", "default_cagra_accumulator"),
    ("em", "navix_reference"),
    ("emis", "navix_reference"),
    ("r", "default_cagra"),
    ("r", "default_cagra_accumulator"),
    ("r", "navix_reference"),
)
NAVIX_REFINEMENT_WORKLOADS = ("em", "r")
NAVIX_REFINEMENT_MAX_L = 31
NAVIX_REFINEMENT_FINALISTS = 4

LOCKED_POINTS = (
    {"workload": "yfcc", "method": "default_cagra", "itopk": 512, "search_width": 4, "max_iterations": 2046},
    {"workload": "yfcc", "method": "default_cagra_accumulator", "itopk": 64, "search_width": 1, "max_iterations": 675},
    {"workload": "emis", "method": "default_cagra", "itopk": 512, "search_width": 4, "max_iterations": 4092},
    {"workload": "emis", "method": "default_cagra_accumulator", "itopk": 512, "search_width": 4, "max_iterations": 536},
)

MEASUREMENT_FIELDS = (
    "group",
    "stage",
    "workload",
    "graph_degree",
    "intermediate_graph_degree",
    "method",
    "itopk",
    "search_width",
    "max_iterations",
    "resolved_iterations",
    "repetition_index",
    "shards",
    "queries",
    "recall",
    "valid_gt_fraction",
    "qps",
    "seconds",
    "filter_violations",
    "sentinel_errors",
    "duplicate_output_query_rate",
    "underfilled_queries",
    "missing_result_slots",
)

SUMMARY_FIELDS = (
    "group",
    "phase",
    "workload",
    "graph_degree",
    "intermediate_graph_degree",
    "method",
    "itopk",
    "search_width",
    "max_iterations",
    "resolved_iterations",
    "repetitions",
    "shards_per_repetition",
    "queries_per_repetition",
    "recall_median",
    "recall_min",
    "recall_max",
    "valid_gt_fraction_min",
    "qps_median",
    "qps_min",
    "qps_max",
    "seconds_median",
    "filter_violations",
    "sentinel_errors",
    "duplicate_output_query_rate_max",
    "underfilled_queries_max",
    "missing_result_slots_max",
    "target_recall",
    "target_reached",
    "within_target_window",
    "selected",
    "paper_included",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def validate_raw_output(
    raw_path: Path, config_path: Path, repetitions: int
) -> dict[str, int]:
    """Reject partial Google Benchmark JSON before a resumable group skips it."""
    if repetitions not in (1, 3):
        raise ValueError("matched-recall raw output uses one or three repetitions")
    raw = json.loads(raw_path.read_text())
    config = json.loads(config_path.read_text())
    expected_searches = sum(
        len(index.get("search_params", [])) for index in config.get("index", [])
    )
    if expected_searches <= 0:
        raise ValueError(f"benchmark config has no search points: {config_path}")
    observations = [
        row for row in raw.get("benchmarks", []) if row.get("run_type") == "iteration"
    ]
    if any(row.get("error_occurred") or row.get("skipped") for row in observations):
        raise ValueError(f"benchmark output contains an error or skipped row: {raw_path}")
    by_name: dict[str, list[dict]] = {}
    for row in observations:
        by_name.setdefault(str(row.get("name", "")), []).append(row)
    if len(by_name) != expected_searches:
        raise ValueError(
            f"benchmark output has {len(by_name)}/{expected_searches} search points: {raw_path}"
        )
    expected_repetitions = list(range(repetitions))
    for name, rows in by_name.items():
        observed_repetitions = sorted(
            int(row.get("repetition_index", -1)) for row in rows
        )
        if observed_repetitions != expected_repetitions:
            raise ValueError(
                f"benchmark {name!r} has repetitions {observed_repetitions}, "
                f"expected {expected_repetitions}: {raw_path}"
            )
    return {
        "search_points": expected_searches,
        "repetitions": repetitions,
        "iteration_rows": len(observations),
    }


def target(workload: str) -> float:
    return TARGETS[workload]


def internal_itopk(itopk: int) -> int:
    """Return CAGRA's 32-element internal top-k capacity for a requested L."""
    if not MIN_L <= itopk <= MAX_L:
        raise ValueError(f"invalid SINGLE_CTA L={itopk}")
    return ((itopk + L_QUANTUM - 1) // L_QUANTUM) * L_QUANTUM


def resolved_b0(itopk: int, width: int, dataset_size: int, degree: int) -> int:
    # search_plan.cuh computes the automatic iteration budget from the caller-requested L and only
    # then rounds the actual in-kernel itopk capacity to 32.  Requested non-multiple values
    # therefore provide real, fine-grained B0 tuning even when they share an internal capacity.
    internal_itopk(itopk)
    iterations = itopk // width
    reachable = 1
    while reachable < dataset_size:
        reachable *= max(2, degree // 2)
        iterations += 1
    return iterations


def resolved_iterations(workload: str, itopk: int, width: int, maximum: int) -> int:
    if maximum:
        return maximum
    spec = workload_spec(workload, PROFILE)
    dataset_size = int(spec["dataset_size"])
    degree = int(spec["graph_degree"])
    return resolved_b0(itopk, width, dataset_size, degree)


def visit_multiplier(method: str) -> int:
    """Match search_plan.cuh's conservative visited-node accounting."""
    return 2 if method == "navix_reference" else 1


def hash_nodes(workload: str, method: str, itopk: int, width: int, maximum: int) -> int:
    degree = int(workload_spec(workload, PROFILE)["graph_degree"])
    iterations = resolved_iterations(workload, itopk, width, maximum)
    return internal_itopk(itopk) + width * degree * visit_multiplier(method) * iterations


def hash_feasible(workload: str, method: str, itopk: int, width: int, maximum: int) -> bool:
    """Whether normal-hash planning remains within cuVS's 20-bit, 0.5-fill limit."""
    return hash_nodes(workload, method, itopk, width, maximum) <= MAX_NORMAL_HASH_ENTRIES


def key(row: dict) -> tuple[str, str, int, int, int]:
    return (
        str(row["workload"]),
        str(row["method"]),
        int(row["itopk"]),
        int(row["search_width"]),
        int(row["max_iterations"]),
    )


def normalize_point(row: dict) -> dict[str, object]:
    workload = str(row["workload"])
    method = str(row["method"])
    itopk = int(row["itopk"])
    width = int(row["search_width"])
    maximum = int(row["max_iterations"])
    if workload not in WORKLOADS or method not in METHODS:
        raise ValueError(f"invalid matched-recall point: {row}")
    if not MIN_L <= itopk <= MAX_L:
        raise ValueError(f"requested L must be in [{MIN_L},{MAX_L}]: {row}")
    if width not in WIDTHS:
        raise ValueError(f"W must be one of {WIDTHS}: {row}")
    if not 0 <= maximum <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations outside [0,{MAX_ITERATIONS}]: {row}")
    if maximum and maximum <= resolved_iterations(workload, itopk, width, 0):
        raise ValueError(f"explicit depth must exceed B0: {row}")
    if not hash_feasible(workload, method, itopk, width, maximum):
        raise ValueError(
            "configuration exceeds SINGLE_CTA's 20-bit normal-hash capacity "
            f"({hash_nodes(workload, method, itopk, width, maximum)} visited nodes): {row}"
        )
    return {
        "workload": workload,
        "method": method,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": maximum,
    }


def write_group(
    *,
    result_root: Path,
    data_root: Path,
    group: str,
    repetitions: int,
    points: list[dict],
    stage: str | None = None,
) -> None:
    if repetitions not in (1, 3):
        raise ValueError("matched-recall groups use one or three repetitions")
    if stage is None:
        stage = "final" if repetitions == 3 else "calibration"
    if stage not in {"calibration", "finalist", "final"}:
        raise ValueError(f"invalid matched-recall stage {stage!r}")
    if stage == "calibration" and repetitions != 1:
        raise ValueError("calibration groups use one repetition")
    if stage in {"finalist", "final"} and repetitions != 3:
        raise ValueError(f"{stage} groups use three repetitions")
    normalized = [normalize_point(row) for row in points]
    if len({key(row) for row in normalized}) != len(normalized):
        raise ValueError("duplicate point in requested group")
    if not normalized:
        raise ValueError("cannot generate an empty group")
    group_root = result_root / "configs" / group
    if group_root.exists():
        raise FileExistsError(group_root)
    group_index = {
        "schema_version": 1,
        "experiment": "retrieve_workshop_matched_recall",
        "group": group,
        "stage": stage,
        "repetitions": repetitions,
        "targets": TARGETS,
        "dataset_profile": profile_record(PROFILE),
        "points": normalized,
    }
    for workload in WORKLOADS:
        local = [row for row in normalized if row["workload"] == workload]
        if not local:
            continue
        paths = dataset_paths(data_root, workload, "throughput", 64)
        source = json.loads(paths.manifest.read_text())
        shards = source.get("shards", [])
        if not shards or sum(int(row["query_count"]) for row in shards) != EXPECTED_QUERIES:
            raise ValueError(f"invalid throughput shard coverage in {paths.manifest}")
        searches = [
            search_point(
                str(row["method"]),
                int(row["itopk"]),
                int(row["search_width"]),
                int(row["max_iterations"]),
            )
            for row in local
        ]
        workload_root = group_root / workload
        workload_root.mkdir(parents=True, exist_ok=False)
        configs: list[dict] = []
        cursor = 0
        for shard_index, shard in enumerate(shards):
            first = int(shard["first_query"])
            count = int(shard["query_count"])
            if first != cursor or count <= 0:
                raise ValueError(f"non-contiguous shard {shard_index} in {paths.manifest}")
            cursor += count
            config = workload_root / f"shard_{shard_index:02d}.json"
            config.write_text(
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
                + "\n",
                encoding="utf-8",
            )
            configs.append(
                {
                    "config": str(config.resolve()),
                    "shard_index": shard_index,
                    "first_query": first,
                    "query_count": count,
                }
            )
        manifest = {
            "schema_version": 1,
            # Reuse the strict raw parser and its runtime/correctness contracts.
            "experiment": "retrieve_workshop_gpu_graph",
            "matched_recall_schema_version": 1,
            "group": group,
            "matched_stage": stage,
            "phase": "throughput",
            "workload": workload,
            "target_recall": target(workload),
            "target_window": TARGET_WINDOW,
            "k": 10,
            "max_queries": MAX_QUERIES,
            "graph_degree": paths.graph_degree,
            "intermediate_graph_degree": paths.intermediate_graph_degree,
            "repetitions": repetitions,
            "expected_queries": EXPECTED_QUERIES,
            "expected_shards": len(configs),
            "source_bitmap_manifest": str(paths.manifest.resolve()),
            "search_points": [point_identity(row) for row in searches],
            "configs": configs,
        }
        (workload_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    group_root.mkdir(parents=True, exist_ok=True)
    (group_root / "manifest.json").write_text(
        json.dumps(group_index, indent=2) + "\n", encoding="utf-8"
    )


def experiment_manifests(result_root: Path) -> list[tuple[Path, dict]]:
    output: list[tuple[Path, dict]] = []
    for path in sorted((result_root / "configs").glob("*/*/manifest.json")):
        payload = json.loads(path.read_text())
        if payload.get("matched_recall_schema_version") != 1:
            continue
        output.append((path, payload))
    return output


def measurements(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path, manifest in experiment_manifests(result_root):
        group = str(manifest["group"])
        if not (result_root / "raw" / group / str(manifest["workload"])).is_dir():
            continue
        raw = load_group(path, manifest, result_root / "raw")
        for point in aggregate_repetitions(raw):
            maximum = int(point.max_iterations)
            row = {
                "group": group,
                "stage": str(manifest["matched_stage"]),
                "workload": point.workload,
                "graph_degree": point.graph_degree,
                "intermediate_graph_degree": point.intermediate_graph_degree,
                "method": point.method,
                "itopk": point.itopk,
                "search_width": point.search_width,
                "max_iterations": maximum,
                "resolved_iterations": resolved_iterations(
                    point.workload, point.itopk, point.search_width, maximum
                ),
                "repetition_index": point.repetition_index,
                "shards": point.shards,
                "queries": point.queries,
                "recall": point.recall,
                "valid_gt_fraction": point.valid_gt_fraction,
                "qps": point.qps,
                "seconds": point.seconds,
                "filter_violations": point.filter_violations,
                "sentinel_errors": point.sentinel_errors,
                "duplicate_output_query_rate": point.duplicate_output_query_rate,
                "underfilled_queries": point.underfilled_queries,
                "missing_result_slots": point.missing_result_slots,
            }
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["stage"]),
            str(row["group"]),
            str(row["workload"]),
            str(row["method"]),
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
            int(row["repetition_index"]),
        )
    )
    return rows


def calibration_rows(result_root: Path) -> list[dict[str, object]]:
    return [row for row in measurements(result_root) if row["stage"] == "calibration"]


def measured_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, int, int, int]]:
    return {key(row) for row in rows}


def execution_fingerprint(workload: str, itopk: int, width: int) -> tuple[int, int]:
    """Kernel capacity and effective B0 budget produced by a requested (L, W)."""
    return (internal_itopk(itopk), resolved_iterations(workload, itopk, width, 0))


def navix_refinement_points() -> dict:
    """Enumerate all distinct below-L32 B0 executions for EM/R NaviX."""
    points: list[dict[str, object]] = []
    for workload in NAVIX_REFINEMENT_WORKLOADS:
        for width in TUNING_WIDTHS:
            fingerprints: set[tuple[int, int]] = set()
            for requested_l in range(MIN_L, NAVIX_REFINEMENT_MAX_L + 1):
                fingerprint = execution_fingerprint(workload, requested_l, width)
                if fingerprint in fingerprints:
                    continue
                fingerprints.add(fingerprint)
                points.append(
                    {
                        "workload": workload,
                        "method": "navix_reference",
                        "itopk": requested_l,
                        "search_width": width,
                        "max_iterations": 0,
                    }
                )
    return {
        "schema_version": 1,
        "experiment": "retrieve_workshop_navix_em_r_refinement",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {workload: target(workload) for workload in NAVIX_REFINEMENT_WORKLOADS},
        "target_window": TARGET_WINDOW,
        "selection_contract": (
            "Every distinct below-L32 B0 execution for W in {1,2}; "
            "no interpolation and no explicit iteration budget"
        ),
        "points": points,
    }


def load_baseline_measurements(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(path):
        if str(row.get("stage", "")) != "calibration" or int(row["max_iterations"]) != 0:
            continue
        rows.append(
            {
                "workload": str(row["workload"]),
                "method": str(row["method"]),
                "itopk": int(row["itopk"]),
                "search_width": int(row["search_width"]),
                "recall": float(row["recall"]),
            }
        )
    if not rows:
        raise ValueError(f"no B0 calibration rows in {path}")
    return rows


def tight_refinement_points(baseline_path: Path) -> dict:
    """Enumerate every distinct requested-L execution between measured target brackets."""
    baseline = load_baseline_measurements(baseline_path)
    points: list[dict[str, object]] = []
    brackets: list[dict[str, object]] = []
    for workload, method in TIGHT_PAIRS:
        goal = target(workload)
        for width in TUNING_WIDTHS:
            local = sorted(
                (
                    row
                    for row in baseline
                    if row["workload"] == workload
                    and row["method"] == method
                    and int(row["search_width"]) == width
                ),
                key=lambda row: int(row["itopk"]),
            )
            passing = [row for row in local if float(row["recall"]) >= goal]
            if not passing:
                raise ValueError(f"baseline has no B0 upper bracket for {workload}/{method}/W{width}")
            upper = min(int(row["itopk"]) for row in passing)
            failures = [
                int(row["itopk"])
                for row in local
                if int(row["itopk"]) < upper and float(row["recall"]) < goal
            ]
            lower = max(failures) if failures else MIN_L
            if lower > upper:
                raise ValueError(f"malformed bracket for {workload}/{method}/W{width}")
            seen_fingerprints: set[tuple[int, int]] = set()
            emitted = 0
            for requested_l in range(lower, upper + 1):
                fingerprint = execution_fingerprint(workload, requested_l, width)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                points.append(
                    {
                        "workload": workload,
                        "method": method,
                        "itopk": requested_l,
                        "search_width": width,
                        "max_iterations": 0,
                    }
                )
                emitted += 1
            brackets.append(
                {
                    "workload": workload,
                    "method": method,
                    "search_width": width,
                    "lower_requested_l": lower,
                    "upper_requested_l": upper,
                    "distinct_executions": emitted,
                }
            )
    return {
        "schema_version": 2,
        "experiment": "retrieve_workshop_tight_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_measurements": str(baseline_path.resolve()),
        "baseline_sha256": sha256(baseline_path),
        "targets": TARGETS,
        "target_window": TARGET_WINDOW,
        "tuning_widths": list(TUNING_WIDTHS),
        "brackets": brackets,
        "points": points,
    }


def grouped_stage_rows(
    result_root: Path, stage: str, expected_repetitions: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, object]]] = {}
    for row in measurements(result_root):
        if row["stage"] == stage:
            if int(row["queries"]) != EXPECTED_QUERIES:
                raise ValueError(f"{stage} row does not cover {EXPECTED_QUERIES} queries")
            if float(row["filter_violations"]) or float(row["sentinel_errors"]):
                raise ValueError(f"{stage} correctness failure")
            if (
                row["method"] != "default_cagra"
                and float(row["duplicate_output_query_rate"]) != 0
            ):
                raise ValueError(f"{stage} Retain/NaviX output is not duplicate-free")
            grouped.setdefault(key(row), []).append(row)
    output: list[dict[str, object]] = []
    for identity, members in sorted(grouped.items()):
        if len(members) != expected_repetitions or sorted(
            int(row["repetition_index"]) for row in members
        ) != list(range(expected_repetitions)):
            raise ValueError(
                f"{stage} point has the wrong repetition set: {identity}"
            )
        recalls = [float(row["recall"]) for row in members]
        qps = [float(row["qps"]) for row in members]
        first = members[0]
        output.append(
            {
                "workload": first["workload"],
                "method": first["method"],
                "itopk": int(first["itopk"]),
                "search_width": int(first["search_width"]),
                "max_iterations": int(first["max_iterations"]),
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "qps_median": statistics.median(qps),
                "repetitions": len(members),
            }
        )
    return output


def select_navix_refinement_finalists(
    rows: list[dict[str, object]],
) -> dict:
    """Select the fastest calibration points inside the closed target window."""
    expected = {key(point) for point in navix_refinement_points()["points"]}
    measured = {key(row): row for row in rows if key(row) in expected}
    missing = sorted(expected - set(measured))
    if missing:
        raise ValueError(
            f"NaviX EM/R refinement is missing {len(missing)} calibration points: "
            f"{missing[:4]}"
        )

    selected: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for workload in NAVIX_REFINEMENT_WORKLOADS:
        goal = target(workload)
        local = [
            row
            for identity, row in measured.items()
            if identity[0] == workload and identity[1] == "navix_reference"
        ]
        within = [
            row
            for row in local
            if float(row["recall_min"]) >= goal
            and float(row["recall_median"]) <= goal + TARGET_WINDOW + 1e-12
        ]
        if not within:
            nearest = sorted(
                local,
                key=lambda row: (
                    abs(float(row["recall_median"]) - goal),
                    -float(row["qps_median"]),
                ),
            )[:4]
            summary = [
                {
                    "itopk": int(row["itopk"]),
                    "search_width": int(row["search_width"]),
                    "recall": float(row["recall_median"]),
                }
                for row in nearest
            ]
            raise ValueError(
                f"no {workload} NaviX calibration point lies in "
                f"[{goal:.3f},{goal + TARGET_WINDOW:.3f}]; nearest={summary}"
            )
        finalists = sorted(
            within, key=lambda row: float(row["qps_median"]), reverse=True
        )[:NAVIX_REFINEMENT_FINALISTS]
        points = [
            {
                "workload": workload,
                "method": "navix_reference",
                "itopk": int(row["itopk"]),
                "search_width": int(row["search_width"]),
                "max_iterations": 0,
            }
            for row in finalists
        ]
        selected.extend(points)
        decisions.append(
            {
                "workload": workload,
                "selection_rule": "four fastest one-run points inside the target window",
                "points": points,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "retrieve_workshop_navix_em_r_refinement",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "final",
        "targets": {workload: target(workload) for workload in NAVIX_REFINEMENT_WORKLOADS},
        "target_window": TARGET_WINDOW,
        "decisions": decisions,
        "points": selected,
    }


def navix_refinement_finalists(result_root: Path) -> dict:
    rows = grouped_stage_rows(result_root, "calibration", 1)
    return select_navix_refinement_finalists(rows)


def tight_finalists(result_root: Path) -> dict:
    rows = grouped_stage_rows(result_root, "calibration", 1)
    selected: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for workload, method in TIGHT_PAIRS:
        goal = target(workload)
        local = [row for row in rows if row["workload"] == workload and row["method"] == method]
        within = [
            row
            for row in local
            if float(row["recall_median"]) >= goal
            and float(row["recall_median"]) <= goal + TARGET_WINDOW + 1e-12
        ]
        if within:
            candidates = sorted(within, key=lambda row: float(row["qps_median"]), reverse=True)[:4]
            rule = "four fastest calibration points inside the target window"
        else:
            candidates = sorted(
                local,
                key=lambda row: (
                    abs(float(row["recall_median"]) - goal),
                    -float(row["qps_median"]),
                ),
            )[:4]
            rule = "no calibration point in the target window; four closest measured points"
        if not candidates:
            raise ValueError(f"missing tight calibration data for {workload}/{method}")
        points = [
            {
                "workload": workload,
                "method": method,
                "itopk": int(row["itopk"]),
                "search_width": int(row["search_width"]),
                "max_iterations": 0,
            }
            for row in candidates
        ]
        selected.extend(points)
        decisions.append({"workload": workload, "method": method, "selection_rule": rule, "points": points})
    return {
        "schema_version": 2,
        "experiment": "retrieve_workshop_tight_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "finalist",
        "decisions": decisions,
        "points": selected,
    }


def tight_paper_points(result_root: Path) -> dict:
    rows = grouped_stage_rows(result_root, "finalist", 3)
    selected: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for workload, method in TIGHT_PAIRS:
        goal = target(workload)
        local = [row for row in rows if row["workload"] == workload and row["method"] == method]
        within = [
            row
            for row in local
            if float(row["recall_min"]) >= goal
            and float(row["recall_median"]) <= goal + TARGET_WINDOW + 1e-12
        ]
        if within:
            chosen = max(within, key=lambda row: float(row["qps_median"]))
            rule = "highest three-run median QPS inside the target window"
        else:
            qualifying = [row for row in local if float(row["recall_min"]) >= goal]
            if qualifying:
                chosen = min(
                    qualifying,
                    key=lambda row: (float(row["recall_median"]) - goal, -float(row["qps_median"])),
                )
                rule = "target reached but no finalist remained inside the window; smallest overshoot"
            else:
                chosen = max(local, key=lambda row: (float(row["recall_median"]), float(row["qps_median"])))
                rule = "target not reached; maximum measured recall"
        point = {
            "workload": workload,
            "method": method,
            "itopk": int(chosen["itopk"]),
            "search_width": int(chosen["search_width"]),
            "max_iterations": 0,
        }
        selected.append(point)
        decisions.append({"workload": workload, "method": method, "selection_rule": rule, "point": point})
    selected.extend(dict(row) for row in LOCKED_POINTS)
    if len(selected) != len(WORKLOADS) * len(METHODS) or len({key(row) for row in selected}) != len(selected):
        raise ValueError("tight paper selection must contain exactly one point per workload/method")
    return {
        "schema_version": 2,
        "experiment": "retrieve_workshop_tight_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "final",
        "selection_contract": "highest final median QPS with recall_min>=target and recall_median<=target+0.002",
        "decisions": decisions,
        "locked_points": [dict(row) for row in LOCKED_POINTS],
        "points": sorted(selected, key=lambda row: (WORKLOADS.index(str(row["workload"])), METHODS.index(str(row["method"])))),
    }


def pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    best_qps = -math.inf
    for row in sorted(rows, key=lambda item: (float(item["recall"]), float(item["qps"])), reverse=True):
        if float(row["qps"]) > best_qps:
            frontier.append(row)
            best_qps = float(row["qps"])
    return frontier


def deep_paths(rows: list[dict[str, object]], workload: str, method: str) -> set[tuple[int, int]]:
    local = [
        row
        for row in rows
        if row["workload"] == workload
        and row["method"] == method
        and int(row["max_iterations"]) == 0
    ]
    paths = {(int(row["itopk"]), int(row["search_width"])) for row in pareto(local)}
    paths.update((MAX_L, width) for width in WIDTHS)
    return paths


def next_points(result_root: Path) -> tuple[str, list[dict[str, object]]]:
    rows = calibration_rows(result_root)
    observed = measured_keys(rows)
    proposals: set[tuple[str, str, int, int, int]] = set()

    # Stage 1: normal B0 anchors and L refinement, independently for every categorical W.
    for workload in WORKLOADS:
        for method in METHODS:
            for width in WIDTHS:
                for itopk in ANCHOR_L:
                    candidate = (workload, method, itopk, width, 0)
                    if candidate not in observed:
                        proposals.add(candidate)
    if proposals:
        reason = "b0_anchors"
    else:
        for workload in WORKLOADS:
            goal = target(workload)
            for method in METHODS:
                for width in WIDTHS:
                    local = sorted(
                        (
                            row
                            for row in rows
                            if row["workload"] == workload
                            and row["method"] == method
                            and int(row["search_width"]) == width
                            and int(row["max_iterations"]) == 0
                        ),
                        key=lambda row: int(row["itopk"]),
                    )
                    for left, right in zip(local, local[1:]):
                        left_pass = float(left["recall"]) >= goal
                        right_pass = float(right["recall"]) >= goal
                        gap = int(right["itopk"]) - int(left["itopk"])
                        if left_pass != right_pass and gap > 1:
                            # Requested L influences B0 before CAGRA rounds its internal frontier
                            # capacity to a multiple of 32.  Refine requested integers and retain
                            # only points with a distinct (capacity, B0) execution fingerprint.
                            midpoint = (int(left["itopk"]) + int(right["itopk"])) // 2
                            existing_fingerprints = {
                                execution_fingerprint(workload, int(row["itopk"]), width)
                                for row in local
                            }
                            candidates = sorted(
                                range(int(left["itopk"]) + 1, int(right["itopk"])),
                                key=lambda value: (abs(value - midpoint), value),
                            )
                            for requested_l in candidates:
                                fingerprint = execution_fingerprint(workload, requested_l, width)
                                candidate = (workload, method, requested_l, width, 0)
                                if fingerprint not in existing_fingerprints and candidate not in observed:
                                    proposals.add(candidate)
                                    break
        reason = "b0_refinement"

    if proposals:
        return reason, [
            {
                "workload": workload,
                "method": method,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": maximum,
            }
            for workload, method, itopk, width, maximum in sorted(proposals)
        ]

    # Stage 2: explicit continuation is legal only when no B0 configuration reaches the target.
    for workload in WORKLOADS:
        goal = target(workload)
        for method in METHODS:
            b0 = [
                row
                for row in rows
                if row["workload"] == workload
                and row["method"] == method
                and int(row["max_iterations"]) == 0
            ]
            if any(float(row["recall"]) >= goal for row in b0):
                continue
            for itopk, width in sorted(deep_paths(rows, workload, method)):
                base_iterations = resolved_iterations(workload, itopk, width, 0)
                series = sorted(
                    (
                        row
                        for row in rows
                        if row["workload"] == workload
                        and row["method"] == method
                        and int(row["itopk"]) == itopk
                        and int(row["search_width"]) == width
                        and int(row["max_iterations"]) > 0
                    ),
                    key=lambda row: int(row["max_iterations"]),
                )
                passing = [row for row in series if float(row["recall"]) >= goal]
                if not series:
                    maximum = min(MAX_ITERATIONS, 2 * base_iterations)
                elif not passing:
                    previous = max(int(row["max_iterations"]) for row in series)
                    if previous >= MAX_ITERATIONS:
                        continue
                    maximum = min(MAX_ITERATIONS, 2 * previous)
                else:
                    upper = min(int(row["max_iterations"]) for row in passing)
                    failures = [
                        base_iterations,
                        *(
                            int(row["max_iterations"])
                            for row in series
                            if int(row["max_iterations"]) < upper
                            and float(row["recall"]) < goal
                        ),
                    ]
                    lower = max(failures)
                    if upper - lower <= 1:
                        continue
                    maximum = (lower + upper) // 2
                candidate = (workload, method, itopk, width, maximum)
                if candidate not in observed and hash_feasible(
                    workload, method, itopk, width, maximum
                ):
                    proposals.add(candidate)
    return "deep_refinement", [
        {
            "workload": workload,
            "method": method,
            "itopk": itopk,
            "search_width": width,
            "max_iterations": maximum,
        }
        for workload, method, itopk, width, maximum in sorted(proposals)
    ]


def candidate_selection(result_root: Path) -> dict:
    rows = calibration_rows(result_root)
    if not rows:
        raise ValueError("no calibration measurements")
    selected: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for workload in WORKLOADS:
        goal = target(workload)
        for method in METHODS:
            local = [row for row in rows if row["workload"] == workload and row["method"] == method]
            b0 = [row for row in local if int(row["max_iterations"]) == 0]
            b0_reaches = any(float(row["recall"]) >= goal for row in b0)
            eligible = b0 if b0_reaches else local
            qualifying = [row for row in eligible if float(row["recall"]) >= goal]
            if qualifying:
                within = [row for row in qualifying if float(row["recall"]) <= goal + TARGET_WINDOW]
                if within:
                    finalists = sorted(within, key=lambda row: float(row["qps"]), reverse=True)[:3]
                    rule = "fastest calibration QPS within target window"
                else:
                    finalists = sorted(
                        qualifying,
                        key=lambda row: (float(row["recall"]) - goal, -float(row["qps"])),
                    )[:3]
                    rule = "smallest measured overshoot"
            else:
                finalists = sorted(local, key=lambda row: (float(row["recall"]), float(row["qps"])), reverse=True)[:3]
                rule = "target not reached; rerun maximum-recall endpoints"
            unique: list[dict[str, object]] = []
            seen: set[tuple[str, str, int, int, int]] = set()
            for row in finalists:
                identity = key(row)
                if identity in seen:
                    continue
                seen.add(identity)
                point = {
                    "workload": workload,
                    "method": method,
                    "itopk": int(row["itopk"]),
                    "search_width": int(row["search_width"]),
                    "max_iterations": int(row["max_iterations"]),
                }
                unique.append(point)
                selected.append(point)
            decisions.append(
                {
                    "workload": workload,
                    "method": method,
                    "target_recall": goal,
                    "b0_reaches_target": b0_reaches,
                    "selection_rule": rule,
                    "calibration_candidates": unique,
                }
            )
    return {
        "schema_version": 1,
        "experiment": "retrieve_workshop_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_window": TARGET_WINDOW,
        "max_l": MAX_L,
        "widths": list(WIDTHS),
        "dataset_profile": profile_record(PROFILE),
        "max_iterations": MAX_ITERATIONS,
        "decisions": decisions,
        "points": selected,
    }


def summarize_final(result_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = [row for row in measurements(result_root) if row["stage"] == "final"]
    if not rows:
        raise ValueError("no final measurements")
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    summaries: list[dict[str, object]] = []
    for identity, members in sorted(grouped.items()):
        if sorted(int(row["repetition_index"]) for row in members) != [0, 1, 2]:
            raise ValueError(f"final point lacks repetitions 0,1,2: {identity}")
        first = members[0]
        recalls = [float(row["recall"]) for row in members]
        qps = [float(row["qps"]) for row in members]
        seconds = [float(row["seconds"]) for row in members]
        goal = target(str(first["workload"]))
        summaries.append(
            {
                "group": "matched_recall_final",
                "phase": "throughput",
                "workload": first["workload"],
                "graph_degree": first["graph_degree"],
                "intermediate_graph_degree": first["intermediate_graph_degree"],
                "method": first["method"],
                "itopk": first["itopk"],
                "search_width": first["search_width"],
                "max_iterations": first["max_iterations"],
                "resolved_iterations": first["resolved_iterations"],
                "repetitions": 3,
                "shards_per_repetition": first["shards"],
                "queries_per_repetition": first["queries"],
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "valid_gt_fraction_min": min(float(row["valid_gt_fraction"]) for row in members),
                "qps_median": statistics.median(qps),
                "qps_min": min(qps),
                "qps_max": max(qps),
                "seconds_median": statistics.median(seconds),
                "filter_violations": sum(float(row["filter_violations"]) for row in members),
                "sentinel_errors": sum(float(row["sentinel_errors"]) for row in members),
                "duplicate_output_query_rate_max": max(float(row["duplicate_output_query_rate"]) for row in members),
                "underfilled_queries_max": max(float(row["underfilled_queries"]) for row in members),
                "missing_result_slots_max": max(float(row["missing_result_slots"]) for row in members),
                "target_recall": goal,
                "target_reached": min(recalls) >= goal,
                "within_target_window": min(recalls) >= goal and statistics.median(recalls) <= goal + TARGET_WINDOW,
                "selected": False,
                "paper_included": True,
            }
        )

    selected: list[dict[str, object]] = []
    for workload in WORKLOADS:
        goal = target(workload)
        for method in METHODS:
            local = [row for row in summaries if row["workload"] == workload and row["method"] == method]
            qualifying = [row for row in local if bool(row["target_reached"])]
            within = [row for row in qualifying if bool(row["within_target_window"])]
            if within:
                chosen = max(within, key=lambda row: float(row["qps_median"]))
                rule = "fastest final median QPS within target window"
            elif qualifying:
                chosen = min(
                    qualifying,
                    key=lambda row: (float(row["recall_median"]) - goal, -float(row["qps_median"])),
                )
                rule = "smallest final measured overshoot"
            else:
                chosen = max(local, key=lambda row: (float(row["recall_median"]), float(row["qps_median"])))
                rule = "target not reached; maximum final recall"
            chosen["selected"] = True
            selected.append({**chosen, "selection_rule": rule})
    return summaries, selected


def validate_navix_refinement(result_root: Path) -> dict:
    """Require strict three-run EM/R NaviX target matches after refinement."""
    _, selected = summarize_final(result_root)
    output: list[dict[str, object]] = []
    for workload in NAVIX_REFINEMENT_WORKLOADS:
        local = [
            row
            for row in selected
            if row["workload"] == workload and row["method"] == "navix_reference"
        ]
        if len(local) != 1:
            raise ValueError(f"expected one selected NaviX point for {workload}")
        row = local[0]
        if (
            not bool(row["target_reached"])
            or not bool(row["within_target_window"])
            or int(row["itopk"]) > NAVIX_REFINEMENT_MAX_L
            or int(row["search_width"]) not in TUNING_WIDTHS
            or int(row["max_iterations"]) != 0
            or float(row["filter_violations"]) != 0
            or float(row["sentinel_errors"]) != 0
            or float(row["duplicate_output_query_rate_max"]) != 0
        ):
            raise ValueError(
                f"selected {workload} NaviX refinement is not a strict, correct B0 match: {row}"
            )
        output.append(row)
    return {
        "schema_version": 1,
        "experiment": "retrieve_workshop_navix_em_r_refinement",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": (
            "recall_min>=0.950, recall_median<=0.952, maximum final median QPS, "
            "zero filter/sentinel/duplicate-output errors"
        ),
        "selected": output,
    }


def provenance(result_root: Path, summaries: list[dict], selected: list[dict]) -> dict:
    run_path = result_root / "provenance" / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(run_path)
    run_payload = json.loads(run_path.read_text())
    provenance_max_queries = int(
        run_payload.get("fixed_contract", {}).get("max_queries", -1)
    )
    if provenance_max_queries != MAX_QUERIES:
        raise ValueError(
            "matched-recall provenance/config max_queries mismatch: "
            f"provenance={provenance_max_queries}, active_profile={MAX_QUERIES}"
        )
    manifests = [
        path
        for path, _ in experiment_manifests(result_root)
        if (result_root / "raw" / path.parent.parent.name / path.parent.name).is_dir()
    ]
    manifest_max_queries = {
        int(json.loads(path.read_text()).get("max_queries", -1)) for path in manifests
    }
    if manifest_max_queries != {MAX_QUERIES}:
        raise ValueError(
            "matched-recall manifests mix max_queries values: "
            f"{sorted(manifest_max_queries)}"
        )
    raw_files = sorted((result_root / "raw").glob("*/*/shard_*.json"))
    state_files = sorted((result_root / "state").glob("*.json"))
    analysis_files = [
        result_root / "analysis" / "measurements.csv",
        result_root / "analysis" / "final_summary.csv",
        result_root / "analysis" / "selected_points.csv",
    ]
    return {
        "schema_version": 1,
        "experiment": "retrieve_workshop_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "max_queries": MAX_QUERIES,
        "targets": TARGETS,
        "target_window": TARGET_WINDOW,
        "selection_contract": (
            "Use B0 only whenever any B0 configuration reaches the workload target. "
            "Otherwise allow explicit continuation through 7569 iterations where the L/W/depth "
            "combination fits SINGLE_CTA's 20-bit normal visited hash at 0.5 fill. Final selection "
            "requires recall_min>=target, prefers median recall<=target+0.002, and chooses maximum "
            "median QPS inside that window; no interpolation."
        ),
        "timing_contract": (
            "Complete cuVS-bench search call; YFCC executes five bitmap shards serially and "
            "QPS=10000/sum(shard seconds) within each repetition."
        ),
        "run_provenance": {"path": str(run_path.resolve()), "sha256": sha256(run_path)},
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "config_manifests": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in manifests],
        "raw_results": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in raw_files],
        "selection_state": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in state_files
        ],
        "analysis_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in analysis_files
        ],
        "summary_rows": len(summaries),
        "selected_rows": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    next_parser = subparsers.add_parser("next")
    next_parser.add_argument("--result-root", type=Path, required=True)
    next_parser.add_argument("--output", type=Path, required=True)

    generate = subparsers.add_parser("generate-group")
    generate.add_argument("--result-root", type=Path, required=True)
    generate.add_argument("--data-root", type=Path, required=True)
    generate.add_argument("--group", required=True)
    generate.add_argument("--repetitions", type=int, choices=(1, 3), required=True)
    generate.add_argument("--stage", choices=("calibration", "finalist", "final"))
    generate.add_argument("--points", type=Path, required=True)

    refine = subparsers.add_parser("tight-refinement-points")
    refine.add_argument("--baseline-measurements", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)

    finalist = subparsers.add_parser("tight-finalists")
    finalist.add_argument("--result-root", type=Path, required=True)
    finalist.add_argument("--output", type=Path, required=True)

    paper = subparsers.add_parser("tight-paper-points")
    paper.add_argument("--result-root", type=Path, required=True)
    paper.add_argument("--output", type=Path, required=True)

    navix_points = subparsers.add_parser("navix-refinement-points")
    navix_points.add_argument("--output", type=Path, required=True)

    navix_finalists = subparsers.add_parser("navix-refinement-finalists")
    navix_finalists.add_argument("--result-root", type=Path, required=True)
    navix_finalists.add_argument("--output", type=Path, required=True)

    navix_validate = subparsers.add_parser("validate-navix-refinement")
    navix_validate.add_argument("--result-root", type=Path, required=True)
    navix_validate.add_argument("--output", type=Path, required=True)

    raw_validate = subparsers.add_parser("validate-raw-output")
    raw_validate.add_argument("--raw", type=Path, required=True)
    raw_validate.add_argument("--config", type=Path, required=True)
    raw_validate.add_argument("--repetitions", type=int, choices=(1, 3), required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--result-root", type=Path, required=True)

    select = subparsers.add_parser("select-final")
    select.add_argument("--result-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--result-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "next":
        reason, points = next_points(args.result_root)
        payload = {"schema_version": 1, "reason": reason, "points": points}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"reason": reason, "points": len(points)}))
    elif args.command == "generate-group":
        payload = json.loads(args.points.read_text())
        write_group(
            result_root=args.result_root,
            data_root=args.data_root,
            group=args.group,
            repetitions=args.repetitions,
            points=list(payload["points"]),
            stage=args.stage,
        )
        print(json.dumps({"group": args.group, "points": len(payload["points"])}))
    elif args.command == "tight-refinement-points":
        payload = tight_refinement_points(args.baseline_measurements)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"points": len(payload["points"]), "brackets": len(payload["brackets"])}))
    elif args.command == "tight-finalists":
        payload = tight_finalists(args.result_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"points": len(payload["points"])}))
    elif args.command == "tight-paper-points":
        payload = tight_paper_points(args.result_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"points": len(payload["points"])}))
    elif args.command == "navix-refinement-points":
        payload = navix_refinement_points()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"points": len(payload["points"])}))
    elif args.command == "navix-refinement-finalists":
        payload = navix_refinement_finalists(args.result_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"points": len(payload["points"])}))
    elif args.command == "validate-navix-refinement":
        payload = validate_navix_refinement(args.result_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"selected": len(payload["selected"])}))
    elif args.command == "validate-raw-output":
        payload = validate_raw_output(args.raw, args.config, args.repetitions)
        print(json.dumps(payload))
    elif args.command == "analyze":
        rows = measurements(args.result_root)
        write_csv(args.result_root / "analysis" / "measurements.csv", MEASUREMENT_FIELDS, rows)
        print(json.dumps({"measurements": len(rows)}))
    elif args.command == "select-final":
        payload = candidate_selection(args.result_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"points": len(payload["points"])}))
    elif args.command == "finalize":
        rows = measurements(args.result_root)
        write_csv(args.result_root / "analysis" / "measurements.csv", MEASUREMENT_FIELDS, rows)
        summaries, selected = summarize_final(args.result_root)
        write_csv(args.result_root / "analysis" / "final_summary.csv", SUMMARY_FIELDS, summaries)
        selected_fields = SUMMARY_FIELDS + ("selection_rule",)
        write_csv(args.result_root / "analysis" / "selected_points.csv", selected_fields, selected)
        payload = provenance(args.result_root, summaries, selected)
        (args.result_root / "analysis" / "provenance.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"final_points": len(summaries), "selected": len(selected)}))


if __name__ == "__main__":
    main()
