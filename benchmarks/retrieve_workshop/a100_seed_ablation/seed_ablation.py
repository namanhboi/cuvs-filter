#!/usr/bin/env python3
"""Generate, calibrate, validate, and package the A100 NaviX seed-count ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_DIR = SCRIPT_DIR.parent / "gpu_graph"
sys.path.insert(0, str(GPU_DIR))

from generate_configs import dataset_paths, search_point, write_group

K = 100
MAX_QUERIES = 2048
TARGET = 0.800
TARGET_HIGH = 0.802
SEED_CAPS = (10, 100)
WIDTHS = (1, 2)
MIN_L = 100
MAX_L = 512
MAX_DEEP_ITERATIONS = 7569
EXPERIMENT = "retrieve_workshop_navix_seed_ablation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(
    path: Path, rows: list[dict[str, object]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def finite(row: dict, key: str, path: Path) -> float:
    if key not in row:
        raise ValueError(f"missing {key} in {path}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {path}")
    return value


def navix_point(
    itopk: int,
    width: int,
    iterations: int,
    seed_cap: int,
    *,
    diagnostic: dict[str, str] | None = None,
) -> dict[str, object]:
    if seed_cap not in SEED_CAPS:
        raise ValueError(f"unsupported seed cap {seed_cap}")
    row = search_point(
        "navix_reference",
        itopk,
        width,
        iterations,
        k=K,
        max_queries=MAX_QUERIES,
    )
    row["navix_seed_cap"] = seed_cap
    if diagnostic:
        row.update(diagnostic)
    return row


def identity(row: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(row["itopk"]),
        int(row["search_width"]),
        int(row["max_iterations"]),
    )


def group_manifest(root: Path, group: str) -> Path:
    return root / "configs" / group / "yfcc" / "manifest.json"


def create_group(
    root: Path,
    data_root: Path,
    group: str,
    phase: str,
    seed_cap: int,
    points: list[dict[str, object]],
) -> None:
    manifest_path = group_manifest(root, group)
    expected = {identity(row) for row in points}
    if len(expected) != len(points):
        raise ValueError(f"duplicate point in {group}")
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        observed = {
            (
                int(row["itopk"]),
                int(row["search_width"]),
                int(row["max_iterations"]),
            )
            for row in existing["search_points"]
        }
        if (
            existing.get("experiment") != EXPERIMENT
            or int(existing.get("navix_seed_cap", -1)) != seed_cap
            or observed != expected
        ):
            raise ValueError(f"immutable group contract changed: {group}")
        return

    searches = [
        navix_point(
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
            seed_cap,
            diagnostic=row.get("diagnostic"),
        )
        for row in points
    ]
    write_group(
        root / "configs",
        data_root,
        group=group,
        workload="yfcc",
        phase=phase,
        searches=searches,
        yfcc_graph_degree=64,
        k=K,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "ablation_schema_version": 1,
            "navix_seed_cap": seed_cap,
            "target_recall_low": TARGET,
            "target_recall_high": TARGET_HIGH,
        }
    )
    for row in manifest["search_points"]:
        row["navix_seed_cap"] = seed_cap
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def selected_reference(path: Path) -> dict[str, object]:
    rows = [
        row
        for row in read_csv(path)
        if row.get("workload") == "yfcc"
        and row.get("method") == "navix_reference"
        and str(row.get("selected", "true")).lower() in {"1", "true", "yes"}
    ]
    if len(rows) != 1:
        raise ValueError(f"require one selected YFCC NaviX point in {path}")
    row = rows[0]
    return {
        "itopk": int(row["itopk"]),
        "search_width": int(row["search_width"]),
        "max_iterations": int(row["max_iterations"]),
        "recall": float(row["recall_median"]),
        "qps": float(row["qps_median"]),
    }


def initialize(
    root: Path,
    data_root: Path,
    reference_selected: Path,
    reference_summary: Path,
    reference_provenance: Path,
) -> None:
    for path in (reference_selected, reference_summary, reference_provenance):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = selected_reference(reference_selected)
    if identity(reference) != (100, 1, 56):
        raise ValueError(
            "the frozen seed=100 control must be the selected YFCC point L=100,W=1,I=56"
        )
    provenance = json.loads(reference_provenance.read_text())
    if (
        int(provenance.get("k", -1)) != K
        or int(provenance.get("max_queries", -1)) != MAX_QUERIES
    ):
        raise ValueError(
            "reference matched-recall provenance is not k=100/max_queries=2048"
        )

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "k": K,
        "max_queries": MAX_QUERIES,
        "dataset": "YFCC-10M",
        "graph_degree": 64,
        "target_window": [TARGET, TARGET_HIGH],
        "seed_caps": list(SEED_CAPS),
        "reference": {
            "selected_csv": str(reference_selected.resolve()),
            "selected_sha256": sha256(reference_selected),
            "summary_csv": str(reference_summary.resolve()),
            "summary_sha256": sha256(reference_summary),
            "provenance_json": str(reference_provenance.resolve()),
            "provenance_sha256": sha256(reference_provenance),
            "selected_point": reference,
        },
        "calibration_policy": {
            "b0": "L in [100,512], W in {1,2}; integer binary search after anchors",
            "deep_fallback": (
                "only if neither W reaches 0.800 at L=512 with max_iterations=0; "
                "then L=512,W=2 with bounded doubling and integer binary search"
            ),
            "maximum_deep_iterations": MAX_DEEP_ITERATIONS,
        },
    }
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    contract_path = state / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text())
        if existing != contract:
            raise ValueError(
                "seed-ablation contract changed inside immutable result root"
            )
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    control = [{"itopk": 100, "search_width": 1, "max_iterations": 56}]
    create_group(
        root, data_root, "correctness_s10", "correctness", 10, control
    )
    create_group(
        root, data_root, "correctness_s100", "correctness", 100, control
    )
    create_group(root, data_root, "control_s100", "throughput", 100, control)
    anchors = [
        {"itopk": itopk, "search_width": width, "max_iterations": 0}
        for itopk in (128, 256, 512)
        for width in WIDTHS
    ]
    create_group(root, data_root, "anchors_s10", "throughput", 10, anchors)
    print(root / "state" / "contract.json")


RAW_FIELDS = [
    "group",
    "phase",
    "seed_cap",
    "itopk",
    "search_width",
    "max_iterations",
    "repetition_index",
    "shard_index",
    "queries",
    "recall",
    "valid_gt_fraction",
    "qps",
    "seconds",
    "filter_violations",
    "sentinel_errors",
    "sentinel_order_errors",
    "invalid_sentinel_distance_errors",
    "duplicate_output_queries",
    "underfilled_queries",
    "missing_result_slots",
    "source_file",
]


def load_group(root: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != EXPERIMENT:
        return []
    phase = str(manifest["phase"])
    group = str(manifest["group"])
    cap = int(manifest["navix_seed_cap"])
    repetitions = int(manifest["repetitions"])
    expected = {
        (
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
        )
        for row in manifest["search_points"]
    }
    raw_dir = root / "raw" / group / "yfcc"
    expected_files = {
        f"shard_{int(row['shard_index']):02d}.json"
        for row in manifest["configs"]
    }
    observed_files = {path.name for path in raw_dir.glob("shard_*.json")}
    if observed_files != expected_files:
        raise ValueError(
            f"incomplete {group}: missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )

    output: list[dict[str, object]] = []
    for shard in manifest["configs"]:
        shard_index = int(shard["shard_index"])
        query_count = int(shard["query_count"])
        raw_path = raw_dir / f"shard_{shard_index:02d}.json"
        payload = json.loads(raw_path.read_text())
        seen = {repetition: set() for repetition in range(repetitions)}
        for record in payload.get("benchmarks", []):
            if record.get("error_occurred") or record.get("skipped"):
                raise ValueError(f"benchmark failure in {raw_path}: {record}")
            if record.get("run_type") != "iteration":
                continue
            repetition = int(record.get("repetition_index", -1))
            if repetition not in seen:
                raise ValueError(f"invalid repetition in {raw_path}")
            label = str(record.get("label", ""))
            if (
                label_value(label, "bitmap_method") != "navix_reference"
                or label_value(label, "algo") != "single_cta"
                or label_value(label, "navix_mode") != "adaptive_kuzu"
                or label_value(label, "navix_scheduler") != "tiled"
            ):
                raise ValueError(
                    f"wrong NaviX execution path in {raw_path}: {label}"
                )
            point = (
                round(finite(record, "itopk", raw_path)),
                round(finite(record, "search_width", raw_path)),
                round(finite(record, "max_iterations", raw_path)),
            )
            if point not in expected or point in seen[repetition]:
                raise ValueError(
                    f"unexpected/duplicate point {point} in {raw_path}"
                )
            seen[repetition].add(point)
            if (
                round(finite(record, "k", raw_path)) != K
                or round(finite(record, "max_queries", raw_path))
                != MAX_QUERIES
                or round(finite(record, "navix_bitmap_seeds", raw_path)) != 1
                or round(finite(record, "navix_seed_cap", raw_path)) != cap
                or round(finite(record, "n_queries", raw_path)) != query_count
            ):
                raise ValueError(
                    f"seed-width/runtime contract failed in {raw_path}"
                )
            recall = finite(record, "ValidGTRecall", raw_path)
            valid_fraction = finite(record, "ValidGTFraction", raw_path)
            qps = finite(record, "items_per_second", raw_path)
            if not (0 <= recall <= 1 and 0 < valid_fraction <= 1 and qps > 0):
                raise ValueError(f"invalid measurement in {raw_path}")
            error_values = {
                "filter_violations": finite(
                    record, "FilterViolations", raw_path
                ),
                "sentinel_errors": finite(
                    record, "InvalidSentinelErrors", raw_path
                ),
                "sentinel_order_errors": finite(
                    record, "SentinelOrderErrors", raw_path
                ),
                "invalid_sentinel_distance_errors": finite(
                    record, "InvalidSentinelDistanceErrors", raw_path
                ),
                "duplicate_output_queries": finite(
                    record, "DuplicateOutputQueries", raw_path
                ),
            }
            if any(value != 0 for value in error_values.values()):
                raise ValueError(
                    f"NaviX correctness failure in {raw_path}: {error_values}"
                )
            output.append(
                {
                    "group": group,
                    "phase": phase,
                    "seed_cap": cap,
                    "itopk": point[0],
                    "search_width": point[1],
                    "max_iterations": point[2],
                    "repetition_index": repetition,
                    "shard_index": shard_index,
                    "queries": query_count,
                    "recall": recall,
                    "valid_gt_fraction": valid_fraction,
                    "qps": qps,
                    "seconds": query_count / qps,
                    **error_values,
                    "underfilled_queries": finite(
                        record, "UnderfilledQueries", raw_path
                    ),
                    "missing_result_slots": finite(
                        record, "MissingResultSlots", raw_path
                    ),
                    "source_file": str(raw_path.resolve()),
                }
            )
        for repetition, points in seen.items():
            if points != expected:
                raise ValueError(
                    f"incomplete repetition {repetition} in {raw_path}: "
                    f"missing={sorted(expected - points)}"
                )
    return output


def load_raw(root: Path) -> list[dict[str, object]]:
    manifests = sorted((root / "configs").glob("*/yfcc/manifest.json"))
    if not manifests:
        raise ValueError(f"no seed-ablation manifests under {root}")
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        raw_dir = root / "raw" / manifest.parent.parent.name / "yfcc"
        if not raw_dir.is_dir():
            raise ValueError(f"missing raw group for {manifest}")
        rows.extend(load_group(root, manifest))
    return rows


SUMMARY_FIELDS = [
    "group",
    "phase",
    "seed_cap",
    "itopk",
    "search_width",
    "max_iterations",
    "repetitions",
    "queries_per_repetition",
    "recall_median",
    "recall_min",
    "recall_max",
    "qps_median",
    "qps_min",
    "qps_max",
    "seconds_median",
    "filter_violations",
    "sentinel_errors",
    "sentinel_order_errors",
    "invalid_sentinel_distance_errors",
    "duplicate_output_queries",
    "underfilled_queries_max",
    "missing_result_slots_max",
]


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    repetitions: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in raw:
        key = (
            row["group"],
            row["phase"],
            row["seed_cap"],
            row["itopk"],
            row["search_width"],
            row["max_iterations"],
            row["repetition_index"],
        )
        repetitions.setdefault(key, []).append(row)
    combined: list[dict[str, object]] = []
    for key, rows in repetitions.items():
        group, phase, cap, itopk, width, iterations, repetition = key
        queries = sum(int(row["queries"]) for row in rows)
        seconds = sum(float(row["seconds"]) for row in rows)
        valid_slots = sum(
            float(row["valid_gt_fraction"]) * int(row["queries"]) * K
            for row in rows
        )
        matches = sum(
            float(row["recall"])
            * float(row["valid_gt_fraction"])
            * int(row["queries"])
            * K
            for row in rows
        )
        expected_queries = 1_000 if phase == "correctness" else 10_000
        expected_shards = 1 if phase == "correctness" else 5
        if queries != expected_queries or len(rows) != expected_shards:
            raise ValueError(
                f"incomplete serial aggregate for {group}, repetition {repetition}"
            )
        combined.append(
            {
                "group": group,
                "phase": phase,
                "seed_cap": cap,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
                "repetition_index": repetition,
                "queries": queries,
                "recall": matches / valid_slots,
                "qps": queries / seconds,
                "seconds": seconds,
                **{
                    field: sum(float(row[field]) for row in rows)
                    for field in (
                        "filter_violations",
                        "sentinel_errors",
                        "sentinel_order_errors",
                        "invalid_sentinel_distance_errors",
                        "duplicate_output_queries",
                    )
                },
                "underfilled_queries": max(
                    float(row["underfilled_queries"]) for row in rows
                ),
                "missing_result_slots": max(
                    float(row["missing_result_slots"]) for row in rows
                ),
            }
        )

    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in combined:
        key = (
            row["group"],
            row["phase"],
            row["seed_cap"],
            row["itopk"],
            row["search_width"],
            row["max_iterations"],
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        group, phase, cap, itopk, width, iterations = key
        recalls = [float(row["recall"]) for row in rows]
        qps_values = [float(row["qps"]) for row in rows]
        seconds = [float(row["seconds"]) for row in rows]
        output.append(
            {
                "group": group,
                "phase": phase,
                "seed_cap": cap,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
                "repetitions": len(rows),
                "queries_per_repetition": int(rows[0]["queries"]),
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "qps_median": statistics.median(qps_values),
                "qps_min": min(qps_values),
                "qps_max": max(qps_values),
                "seconds_median": statistics.median(seconds),
                **{
                    field: sum(float(row[field]) for row in rows)
                    for field in (
                        "filter_violations",
                        "sentinel_errors",
                        "sentinel_order_errors",
                        "invalid_sentinel_distance_errors",
                        "duplicate_output_queries",
                    )
                },
                "underfilled_queries_max": max(
                    float(row["underfilled_queries"]) for row in rows
                ),
                "missing_result_slots_max": max(
                    float(row["missing_result_slots"]) for row in rows
                ),
            }
        )
    return output


def measured_identities(
    rows: list[dict[str, object]],
) -> set[tuple[int, int, int]]:
    return {
        (
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
        )
        for row in rows
        if int(row["seed_cap"]) == 10 and row["phase"] == "throughput"
    }


def in_window(row: dict[str, object]) -> bool:
    return (
        float(row["recall_min"]) >= TARGET
        and float(row["recall_median"]) <= TARGET_HIGH
    )


def next_integer(
    rows: list[dict[str, object]],
    coordinate: str,
    minimum: int,
    maximum: int,
) -> list[int]:
    values = sorted({int(row[coordinate]) for row in rows})
    by_value = {int(row[coordinate]): row for row in rows}
    window = [value for value in values if in_window(by_value[value])]
    if window:
        # Once any in-window point has both immediate neighbors measured, its local
        # neighborhood is resolved.  Do not let a newly measured in-window neighbor
        # recursively grow a linear sweep across the entire target band.
        if any(
            all(
                neighbor < minimum
                or neighbor > maximum
                or neighbor in by_value
                for neighbor in (value - 1, value + 1)
            )
            for value in window
        ):
            return []
        center = min(
            window,
            key=lambda value: (
                abs(float(by_value[value]["recall_median"]) - TARGET),
                value,
            ),
        )
        candidates = {
            neighbor
            for neighbor in (center - 1, center + 1)
            if minimum <= neighbor <= maximum and neighbor not in by_value
        }
        return sorted(candidates)
    adjacent = []
    for left, right in pairwise(values):
        if (
            float(by_value[left]["recall_min"]) < TARGET
            and float(by_value[right]["recall_min"]) >= TARGET
        ):
            adjacent.append((right - left, left, right))
    if adjacent:
        _, left, right = min(adjacent)
        return [(left + right) // 2] if right - left > 1 else []
    if all(float(by_value[value]["recall_min"]) >= TARGET for value in values):
        low = min(values)
        if low == minimum:
            return []
        return [(minimum + low) // 2]
    if all(float(by_value[value]["recall_min"]) < TARGET for value in values):
        high = max(values)
        if high == maximum:
            return []
        return [(high + maximum + 1) // 2]
    return []


def plan_next(root: Path, data_root: Path) -> None:
    summaries = summarize(load_raw(root))
    throughput = [
        row
        for row in summaries
        if row["phase"] == "throughput" and int(row["seed_cap"]) == 10
    ]
    measured = measured_identities(throughput)
    points: list[dict[str, object]] = []
    for width in WIDTHS:
        local = [
            row
            for row in throughput
            if int(row["search_width"]) == width
            and int(row["max_iterations"]) == 0
        ]
        if not local:
            continue
        for itopk in next_integer(local, "itopk", MIN_L, MAX_L):
            point = {
                "itopk": itopk,
                "search_width": width,
                "max_iterations": 0,
            }
            if identity(point) not in measured:
                points.append(point)

    reason = "continue B0 integer search"
    if not points:
        b0_reached = any(
            int(row["max_iterations"]) == 0
            and float(row["recall_min"]) >= TARGET
            for row in throughput
        )
        if b0_reached:
            reason = "B0 reaches target; deep fallback forbidden"
        else:
            deep = [
                row
                for row in throughput
                if int(row["itopk"]) == MAX_L
                and int(row["search_width"]) == 2
                and int(row["max_iterations"]) > 0
            ]
            if not deep:
                next_iterations = [522]
            elif all(float(row["recall_min"]) < TARGET for row in deep):
                largest = max(int(row["max_iterations"]) for row in deep)
                next_iterations = (
                    [min(largest * 2, MAX_DEEP_ITERATIONS)]
                    if largest < MAX_DEEP_ITERATIONS
                    else []
                )
            else:
                next_iterations = next_integer(
                    deep, "max_iterations", 1, MAX_DEEP_ITERATIONS
                )
            for iterations in next_iterations:
                point = {
                    "itopk": MAX_L,
                    "search_width": 2,
                    "max_iterations": iterations,
                }
                if identity(point) not in measured:
                    points.append(point)
            reason = (
                "L=512 B0 cannot reach target; continue bounded deep fallback"
                if points
                else "deep fallback exhausted"
            )

    existing_rounds = sorted(
        (root / "configs").glob("calibration_r*/yfcc/manifest.json")
    )
    group = f"calibration_r{len(existing_rounds):02d}"
    if points:
        create_group(root, data_root, group, "throughput", 10, points)
    payload = {
        "schema_version": 1,
        "complete": not points,
        "reason": reason,
        "next_group": group if points else None,
        "next_points": points,
        "measured_points": len(throughput),
    }
    state = root / "state" / "calibration_state.json"
    state.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def select_point(rows: list[dict[str, object]], cap: int) -> dict[str, object]:
    local = [
        row
        for row in rows
        if row["phase"] == "throughput" and int(row["seed_cap"]) == cap
    ]
    if not local:
        raise ValueError(f"no throughput points for seed cap {cap}")
    within = [row for row in local if in_window(row)]
    if within:
        selected = max(within, key=lambda row: float(row["qps_median"]))
        status = "matched"
    else:
        reached = [row for row in local if float(row["recall_min"]) >= TARGET]
        if reached:
            selected = min(
                reached,
                key=lambda row: (
                    float(row["recall_median"]) - TARGET,
                    -float(row["qps_median"]),
                ),
            )
            status = "nearest_overshoot"
        else:
            selected = max(
                local,
                key=lambda row: (
                    float(row["recall_median"]),
                    float(row["qps_median"]),
                ),
            )
            status = "unreached"
    return {**selected, "status": status, "selected": True}


def qps_label(qps: float) -> str:
    return f"{qps / 1000:.1f}K" if qps >= 1000 else f"{qps:.0f}"


def diagnostic_summary(root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for cap in SEED_CAPS:
        directory = root / "diagnostics" / f"seed_{cap}"
        if not directory.is_dir():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = read_csv(directory / "query_summary.csv")
        if int(manifest.get("topk", -1)) != K or len(rows) != 1_000:
            raise ValueError(f"invalid seed diagnostic for cap {cap}")
        counts = [int(row["navix_seed_count"]) for row in rows]
        inspected = [int(row["seed_inspected_units"]) for row in rows]
        if any(count < 0 or count > cap for count in counts):
            raise ValueError(f"diagnostic seed count exceeds cap {cap}")
        output.append(
            {
                "seed_cap": cap,
                "queries": len(rows),
                "seed_count_mean": statistics.mean(counts),
                "seed_count_min": min(counts),
                "seed_count_max": max(counts),
                "queries_at_cap_fraction": sum(
                    count == cap for count in counts
                )
                / len(counts),
                "bitmap_words_inspected_mean": statistics.mean(inspected),
                "bitmap_words_inspected_min": min(inspected),
                "bitmap_words_inspected_max": max(inspected),
            }
        )
    return output


def reference_b0_points(contract: dict[str, object]) -> list[dict[str, float]]:
    path = Path(str(contract["reference"]["summary_csv"]))
    points = []
    for row in read_csv(path):
        if (
            row.get("phase") == "throughput"
            and row.get("workload") == "yfcc"
            and row.get("method") == "navix_reference"
            and row.get("group") == "b0"
            and int(row["max_iterations"]) == 0
        ):
            points.append(
                {
                    "recall": float(row["recall_median"]),
                    "qps": float(row["qps_median"]),
                }
            )
    if not points:
        raise ValueError(f"no frozen seed=100 YFCC B0 points in {path}")
    return points


def analyze(root: Path, require_diagnostics: bool) -> None:
    raw = load_raw(root)
    summaries = summarize(raw)
    correctness_caps = {
        int(row["seed_cap"])
        for row in summaries
        if row["phase"] == "correctness"
    }
    if correctness_caps != set(SEED_CAPS):
        raise ValueError(f"missing correctness gates: {correctness_caps}")
    selected = [select_point(summaries, cap) for cap in SEED_CAPS]
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "raw_points.csv", raw, RAW_FIELDS)
    write_csv(analysis / "measurements.csv", summaries, SUMMARY_FIELDS)
    selected_fields = SUMMARY_FIELDS + ["status", "selected"]
    write_csv(analysis / "selected_points.csv", selected, selected_fields)

    contract = json.loads((root / "state" / "contract.json").read_text())
    frozen_b0 = reference_b0_points(contract)
    reference = contract["reference"]["selected_point"]
    control = next(row for row in selected if int(row["seed_cap"]) == 100)
    control_recall_delta = float(control["recall_median"]) - float(
        reference["recall"]
    )
    control_qps_ratio = float(control["qps_median"]) / float(reference["qps"])
    if abs(control_recall_delta) > 1e-9:
        raise ValueError(
            f"seed=100 control recall changed from frozen reference: {control_recall_delta}"
        )
    if not 0.85 <= control_qps_ratio <= 1.15:
        raise ValueError(
            f"seed=100 control QPS drift is too large: {control_qps_ratio}"
        )

    diagnostics = diagnostic_summary(root)
    if require_diagnostics and {
        int(row["seed_cap"]) for row in diagnostics
    } != set(SEED_CAPS):
        raise ValueError("both seed-count diagnostic captures are required")
    if diagnostics:
        write_csv(
            analysis / "seed_prepass_diagnostics.csv",
            diagnostics,
            list(diagnostics[0]),
        )

    selected_by_cap = {int(row["seed_cap"]): row for row in selected}
    rows_tex = [
        "% Generated by a100_seed_ablation/seed_ablation.py; do not edit.",
        r"\newcommand{\YFCCSeedAblationRows}{%",
    ]
    for cap in SEED_CAPS:
        row = selected_by_cap[cap]
        recall = float(row["recall_median"])
        status = str(row["status"])
        recall_text = (
            f"{recall:.4f}" if status == "matched" else f"max {recall:.4f}"
        )
        rows_tex.append(
            f"{cap} & {qps_label(float(row['qps_median']))} & {recall_text} \\\\%"
        )
    rows_tex.append("}")
    (analysis / "yfcc_seed_ablation_results.tex").write_text(
        "\n".join(rows_tex) + "\n"
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for cap, color, marker in ((10, "#1f77b4", "o"), (100, "#d62728", "s")):
        if cap == 100:
            local = [
                {"recall_median": row["recall"], "qps_median": row["qps"]}
                for row in frozen_b0
            ] + [
                row
                for row in summaries
                if row["phase"] == "throughput" and int(row["seed_cap"]) == cap
            ]
        else:
            local = [
                row
                for row in summaries
                if row["phase"] == "throughput" and int(row["seed_cap"]) == cap
            ]
        axis.scatter(
            [float(row["recall_median"]) for row in local],
            [float(row["qps_median"]) for row in local],
            color=color,
            marker=marker,
            label=f"{cap} passing seeds",
            alpha=0.85,
        )
    axis.axvspan(TARGET, TARGET_HIGH, color="#777777", alpha=0.10)
    axis.set_xlabel("Recall@100")
    axis.set_ylabel("Queries per second")
    axis.set_title("YFCC-10M NaviX seed-count ablation")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    for extension in ("pdf", "png"):
        figure.savefig(analysis / f"yfcc_seed_ablation.{extension}", dpi=220)
    plt.close(figure)

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "k": K,
        "max_queries": MAX_QUERIES,
        "target_window": [TARGET, TARGET_HIGH],
        "selected": selected,
        "seed_100_reference_check": {
            "recall_delta": control_recall_delta,
            "qps_ratio": control_qps_ratio,
            "frozen_b0_points": frozen_b0,
        },
        "diagnostics": diagnostics,
    }
    (analysis / "seed_ablation_results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    provenance = {
        **contract,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted((root / "raw").glob("*/yfcc/shard_*.json"))
        ],
        "selected": selected,
        "diagnostics": diagnostics,
    }
    (analysis / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        json.dumps({"measurements": len(summaries), "selected": len(selected)})
    )


def generate_diagnostics(root: Path, data_root: Path) -> None:
    selected_path = root / "analysis" / "selected_points.csv"
    if not selected_path.is_file():
        raise FileNotFoundError("run analyze before generating diagnostics")
    selected = {int(row["seed_cap"]): row for row in read_csv(selected_path)}
    paths = dataset_paths(data_root, "yfcc", "correctness", 64)
    source = json.loads(paths.manifest.read_text())
    shard = source["shards"][0]
    groundtruth = str(
        (Path(shard["directory"]) / "groundtruth.ibin").resolve()
    )
    for cap in SEED_CAPS:
        row = selected[cap]
        variant = f"seed_{cap}"
        diagnostic = {
            "favor_diagnostics_output": str(
                (root / "diagnostics" / variant).resolve()
            ),
            "favor_diagnostics_groundtruth": groundtruth,
            "favor_diagnostics_dataset": "yfcc10m-a100-k100-seed-ablation",
            "favor_diagnostics_variant": variant,
        }
        point = {
            "itopk": int(row["itopk"]),
            "search_width": int(row["search_width"]),
            "max_iterations": int(row["max_iterations"]),
            "diagnostic": diagnostic,
        }
        create_group(
            root, data_root, f"diagnostics_s{cap}", "correctness", cap, [point]
        )


def bundle(root: Path) -> None:
    analysis = root / "analysis"
    if not (analysis / "seed_ablation_results.json").is_file():
        raise FileNotFoundError("analyze the seed ablation before bundling")
    destination = root / "paper_gpu_bundle_seed_ablation"
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "analysis").mkdir(parents=True)
    for path in analysis.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / "analysis" / path.name)
    shutil.copytree(root / "state", destination / "state")
    if (root / "provenance").is_dir():
        shutil.copytree(root / "provenance", destination / "provenance")
    for cap in SEED_CAPS:
        source = root / "diagnostics" / f"seed_{cap}"
        if source.is_dir():
            target = destination / "diagnostics" / f"seed_{cap}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.json", "query_summary.csv"):
                shutil.copy2(source / name, target / name)
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(destination)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (destination / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n"
    )
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("--root", type=Path, required=True)
    initialize_parser.add_argument("--data-root", type=Path, required=True)
    initialize_parser.add_argument(
        "--reference-selected", type=Path, required=True
    )
    initialize_parser.add_argument(
        "--reference-summary", type=Path, required=True
    )
    initialize_parser.add_argument(
        "--reference-provenance", type=Path, required=True
    )
    next_parser = subparsers.add_parser("plan-next")
    next_parser.add_argument("--root", type=Path, required=True)
    next_parser.add_argument("--data-root", type=Path, required=True)
    diagnostics_parser = subparsers.add_parser("generate-diagnostics")
    diagnostics_parser.add_argument("--root", type=Path, required=True)
    diagnostics_parser.add_argument("--data-root", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--root", type=Path, required=True)
    analyze_parser.add_argument("--require-diagnostics", action="store_true")
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize(
            args.root.resolve(),
            args.data_root.resolve(),
            args.reference_selected.resolve(),
            args.reference_summary.resolve(),
            args.reference_provenance.resolve(),
        )
    elif args.command == "plan-next":
        plan_next(args.root.resolve(), args.data_root.resolve())
    elif args.command == "generate-diagnostics":
        generate_diagnostics(args.root.resolve(), args.data_root.resolve())
    elif args.command == "analyze":
        analyze(args.root.resolve(), args.require_diagnostics)
    else:
        bundle(args.root.resolve())


if __name__ == "__main__":
    main()
