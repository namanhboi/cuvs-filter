#!/usr/bin/env python3
"""Run and analyze the k=10 YFCC NaviX k-seed versus W*D-seed ablation."""

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

K = 10
MAX_QUERIES = 2048
GRAPH_DEGREE = 64
TARGET_LOW = 0.800
TARGET_HIGH = 0.802
WIDTHS = (1, 2)
ANCHORS = (10, 32, 64, 128, 256, 512)
MAX_L = 512
MAX_DEEP_ITERATIONS = 7569
REFERENCE_POINT = (129, 2, 0)
EXPERIMENT = "retrieve_workshop_navix_k10_wd_seed_ablation"
POLICY_K = "k_seed"
POLICY_WD = "wd_seed"


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


def finite(row: dict, key: str, path: Path) -> float:
    if key not in row:
        raise ValueError(f"missing {key} in {path}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {path}")
    return value


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def seed_cap(policy: str, width: int) -> int:
    if policy == POLICY_K:
        return K
    if policy == POLICY_WD:
        return width * GRAPH_DEGREE
    raise ValueError(f"unknown seed policy {policy!r}")


def point(
    policy: str,
    itopk: int,
    width: int,
    iterations: int,
    *,
    diagnostic: dict[str, str] | None = None,
) -> dict[str, object]:
    if not K <= itopk <= MAX_L:
        raise ValueError(f"L must be in [{K},{MAX_L}], got {itopk}")
    if width not in WIDTHS:
        raise ValueError(f"unsupported search width {width}")
    row = search_point(
        "navix_reference",
        itopk,
        width,
        iterations,
        k=K,
        max_queries=MAX_QUERIES,
    )
    row["navix_seed_cap"] = seed_cap(policy, width)
    if diagnostic:
        row.update(diagnostic)
    return row


def identity(row: dict[str, object]) -> tuple[str, int, int, int, int]:
    policy = str(row["seed_policy"])
    width = int(row["search_width"])
    return (
        policy,
        int(row["itopk"]),
        width,
        int(row["max_iterations"]),
        seed_cap(policy, width),
    )


def group_manifest(root: Path, group: str) -> Path:
    return root / "configs" / group / "yfcc" / "manifest.json"


def create_group(
    root: Path,
    data_root: Path,
    group: str,
    phase: str,
    points: list[dict[str, object]],
    repetitions: int,
) -> None:
    expected = {identity(row) for row in points}
    if len(expected) != len(points):
        raise ValueError(f"duplicate point in {group}")
    manifest_path = group_manifest(root, group)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        observed = {
            (
                str(row["seed_policy"]),
                int(row["itopk"]),
                int(row["search_width"]),
                int(row["max_iterations"]),
                int(row["navix_seed_cap"]),
            )
            for row in manifest["search_points"]
        }
        if (
            manifest.get("experiment") != EXPERIMENT
            or int(manifest["repetitions"]) != repetitions
            or observed != expected
        ):
            raise ValueError(f"immutable group contract changed: {group}")
        return

    searches = [
        point(
            str(row["seed_policy"]),
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
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
        yfcc_graph_degree=GRAPH_DEGREE,
        k=K,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "ablation_schema_version": 1,
            "repetitions": repetitions,
            "target_recall_low": TARGET_LOW,
            "target_recall_high": TARGET_HIGH,
            "seed_policy_contract": {
                POLICY_K: "up to k passing bitmap seeds",
                POLICY_WD: "up to search_width * graph_degree passing bitmap seeds",
            },
        }
    )
    manifest["search_points"] = [
        {
            "method": "navix_reference",
            "seed_policy": str(row["seed_policy"]),
            "navix_seed_cap": seed_cap(
                str(row["seed_policy"]), int(row["search_width"])
            ),
            "itopk": int(row["itopk"]),
            "search_width": int(row["search_width"]),
            "max_iterations": int(row["max_iterations"]),
        }
        for row in points
    ]
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
    result = {
        "itopk": int(row["itopk"]),
        "search_width": int(row["search_width"]),
        "max_iterations": int(row["max_iterations"]),
        "recall": float(row["recall_median"]),
        "recall_min": float(row["recall_min"]),
        "qps": float(row["qps_median"]),
    }
    if (
        result["itopk"],
        result["search_width"],
        result["max_iterations"],
    ) != REFERENCE_POINT:
        raise ValueError(
            "frozen k=10 YFCC NaviX reference must be L=129,W=2,max_iterations=0"
        )
    if not TARGET_LOW <= float(result["recall"]) <= TARGET_HIGH:
        raise ValueError(
            "frozen reference is outside the target-recall window"
        )
    return result


def initialize(
    root: Path,
    data_root: Path,
    reference_selected: Path,
    reference_provenance: Path,
) -> None:
    for path in (reference_selected, reference_provenance):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = selected_reference(reference_selected)
    provenance = json.loads(reference_provenance.read_text())
    if int(provenance.get("max_queries", -1)) != MAX_QUERIES:
        raise ValueError("reference provenance is not max_queries=2048")
    if "k" in provenance and int(provenance["k"]) != K:
        raise ValueError("reference provenance is not k=10")
    if "k" not in provenance and (
        provenance.get("experiment") != "retrieve_workshop_matched_recall"
        or float(provenance.get("targets", {}).get("yfcc", -1)) != TARGET_LOW
    ):
        raise ValueError(
            "reference provenance does not identify the k=10 matched-recall study"
        )

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "k": K,
        "max_queries": MAX_QUERIES,
        "dataset": "YFCC-10M",
        "graph_degree": GRAPH_DEGREE,
        "target_window": [TARGET_LOW, TARGET_HIGH],
        "policies": {
            POLICY_K: {"cap": K},
            POLICY_WD: {"cap": "search_width * graph_degree"},
        },
        "reference": {
            "selected_csv": str(reference_selected.resolve()),
            "selected_sha256": sha256(reference_selected),
            "provenance_json": str(reference_provenance.resolve()),
            "provenance_sha256": sha256(reference_provenance),
            "selected_point": reference,
        },
        "selection": (
            "highest measured three-repetition median QPS with recall_min>=0.800 "
            "and recall_median<=0.802; no interpolation"
        ),
        "maximum_deep_iterations": MAX_DEEP_ITERATIONS,
    }
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    contract_path = state / "contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "ablation contract changed inside immutable result root"
            )
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    correctness = [
        {
            "seed_policy": POLICY_K,
            "itopk": 64,
            "search_width": 1,
            "max_iterations": 0,
        },
        {
            "seed_policy": POLICY_WD,
            "itopk": 64,
            "search_width": 1,
            "max_iterations": 0,
        },
        {
            "seed_policy": POLICY_WD,
            "itopk": 128,
            "search_width": 2,
            "max_iterations": 0,
        },
    ]
    create_group(root, data_root, "correctness", "correctness", correctness, 1)
    incumbent = [
        {
            "seed_policy": policy,
            "itopk": REFERENCE_POINT[0],
            "search_width": REFERENCE_POINT[1],
            "max_iterations": REFERENCE_POINT[2],
        }
        for policy in (POLICY_K, POLICY_WD)
    ]
    create_group(
        root, data_root, "paired_incumbent", "throughput", incumbent, 3
    )
    anchors = [
        {
            "seed_policy": POLICY_WD,
            "itopk": itopk,
            "search_width": width,
            "max_iterations": 0,
        }
        for width in WIDTHS
        for itopk in ANCHORS
    ]
    create_group(root, data_root, "anchors_wd", "throughput", anchors, 1)
    print(contract_path)


RAW_FIELDS = [
    "group",
    "phase",
    "seed_policy",
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
    group = str(manifest["group"])
    phase = str(manifest["phase"])
    repetitions = int(manifest["repetitions"])
    expected = {
        (
            str(row["seed_policy"]),
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
            int(row["navix_seed_cap"]),
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
                raise ValueError(f"wrong NaviX path in {raw_path}: {label}")
            itopk = round(finite(record, "itopk", raw_path))
            width = round(finite(record, "search_width", raw_path))
            iterations = round(finite(record, "max_iterations", raw_path))
            cap = round(finite(record, "navix_seed_cap", raw_path))
            policy = POLICY_K if cap == K else POLICY_WD
            key = (policy, itopk, width, iterations, cap)
            if key not in expected or key in seen[repetition]:
                raise ValueError(
                    f"unexpected/duplicate point {key} in {raw_path}"
                )
            seen[repetition].add(key)
            if (
                round(finite(record, "k", raw_path)) != K
                or round(finite(record, "max_queries", raw_path))
                != MAX_QUERIES
                or round(finite(record, "navix_bitmap_seeds", raw_path)) != 1
                or round(finite(record, "n_queries", raw_path)) != query_count
                or cap != seed_cap(policy, width)
            ):
                raise ValueError(f"seed/runtime contract failed in {raw_path}")
            recall = finite(record, "ValidGTRecall", raw_path)
            valid_fraction = finite(record, "ValidGTFraction", raw_path)
            qps = finite(record, "items_per_second", raw_path)
            if not (0 <= recall <= 1 and 0 < valid_fraction <= 1 and qps > 0):
                raise ValueError(f"invalid measurement in {raw_path}")
            errors = {
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
            if any(value != 0 for value in errors.values()):
                raise ValueError(
                    f"NaviX correctness failure in {raw_path}: {errors}"
                )
            output.append(
                {
                    "group": group,
                    "phase": phase,
                    "seed_policy": policy,
                    "seed_cap": cap,
                    "itopk": itopk,
                    "search_width": width,
                    "max_iterations": iterations,
                    "repetition_index": repetition,
                    "shard_index": shard_index,
                    "queries": query_count,
                    "recall": recall,
                    "valid_gt_fraction": valid_fraction,
                    "qps": qps,
                    "seconds": query_count / qps,
                    **errors,
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
        raise ValueError(f"no ablation manifests under {root}")
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        raw_dir = root / "raw" / manifest.parent.parent.name / "yfcc"
        if raw_dir.is_dir():
            rows.extend(load_group(root, manifest))
    return rows


SUMMARY_FIELDS = [
    "group",
    "phase",
    "seed_policy",
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
        key = tuple(
            row[field]
            for field in (
                "group",
                "phase",
                "seed_policy",
                "seed_cap",
                "itopk",
                "search_width",
                "max_iterations",
                "repetition_index",
            )
        )
        repetitions.setdefault(key, []).append(row)
    combined: list[dict[str, object]] = []
    for key, rows in repetitions.items():
        group, phase, policy, cap, itopk, width, iterations, repetition = key
        queries = sum(int(row["queries"]) for row in rows)
        expected_queries = 1_000 if phase == "correctness" else 10_000
        expected_shards = 1 if phase == "correctness" else 5
        if queries != expected_queries or len(rows) != expected_shards:
            raise ValueError(
                f"incomplete serial aggregate for {group}, repetition {repetition}"
            )
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
        combined.append(
            {
                "group": group,
                "phase": phase,
                "seed_policy": policy,
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
        key = tuple(
            row[field]
            for field in (
                "group",
                "phase",
                "seed_policy",
                "seed_cap",
                "itopk",
                "search_width",
                "max_iterations",
            )
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, rows in sorted(groups.items()):
        group, phase, policy, cap, itopk, width, iterations = key
        recalls = [float(row["recall"]) for row in rows]
        qps = [float(row["qps"]) for row in rows]
        output.append(
            {
                "group": group,
                "phase": phase,
                "seed_policy": policy,
                "seed_cap": cap,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
                "repetitions": len(rows),
                "queries_per_repetition": int(rows[0]["queries"]),
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "qps_median": statistics.median(qps),
                "qps_min": min(qps),
                "qps_max": max(qps),
                "seconds_median": statistics.median(
                    float(row["seconds"]) for row in rows
                ),
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


def in_window(row: dict[str, object]) -> bool:
    return (
        float(row["recall_min"]) >= TARGET_LOW
        and float(row["recall_median"]) <= TARGET_HIGH
    )


def next_integer(
    rows: list[dict[str, object]], coordinate: str, minimum: int, maximum: int
) -> list[int]:
    values = sorted({int(row[coordinate]) for row in rows})
    by_value = {int(row[coordinate]): row for row in rows}
    window = [value for value in values if in_window(by_value[value])]
    if window:
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
                abs(float(by_value[value]["recall_median"]) - TARGET_LOW),
                value,
            ),
        )
        return sorted(
            neighbor
            for neighbor in (center - 1, center + 1)
            if minimum <= neighbor <= maximum and neighbor not in by_value
        )
    adjacent = []
    for left, right in pairwise(values):
        if (
            float(by_value[left]["recall_min"]) < TARGET_LOW
            and float(by_value[right]["recall_min"]) >= TARGET_LOW
        ):
            adjacent.append((right - left, left, right))
    if adjacent:
        _, left, right = min(adjacent)
        return [(left + right) // 2] if right - left > 1 else []
    if all(
        float(by_value[value]["recall_min"]) >= TARGET_LOW for value in values
    ):
        low = min(values)
        return [] if low == minimum else [(minimum + low) // 2]
    if all(
        float(by_value[value]["recall_min"]) < TARGET_LOW for value in values
    ):
        high = max(values)
        return [] if high == maximum else [(high + maximum + 1) // 2]
    return []


def b0_iterations(itopk: int, width: int) -> int:
    reach = 1
    graph_steps = 0
    while reach < 10_000_000:
        reach *= max(2, GRAPH_DEGREE // 2)
        graph_steps += 1
    # CAGRA derives the automatic B0 budget from the requested L before its
    # CTA-local result buffer is rounded to a warp boundary.
    return itopk // width + graph_steps


def measured_key(row: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(row["itopk"]),
        int(row["search_width"]),
        int(row["max_iterations"]),
    )


def next_for_width(
    rows: list[dict[str, object]], width: int
) -> list[dict[str, object]]:
    local = [
        row
        for row in rows
        if row["phase"] == "throughput"
        and row["seed_policy"] == POLICY_WD
        and int(row["search_width"]) == width
    ]
    b0 = [row for row in local if int(row["max_iterations"]) == 0]
    measured = {measured_key(row) for row in local}
    next_l = next_integer(b0, "itopk", K, MAX_L)
    if next_l:
        return [
            {
                "seed_policy": POLICY_WD,
                "itopk": value,
                "search_width": width,
                "max_iterations": 0,
            }
            for value in next_l
            if (value, width, 0) not in measured
        ]
    if any(in_window(row) for row in b0):
        return []

    reached = [row for row in b0 if float(row["recall_min"]) >= TARGET_LOW]
    if reached:
        upper = min(reached, key=lambda row: int(row["itopk"]))
        itopk = int(upper["itopk"])
        budget = b0_iterations(itopk, width)
        explicit = [
            row
            for row in local
            if int(row["itopk"]) == itopk and int(row["max_iterations"]) > 0
        ]
        pseudo = explicit + [{**upper, "max_iterations": budget}]
        values = next_integer(pseudo, "max_iterations", 1, budget)
        return [
            {
                "seed_policy": POLICY_WD,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": value,
            }
            for value in values
            if (itopk, width, value) not in measured and value != budget
        ]

    l512 = [row for row in b0 if int(row["itopk"]) == MAX_L]
    if not l512:
        return []
    base = l512[0]
    budget = b0_iterations(MAX_L, width)
    deep = [
        row
        for row in local
        if int(row["itopk"]) == MAX_L and int(row["max_iterations"]) > 0
    ]
    if any(float(row["recall_min"]) >= TARGET_LOW for row in deep):
        pseudo = deep + [{**base, "max_iterations": budget}]
        values = next_integer(
            pseudo, "max_iterations", budget, MAX_DEEP_ITERATIONS
        )
    else:
        largest = max([budget] + [int(row["max_iterations"]) for row in deep])
        values = (
            [min(MAX_DEEP_ITERATIONS, max(522, largest * 2))]
            if largest < MAX_DEEP_ITERATIONS
            else []
        )
    return [
        {
            "seed_policy": POLICY_WD,
            "itopk": MAX_L,
            "search_width": width,
            "max_iterations": value,
        }
        for value in values
        if (MAX_L, width, value) not in measured and value != budget
    ]


def plan_next(root: Path, data_root: Path) -> None:
    summaries = [
        row
        for row in summarize(load_raw(root))
        if row["group"] == "anchors_wd"
        or str(row["group"]).startswith("calibration_r")
    ]
    points: list[dict[str, object]] = []
    for width in WIDTHS:
        points.extend(next_for_width(summaries, width))
    rounds = sorted(
        (root / "configs").glob("calibration_r*/yfcc/manifest.json")
    )
    group = f"calibration_r{len(rounds):02d}"
    if points:
        create_group(root, data_root, group, "throughput", points, 1)
    payload = {
        "schema_version": 1,
        "complete": not points,
        "next_group": group if points else None,
        "next_points": points,
        "measured_wd_points": sum(
            row["phase"] == "throughput" and row["seed_policy"] == POLICY_WD
            for row in summaries
        ),
    }
    (root / "state" / "calibration_state.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload))


def choose(rows: list[dict[str, object]]) -> tuple[dict[str, object], str]:
    within = [row for row in rows if in_window(row)]
    if within:
        return max(within, key=lambda row: float(row["qps_median"])), "matched"
    reached = [row for row in rows if float(row["recall_min"]) >= TARGET_LOW]
    if reached:
        return min(
            reached,
            key=lambda row: (
                float(row["recall_median"]) - TARGET_LOW,
                -float(row["qps_median"]),
            ),
        ), "nearest_overshoot"
    return max(
        rows,
        key=lambda row: (
            float(row["recall_median"]),
            float(row["qps_median"]),
        ),
    ), "unreached"


def prepare_finalists(root: Path, data_root: Path) -> None:
    rows = [
        row
        for row in summarize(load_raw(root))
        if row["phase"] == "throughput"
        and row["seed_policy"] == POLICY_WD
        and (
            row["group"] == "anchors_wd"
            or str(row["group"]).startswith("calibration_r")
        )
    ]
    within = sorted(
        (row for row in rows if in_window(row)),
        key=lambda row: float(row["qps_median"]),
        reverse=True,
    )
    candidates = within[:3]
    if not candidates:
        selected, _ = choose(rows)
        candidates = [selected]
    points = [
        {
            "seed_policy": POLICY_WD,
            "itopk": int(row["itopk"]),
            "search_width": int(row["search_width"]),
            "max_iterations": int(row["max_iterations"]),
        }
        for row in candidates
    ]
    create_group(root, data_root, "finalists_wd", "throughput", points, 3)
    print(group_manifest(root, "finalists_wd"))


def prepare_controls(root: Path, data_root: Path) -> None:
    rows = [
        row
        for row in summarize(load_raw(root))
        if row["group"] == "finalists_wd" and row["seed_policy"] == POLICY_WD
    ]
    winner, status = choose(rows)
    point_data = {
        "itopk": int(winner["itopk"]),
        "search_width": int(winner["search_width"]),
        "max_iterations": int(winner["max_iterations"]),
    }
    state = {
        "schema_version": 1,
        "status": status,
        "winner": point_data,
    }
    winner_path = root / "state" / "winner.json"
    if winner_path.is_file() and json.loads(winner_path.read_text()) != state:
        raise ValueError("winner changed inside immutable result root")
    winner_path.write_text(json.dumps(state, indent=2) + "\n")
    points = [
        {"seed_policy": policy, **point_data}
        for policy in (POLICY_K, POLICY_WD)
    ]
    create_group(root, data_root, "paired_winner", "throughput", points, 3)
    print(winner_path)


def diagnostic_summary(root: Path) -> list[dict[str, object]]:
    winner = json.loads((root / "state" / "winner.json").read_text())["winner"]
    winner_width = int(winner["search_width"])
    expected_caps = {
        "incumbent_k": K,
        "incumbent_wd": REFERENCE_POINT[1] * GRAPH_DEGREE,
        "winner_k": K,
        "winner_wd": winner_width * GRAPH_DEGREE,
    }
    output = []
    for directory in sorted((root / "diagnostics").glob("*")):
        path = directory / "query_summary.csv"
        if not path.is_file():
            continue
        rows = read_csv(path)
        if len(rows) != 1_000:
            raise ValueError(
                f"diagnostic {directory.name} has {len(rows)} queries"
            )
        manifest = json.loads((directory / "manifest.json").read_text())
        cap = expected_caps.get(directory.name)
        if cap is None:
            raise ValueError(f"unexpected diagnostic variant {directory.name}")
        if int(manifest.get("topk", -1)) != K:
            raise ValueError(
                f"diagnostic {directory.name} has the wrong top-k"
            )
        counts = [int(row["navix_seed_count"]) for row in rows]
        if any(count < 0 or count > cap for count in counts):
            raise ValueError(
                f"diagnostic seed count exceeds cap for {directory.name}"
            )
        summary: dict[str, object] = {
            "variant": directory.name,
            "seed_cap": cap,
            "queries": len(rows),
            "seed_count_mean": statistics.mean(counts),
            "seed_count_min": min(counts),
            "seed_count_max": max(counts),
            "queries_at_cap_fraction": sum(count == cap for count in counts)
            / len(counts),
        }
        for field in (
            "seed_inspected_units",
            "iterations",
            "graph_rows_read",
            "predicate_probes",
            "distance_evaluations",
            "passing_admissions",
            "output_count",
        ):
            if field in rows[0]:
                summary[f"{field}_mean"] = statistics.mean(
                    float(row[field]) for row in rows
                )
        if "gt_seen_mask" in rows[0]:
            summary["gt_seen_fraction_mean"] = statistics.mean(
                int(row["gt_seen_mask"]).bit_count() / K for row in rows
            )
        output.append(summary)
    return output


def qps_label(value: float) -> str:
    return f"{value / 1000:.1f}K" if value >= 1000 else f"{value:.0f}"


def analyze(root: Path, require_diagnostics: bool) -> None:
    raw = load_raw(root)
    summaries = summarize(raw)
    correctness_caps = {
        int(row["seed_cap"])
        for row in summaries
        if row["group"] == "correctness"
    }
    if correctness_caps != {10, 64, 128}:
        raise ValueError(f"missing correctness gates: {correctness_caps}")
    incumbent_rows = [
        row for row in summaries if row["group"] == "paired_incumbent"
    ]
    winner_rows = [row for row in summaries if row["group"] == "paired_winner"]
    if len(incumbent_rows) != 2 or len(winner_rows) != 2:
        raise ValueError("paired controls are incomplete")
    incumbent = {str(row["seed_policy"]): row for row in incumbent_rows}
    winner = {str(row["seed_policy"]): row for row in winner_rows}
    winner_state = json.loads((root / "state" / "winner.json").read_text())
    contract = json.loads((root / "state" / "contract.json").read_text())
    reference = contract["reference"]["selected_point"]
    recall_delta = float(incumbent[POLICY_K]["recall_median"]) - float(
        reference["recall"]
    )
    qps_ratio = float(incumbent[POLICY_K]["qps_median"]) / float(
        reference["qps"]
    )
    if abs(recall_delta) > 1e-9:
        raise ValueError(f"10-seed reference recall drifted by {recall_delta}")
    if not 0.85 <= qps_ratio <= 1.15:
        raise ValueError(
            f"10-seed reference QPS drift is too large: {qps_ratio}"
        )

    baseline = incumbent[POLICY_K]
    wd = winner[POLICY_WD]
    target_rows = []
    for label, row, status in (
        ("FAISS-matched k seeds", baseline, "matched"),
        ("CAGRA-native W*D seeds", wd, str(winner_state["status"])),
    ):
        target_rows.append(
            {
                "policy": label,
                "status": status,
                "seed_cap": int(row["seed_cap"]),
                "itopk": int(row["itopk"]),
                "search_width": int(row["search_width"]),
                "max_iterations": int(row["max_iterations"]),
                "recall_median": float(row["recall_median"]),
                "recall_min": float(row["recall_min"]),
                "recall_max": float(row["recall_max"]),
                "qps_median": float(row["qps_median"]),
                "qps_min": float(row["qps_min"]),
                "qps_max": float(row["qps_max"]),
                "speedup_vs_k_seed": float(row["qps_median"])
                / float(baseline["qps_median"]),
            }
        )
    paired = [
        {
            "configuration": "incumbent",
            "itopk": int(incumbent[POLICY_K]["itopk"]),
            "search_width": int(incumbent[POLICY_K]["search_width"]),
            "max_iterations": int(incumbent[POLICY_K]["max_iterations"]),
            "k_seed_recall": float(incumbent[POLICY_K]["recall_median"]),
            "wd_seed_recall": float(incumbent[POLICY_WD]["recall_median"]),
            "k_seed_qps": float(incumbent[POLICY_K]["qps_median"]),
            "wd_seed_qps": float(incumbent[POLICY_WD]["qps_median"]),
            "wd_over_k_qps": float(incumbent[POLICY_WD]["qps_median"])
            / float(incumbent[POLICY_K]["qps_median"]),
        },
        {
            "configuration": "wd_winner",
            "itopk": int(winner[POLICY_WD]["itopk"]),
            "search_width": int(winner[POLICY_WD]["search_width"]),
            "max_iterations": int(winner[POLICY_WD]["max_iterations"]),
            "k_seed_recall": float(winner[POLICY_K]["recall_median"]),
            "wd_seed_recall": float(winner[POLICY_WD]["recall_median"]),
            "k_seed_qps": float(winner[POLICY_K]["qps_median"]),
            "wd_seed_qps": float(winner[POLICY_WD]["qps_median"]),
            "wd_over_k_qps": float(winner[POLICY_WD]["qps_median"])
            / float(winner[POLICY_K]["qps_median"]),
        },
    ]

    diagnostics = diagnostic_summary(root)
    if require_diagnostics and len(diagnostics) != 4:
        raise ValueError("all four paired diagnostic captures are required")
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "raw_points.csv", raw, RAW_FIELDS)
    write_csv(analysis / "measurements.csv", summaries, SUMMARY_FIELDS)
    write_csv(
        analysis / "target_comparison.csv", target_rows, list(target_rows[0])
    )
    write_csv(analysis / "paired_controls.csv", paired, list(paired[0]))
    if diagnostics:
        diagnostic_fields = sorted({key for row in diagnostics for key in row})
        write_csv(analysis / "diagnostics.csv", diagnostics, diagnostic_fields)

    tex = [
        "% Generated by a100_k10_wd_seed_ablation/wd_seed_ablation.py; do not edit.",
        r"\newcommand{\YFCCKTenWDSeedRows}{%",
    ]
    for row in target_rows:
        recall = float(row["recall_median"])
        recall_text = (
            f"{recall:.4f}"
            if row["status"] == "matched"
            else f"max {recall:.4f}"
        )
        tex.append(
            f"{row['policy']} & {qps_label(float(row['qps_median']))} & {recall_text} \\\\%"
        )
    tex.append("}")
    (analysis / "yfcc_k10_wd_seed_ablation.tex").write_text(
        "\n".join(tex) + "\n"
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    wd_points = [
        row
        for row in summaries
        if row["phase"] == "throughput"
        and row["seed_policy"] == POLICY_WD
        and not str(row["group"]).startswith("diagnostic_")
    ]
    axis.scatter(
        [float(row["recall_median"]) for row in wd_points],
        [float(row["qps_median"]) for row in wd_points],
        color="#d62728",
        alpha=0.75,
        label="up to W*D seeds",
    )
    axis.scatter(
        [float(baseline["recall_median"])],
        [float(baseline["qps_median"])],
        color="#1f77b4",
        label="up to k seeds",
    )
    axis.axvspan(TARGET_LOW, TARGET_HIGH, color="#777777", alpha=0.10)
    axis.set_xlabel("Recall@10")
    axis.set_ylabel("Queries per second")
    axis.set_title("YFCC-10M NaviX initialization-width ablation")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    for extension in ("pdf", "png"):
        figure.savefig(
            analysis / f"yfcc_k10_wd_seed_ablation.{extension}", dpi=220
        )
    plt.close(figure)

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "k": K,
        "max_queries": MAX_QUERIES,
        "target_window": [TARGET_LOW, TARGET_HIGH],
        "target_comparison": target_rows,
        "paired_controls": paired,
        "reference_replay": {
            "recall_delta": recall_delta,
            "qps_ratio": qps_ratio,
        },
        "diagnostics": diagnostics,
        "automatic_promotion": False,
    }
    (analysis / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    provenance = {
        **contract,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted((root / "raw").glob("*/yfcc/shard_*.json"))
        ],
        "result": payload,
    }
    (analysis / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"measurements": len(summaries), "diagnostics": len(diagnostics)}
        )
    )


def generate_diagnostics(root: Path, data_root: Path) -> None:
    winner_state = json.loads((root / "state" / "winner.json").read_text())
    winner = winner_state["winner"]
    paths = dataset_paths(data_root, "yfcc", "correctness", GRAPH_DEGREE)
    source = json.loads(paths.manifest.read_text())
    groundtruth = str(
        (Path(source["shards"][0]["directory"]) / "groundtruth.ibin").resolve()
    )
    variants = {
        "incumbent_k": (POLICY_K, REFERENCE_POINT),
        "incumbent_wd": (POLICY_WD, REFERENCE_POINT),
        "winner_k": (
            POLICY_K,
            (
                int(winner["itopk"]),
                int(winner["search_width"]),
                int(winner["max_iterations"]),
            ),
        ),
        "winner_wd": (
            POLICY_WD,
            (
                int(winner["itopk"]),
                int(winner["search_width"]),
                int(winner["max_iterations"]),
            ),
        ),
    }
    for variant, (policy, coordinates) in variants.items():
        itopk, width, iterations = coordinates
        diagnostic = {
            "favor_diagnostics_output": str(
                (root / "diagnostics" / variant).resolve()
            ),
            "favor_diagnostics_groundtruth": groundtruth,
            "favor_diagnostics_dataset": "yfcc10m-a100-k10-wd-seed-ablation",
            "favor_diagnostics_variant": variant,
        }
        create_group(
            root,
            data_root,
            f"diagnostic_{variant}",
            "correctness",
            [
                {
                    "seed_policy": policy,
                    "itopk": itopk,
                    "search_width": width,
                    "max_iterations": iterations,
                    "diagnostic": diagnostic,
                }
            ],
            1,
        )


def bundle(root: Path) -> None:
    if not (root / "analysis" / "results.json").is_file():
        raise FileNotFoundError("analyze the ablation before bundling")
    destination = root / "paper_gpu_bundle_k10_wd_seed_ablation"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in (
        "analysis",
        "state",
        "provenance",
        "configs",
        "raw",
        "diagnostics",
    ):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
    files = [
        {
            "path": str(path.relative_to(destination)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    (destination / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n"
    )
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("initialize")
    init_parser.add_argument("--root", type=Path, required=True)
    init_parser.add_argument("--data-root", type=Path, required=True)
    init_parser.add_argument("--reference-selected", type=Path, required=True)
    init_parser.add_argument(
        "--reference-provenance", type=Path, required=True
    )
    for command in (
        "plan-next",
        "prepare-finalists",
        "prepare-controls",
        "generate-diagnostics",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True)
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
            args.reference_provenance.resolve(),
        )
    elif args.command == "plan-next":
        plan_next(args.root.resolve(), args.data_root.resolve())
    elif args.command == "prepare-finalists":
        prepare_finalists(args.root.resolve(), args.data_root.resolve())
    elif args.command == "prepare-controls":
        prepare_controls(args.root.resolve(), args.data_root.resolve())
    elif args.command == "generate-diagnostics":
        generate_diagnostics(args.root.resolve(), args.data_root.resolve())
    elif args.command == "analyze":
        analyze(args.root.resolve(), args.require_diagnostics)
    else:
        bundle(args.root.resolve())


if __name__ == "__main__":
    main()
