#!/usr/bin/env python3
"""Build, validate, analyze, and bundle the all-workload k=10 W*D NaviX rerun."""

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
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RETRIEVE_DIR = SCRIPT_DIR.parent
GPU_DIR = RETRIEVE_DIR / "gpu_graph"
MATCHED_DIR = RETRIEVE_DIR / "matched_recall"
PAPER_DIR = RETRIEVE_DIR / "a100_paper"
sys.path.insert(0, str(GPU_DIR))
sys.path.insert(0, str(PAPER_DIR))

from dataset_profile import load_profile, profile_record, workload_spec
from generate_configs import (
    B0_CELLS,
    config_payload,
    dataset_paths,
    search_point,
    write_group,
)

K = 10
MAX_QUERIES = 2_048
WORKLOADS = ("yfcc", "em", "emis", "r")
TARGETS = {"yfcc": 0.800, "em": 0.950, "emis": 0.950, "r": 0.950}
TARGET_WINDOW = 0.002
REFERENCE_REPLAY_MAX_SLOT_DRIFT = 10
POLICY_K = "k_seed"
POLICY_WD = "wd_seed"
METHOD = "navix_reference"
EXPERIMENT = "retrieve_workshop_navix_k10_wd_all"
RESOURCE_PREFIX = "CAGRA_KERNEL_RESOURCES "
DIAGNOSTIC_SCHEMA = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"cannot infer columns for empty CSV {path}")
    columns = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def seed_cap(policy: str, width: int, degree: int) -> int:
    if policy == POLICY_K:
        return K
    if policy == POLICY_WD:
        return width * degree
    raise ValueError(f"unknown seed policy {policy!r}")


def navix_search(
    policy: str,
    itopk: int,
    width: int,
    iterations: int,
    degree: int,
    diagnostic: dict[str, str] | None = None,
) -> dict[str, object]:
    row = search_point(METHOD, itopk, width, iterations, k=K, max_queries=MAX_QUERIES)
    row["navix_seed_cap"] = seed_cap(policy, width, degree)
    if diagnostic:
        row.update(diagnostic)
    return row


def selected_reference(path: Path) -> dict[str, dict[str, object]]:
    rows = [
        row
        for row in read_csv(path)
        if row.get("method") == METHOD
        and str(row.get("selected", "true")).lower() in {"1", "true", "yes"}
    ]
    if {row["workload"] for row in rows} != set(WORKLOADS) or len(rows) != len(WORKLOADS):
        raise ValueError(f"require one selected k-seed NaviX row per workload in {path}")
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        workload = str(row["workload"])
        recall = float(row["recall_median"])
        target = TARGETS[workload]
        if not target <= recall <= target + TARGET_WINDOW:
            raise ValueError(f"reference {workload} point is outside its target window")
        output[workload] = {
            "itopk": int(row["itopk"]),
            "search_width": int(row["search_width"]),
            "max_iterations": int(row["max_iterations"]),
            "recall_median": recall,
            "recall_min": float(row["recall_min"]),
            "qps_median": float(row["qps_median"]),
        }
    return output


def enrich_manifest(path: Path, policy: str) -> None:
    manifest = json.loads(path.read_text())
    degree = int(manifest["graph_degree"])
    manifest.update(
        {
            # Keep the shared GPU analyzer's experiment discriminator and add the
            # W*D workflow identity as an orthogonal contract field.
            "experiment": "retrieve_workshop_gpu_graph",
            "wd_all_experiment": EXPERIMENT,
            "wd_all_schema_version": 1,
            "navix_seed_policy": "wd" if policy == POLICY_WD else "k",
            "navix_seed_cap_contract": (
                "search_width * graph_degree" if policy == POLICY_WD else "result k"
            ),
        }
    )
    searches: list[dict[str, object]] = []
    for config_row in manifest["configs"][:1]:
        config = json.loads(Path(config_row["config"]).read_text())
        for index in config["index"]:
            for row in index["search_params"]:
                searches.append(
                    {
                        "method": str(row["bitmap_method"]),
                        "itopk": int(row["itopk"]),
                        "search_width": int(row["search_width"]),
                        "max_iterations": int(row["max_iterations"]),
                        "navix_seed_cap": int(row["navix_seed_cap"]),
                    }
                )
    for row in searches:
        expected = seed_cap(policy, int(row["search_width"]), degree)
        if int(row["navix_seed_cap"]) != expected:
            raise ValueError(f"wrong seed cap in {path}: {row}")
    manifest["search_points"] = searches
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def create_frontier(root: Path, data_root: Path, reference: Path, provenance: Path) -> None:
    profile = load_profile()
    if int(profile["max_queries"]) != MAX_QUERIES:
        raise ValueError("all-workload W*D rerun requires max_queries=2048")
    references = selected_reference(reference)
    reference_provenance = json.loads(provenance.read_text())
    reference_k = int(reference_provenance.get("k", -1))
    if reference_k < 0:
        run_path = Path(
            str(reference_provenance.get("run_provenance", {}).get("path", ""))
        )
        if run_path.is_file():
            reference_k = int(
                json.loads(run_path.read_text())
                .get("fixed_contract", {})
                .get("k", -1)
            )
    if (
        reference_k != K
        or int(reference_provenance.get("max_queries", -1)) != MAX_QUERIES
        or str(reference_provenance.get("navix_seed_policy", "k")) != "k"
    ):
        raise ValueError(
            "reference matched-recall provenance is not the k=10, max_queries=2048, "
            "k-seed run"
        )
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "k": K,
        "max_queries": MAX_QUERIES,
        "workloads": list(WORKLOADS),
        "targets": TARGETS,
        "target_window": TARGET_WINDOW,
        "b0_cells": [list(cell) for cell in B0_CELLS],
        "seed_policy": {
            "primary": POLICY_WD,
            "cap": "search_width * graph_degree",
            "control": POLICY_K,
            "control_cap": K,
        },
        "profile": profile_record(profile),
        "reference": {
            "selected": str(reference.resolve()),
            "selected_sha256": sha256(reference),
            "provenance": str(provenance.resolve()),
            "provenance_sha256": sha256(provenance),
            "points": references,
        },
        "selection": (
            "highest three-repetition median QPS with recall_min>=target and "
            "recall_median<=target+0.002; no interpolation"
        ),
    }
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    contract_path = state / "contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("experiment contract changed inside immutable result root")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    frontier = root / "frontier"
    for workload in WORKLOADS:
        degree = int(workload_spec(workload, profile)["graph_degree"])
        correctness = [
            navix_search(POLICY_WD, 64, 1, 0, degree),
            navix_search(POLICY_WD, 128, 2, 0, degree),
        ]
        b0 = [
            navix_search(POLICY_WD, itopk, width, 0, degree)
            for itopk, width in B0_CELLS
        ]
        for group, phase, searches in (
            ("correctness", "correctness", correctness),
            ("b0", "throughput", b0),
        ):
            manifest_path = frontier / "configs" / group / workload / "manifest.json"
            if not manifest_path.is_file():
                write_group(
                    frontier / "configs",
                    data_root,
                    group=group,
                    workload=workload,
                    phase=phase,
                    searches=searches,
                    yfcc_graph_degree=degree,
                    k=K,
                )
                enrich_manifest(manifest_path, POLICY_WD)
    print(contract_path)


def validate_frontier(root: Path) -> list[dict[str, str]]:
    path = root / "frontier" / "analysis" / "summary_points.csv"
    rows = [
        row
        for row in read_csv(path)
        if row["group"] == "b0"
        and row["phase"] == "throughput"
        and row["method"] == METHOD
        and truth(row["paper_included"])
    ]
    if len(rows) != len(WORKLOADS) * len(B0_CELLS):
        raise ValueError(f"W*D B0 frontier has {len(rows)} rows, expected 24")
    for workload in WORKLOADS:
        local = [row for row in rows if row["workload"] == workload]
        observed = {(int(row["itopk"]), int(row["search_width"])) for row in local}
        if observed != set(B0_CELLS):
            raise ValueError(f"incomplete B0 cells for {workload}: {sorted(observed)}")
        if any(
            int(row["repetitions"]) != 3
            or int(row["shards_per_repetition"]) != 5
            or int(row["queries_per_repetition"]) != 10_000
            or float(row["filter_violations"]) != 0
            or float(row["sentinel_errors"]) != 0
            or float(row["duplicate_output_query_rate_max"]) != 0
            for row in local
        ):
            raise ValueError(f"frontier correctness/coverage contract failed for {workload}")
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "b0_navix_wd.csv", rows, list(rows[0]))
    print(json.dumps({"b0_points": len(rows), "workloads": list(WORKLOADS)}))
    return rows


def create_mixed_group(
    output: Path,
    data_root: Path,
    group: str,
    phase: str,
    points: dict[str, list[dict[str, object]]],
) -> None:
    profile = load_profile()
    for workload, specifications in points.items():
        degree = int(workload_spec(workload, profile)["graph_degree"])
        searches = [
            navix_search(
                str(row["seed_policy"]),
                int(row["itopk"]),
                int(row["search_width"]),
                int(row["max_iterations"]),
                degree,
            )
            for row in specifications
        ]
        manifest_path = output / "configs" / group / workload / "manifest.json"
        if manifest_path.is_file():
            continue
        write_group(
            output / "configs",
            data_root,
            group=group,
            workload=workload,
            phase=phase,
            searches=searches,
            yfcc_graph_degree=degree,
            k=K,
        )
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "experiment": EXPERIMENT,
                "wd_all_schema_version": 1,
                "mixed_seed_policies": True,
                "search_points": [
                    {
                        "method": METHOD,
                        "seed_policy": str(row["seed_policy"]),
                        "navix_seed_cap": seed_cap(
                            str(row["seed_policy"]), int(row["search_width"]), degree
                        ),
                        "itopk": int(row["itopk"]),
                        "search_width": int(row["search_width"]),
                        "max_iterations": int(row["max_iterations"]),
                    }
                    for row in specifications
                ],
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def create_controls(root: Path, data_root: Path, selected: Path) -> None:
    references = json.loads((root / "state" / "contract.json").read_text())["reference"]["points"]
    winners = selected_reference(selected)
    controls = root / "controls"
    for group, source in (("paired_incumbent", references), ("paired_winner", winners)):
        points: dict[str, list[dict[str, object]]] = {}
        for workload in WORKLOADS:
            coordinate = source[workload]
            points[workload] = [
                {
                    "seed_policy": policy,
                    "itopk": coordinate["itopk"],
                    "search_width": coordinate["search_width"],
                    "max_iterations": coordinate["max_iterations"],
                }
                for policy in (POLICY_K, POLICY_WD)
            ]
        create_mixed_group(controls, data_root, group, "throughput", points)
    correctness = {
        workload: [
            {
                "seed_policy": POLICY_K,
                "itopk": 64,
                "search_width": 1,
                "max_iterations": 0,
            }
        ]
        for workload in WORKLOADS
    }
    create_mixed_group(controls, data_root, "correctness_k", "correctness", correctness)
    print(controls / "configs")


RAW_FIELDS = [
    "group",
    "phase",
    "workload",
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


def load_mixed_group(root: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    group = str(manifest["group"])
    workload = str(manifest["workload"])
    phase = str(manifest["phase"])
    repetitions = int(manifest["repetitions"])
    degree = int(manifest["graph_degree"])
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
    raw_dir = root / "raw" / group / workload
    expected_files = {
        f"shard_{int(row['shard_index']):02d}.json" for row in manifest["configs"]
    }
    observed_files = {path.name for path in raw_dir.glob("shard_*.json")}
    if observed_files != expected_files:
        raise ValueError(
            f"incomplete {group}/{workload}: missing={sorted(expected_files-observed_files)}, "
            f"extra={sorted(observed_files-expected_files)}"
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
            if label_value(label, "bitmap_method") != METHOD:
                raise ValueError(f"wrong benchmark method in {raw_path}: {label}")
            itopk = round(finite(record, "itopk", raw_path))
            width = round(finite(record, "search_width", raw_path))
            iterations = round(finite(record, "max_iterations", raw_path))
            cap = round(finite(record, "navix_seed_cap", raw_path))
            policy = POLICY_K if cap == K else POLICY_WD
            identity = (policy, itopk, width, iterations, cap)
            if identity not in expected or identity in seen[repetition]:
                raise ValueError(f"unexpected/duplicate point {identity} in {raw_path}")
            seen[repetition].add(identity)
            if (
                round(finite(record, "k", raw_path)) != K
                or round(finite(record, "max_queries", raw_path)) != MAX_QUERIES
                or round(finite(record, "n_queries", raw_path)) != query_count
                or cap != seed_cap(policy, width, degree)
            ):
                raise ValueError(f"runtime/seed contract failed in {raw_path}")
            errors = {
                "filter_violations": finite(record, "FilterViolations", raw_path),
                "sentinel_errors": finite(record, "InvalidSentinelErrors", raw_path),
                "sentinel_order_errors": finite(record, "SentinelOrderErrors", raw_path),
                "invalid_sentinel_distance_errors": finite(
                    record, "InvalidSentinelDistanceErrors", raw_path
                ),
                "duplicate_output_queries": finite(
                    record, "DuplicateOutputQueries", raw_path
                ),
            }
            if any(value != 0 for value in errors.values()):
                raise ValueError(f"NaviX correctness failure in {raw_path}: {errors}")
            qps = finite(record, "items_per_second", raw_path)
            output.append(
                {
                    "group": group,
                    "phase": phase,
                    "workload": workload,
                    "seed_policy": policy,
                    "seed_cap": cap,
                    "itopk": itopk,
                    "search_width": width,
                    "max_iterations": iterations,
                    "repetition_index": repetition,
                    "shard_index": shard_index,
                    "queries": query_count,
                    "recall": finite(record, "ValidGTRecall", raw_path),
                    "valid_gt_fraction": finite(record, "ValidGTFraction", raw_path),
                    "qps": qps,
                    "seconds": query_count / qps,
                    **errors,
                    "underfilled_queries": finite(record, "UnderfilledQueries", raw_path),
                    "missing_result_slots": finite(record, "MissingResultSlots", raw_path),
                    "source_file": str(raw_path.resolve()),
                }
            )
        for repetition, identities in seen.items():
            if identities != expected:
                raise ValueError(
                    f"incomplete repetition {repetition} in {raw_path}: "
                    f"missing={sorted(expected-identities)}"
                )
    return output


def summarize_controls(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw: list[dict[str, object]] = []
    for manifest in sorted((root / "controls" / "configs").glob("*/*/manifest.json")):
        raw.extend(load_mixed_group(root / "controls", manifest))
    grouped_repetitions: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in raw:
        identity = (
            row["group"],
            row["phase"],
            row["workload"],
            row["seed_policy"],
            row["seed_cap"],
            row["itopk"],
            row["search_width"],
            row["max_iterations"],
            row["repetition_index"],
        )
        grouped_repetitions.setdefault(identity, []).append(row)
    repetitions: list[dict[str, object]] = []
    for identity, members in grouped_repetitions.items():
        group, phase, workload, policy, cap, itopk, width, iterations, repetition = identity
        expected_queries = 1_000 if phase == "correctness" else 10_000
        expected_shards = 1 if phase == "correctness" else 5
        queries = sum(int(row["queries"]) for row in members)
        if queries != expected_queries or len(members) != expected_shards:
            raise ValueError(f"incomplete serial control aggregate: {identity}")
        seconds = sum(float(row["seconds"]) for row in members)
        slots = sum(
            float(row["valid_gt_fraction"]) * int(row["queries"]) * K for row in members
        )
        matches = sum(
            float(row["recall"])
            * float(row["valid_gt_fraction"])
            * int(row["queries"])
            * K
            for row in members
        )
        repetitions.append(
            {
                "group": group,
                "phase": phase,
                "workload": workload,
                "seed_policy": policy,
                "seed_cap": cap,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
                "repetition_index": repetition,
                "queries": queries,
                "recall": matches / slots,
                "qps": queries / seconds,
                "seconds": seconds,
            }
        )
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in repetitions:
        identity = tuple(
            row[field]
            for field in (
                "group",
                "phase",
                "workload",
                "seed_policy",
                "seed_cap",
                "itopk",
                "search_width",
                "max_iterations",
            )
        )
        grouped.setdefault(identity, []).append(row)
    summary: list[dict[str, object]] = []
    for identity, members in sorted(grouped.items()):
        group, phase, workload, policy, cap, itopk, width, iterations = identity
        expected_repetitions = 1 if phase == "correctness" else 3
        observed_repetitions = sorted(
            int(row["repetition_index"]) for row in members
        )
        if observed_repetitions != list(range(expected_repetitions)):
            raise ValueError(
                f"control point has the wrong repetition set: {identity}, "
                f"observed={observed_repetitions}"
            )
        recalls = [float(row["recall"]) for row in members]
        qps = [float(row["qps"]) for row in members]
        summary.append(
            {
                "group": group,
                "phase": phase,
                "workload": workload,
                "seed_policy": policy,
                "seed_cap": cap,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
                "repetitions": len(members),
                "queries_per_repetition": int(members[0]["queries"]),
                "recall_median": statistics.median(recalls),
                "recall_min": min(recalls),
                "recall_max": max(recalls),
                "qps_median": statistics.median(qps),
                "qps_min": min(qps),
                "qps_max": max(qps),
            }
        )
    return raw, summary


def create_diagnostics(root: Path, data_root: Path, selected: Path) -> None:
    references = json.loads((root / "state" / "contract.json").read_text())["reference"]["points"]
    winners = selected_reference(selected)
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    manifest_records: dict[str, dict[str, object]] = {}
    for workload in WORKLOADS:
        paths = dataset_paths(data_root, workload, "correctness", 64)
        source = json.loads(paths.manifest.read_text())
        shards = source.get("shards", [])
        if len(shards) != 1 or int(shards[0].get("query_count", -1)) != 1_000:
            raise ValueError(f"{paths.manifest} must contain one 1,000-query shard")
        shard = shards[0]
        shard_dir = Path(shard["directory"])
        groundtruth = shard_dir / "groundtruth.ibin"
        degree = int(paths.graph_degree)
        variants: list[tuple[str, str, dict[str, object]]] = []
        for prefix, coordinates in (("incumbent", references[workload]), ("winner", winners[workload])):
            for policy in (POLICY_K, POLICY_WD):
                variants.append((f"{prefix}_{policy}", policy, coordinates))
        resource_coordinate: dict[str, object] = {
            "itopk": 64,
            "search_width": 1,
            "max_iterations": 0,
        }
        variants.append(("resource_wd", POLICY_WD, resource_coordinate))
        searches: list[dict[str, object]] = []
        for variant, policy, coordinate in variants:
            diagnostic = {
                "favor_diagnostics_output": str((diagnostics / "captures" / workload / variant).resolve()),
                "favor_diagnostics_groundtruth": str(groundtruth.resolve()),
                "favor_diagnostics_dataset": f"{workload}-a100-k10-wd-all",
                "favor_diagnostics_variant": variant,
            }
            searches.append(
                navix_search(
                    policy,
                    int(coordinate["itopk"]),
                    int(coordinate["search_width"]),
                    int(coordinate["max_iterations"]),
                    degree,
                    diagnostic,
                )
            )
        config = diagnostics / "configs" / f"{workload}.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        if not config.is_file():
            config.write_text(
                json.dumps(
                    config_payload(
                        workload=workload,
                        phase="correctness",
                        shard=shard,
                        paths=paths,
                        searches=searches,
                        k=K,
                    ),
                    indent=2,
                )
                + "\n"
            )

        resource_search = navix_search(POLICY_WD, 64, 1, 0, degree)
        resource_config = diagnostics / "resource_configs" / f"{workload}.json"
        resource_config.parent.mkdir(parents=True, exist_ok=True)
        if not resource_config.is_file():
            resource_config.write_text(
                json.dumps(
                    config_payload(
                        workload=workload,
                        phase="correctness",
                        shard=shard,
                        paths=paths,
                        searches=[resource_search],
                        k=K,
                    ),
                    indent=2,
                )
                + "\n"
            )
        manifest_records[workload] = {
            "diagnostic_config": str(config.resolve()),
            "resource_config": str(resource_config.resolve()),
            "graph_degree": degree,
            "variants": [
                {
                    "name": variant,
                    "seed_policy": policy,
                    "seed_cap": seed_cap(
                        policy, int(coordinate["search_width"]), degree
                    ),
                    "itopk": int(coordinate["itopk"]),
                    "search_width": int(coordinate["search_width"]),
                    "max_iterations": int(coordinate["max_iterations"]),
                }
                for variant, policy, coordinate in variants
            ],
        }
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "k": K,
        "max_queries": MAX_QUERIES,
        "queries": 1_000,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA,
        "workloads": manifest_records,
    }
    manifest_path = diagnostics / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("diagnostic contract changed inside immutable result root")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def mean(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"invalid diagnostic field {field}")
    return statistics.fmean(values)


def diagnostic_summary(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    diagnostics = root / "diagnostics"
    experiment = json.loads((diagnostics / "manifest.json").read_text())
    summaries: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    for workload, workload_contract in experiment["workloads"].items():
        by_variant = {row["name"]: row for row in workload_contract["variants"]}
        for variant, contract in by_variant.items():
            directory = diagnostics / "captures" / workload / variant
            manifest_path = directory / "manifest.json"
            rows_path = directory / "query_summary.csv"
            manifest = json.loads(manifest_path.read_text())
            rows = read_csv(rows_path)
            if (
                int(manifest.get("schema_version", -1)) != DIAGNOSTIC_SCHEMA
                or int(manifest.get("num_queries", -1)) != 1_000
                or int(manifest.get("topk", -1)) != K
                or str(manifest.get("variant")) != variant
                or bool(manifest.get("timing_valid", True))
                or len(rows) != 1_000
                or [int(row["query_id"]) for row in rows] != list(range(1_000))
            ):
                raise ValueError(f"diagnostic contract failed: {directory}")
            cap = int(contract["seed_cap"])
            counts = [int(row["navix_seed_count"]) for row in rows]
            if any(count < 0 or count > cap for count in counts):
                raise ValueError(f"seed count exceeds cap in {directory}")
            summary: dict[str, object] = {
                "workload": workload,
                "variant": variant,
                "seed_policy": contract["seed_policy"],
                "seed_cap": cap,
                "itopk": int(contract["itopk"]),
                "search_width": int(contract["search_width"]),
                "max_iterations": int(contract["max_iterations"]),
                "queries": 1_000,
                "seed_count_mean": statistics.fmean(counts),
                "seed_count_min": min(counts),
                "seed_count_max": max(counts),
                "queries_at_cap_fraction": sum(count == cap for count in counts) / len(counts),
            }
            for field in (
                "seed_inspected_units",
                "iterations",
                "graph_rows_read",
                "predicate_probes",
                "distance_evaluations",
                "passing_admissions",
                "output_count",
                "recall",
            ):
                if field in rows[0]:
                    summary[f"{field}_mean"] = mean(rows, field)
            if "gt_seen_mask" in rows[0]:
                summary["gt_seen_fraction_mean"] = statistics.fmean(
                    int(row["gt_seen_mask"]).bit_count() / K for row in rows
                )
            summaries.append(summary)
            if variant == "resource_wd":
                resource_rows.append(
                    {
                        "workload": workload,
                        "method": METHOD,
                        "queries": 1_000,
                        "itopk": 64,
                        "search_width": 1,
                        "max_iterations": 0,
                        "max_queries": MAX_QUERIES,
                        "recall": float(summary.get("recall_mean", 0.0)),
                        "graph_rows_per_query": float(summary["graph_rows_read_mean"]),
                        "seed_bitmap_words_per_query": float(summary["seed_inspected_units_mean"]),
                        "bitmap_probes_per_query": float(summary["predicate_probes_mean"]),
                        "distance_evaluations_per_query": float(summary["distance_evaluations_mean"]),
                        "passing_admissions_per_query": float(summary["passing_admissions_mean"]),
                    }
                )
    if len(summaries) != len(WORKLOADS) * 5 or len(resource_rows) != len(WORKLOADS):
        raise ValueError("all-workload paired diagnostics are incomplete")
    return summaries, resource_rows


def load_resource_tuple(path: Path, graph_degree: int) -> dict[str, object]:
    records: set[tuple[object, ...]] = set()
    values: dict[tuple[object, ...], dict[str, object]] = {}
    for line in path.read_text().splitlines():
        position = line.find(RESOURCE_PREFIX)
        if position < 0:
            continue
        record = json.loads(line[position + len(RESOURCE_PREFIX) :])
        if (
            bool(record.get("diagnostics"))
            or str(record.get("method")) != "navix"
            or int(record.get("graph_degree", -1)) != graph_degree
            or int(record.get("itopk", -1)) != 64
            or int(record.get("search_width", -1)) != 1
        ):
            continue
        identity = tuple(
            int(record[key])
            for key in (
                "threads_per_cta",
                "registers_per_thread",
                "dynamic_smem_bytes",
                "static_smem_bytes",
                "active_ctas_per_sm",
            )
        )
        records.add(identity)
        values[identity] = record
    if len(records) != 1:
        raise ValueError(f"expected one NaviX resource tuple in {path}, got {records}")
    return values[next(iter(records))]


def analyze(root: Path) -> None:
    b0 = validate_frontier(root)
    selected_path = root / "matched_recall" / "analysis" / "selected_points.csv"
    selected = [row for row in read_csv(selected_path) if row["method"] == METHOD]
    if len(selected) != len(WORKLOADS):
        raise ValueError("matched-recall analysis must select one W*D NaviX point per workload")
    for row in selected:
        workload = str(row["workload"])
        if (
            int(row["repetitions"]) != 3
            or int(row["shards_per_repetition"]) != 5
            or int(row["queries_per_repetition"]) != 10_000
            or float(row["filter_violations"]) != 0
            or float(row["sentinel_errors"]) != 0
            or float(row["duplicate_output_query_rate_max"]) != 0
        ):
            raise ValueError(f"selected W*D point violates the execution contract: {row}")
        if truth(row["target_reached"]) and not truth(row["within_target_window"]):
            raise ValueError(f"selected W*D point overshoots the target window: {row}")
        if float(row["target_recall"]) != TARGETS[workload]:
            raise ValueError(f"target mismatch in selected row: {row}")

    raw_controls, controls = summarize_controls(root)
    diagnostics, resource_dynamic = diagnostic_summary(root)
    resource_by_workload = {str(row["workload"]): row for row in resource_dynamic}
    for workload in WORKLOADS:
        resource = load_resource_tuple(
            root / "diagnostics" / "resources" / f"{workload}.log", 64
        )
        row = resource_by_workload[workload]
        row.update(
            {
                "threads_per_cta": int(resource["threads_per_cta"]),
                "registers_per_thread": int(resource["registers_per_thread"]),
                "dynamic_smem_bytes": int(resource["dynamic_smem_bytes"]),
                "static_smem_bytes": int(resource["static_smem_bytes"]),
                "active_ctas_per_sm": int(resource["active_ctas_per_sm"]),
            }
        )

    control_by_key = {
        (str(row["group"]), str(row["workload"]), str(row["seed_policy"])): row
        for row in controls
        if row["phase"] == "throughput"
    }
    paired: list[dict[str, object]] = []
    for group, label in (("paired_incumbent", "k_seed_incumbent"), ("paired_winner", "wd_winner")):
        for workload in WORKLOADS:
            k_row = control_by_key[(group, workload, POLICY_K)]
            wd_row = control_by_key[(group, workload, POLICY_WD)]
            paired.append(
                {
                    "workload": workload,
                    "configuration": label,
                    "itopk": int(k_row["itopk"]),
                    "search_width": int(k_row["search_width"]),
                    "max_iterations": int(k_row["max_iterations"]),
                    "k_seed_cap": int(k_row["seed_cap"]),
                    "wd_seed_cap": int(wd_row["seed_cap"]),
                    "k_seed_recall": float(k_row["recall_median"]),
                    "wd_seed_recall": float(wd_row["recall_median"]),
                    "k_seed_qps": float(k_row["qps_median"]),
                    "wd_seed_qps": float(wd_row["qps_median"]),
                    "wd_over_k_qps": float(wd_row["qps_median"]) / float(k_row["qps_median"]),
                }
            )

    references = json.loads((root / "state" / "contract.json").read_text())["reference"]["points"]
    for workload in WORKLOADS:
        replay = control_by_key[("paired_incumbent", workload, POLICY_K)]
        recall_delta = float(replay["recall_median"]) - float(references[workload]["recall_median"])
        qps_ratio = float(replay["qps_median"]) / float(references[workload]["qps_median"])
        replay_tolerance = REFERENCE_REPLAY_MAX_SLOT_DRIFT / (10_000 * K)
        if abs(recall_delta) > replay_tolerance + 1e-12:
            raise ValueError(f"{workload} k-seed reference recall drifted by {recall_delta}")
        if not 0.85 <= qps_ratio <= 1.15:
            raise ValueError(f"{workload} k-seed reference QPS drift is too large: {qps_ratio}")

    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "b0_navix_wd.csv", [dict(row) for row in b0], list(b0[0]))
    write_csv(output / "selected_navix_wd.csv", [dict(row) for row in selected], list(selected[0]))
    write_csv(output / "control_raw_points.csv", raw_controls, RAW_FIELDS)
    write_csv(output / "control_summary.csv", controls)
    write_csv(output / "paired_controls.csv", paired)
    write_csv(output / "diagnostics.csv", diagnostics, sorted({key for row in diagnostics for key in row}))
    write_csv(output / "navix_resource_work.csv", resource_dynamic)

    target_rows = []
    for row in selected:
        target_rows.append(
            {
                "workload": row["workload"],
                "target_low": float(row["target_recall"]),
                "target_high": float(row["target_recall"]) + TARGET_WINDOW,
                "target_reached": truth(row["target_reached"]),
                "within_target_window": truth(row["within_target_window"]),
                "itopk": int(row["itopk"]),
                "search_width": int(row["search_width"]),
                "max_iterations": int(row["max_iterations"]),
                "seed_cap": int(row["search_width"]) * 64,
                "recall_median": float(row["recall_median"]),
                "recall_min": float(row["recall_min"]),
                "recall_max": float(row["recall_max"]),
                "qps_median": float(row["qps_median"]),
                "qps_min": float(row["qps_min"]),
                "qps_max": float(row["qps_max"]),
            }
        )
    write_csv(output / "target_results.csv", target_rows)
    (output / "results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "experiment": EXPERIMENT,
                "b0_points": len(b0),
                "selected_points": len(selected),
                "paired_controls": len(paired),
                "diagnostics": len(diagnostics),
                "reference_replay_max_slot_drift": REFERENCE_REPLAY_MAX_SLOT_DRIFT,
                "target_results": target_rows,
            },
            indent=2,
        )
        + "\n"
    )
    plot_results(output, b0, selected)
    write_tex(output, target_rows)
    print(output)


def pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    best = -math.inf
    for row in sorted(rows, key=lambda item: (float(item["recall_median"]), float(item["qps_median"])), reverse=True):
        qps = float(row["qps_median"])
        if qps > best:
            kept.append(row)
            best = qps
    return sorted(kept, key=lambda item: float(item["recall_median"]))


def plot_results(output: Path, b0: list[dict[str, object]], selected: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 2.8), constrained_layout=True)
    labels = {"yfcc": "YFCC-10M", "em": "ArXiv-large EM", "emis": "ArXiv-large EMIS", "r": "ArXiv-large R"}
    for axis, workload in zip(axes, WORKLOADS, strict=True):
        points = pareto([dict(row) for row in b0 if row["workload"] == workload])
        axis.plot(
            [float(row["recall_median"]) for row in points],
            [float(row["qps_median"]) for row in points],
            color="#2ca02c",
            marker="^",
            linewidth=1.2,
            markersize=3.5,
        )
        winner = next(row for row in selected if row["workload"] == workload)
        axis.scatter(
            [float(winner["recall_median"])],
            [float(winner["qps_median"])],
            facecolors="none",
            edgecolors="#2ca02c",
            marker="^",
            s=70,
            linewidths=1.4,
            zorder=5,
        )
        axis.axvspan(TARGETS[workload], TARGETS[workload] + TARGET_WINDOW, color="#777777", alpha=0.10)
        axis.set_title(labels[workload], fontsize=8.5)
        axis.set_xlabel("Recall@10", fontsize=8)
        axis.grid(alpha=0.22)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Queries/s", fontsize=8)
    for extension in ("pdf", "png"):
        figure.savefig(output / f"navix_wd_qps_recall.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_tex(output: Path, rows: list[dict[str, object]]) -> None:
    labels = {"yfcc": "YFCC-10M", "em": "ArXiv-large EM", "emis": "ArXiv-large EMIS", "r": "ArXiv-large R"}
    lines = ["% Generated by a100_k10_wd_all/workflow.py; do not edit."]
    for workload in WORKLOADS:
        row = next(item for item in rows if item["workload"] == workload)
        qps = float(row["qps_median"])
        qps_text = f"{qps / 1000:.1f}K" if qps >= 1000 else f"{qps:.0f}"
        recall = float(row["recall_median"])
        recall_text = f"{recall:.4f}" if row["within_target_window"] else f"max {recall:.4f}"
        lines.append(f"{labels[workload]} & {qps_text} & {recall_text} \\\\")
    (output / "navix_wd_target_rows.tex").write_text("\n".join(lines) + "\n")


def copy_tree_files(source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    if not source.is_dir():
        return copied
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def make_manifest(root: Path, metadata: dict[str, object]) -> None:
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    ]
    (root / "manifest.json").write_text(
        json.dumps({**metadata, "files": files}, indent=2) + "\n"
    )


def validate_hash_manifest(root: Path) -> dict[str, object]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    listed = {str(row["path"]): row for row in manifest.get("files", [])}
    actual = {
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file() and item != path
    }
    if set(listed) != actual:
        raise ValueError(
            f"bundle manifest mismatch: missing={sorted(set(listed)-actual)}, "
            f"extra={sorted(actual-set(listed))}"
        )
    for relative, record in listed.items():
        item = root / relative
        if item.stat().st_size != int(record["bytes"]) or sha256(item) != record["sha256"]:
            raise ValueError(f"bundle hash mismatch: {relative}")
    return manifest


def bundle(root: Path) -> None:
    results = root / "analysis" / "results.json"
    if not results.is_file() or json.loads(results.read_text()).get("status") != "PASS":
        raise FileNotFoundError("analyze the all-workload W*D rerun before bundling")
    destination = root / "paper_gpu_bundle_k10_wd_all"
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable bundle: {destination}")
    destination.mkdir(parents=True)
    sources = {
        "analysis": root / "analysis",
        "state": root / "state",
        "provenance": root / "provenance",
        "frontier": root / "frontier",
        "matched_recall": root / "matched_recall",
        "controls": root / "controls",
        "diagnostics": root / "diagnostics",
    }
    for name, source in sources.items():
        copy_tree_files(source, destination / name)
    make_manifest(
        destination,
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": EXPERIMENT,
            "profile": load_profile(),
            "execution_contract": {
                "k": K,
                "max_queries": MAX_QUERIES,
                "throughput_queries": 10_000,
                "throughput_shards": [2_048, 2_048, 2_048, 2_048, 1_808],
                "reported_repetitions": 3,
                "navix_seed_policy": "search_width * graph_degree",
            },
        },
    )
    validate_hash_manifest(destination)
    print(destination)


def resolve_bundle(source: Path, temporary: Path, names: tuple[str, ...]) -> Path:
    if source.is_dir():
        for name in names:
            candidate = source if source.name == name else source / name
            if (candidate / "manifest.json").is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"cannot find bundle root under {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(
            member.name.startswith("/") or ".." in Path(member.name).parts
            for member in members
        ):
            raise ValueError(f"unsafe or empty archive: {source}")
        archive.extractall(temporary, filter="data")
    for name in names:
        candidate = temporary / name
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise ValueError(f"archive {source} does not contain one of {names}")


def replace_method_rows(
    destination: Path,
    replacements: list[dict[str, str]],
    *,
    predicate,
) -> list[dict[str, str]]:
    original = read_csv(destination)
    fields = list(original[0])
    if set(replacements[0]) != set(fields):
        raise ValueError(
            f"replacement schema mismatch for {destination}: "
            f"old={fields}, new={list(replacements[0])}"
        )
    merged = [row for row in original if not predicate(row)] + replacements
    write_csv(destination, merged, fields)
    return [row for row in original if predicate(row)]


def merge_bundle(reference_source: Path, wd_source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable merged bundle: {output}")
    with tempfile.TemporaryDirectory(prefix="retrieve-wd-merge-") as temporary_name:
        temporary = Path(temporary_name)
        reference = resolve_bundle(reference_source, temporary / "reference", ("paper_gpu_bundle",))
        wd = resolve_bundle(
            wd_source,
            temporary / "wd",
            ("paper_gpu_bundle_k10_wd_all",),
        )
        reference_manifest = validate_hash_manifest(reference)
        wd_manifest = validate_hash_manifest(wd)
        reference_b0_provenance = json.loads(
            (reference / "b0" / "provenance.json").read_text()
        )
        reference_k = int(
            reference_b0_provenance.get("run_provenance", {})
            .get("payload", {})
            .get("fixed_contract", {})
            .get("k", -1)
        )
        if reference_k != K:
            raise ValueError("reference paper bundle is not k=10")
        if wd_manifest.get("experiment") != EXPERIMENT:
            raise ValueError("unexpected W*D overlay experiment")
        shutil.copytree(reference, output)
        (output / "manifest.json").unlink()

        ablation = output / "seed_ablation"
        ablation.mkdir(parents=True, exist_ok=True)
        copy_tree_files(wd, ablation / "wd_all")

        new_b0 = read_csv(wd / "analysis" / "b0_navix_wd.csv")
        old_b0 = replace_method_rows(
            output / "b0" / "summary_points.csv",
            new_b0,
            predicate=lambda row: row["group"] == "b0"
            and row["phase"] == "throughput"
            and row["method"] == METHOD,
        )
        write_csv(ablation / "k_seed_b0.csv", old_b0, list(old_b0[0]))

        new_selected = read_csv(wd / "analysis" / "selected_navix_wd.csv")
        old_selected = replace_method_rows(
            output / "matched_recall" / "selected_points.csv",
            new_selected,
            predicate=lambda row: row["method"] == METHOD,
        )
        write_csv(ablation / "k_seed_selected_points.csv", old_selected, list(old_selected[0]))

        for filename in ("measurements.csv", "final_summary.csv"):
            replacement = read_csv(wd / "matched_recall" / "analysis" / filename)
            old = replace_method_rows(
                output / "matched_recall" / filename,
                replacement,
                predicate=lambda row: row["method"] == METHOD,
            )
            write_csv(ablation / f"k_seed_{filename}", old, list(old[0]))

        old_b0_provenance = output / "b0" / "provenance.json"
        old_matched_provenance = output / "matched_recall" / "provenance.json"
        shutil.copy2(old_b0_provenance, ablation / "k_seed_b0_provenance.json")
        shutil.copy2(old_matched_provenance, ablation / "k_seed_matched_provenance.json")
        composite = {
            "schema_version": 1,
            "experiment": "retrieve_workshop_a100_composite_wd",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference_bundle": {
                "path": str(reference_source.resolve()),
                "sha256": sha256(reference_source) if reference_source.is_file() else None,
                "manifest_sha256": sha256(reference / "manifest.json"),
            },
            "wd_bundle": {
                "path": str(wd_source.resolve()),
                "sha256": sha256(wd_source) if wd_source.is_file() else None,
                "manifest_sha256": sha256(wd / "manifest.json"),
            },
            "reuse_contract": (
                "Base, Retain, exact scan, dataset statistics, max-query evidence, and the "
                "representative resource table are copied byte-for-byte from the reviewed "
                "reference. NaviX B0 and target rows come from the W*D rerun; its paired "
                "diagnostics and resource-only observations remain separate ablation evidence."
            ),
        }
        old_b0_provenance.write_text(json.dumps({**composite, "component": "b0"}, indent=2) + "\n")
        old_matched_provenance.write_text(
            json.dumps({**composite, "component": "matched_recall"}, indent=2) + "\n"
        )

        selected = read_csv(output / "matched_recall" / "selected_points.csv")
        exact = read_csv(output / "exact_scan" / "exact_summary.csv")
        b0 = read_csv(output / "b0" / "summary_points.csv")
        from bundle import write_gpu_plot, write_headline_tex

        write_gpu_plot(output, b0, selected, exact)
        write_headline_tex(output, selected)
        (output / "gpu_headline_inputs.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "navix_seed_policy": "search_width * graph_degree",
                    "matched_recall_points": selected,
                    "exact_scan_points": exact,
                    "composite_provenance": composite,
                },
                indent=2,
            )
            + "\n"
        )
        claims_path = output / "claim_to_source.json"
        claims = json.loads(claims_path.read_text()) if claims_path.is_file() else {}
        claims.update(
            {
                "b0_qps_recall_and_seed_controls": "b0/summary_points.csv",
                "matched_recall_headlines_and_parameters": "matched_recall/selected_points.csv",
                "navix_wd_seed_ablation": "seed_ablation/wd_all/analysis/paired_controls.csv",
                "navix_wd_mechanism_diagnostics": "seed_ablation/wd_all/analysis/diagnostics.csv",
                "navix_wd_dynamic_work": (
                    "seed_ablation/wd_all/analysis/navix_resource_work.csv"
                ),
            }
        )
        claims.pop("serialized_single_query_latency", None)
        claims_path.write_text(json.dumps(claims, indent=2) + "\n")
        make_manifest(
            output,
            {
                "schema_version": 2,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "profile": reference_manifest["profile"],
                "execution_contract": {
                    **reference_manifest["execution_contract"],
                    "k": K,
                    "navix_seed_policy": "search_width * graph_degree",
                },
                "composite_sources": composite,
            },
        )
        validate_hash_manifest(output)
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--root", type=Path, required=True)
    initialize.add_argument("--data-root", type=Path, required=True)
    initialize.add_argument("--reference-selected", type=Path, required=True)
    initialize.add_argument("--reference-provenance", type=Path, required=True)

    frontier = subparsers.add_parser("validate-frontier")
    frontier.add_argument("--root", type=Path, required=True)

    controls = subparsers.add_parser("create-controls")
    controls.add_argument("--root", type=Path, required=True)
    controls.add_argument("--data-root", type=Path, required=True)
    controls.add_argument("--selected", type=Path, required=True)

    diagnostics = subparsers.add_parser("create-diagnostics")
    diagnostics.add_argument("--root", type=Path, required=True)
    diagnostics.add_argument("--data-root", type=Path, required=True)
    diagnostics.add_argument("--selected", type=Path, required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--root", type=Path, required=True)

    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--root", type=Path, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--reference-bundle", type=Path, required=True)
    merge.add_argument("--wd-bundle", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "initialize":
        create_frontier(
            args.root.resolve(),
            args.data_root.resolve(),
            args.reference_selected.resolve(),
            args.reference_provenance.resolve(),
        )
    elif args.command == "validate-frontier":
        validate_frontier(args.root.resolve())
    elif args.command == "create-controls":
        create_controls(args.root.resolve(), args.data_root.resolve(), args.selected.resolve())
    elif args.command == "create-diagnostics":
        create_diagnostics(args.root.resolve(), args.data_root.resolve(), args.selected.resolve())
    elif args.command == "analyze":
        analyze(args.root.resolve())
    elif args.command == "bundle":
        bundle(args.root.resolve())
    else:
        merge_bundle(
            args.reference_bundle.resolve(),
            args.wd_bundle.resolve(),
            args.output.resolve(),
        )


if __name__ == "__main__":
    main()
