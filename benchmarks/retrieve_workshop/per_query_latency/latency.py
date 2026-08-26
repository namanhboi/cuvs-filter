#!/usr/bin/env python3
"""Generate and analyze serialized, one-query CAGRA latency experiments.

This pipeline is deliberately benchmark-private.  It freezes the twelve selected Recall@10 graph
configurations, issues exactly one query per public search call, and records host API-to-GPU-
completion time without changing any production CAGRA kernel or public cuVS API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_GRAPH_DIR = SCRIPT_DIR.parent / "gpu_graph"
sys.path.insert(0, str(GPU_GRAPH_DIR))

from dataset_profile import load_profile, profile_record
from generate_configs import (
    config_payload,
    dataset_paths,
    search_point,
)

WORKLOADS = ("yfcc", "em", "emis", "r")
GRAPH_METHODS = (
    "default_cagra",
    "default_cagra_accumulator",
    "navix_reference",
)
EXACT_METHOD = "cuvs_brute_force_knn"
METHODS = GRAPH_METHODS + (EXACT_METHOD,)
METHOD_LABELS = {
    "default_cagra": "CAGRA-Base",
    "default_cagra_accumulator": "CAGRA-Retain",
    "navix_reference": "CAGRA-NaviX",
    EXACT_METHOD: "Masked cuVS Brute Force KNN",
}
WORKLOAD_LABELS = {
    "yfcc": "YFCC-10M",
    "em": "ArXiv-large EM",
    "emis": "ArXiv-large EMIS",
    "r": "ArXiv-large R",
}
METHOD_COLORS = {
    "default_cagra": "#1f77b4",
    "default_cagra_accumulator": "#d62728",
    "navix_reference": "#2ca02c",
    EXACT_METHOD: "#111111",
}
EXPECTED_QUERIES = 10_000
GATE_QUERIES = 1_000
REPETITIONS = 3
WARMUP_QUERIES = 32
TIMER_FLOOR_SAMPLES = 128
TRACE_SCHEMA = 1
TARGETS = {"yfcc": 0.80, "em": 0.95, "emis": 0.95, "r": 0.95}
RECALL_DRIFT_LIMIT = 1.0e-4
QPS_REGRESSION_LIMIT = 0.02


def truth(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(
    path: Path, rows: list[dict[str, object]], fields: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def load_selected(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows = read_csv(path)
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if "selected" in row and not truth(row["selected"]):
            continue
        workload = row.get("workload", "")
        method = row.get("method", "")
        if workload not in WORKLOADS or method not in GRAPH_METHODS:
            continue
        key = (workload, method)
        if key in selected:
            raise ValueError(
                f"duplicate selected point for {workload}/{method}"
            )
        selected[key] = {
            "workload": workload,
            "method": method,
            "itopk": int(row["itopk"]),
            "search_width": int(row["search_width"]),
            "max_iterations": int(row["max_iterations"]),
            "recall_median": float(row["recall_median"]),
            "qps_median": float(row["qps_median"]),
            "target_recall": float(
                row.get("target_recall", TARGETS[workload])
            ),
        }
    expected = {
        (workload, method)
        for workload in WORKLOADS
        for method in GRAPH_METHODS
    }
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(
            f"selected-point matrix mismatch: missing={missing}, extra={extra}"
        )
    return selected


def validate_shards(
    manifest: dict[str, Any], expected_queries: int
) -> list[dict[str, Any]]:
    shards = list(manifest.get("shards", []))
    if not shards:
        raise ValueError("source manifest has no shards")
    cursor = 0
    for number, shard in enumerate(shards):
        first = int(shard["first_query"])
        count = int(shard["query_count"])
        if first != cursor or count <= 0:
            raise ValueError(
                f"invalid shard {number}: first={first}, expected={cursor}, count={count}"
            )
        cursor += count
    if cursor != expected_queries:
        raise ValueError(
            f"shards cover {cursor} queries, expected {expected_queries}"
        )
    return shards


def trace_fields(
    *,
    root: Path,
    stage: str,
    workload: str,
    method: str,
    repetition: int,
    shard_index: int,
    first_query: int,
    query_count: int,
    reference_recall: float,
    output_neighbors: Path | None = None,
) -> dict[str, object]:
    trace = (
        root
        / "traces"
        / stage
        / workload
        / method
        / f"rep_{repetition}"
        / f"shard_{shard_index:02d}.csv"
    )
    row: dict[str, object] = {
        "benchmark_latency_trace_file": str(trace.resolve()),
        "benchmark_latency_workload": workload,
        "benchmark_latency_method": method,
        "benchmark_latency_repetition": repetition,
        "benchmark_latency_shard_index": shard_index,
        "benchmark_latency_global_query_offset": first_query,
        # Rotate complete passes to avoid always assigning the same queries to early-run thermal
        # conditions. Every pass still covers the shard exactly once.
        "benchmark_latency_start_query": (repetition * query_count)
        // REPETITIONS,
        "benchmark_latency_warmup_queries": WARMUP_QUERIES,
        "benchmark_latency_timer_floor_samples": TIMER_FLOOR_SAMPLES,
        "benchmark_latency_reference_recall": reference_recall,
        "benchmark_latency_target_recall": TARGETS[workload],
        "benchmark_latency_source_max_queries": 2_048,
        "max_queries": 1,
    }
    if output_neighbors is not None:
        row["benchmark_output_neighbors_file"] = str(
            output_neighbors.resolve()
        )
        row["benchmark_latency_start_query"] = 0
    return row


def graph_search(
    selected: dict[str, object],
    *,
    max_queries: int,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    row = search_point(
        str(selected["method"]),
        int(selected["itopk"]),
        int(selected["search_width"]),
        int(selected["max_iterations"]),
        k=10,
        max_queries=max_queries,
    )
    if extras:
        row.update(extras)
    return row


def exact_search(extras: dict[str, object] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "exact_control": "bitmap_count_csr_search",
        "native_l2_cutoff_validation": True,
        "resident_bitmap": True,
        # Ignored by Brute Force KNN itself, but required by the trace contract and emitted as
        # provenance alongside graph-search max_queries=1.
        "max_queries": 1,
    }
    if extras:
        row.update(extras)
    return row


def exact_manifest_path(exact_root: Path, workload: str, phase: str) -> Path:
    phase_directory = (
        "throughput_10000" if phase == "throughput" else "correctness_1000"
    )
    if workload == "yfcc":
        return exact_root / "yfcc" / phase_directory / "manifest.json"
    namespace = exact_root / "arxiv-large" / workload
    return namespace / phase_directory / "manifest.json"


def exact_config(
    *,
    workload: str,
    phase: str,
    shard: dict[str, Any],
    searches: list[dict[str, object]],
    marker: Path,
    batch_size: int,
) -> dict[str, object]:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    query_count = int(shard["query_count"])
    return {
        "dataset": {
            "name": (
                f"retrieve-latency-exact-{workload}-{phase}-"
                f"q{int(shard['first_query']):05d}-{query_count:05d}"
            ),
            "base_file": shard["base_file"],
            "query_file": shard["query_file"],
            "groundtruth_neighbors_file": shard["groundtruth_file"],
            "distance": "euclidean",
            "dtype": "float",
            "filter": {"kind": "bitmap", "file": shard["bitmap_file"]},
        },
        "search_basic_param": {"batch_size": batch_size, "k": 10},
        "index": [
            {
                "name": "cuvs-exact-bitmap",
                "algo": "cuvs_brute_force",
                "file": str(marker.resolve()),
                "build_param": {},
                "search_params": searches,
            }
        ],
    }


def case_record(
    slot: int,
    method: str,
    repetition: int,
    search: dict[str, object],
    reference: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "slot": slot,
        "method": method,
        "repetition": repetition,
        "itopk": int(search.get("itopk", 0)),
        "search_width": int(search.get("search_width", 0)),
        "max_iterations": int(search.get("max_iterations", 0)),
        "max_queries": int(search.get("max_queries", 0)),
        "trace": search.get("benchmark_latency_trace_file"),
        "output_neighbors": search.get("benchmark_output_neighbors_file"),
        "reference_recall": None
        if reference is None
        else float(reference["recall_median"]),
        "reference_qps": None
        if reference is None
        else float(reference["qps_median"]),
    }


def generate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    data_root = args.data_root.resolve()
    exact_root = args.exact_data_root.resolve()
    selected_path = args.selected_points.resolve()
    selected = load_selected(selected_path)
    if args.profile:
        # dataset_paths() resolves the active profile through the shared environment contract.
        # Set it explicitly for direct CLI use as well as the shell runner.
        os.environ["RETRIEVE_DATASET_PROFILE"] = str(args.profile.resolve())
    profile = load_profile(args.profile.resolve() if args.profile else None)
    if int(profile["max_queries"]) != 2_048:
        raise ValueError(
            "latency source profile must be the A100 max_queries=2048 profile"
        )

    records: list[dict[str, object]] = []
    config_root = root / "configs"
    marker_root = root / "markers"

    def add_record(
        *,
        stage: str,
        engine: str,
        workload: str,
        shard_index: int,
        first_query: int,
        query_count: int,
        config: Path,
        cases: list[dict[str, object]],
        mode: str,
        repetitions: int,
        min_time: str,
    ) -> None:
        raw = root / "raw" / stage / workload / f"shard_{shard_index:02d}.json"
        records.append(
            {
                "stage": stage,
                "engine": engine,
                "workload": workload,
                "shard_index": shard_index,
                "first_query": first_query,
                "query_count": query_count,
                "config": str(config.resolve()),
                "raw": str(raw.resolve()),
                "cases": cases,
                "mode": mode,
                "benchmark_repetitions": repetitions,
                "benchmark_min_time": min_time,
            }
        )

    for workload in WORKLOADS:
        graph_paths = dataset_paths(data_root, workload, "throughput", 64)
        graph_source = json.loads(graph_paths.manifest.read_text())
        graph_shards = validate_shards(graph_source, EXPECTED_QUERIES)
        for shard_index, shard in enumerate(graph_shards):
            searches: list[dict[str, object]] = []
            cases: list[dict[str, object]] = []
            for repetition in range(REPETITIONS):
                order = GRAPH_METHODS[repetition:] + GRAPH_METHODS[:repetition]
                for method in order:
                    reference = selected[(workload, method)]
                    extras = trace_fields(
                        root=root,
                        stage="trace",
                        workload=workload,
                        method=method,
                        repetition=repetition,
                        shard_index=shard_index,
                        first_query=int(shard["first_query"]),
                        query_count=int(shard["query_count"]),
                        reference_recall=float(reference["recall_median"]),
                    )
                    search = graph_search(
                        reference, max_queries=1, extras=extras
                    )
                    cases.append(
                        case_record(
                            len(searches),
                            method,
                            repetition,
                            search,
                            reference,
                        )
                    )
                    searches.append(search)
            config = (
                config_root
                / "trace_graph"
                / workload
                / f"shard_{shard_index:02d}.json"
            )
            config.parent.mkdir(parents=True, exist_ok=True)
            payload = config_payload(
                workload=workload,
                phase="latency",
                shard=shard,
                paths=graph_paths,
                searches=searches,
                k=10,
            )
            payload["search_basic_param"]["batch_size"] = 1
            write_json(config, payload)
            add_record(
                stage="trace_graph",
                engine="graph",
                workload=workload,
                shard_index=shard_index,
                first_query=int(shard["first_query"]),
                query_count=int(shard["query_count"]),
                config=config,
                cases=cases,
                mode="latency",
                repetitions=1,
                min_time="0.001s",
            )

        exact_path = exact_manifest_path(exact_root, workload, "throughput")
        exact_source = json.loads(exact_path.read_text())
        if exact_source.get("method") != "cuvs_brute_force_bitmap":
            raise ValueError(f"stale exact manifest: {exact_path}")
        exact_shards = validate_shards(exact_source, EXPECTED_QUERIES)
        for shard_index, source_shard in enumerate(exact_shards):
            shard = dict(source_shard)
            shard["base_file"] = exact_source["base_file"]
            searches = []
            cases = []
            for repetition in range(REPETITIONS):
                extras = trace_fields(
                    root=root,
                    stage="trace",
                    workload=workload,
                    method=EXACT_METHOD,
                    repetition=repetition,
                    shard_index=shard_index,
                    first_query=int(shard["first_query"]),
                    query_count=int(shard["query_count"]),
                    reference_recall=1.0,
                )
                search = exact_search(extras)
                cases.append(
                    case_record(
                        len(searches), EXACT_METHOD, repetition, search, None
                    )
                )
                searches.append(search)
            config = (
                config_root
                / "trace_exact"
                / workload
                / f"shard_{shard_index:02d}.json"
            )
            write_json(
                config,
                exact_config(
                    workload=workload,
                    phase="latency",
                    shard=shard,
                    searches=searches,
                    marker=marker_root / f"{workload}.index",
                    batch_size=1,
                ),
            )
            add_record(
                stage="trace_exact",
                engine="exact",
                workload=workload,
                shard_index=shard_index,
                first_query=int(shard["first_query"]),
                query_count=int(shard["query_count"]),
                config=config,
                cases=cases,
                mode="latency",
                repetitions=1,
                min_time="0.001s",
            )

        # The 1,000-query gate compares the ordinary max_queries=2048 batch with exactly one
        # query/call and max_queries=1 before the expensive 10,000-query traces are accepted.
        gate_paths = dataset_paths(data_root, workload, "correctness", 64)
        gate_source = json.loads(gate_paths.manifest.read_text())
        gate_shards = validate_shards(gate_source, GATE_QUERIES)
        if len(gate_shards) != 1:
            raise ValueError(
                f"latency gate expects one 1,000-query shard for {workload}"
            )
        gate_shard = gate_shards[0]
        for serial in (False, True):
            searches = []
            cases = []
            stage = "gate_serial_graph" if serial else "gate_batch_graph"
            for method in GRAPH_METHODS:
                reference = selected[(workload, method)]
                output = (
                    root
                    / "gate"
                    / "neighbors"
                    / workload
                    / f"{method}_{'serial' if serial else 'batch'}.ibin"
                )
                extras: dict[str, object] = {
                    "benchmark_output_neighbors_file": str(output.resolve())
                }
                if serial:
                    extras.update(
                        trace_fields(
                            root=root,
                            stage="gate",
                            workload=workload,
                            method=method,
                            repetition=0,
                            shard_index=0,
                            first_query=0,
                            query_count=GATE_QUERIES,
                            reference_recall=float(reference["recall_median"]),
                            output_neighbors=output,
                        )
                    )
                search = graph_search(
                    reference,
                    max_queries=1 if serial else 2_048,
                    extras=extras,
                )
                cases.append(
                    case_record(len(searches), method, 0, search, reference)
                )
                searches.append(search)
            config = config_root / stage / workload / "shard_00.json"
            payload = config_payload(
                workload=workload,
                phase="latency-gate",
                shard=gate_shard,
                paths=gate_paths,
                searches=searches,
                k=10,
            )
            payload["search_basic_param"]["batch_size"] = (
                1 if serial else GATE_QUERIES
            )
            write_json(config, payload)
            add_record(
                stage=stage,
                engine="graph",
                workload=workload,
                shard_index=0,
                first_query=0,
                query_count=GATE_QUERIES,
                config=config,
                cases=cases,
                mode="latency" if serial else "throughput",
                repetitions=1,
                min_time="0.001s",
            )

        gate_exact_path = exact_manifest_path(
            exact_root, workload, "correctness"
        )
        gate_exact_source = json.loads(gate_exact_path.read_text())
        gate_exact_shards = validate_shards(gate_exact_source, GATE_QUERIES)
        if len(gate_exact_shards) != 1:
            raise ValueError(
                f"exact latency gate expects one 1,000-query shard for {workload}"
            )
        exact_shard = dict(gate_exact_shards[0])
        exact_shard["base_file"] = gate_exact_source["base_file"]
        for serial in (False, True):
            stage = "gate_serial_exact" if serial else "gate_batch_exact"
            output = (
                root
                / "gate"
                / "neighbors"
                / workload
                / f"{EXACT_METHOD}_{'serial' if serial else 'batch'}.ibin"
            )
            extras: dict[str, object] = {
                "benchmark_output_neighbors_file": str(output.resolve())
            }
            if serial:
                extras.update(
                    trace_fields(
                        root=root,
                        stage="gate",
                        workload=workload,
                        method=EXACT_METHOD,
                        repetition=0,
                        shard_index=0,
                        first_query=0,
                        query_count=GATE_QUERIES,
                        reference_recall=1.0,
                        output_neighbors=output,
                    )
                )
            search = exact_search(extras)
            config = config_root / stage / workload / "shard_00.json"
            write_json(
                config,
                exact_config(
                    workload=workload,
                    phase="latency-gate",
                    shard=exact_shard,
                    searches=[search],
                    marker=marker_root / f"{workload}.index",
                    batch_size=1 if serial else GATE_QUERIES,
                ),
            )
            add_record(
                stage=stage,
                engine="exact",
                workload=workload,
                shard_index=0,
                first_query=0,
                query_count=GATE_QUERIES,
                config=config,
                cases=[case_record(0, EXACT_METHOD, 0, search, None)],
                mode="latency" if serial else "throughput",
                repetitions=1,
                min_time="0.001s",
            )

        # Ordinary batched rerun of the twelve selected graph points. This is the explicit guard
        # that benchmark-only tracing support did not regress the paper's existing QPS path.
        for shard_index, shard in enumerate(graph_shards):
            searches = []
            cases = []
            for method in GRAPH_METHODS:
                reference = selected[(workload, method)]
                search = graph_search(reference, max_queries=2_048)
                cases.append(
                    case_record(len(searches), method, 0, search, reference)
                )
                searches.append(search)
            config = (
                config_root
                / "throughput_gate"
                / workload
                / f"shard_{shard_index:02d}.json"
            )
            write_json(
                config,
                config_payload(
                    workload=workload,
                    phase="latency-throughput-gate",
                    shard=shard,
                    paths=graph_paths,
                    searches=searches,
                    k=10,
                ),
            )
            add_record(
                stage="throughput_gate",
                engine="graph",
                workload=workload,
                shard_index=shard_index,
                first_query=int(shard["first_query"]),
                query_count=int(shard["query_count"]),
                config=config,
                cases=cases,
                mode="throughput",
                repetitions=3,
                min_time="0.10s",
            )

    manifest = {
        "schema_version": 1,
        "experiment": "retrieve_serialized_per_query_latency",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_record(profile),
        "selected_points": str(selected_path),
        "selected_points_sha256": sha256(selected_path),
        "contracts": {
            "k": 10,
            "source_max_queries": 2_048,
            "serialized_max_queries": 1,
            "queries_per_search_call": 1,
            "query_samples_per_workload_method": EXPECTED_QUERIES,
            "complete_passes": REPETITIONS,
            "warmup_calls_per_config": WARMUP_QUERIES,
            "timer_floor_samples_per_config": TIMER_FLOOR_SAMPLES,
            "latency_definition": "host API entry through synchronized GPU completion",
            "percentile_unit": "per-query median across three complete passes",
            "percentile_rule": "nearest-rank",
            "trace_overhead_subtraction": False,
            "recall_drift_limit": RECALL_DRIFT_LIMIT,
            "qps_regression_limit": QPS_REGRESSION_LIMIT,
        },
        "records": records,
    }
    write_json(root / "manifest.json", manifest)
    print(root / "manifest.json")


SLOT_PATTERN = re.compile(r"/(\d+)/process_time/")


def iteration_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in raw.get("benchmarks", [])
        if row.get("run_type") == "iteration"
    ]


def map_rows_to_cases(
    record: dict[str, Any], *, all_repetitions: bool
) -> list[tuple[dict, dict]]:
    raw_path = Path(record["raw"])
    raw = json.loads(raw_path.read_text())
    rows = iteration_rows(raw)
    cases = list(record["cases"])
    expected_repetitions = int(record["benchmark_repetitions"])
    expected = len(cases) * expected_repetitions
    if len(rows) != expected:
        raise ValueError(
            f"{raw_path}: found {len(rows)}/{expected} iteration rows"
        )
    mapped: list[tuple[dict, dict]] = []
    for order, row in enumerate(rows):
        if row.get("error_occurred") or row.get("skipped"):
            raise ValueError(
                f"benchmark failure in {raw_path}: {row.get('error_message', row)}"
            )
        match = SLOT_PATTERN.search(str(row.get("name", "")))
        slot = int(match.group(1)) if match else order // expected_repetitions
        if not 0 <= slot < len(cases):
            raise ValueError(f"cannot map benchmark slot {slot} in {raw_path}")
        case = cases[slot]
        if all_repetitions:
            repetition_index = int(row.get("repetition_index", -1))
            if not 0 <= repetition_index < expected_repetitions:
                raise ValueError(
                    f"bad repetition index in {raw_path}: {repetition_index}"
                )
        mapped.append((case, row))
    return mapped


def validate_correctness(
    method: str, row: dict[str, Any], raw_path: Path
) -> None:
    required_zero = (
        "FilterViolations",
        "InvalidSentinelErrors",
        "SentinelOrderErrors",
    )
    for field in required_zero:
        if float(row.get(field, 0.0)) != 0.0:
            raise ValueError(f"{raw_path}: {method} has nonzero {field}")
    if (
        method != "default_cagra"
        and float(row.get("DuplicateOutputQueries", 0.0)) != 0.0
    ):
        raise ValueError(f"{raw_path}: {method} emitted duplicate valid IDs")
    if (
        method in ("navix_reference", EXACT_METHOD)
        and float(row.get("InvalidSentinelDistanceErrors", 0.0)) != 0.0
    ):
        raise ValueError(
            f"{raw_path}: {method} has noncanonical invalid-slot distances"
        )
    if method == EXACT_METHOD:
        if float(row.get("NativeL2CutoffRecall", 0.0)) < 0.9999:
            raise ValueError(
                f"{raw_path}: exact native-L2 cutoff recall is below 0.9999"
            )
        if float(row.get("NativeL2CutoffErrors", 1.0)) > 0.0001:
            raise ValueError(
                f"{raw_path}: exact native-L2 cutoff errors exceed 0.0001"
            )
        if float(row.get("NativeL2StrictPrefixErrors", 1.0)) != 0.0:
            raise ValueError(
                f"{raw_path}: exact scan missed a strict-prefix neighbor"
            )


def validate_trace(
    case: dict[str, Any], record: dict[str, Any]
) -> dict[str, object]:
    trace_path = Path(str(case["trace"]))
    rows = read_csv(trace_path)
    query_rows = [row for row in rows if row["record_kind"] == "query"]
    floor_rows = [row for row in rows if row["record_kind"] == "timer_floor"]
    expected = int(record["query_count"])
    first = int(record["first_query"])
    if len(query_rows) != expected:
        raise ValueError(
            f"{trace_path}: found {len(query_rows)}/{expected} query samples"
        )
    if len(floor_rows) != TIMER_FLOOR_SAMPLES:
        raise ValueError(
            f"{trace_path}: found {len(floor_rows)}/{TIMER_FLOOR_SAMPLES} timer-floor samples"
        )
    local_ids = [int(row["query_local_id"]) for row in query_rows]
    global_ids = [int(row["query_global_id"]) for row in query_rows]
    orders = [int(row["call_order"]) for row in query_rows]
    if sorted(local_ids) != list(range(expected)):
        raise ValueError(
            f"{trace_path}: local query IDs do not form a complete shard pass"
        )
    if sorted(global_ids) != list(range(first, first + expected)):
        raise ValueError(
            f"{trace_path}: global query IDs do not match the shard"
        )
    if orders != list(range(expected)):
        raise ValueError(f"{trace_path}: call order is not contiguous")
    for row in rows:
        if int(row["schema_version"]) != TRACE_SCHEMA:
            raise ValueError(f"{trace_path}: unsupported trace schema")
        if (
            row["workload"] != record["workload"]
            or row["method"] != case["method"]
        ):
            raise ValueError(f"{trace_path}: trace identity mismatch")
        if int(row["repetition"]) != int(case["repetition"]):
            raise ValueError(f"{trace_path}: trace repetition mismatch")
        if int(row["shard_index"]) != int(record["shard_index"]):
            raise ValueError(f"{trace_path}: trace shard mismatch")
        if int(row["max_queries"]) != 1 or int(row["k"]) != 10:
            raise ValueError(f"{trace_path}: wrong serialized search contract")
        if int(row["host_latency_ns"]) <= 0:
            raise ValueError(f"{trace_path}: nonpositive host latency")
    return {
        "trace": str(trace_path),
        "queries": len(query_rows),
        "timer_floor_samples": len(floor_rows),
    }


def validate_one(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    records = [
        row
        for row in manifest["records"]
        if Path(row["raw"]) == args.raw.resolve()
    ]
    if len(records) != 1:
        raise ValueError(f"cannot resolve one record for {args.raw}")
    record = records[0]
    mapped = map_rows_to_cases(
        record, all_repetitions=int(record["benchmark_repetitions"]) > 1
    )
    for case, row in mapped:
        validate_correctness(str(case["method"]), row, args.raw)
    if str(record["stage"]).startswith("trace_") or str(
        record["stage"]
    ).startswith("gate_serial"):
        for case in record["cases"]:
            validate_trace(case, record)
    print(json.dumps({"raw": str(args.raw), "iteration_rows": len(mapped)}))


def load_ibin(path: Path) -> tuple[int, int, list[list[int]]]:
    import struct

    with path.open("rb") as source:
        header = source.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated neighbor file: {path}")
        rows, cols = struct.unpack("<II", header)
        payload = source.read()
    if len(payload) != rows * cols * 4:
        raise ValueError(f"neighbor file size mismatch: {path}")
    values = struct.unpack(f"<{rows * cols}I", payload)
    return (
        rows,
        cols,
        [list(values[i * cols : (i + 1) * cols]) for i in range(rows)],
    )


def observation(record: dict, case: dict, row: dict) -> dict[str, object]:
    return {
        "stage": record["stage"],
        "engine": record["engine"],
        "workload": record["workload"],
        "shard_index": record["shard_index"],
        "first_query": record["first_query"],
        "query_count": record["query_count"],
        "method": case["method"],
        "repetition": case["repetition"],
        "itopk": case["itopk"],
        "search_width": case["search_width"],
        "max_iterations": case["max_iterations"],
        "max_queries": case["max_queries"],
        "recall": float(row.get("ValidGTRecall", row.get("Recall", 0.0))),
        "valid_gt_fraction": float(row.get("ValidGTFraction", 1.0)),
        "qps": float(row.get("items_per_second", 0.0)),
        "filter_violations": float(row.get("FilterViolations", 0.0)),
        "invalid_sentinel_errors": float(
            row.get("InvalidSentinelErrors", 0.0)
        ),
        "sentinel_order_errors": float(row.get("SentinelOrderErrors", 0.0)),
        "invalid_sentinel_distance_errors": float(
            row.get("InvalidSentinelDistanceErrors", 0.0)
        ),
        "duplicate_output_queries": float(
            row.get("DuplicateOutputQueries", 0.0)
        ),
        "underfilled_queries": float(row.get("UnderfilledQueries", 0.0)),
        "missing_result_slots": float(row.get("MissingResultSlots", 0.0)),
        "native_l2_cutoff_recall": float(row.get("NativeL2CutoffRecall", 0.0)),
        "native_l2_cutoff_errors": float(row.get("NativeL2CutoffErrors", 0.0)),
        "reference_recall": case["reference_recall"],
        "reference_qps": case["reference_qps"],
    }


def collect_observations(
    manifest: dict, stages: set[str]
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for record in manifest["records"]:
        if record["stage"] not in stages:
            continue
        raw_path = Path(record["raw"])
        for case, row in map_rows_to_cases(
            record, all_repetitions=int(record["benchmark_repetitions"]) > 1
        ):
            validate_correctness(str(case["method"]), row, raw_path)
            item = observation(record, case, row)
            if int(record["benchmark_repetitions"]) > 1:
                item["repetition"] = int(row["repetition_index"])
            observations.append(item)
    return observations


def aggregate_recall(rows: list[dict[str, object]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = (
            int(row["query_count"])
            * 10
            * float(row.get("valid_gt_fraction", 1.0))
        )
        numerator += float(row["recall"]) * weight
        denominator += weight
    if denominator <= 0:
        raise ValueError("cannot aggregate recall with no valid ground truth")
    return numerator / denominator


def analyze_gate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    observations = collect_observations(
        manifest,
        {
            "gate_batch_graph",
            "gate_serial_graph",
            "gate_batch_exact",
            "gate_serial_exact",
        },
    )
    by_key = {
        (str(row["stage"]), str(row["workload"]), str(row["method"])): row
        for row in observations
    }
    summary: list[dict[str, object]] = []
    failures: list[str] = []
    for workload in WORKLOADS:
        for method in METHODS:
            prefix = "exact" if method == EXACT_METHOD else "graph"
            batch = by_key[(f"gate_batch_{prefix}", workload, method)]
            serial = by_key[(f"gate_serial_{prefix}", workload, method)]
            recall_delta = abs(
                float(batch["recall"]) - float(serial["recall"])
            )
            batch_case = next(
                case
                for record in manifest["records"]
                if record["stage"] == f"gate_batch_{prefix}"
                and record["workload"] == workload
                for case in record["cases"]
                if case["method"] == method
            )
            serial_case = next(
                case
                for record in manifest["records"]
                if record["stage"] == f"gate_serial_{prefix}"
                and record["workload"] == workload
                for case in record["cases"]
                if case["method"] == method
            )
            batch_rows, batch_k, batch_ids = load_ibin(
                Path(str(batch_case["output_neighbors"]))
            )
            serial_rows, serial_k, serial_ids = load_ibin(
                Path(str(serial_case["output_neighbors"]))
            )
            if (batch_rows, batch_k) != (serial_rows, serial_k):
                raise ValueError(
                    f"gate neighbor shape mismatch for {workload}/{method}"
                )
            sentinel = (1 << 32) - 1
            matches = sum(
                {value for value in batch_row if value != sentinel}
                == {value for value in serial_row if value != sentinel}
                for batch_row, serial_row in zip(
                    batch_ids, serial_ids, strict=True
                )
            )
            set_match_fraction = matches / batch_rows
            # Exact batched and one-query GEMM paths can exchange legal rank-k ties. Graph methods
            # are required to preserve their complete output sets; every method must preserve
            # aggregate recall to the paper-level tolerance.
            passed = recall_delta <= RECALL_DRIFT_LIMIT and (
                method == EXACT_METHOD or set_match_fraction == 1.0
            )
            if not passed:
                failures.append(
                    f"{workload}/{method}: recall_delta={recall_delta}, "
                    f"set_match_fraction={set_match_fraction}"
                )
            summary.append(
                {
                    "workload": workload,
                    "method": method,
                    "batch_recall": batch["recall"],
                    "serialized_recall": serial["recall"],
                    "absolute_recall_delta": recall_delta,
                    "output_set_match_fraction": set_match_fraction,
                    "passed": passed,
                }
            )
    output = root / "gate" / "analysis"
    fields = (
        "workload",
        "method",
        "batch_recall",
        "serialized_recall",
        "absolute_recall_delta",
        "output_set_match_fraction",
        "passed",
    )
    write_csv(output / "batch_vs_serial_gate.csv", summary, fields)
    write_json(
        output / "batch_vs_serial_gate.json",
        {
            "schema_version": 1,
            "status": "FAIL" if failures else "PASS",
            "recall_drift_limit": RECALL_DRIFT_LIMIT,
            "rows": summary,
            "failures": failures,
        },
    )
    if failures:
        raise ValueError(
            "batch-vs-serialized gate failed:\n" + "\n".join(failures)
        )
    print(output)


def analyze_throughput_gate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    rows = collect_observations(manifest, {"throughput_gate"})
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (str(row["workload"]), str(row["method"]), int(row["repetition"]))
        ].append(row)
    aggregates: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(
        list
    )
    for (workload, method, repetition), shards in grouped.items():
        if (
            len(shards) != 5
            or sum(int(row["query_count"]) for row in shards)
            != EXPECTED_QUERIES
        ):
            raise ValueError(
                f"incomplete throughput gate for {workload}/{method}/rep{repetition}"
            )
        seconds = sum(
            int(row["query_count"]) / float(row["qps"]) for row in shards
        )
        aggregates[(workload, method)].append(
            {
                "qps": EXPECTED_QUERIES / seconds,
                "recall": aggregate_recall(shards),
            }
        )
    summaries: list[dict[str, object]] = []
    failures: list[str] = []
    for workload in WORKLOADS:
        for method in GRAPH_METHODS:
            points = aggregates[(workload, method)]
            if len(points) != REPETITIONS:
                raise ValueError(
                    f"missing throughput repetitions for {workload}/{method}"
                )
            qps = statistics.median(point["qps"] for point in points)
            recall = statistics.median(point["recall"] for point in points)
            reference = next(
                row
                for row in rows
                if row["workload"] == workload and row["method"] == method
            )
            reference_qps = float(reference["reference_qps"])
            reference_recall = float(reference["reference_recall"])
            ratio = qps / reference_qps
            recall_delta = abs(recall - reference_recall)
            passed = (
                ratio >= 1.0 - QPS_REGRESSION_LIMIT
                and recall_delta <= RECALL_DRIFT_LIMIT
            )
            if not passed:
                failures.append(
                    f"{workload}/{method}: qps_ratio={ratio}, recall_delta={recall_delta}"
                )
            summaries.append(
                {
                    "workload": workload,
                    "method": method,
                    "qps_median": qps,
                    "reference_qps": reference_qps,
                    "qps_ratio": ratio,
                    "recall_median": recall,
                    "reference_recall": reference_recall,
                    "absolute_recall_delta": recall_delta,
                    "passed": passed,
                }
            )
    output = root / "throughput_gate" / "analysis"
    fields = tuple(summaries[0])
    write_csv(output / "throughput_regression_gate.csv", summaries, fields)
    write_json(
        output / "throughput_regression_gate.json",
        {
            "schema_version": 1,
            "status": "FAIL" if failures else "PASS",
            "maximum_qps_regression": QPS_REGRESSION_LIMIT,
            "recall_drift_limit": RECALL_DRIFT_LIMIT,
            "rows": summaries,
            "failures": failures,
        },
    )
    if failures:
        raise ValueError(
            "throughput regression gate failed:\n" + "\n".join(failures)
        )
    print(output)


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def analyze(args: argparse.Namespace) -> None:
    # Import plotting only in the analysis path, so config and raw-validation commands remain
    # usable in minimal benchmark environments.
    import matplotlib.pyplot as plt

    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    gate = json.loads(
        (root / "gate/analysis/batch_vs_serial_gate.json").read_text()
    )
    throughput_gate = json.loads(
        (
            root / "throughput_gate/analysis/throughput_regression_gate.json"
        ).read_text()
    )
    if gate["status"] != "PASS" or throughput_gate["status"] != "PASS":
        raise ValueError(
            "latency analysis requires both preflight gates to pass"
        )

    observations = collect_observations(
        manifest, {"trace_graph", "trace_exact"}
    )
    recall_groups: dict[tuple[str, str, int], list[dict[str, object]]] = (
        defaultdict(list)
    )
    samples: dict[tuple[str, str, int], dict[int, int]] = defaultdict(dict)
    floors: dict[tuple[str, str], list[int]] = defaultdict(list)
    validated_traces: list[dict[str, object]] = []
    for record in manifest["records"]:
        if record["stage"] not in ("trace_graph", "trace_exact"):
            continue
        for case in record["cases"]:
            validated_traces.append(validate_trace(case, record))
            for row in read_csv(Path(str(case["trace"]))):
                latency = int(row["host_latency_ns"])
                key = (record["workload"], case["method"])
                if row["record_kind"] == "query":
                    sample_key = (key[0], key[1], int(row["query_global_id"]))
                    repetition = int(row["repetition"])
                    if repetition in samples[sample_key]:
                        raise ValueError(
                            f"duplicate repetition {repetition} for "
                            f"{sample_key[0]}/{sample_key[1]}/query{sample_key[2]}"
                        )
                    samples[sample_key][repetition] = latency
                elif row["record_kind"] == "timer_floor":
                    floors[key].append(latency)
    for row in observations:
        recall_groups[
            (str(row["workload"]), str(row["method"]), int(row["repetition"]))
        ].append(row)

    summaries: list[dict[str, object]] = []
    median_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for workload in WORKLOADS:
        for method in METHODS:
            for query_id in range(EXPECTED_QUERIES):
                by_repetition = samples[(workload, method, query_id)]
                if set(by_repetition) != set(range(REPETITIONS)):
                    raise ValueError(
                        f"{workload}/{method}/query{query_id} has repetitions "
                        f"{sorted(by_repetition)}, expected 0..{REPETITIONS - 1}"
                    )
                values = [
                    by_repetition[repetition]
                    for repetition in range(REPETITIONS)
                ]
                median_ns = float(statistics.median(values))
                median_samples[(workload, method)].append(median_ns)
                sample_rows.append(
                    {
                        "workload": workload,
                        "method": method,
                        "query_id": query_id,
                        "rep0_ns": values[0],
                        "rep1_ns": values[1],
                        "rep2_ns": values[2],
                        "median_ns": median_ns,
                    }
                )
            recalls = []
            for repetition in range(REPETITIONS):
                shards = recall_groups[(workload, method, repetition)]
                if (
                    len(shards) != 5
                    or sum(int(row["query_count"]) for row in shards)
                    != EXPECTED_QUERIES
                ):
                    raise ValueError(
                        f"incomplete trace recall for {workload}/{method}/rep{repetition}"
                    )
                recalls.append(aggregate_recall(shards))
            recall_median = statistics.median(recalls)
            reference_recall = (
                1.0
                if method == EXACT_METHOD
                else float(
                    recall_groups[(workload, method, 0)][0]["reference_recall"]
                )
            )
            recall_delta = abs(recall_median - reference_recall)
            if method != EXACT_METHOD and recall_delta > RECALL_DRIFT_LIMIT:
                failures.append(
                    f"{workload}/{method}: serialized recall {recall_median} differs from "
                    f"batched reference {reference_recall} by {recall_delta}"
                )
            values_ns = median_samples[(workload, method)]
            floor_ns = [float(value) for value in floors[(workload, method)]]
            if len(floor_ns) != 5 * REPETITIONS * TIMER_FLOOR_SAMPLES:
                raise ValueError(
                    f"incomplete timer-floor samples for {workload}/{method}"
                )
            mean_ns = statistics.fmean(values_ns)
            summaries.append(
                {
                    "workload": workload,
                    "method": method,
                    "queries": EXPECTED_QUERIES,
                    "complete_passes": REPETITIONS,
                    "mean_us": mean_ns / 1_000.0,
                    "std_us": statistics.pstdev(values_ns) / 1_000.0,
                    "p50_us": nearest_rank(values_ns, 0.50) / 1_000.0,
                    "p95_us": nearest_rank(values_ns, 0.95) / 1_000.0,
                    "p99_us": nearest_rank(values_ns, 0.99) / 1_000.0,
                    "max_us": max(values_ns) / 1_000.0,
                    "serialized_qps": 1.0e9 / mean_ns,
                    "timer_floor_p50_us": nearest_rank(floor_ns, 0.50)
                    / 1_000.0,
                    "timer_floor_p95_us": nearest_rank(floor_ns, 0.95)
                    / 1_000.0,
                    "serialized_recall_median": recall_median,
                    "batched_reference_recall": reference_recall,
                    "absolute_recall_delta": recall_delta,
                }
            )

    output = root / "analysis"
    sample_fields = (
        "workload",
        "method",
        "query_id",
        "rep0_ns",
        "rep1_ns",
        "rep2_ns",
        "median_ns",
    )
    summary_fields = tuple(summaries[0])
    write_csv(
        output / "per_query_latency_samples.csv", sample_rows, sample_fields
    )
    write_csv(output / "latency_summary.csv", summaries, summary_fields)

    fig, axes = plt.subplots(
        1, 4, figsize=(16.0, 3.7), constrained_layout=True
    )
    handles = []
    labels = []
    for axis, workload in zip(axes, WORKLOADS, strict=True):
        for method in METHODS:
            values_us = sorted(
                value / 1_000.0 for value in median_samples[(workload, method)]
            )
            cumulative = [
                (index + 1) / len(values_us) for index in range(len(values_us))
            ]
            line = axis.plot(
                values_us,
                cumulative,
                color=METHOD_COLORS[method],
                linewidth=1.5,
                label=METHOD_LABELS[method],
            )[0]
            if METHOD_LABELS[method] not in labels:
                handles.append(line)
                labels.append(METHOD_LABELS[method])
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.005)
        axis.set_title(WORKLOAD_LABELS[workload])
        axis.set_xlabel("Serialized latency (us, log scale)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Fraction of queries")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"per_query_latency_cdf.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    tex_lines = []
    for row in summaries:
        tex_lines.append(
            f"{WORKLOAD_LABELS[str(row['workload'])]} & {METHOD_LABELS[str(row['method'])]} & "
            f"{float(row['mean_us']):,.1f} & {float(row['p50_us']):,.1f} & "
            f"{float(row['p95_us']):,.1f} & {float(row['p99_us']):,.1f} & "
            f"{float(row['max_us']):,.1f} \\\\"
        )
    (output / "latency_table_rows.tex").write_text("\n".join(tex_lines) + "\n")
    payload = {
        "schema_version": 1,
        "status": "FAIL" if failures else "PASS",
        "measurement_contract": manifest["contracts"],
        "summary": summaries,
        "validated_trace_files": len(validated_traces),
        "query_trace_rows": len(sample_rows) * REPETITIONS,
        "failures": failures,
    }
    write_json(output / "latency_summary.json", payload)
    if failures:
        raise ValueError(
            "serialized latency validation failed:\n" + "\n".join(failures)
        )
    print(output)


def provenance(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest = root / "manifest.json"
    paths = {
        "manifest": manifest,
        "selected_points": args.selected_points.resolve(),
        "dataset_profile": args.profile.resolve(),
        "cagra_benchmark": args.graph_binary.resolve(),
        "exact_benchmark": args.exact_binary.resolve(),
        "libcuvs": args.libcuvs.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing provenance input {name}: {path}")
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=args.repo.resolve(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=args.repo.resolve(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    write_json(
        root / "provenance" / "run.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_dirty": dirty,
            "files": {
                name: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for name, path in paths.items()
            },
            "contract": {
                "graph_source_max_queries": 2_048,
                "serialized_max_queries": 1,
                "queries_per_call": 1,
                "latency": "host API entry through synchronized GPU completion",
            },
        },
    )
    print(root / "provenance" / "run.json")


def list_records(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    stages = set(args.stage)
    for record in manifest["records"]:
        if stages and record["stage"] not in stages:
            continue
        print(
            "\t".join(
                (
                    str(record["engine"]),
                    str(record["mode"]),
                    str(record["benchmark_repetitions"]),
                    str(record["benchmark_min_time"]),
                    str(record["config"]),
                    str(record["raw"]),
                )
            )
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("generate")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--exact-data-root", type=Path, required=True)
    command.add_argument("--selected-points", type=Path, required=True)
    command.add_argument("--profile", type=Path)
    command.set_defaults(func=generate)

    command = subparsers.add_parser("list-records")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--stage", action="append", default=[])
    command.set_defaults(func=list_records)

    command = subparsers.add_parser("validate-one")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--raw", type=Path, required=True)
    command.set_defaults(func=validate_one)

    for name, function in (
        ("analyze-gate", analyze_gate),
        ("analyze-throughput-gate", analyze_throughput_gate),
        ("analyze", analyze),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.set_defaults(func=function)

    command = subparsers.add_parser("provenance")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--selected-points", type=Path, required=True)
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--graph-binary", type=Path, required=True)
    command.add_argument("--exact-binary", type=Path, required=True)
    command.add_argument("--libcuvs", type=Path, required=True)
    command.set_defaults(func=provenance)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
