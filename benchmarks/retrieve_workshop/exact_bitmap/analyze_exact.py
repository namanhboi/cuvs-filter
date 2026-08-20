#!/usr/bin/env python3
"""Aggregate sharded cuVS exact-bitmap runs and enforce exact-search correctness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

MATRIX_HEADER = struct.Struct("<II")
WORKLOADS = ("yfcc", "em", "emis", "r")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gt_statistics(
    path: Path, base_rows: int, k: int
) -> tuple[int, int, int, int]:
    with path.open("rb") as stream:
        header = stream.read(MATRIX_HEADER.size)
    if len(header) != MATRIX_HEADER.size:
        raise ValueError(f"truncated ground truth: {path}")
    rows, cols = MATRIX_HEADER.unpack(header)
    if (
        cols != k
        or path.stat().st_size != MATRIX_HEADER.size + rows * cols * 4
    ):
        raise ValueError(
            f"ground truth must have exactly k={k} columns: {path}"
        )
    values = np.memmap(
        path,
        dtype="<u4",
        mode="r",
        offset=MATRIX_HEADER.size,
        shape=(rows, cols),
    )
    valid = values < np.uint32(base_rows)
    valid_per_query = np.count_nonzero(valid, axis=1)
    return (
        rows,
        int(np.count_nonzero(valid)),
        int(np.count_nonzero(valid_per_query < k)),
        int(np.sum(k - valid_per_query)),
    )


def iteration_rows(raw_path: Path) -> list[dict[str, object]]:
    raw = json.loads(raw_path.read_text())
    rows = [
        row
        for row in raw.get("benchmarks", [])
        if row.get("run_type") == "iteration"
    ]
    if not rows:
        raise ValueError(f"no benchmark iteration records: {raw_path}")
    for row in rows:
        if row.get("error_occurred") or row.get("skipped"):
            raise ValueError(
                f"failed benchmark record in {raw_path}: {row.get('error_message')}"
            )
    return rows


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        action="append",
        choices=("correctness", "throughput", "smoke"),
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    phases = args.phase or ["correctness", "throughput"]

    shard_rows: list[dict[str, object]] = []
    contracts: dict[tuple[str, str], dict[str, object]] = {}
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(
        list
    )
    for phase in phases:
        config_phase = args.result_root / "configs" / phase
        if not config_phase.exists():
            continue
        for workload_directory in sorted(
            path for path in config_phase.iterdir() if path.is_dir()
        ):
            config_manifest_path = workload_directory / "manifest.json"
            if not config_manifest_path.exists():
                continue
            config_manifest = json.loads(config_manifest_path.read_text())
            workload = str(config_manifest["workload"])
            exact_manifest = json.loads(
                Path(config_manifest["exact_manifest"]).read_text()
            )
            base_rows = int(exact_manifest["base_rows"])
            exact_shards = {
                int(row["shard_number"]): row
                for row in exact_manifest["shards"]
            }
            if set(exact_shards) != set(range(len(exact_shards))):
                raise ValueError(
                    f"exact shard numbers are not contiguous in {config_manifest['exact_manifest']}"
                )
            cursor = 0
            for shard_number in range(len(exact_shards)):
                shard = exact_shards[shard_number]
                first_query = int(shard["first_query"])
                query_count = int(shard["query_count"])
                if first_query != cursor or query_count <= 0:
                    raise ValueError(
                        f"non-contiguous/invalid exact shard {shard_number}: "
                        f"first={first_query}, expected={cursor}, count={query_count}"
                    )
                cursor += query_count
            if cursor != int(exact_manifest["query_rows"]):
                raise ValueError(
                    "exact shard coverage does not match query_rows"
                )
            expected_repetitions = int(config_manifest["expected_repetitions"])
            expected_shards = set(exact_shards)
            configured_shards = {
                int(row["shard_number"]) for row in config_manifest["configs"]
            }
            if configured_shards != expected_shards:
                raise ValueError(
                    f"configured shard set does not match exact manifest: {config_manifest_path}"
                )
            expected_queries = int(config_manifest["expected_queries"])
            if expected_queries != int(exact_manifest["query_rows"]):
                raise ValueError(
                    f"configured query total does not match exact manifest: {config_manifest_path}"
                )
            contract_key = (workload, phase)
            if contract_key in contracts:
                raise ValueError(
                    f"duplicate workload/phase manifest: {contract_key}"
                )
            contracts[contract_key] = {
                "repetitions": expected_repetitions,
                "shards": expected_shards,
                "queries": expected_queries,
            }
            raw_directory = args.result_root / "raw" / phase / workload
            expected_files = {
                f"shard_{shard_number:02d}.json"
                for shard_number in expected_shards
            }
            observed_files = {
                path.name for path in raw_directory.glob("shard_*.json")
            }
            if observed_files != expected_files:
                raise ValueError(
                    f"incomplete raw shard set for {workload}/{phase}: "
                    f"missing={sorted(expected_files - observed_files)}, "
                    f"extra={sorted(observed_files - expected_files)}"
                )
            for config_row in config_manifest["configs"]:
                shard_number = int(config_row["shard_number"])
                exact_shard = exact_shards[shard_number]
                estimate = exact_shard.get("gpu_memory_estimate")
                if not isinstance(estimate, dict):
                    raise TypeError(
                        f"exact shard lacks GPU-memory estimate: {shard_number}"
                    )
                required_free = int(
                    estimate.get("required_free_device_bytes", -1)
                )
                if (
                    required_free <= 0
                    or int(config_row.get("required_free_device_bytes", -1))
                    != required_free
                ):
                    raise ValueError(
                        f"config/estimate memory contract mismatch: {config_manifest_path}"
                    )
                if phase in {"correctness", "throughput"}:
                    memory_record_path = (
                        args.result_root
                        / "provenance"
                        / "gpu_memory"
                        / phase
                        / workload
                        / f"shard_{shard_number:02d}.json"
                    )
                    if not memory_record_path.is_file():
                        raise FileNotFoundError(
                            f"missing mandatory GPU-memory preflight: {memory_record_path}"
                        )
                    memory_record = json.loads(memory_record_path.read_text())
                    if (
                        memory_record.get("status") != "PASS"
                        or int(memory_record.get("shard_number", -1))
                        != shard_number
                        or int(
                            memory_record.get("required_free_device_bytes", -1)
                        )
                        != required_free
                        or int(memory_record.get("available_bytes", -1))
                        < required_free
                        or Path(
                            str(memory_record.get("exact_manifest", ""))
                        ).resolve()
                        != Path(config_manifest["exact_manifest"]).resolve()
                    ):
                        raise ValueError(
                            f"invalid GPU-memory preflight: {memory_record_path}"
                        )
                config_path = Path(config_row["config"])
                config = json.loads(config_path.read_text())
                dataset = config["dataset"]
                basic = config["search_basic_param"]
                index = config["index"]
                if (
                    len(index) != 1
                    or index[0]["algo"] != "cuvs_brute_force"
                    or len(index[0]["search_params"]) != 1
                    or int(basic["batch_size"])
                    != int(config_row["query_count"])
                    or int(basic["k"]) != args.k
                    or Path(dataset["base_file"]).resolve()
                    != Path(exact_manifest["base_file"]).resolve()
                    or Path(dataset["query_file"]).resolve()
                    != Path(exact_shard["query_file"]).resolve()
                    or Path(dataset["groundtruth_neighbors_file"]).resolve()
                    != Path(exact_shard["groundtruth_file"]).resolve()
                    or dataset["filter"]["kind"] != "bitmap"
                    or Path(dataset["filter"]["file"]).resolve()
                    != Path(exact_shard["bitmap_file"]).resolve()
                    or index[0]["search_params"][0]
                    != {
                        "exact_control": "bitmap_count_csr_search",
                        "resident_bitmap": True,
                    }
                ):
                    raise ValueError(
                        f"exact benchmark config violates the frozen contract: {config_path}"
                    )
                raw_path = (
                    args.result_root
                    / "raw"
                    / phase
                    / workload
                    / f"shard_{shard_number:02d}.json"
                )
                (
                    query_count,
                    valid_slots,
                    underfilled_queries,
                    missing_slots,
                ) = gt_statistics(
                    Path(exact_shard["groundtruth_file"]), base_rows, args.k
                )
                if query_count != int(config_row["query_count"]):
                    raise ValueError(f"query-count mismatch for {raw_path}")
                expected_underfilled = underfilled_queries / query_count
                expected_missing = missing_slots / (query_count * args.k)
                expected_valid_fraction = valid_slots / (query_count * args.k)
                records = iteration_rows(raw_path)
                repetitions = [
                    int(record.get("repetition_index", -1))
                    for record in records
                ]
                if sorted(repetitions) != list(range(expected_repetitions)):
                    raise ValueError(
                        f"wrong repetition set in {raw_path}: observed={sorted(repetitions)}, "
                        f"expected={list(range(expected_repetitions))}"
                    )
                for record, repetition in zip(
                    records, repetitions, strict=True
                ):
                    qps = float(record["items_per_second"])
                    legacy_recall = float(record["Recall"])
                    valid_recall = float(record["ValidGTRecall"])
                    valid_fraction = float(
                        record.get("ValidGTFraction", math.nan)
                    )
                    label = str(record.get("label", ""))
                    if (
                        int(record.get("n_queries", -1)) != query_count
                        or int(record.get("k", -1)) != args.k
                        or label_value(label, "exact_control")
                        != "bitmap_count_csr_search"
                        or not close(
                            float(record.get("resident_bitmap", math.nan)), 1.0
                        )
                    ):
                        raise ValueError(
                            f"raw exact record disagrees with config in {raw_path}"
                        )
                    filter_violations = float(
                        record.get("FilterViolations", math.nan)
                    )
                    sentinel_errors = float(
                        record.get("InvalidSentinelErrors", math.nan)
                    )
                    observed_underfilled = float(
                        record.get("UnderfilledQueries", math.nan)
                    )
                    observed_missing = float(
                        record.get("MissingResultSlots", math.nan)
                    )
                    duplicate_queries = float(
                        record.get("DuplicateOutputQueries", math.nan)
                    )
                    output_set_semantics = float(
                        record.get("OutputSetSemanticsVersion", math.nan)
                    )
                    if (
                        not math.isfinite(qps)
                        or qps <= 0.0
                        or not math.isfinite(legacy_recall)
                        or not 0.0 <= legacy_recall <= 1.0
                        or not math.isfinite(valid_recall)
                        or not 0.0 <= valid_recall <= 1.0
                        or not close(output_set_semantics, 1.0)
                    ):
                        raise ValueError(
                            f"invalid exact QPS/recall/output semantics in {raw_path}: "
                            f"qps={qps}, legacy={legacy_recall}, valid={valid_recall}"
                        )
                    correct = (
                        close(valid_recall, 1.0, 1e-7)
                        and close(valid_fraction, expected_valid_fraction)
                        and close(filter_violations, 0.0)
                        and close(sentinel_errors, 0.0)
                        and close(observed_underfilled, expected_underfilled)
                        and close(observed_missing, expected_missing)
                        and close(duplicate_queries, 0.0)
                    )
                    row: dict[str, object] = {
                        "workload": workload,
                        "phase": phase,
                        "shard": shard_number,
                        "first_query": int(config_row["first_query"]),
                        "queries": query_count,
                        "repetition": repetition,
                        "qps": qps,
                        "seconds": query_count / qps,
                        "legacy_recall": legacy_recall,
                        "valid_gt_recall": valid_recall,
                        "valid_gt_fraction": valid_fraction,
                        "valid_gt_slots": valid_slots,
                        "underfilled_queries": observed_underfilled,
                        "expected_underfilled_queries": expected_underfilled,
                        "missing_result_slots": observed_missing,
                        "expected_missing_result_slots": expected_missing,
                        "filter_violations": filter_violations,
                        "invalid_sentinel_errors": sentinel_errors,
                        "duplicate_output_queries": duplicate_queries,
                        "output_set_semantics_version": output_set_semantics,
                        "correct": correct,
                    }
                    shard_rows.append(row)
                    grouped[(workload, phase, repetition)].append(row)

    if not shard_rows:
        raise ValueError("no exact-control results found")
    for phase in set(phases) & {"correctness", "throughput"}:
        observed_workloads = {
            workload
            for workload, contract_phase in contracts
            if contract_phase == phase
        }
        if observed_workloads != set(WORKLOADS):
            raise ValueError(
                f"{phase} exact workload set is incomplete: "
                f"observed={sorted(observed_workloads)}, expected={list(WORKLOADS)}"
            )

    aggregate_rows: list[dict[str, object]] = []
    for (workload, phase, repetition), rows in sorted(grouped.items()):
        contract = contracts[(workload, phase)]
        observed_shards = {int(row["shard"]) for row in rows}
        if observed_shards != contract["shards"] or len(rows) != len(
            contract["shards"]
        ):
            raise ValueError(
                f"incomplete/duplicate shards for {workload}/{phase}/rep{repetition}: "
                f"observed={sorted(observed_shards)}, expected={sorted(contract['shards'])}"
            )
        total_queries = sum(int(row["queries"]) for row in rows)
        if total_queries != contract["queries"]:
            raise ValueError(
                f"query total mismatch for {workload}/{phase}/rep{repetition}: "
                f"{total_queries} != {contract['queries']}"
            )
        total_seconds = sum(float(row["seconds"]) for row in rows)
        total_valid = sum(int(row["valid_gt_slots"]) for row in rows)
        matched = sum(
            float(row["valid_gt_recall"]) * int(row["valid_gt_slots"])
            for row in rows
        )
        aggregate_rows.append(
            {
                "workload": workload,
                "phase": phase,
                "repetition": repetition,
                "shards": len(rows),
                "queries": total_queries,
                "qps": total_queries / total_seconds,
                "seconds": total_seconds,
                "valid_gt_recall": matched / total_valid,
                "filter_violations": max(
                    float(row["filter_violations"]) for row in rows
                ),
                "invalid_sentinel_errors": max(
                    float(row["invalid_sentinel_errors"]) for row in rows
                ),
                "duplicate_output_queries": max(
                    float(row["duplicate_output_queries"]) for row in rows
                ),
                "correct": all(bool(row["correct"]) for row in rows),
            }
        )

    summaries: list[dict[str, object]] = []
    by_workload_phase: dict[tuple[str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in aggregate_rows:
        by_workload_phase[(str(row["workload"]), str(row["phase"]))].append(
            row
        )
    for (workload, phase), rows in sorted(by_workload_phase.items()):
        contract = contracts[(workload, phase)]
        observed_repetitions = {int(row["repetition"]) for row in rows}
        expected_repetition_set = set(range(int(contract["repetitions"])))
        if observed_repetitions != expected_repetition_set or len(rows) != len(
            expected_repetition_set
        ):
            raise ValueError(
                f"incomplete repetitions for {workload}/{phase}: "
                f"observed={sorted(observed_repetitions)}, "
                f"expected={sorted(expected_repetition_set)}"
            )
        qps_values = [float(row["qps"]) for row in rows]
        summaries.append(
            {
                "workload": workload,
                "phase": phase,
                "repetitions": len(rows),
                "queries": int(rows[0]["queries"]),
                "median_qps": statistics.median(qps_values),
                "min_qps": min(qps_values),
                "max_qps": max(qps_values),
                "valid_gt_recall": min(
                    float(row["valid_gt_recall"]) for row in rows
                ),
                "correct": all(bool(row["correct"]) for row in rows),
            }
        )

    analysis = args.result_root / "analysis"
    write_csv(analysis / "exact_shard_measurements.csv", shard_rows)
    write_csv(analysis / "exact_aggregate_measurements.csv", aggregate_rows)
    write_csv(analysis / "exact_summary.csv", summaries)
    result = {
        "schema_version": 2,
        "method": "cuvs_brute_force_bitmap",
        "timing_contract": {
            "included": [
                "bitmap count",
                "query L2 norms",
                "bitmap-to-CSR and CSR-to-COO construction when sparse path is selected",
                "masked sparse or tiled dense distance evaluation",
                "distance and top-k epilogues",
                "temporary device allocation and deallocation",
            ],
            "excluded": [
                "predicate evaluation",
                "bitmap materialization",
                "bitmap upload",
                "uint8-to-float conversion",
            ],
            "yfcc_aggregation": "10000 / sum(shard wall-search seconds)",
        },
        "correctness": {
            "recall": "distinct valid output-ID matches divided by valid GT IDs; out-of-range padding IDs are excluded",
            "output_set_semantics": "distinct_valid_output_ids_v1",
            "requirements": [
                "valid_gt_recall == 1",
                "zero predicate violations",
                "zero invalid-sentinel errors",
                "zero duplicate-output queries",
                "underfill and missing slots match legal GT cardinality",
            ],
            "all_passed": all(bool(row["correct"]) for row in summaries),
        },
        "summaries": summaries,
    }
    (analysis / "exact_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not result["correctness"]["all_passed"]:
        raise SystemExit("exact-control correctness failed")

    # Bind the published aggregate to every config, raw result, source manifest, and resident
    # input used by this result root.  Full SHA-256 is intentional here: this is generated once for
    # the paper artifact, and avoids treating path/mtime identity as data provenance.
    bound_files: set[Path] = {Path(__file__).resolve()}
    for phase in phases:
        for manifest_path in (args.result_root / "configs" / phase).glob(
            "*/manifest.json"
        ):
            bound_files.add(manifest_path.resolve())
            manifest = json.loads(manifest_path.read_text())
            exact_manifest_path = Path(manifest["exact_manifest"]).resolve()
            bound_files.add(exact_manifest_path)
            exact_payload = json.loads(exact_manifest_path.read_text())
            if exact_payload.get("source_bitmap_manifest"):
                bound_files.add(
                    Path(exact_payload["source_bitmap_manifest"]).resolve()
                )
            bound_files.add(Path(exact_payload["base_file"]).resolve())
            for shard in exact_payload["shards"]:
                bound_files.update(
                    {
                        Path(shard["query_file"]).resolve(),
                        Path(shard["groundtruth_file"]).resolve(),
                        Path(shard["bitmap_file"]).resolve(),
                    }
                )
            for row in manifest["configs"]:
                bound_files.add(Path(row["config"]).resolve())
    bound_files.update(
        path.resolve() for path in (args.result_root / "raw").glob("**/*.json")
    )
    bound_files.update(
        path.resolve()
        for path in (args.result_root / "provenance" / "gpu_memory").glob(
            "**/*.json"
        )
    )
    run_provenance = args.result_root / "provenance" / "run.json"
    if (
        set(phases) & {"correctness", "throughput"}
        and not run_provenance.is_file()
    ):
        raise FileNotFoundError(
            f"production exact analysis requires run provenance: {run_provenance}"
        )
    if run_provenance.is_file():
        run_payload = json.loads(run_provenance.read_text())
        if (
            run_payload.get("schema_version") != 2
            or run_payload.get("fixed_contract", {}).get(
                "output_set_semantics"
            )
            != "distinct_valid_output_ids_v1"
        ):
            raise ValueError(
                "exact result root uses legacy output/recall semantics"
            )
        bound_files.add(run_provenance.resolve())
    bound_files.update(
        path.resolve()
        for path in analysis.glob("*")
        if path.is_file() and path.name != "final_hash_manifest.json"
    )
    missing = [str(path) for path in sorted(bound_files) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing files in exact provenance set: {missing}"
        )
    hash_manifest = {
        "schema_version": 2,
        "algorithm": "sha256",
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(bound_files)
        ],
    }
    (analysis / "final_hash_manifest.json").write_text(
        json.dumps(hash_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
