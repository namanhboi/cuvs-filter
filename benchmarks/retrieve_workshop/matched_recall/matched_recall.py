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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GPU_GRAPH_DIR = SCRIPT_DIR.parent / "gpu_graph"
sys.path.insert(0, str(GPU_GRAPH_DIR))

from analyze_gpu_graph import aggregate_repetitions, load_group  # noqa: E402
from generate_configs import (  # noqa: E402
    config_payload,
    dataset_paths,
    point_identity,
    search_point,
)

WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")
WIDTHS = (1, 2, 4)
ANCHOR_L = (32, 64, 128, 256, 512)
MIN_L = 32
MAX_L = 512
L_QUANTUM = 32
MAX_ITERATIONS = 7569
TARGETS = {"yfcc": 0.80, "em": 0.95, "emis": 0.95, "r": 0.95}
TARGET_WINDOW = 0.002
EXPECTED_QUERIES = 10_000
MAX_QUERIES = 512

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


def target(workload: str) -> float:
    return TARGETS[workload]


def resolved_b0(itopk: int, width: int, dataset_size: int, degree: int) -> int:
    if itopk % L_QUANTUM or not MIN_L <= itopk <= MAX_L:
        raise ValueError(f"invalid SINGLE_CTA L={itopk}")
    iterations = itopk // width
    reachable = 1
    while reachable < dataset_size:
        reachable *= max(2, degree // 2)
        iterations += 1
    return iterations


def resolved_iterations(workload: str, itopk: int, width: int, maximum: int) -> int:
    if maximum:
        return maximum
    dataset_size = 10_000_000 if workload == "yfcc" else 100_000
    degree = 64 if workload == "yfcc" else 32
    return resolved_b0(itopk, width, dataset_size, degree)


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
    if itopk % L_QUANTUM or not MIN_L <= itopk <= MAX_L:
        raise ValueError(f"L must be a multiple of 32 in [32,512]: {row}")
    if width not in WIDTHS:
        raise ValueError(f"W must be one of {WIDTHS}: {row}")
    if not 0 <= maximum <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations outside [0,{MAX_ITERATIONS}]: {row}")
    if maximum and maximum <= resolved_iterations(workload, itopk, width, 0):
        raise ValueError(f"explicit depth must exceed B0: {row}")
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
) -> None:
    if repetitions not in (1, 3):
        raise ValueError("matched-recall groups use one or three repetitions")
    normalized = [normalize_point(row) for row in points]
    if len({key(row) for row in normalized}) != len(normalized):
        raise ValueError("duplicate point in requested group")
    if not normalized:
        raise ValueError("cannot generate an empty group")
    group_root = result_root / "configs" / group
    if group_root.exists():
        raise FileExistsError(group_root)
    stage = "final" if repetitions == 3 else "calibration"
    group_index = {
        "schema_version": 1,
        "experiment": "retrieve_workshop_matched_recall",
        "group": group,
        "stage": stage,
        "repetitions": repetitions,
        "targets": TARGETS,
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
                        if left_pass != right_pass and gap > L_QUANTUM:
                            midpoint_index = (
                                (int(left["itopk"]) // L_QUANTUM)
                                + (int(right["itopk"]) // L_QUANTUM)
                            ) // 2
                            midpoint = midpoint_index * L_QUANTUM
                            candidate = (workload, method, midpoint, width, 0)
                            if candidate not in observed:
                                proposals.add(candidate)
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
                if candidate not in observed:
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


def provenance(result_root: Path, summaries: list[dict], selected: list[dict]) -> dict:
    run_path = result_root / "provenance" / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(run_path)
    manifests = [path for path, _ in experiment_manifests(result_root)]
    raw_files = sorted((result_root / "raw").glob("*/*/shard_*.json"))
    return {
        "schema_version": 1,
        "experiment": "retrieve_workshop_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "target_window": TARGET_WINDOW,
        "selection_contract": (
            "Use B0 only whenever any B0 configuration reaches the workload target. "
            "Otherwise allow explicit continuation through 7569 iterations. Final selection "
            "requires recall_min>=target, prefers median recall<=target+0.002, and chooses maximum "
            "median QPS inside that window; no interpolation."
        ),
        "timing_contract": (
            "Complete cuVS-bench search call; YFCC executes five bitmap shards serially and "
            "QPS=10000/sum(shard seconds) within each repetition."
        ),
        "run_provenance": {"path": str(run_path.resolve()), "sha256": sha256(run_path)},
        "config_manifests": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in manifests],
        "raw_results": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in raw_files],
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
    generate.add_argument("--points", type=Path, required=True)

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
        )
        print(json.dumps({"group": args.group, "points": len(payload["points"])}))
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
