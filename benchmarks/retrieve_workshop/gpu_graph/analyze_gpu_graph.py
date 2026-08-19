#!/usr/bin/env python3
"""Strictly validate, aggregate, and plot the RETRIEVE GPU graph experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

WORKLOADS = ("yfcc", "em", "emis", "r")
WORKLOAD_LABELS = {
    "yfcc": "YFCC-10M",
    "em": "ArXiv-EM",
    "emis": "ArXiv-EMIS",
    "r": "ArXiv-R",
}
METHOD_LABELS = {
    "default_cagra": "CAGRA-Base",
    "default_cagra_accumulator": "CAGRA-Retain",
    "navix_reference": "CAGRA-NaviX",
    "default_cagra_seeded": "CAGRA-Base + matched seeds",
    "default_cagra_accumulator_seeded": "CAGRA-Retain + matched seeds",
}
PRIMARY_METHODS = (
    "default_cagra",
    "default_cagra_accumulator",
    "navix_reference",
)
SEED_METHODS = (
    "default_cagra",
    "default_cagra_seeded",
    "default_cagra_accumulator",
    "default_cagra_accumulator_seeded",
    "navix_reference",
)
COLORS = {
    "default_cagra": "#4c78a8",
    "default_cagra_accumulator": "#e45756",
    "navix_reference": "#f58518",
    "default_cagra_seeded": "#72a5cf",
    "default_cagra_accumulator_seeded": "#ed8b87",
}
METHOD_MARKERS = {
    "default_cagra": "o",
    "default_cagra_accumulator": "D",
    "navix_reference": "^",
    "default_cagra_seeded": "s",
    "default_cagra_accumulator_seeded": "P",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def finite(
    record: dict, key: str, path: Path, default: float | None = None
) -> float:
    if key not in record:
        if default is None:
            raise ValueError(f"missing {key} in {path}")
        return default
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {path}")
    return value


@dataclass(frozen=True)
class RawPoint:
    group: str
    phase: str
    workload: str
    shard_index: int
    first_query: int
    repetition_index: int
    method: str
    itopk: int
    search_width: int
    max_iterations: int
    queries: int
    recall: float
    valid_gt_fraction: float
    qps: float
    seconds: float
    filter_violations: float
    sentinel_errors: float
    duplicate_output_queries: float
    underfilled_queries: float
    missing_result_slots: float
    source_file: str


@dataclass(frozen=True)
class RepetitionPoint:
    group: str
    phase: str
    workload: str
    repetition_index: int
    method: str
    itopk: int
    search_width: int
    max_iterations: int
    shards: int
    queries: int
    recall: float
    valid_gt_fraction: float
    qps: float
    seconds: float
    filter_violations: float
    sentinel_errors: float
    duplicate_output_queries: float
    underfilled_queries: float
    missing_result_slots: float


@dataclass(frozen=True)
class SummaryPoint:
    group: str
    phase: str
    workload: str
    method: str
    itopk: int
    search_width: int
    max_iterations: int
    repetitions: int
    shards_per_repetition: int
    queries_per_repetition: int
    recall_median: float
    recall_min: float
    recall_max: float
    valid_gt_fraction_min: float
    qps_median: float
    qps_min: float
    qps_max: float
    seconds_median: float
    filter_violations: float
    sentinel_errors: float
    duplicate_output_queries: float
    underfilled_queries_max: float
    missing_result_slots_max: float
    paper_included: bool


def point_key(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["method"]),
        int(row["itopk"]),
        int(row["search_width"]),
        int(row["max_iterations"]),
    )


def validate_runtime_flags(
    record: dict, label: str, method: str, path: Path
) -> None:
    accumulator = round(
        finite(record, "favor_udf_passing_accumulator", path, 0.0)
    )
    cagra_seeds = round(finite(record, "cagra_bitmap_seeds", path, 0.0))
    navix_seeds = round(finite(record, "navix_bitmap_seeds", path, 0.0))
    identity_ids = round(finite(record, "require_identity_source_indices", path))
    expected_accumulator = int("accumulator" in method)
    expected_cagra_seeds = int(method.endswith("_seeded"))
    expected_navix_seeds = int(method == "navix_reference")
    if (
        accumulator != expected_accumulator
        or cagra_seeds != expected_cagra_seeds
        or navix_seeds != expected_navix_seeds
        or identity_ids != 1
    ):
        raise ValueError(
            f"runtime flags disagree with {method} in {path}: accumulator={accumulator}, "
            f"cagra_seeds={cagra_seeds}, navix_seeds={navix_seeds}, "
            f"identity_ids={identity_ids}"
        )
    mode = label_value(label, "navix_mode")
    if method == "navix_reference" and mode != "adaptive_kuzu":
        raise ValueError(
            f"{method} did not run adaptive_kuzu in {path}: {label}"
        )
    if method != "navix_reference" and mode:
        raise ValueError(f"non-NaviX method has navix_mode={mode} in {path}")
    if label_value(label, "algo") != "single_cta":
        raise ValueError(f"non-SINGLE_CTA result in {path}: {label}")
    if label_value(label, "filter_mode") != "default":
        raise ValueError(f"non-default filter mode in {path}: {label}")
    if round(finite(record, "k", path)) != 10:
        raise ValueError(f"paper run does not use k=10 in {path}")
    if round(finite(record, "max_queries", path)) != 512:
        raise ValueError(f"paper run does not use max_queries=512 in {path}")
    scheduler = label_value(label, "navix_scheduler")
    variant = label_value(label, "navix_kernel_variant")
    if method == "navix_reference":
        if scheduler != "tiled" or variant != "reference":
            raise ValueError(
                f"unexpected NaviX scheduler/variant in {path}: "
                f"scheduler={scheduler!r}, variant={variant!r}"
            )
    elif scheduler or variant:
        raise ValueError(
            f"non-NaviX method has NaviX scheduler/variant in {path}"
        )


def load_manifests(config_root: Path) -> list[tuple[Path, dict]]:
    manifests: list[tuple[Path, dict]] = []
    for path in sorted(config_root.glob("*/*/manifest.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment") != "retrieve_workshop_gpu_graph":
            continue
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported manifest schema in {path}")
        manifests.append((path, payload))
    if not manifests:
        raise ValueError(f"no experiment manifests under {config_root}")
    return manifests


def load_group(path: Path, manifest: dict, raw_root: Path) -> list[RawPoint]:
    group = str(manifest["group"])
    phase = str(manifest["phase"])
    workload = str(manifest["workload"])
    repetitions = int(manifest["repetitions"])
    expected_points = {point_key(row) for row in manifest["search_points"]}
    if len(expected_points) != len(manifest["search_points"]):
        raise ValueError(f"duplicate search point in {path}")
    source_manifest_path = Path(manifest["source_bitmap_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text())
    cursor = 0
    source_shards = source_manifest.get("shards", [])
    for shard_index, shard in enumerate(source_shards):
        first_query = int(shard["first_query"])
        count = int(shard["query_count"])
        if first_query != cursor or count <= 0:
            raise ValueError(
                f"non-contiguous/invalid source shard {shard_index} in "
                f"{source_manifest_path}: first={first_query}, expected={cursor}, count={count}"
            )
        cursor += count
    if cursor != int(manifest["expected_queries"]):
        raise ValueError(
            f"source manifest query coverage mismatch in {source_manifest_path}: "
            f"{cursor} != {manifest['expected_queries']}"
        )
    if "query_rows" in source_manifest and int(source_manifest["query_rows"]) != cursor:
        raise ValueError(f"source manifest query_rows mismatch in {source_manifest_path}")

    raw_directory = raw_root / group / workload
    expected_files = {
        f"shard_{int(row['shard_index']):02d}.json"
        for row in manifest["configs"]
    }
    observed_files = {item.name for item in raw_directory.glob("shard_*.json")}
    if observed_files != expected_files:
        raise ValueError(
            f"incomplete raw files for {group}/{workload}: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )

    result: list[RawPoint] = []
    for shard in manifest["configs"]:
        shard_index = int(shard["shard_index"])
        first_query = int(shard["first_query"])
        query_count = int(shard["query_count"])
        raw_path = raw_directory / f"shard_{shard_index:02d}.json"
        payload = json.loads(raw_path.read_text())
        rows_by_repetition: dict[int, set[tuple[str, int, int, int]]] = {
            index: set() for index in range(repetitions)
        }
        for record in payload.get("benchmarks", []):
            if record.get("error_occurred"):
                raise ValueError(
                    f"benchmark error in {raw_path}: "
                    f"{record.get('error_message', record.get('name', 'unknown error'))}"
                )
            if record.get("skipped"):
                raise ValueError(f"skipped benchmark record in {raw_path}")
            if record.get("run_type") != "iteration":
                continue
            repetition = int(record.get("repetition_index", -1))
            if repetition not in rows_by_repetition:
                raise ValueError(
                    f"invalid repetition_index={repetition} in {raw_path}"
                )
            label = str(record.get("label", ""))
            method = label_value(label, "bitmap_method")
            if method not in METHOD_LABELS:
                raise ValueError(
                    f"unknown/missing bitmap_method in {raw_path}: {label}"
                )
            key = (
                method,
                round(finite(record, "itopk", raw_path)),
                round(finite(record, "search_width", raw_path)),
                round(finite(record, "max_iterations", raw_path)),
            )
            if key not in expected_points:
                raise ValueError(
                    f"unexpected search point {key} in {raw_path}"
                )
            if key in rows_by_repetition[repetition]:
                raise ValueError(
                    f"duplicate search point {key}, repetition {repetition} in {raw_path}"
                )
            rows_by_repetition[repetition].add(key)
            validate_runtime_flags(record, label, method, raw_path)

            queries = round(finite(record, "n_queries", raw_path))
            qps = finite(record, "items_per_second", raw_path)
            # Filtered ground truth can contain out-of-range padding sentinels.  The legacy Recall
            # counter retains historical behavior; paper results require the valid-ID-only counter.
            recall = finite(record, "ValidGTRecall", raw_path)
            valid_gt_fraction = finite(record, "ValidGTFraction", raw_path)
            if queries != query_count:
                raise ValueError(
                    f"{raw_path} reports {queries} queries, expected shard count {query_count}"
                )
            if (
                qps <= 0
                or not 0.0 <= recall <= 1.0
                or not 0.0 < valid_gt_fraction <= 1.0
            ):
                raise ValueError(
                    f"invalid QPS/recall/valid-GT fraction in {raw_path}: "
                    f"qps={qps}, recall={recall}, valid_gt_fraction={valid_gt_fraction}"
                )
            violations = finite(record, "FilterViolations", raw_path)
            sentinels = finite(record, "InvalidSentinelErrors", raw_path)
            # Mandatory for paper data: absence must not masquerade as a successful check.
            duplicates = finite(record, "DuplicateOutputQueries", raw_path)
            if violations != 0 or sentinels != 0 or duplicates != 0:
                raise ValueError(
                    f"correctness failure in {raw_path}, repetition {repetition}, {key}: "
                    f"filter={violations}, sentinel={sentinels}, duplicate={duplicates}"
                )
            result.append(
                RawPoint(
                    group=group,
                    phase=phase,
                    workload=workload,
                    shard_index=shard_index,
                    first_query=first_query,
                    repetition_index=repetition,
                    method=method,
                    itopk=key[1],
                    search_width=key[2],
                    max_iterations=key[3],
                    queries=queries,
                    recall=recall,
                    valid_gt_fraction=valid_gt_fraction,
                    qps=qps,
                    seconds=queries / qps,
                    filter_violations=violations,
                    sentinel_errors=sentinels,
                    duplicate_output_queries=duplicates,
                    underfilled_queries=finite(
                        record, "UnderfilledQueries", raw_path, 0.0
                    ),
                    missing_result_slots=finite(
                        record, "MissingResultSlots", raw_path, 0.0
                    ),
                    source_file=str(raw_path.resolve()),
                )
            )
        for repetition, observed in rows_by_repetition.items():
            if observed != expected_points:
                raise ValueError(
                    f"incomplete points in {raw_path}, repetition {repetition}: "
                    f"missing={sorted(expected_points - observed)}, "
                    f"extra={sorted(observed - expected_points)}"
                )
    return result


def aggregate_repetitions(points: list[RawPoint]) -> list[RepetitionPoint]:
    groups: dict[tuple[object, ...], list[RawPoint]] = {}
    for point in points:
        key = (
            point.group,
            point.phase,
            point.workload,
            point.repetition_index,
            point.method,
            point.itopk,
            point.search_width,
            point.max_iterations,
        )
        groups.setdefault(key, []).append(point)
    output: list[RepetitionPoint] = []
    for key, members in sorted(groups.items()):
        (
            group,
            phase,
            workload,
            repetition,
            method,
            itopk,
            width,
            iterations,
        ) = key
        total_queries = sum(row.queries for row in members)
        total_seconds = sum(row.seconds for row in members)
        expected_shards = (
            5 if workload == "yfcc" and phase == "throughput" else 1
        )
        expected_queries = 1_000 if phase == "correctness" else 10_000
        if (
            len(members) != expected_shards
            or total_queries != expected_queries
        ):
            raise ValueError(
                f"incomplete serial aggregate for {group}/{workload}/{method}/rep{repetition}: "
                f"shards={len(members)}/{expected_shards}, queries={total_queries}/{expected_queries}"
            )
        if len({row.shard_index for row in members}) != expected_shards:
            raise ValueError(
                f"duplicate shard in {group}/{workload}/{method}/rep{repetition}"
            )
        valid_gt_slots = sum(
            row.valid_gt_fraction * row.queries * 10 for row in members
        )
        valid_gt_matches = sum(
            row.recall * row.valid_gt_fraction * row.queries * 10
            for row in members
        )
        if valid_gt_slots <= 0:
            raise ValueError(
                f"no valid filtered ground truth for {group}/{workload}/{method}/rep{repetition}"
            )
        output.append(
            RepetitionPoint(
                group=str(group),
                phase=str(phase),
                workload=str(workload),
                repetition_index=int(repetition),
                method=str(method),
                itopk=int(itopk),
                search_width=int(width),
                max_iterations=int(iterations),
                shards=len(members),
                queries=total_queries,
                recall=valid_gt_matches / valid_gt_slots,
                valid_gt_fraction=valid_gt_slots / (total_queries * 10),
                # Shards are sequential host calls: aggregate QPS is total queries / total time.
                qps=total_queries / total_seconds,
                seconds=total_seconds,
                filter_violations=sum(
                    row.filter_violations for row in members
                ),
                sentinel_errors=sum(row.sentinel_errors for row in members),
                duplicate_output_queries=sum(
                    row.duplicate_output_queries for row in members
                ),
                underfilled_queries=sum(
                    row.underfilled_queries * row.queries for row in members
                )
                / total_queries,
                missing_result_slots=sum(
                    row.missing_result_slots * row.queries for row in members
                )
                / total_queries,
            )
        )
    return output


def summarize(
    points: list[RepetitionPoint], target: float
) -> list[SummaryPoint]:
    groups: dict[tuple[object, ...], list[RepetitionPoint]] = {}
    for point in points:
        key = (
            point.group,
            point.phase,
            point.workload,
            point.method,
            point.itopk,
            point.search_width,
            point.max_iterations,
        )
        groups.setdefault(key, []).append(point)
    rows: list[SummaryPoint] = []
    for key, members in sorted(groups.items()):
        group, phase, workload, method, itopk, width, iterations = key
        indices = sorted(row.repetition_index for row in members)
        expected_indices = [0] if phase == "correctness" else [0, 1, 2]
        if indices != expected_indices:
            raise ValueError(
                f"expected repetitions {expected_indices} for {key}, got {indices}"
            )
        recalls = [row.recall for row in members]
        valid_gt_fractions = [row.valid_gt_fraction for row in members]
        rates = [row.qps for row in members]
        seconds = [row.seconds for row in members]
        rows.append(
            SummaryPoint(
                group=str(group),
                phase=str(phase),
                workload=str(workload),
                method=str(method),
                itopk=int(itopk),
                search_width=int(width),
                max_iterations=int(iterations),
                repetitions=len(members),
                shards_per_repetition=members[0].shards,
                queries_per_repetition=members[0].queries,
                recall_median=statistics.median(recalls),
                recall_min=min(recalls),
                recall_max=max(recalls),
                valid_gt_fraction_min=min(valid_gt_fractions),
                qps_median=statistics.median(rates),
                qps_min=min(rates),
                qps_max=max(rates),
                seconds_median=statistics.median(seconds),
                filter_violations=sum(
                    row.filter_violations for row in members
                ),
                sentinel_errors=sum(row.sentinel_errors for row in members),
                duplicate_output_queries=sum(
                    row.duplicate_output_queries for row in members
                ),
                underfilled_queries_max=max(
                    row.underfilled_queries for row in members
                ),
                missing_result_slots_max=max(
                    row.missing_result_slots for row in members
                ),
                paper_included=True,
            )
        )

    # A deep series is included only through its first target-reaching point.  Raw and summary
    # CSVs retain every measured point, but later over-search points cannot distort the paper plot.
    included: set[tuple[object, ...]] = set()
    series: dict[tuple[str, str, int, int], list[SummaryPoint]] = {}
    for row in rows:
        if row.phase == "throughput":
            series.setdefault(
                (row.workload, row.method, row.itopk, row.search_width), []
            ).append(row)
    for members in series.values():
        ordered = sorted(members, key=lambda row: row.max_iterations)
        reached = False
        for row in ordered:
            identity = (
                row.group,
                row.workload,
                row.method,
                row.itopk,
                row.search_width,
                row.max_iterations,
            )
            if row.max_iterations == 0 or not reached:
                included.add(identity)
            if row.recall_median >= target:
                reached = True
    return [
        replace(
            row,
            paper_included=(
                row.phase == "correctness"
                or (
                    row.group,
                    row.workload,
                    row.method,
                    row.itopk,
                    row.search_width,
                    row.max_iterations,
                )
                in included
            ),
        )
        for row in rows
    ]


def pareto(rows: list[SummaryPoint]) -> list[SummaryPoint]:
    frontier: list[SummaryPoint] = []
    best_qps = -math.inf
    for row in sorted(
        rows,
        key=lambda item: (item.recall_median, item.qps_median),
        reverse=True,
    ):
        if row.qps_median > best_qps:
            frontier.append(row)
            best_qps = row.qps_median
    return sorted(frontier, key=lambda item: item.recall_median)


def plot_workload(
    output: Path,
    workload: str,
    rows: list[SummaryPoint],
    methods: tuple[str, ...],
    suffix: str,
    title_suffix: str,
) -> None:
    selected = [
        row
        for row in rows
        if row.phase == "throughput"
        and row.workload == workload
        and row.method in methods
        and row.paper_included
    ]
    if not selected:
        return
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for method in methods:
        method_rows = [row for row in selected if row.method == method]
        if not method_rows:
            continue
        color = COLORS[method]
        for depth, marker, label in (
            ("b0", METHOD_MARKERS[method], "B0"),
            ("deep", "X", "deep"),
        ):
            depth_rows = [
                row
                for row in method_rows
                if (row.max_iterations == 0) == (depth == "b0")
            ]
            if depth_rows:
                axis.scatter(
                    [row.recall_median for row in depth_rows],
                    [row.qps_median for row in depth_rows],
                    color=color,
                    marker=marker,
                    s=42 if depth == "b0" else 54,
                    alpha=0.72,
                    label=f"{METHOD_LABELS[method]} ({label})",
                )
        front = pareto(method_rows)
        axis.plot(
            [row.recall_median for row in front],
            [row.qps_median for row in front],
            color=color,
            linewidth=1.8,
        )
    minimum = min(row.recall_median for row in selected)
    axis.set_xlim(max(0.0, minimum - 0.02), 1.0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Recall@10")
    axis.set_ylabel("Queries per second")
    axis.set_title(f"{WORKLOAD_LABELS[workload]}: {title_suffix}")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    # Avoid duplicate legend entries when a method contributes several cells at a depth.
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{workload}_{suffix}.png", dpi=200)
    fig.savefig(output / f"{workload}_{suffix}.pdf")
    plt.close(fig)


def write_csv(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=asdict(rows[0]).keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def deep_candidates(rows: list[SummaryPoint], target: float) -> dict:
    pairs: list[dict] = []
    for workload in WORKLOADS:
        for method in METHOD_LABELS:
            points = [
                row
                for row in rows
                if row.group == "b0"
                and row.workload == workload
                and row.method == method
            ]
            if points and max(row.recall_median for row in points) < target:
                best = max(points, key=lambda row: row.recall_median)
                pairs.append(
                    {
                        "workload": workload,
                        "method": method,
                        "best_b0_recall": best.recall_median,
                        "target": target,
                    }
                )
    return {
        "schema_version": 1,
        "description": (
            "Candidate workload/method pairs whose best six-cell B0 recall is below target. "
            "Review/edit this file before passing it to generate_configs.py --deep-selection."
        ),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument(
        "--require-group",
        action="append",
        default=[],
        help="fail unless this complete group is present (for example correctness or b0)",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--skip-full-input-hashes",
        action="store_true",
        help="internal staged-analysis optimization; final/paper analysis must not use this",
    )
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1:
        raise ValueError("target recall must be in (0,1]")

    config_root = args.result_root / "configs"
    raw_root = args.result_root / "raw"
    output = args.output or args.result_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests(config_root)
    available_groups = {
        path.parent.parent.name
        for path, manifest in manifests
        if (raw_root / str(manifest["group"])).is_dir()
    }
    missing_required = set(args.require_group) - available_groups
    if missing_required:
        raise ValueError(
            f"missing required raw groups: {sorted(missing_required)}"
        )
    if not available_groups:
        raise ValueError(f"no raw result groups under {raw_root}")
    for complete_group in ("correctness", "b0"):
        if complete_group not in available_groups:
            continue
        observed_workloads = {
            str(manifest["workload"])
            for _, manifest in manifests
            if str(manifest["group"]) == complete_group
        }
        if observed_workloads != set(WORKLOADS):
            raise ValueError(
                f"{complete_group} workload manifest set is incomplete: "
                f"observed={sorted(observed_workloads)}, expected={list(WORKLOADS)}"
            )

    raw_points: list[RawPoint] = []
    used_manifests: list[Path] = []
    for path, manifest in manifests:
        if str(manifest["group"]) not in available_groups:
            continue
        raw_points.extend(load_group(path, manifest, raw_root))
        used_manifests.append(path)
    repetition_points = aggregate_repetitions(raw_points)
    summary_points = summarize(repetition_points, args.target_recall)
    pareto_points: list[SummaryPoint] = []
    for workload in WORKLOADS:
        for method in METHOD_LABELS:
            pareto_points.extend(
                pareto(
                    [
                        row
                        for row in summary_points
                        if row.phase == "throughput"
                        and row.workload == workload
                        and row.method == method
                        and row.paper_included
                    ]
                )
            )

    write_csv(output / "raw_points.csv", raw_points)
    write_csv(output / "repetition_aggregates.csv", repetition_points)
    write_csv(output / "summary_points.csv", summary_points)
    write_csv(output / "pareto_points.csv", pareto_points)
    candidates = deep_candidates(summary_points, args.target_recall)
    (output / "deep_candidates.json").write_text(
        json.dumps(candidates, indent=2) + "\n"
    )

    if not args.no_plots:
        for workload in WORKLOADS:
            plot_workload(
                output,
                workload,
                summary_points,
                PRIMARY_METHODS,
                "gpu_graph_qps_recall",
                "GPU graph search (3-repetition median)",
            )
            plot_workload(
                output,
                workload,
                summary_points,
                SEED_METHODS,
                "matched_seed_control",
                "matched deterministic passing-seed control",
            )

    raw_files = sorted({Path(row.source_file) for row in raw_points})
    config_files: set[Path] = set()
    source_manifests: set[Path] = set()
    for manifest_path in used_manifests:
        manifest = json.loads(manifest_path.read_text())
        config_files.update(Path(row["config"]) for row in manifest["configs"])
        source_manifests.add(Path(manifest["source_bitmap_manifest"]))
    run_provenance = args.result_root / "provenance" / "run.json"
    if not run_provenance.is_file():
        raise FileNotFoundError(
            f"paper analysis requires staged run provenance: {run_provenance}"
        )
    run_payload = json.loads(run_provenance.read_text())
    data_root = Path(run_payload["data_root"])

    def data_path(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else data_root / path).resolve()

    resident_inputs: set[Path] = set()
    for config_path in config_files:
        config = json.loads(config_path.read_text())
        dataset = config["dataset"]
        resident_inputs.update(
            {
                data_path(str(dataset["base_file"])),
                data_path(str(dataset["query_file"])),
                data_path(str(dataset["groundtruth_neighbors_file"])),
                data_path(str(dataset["filter"]["file"])),
            }
        )
        resident_inputs.update(
            data_path(str(index["file"])) for index in config["index"]
        )
    missing_inputs = [str(path) for path in sorted(resident_inputs) if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            f"missing resident benchmark inputs: {missing_inputs}"
        )
    if args.skip_full_input_hashes:
        resident_input_records: list[dict[str, object]] = []
        input_hash_status = "skipped for internal staged analysis; not paper-valid"
    else:
        resident_input_records = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(resident_inputs)
        ]
        input_hash_status = "complete SHA-256 over all unique resident inputs"
    provenance = {
        "schema_version": 1,
        "experiment": "retrieve_workshop_gpu_graph",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "result_root": str(args.result_root.resolve()),
        "target_recall": args.target_recall,
        "groups": sorted(available_groups),
        "timing_contract": (
            "Each repetition times cuVS-bench's complete search call. For a sharded workload, "
            "shards execute serially and QPS=10000/sum(shard_seconds) within the same "
            "repetition_index. Median/min/max are computed only after serial aggregation."
        ),
        "plot_contract": (
            "Median QPS and recall over three repetitions; B0 and explicit deep points use "
            "different markers; deep series stop in paper outputs after first recall>=target."
        ),
        "config_manifests": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in used_manifests
        ],
        "generated_configs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(config_files)
        ],
        "source_bitmap_manifests": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in sorted(source_manifests)
        ],
        "raw_results": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in raw_files
        ],
        "resident_input_hash_status": input_hash_status,
        "resident_inputs": resident_input_records,
        "run_provenance": {
            "path": str(run_provenance.resolve()),
            "sha256": sha256(run_provenance),
            "payload": run_payload,
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    summary = {
        "raw_points": len(raw_points),
        "repetition_aggregates": len(repetition_points),
        "summary_points": len(summary_points),
        "pareto_points": len(pareto_points),
        "groups": sorted(available_groups),
        "correctness_error_total": sum(
            row.filter_violations
            + row.sentinel_errors
            + row.duplicate_output_queries
            for row in raw_points
        ),
        "deep_candidate_pairs": len(candidates["pairs"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
