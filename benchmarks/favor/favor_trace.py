#!/usr/bin/env python3
"""Parse and compare bounded SINGLE_CTA FAVOR diagnostic captures."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import signal
import statistics
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


STOP_REASONS = {
    0: "UNKNOWN",
    1: "MAX_WITH_UNEXPANDED_FRONTIER",
    2: "MAX_WITH_EMPTY_FRONTIER",
    3: "FRONTIER_EXHAUSTED",
    4: "FILTER_EMPTY_OR_SKIPPED",
    5: "ADAPTIVE_CONVERGED",
    6: "ADAPTIVE_SAFETY_CAP",
}

HASH_OUTCOMES = {0: "UNKNOWN", 1: "INSERTED", 2: "DUPLICATE", 3: "FULL"}

ITERATION_DTYPE = np.dtype(
    [
        ("query_id", "<u4"), ("iteration", "<u4"), ("valid", "<u4"),
        ("passing", "<u4"), ("rejected", "<u4"),
        ("unexpanded_passing", "<u4"), ("unexpanded_rejected", "<u4"),
        ("selected_passing_parents", "<u4"), ("selected_rejected_parents", "<u4"),
        ("child_attempts", "<u4"), ("child_evaluations", "<u4"),
        ("child_duplicate_or_full", "<u4"), ("child_duplicates", "<u4"),
        ("child_hash_full", "<u4"), ("child_passing", "<u4"),
        ("child_rejected", "<u4"), ("stop_reason", "<u4"),
        ("penalty", "<f4"), ("cutoff", "<f4"),
        ("best_unexpanded_distance", "<f4"), ("worst_retained_distance", "<f4"),
    ]
)

CANDIDATE_DTYPE = np.dtype(
    [
        ("query_id", "<u4"), ("iteration", "<u4"), ("parent_id", "<u4"),
        ("child_id", "<u4"), ("raw_distance", "<f4"),
        ("effective_penalty", "<f4"), ("final_distance", "<f4"),
        ("ground_truth_rank", "<i2"), ("passes_filter", "u1"),
        ("hash_result", "u1"), ("survived_next_merge", "u1"), ("valid", "u1"),
        ("reserved", "<u2"),
    ]
)


def load_manifest(run: Path) -> dict:
    with (run / "manifest.json").open() as f:
        manifest = json.load(f)
    if manifest.get("schema_version") not in (2, 3, 4, 5):
        raise ValueError(f"unsupported schema version: {manifest.get('schema_version')}")
    return manifest


def load_summary(run: Path) -> list[dict[str, str]]:
    with (run / "query_summary.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def load_selected(run: Path) -> dict[int, int]:
    path = run / "selected_queries.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {int(row["query_id"]): int(row["trace_slot"]) for row in csv.DictReader(f)}


def binary_bytes(run: Path, stem: str) -> bytes:
    raw = run / f"{stem}.bin"
    compressed = run / f"{stem}.bin.zst"
    if compressed.exists():
        proc = subprocess.run(
            ["zstd", "-q", "-d", "-c", str(compressed)], check=True, stdout=subprocess.PIPE
        )
        return proc.stdout
    return raw.read_bytes()


def load_iterations(run: Path, manifest: dict) -> np.ndarray:
    data = np.frombuffer(binary_bytes(run, "iteration_trace"), dtype=ITERATION_DTYPE)
    shape = (manifest["trace_slots"], manifest["max_trace_iterations"])
    return data.reshape(shape)


def load_candidates(run: Path, manifest: dict) -> np.ndarray:
    data = np.frombuffer(binary_bytes(run, "candidate_trace"), dtype=CANDIDATE_DTYPE)
    shape = (
        manifest["trace_slots"],
        manifest["max_trace_iterations"],
        manifest["candidates_per_iteration"],
    )
    return data.reshape(shape)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def summarize(args: argparse.Namespace) -> None:
    run = Path(args.run)
    manifest = load_manifest(run)
    rows = load_summary(run)
    recalls = [as_float(row, "recall") for row in rows]
    iterations = [as_float(row, "iterations") for row in rows]
    reasons = Counter(STOP_REASONS.get(int(row["stop_reason"]), row["stop_reason"]) for row in rows)
    seen = [int(row["gt_seen_mask"]).bit_count() for row in rows]
    underfilled = sum(int(row["output_count"]) < manifest["topk"] for row in rows)
    duplicates = [int(row.get("candidate_duplicates", 0)) for row in rows]
    hash_full = [int(row.get("candidate_hash_full", 0)) for row in rows]
    print(f"dataset={manifest.get('dataset', '')} variant={manifest.get('variant', '')}")
    print(f"queries={len(rows)} mean_recall={statistics.fmean(recalls):.6f} "
          f"p10/p50/p90={np.percentile(recalls, [10, 50, 90])}")
    print(f"mean_iterations={statistics.fmean(iterations):.2f} "
          f"mean_gt_seen={statistics.fmean(seen):.3f} underfilled={underfilled}")
    print(f"mean_hash_duplicates={statistics.fmean(duplicates):.2f} "
          f"hash_full_total={sum(hash_full)}")
    print("stop_reasons=" + ", ".join(f"{key}:{value}" for key, value in reasons.most_common()))


def inspect_query(args: argparse.Namespace) -> None:
    run = Path(args.run)
    manifest = load_manifest(run)
    rows = {int(row["query_id"]): row for row in load_summary(run)}
    selected = load_selected(run)
    query = args.query_id
    if query not in rows:
        raise ValueError(f"query {query} is absent from summary")
    row = rows[query]
    print(json.dumps(row, indent=2))
    if query not in selected:
        print("No deep trace for this query.")
        return
    slot = selected[query]
    iteration_count = int(row["iterations"])
    iterations = load_iterations(run, manifest)[slot, :iteration_count]
    print("\niteration valid pass reject unexp_pass unexp_reject parents(p/r) "
          "children(eval/attempt,dup/full) stop")
    for item in iterations:
        print(
            f"{item['iteration']:9d} {item['valid']:5d} {item['passing']:4d} "
            f"{item['rejected']:6d} {item['unexpanded_passing']:10d} "
            f"{item['unexpanded_rejected']:12d} "
            f"{item['selected_passing_parents']}/{item['selected_rejected_parents']} "
            f"{item['child_evaluations']}/{item['child_attempts']},"
            f"{item['child_duplicates']}/{item['child_hash_full']} "
            f"{STOP_REASONS.get(int(item['stop_reason']), int(item['stop_reason']))}"
        )
    if args.candidates:
        candidates = load_candidates(run, manifest)[slot, :iteration_count]
        print("\niteration parent child raw penalty final pass hash gt_rank survived")
        for item in candidates[candidates["valid"] != 0]:
            print(
                f"{item['iteration']:9d} {item['parent_id']:6d} {item['child_id']:6d} "
                f"{item['raw_distance']:.7g} {item['effective_penalty']:.7g} "
                f"{item['final_distance']:.7g} {item['passes_filter']} "
                f"{HASH_OUTCOMES.get(int(item['hash_result']), int(item['hash_result'])):9} "
                f"{item['ground_truth_rank']:7d} {item['survived_next_merge']}"
            )


def compare(args: argparse.Namespace) -> None:
    left = {int(r["query_id"]): r for r in load_summary(Path(args.left))}
    right = {int(r["query_id"]): r for r in load_summary(Path(args.right))}
    common = sorted(left.keys() & right.keys())
    recall_delta = [as_float(right[q], "recall") - as_float(left[q], "recall") for q in common]
    iteration_delta = [as_float(right[q], "iterations") - as_float(left[q], "iterations") for q in common]
    gained_gt = [
        (int(right[q]["gt_seen_mask"]) & ~int(left[q]["gt_seen_mask"])).bit_count() for q in common
    ]
    print(f"queries={len(common)} right-left mean_recall={statistics.fmean(recall_delta):+.6f} "
          f"mean_iterations={statistics.fmean(iteration_delta):+.2f} "
          f"mean_new_gt_seen={statistics.fmean(gained_gt):+.3f}")
    print(f"recall improved/equal/worse="
          f"{sum(x > 0 for x in recall_delta)}/{sum(x == 0 for x in recall_delta)}/"
          f"{sum(x < 0 for x in recall_delta)}")


def export_csv(args: argparse.Namespace) -> None:
    run = Path(args.run)
    manifest = load_manifest(run)
    array = load_iterations(run, manifest) if args.table == "iteration" else load_candidates(run, manifest)
    selected = load_selected(run)
    rows = {int(row["query_id"]): row for row in load_summary(run)}
    valid_chunks = []
    for query, slot in selected.items():
        count = min(int(rows[query]["iterations"]), manifest["max_trace_iterations"])
        valid_chunks.append(array[slot, :count].reshape(-1))
    flat = np.concatenate(valid_chunks) if valid_chunks else array.reshape(-1)[:0]
    if args.table == "candidate":
        flat = flat[flat["valid"] != 0]
    output = Path(args.output)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(flat.dtype.names)
        for item in flat:
            writer.writerow(item[name].item() for name in flat.dtype.names)


def validate(args: argparse.Namespace) -> None:
    run = Path(args.run)
    manifest = load_manifest(run)
    rows = load_summary(run)
    errors: list[str] = []
    if len(rows) != manifest["num_queries"]:
        errors.append("query_summary row count does not match manifest")
    if manifest["iteration_record_size"] != ITERATION_DTYPE.itemsize:
        errors.append("iteration record size mismatch")
    if manifest["candidate_record_size"] != CANDIDATE_DTYPE.itemsize:
        errors.append("candidate record size mismatch")
    expected_iterations = manifest["trace_slots"] * manifest["max_trace_iterations"]
    expected_candidates = expected_iterations * manifest["candidates_per_iteration"]
    if len(binary_bytes(run, "iteration_trace")) != expected_iterations * ITERATION_DTYPE.itemsize:
        errors.append("iteration trace byte count mismatch")
    if len(binary_bytes(run, "candidate_trace")) != expected_candidates * CANDIDATE_DTYPE.itemsize:
        errors.append("candidate trace byte count mismatch")
    if errors:
        print("INVALID: " + "; ".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("OK")


def select_queries(args: argparse.Namespace) -> None:
    rows = load_summary(Path(args.run))
    entries = [(float(r["recall"]), int(r["stop_reason"]), int(r["query_id"])) for r in rows]

    def balanced(pool: list[tuple[float, int, int]], count: int) -> list[int]:
        result: list[int] = []
        by_reason: dict[int, list[tuple[float, int, int]]] = {}
        for item in pool:
            by_reason.setdefault(item[1], []).append(item)
        while len(result) < count and any(by_reason.values()):
            for reason in sorted(by_reason):
                if by_reason[reason] and len(result) < count:
                    result.append(by_reason[reason].pop(0)[2])
        return result

    low = sorted(entries)[: max(args.count * 3, args.count)]
    near = sorted(entries, key=lambda x: (abs(x[0] - args.target), x[2]))[: max(args.count * 3, args.count)]
    high = sorted(entries, key=lambda x: (-x[0], x[2]))[: max(args.count * 3, args.count)]
    chosen: list[int] = []
    for pool in (low, near, high):
        for query in balanced(pool, args.count):
            if query not in chosen:
                chosen.append(query)
    for _, _, query in sorted(entries, key=lambda x: (abs(x[0] - args.target), x[2])):
        if len(chosen) >= args.count * 3:
            break
        if query not in chosen:
            chosen.append(query)
    Path(args.output).write_text("".join(f"{query}\n" for query in chosen[: args.count * 3]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("summarize"); p.add_argument("run"); p.set_defaults(func=summarize)
    p = sub.add_parser("inspect-query"); p.add_argument("run"); p.add_argument("--query-id", type=int, required=True); p.add_argument("--candidates", action="store_true"); p.set_defaults(func=inspect_query)
    p = sub.add_parser("compare"); p.add_argument("left"); p.add_argument("right"); p.set_defaults(func=compare)
    p = sub.add_parser("export-csv"); p.add_argument("run"); p.add_argument("--table", choices=("iteration", "candidate"), required=True); p.add_argument("--output", required=True); p.set_defaults(func=export_csv)
    p = sub.add_parser("validate"); p.add_argument("run"); p.set_defaults(func=validate)
    p = sub.add_parser("select-queries"); p.add_argument("run"); p.add_argument("--target", type=float, default=0.9); p.add_argument("--count", type=int, default=16); p.add_argument("--output", required=True); p.set_defaults(func=select_queries)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
