#!/usr/bin/env python3
"""Validate and aggregate a completed native CPU context run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_LABELS = {
    "faiss_navix": "FAISS-NaviX",
    "acorn_1": "ACORN-1 graph-only",
    "acorn_gamma": "ACORN-gamma graph-only",
}
WORKLOAD_LABELS = {
    "yfcc": "YFCC-10M", "em": "ArXiv-EM", "emis": "ArXiv-EMIS", "r": "ArXiv-R",
}
COLORS = {"faiss_navix": "#4c78a8", "acorn_1": "#e45756", "acorn_gamma": "#59a14f"}
MARKERS = {"faiss_navix": "o", "acorn_1": "s", "acorn_gamma": "^"}


@dataclass(frozen=True)
class PerRep:
    repetition: int
    workload: str
    method: str
    ef_search: int
    queries: int
    shards: int
    recall: float
    qps: float
    search_seconds: float
    filter_violations: int
    underfilled_queries: int
    intrinsic_underfilled_queries: int
    invalid_count_mismatch_queries: int
    sentinel_error_queries: int
    duplicate_output_queries: int
    search_threads: int
    chunk: int
    index_bytes: int


def integer(row: dict[str, str], key: str, path: pathlib.Path) -> int:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid/missing {key} in {path}: {error}") from error
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{key} must be a finite integer in {path}: {value}")
    return int(value)


def number(row: dict[str, str], key: str, path: pathlib.Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid/missing {key} in {path}: {error}") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite in {path}")
    return value


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: pathlib.Path, payload: Any) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def expected_counts(created: dict[str, Any], workload: str) -> list[int]:
    key = "yfcc_shard_counts" if workload == "yfcc" else "arxiv_shard_counts"
    return [int(value) for value in created["run_config"][key]]


def load_per_rep(run_root: pathlib.Path, created: dict[str, Any]) -> list[PerRep]:
    config = created["run_config"]
    repetitions = int(config["repetitions"])
    if repetitions != 3:
        raise ValueError(f"production analysis requires exactly three repetitions, got {repetitions}")
    methods = list(config["methods"])
    workloads = list(config["workloads"])
    ef_values = [int(value) for value in config["ef_search"]]
    if len(ef_values) != len(set(ef_values)):
        raise ValueError("efSearch list contains duplicates")

    result: list[PerRep] = []
    for repetition in range(1, repetitions + 1):
        rep_root = run_root / f"rep_{repetition:02d}" / "raw"
        if not rep_root.is_dir():
            raise FileNotFoundError(rep_root)
        for method in methods:
            for workload in workloads:
                directory = rep_root / method / workload
                files = sorted(directory.glob("shard_*.csv")) if directory.is_dir() else []
                counts = expected_counts(created, workload)
                expected_names = []
                cursor = 0
                for count in counts:
                    expected_names.append(f"shard_{cursor:05d}_{cursor + count:05d}.csv")
                    cursor += count
                if [path.name for path in files] != expected_names:
                    raise ValueError(
                        f"shard set mismatch for rep={repetition}, {method}/{workload}: "
                        f"expected {expected_names}, found {[path.name for path in files]}"
                    )
                by_ef: dict[int, list[tuple[pathlib.Path, dict[str, str]]]] = {
                    ef: [] for ef in ef_values
                }
                for shard_index, (path, expected_queries) in enumerate(zip(files, counts)):
                    with path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    if len(rows) != len(ef_values):
                        raise ValueError(f"incomplete efSearch curve in {path}")
                    seen_ef: set[int] = set()
                    for row in rows:
                        ef = integer(row, "ef_search", path)
                        if ef not in by_ef or ef in seen_ef:
                            raise ValueError(f"unexpected/duplicate efSearch={ef} in {path}")
                        seen_ef.add(ef)
                        if row.get("method") != method:
                            raise ValueError(f"method mismatch in {path}: {row.get('method')} != {method}")
                        if integer(row, "queries", path) != expected_queries:
                            raise ValueError(f"query count mismatch in {path}")
                        expected_chunk = 512 if workload == "yfcc" else 10_000
                        expected_threads = 16 if method == "faiss_navix" and workload != "yfcc" else 32
                        if integer(row, "chunk", path) != expected_chunk:
                            raise ValueError(f"chunk mismatch in {path}")
                        if integer(row, "search_threads", path) != expected_threads:
                            raise ValueError(f"thread mismatch in {path}")
                        if number(row, "build_seconds", path) != 0:
                            raise ValueError(f"graph construction occurred during search run: {path}")
                        expected_index_bytes = int(
                            created["graph_configurations"][f"{method}/{workload}"]["bytes"]
                        )
                        if integer(row, "index_bytes", path) != expected_index_bytes:
                            raise ValueError(f"index byte-size mismatch in {path}")
                        if method.startswith("acorn"):
                            expected_seeds = 10 if method.endswith("_navix_seeded") else 0
                            if integer(row, "filtered_seeds", path) != expected_seeds:
                                raise ValueError(f"filtered-seed setting mismatch in {path}")
                        seconds = number(row, "search_seconds", path)
                        qps = number(row, "qps", path)
                        recall = number(row, "recall", path)
                        if seconds <= 0 or qps <= 0 or not 0 <= recall <= 1:
                            raise ValueError(f"invalid timing/recall in {path}")
                        if abs(qps - expected_queries / seconds) > max(0.1, 0.002 * qps):
                            raise ValueError(f"qps/search_seconds inconsistency in {path}")
                        for field in (
                            "filter_violations", "underfilled_queries",
                            "intrinsic_underfilled_queries", "invalid_count_mismatch_queries",
                            "sentinel_error_queries", "duplicate_output_queries",
                        ):
                            value = integer(row, field, path)
                            if value < 0 or value > expected_queries:
                                raise ValueError(f"invalid {field} in {path}")
                        for field in (
                            "filter_violations", "sentinel_error_queries", "duplicate_output_queries"
                        ):
                            if integer(row, field, path) != 0:
                                raise ValueError(f"correctness failure: {field} != 0 in {path}")
                        if integer(row, "intrinsic_underfilled_queries", path) > integer(
                            row, "underfilled_queries", path
                        ):
                            raise ValueError(f"intrinsic underfill exceeds total underfill in {path}")
                        by_ef[ef].append((path, row))
                    if seen_ef != set(ef_values):
                        raise ValueError(f"efSearch set mismatch in {path}")

                for ef in ef_values:
                    members = by_ef[ef]
                    if len(members) != len(counts):
                        raise ValueError(f"missing shard at efSearch={ef} for {method}/{workload}")
                    queries = sum(integer(row, "queries", path) for path, row in members)
                    if queries != 10_000:
                        raise ValueError(
                            f"aggregate queries={queries}, expected 10000 for "
                            f"rep={repetition}, {method}/{workload}, ef={ef}"
                        )
                    seconds = sum(number(row, "search_seconds", path) for path, row in members)
                    recall = sum(
                        number(row, "recall", path) * integer(row, "queries", path)
                        for path, row in members
                    ) / queries
                    # In particular, YFCC is exactly 10,000 divided by the sum of all five shard
                    # times. Never average shard QPS.
                    qps = queries / seconds
                    threads = {
                        integer(row, "search_threads", path) for path, row in members
                    }
                    chunks = {integer(row, "chunk", path) for path, row in members}
                    index_bytes = {
                        integer(row, "index_bytes", path) for path, row in members
                    }
                    if len(threads) != 1 or len(chunks) != 1 or len(index_bytes) != 1:
                        raise ValueError(f"inconsistent resource/index metadata for {method}/{workload}")
                    result.append(PerRep(
                        repetition, workload, method, ef, queries, len(members), recall, qps, seconds,
                        sum(integer(row, "filter_violations", path) for path, row in members),
                        sum(integer(row, "underfilled_queries", path) for path, row in members),
                        sum(integer(row, "intrinsic_underfilled_queries", path) for path, row in members),
                        sum(integer(row, "invalid_count_mismatch_queries", path) for path, row in members),
                        sum(integer(row, "sentinel_error_queries", path) for path, row in members),
                        sum(integer(row, "duplicate_output_queries", path) for path, row in members),
                        next(iter(threads)), next(iter(chunks)), next(iter(index_bytes)),
                    ))
    return result


def summarize(rows: list[PerRep]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[PerRep]] = {}
    for row in rows:
        groups.setdefault((row.workload, row.method, row.ef_search), []).append(row)
    result: list[dict[str, Any]] = []
    for (workload, method, ef), members in sorted(groups.items()):
        if len(members) != 3 or {row.repetition for row in members} != {1, 2, 3}:
            raise ValueError(f"incorrect repetition group for {method}/{workload}/ef={ef}")
        if len({row.recall for row in members}) > 1:
            # Deterministic CPU graph traversal is expected to return identical outputs; retain
            # min/max regardless, but flag drift rather than silently averaging unlike results.
            drift = max(row.recall for row in members) - min(row.recall for row in members)
            if drift > 1e-9:
                raise ValueError(f"recall changed across repetitions for {method}/{workload}/ef={ef}")
        record: dict[str, Any] = {
            "workload": workload, "method": method, "ef_search": ef,
            "repetitions": 3, "queries_per_repetition": 10_000,
            "shards": members[0].shards, "search_threads": members[0].search_threads,
            "chunk": members[0].chunk, "index_bytes": members[0].index_bytes,
        }
        for field in ("recall", "qps", "search_seconds"):
            values = [float(getattr(row, field)) for row in members]
            record[f"{field}_median"] = statistics.median(values)
            record[f"{field}_min"] = min(values)
            record[f"{field}_max"] = max(values)
        for field in (
            "filter_violations", "underfilled_queries", "intrinsic_underfilled_queries",
            "invalid_count_mismatch_queries", "sentinel_error_queries", "duplicate_output_queries",
        ):
            values = [int(getattr(row, field)) for row in members]
            record[f"{field}_min"] = min(values)
            record[f"{field}_max"] = max(values)
        result.append(record)
    return result


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    best_qps = -math.inf
    for row in sorted(rows, key=lambda item: (item["recall_median"], item["qps_median"]), reverse=True):
        if row["qps_median"] > best_qps:
            frontier.append(row)
            best_qps = row["qps_median"]
    return sorted(frontier, key=lambda item: item["recall_median"])


def plot_summary(path: pathlib.Path, summary: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for ax, workload in zip(axes.flat, ("yfcc", "em", "emis", "r")):
        workload_rows = [row for row in summary if row["workload"] == workload]
        for method, label in METHOD_LABELS.items():
            points = pareto([row for row in workload_rows if row["method"] == method])
            if not points:
                raise ValueError(f"no plotted points for {method}/{workload}")
            x = [row["recall_median"] for row in points]
            y = [row["qps_median"] for row in points]
            xerr = [
                [row["recall_median"] - row["recall_min"] for row in points],
                [row["recall_max"] - row["recall_median"] for row in points],
            ]
            yerr = [
                [row["qps_median"] - row["qps_min"] for row in points],
                [row["qps_max"] - row["qps_median"] for row in points],
            ]
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr, label=label, color=COLORS[method],
                marker=MARKERS[method], linewidth=1.8, markersize=5, capsize=2,
            )
        all_recall = [row["recall_median"] for row in workload_rows if row["method"] in METHOD_LABELS]
        lower = max(0.0, min(all_recall) - max(0.01, 0.04 * (max(all_recall) - min(all_recall))))
        ax.set_xlim(lower, min(1.005, max(all_recall) + 0.02))
        ax.set_ylim(bottom=0)
        ax.set_yscale("linear")
        ax.grid(True, alpha=0.25)
        ax.set_title(WORKLOAD_LABELS[workload])
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("Queries/s (native CPU search call)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Native CPU bitmap-search context (median of 3 repetitions)", y=1.035)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def seeded_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Method | efSearch | Recall (median [min,max]) | QPS (median [min,max]) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {WORKLOAD_LABELS[row['workload']]} | {row['method']} | {row['ef_search']} | "
            f"{row['recall_median']:.5f} [{row['recall_min']:.5f},{row['recall_max']:.5f}] | "
            f"{row['qps_median']:.1f} [{row['qps_min']:.1f},{row['qps_max']:.1f}] |"
        )
    return "\n".join(lines) + "\n"


def analyze_run(run_root: pathlib.Path, output: pathlib.Path | None = None) -> pathlib.Path:
    created_path = run_root / "run.created.json"
    completed_path = run_root / "run.completed.json"
    if not created_path.is_file() or not completed_path.is_file():
        raise FileNotFoundError("analysis requires run.created.json and run.completed.json")
    created = json.loads(created_path.read_text())
    completed = json.loads(completed_path.read_text())
    if completed.get("run_id") != created.get("run_id"):
        raise ValueError("run ID mismatch between created/completed manifests")
    output = output or run_root / "analysis"
    per_rep = load_per_rep(run_root, created)
    summary = summarize(per_rep)
    expected_commands = (
        int(created["run_config"]["repetitions"])
        * len(created["run_config"]["methods"])
        * (5 + 1 + 1 + 1)
    )
    if "commands_completed" in completed and int(completed["commands_completed"]) != expected_commands:
        raise ValueError(
            f"completed command count {completed['commands_completed']} != {expected_commands}"
        )
    output.mkdir(parents=True, exist_ok=False)
    per_rep_dicts = [asdict(row) for row in per_rep]
    write_csv(output / "per_rep.csv", per_rep_dicts)
    write_json(output / "per_rep.json", {
        "schema_version": 1, "run_id": created["run_id"], "rows": per_rep_dicts,
    })
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", {
        "schema_version": 1, "run_id": created["run_id"], "rows": summary,
    })
    seeded_methods = set(created["run_config"]["seeded_methods"])
    seeded = [row for row in summary if row["method"] in seeded_methods]
    write_csv(output / "seeded_acorn_summary.csv", seeded)
    write_json(output / "seeded_acorn_summary.json", {
        "schema_version": 1, "run_id": created["run_id"], "rows": seeded,
    })
    with (output / "seeded_acorn_summary.md").open("x") as stream:
        stream.write(seeded_markdown(seeded))
    plot_summary(output / "cpu_context_pareto.png", summary)
    write_json(output / "validation.json", {
        "schema_version": 1, "status": "PASS", "repetitions": 3,
        "queries_per_method_workload_ef_repetition": 10_000,
        "correctness": {
            "filter_violations": 0, "sentinel_error_queries": 0,
            "duplicate_output_queries": 0,
        },
        "yfcc_qps_formula": "10000 / sum(five shard search_seconds)",
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    print(analyze_run(args.run_root, args.output))


if __name__ == "__main__":
    main()
