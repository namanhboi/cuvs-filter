#!/usr/bin/env python3
"""Aggregate optimized GPU NaviX and native CPU bitmap baseline results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

WORKLOADS = ("yfcc", "em", "emis", "r")
WORKLOAD_LABELS = {"yfcc": "YFCC-10M", "em": "ArXiv-EM", "emis": "ArXiv-EMIS", "r": "ArXiv-R"}
GPU_LABELS = {
    "default_cagra": "Default CAGRA",
    "default_cagra_accumulator": "CAGRA + accumulator",
    "navix_reference": "GPU NaviX reference",
    "navix_optimized": "NaviX direct lookup (rejected)",
}
GPU_PRIMARY_LABELS = {
    name: label for name, label in GPU_LABELS.items() if name != "navix_optimized"
}
CPU_LABELS = {
    "faiss_navix": "FAISS-NaviX",
    "acorn_1": "ACORN-1",
    "acorn_1_navix_seeded": "ACORN-1 + NaviX seeds",
    "acorn_gamma": "ACORN-gamma",
    "acorn_gamma_navix_seeded": "ACORN-gamma + NaviX seeds",
}
ACORN_SEED_LABELS = {
    name: CPU_LABELS[name]
    for name in (
        "acorn_1", "acorn_1_navix_seeded", "acorn_gamma", "acorn_gamma_navix_seeded"
    )
}
COLORS = {
    "default_cagra": "#4c78a8", "default_cagra_accumulator": "#e45756",
    "navix_reference": "#f58518", "navix_optimized": "#54a24b",
    "faiss_navix": "#b279a2", "acorn_1": "#9d755d", "acorn_gamma": "#59a14f",
    "acorn_1_navix_seeded": "#e15759", "acorn_gamma_navix_seeded": "#76b7b2",
}
MARKERS = {
    name: marker
    for name, marker in zip(COLORS, ("o", "D", "^", "s", "P", "X", "v", "<", "h"))
}


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def number(row: dict, key: str, path: Path) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {path}")
    return value


@dataclass(frozen=True)
class GPUPoint:
    group: str
    workload: str
    shard: str
    method: str
    queries: int
    recall: float
    qps: float
    itopk: int
    width: int
    max_iterations: int
    block_size: int
    filter_violations: float
    sentinel_errors: float
    underfilled_queries: float
    missing_result_slots: float


@dataclass(frozen=True)
class Aggregate:
    platform: str
    group: str
    workload: str
    method: str
    queries: int
    shards: int
    recall: float
    qps: float
    parameter: str
    filter_violations: float
    sentinel_errors: float
    underfilled_queries: float
    missing_result_slots: float
    duplicate_output_queries: float


def gpu_points(root: Path) -> list[GPUPoint]:
    result = []
    errors = []
    for path in sorted((root / "raw").glob("*/*/shard_*.json")):
        group, workload = path.parts[-3:-1]
        payload = json.loads(path.read_text())
        for row in payload.get("benchmarks", []):
            if row.get("run_type") != "iteration":
                continue
            if row.get("error_occurred"):
                errors.append(f"{path}: {row.get('error_message', 'benchmark error')}")
                continue
            label = str(row.get("label", ""))
            method = label_value(label, "bitmap_method")
            if method not in GPU_LABELS:
                continue
            result.append(
                GPUPoint(
                    group, workload, path.stem, method,
                    round(number(row, "n_queries", path)), number(row, "Recall", path),
                    number(row, "items_per_second", path), round(number(row, "itopk", path)),
                    round(number(row, "search_width", path)),
                    round(number(row, "max_iterations", path)),
                    round(float(row.get("thread_block_size", 0))),
                    float(row.get("FilterViolations", 0)),
                    float(row.get("InvalidSentinelErrors", 0)),
                    float(row.get("UnderfilledQueries", 0)),
                    float(row.get("MissingResultSlots", 0)),
                )
            )
    if errors:
        raise SystemExit("GPU benchmark failures:\n" + "\n".join(errors))
    return result


def aggregate_gpu(points: list[GPUPoint]) -> list[Aggregate]:
    groups: dict[tuple, list[GPUPoint]] = {}
    for p in points:
        key = (p.group, p.workload, p.method, p.itopk, p.width, p.max_iterations, p.block_size)
        groups.setdefault(key, []).append(p)
    rows = []
    for key, members in sorted(groups.items()):
        group, workload, method, L, W, iterations, threads = key
        queries = sum(x.queries for x in members)
        seconds = sum(x.queries / x.qps for x in members)
        recall = sum(x.recall * x.queries for x in members) / queries
        rows.append(Aggregate(
            "GPU", group, workload, method, queries, len(members), recall, queries / seconds,
            f"L={L},W={W},iterations={iterations},threads={threads}",
            sum(x.filter_violations for x in members), sum(x.sentinel_errors for x in members),
            sum(x.underfilled_queries for x in members),
            sum(x.missing_result_slots for x in members), 0))
    return rows


def aggregate_cpu(root: Path) -> list[Aggregate]:
    rows = []
    for method in CPU_LABELS:
        for workload in WORKLOADS:
            # Only consume the clean, explicitly resource-labelled measurements.  The artifact
            # tree also retains older unsuffixed eight-thread CSVs for provenance; mixing those
            # with the final _t8 files silently double-counts each query shard and distorts QPS.
            files = [
                path
                for path in sorted((root / "results" / method / workload).glob("shard_*.csv"))
                if re.search(r"_t(?:8|16|32)\.csv$", path.name)
                and "sensitivity" not in path.name
            ]
            by_parameter: dict[str, list[dict]] = {}
            for path in files:
                for row in csv.DictReader(path.open()):
                    match = re.search(r"_t(8|16|32)\.csv$", path.name)
                    if match is None:
                        raise ValueError(f"missing thread suffix in {path}")
                    threads = row.get("search_threads") or match.group(1)
                    if threads != match.group(1):
                        raise ValueError(
                            f"thread metadata {threads} disagrees with {path.name}"
                        )
                    parameter = (
                        f"efSearch={row['ef_search']},chunk={row['chunk']},threads={threads}"
                    )
                    by_parameter.setdefault(parameter, []).append(row)
            for parameter, members in by_parameter.items():
                queries = sum(int(x["queries"]) for x in members)
                seconds = sum(float(x["search_seconds"]) for x in members)
                recall = sum(float(x["recall"]) * int(x["queries"]) for x in members) / queries
                rows.append(Aggregate(
                    "CPU", "ef_search", workload, method, queries, len(members), recall,
                    queries / seconds, parameter,
                    sum(float(x["filter_violations"]) for x in members),
                    sum(float(x.get("sentinel_error_queries", 0)) for x in members),
                    sum(float(x.get("underfilled_queries", 0)) for x in members),
                    0,
                    sum(float(x.get("duplicate_output_queries", 0)) for x in members)))
    return rows


def pareto(rows: list[Aggregate]) -> list[Aggregate]:
    frontier, best = [], -math.inf
    for row in sorted(rows, key=lambda x: (x.recall, x.qps), reverse=True):
        if row.qps > best:
            frontier.append(row)
            best = row.qps
    return sorted(frontier, key=lambda x: x.recall)


def draw(path: Path, workload: str, rows: list[Aggregate], labels: dict[str, str], title: str) -> None:
    selected = [x for x in rows if x.workload == workload and x.method in labels]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    for method, label in labels.items():
        method_rows = [x for x in selected if x.method == method]
        if not method_rows:
            continue
        ax.scatter([x.recall for x in method_rows], [x.qps for x in method_rows],
                   color=COLORS[method], marker=MARKERS[method], alpha=.25, s=30)
        front = pareto(method_rows)
        ax.plot([x.recall for x in front], [x.qps for x in front], color=COLORS[method],
                marker=MARKERS[method], linewidth=2, label=label)
    minimum = min(x.recall for x in selected)
    ax.set_xlim(max(0, minimum - .02), 1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second")
    ax.set_title(f"{WORKLOAD_LABELS[workload]}: {title}")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=asdict(rows[0]).keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-results",
        type=Path,
        action="append",
        required=True,
        help="GPU result root; repeat the option to combine B0 and deeper sweeps",
    )
    parser.add_argument("--cpu-artifacts", type=Path, default=Path("/home/ubuntu/navix_cpu_artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    points = [point for root in args.gpu_results for point in gpu_points(root)]
    gpu = [x for x in aggregate_gpu(points) if x.group.startswith("sweep_")]
    cpu = aggregate_cpu(args.cpu_artifacts)
    write_csv(args.output / "gpu_points.csv", points)
    write_csv(args.output / "gpu_aggregate.csv", gpu)
    write_csv(args.output / "cpu_aggregate.csv", cpu)
    for workload in WORKLOADS:
        draw(args.output / f"{workload}_gpu_qps_recall.png", workload, gpu, GPU_PRIMARY_LABELS,
             "GPU bitmap search, 10,000 queries")
        draw(args.output / f"{workload}_cpu_qps_recall.png", workload, cpu, CPU_LABELS,
             "CPU bitmap search, 10,000 queries")
        draw(args.output / f"{workload}_acorn_seed_ablation.png", workload, cpu,
             ACORN_SEED_LABELS, "ACORN deterministic-seed ablation, 10,000 queries")
        draw(args.output / f"{workload}_cpu_gpu_overlay.png", workload, gpu + cpu,
             {**GPU_PRIMARY_LABELS, **CPU_LABELS}, "absolute CPU/GPU QPS (appendix)")
    violations = sum(
        x.filter_violations + x.sentinel_errors + x.duplicate_output_queries
        for x in gpu + cpu
    )
    summary = {
        "gpu_raw_points": len(points), "gpu_aggregate_points": len(gpu),
        "cpu_aggregate_points": len(cpu), "correctness_error_total": violations,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
