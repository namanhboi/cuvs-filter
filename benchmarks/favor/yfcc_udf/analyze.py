#!/usr/bin/env python3
"""Analyze YFCC UDF benchmark output and generate the experiment report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path

import numpy as np


def benchmark_rows(result_root: Path, name: str) -> list[dict]:
    raw = json.loads((result_root / "raw" / f"{name}.json").read_text())["benchmarks"]
    rows = [r for r in raw if r.get("run_type") == "iteration"]
    by_family = {}
    for row in rows:
        by_family.setdefault(int(row["family_index"]), row)
    config = json.loads((result_root / "configs" / f"{name}.json").read_text())
    searches = config["index"][0]["search_params"]
    if set(by_family) != set(range(len(searches))):
        raise RuntimeError(f"incomplete benchmark result: {name}")
    output = []
    for index, search in enumerate(searches):
        row = dict(by_family[index])
        row["search"] = search
        output.append(row)
    return output


def variant(search: dict) -> str:
    if search.get("filter_mode", "default") == "default":
        return (
            "default_accumulator"
            if search.get("favor_udf_passing_accumulator", False)
            else "default_cagra"
        )
    suffix = "accumulator" if search.get("favor_udf_passing_accumulator", True) else "legacy"
    if search.get("favor_udf_sample_offset", 0):
        suffix += "_shifted"
    if search.get("favor_udf_include_sampling", True):
        suffix += "_end_to_end"
    return f"automatic_{suffix}"


def canonical_variant(variant_name: str) -> str:
    return variant_name[:-11] if variant_name.endswith("_end_to_end") else variant_name


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(row: dict, name: str, default: float = 0.0) -> float:
    value = row.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(row: dict, name: str, default: int = 0) -> int:
    value = row.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _query_batch_size(result_root: Path, benchmark_name: str) -> int:
    path = result_root / "raw" / f"{benchmark_name}.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return 0
    try:
        return int(data.get("context", {}).get("max_n_queries", 0))
    except (TypeError, ValueError):
        return 0


def _pretty_batch_size(batch_size: int) -> str:
    if batch_size <= 0:
        return "unknown batch size"
    if batch_size % 1_000_000 == 0:
        return f"{batch_size // 1_000_000}M"
    if batch_size % 1_000 == 0:
        return f"{batch_size // 1_000}k"
    return f"{batch_size}"


class Spmat:
    def __init__(self, path: Path):
        with path.open("rb") as stream:
            self.rows, self.cols, self.nnz = struct.unpack("<qqq", stream.read(24))
        self.offsets = np.memmap(path, "<i8", "r", offset=24, shape=(self.rows + 1,))
        self.columns = np.memmap(
            path, "<i4", "r", offset=24 + 8 * (self.rows + 1), shape=(self.nnz,)
        )

    def tags(self, row: int) -> frozenset[int]:
        return frozenset(
            int(x) for x in self.columns[int(self.offsets[row]) : int(self.offsets[row + 1])]
        )


def sampler_analysis(data_root: Path, selection_path: Path, result_root: Path) -> list[dict]:
    base = Spmat(data_root / "yfcc-10M/base.metadata.10M.spmat")
    query = Spmat(data_root / "yfcc-10M/query.metadata.public.100K.spmat")
    selected = json.loads(selection_path.read_text())["queries"]
    step = base.rows // 10_000
    sample_count = math.ceil(base.rows / step)
    exact = {int(row["query_id"]): int(row["passing_count"]) for row in selected}
    labels = {
        int(row["query_id"]): (int(row["arity"]), int(row["selectivity_decile"]) + 1)
        for row in selected
    }
    rows = []
    for offset in (0, 499):
        sampled_tags = [base.tags((offset + i * step) % base.rows) for i in range(sample_count)]
        for query_id in sorted(exact):
            required = query.tags(query_id)
            hits = sum(required.issubset(tags) for tags in sampled_tags)
            effective_hits = max(1, hits)
            estimate = effective_hits / sample_count
            exact_rate = exact[query_id] / base.rows
            arity, decile = labels[query_id]
            rows.append(
                {
                    "query_id": query_id,
                    "arity": arity,
                    "selectivity_decile": decile,
                    "sample_offset": offset,
                    "sample_count": sample_count,
                    "sample_hits": hits,
                    "underresolved": int(hits == 0),
                    "estimated_selectivity": estimate,
                    "exact_selectivity_posthoc": exact_rate,
                    "absolute_error": abs(estimate - exact_rate),
                    "estimate_over_exact": estimate / exact_rate,
                }
            )
    write_csv(result_root / "sampling_query_summary.csv", rows)
    return rows


def summarize_diagnostics(result_root: Path) -> list[dict]:
    output = []
    for path in sorted((result_root / "diagnostics").glob("*/query_summary.csv")):
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        manifest = json.loads((path.parent / "manifest.json").read_text())
        output.append(
            {
                "variant": path.parent.name,
                "queries": len(rows),
                "recall": statistics.fmean(float(r["recall"]) for r in rows),
                "gt_seen": statistics.fmean(int(r["gt_seen_mask"]).bit_count() / 10 for r in rows),
                "passing_discoveries": statistics.fmean(
                    float(r["passing_candidates"]) for r in rows
                ),
                "distance_evaluations": statistics.fmean(
                    float(r["candidate_evaluations"]) for r in rows
                ),
                "frontier_exhaustion": statistics.fmean(
                    r["stop_reason"] in {"2", "3"} for r in rows
                ),
                "underfilled": statistics.fmean(int(r["output_count"]) < 10 for r in rows),
                "block_size": int(manifest.get("block_size", 0)),
                "dynamic_smem_bytes": int(manifest.get("dynamic_smem_bytes", 0)),
                "active_blocks_per_sm": int(manifest.get("active_blocks_per_sm", 0)),
                "occupancy": float(manifest.get("occupancy", 0)),
            }
        )
    write_csv(result_root / "diagnostic_summary.csv", output)
    return output


def summarize_group_diagnostics(result_root: Path) -> list[dict]:
    output = []
    for path in sorted((result_root / "diagnostics" / "groups").glob("*/query_summary.csv")):
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        parts = path.parent.name.split("_")
        arity = int(parts[0][1:])
        decile = int(parts[1][1:])
        checkpoint = parts[2]
        output.append(
            {
                "arity": arity,
                "decile": decile,
                "checkpoint": checkpoint,
                "recall": statistics.fmean(float(r["recall"]) for r in rows),
                "gt_seen": statistics.fmean(int(r["gt_seen_mask"]).bit_count() / 10 for r in rows),
                "passing_discoveries": statistics.fmean(
                    float(r["passing_candidates"]) for r in rows
                ),
                "distance_evaluations": statistics.fmean(
                    float(r["candidate_evaluations"]) for r in rows
                ),
                "underfilled": statistics.fmean(int(r["output_count"]) < 10 for r in rows),
                "frontier_exhaustion": statistics.fmean(
                    r["stop_reason"] in {"2", "3"} for r in rows
                ),
            }
        )
    output.sort(key=lambda row: (row["arity"], row["decile"], row["checkpoint"] != "b0"))
    write_csv(result_root / "arity_decile_diagnostic_summary.csv", output)
    return output


def summarize_sampling_overhead(
    sampling_rows: list[dict],
    throughput_batch: int,
) -> tuple[list[dict], list[dict]]:
    by_key: dict[tuple[str, int, int, int], dict[str, dict]] = {}
    for row in sampling_rows:
        search = row["search"]
        if search.get("filter_mode") != "favor":
            continue
        include_sampling = bool(search.get("favor_udf_include_sampling", False))
        method = (
            "automatic_legacy"
            if not search.get("favor_udf_passing_accumulator", True)
            else "automatic_accumulator"
        )
        sample_key = _safe_float(row, "items_per_second", _safe_float(row, "qps", 0.0))
        sample_recall = _safe_float(row, "Recall", _safe_float(row, "recall", 0.0))
        key = (
            method,
            _safe_int(search, "itopk", 0),
            _safe_int(search, "search_width", 0),
            _safe_int(search, "max_iterations", 0),
        )
        bucket = by_key.setdefault(key, {})
        bucket["end_to_end" if include_sampling else "traversal"] = {
            "qps": sample_key,
            "recall": sample_recall,
        }

    per_point = []
    aggregates = []
    for (method, itopk, width, iterations), values in sorted(by_key.items()):
        traversal = values.get("traversal")
        end_to_end = values.get("end_to_end")
        if not traversal or not end_to_end:
            continue
        traversal_qps = traversal["qps"]
        end_to_end_qps = end_to_end["qps"]
        if traversal_qps <= 0.0 or end_to_end_qps <= 0.0:
            continue
        throughput_drop = (traversal_qps - end_to_end_qps) / traversal_qps * 100.0
        delta_ms_per_query = max(0.0, (1.0 / end_to_end_qps - 1.0 / traversal_qps) * 1000.0)
        delta_ms_batch = delta_ms_per_query * throughput_batch
        per_point.append(
            {
                "method": method,
                "itopk": itopk,
                "search_width": width,
                "max_iterations": iterations,
                "recall_traversal": traversal["recall"],
                "recall_end_to_end": end_to_end["recall"],
                "qps_traversal": traversal_qps,
                "qps_end_to_end": end_to_end_qps,
                "delta_qps": traversal_qps - end_to_end_qps,
                "delta_pct": throughput_drop,
                "delta_ms_per_query": delta_ms_per_query,
                "delta_ms_per_batch": delta_ms_batch,
            }
        )
        aggregates.append((method, throughput_drop, delta_ms_per_query, delta_ms_batch))

    aggregate_rows: list[dict] = []
    if aggregates:
        by_method: dict[str, list[tuple[float, float, float]]] = {}
        for method, drop_pct, delta_ms_query, delta_ms_batch in aggregates:
            by_method.setdefault(method, []).append(
                (drop_pct, delta_ms_query, delta_ms_batch)
            )
        for method, values in sorted(by_method.items()):
            drops = [v[0] for v in values]
            per_q = [v[1] for v in values]
            per_b = [v[2] for v in values]
            aggregate_rows.append(
                {
                    "method": method,
                    "samples": len(values),
                    "mean_drop_pct": statistics.fmean(drops),
                    "median_drop_pct": statistics.median(drops),
                    "mean_ms_per_query": statistics.fmean(per_q),
                    "mean_ms_per_10k": statistics.fmean(per_b),
                }
            )
    return per_point, aggregate_rows


def pareto_frontier(points: list[dict], maximize_y: bool = True) -> list[dict]:
    if not points:
        return []
    filtered = sorted(points, key=lambda p: (float(p["recall"]), float(p["qps"])), reverse=True)
    frontier: list[dict] = []
    best_qps = -1.0
    for row in filtered:
        x = float(row["recall"])
        y = float(row["qps"])
        if maximize_y:
            if y >= best_qps:
                frontier.append(row)
                best_qps = y
        else:
            if y <= best_qps or best_qps < 0:
                frontier.append(row)
                best_qps = y
    frontier.sort(key=lambda row: float(row["recall"]))
    return frontier


def plot_results(
    result_root: Path,
    correctness: list[dict],
    latency: list[dict],
    group_diagnostics: list[dict],
    throughput: list[dict],
    throughput_batch: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_rows = throughput or correctness
    rows_by_method: dict[str, list[dict]] = {}
    for row in plot_rows:
        if row.get("qps") is None:
            continue
        rows_by_method.setdefault(canonical_variant(row["variant"]), []).append(row)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method in sorted(rows_by_method):
        rows = rows_by_method[method]
        b0_rows = [r for r in rows if int(r.get("max_iterations", 0)) == 0]
        frontier = pareto_frontier(b0_rows if b0_rows else rows, maximize_y=True)
        if frontier:
            ax.plot(
                [r["recall"] for r in frontier],
                [r["qps"] for r in frontier],
                marker="o",
                label=method,
            )
    ax.set(
        xlabel="Recall@10",
        ylabel="QPS",
        title=f"YFCC B0 Pareto frontier ({_pretty_batch_size(throughput_batch)}-query batch)",
    )
    ax.set_xlim(0.0, 1.0)
    all_qps = [float(r["qps"]) for rows in rows_by_method.values() for r in rows]
    if all_qps:
        qps_max = max(all_qps)
        ax.set_ylim(bottom=0.0, top=qps_max * 1.05)
    else:
        ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "b0_recall_qps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method in sorted(rows_by_method):
        rows = rows_by_method[method]
        frontier = pareto_frontier(rows, maximize_y=True)
        if frontier:
            ax.plot(
                [r["recall"] for r in frontier],
                [r["qps"] for r in frontier],
                marker="o",
                label=method,
            )
    ax.set(
        xlabel="Recall@10",
        ylabel="QPS",
        title=f"YFCC Recall-vs-QPS Pareto sweep ({_pretty_batch_size(throughput_batch)}-query batch)",
    )
    ax.set_xlim(0.0, 1.0)
    all_qps = [float(r["qps"]) for rows in rows_by_method.values() for r in rows]
    if all_qps:
        qps_max = max(all_qps)
        ax.set_ylim(bottom=0.0, top=qps_max * 1.05)
    else:
        ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "qps_recall_sweep.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for arity, ax in zip((1, 2), axes, strict=True):
        for method in ("default_cagra", "automatic_accumulator", "automatic_accumulator_deep"):
            rows = [r for r in latency if r["arity"] == arity and r["variant"] == method]
            ax.plot(
                [r["decile"] for r in rows],
                [r["recall"] for r in rows],
                marker="o",
                label=method,
            )
        ax.set(title=f"Arity {arity}", xlabel="Exact-selectivity decile (post-hoc label)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Recall@10")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "recall_by_arity_decile.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for arity, ax in zip((1, 2), axes, strict=True):
        for checkpoint in ("b0", "deep"):
            rows = [
                r
                for r in group_diagnostics
                if r["arity"] == arity and r["checkpoint"] == checkpoint
            ]
            ax.plot(
                [r["decile"] for r in rows],
                [r["distance_evaluations"] for r in rows],
                marker="o",
                label=checkpoint,
            )
        ax.set(title=f"Arity {arity}", xlabel="Exact-selectivity decile (post-hoc label)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean exact distance evaluations/query")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "distance_work_by_arity_decile.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    correctness = []
    for row in benchmark_rows(args.result_root, "correctness"):
        search = row["search"]
        correctness.append(
            {
                "variant": variant(search),
                "itopk": _safe_int(search, "itopk", 0),
                "search_width": _safe_int(search, "search_width", 0),
                "max_iterations": _safe_int(search, "max_iterations", 0),
                "recall": _safe_float(row, "Recall"),
                "qps": _safe_float(row, "items_per_second"),
                "underfilled_queries": _safe_float(row, "UnderfilledQueries", 0.0),
                "missing_result_slots": _safe_float(row, "MissingResultSlots", 0.0),
                "filter_violations": _safe_float(row, "FilterViolations", 0.0),
                "invalid_sentinel_errors": _safe_float(row, "InvalidSentinelErrors", 0.0),
            }
        )
    write_csv(args.result_root / "correctness_summary.csv", correctness)

    throughput = []
    for row in benchmark_rows(args.result_root, "throughput"):
        search = row["search"]
        throughput.append(
            {
                "variant": variant(search),
                "recall": _safe_float(row, "Recall"),
                "qps": _safe_float(row, "items_per_second"),
                "itopk": _safe_int(search, "itopk", 0),
                "search_width": _safe_int(search, "search_width", 0),
                "max_iterations": _safe_int(search, "max_iterations", 0),
                "latency_seconds": _safe_float(row, "Latency", 0.0),
                "filter_violations": _safe_float(row, "FilterViolations", 0.0),
                "invalid_sentinel_errors": _safe_float(row, "InvalidSentinelErrors", 0.0),
            }
        )
    write_csv(args.result_root / "throughput_summary.csv", throughput)

    throughput_batch = _query_batch_size(args.result_root, "throughput")
    sampling_batch = _query_batch_size(args.result_root, "throughput_sampling")
    sampling_points: list[dict] = []
    if (args.result_root / "raw" / "throughput_sampling.json").exists():
        sampling_rows = []
        for row in benchmark_rows(args.result_root, "throughput_sampling"):
            search = row["search"]
            sampling_rows.append(
                {
                    "method": "automatic_legacy"
                    if search.get("favor_udf_passing_accumulator", True) is False
                    else "automatic_accumulator",
                    "variant": variant(search),
                    "recall": _safe_float(row, "Recall"),
                    "items_per_second": _safe_float(row, "items_per_second"),
                    "itopk": _safe_int(search, "itopk", 0),
                    "search_width": _safe_int(search, "search_width", 0),
                    "max_iterations": _safe_int(search, "max_iterations", 0),
                    "include_sampling": bool(search.get("favor_udf_include_sampling", False)),
                    "latency_seconds": _safe_float(row, "Latency", 0.0),
                    "filter_violations": _safe_float(row, "FilterViolations", 0.0),
                    "invalid_sentinel_errors": _safe_float(row, "InvalidSentinelErrors", 0.0),
                    "search": search,
                }
            )
        sampling_points, sampling_summary = summarize_sampling_overhead(
            sampling_rows, sampling_batch or throughput_batch
        )
        write_csv(
            args.result_root / "throughput_sampling_summary.csv",
            sampling_points,
        )
        write_csv(
            args.result_root / "sampling_overhead_summary.csv",
            sampling_summary,
        )
    else:
        sampling_points = []
        sampling_summary = []

    latency = []
    for arity in (1, 2):
        for decile in range(1, 11):
            for row in benchmark_rows(args.result_root, f"latency_a{arity}_d{decile}"):
                search = row["search"]
                method = variant(search)
                if _safe_int(search, "max_iterations", 0):
                    method = "automatic_accumulator_deep"
                latency.append(
                    {
                        "arity": arity,
                        "decile": decile,
                        "variant": method,
                        "recall": _safe_float(row, "Recall"),
                        "latency_seconds": _safe_float(row, "Latency", 0.0),
                        "underfilled_queries": _safe_float(row, "UnderfilledQueries", 0.0),
                        "filter_violations": _safe_float(row, "FilterViolations", 0.0),
                        "invalid_sentinel_errors": _safe_float(row, "InvalidSentinelErrors", 0.0),
                    }
                )
    write_csv(args.result_root / "arity_decile_summary.csv", latency)

    if sampling_batch <= 0:
        sampling_batch = throughput_batch
    samples = sampler_analysis(args.data_root, args.selection_json, args.result_root)
    diagnostics = summarize_diagnostics(args.result_root)
    group_diagnostics = summarize_group_diagnostics(args.result_root)
    plot_results(
        args.result_root,
        correctness,
        latency,
        group_diagnostics,
        throughput,
        throughput_batch,
    )

    correctness_by_variant: dict[str, list[dict]] = {}
    for row in correctness:
        correctness_by_variant.setdefault(canonical_variant(row["variant"]), []).append(row)
    throughput_by_variant: dict[str, list[dict]] = {}
    for row in throughput:
        throughput_by_variant.setdefault(canonical_variant(row["variant"]), []).append(row)

    def _best(rowset: list[dict], *, key) -> dict:
        if not rowset:
            raise RuntimeError("missing required variant group in benchmark results")
        return max(rowset, key=key)

    best_default = _best(correctness_by_variant.get("default_cagra", []), key=lambda r: r["recall"])
    best_default_accumulator = _best(
        correctness_by_variant.get("default_accumulator", []),
        key=lambda r: r["recall"],
    )
    best_legacy = _best(
        correctness_by_variant.get("automatic_legacy", []),
        key=lambda r: r["recall"],
    )
    best_accumulator = _best(
        correctness_by_variant.get("automatic_accumulator", []),
        key=lambda r: r["recall"],
    )
    deepest_auto = _best(
        correctness_by_variant.get("automatic_accumulator", []),
        key=lambda r: r["max_iterations"],
    )
    verdict = (
        "baseline_sufficient"
        if best_default["recall"] >= 0.905
        else "budget_opportunity"
        if deepest_auto["recall"] >= 0.905
        else "target_not_reached"
    )
    zero_rate = statistics.fmean(r["underresolved"] for r in samples if r["sample_offset"] == 0)
    primary_samples = [r for r in samples if r["sample_offset"] == 0]
    shifted_samples = [r for r in samples if r["sample_offset"] == 499]
    sample_hits = [r["sample_hits"] for r in primary_samples]
    sample_mae = statistics.fmean(r["absolute_error"] for r in primary_samples)
    sample_median_ratio = statistics.median(r["estimate_over_exact"] for r in primary_samples)
    sample_p95_error = float(np.quantile([r["absolute_error"] for r in primary_samples], 0.95))
    shifted_by_query = {r["query_id"]: r for r in shifted_samples}
    shifted_mean_delta = statistics.fmean(
        abs(r["estimated_selectivity"] - shifted_by_query[r["query_id"]]["estimated_selectivity"])
        for r in primary_samples
    )
    sample_overhead_rows = [
        f"| {row['method']} | {row['itopk']} | {row['search_width']} | {row['max_iterations']} | "
        f"{row['recall_traversal']:.4f} | {row['recall_end_to_end']:.4f} | "
        f"{row['qps_traversal']:.1f} | {row['qps_end_to_end']:.1f} | "
        f"{row['delta_pct']:.2f} | {row['delta_ms_per_query']:.3f} | {row['delta_ms_per_batch']:.1f} |"
        for row in sampling_points
    ]
    sample_summary_text = "\n".join(
        f"| {row['method']} | {row['samples']} | {row['mean_drop_pct']:.2f} | "
        f"{row['median_drop_pct']:.2f} | {row['mean_ms_per_query']:.3f} | {row['mean_ms_per_10k']:.1f} |"
        for row in sampling_summary
    ) or "| unavailable | 0 | 0.00 | 0.00 | 0.000 | 0.0 |"
    if sampling_points:
        sample_timing_text = (
            "Focused paired sampling-overhead sweep in `throughput_sampling.json` reports direct traversal "
            "vs end-to-end comparisons at matching (L, W, i) settings. "
            f"Batch size was {sampling_batch}."
        )
    else:
        sample_timing_text = "No traversal/e2e paired FAVOR run was found for sampling-overhead estimation."

    sampling_overhead_table = (
        """| Method | L | W | I | Recall (traversal) | Recall (end-to-end) | QPS (traversal) | QPS (end-to-end) | Sampling ΔQPS % | Δms/query | Δms/10000 queries |
|-
""" + "\n".join(sample_overhead_rows)
        if sample_overhead_rows
        else ""
    )
    sampling_summary_table = (
        "| Method | Samples | Mean ΔQPS % | Median ΔQPS % | Mean Δms/query | Mean Δms/10000 queries |\n"
        "|-\n"
        + sample_summary_text
    )

    sample_hit_text = (
        "The systematic sample evaluates 10,000 base rows per query. "
        f"Offset-zero samples had a {zero_rate:.1%} zero-hit/underresolved rate. "
        "Exact counts are used only here, after search, to measure estimator error; they are never loaded by cuVS."
    )
    throughput_timing_text = (
        "For the throughput sweep, all rows for FAVOR variants include end-to-end sampling."
        if any(v["variant"].endswith("_end_to_end") for v in throughput)
        else "For the throughput sweep, FAVOR variants are traversal-only (no end-to-end sampling)."
    )

    diagnostic_text = "\n".join(
        f"| {r['variant']} | {r['recall']:.4f} | {r['gt_seen']:.4f} | "
        f"{r['passing_discoveries']:.1f} | {r['distance_evaluations']:.0f} | "
        f"{r['underfilled']:.3f} | {r['frontier_exhaustion']:.3f} |"
        for r in diagnostics
    ) or "| unavailable | | | | | | |"
    best_rows = [
        best_default,
        best_default_accumulator,
        best_legacy,
        best_accumulator,
    ]
    correctness_text = "\n".join(
        f"| {canonical_variant(r['variant'])} | {r['itopk']} | {r['search_width']} | {r['max_iterations']} | "
        f"{r['recall']:.4f} | {r['qps']:.1f} | {r['underfilled_queries']:.3f} |"
        for r in best_rows
    )
    best_throughput_rows = [
        _best(throughput_by_variant[method], key=lambda r: r["qps"])
        for method in sorted(throughput_by_variant)
    ]
    throughput_text = "\n".join(
        f"| {canonical_variant(r['variant'])} | {r['recall']:.4f} | {r['qps']:.1f} | "
        f"{1000 * r['latency_seconds']:.3f} |"
        for r in best_throughput_rows
    )
    diag_by_variant = {r["variant"]: r for r in diagnostics}
    legacy_diag = diag_by_variant.get("legacy_b0")
    accumulator_diag = diag_by_variant.get("accumulator_b0")
    if legacy_diag and accumulator_diag:
        candidate_delta = accumulator_diag["distance_evaluations"] - legacy_diag["distance_evaluations"]
        seen_delta = accumulator_diag["gt_seen"] - legacy_diag["gt_seen"]
        accumulator_smem = accumulator_diag["dynamic_smem_bytes"] - legacy_diag["dynamic_smem_bytes"]
        resource_text = (
            f"At L=512, W=2, k=10 the accumulator adds {accumulator_smem} bytes of dynamic shared "
            f"memory ({legacy_diag['dynamic_smem_bytes']} to {accumulator_diag['dynamic_smem_bytes']} bytes). "
            f"Both kernels use {accumulator_diag['block_size']} threads; measured CUDA occupancy is "
            f"{legacy_diag['occupancy']:.1%} legacy and {accumulator_diag['occupancy']:.1%} with the "
            f"accumulator ({legacy_diag['active_blocks_per_sm']} versus "
            f"{accumulator_diag['active_blocks_per_sm']} active CTAs/SM)."
        )
        invariance_text = (
            f"The B0 accumulator-minus-legacy mean distance-work delta is {candidate_delta:.3f} and "
            f"the GT-seen delta is {seen_delta:.6f}; output recall can differ because passing "
            "neighbors are no longer evicted by the mixed traversal queue."
        )
    else:
        resource_text = "Launch-resource measurements are unavailable."
        invariance_text = "The diagnostic accumulator-invariance comparison is unavailable."
    args.report.write_text(
        f"""#+title: YFCC-10M SINGLE_CTA UDF Filtering Report

* Scope

This report contains only the GPU results produced by =benchmarks/favor/yfcc_udf=.  The predicate
is exact contains-all semantics: every query tag must occur in the candidate image's tag row.
Search never receives precomputed or exact selectivity; FAVOR samples the predicate on GPU.

* Correctness and sweep results

- Verdict: ={verdict}=.
- Best default CAGRA (any max_iterations): recall {best_default['recall']:.4f} at L={best_default['itopk']}, W={best_default['search_width']}, i={best_default['max_iterations']}.
- Best default CAGRA + passing-accumulator: recall {best_default_accumulator['recall']:.4f} at L={best_default_accumulator['itopk']}, W={best_default_accumulator['search_width']}, i={best_default_accumulator['max_iterations']}.
- Best automatic legacy: recall {best_legacy['recall']:.4f} at L={best_legacy['itopk']}, W={best_legacy['search_width']}, i={best_legacy['max_iterations']}.
- Best automatic accumulator: recall {best_accumulator['recall']:.4f} at L={best_accumulator['itopk']}, W={best_accumulator['search_width']}, i={best_accumulator['max_iterations']}.
- All reported runs have zero filter violations and zero invalid-sentinel errors (the analyzer fails otherwise).

| Method | L | W | Max iterations | Recall@10 | QPS | Underfilled queries |
|-
{correctness_text}

[[file:benchmarks/favor/yfcc_udf/results/plots/b0_recall_qps.png]]
[[file:benchmarks/favor/yfcc_udf/results/plots/qps_recall_sweep.png]]

* Sampling

{sample_hit_text}
{sample_timing_text}

The sample-hit distribution is min/median/p95/max = {min(sample_hits)}/
{statistics.median(sample_hits):.0f}/{np.quantile(sample_hits, 0.95):.0f}/{max(sample_hits)}.
Post-hoc mean absolute selectivity error is
{sample_mae:.6g}, p95 absolute error is {sample_p95_error:.6g}, and the median estimate/exact ratio
is {sample_median_ratio:.3f}.  Shifting the systematic schedule by 499 rows changes estimated
selectivity by {shifted_mean_delta:.6g} on average.

* Sampling overhead (traversal vs end-to-end)

Paired FAVOR runs from =throughput_sampling.json= compare the same (L, W, i) point with and without
selectivity sampling. Δ values report the end-to-end penalty relative to pure traversal.

{sampling_overhead_table if sampling_overhead_table else "| unavailable |  |  |  |  |  |  |  |  |  |  |"}

{sampling_summary_table if sampling_summary_table else "| unavailable | 0 | 0.00 | 0.00 | 0.000 | 0.0 |"}

* Diagnostics

| Variant | Recall | GT seen | Passing discoveries | Mean exact distance evaluations | Underfilled | Frontier exhaustion |
|-
{diagnostic_text}

{invariance_text}

{resource_text}

* Arity and selectivity deciles

Deciles are post-hoc workload labels and are not passed to search.

[[file:benchmarks/favor/yfcc_udf/results/plots/recall_by_arity_decile.png]]

The diagnostic companion records passing discoveries, GT-seen rate, underfill, frontier
exhaustion, and exact distance work for every arity/decile cell in
=arity_decile_diagnostic_summary.csv=.

[[file:benchmarks/favor/yfcc_udf/results/plots/distance_work_by_arity_decile.png]]

* Throughput

{throughput_timing_text}

| Method | Recall@10 | QPS | Batch latency (ms) |
|-
{throughput_text}

* Reproduction

The timed runs use one repetition after warmup on an NVIDIA L4.  JIT compilation, index loading,
and metadata upload are outside the timed search; the end-to-end FAVOR row includes sampling.

#+begin_src sh
benchmarks/favor/yfcc_udf/prepare_workloads.py --source /home/ubuntu/big-ann-benchmarks/data/yfcc100M --index /home/ubuntu/FAVOR/.artifacts/yfcc/yfcc10m.cagra_g32_ig64.index --delta-d /home/ubuntu/FAVOR/.artifacts/yfcc/yfcc10m.cagra_g32_ig64.index.delta_d --selection-json /home/ubuntu/FAVOR/.artifacts/yfcc/raw/automatic_retention_accumulator_L512_W2_accumulator.json --output datasets/yfcc-10M
benchmarks/favor/yfcc_udf/run_experiment.sh all
python benchmarks/favor/yfcc_udf/analyze.py --result-root benchmarks/favor/yfcc_udf/results --data-root datasets --selection-json /home/ubuntu/FAVOR/.artifacts/yfcc/raw/automatic_retention_accumulator_L512_W2_accumulator.json --report YFCC_SINGLE_CTA_UDF_REPORT.org
#+end_src
"""
    )

    if (
        any(r["filter_violations"] or r["invalid_sentinel_errors"] for r in correctness)
        or any(r["filter_violations"] or r["invalid_sentinel_errors"] for r in throughput)
        or any(r["filter_violations"] or r["invalid_sentinel_errors"] for r in latency)
    ):
        raise RuntimeError("correctness run contains filter violations or invalid sentinels")


if __name__ == "__main__":
    main()
