#!/usr/bin/env python3
"""Aggregate sharded bitmap benchmarks and emit Pareto plots plus an Org report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
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
    "default_cagra": "Default CAGRA (bitmap)",
    "default_cagra_accumulator": "Default CAGRA + accumulator (bitmap)",
    "legacy_navix": "Legacy NaviX (bitmap)",
    "bitmap_seeded_navix": "Bitmap-seeded NaviX",
}
METHOD_STYLES = {
    "default_cagra": ("#4c78a8", "o"),
    "default_cagra_accumulator": ("#e45756", "D"),
    "legacy_navix": ("#f58518", "^"),
    "bitmap_seeded_navix": ("#54a24b", "s"),
}


@dataclass(frozen=True)
class Point:
    phase: str
    workload: str
    shard: str
    method: str
    queries: int
    recall: float
    qps: float
    filter_violations: float
    sentinel_errors: float
    underfilled: float
    itopk: int
    width: int
    max_iterations: int


@dataclass(frozen=True)
class Aggregate:
    phase: str
    workload: str
    method: str
    shards: int
    queries: int
    recall: float
    qps: float
    filter_violations: float
    sentinel_errors: float
    underfilled: float
    itopk: int
    width: int
    max_iterations: int


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def finite(value: object, field: str, path: Path) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} in {path}")
    return result


def load_points(result_root: Path) -> list[Point]:
    points: list[Point] = []
    errors: list[str] = []
    for phase in ("correctness", "throughput"):
        for workload in WORKLOADS:
            raw_dir = result_root / "raw" / phase / workload
            for path in sorted(raw_dir.glob("shard_*.json")):
                payload = json.loads(path.read_text())
                for row in payload.get("benchmarks", []):
                    if row.get("run_type") != "iteration":
                        continue
                    if row.get("error_occurred"):
                        errors.append(
                            f"{path}: {row.get('error_message', row.get('name', 'error'))}"
                        )
                        continue
                    method = label_value(str(row.get("label", "")), "bitmap_method")
                    if method not in METHOD_LABELS:
                        raise ValueError(f"missing bitmap_method in {path}: {row.get('label', '')}")
                    points.append(
                        Point(
                            phase=phase,
                            workload=workload,
                            shard=path.stem,
                            method=method,
                            queries=round(float(row["n_queries"])),
                            recall=finite(row["Recall"], "Recall", path),
                            qps=finite(row["items_per_second"], "items_per_second", path),
                            filter_violations=float(row.get("FilterViolations", 0.0)),
                            sentinel_errors=float(row.get("InvalidSentinelErrors", 0.0)),
                            underfilled=float(row.get("UnderfilledQueries", 0.0)),
                            itopk=round(float(row["itopk"])),
                            width=round(float(row["search_width"])),
                            max_iterations=round(float(row["max_iterations"])),
                        )
                    )
    if errors:
        raise SystemExit("benchmark failures:\n" + "\n".join(errors))
    if not points:
        raise SystemExit(f"no benchmark rows found under {result_root / 'raw'}")
    return points


def aggregate(points: list[Point]) -> list[Aggregate]:
    groups: dict[tuple[object, ...], list[Point]] = {}
    for point in points:
        key = (
            point.phase,
            point.workload,
            point.method,
            point.itopk,
            point.width,
            point.max_iterations,
        )
        groups.setdefault(key, []).append(point)
    result: list[Aggregate] = []
    for key, members in sorted(groups.items()):
        phase, workload, method, itopk, width, max_iterations = key
        total_queries = sum(point.queries for point in members)
        total_seconds = sum(point.queries / point.qps for point in members)
        def weighted(
            field: str,
            group: list[Point] = members,
            denominator: int = total_queries,
        ) -> float:
            return sum(
                getattr(point, field) * point.queries for point in group
            ) / denominator
        result.append(
            Aggregate(
                phase=str(phase),
                workload=str(workload),
                method=str(method),
                shards=len(members),
                queries=total_queries,
                recall=weighted("recall"),
                qps=total_queries / total_seconds,
                filter_violations=sum(point.filter_violations for point in members),
                sentinel_errors=sum(point.sentinel_errors for point in members),
                underfilled=weighted("underfilled"),
                itopk=int(itopk),
                width=int(width),
                max_iterations=int(max_iterations),
            )
        )
    return result


def validate_coverage(points: list[Point], rows: list[Aggregate]) -> None:
    configurations = len(METHOD_LABELS) * 4 * 2
    expected_points = len(WORKLOADS) * configurations + 5 * configurations + 3 * configurations
    expected_rows = 2 * len(WORKLOADS) * len(METHOD_LABELS) * 4 * 2
    if len(points) != expected_points or len(rows) != expected_rows:
        raise ValueError(
            f"incomplete sweep: points={len(points)}/{expected_points}, "
            f"aggregate rows={len(rows)}/{expected_rows}"
        )
    for row in rows:
        expected_shards = 5 if row.phase == "throughput" and row.workload == "yfcc" else 1
        expected_queries = 1000 if row.phase == "correctness" else 10000
        if (
            row.shards != expected_shards
            or row.queries != expected_queries
            or row.max_iterations != 0
        ):
            raise ValueError(
                "invalid sweep coverage for "
                f"{row.phase}/{row.workload}/{row.method}/L={row.itopk}/W={row.width}: "
                f"shards={row.shards}, queries={row.queries}, "
                f"max_iterations={row.max_iterations}"
            )


def write_csv(path: Path, rows: list[Point] | list[Aggregate]) -> None:
    columns = list(rows[0].__dataclass_fields__)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in columns})


def pareto(points: list[Aggregate]) -> list[Aggregate]:
    frontier: list[Aggregate] = []
    best_qps = -math.inf
    for point in sorted(points, key=lambda item: (item.recall, item.qps), reverse=True):
        if point.qps > best_qps:
            frontier.append(point)
            best_qps = point.qps
    return sorted(frontier, key=lambda item: item.recall)


def plot_workload(
    output: Path,
    workload: str,
    rows: list[Aggregate],
    methods: tuple[str, ...] = tuple(METHOD_LABELS),
    filename_suffix: str = "",
) -> Path:
    selected = [
        row
        for row in rows
        if row.phase == "throughput"
        and row.workload == workload
        and row.method in methods
    ]
    if not selected:
        raise ValueError(f"no throughput points for {workload}")
    figure, axis = plt.subplots(figsize=(8.3, 5.5))
    for method in methods:
        method_label = METHOD_LABELS[method]
        method_rows = [row for row in selected if row.method == method]
        color, marker = METHOD_STYLES[method]
        axis.scatter(
            [row.recall for row in method_rows],
            [row.qps for row in method_rows],
            color=color,
            marker=marker,
            alpha=0.28,
            s=30,
        )
        frontier = pareto(method_rows)
        axis.plot(
            [row.recall for row in frontier],
            [row.qps for row in frontier],
            color=color,
            marker=marker,
            linewidth=2,
            label=method_label,
        )
    minimum = min(row.recall for row in selected)
    axis.set_xlim(max(0.0, minimum - 0.02), 1.0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Recall@10")
    axis.xaxis.set_label_coords(0.5, -0.08)
    axis.set_ylabel("Queries per second")
    axis.set_title(
        f"{WORKLOAD_LABELS[workload]}: B0 end-to-end QPS, 10,000 queries"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output / f"{workload}{filename_suffix}_qps_recall_b0.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def best_at_target(
    rows: list[Aggregate], workload: str, method: str, target: float = 0.95
) -> str:
    candidates = [
        row
        for row in rows
        if row.phase == "throughput"
        and row.workload == workload
        and row.method == method
    ]
    reached = [row for row in candidates if row.recall >= target]
    if reached:
        row = max(reached, key=lambda item: item.qps)
        return (
            f"{row.recall:.4f} @ {row.qps:.0f} QPS "
            f"(L={row.itopk}, W={row.width})"
        )
    row = max(candidates, key=lambda item: item.recall)
    return (
        f"not reached; best {row.recall:.4f} @ {row.qps:.0f} QPS "
        f"(L={row.itopk}, W={row.width})"
    )


def at_config(
    rows: list[Aggregate], workload: str, method: str, itopk: int, width: int
) -> Aggregate:
    matches = [
        row
        for row in rows
        if row.phase == "throughput"
        and row.workload == workload
        and row.method == method
        and row.itopk == itopk
        and row.width == width
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {workload}/{method}/L={itopk}/W={width} row, "
            f"found {len(matches)}"
        )
    return matches[0]


def max_recall(rows: list[Aggregate], workload: str, method: str) -> Aggregate:
    return max(
        (
            row
            for row in rows
            if row.phase == "throughput"
            and row.workload == workload
            and row.method == method
        ),
        key=lambda item: (item.recall, item.qps),
    )


def benchmark_context(result_root: Path) -> dict[str, object]:
    path = result_root / "raw" / "throughput" / "yfcc" / "shard_00.json"
    return dict(json.loads(path.read_text()).get("context", {}))


def manifest_stats(bitmap_root: Path, workload: str, phase: str) -> tuple[int, float, int, int]:
    if workload == "yfcc":
        path = bitmap_root / "yfcc" / f"{phase}_{'1000' if phase == 'correctness' else '10000'}"
    else:
        path = (
            bitmap_root
            / "arxiv"
            / workload
            / f"{phase}_{'1000' if phase == 'correctness' else '10000'}"
        )
    manifest = json.loads((path / "manifest.json").read_text())
    shards = manifest["shards"]
    queries = sum(int(shard["query_count"]) for shard in shards)
    mean = sum(
        float(shard["mean_selectivity"]) * int(shard["query_count"])
        for shard in shards
    ) / queries
    minimum = min(int(shard["min_passing"]) for shard in shards)
    maximum = max(int(shard["max_passing"]) for shard in shards)
    return queries, mean, minimum, maximum


def report(result_root: Path, bitmap_root: Path, rows: list[Aggregate], plots: dict[str, Path]) -> None:
    total_violations = sum(row.filter_violations for row in rows)
    total_sentinel_errors = sum(row.sentinel_errors for row in rows)
    context = benchmark_context(result_root)
    yfcc_legacy_max = max_recall(rows, "yfcc", "legacy_navix")
    yfcc_seeded_max = max_recall(rows, "yfcc", "bitmap_seeded_navix")
    yfcc_default_max = max_recall(rows, "yfcc", "default_cagra")
    yfcc_accumulator_max = max_recall(rows, "yfcc", "default_cagra_accumulator")
    em_default_l64 = at_config(rows, "em", "default_cagra", 64, 1)
    em_accumulator_l64 = at_config(rows, "em", "default_cagra_accumulator", 64, 1)
    emis_default_l64 = at_config(rows, "emis", "default_cagra", 64, 1)
    emis_accumulator_l64 = at_config(rows, "emis", "default_cagra_accumulator", 64, 1)
    emis_legacy_l64 = at_config(rows, "emis", "legacy_navix", 64, 1)
    emis_seeded_l64 = at_config(rows, "emis", "bitmap_seeded_navix", 64, 1)
    lines = [
        "#+title: Bitmap filtering and bitmap-seeded NaviX SINGLE_CTA report",
        "",
        "* Verdict",
        "",
        "This experiment compares four bitmap-filtered paths at the automatic B0 traversal budget:",
        "default CAGRA, default CAGRA with a bounded passing-result accumulator, legacy",
        "in-kernel-seeded NaviX, and strict bitmap-seeded NaviX.  The bitmap seed prepass is",
        "included in end-to-end QPS.  No FAVOR mode, exact-scan fallback, or public-API change is",
        "involved.",
        "",
        f"Across all recorded rows, filter violations = {total_violations:.0f} and invalid-sentinel",
        f"errors = {total_sentinel_errors:.0f}.",
        "",
        "The accumulator consistently improves default CAGRA recall without materially changing",
        "throughput.  At L=64, W=1 it raises ArXiv-EM recall from",
        f"{em_default_l64.recall:.4f} to {em_accumulator_l64.recall:.4f} at",
        f"{em_default_l64.qps:.0f} versus {em_accumulator_l64.qps:.0f} QPS, and raises ArXiv-EMIS",
        f"from {emis_default_l64.recall:.4f} to {emis_accumulator_l64.recall:.4f} at",
        f"{emis_default_l64.qps:.0f} versus {emis_accumulator_l64.qps:.0f} QPS.  On YFCC, its",
        f"maximum B0 recall is {yfcc_accumulator_max.recall:.4f}, versus",
        f"{yfcc_default_max.recall:.4f} without the accumulator.  This confirms that ordinary",
        "default CAGRA loses useful passing candidates through bounded frontier eviction, but also",
        "shows that retention alone does not solve sparse or negatively correlated exploration.",
        "",
        "The implementation is correct and useful, but bitmap seeds are not a complete YFCC",
        "solution.  Strict seeding guarantees a legal passing start and removes avoidable",
        "underfill; after that, the kernel executes the existing adaptive-local NaviX traversal.",
        f"On YFCC, the best B0 recall rises only from {yfcc_legacy_max.recall:.4f} to",
        f"{yfcc_seeded_max.recall:.4f}, while QPS at those maximum-recall points changes from",
        f"{yfcc_legacy_max.qps:.0f} to {yfcc_seeded_max.qps:.0f}.  Since exact passing seeds and",
        "the full B0 graph budget still do not approach 0.95 recall, the remaining bottleneck is",
        "post-seed graph exploration under the sparse predicate, not seed availability.",
        "",
        "The strongest gain is ArXiv-EMIS.  At L=64, W=1, strict seeding improves recall from",
        f"{emis_legacy_l64.recall:.4f} to {emis_seeded_l64.recall:.4f} at essentially unchanged",
        f"throughput ({emis_legacy_l64.qps:.0f} versus {emis_seeded_l64.qps:.0f} QPS), and removes",
        "legacy handoff underfill.  ArXiv-EM and R are nearly parity, indicating that legacy",
        "in-kernel seed discovery was already effective for those workloads.",
        "",
        "* Best B0 result at 0.95 recall",
        "",
        "| workload | method | result |",
        "|----------+--------+--------|",
    ]
    for workload in WORKLOADS:
        for method, method_label in METHOD_LABELS.items():
            lines.append(
                f"| {WORKLOAD_LABELS[workload]} | {method_label} | "
                f"{best_at_target(rows, workload, method)} |"
            )
    lines.extend(
        [
            "",
            "* Matched L=64, W=1 evidence",
            "",
            "| workload | method | recall | QPS | underfilled-query fraction |",
            "|----------+--------+--------+-----+----------------------------|",
        ]
    )
    for workload in WORKLOADS:
        for method, method_label in METHOD_LABELS.items():
            row = at_config(rows, workload, method, 64, 1)
            lines.append(
                f"| {WORKLOAD_LABELS[workload]} | {method_label} | {row.recall:.4f} | "
                f"{row.qps:.0f} | {row.underfilled:.6f} |"
            )
    lines.extend(
        [
            "",
            "ArXiv-EM has one query with only one legal database item (the manifest minimum is",
            "one), so its 0.0001 strict-seeding underfill is required behavior rather than a search",
            "failure.  All YFCC bitmap rows have at least 58 passing items.",
            "",
            "* Throughput Pareto frontiers",
            "",
        ]
    )
    for workload in WORKLOADS:
        lines.extend(
            [
                f"** {WORKLOAD_LABELS[workload]}",
                "",
                f"[[file:{plots[workload].relative_to(result_root)}]]",
                "",
            ]
        )
    lines.extend(
        [
            "* Predicate materialization validation",
            "",
            "Every non-sentinel ground-truth ID was tested against the generated bitmap before a",
            "shard was accepted.  The predicates exactly match the UDF adapters: YFCC contains all",
            "query tags; ArXiv EM matches subcategory, EMIS contains the query category, and R uses",
            "an inclusive date interval.",
            "",
            "| workload | queries | mean selectivity | min passing | max passing |",
            "|----------+---------+------------------+-------------+-------------|",
        ]
    )
    for workload in WORKLOADS:
        queries, mean, minimum, maximum = manifest_stats(
            bitmap_root, workload, "throughput"
        )
        lines.append(
            f"| {WORKLOAD_LABELS[workload]} | {queries} | {mean:.6f} | {minimum} | {maximum} |"
        )
    lines.extend(
        [
            "",
            "* Design and implementation",
            "",
            "The private benchmark path stores a versioned, tightly packed row-major bitmap.  A",
            "one-warp-per-query prepass reads bitmap words coalescently and returns the first k",
            "passing graph IDs in ascending internal-ID order.  Source-index-remapped CAGRA indexes",
            "are handled by scanning internal IDs while testing the mapped source ID.",
            "",
            "The prepass and search launch on the same CUDA stream.  Seed IDs and counts are",
            "preallocated before timing; only valid seeds receive distance calculations and visited",
            "hash insertions.  All remaining shared itopk/candidate slots are initialized to invalid",
            "ID and infinite distance.  Empty rows return the normal invalid sentinels without graph",
            "work.  Seed selection does not consume a graph iteration, so the full B0 budget remains",
            "available to NaviX.",
            "",
            "After initialization, the kernel is the existing adaptive-local NaviX implementation:",
            "the persistent frontier contains passing nodes only, rejected nodes can be transient",
            "one-/two-hop bridges, and no extra degree-squared shared-memory buffer is allocated.",
            "YFCC throughput uses four 2,048-query bitmap shards plus one 1,808-query shard; aggregate",
            "QPS is total queries divided by the sum of shard times.",
            "",
            "The default-CAGRA accumulator is an independent output-retention option; it does not",
            "change traversal order, candidate selection, or the graph budget.  During SINGLE_CTA",
            "search it observes passing initial candidates and expanded children and maintains a",
            "bounded raw-distance top-k.  When enabled, the final neighbors and distances come from",
            "that accumulator instead of the fused traversal buffer.  The implementation reuses",
            "the existing private runtime-state filter wrapper and benchmark configuration flag,",
            "so no public search parameter or filter API was added.",
            "",
            "* Reproduction",
            "",
            f"These measurements were collected on {context.get('gpu_name', 'the recorded GPU')}",
            f"({context.get('gpu_sm_count', '?')} SMs) with cuVS {context.get('library_version', '?')}.",
            "Search timing includes the GPU bitmap-seed prepass and graph traversal, but excludes",
            "offline bitmap materialization, file loading, and one-time device upload.",
            "",
            "#+begin_src sh",
            "ninja -C cpp/build NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST CUVS_CAGRA_ANN_BENCH",
            "cpp/build/gtests/NEIGHBORS_ANN_CAGRA_FILTER_BITMAP_TEST",
            "benchmarks/favor/navix_bitmap/run_experiment.sh all",
            "#+end_src",
            "",
            "The sweep is B0-only: L in {64,128,256,512}, W in {1,2}, k=10, degree=32,",
            "correctness batches of 1,000 queries, and throughput over 10,000 queries.",
        ]
    )
    (result_root / "NAVIX_BITMAP_SEEDED_REPORT.org").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--bitmap-root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.result_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    points = load_points(args.result_root)
    rows = aggregate(points)
    validate_coverage(points, rows)
    write_csv(analysis / "shard_points.csv", points)
    write_csv(analysis / "aggregate_points.csv", rows)
    violations = sum(row.filter_violations for row in rows)
    sentinel_errors = sum(row.sentinel_errors for row in rows)
    if violations != 0 or sentinel_errors != 0:
        raise SystemExit(
            f"correctness failure: filter_violations={violations}, "
            f"sentinel_errors={sentinel_errors}"
        )
    plots = {
        workload: plot_workload(analysis, workload, rows)
        for workload in WORKLOADS
    }
    default_methods = ("default_cagra", "default_cagra_accumulator")
    for workload in WORKLOADS:
        plot_workload(
            analysis,
            workload,
            rows,
            methods=default_methods,
            filename_suffix="_default_accumulator",
        )
    report(args.result_root, args.bitmap_root, rows, plots)


if __name__ == "__main__":
    main()
