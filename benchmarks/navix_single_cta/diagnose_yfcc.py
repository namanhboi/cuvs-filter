#!/usr/bin/env python3
"""Diagnose low YFCC recall in the benchmark-only NaviX SINGLE_CTA path.

The GPU capture is authoritative for what the implementation did.  The optional CPU traversal is
an intentionally favorable counterfactual: it starts from the exact captured handoff seeds, uses
an unbounded best-first passing frontier, and selectively removes the per-parent cap/policy.  It is
therefore an attribution oracle, not a throughput model or a bit-exact CAGRA replay.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAPTURES = (
    "yfcc_navix_diag_b0_l64_w1",
    "yfcc_navix_diag_i1044_l64_w1",
    "yfcc_navix_diag_i1044_l512_w2",
)
LABELS = {
    CAPTURES[0]: "L64 W1 B0",
    CAPTURES[1]: "L64 W1 i1044",
    CAPTURES[2]: "L512 W2 i1044",
}
INVALID = np.uint32(0xFFFFFFFF)


def _load_scalar(stream: BinaryIO, name: str) -> int:
    value = np.load(stream, allow_pickle=False)
    if value.shape != ():
        raise ValueError(f"CAGRA {name} is not a scalar")
    return int(value)


def _read_array_header(stream: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NumPy array header version {version}")
    return shape, fortran, np.dtype(dtype), stream.tell()


def read_cagra_graph(path: Path) -> np.memmap:
    """Map the graph array from the graph-only cuVS serialization used by this benchmark."""
    with path.open("rb") as stream:
        stream.read(4)
        version = _load_scalar(stream, "version")
        rows = _load_scalar(stream, "size")
        _load_scalar(stream, "dimension")
        degree = _load_scalar(stream, "graph degree")
        _load_scalar(stream, "metric")
        shape, fortran, dtype, offset = _read_array_header(stream)
    if version != 5 or shape != (rows, degree) or fortran or dtype != np.dtype("<u4"):
        raise ValueError(
            f"unexpected CAGRA graph serialization: version={version}, shape={shape}, "
            f"fortran={fortran}, dtype={dtype}"
        )
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="C")


def read_bin(path: Path, dtype: np.dtype) -> np.memmap:
    with path.open("rb") as stream:
        rows, cols = struct.unpack("<II", stream.read(8))
    expected = 8 + rows * cols * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"invalid bin matrix size for {path}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, cols), order="C")


class Spmat:
    def __init__(self, path: Path):
        with path.open("rb") as stream:
            self.rows, self.cols, self.nnz = struct.unpack("<qqq", stream.read(24))
        expected = 24 + 8 * (self.rows + 1) + 8 * self.nnz
        if self.rows <= 0 or self.cols <= 0 or self.nnz < 0 or path.stat().st_size != expected:
            raise ValueError(f"invalid spmat: {path}")
        self.offsets = np.memmap(path, dtype="<i8", mode="r", offset=24, shape=(self.rows + 1,))
        self.columns = np.memmap(
            path,
            dtype="<i4",
            mode="r",
            offset=24 + 8 * (self.rows + 1),
            shape=(self.nnz,),
        )
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.nnz:
            raise ValueError(f"invalid spmat offsets: {path}")

    def row(self, row: int) -> np.ndarray:
        begin, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return self.columns[begin:end]


def read_csv(path: Path) -> list[dict[str, float | int]]:
    integer_fields = {
        "query_id",
        "iterations",
        "resolved_max_iterations",
        "stop_reason",
        "terminal_valid",
        "terminal_pass",
        "terminal_reject",
        "terminal_unexpanded_pass",
        "terminal_unexpanded_reject",
        "output_count",
        "navix_seed_found",
        "navix_seed_iteration",
        "navix_seed_count",
        "navix_post_seed_iterations",
        "navix_terminal_phase",
        "navix_one_hop_parents",
        "navix_directed_parents",
        "navix_blind_parents",
        "navix_first_hop_checks",
        "navix_first_hop_passing",
        "navix_bridge_rows",
        "navix_bridge_rows_loaded",
        "navix_bridge_rows_after_cap",
        "navix_second_hop_checks",
        "navix_second_hop_passing",
        "navix_admitted_candidates",
        "navix_cap_blocked_unique",
        "candidate_hash_full",
        "navix_gt_first_hop_mask",
        "navix_gt_second_hop_mask",
        "navix_gt_admitted_mask",
        "navix_gt_retained_mask",
        "navix_gt_cap_blocked_mask",
        "navix_gt_hash_full_mask",
        "navix_gt_output_mask",
    }
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, float | int] = {}
            for key, value in raw.items():
                row[key] = int(value) if key in integer_fields or key.startswith("navix_local_p_") else float(value)
            rows.append(row)
    return rows


def popcount(value: int) -> int:
    return int(value).bit_count()


@dataclass
class Capture:
    name: str
    directory: Path
    manifest: dict
    rows: list[dict[str, float | int]]
    seeds: np.ndarray
    seed_distances: np.ndarray
    outputs: np.ndarray


def load_capture(root: Path, name: str) -> Capture:
    directory = root / "diagnostics" / name
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_csv(directory / "query_summary.csv")
    nq, stride = int(manifest["num_queries"]), int(manifest["navix_seed_stride"])
    seeds = np.fromfile(directory / "navix_seed_ids.bin", dtype="<u4")
    distances = np.fromfile(directory / "navix_seed_distances.bin", dtype="<f4")
    outputs = np.fromfile(directory / "result_indices.i64bin", dtype="<i8")
    topk = int(manifest["topk"])
    if (
        seeds.size != nq * stride
        or distances.size != nq * stride
        or outputs.size != nq * topk
        or len(rows) != nq
    ):
        raise ValueError(f"inconsistent diagnostic capture in {directory}")
    return Capture(
        name,
        directory,
        manifest,
        rows,
        seeds.reshape(nq, stride),
        distances.reshape(nq, stride),
        outputs.reshape(nq, topk),
    )


def validate_capture(capture: Capture, result_root: Path) -> dict[str, float | int]:
    """Fail closed on schema, mask-stage, seed, and regular-search equivalence errors."""
    if int(capture.manifest["schema_version"]) != 6:
        raise ValueError(f"unexpected diagnostic schema for {capture.name}")
    graph_degree = int(capture.manifest["graph_degree"])
    seed_slots = 0
    for expected_query, row in enumerate(capture.rows):
        query_id = int(row["query_id"])
        if query_id != expected_query:
            raise ValueError(f"non-contiguous query IDs in {capture.name}")
        valid_seeds = [int(node) for node in capture.seeds[query_id] if node != INVALID]
        seed_slots += len(valid_seeds)
        if len(valid_seeds) != int(row["navix_seed_count"]):
            raise ValueError(f"seed-count mismatch for {capture.name} query {query_id}")
        if len(valid_seeds) != len(set(valid_seeds)):
            raise ValueError(f"duplicate captured seed for {capture.name} query {query_id}")
        seeded = int(row["navix_seed_found"]) != 0
        if seeded != bool(valid_seeds):
            raise ValueError(f"seed-state mismatch for {capture.name} query {query_id}")
        expected_phase = 2 if seeded else 1
        if int(row["navix_terminal_phase"]) != expected_phase:
            raise ValueError(f"terminal-phase mismatch for {capture.name} query {query_id}")

        policy_count = (
            int(row["navix_one_hop_parents"])
            + int(row["navix_directed_parents"])
            + int(row["navix_blind_parents"])
        )
        histogram_count = sum(int(row[f"navix_local_p_{p}"]) for p in range(33))
        if policy_count != histogram_count:
            raise ValueError(f"policy-histogram mismatch for {capture.name} query {query_id}")
        if int(row["navix_first_hop_checks"]) != policy_count * graph_degree:
            raise ValueError(f"first-hop parent accounting mismatch for {capture.name} query {query_id}")
        local_checks = int(row["navix_first_hop_checks"]) + int(row["navix_second_hop_checks"])
        if int(row["candidate_attempts"]) != local_checks or int(row["candidate_evaluations"]) != local_checks:
            raise ValueError(f"candidate accounting mismatch for {capture.name} query {query_id}")

        checked = int(row["navix_gt_first_hop_mask"]) | int(row["navix_gt_second_hop_mask"])
        admitted = int(row["navix_gt_admitted_mask"])
        retained = int(row["navix_gt_retained_mask"])
        output = int(row["navix_gt_output_mask"])
        if retained & ~admitted or output & ~retained:
            raise ValueError(f"non-monotone GT stage masks for {capture.name} query {query_id}")
        if int(row["navix_gt_cap_blocked_mask"]) & ~int(row["navix_gt_second_hop_mask"]):
            raise ValueError(f"cap-blocked GT was not checked for {capture.name} query {query_id}")
        if int(row["navix_gt_hash_full_mask"]) & ~checked:
            raise ValueError(f"hash-full GT was not checked for {capture.name} query {query_id}")
        valid_outputs = [
            int(node)
            for node in capture.outputs[query_id]
            if 0 <= int(node) < int(capture.manifest["dataset_size"])
        ]
        if len(valid_outputs) != int(row["output_count"]):
            raise ValueError(f"output-count mismatch for {capture.name} query {query_id}")
        if len(valid_outputs) != len(set(valid_outputs)):
            raise ValueError(f"duplicate output ID for {capture.name} query {query_id}")
        expected_recall = popcount(output) / 10.0
        if not math.isclose(float(row["recall"]), expected_recall, abs_tol=2e-6):
            raise ValueError(f"host recall/mask mismatch for {capture.name} query {query_id}")

    raw = json.loads((result_root / "raw" / f"{capture.name}.json").read_text())
    benchmark_rows = [row for row in raw["benchmarks"] if "Recall" in row]
    if len(benchmark_rows) != 1:
        raise ValueError(f"expected one regular benchmark row for {capture.name}")
    regular_recall = float(benchmark_rows[0]["Recall"])
    diagnostic_recall = float(np.mean([float(row["recall"]) for row in capture.rows]))
    regular_underfilled = float(benchmark_rows[0]["UnderfilledQueries"])
    diagnostic_underfilled = ratio(
        sum(int(row["output_count"]) < 10 for row in capture.rows), len(capture.rows)
    )
    if not math.isclose(regular_recall, diagnostic_recall, abs_tol=2e-4):
        raise ValueError(f"diagnostic/regular recall mismatch for {capture.name}")
    if not math.isclose(regular_underfilled, diagnostic_underfilled, abs_tol=2e-4):
        raise ValueError(f"diagnostic/regular underfill mismatch for {capture.name}")
    return {
        "queries": len(capture.rows),
        "seed_slots": seed_slots,
        "regular_recall": regular_recall,
        "diagnostic_recall": diagnostic_recall,
        "regular_underfilled": regular_underfilled,
        "diagnostic_underfilled": diagnostic_underfilled,
    }


def validate_predicates(captures: list[Capture], data_root: Path) -> tuple[int, int]:
    base_metadata = Spmat(data_root / "yfcc-10M/base.metadata.10M.spmat")
    query_metadata = Spmat(
        data_root / "yfcc-10M/workloads/correctness_1000/query.metadata.spmat"
    )
    checked_seeds = 0
    checked_outputs = 0
    for capture in captures:
        for query_id, row in enumerate(capture.rows):
            tags = query_metadata.row(query_id)
            for raw in capture.seeds[query_id]:
                if raw == INVALID:
                    continue
                node = int(raw)
                if node >= base_metadata.rows:
                    raise ValueError(f"out-of-range seed in {capture.name} query {query_id}")
                columns = base_metadata.row(node)
                if any(not np.any(columns == tag) for tag in tags):
                    raise ValueError(f"predicate-invalid seed in {capture.name} query {query_id}")
                checked_seeds += 1
            for raw in capture.outputs[query_id]:
                node = int(raw)
                if node < 0 or node >= base_metadata.rows:
                    continue
                columns = base_metadata.row(node)
                if any(not np.any(columns == tag) for tag in tags):
                    raise ValueError(f"predicate-invalid output in {capture.name} query {query_id}")
                checked_outputs += 1
            if bool(int(row["navix_seed_found"])) != (int(row["navix_seed_count"]) != 0):
                raise ValueError(f"seed summary mismatch in {capture.name} query {query_id}")
    return checked_seeds, checked_outputs


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(capture: Capture) -> dict[str, float | int | dict[str, int]]:
    rows = capture.rows
    seeded = [row for row in rows if int(row["navix_seed_found"]) != 0]
    first_checks = sum(int(row["navix_first_hop_checks"]) for row in rows)
    second_checks = sum(int(row["navix_second_hop_checks"]) for row in rows)
    bridge_loaded = sum(int(row["navix_bridge_rows_loaded"]) for row in rows)
    policies = {
        "one_hop": sum(int(row["navix_one_hop_parents"]) for row in rows),
        "directed": sum(int(row["navix_directed_parents"]) for row in rows),
        "blind": sum(int(row["navix_blind_parents"]) for row in rows),
    }
    gt_fields = (
        "navix_gt_first_hop_mask",
        "navix_gt_second_hop_mask",
        "navix_gt_admitted_mask",
        "navix_gt_retained_mask",
        "navix_gt_output_mask",
    )
    result: dict[str, float | int | dict[str, int]] = {
        "queries": len(rows),
        "mean_recall": float(np.mean([float(row["recall"]) for row in rows])),
        "seedless_fraction": ratio(len(rows) - len(seeded), len(rows)),
        "underfilled_fraction": ratio(sum(int(row["output_count"]) < 10 for row in rows), len(rows)),
        "seeded_underfilled_fraction": ratio(
            sum(int(row["output_count"]) < 10 for row in seeded), len(rows)
        ),
        "mean_seed_iteration_seeded": float(
            np.mean([int(row["navix_seed_iteration"]) for row in seeded]) if seeded else math.nan
        ),
        "p95_seed_iteration_seeded": float(
            np.percentile([int(row["navix_seed_iteration"]) for row in seeded], 95)
            if seeded
            else math.nan
        ),
        "frontier_exhausted_fraction": ratio(
            sum(int(row["stop_reason"]) == 3 for row in rows), len(rows)
        ),
        "max_cap_fraction": ratio(
            sum(int(row["stop_reason"]) in (1, 2) for row in rows), len(rows)
        ),
        "first_hop_yield": ratio(
            sum(int(row["navix_first_hop_passing"]) for row in rows), first_checks
        ),
        "second_hop_yield": ratio(
            sum(int(row["navix_second_hop_passing"]) for row in rows), second_checks
        ),
        "bridge_rows_after_cap_fraction": ratio(
            sum(int(row["navix_bridge_rows_after_cap"]) for row in rows), bridge_loaded
        ),
        "queries_with_cap_blocking_fraction": ratio(
            sum(int(row["navix_cap_blocked_unique"]) != 0 for row in rows), len(rows)
        ),
        "queries_with_hash_full_fraction": ratio(
            sum(int(row["candidate_hash_full"]) != 0 for row in rows), len(rows)
        ),
        "policies": policies,
    }
    for field in gt_fields:
        result[f"mean_{field.removesuffix('_mask')}_recall"] = float(
            np.mean([popcount(int(row[field])) / 10.0 for row in rows])
        )
    return result


class QueryOracle:
    def __init__(
        self,
        graph: np.memmap,
        base: np.memmap,
        query: np.ndarray,
        base_metadata: Spmat,
        required_tags: np.ndarray,
        ground_truth: np.ndarray,
    ):
        self.graph = graph
        self.base = base
        self.query = query.astype(np.int16)
        self.base_metadata = base_metadata
        self.required_tags = np.asarray(required_tags, dtype=np.int32)
        self.ground_truth = {int(node) for node in ground_truth}
        self.pass_cache: dict[int, bool] = {}
        self.distance_cache: dict[int, float] = {}

    def passes(self, node: int) -> bool:
        cached = self.pass_cache.get(node)
        if cached is not None:
            return cached
        columns = self.base_metadata.row(node)
        # YFCC CSR rows are not sorted. Match the benchmark UDF's order-independent containment
        # semantics exactly; query arity is only one or two, so each NumPy scan stays small.
        for tag in self.required_tags:
            if not np.any(columns == tag):
                self.pass_cache[node] = False
                return False
        self.pass_cache[node] = True
        return True

    def distance(self, node: int) -> float:
        cached = self.distance_cache.get(node)
        if cached is not None:
            return cached
        delta = self.base[node].astype(np.int16) - self.query
        value = float(np.dot(delta.astype(np.int32), delta.astype(np.int32)))
        self.distance_cache[node] = value
        return value

    def local_candidates(
        self,
        parent: int,
        visited: set[int],
        *,
        remove_cap: bool,
        force_two_hop: bool,
    ) -> list[int]:
        row = [int(node) for node in self.graph[parent] if node != INVALID]
        passing = [self.passes(node) for node in row]
        p = sum(passing)
        if force_two_hop:
            policy = "blind"
        else:
            policy = "one" if p >= 13 else ("blind" if p <= 4 else "directed")
        order = list(range(len(row)))
        if policy == "directed":
            order.sort(key=lambda i: (self.distance(row[i]), i))

        output: list[int] = []
        bridges: list[int] = []
        for i in order:
            node = row[i]
            if passing[i]:
                if node not in visited:
                    visited.add(node)
                    output.append(node)
            elif policy != "one" and node not in visited:
                visited.add(node)
                bridges.append(node)

        if policy == "one":
            return output
        cap = None if remove_cap else int(self.graph.shape[1])
        for bridge in bridges:
            for raw in self.graph[bridge]:
                grandchild = int(raw)
                if raw == INVALID or not self.passes(grandchild) or grandchild in visited:
                    continue
                if cap is not None and len(output) >= cap:
                    continue
                visited.add(grandchild)
                output.append(grandchild)
        return output

    def traverse(
        self,
        seed_ids: Iterable[int],
        *,
        max_expanded: int,
        remove_cap: bool,
        force_two_hop: bool,
    ) -> dict[str, float | int]:
        seeds = list(dict.fromkeys(int(node) for node in seed_ids if int(node) != int(INVALID)))
        if any(not self.passes(node) for node in seeds):
            raise ValueError("captured NaviX seed does not satisfy the query predicate")
        visited = set(seeds)
        passing_seen = set(seeds)
        frontier = [(self.distance(node), node) for node in seeds]
        heapq.heapify(frontier)
        expanded: set[int] = set()
        while frontier and len(expanded) < max_expanded:
            _, parent = heapq.heappop(frontier)
            if parent in expanded:
                continue
            expanded.add(parent)
            for child in self.local_candidates(
                parent, visited, remove_cap=remove_cap, force_two_hop=force_two_hop
            ):
                passing_seen.add(child)
                heapq.heappush(frontier, (self.distance(child), child))
        nearest = heapq.nsmallest(10, passing_seen, key=self.distance)
        return {
            "recall": len(self.ground_truth.intersection(nearest)) / 10.0,
            "gt_seen_recall": len(self.ground_truth.intersection(passing_seen)) / 10.0,
            "passing_seen": len(passing_seen),
            "expanded": len(expanded),
            "frontier_remaining": len(frontier),
            "predicate_checks": len(self.pass_cache),
            "distance_evaluations": len(self.distance_cache),
        }


def run_oracle(
    capture: Capture,
    data_root: Path,
    sample_size: int,
    max_expanded: int,
) -> list[dict[str, float | int | str]]:
    if sample_size == 0:
        return []
    graph = read_cagra_graph(data_root / "yfcc-10M/cagra_g32_ig64.index")
    base = read_bin(data_root / "yfcc-10M/base.10M.u8bin", np.dtype("u1"))
    queries = read_bin(
        data_root / "yfcc-10M/workloads/correctness_1000/query.u8bin", np.dtype("u1")
    )
    gt = read_bin(
        data_root / "yfcc-10M/workloads/correctness_1000/groundtruth.ibin", np.dtype("<u4")
    )
    base_metadata = Spmat(data_root / "yfcc-10M/base.metadata.10M.spmat")
    query_metadata = Spmat(
        data_root / "yfcc-10M/workloads/correctness_1000/query.metadata.spmat"
    )
    hard = [
        row
        for row in capture.rows
        if int(row["navix_seed_found"]) != 0 and float(row["recall"]) < 0.9
    ]
    # Sample three distinct failure surfaces. Looking only at the absolute lowest-recall rows
    # over-selects one-seed dead ends and cannot tell us whether cap removal helps otherwise-filled
    # queries. The groups remain deterministic and mutually exclusive.
    group_size = max(1, sample_size // 3)
    chosen: dict[int, tuple[dict[str, float | int], str]] = {}

    def choose(candidates: Iterable[dict[str, float | int]], group: str, limit: int) -> None:
        for row in sorted(
            candidates,
            key=lambda item: (
                float(item["recall"]),
                -int(item["navix_seed_iteration"]),
                int(item["query_id"]),
            ),
        ):
            query_id = int(row["query_id"])
            if query_id not in chosen:
                chosen[query_id] = (row, group)
                limit -= 1
                if limit == 0:
                    return

    choose((row for row in hard if int(row["output_count"]) < 10), "underfilled", group_size)
    choose(
        (row for row in hard if int(row["navix_gt_cap_blocked_mask"]) != 0),
        "gt_cap_blocked",
        group_size,
    )
    choose((row for row in hard if int(row["output_count"]) >= 10), "filled_low_recall", sample_size)
    choose(hard, "remaining_hard", sample_size)
    selected = list(chosen.values())[:sample_size]
    variants = (
        ("adaptive_cap_unbounded_frontier", False, False),
        ("adaptive_uncapped_unbounded_frontier", True, False),
        ("always_two_hop_uncapped_unbounded_frontier", True, True),
    )
    results: list[dict[str, float | int | str]] = []
    for position, (row, sample_group) in enumerate(selected, start=1):
        query_id = int(row["query_id"])
        oracle = QueryOracle(
            graph,
            base,
            queries[query_id],
            base_metadata,
            query_metadata.row(query_id),
            gt[query_id],
        )
        seed_ids = capture.seeds[query_id]
        for variant, remove_cap, force_two_hop in variants:
            outcome = oracle.traverse(
                seed_ids,
                max_expanded=max_expanded,
                remove_cap=remove_cap,
                force_two_hop=force_two_hop,
            )
            results.append(
                {
                    "sample_position": position,
                    "sample_group": sample_group,
                    "query_id": query_id,
                    "gpu_recall": float(row["recall"]),
                    "gpu_output_count": int(row["output_count"]),
                    "seed_iteration": int(row["navix_seed_iteration"]),
                    "variant": variant,
                    **outcome,
                }
            )
        print(f"oracle {position}/{len(selected)}: query {query_id}", flush=True)
    return results


def plot_failure_breakdown(captures: list[Capture], output: Path) -> None:
    categories = ("No passing seed", "Seeded, underfilled", "Filled, recall < 0.9", "Recall >= 0.9")
    colors = ("#d73027", "#fc8d59", "#fee08b", "#1a9850")
    values: list[list[float]] = []
    for capture in captures:
        rows = capture.rows
        counts = [
            sum(int(row["navix_seed_found"]) == 0 for row in rows),
            sum(int(row["navix_seed_found"]) != 0 and int(row["output_count"]) < 10 for row in rows),
            sum(
                int(row["navix_seed_found"]) != 0
                and int(row["output_count"]) >= 10
                and float(row["recall"]) < 0.9
                for row in rows
            ),
            sum(
                int(row["navix_seed_found"]) != 0
                and int(row["output_count"]) >= 10
                and float(row["recall"]) >= 0.9
                for row in rows
            ),
        ]
        values.append([count / len(rows) for count in counts])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bottom = np.zeros(len(captures))
    x = np.arange(len(captures))
    for i, category in enumerate(categories):
        height = np.asarray([row[i] for row in values])
        ax.bar(x, height, bottom=bottom, label=category, color=colors[i])
        bottom += height
    ax.set_xticks(x, [LABELS[capture.name] for capture in captures])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of queries")
    ax.set_title("YFCC NaviX query outcomes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_gt_stages(captures: list[Capture], output: Path) -> None:
    fields = (
        ("Checked at hop 1/2", None),
        ("Admitted", "navix_gt_admitted_mask"),
        ("Ever in fused frontier", "navix_gt_retained_mask"),
        ("Returned", "navix_gt_output_mask"),
    )
    x = np.arange(len(fields))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for capture_index, capture in enumerate(captures):
        means = []
        for _, field in fields:
            if field is None:
                values = [
                    popcount(
                        int(row["navix_gt_first_hop_mask"])
                        | int(row["navix_gt_second_hop_mask"])
                    )
                    / 10.0
                    for row in capture.rows
                ]
            else:
                values = [popcount(int(row[field])) / 10.0 for row in capture.rows]
            means.append(float(np.mean(values)))
        ax.bar(x + (capture_index - 1) * width, means, width, label=LABELS[capture.name])
    ax.set_xticks(x, [name for name, _ in fields])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean exact-GT fraction")
    ax.set_title("Where exact YFCC neighbors leave the NaviX pipeline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_local_yield(capture: Capture, output: Path) -> None:
    counts = np.asarray(
        [sum(int(row[f"navix_local_p_{p}"]) for row in capture.rows) for p in range(33)],
        dtype=np.float64,
    )
    fractions = counts / counts.sum() if counts.sum() else counts
    colors = ["#d73027" if p <= 4 else "#4575b4" if p <= 12 else "#1a9850" for p in range(33)]
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(np.arange(33), fractions, color=colors, width=0.9)
    ax.axvline(4.5, color="black", linestyle="--", linewidth=1)
    ax.axvline(12.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-0.6, 32.6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Passing first-hop neighbors P out of D=32")
    ax.set_ylabel("Fraction of expanded parents")
    ax.set_title(f"Adaptive-local decisions: {LABELS[capture.name]}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_oracle(
    results: list[dict[str, float | int | str]], output: Path, max_expanded: int
) -> None:
    if not results:
        return
    labels = {
        "gpu": "GPU current",
        "adaptive_cap_unbounded_frontier": "Current local rule\n+unbounded frontier",
        "adaptive_uncapped_unbounded_frontier": "Remove D cap",
        "always_two_hop_uncapped_unbounded_frontier": "Always 2-hop\nremove D cap",
    }
    variants = list(labels)[1:]
    groups = list(dict.fromkeys(str(row["sample_group"]) for row in results))
    x = np.arange(len(groups))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    for series_index, key in enumerate(labels):
        means = []
        for group in groups:
            group_rows = [row for row in results if row["sample_group"] == group]
            if key == "gpu":
                by_query = {
                    int(row["query_id"]): float(row["gpu_recall"]) for row in group_rows
                }
                means.append(float(np.mean(list(by_query.values()))))
            else:
                means.append(
                    float(
                        np.mean(
                            [float(row["recall"]) for row in group_rows if row["variant"] == key]
                        )
                    )
                )
        ax.bar(
            x + (series_index - 1.5) * width,
            means,
            width,
            label=labels[key],
            color=("#555555", "#4575b4", "#fdae61", "#1a9850")[series_index],
        )
    ax.set_xticks(x, [group.replace("_", "\n") for group in groups])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean recall on hard seeded sample")
    ax.set_title(
        f"CPU counterfactual (up to {max_expanded} parent expansions; not a timing replay)"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    captures: list[Capture],
    summaries: dict[str, dict],
    oracle_results: list[dict[str, float | int | str]],
    oracle_max_expanded: int,
    validation: dict[str, dict[str, float | int]],
    predicate_checked_seeds: int,
    predicate_checked_outputs: int,
) -> None:
    b0 = summaries[CAPTURES[0]]
    deep = summaries[CAPTURES[-1]]
    deep_capture = captures[-1]
    checked_gt = float(
        np.mean(
            [
                popcount(
                    int(row["navix_gt_first_hop_mask"])
                    | int(row["navix_gt_second_hop_mask"])
                )
                / 10.0
                for row in deep_capture.rows
            ]
        )
    )
    admitted_gt = float(deep["mean_navix_gt_admitted_recall"])
    retained_gt = float(deep["mean_navix_gt_retained_recall"])
    output_gt = float(deep["mean_navix_gt_output_recall"])
    local_histogram = [
        sum(int(row[f"navix_local_p_{p}"]) for row in deep_capture.rows) for p in range(33)
    ]
    local_total = sum(local_histogram)
    p_zero_fraction = ratio(local_histogram[0], local_total)
    blind_fraction = ratio(sum(local_histogram[:5]), local_total)
    hard_gt_cap_blocked = sum(
        int(row["navix_seed_found"]) != 0
        and float(row["recall"]) < 0.9
        and int(row["navix_gt_cap_blocked_mask"]) != 0
        for row in deep_capture.rows
    )
    lines = [
        "#+TITLE: YFCC NaviX SINGLE_CTA root-cause diagnostic",
        "#+OPTIONS: toc:2 num:nil",
        "",
        "* Scope",
        "",
        "This report diagnoses the current GPU variant: the benchmark-only adaptive-local NaviX "
        "implementation fused into CAGRA SINGLE_CTA. Instrumented captures are deliberately "
        "untimed and do not change the public cuVS API or the production/default CAGRA path.",
        "",
        "The adaptive thresholds match the released Kuzu degree-32 policy (blind at P<=4, "
        "directed at 5<=P<=12, one hop at P>=13), but this is not claimed to be an exact Kuzu "
        "traversal: it retains CAGRA's fused bounded frontier, CAGRA seeding and termination, and "
        "a D=32 passing-candidate cap per parent.",
        "",
        "* Verdict",
        "",
        "The dominant YFCC failure is local passing-only reachability on the CAGRA graph, not "
        "insufficient max_iterations, fused-frontier retention, the D cap, or the visited hash. "
        "After a seed is found, many query predicates lead to a tiny passing component whose "
        "one-/two-hop closure contains neither ten results nor the exact neighbors.",
        "",
        f"- The widest deep run ends {deep['frontier_exhausted_fraction']:.1%} of queries by "
        f"frontier exhaustion, while only {deep['seedless_fraction']:.1%} remain in seed discovery. "
        f"It still underfills {deep['underfilled_fraction']:.1%} of queries, including "
        f"{deep['seeded_underfilled_fraction']:.1%} that did find a seed.",
        f"- NaviX checks only {checked_gt:.4f} of exact GT. Once checked, retention is nearly "
        f"lossless: {admitted_gt:.4f} is admitted, {retained_gt:.4f} ever reaches the fused "
        f"frontier, and {output_gt:.4f} is returned. Thus {1.0 - checked_gt:.1%} of GT is never "
        "locally exposed at all.",
        f"- The current rule is already in its maximal blind-two-hop region for "
        f"{blind_fraction:.1%} of parents, and P=0 alone accounts for {p_zero_fraction:.1%}. "
        "Changing the adaptive thresholds therefore has little room to repair the sparse cases.",
        f"- Hash-full occurs for {deep['queries_with_hash_full_fraction']:.1%} of queries. The "
        f"checked-to-admitted GT loss is only {checked_gt - admitted_gt:.4f}, so the per-parent D "
        f"cap is visible but not the aggregate bottleneck. Only {hard_gt_cap_blocked} of "
        f"{len(deep_capture.rows)} queries are simultaneously seeded, below 0.9 recall, and "
        "observed with an exact-GT candidate beyond that cap.",
        f"- Increasing from L64/W1 B0 to L512/W2/i1044 reduces seedless queries from "
        f"{b0['seedless_fraction']:.1%} to {deep['seedless_fraction']:.1%}, but recall moves only "
        f"from {b0['mean_recall']:.4f} to {deep['mean_recall']:.4f} and underfill only from "
        f"{b0['underfilled_fraction']:.1%} to {deep['underfilled_fraction']:.1%}.",
        "",
        "* GPU results",
        "",
        "| configuration | recall | no seed | underfilled | seeded underfilled | frontier exhausted | hard cap | first-hop yield | second-hop yield | cap-blocked queries | hash-full queries |",
        "|-",
    ]
    for capture in captures:
        s = summaries[capture.name]
        lines.append(
            f"| {LABELS[capture.name]} | {s['mean_recall']:.4f} | {s['seedless_fraction']:.3f} | "
            f"{s['underfilled_fraction']:.3f} | {s['seeded_underfilled_fraction']:.3f} | "
            f"{s['frontier_exhausted_fraction']:.3f} | {s['max_cap_fraction']:.3f} | "
            f"{s['first_hop_yield']:.5f} | {s['second_hop_yield']:.5f} | "
            f"{s['queries_with_cap_blocking_fraction']:.3f} | "
            f"{s['queries_with_hash_full_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "#+CAPTION: Query-level failure decomposition.",
            "[[file:plots/yfcc_navix_failure_breakdown.png]]",
            "",
            "#+CAPTION: Cumulative exact-ground-truth visibility through the traversal pipeline.",
            "[[file:plots/yfcc_navix_gt_stage_waterfall.png]]",
            "",
            "#+CAPTION: Local-yield distribution and adaptive policy regions for the widest deep run.",
            "[[file:plots/yfcc_navix_local_yield.png]]",
            "",
            "* Counterfactual oracle",
            "",
            "The CPU oracle starts each query from the exact passing seed batch saved by the GPU. "
            "It uses an unbounded raw-distance best-first passing frontier. The first variant keeps "
            "the current adaptive local policy and D cap; the second removes only the D cap; the "
            "third also forces every parent to use two hops. These are favorable attribution "
            f"controls, not QPS measurements and not a bit-exact replay of CAGRA's merge order. "
            f"Each variant is allowed up to {oracle_max_expanded} parent expansions.",
            "",
        ]
    )
    if oracle_results:
        variants = list(dict.fromkeys(str(row["variant"]) for row in oracle_results))
        groups = list(dict.fromkeys(str(row["sample_group"]) for row in oracle_results))
        lines.extend(["| sample group | variant | recall | GT seen | mean expanded |", "|-"])
        for group in groups:
            for variant in variants:
                rows = [
                    row
                    for row in oracle_results
                    if row["variant"] == variant and row["sample_group"] == group
                ]
                lines.append(
                    f"| {group} | {variant} | "
                    f"{np.mean([float(row['recall']) for row in rows]):.4f} | "
                    f"{np.mean([float(row['gt_seen_recall']) for row in rows]):.4f} | "
                    f"{np.mean([int(row['expanded']) for row in rows]):.1f} |"
                )
        lines.extend(
            [
                "",
                "#+CAPTION: Counterfactual recall on stratified hard seeded queries from the widest deep run.",
                "[[file:plots/yfcc_navix_counterfactual.png]]",
            ]
        )
    else:
        lines.append("The CPU oracle was disabled for this invocation.")
    lines.extend(
        [
            "",
            "* Interpretation",
            "",
            "The controls support the GPU-stage attribution:",
            "",
            "- Queries ending in seed discovery isolate a seed-availability failure; no NaviX "
            "policy ran for them.",
            "- A max-cap stop with an unexpanded passing frontier is a budget failure. Frontier "
            "exhaustion after seeding is instead a passing-connectivity/local-expansion failure.",
            "- GT checked at hop 1/2 but not admitted attributes loss to the per-parent cap, visited "
            "suppression, or hash saturation; the explicit cap/hash masks disambiguate them.",
            "- GT admitted but never retained attributes loss to the fused L-sized CAGRA frontier. "
            "GT retained but not returned attributes final bounded selection/order.",
            "- If the unbounded-frontier oracle recovers recall while retaining the current local "
            "rule, queue retention is implicated. Gains only after removing D implicate the "
            "per-parent cap. Gains only after forced two-hop implicate the adaptive threshold.",
            "- In the observed underfilled group, all three controls exhaust after only a few "
            "passing parents and are identical. In the filled-but-wrong group, even the "
            "always-two-hop uncapped control exposes almost no GT within the expanded budget. "
            "The targeted cap-blocked group can benefit from a larger frontier/deeper work, but "
            "those queries are rare and already high recall; that is not the aggregate YFCC gap.",
            "",
            "The design implication is that increasing L, max_iterations, or the D cap cannot by "
            "itself solve YFCC. A robust method must cross more than one rejected bridge hop, "
            "obtain predicate-aware seeds, or invoke a fallback that can inspect nodes outside the "
            "local passing component. Under an opaque scalar UDF, that fallback has inherently "
            "less predictable work than a predicate index.",
            "",
            "* Validation",
            "",
            f"- All {sum(int(row['queries']) for row in validation.values())} query summaries "
            "passed schema, phase, policy-histogram, candidate-accounting, GT-mask, and output "
            "recall invariants.",
            f"- All {predicate_checked_seeds} captured passing seed slots were reevaluated against "
            f"the CPU order-independent YFCC contains-all predicate, as were all "
            f"{predicate_checked_outputs} valid output slots; there were zero violations or "
            "duplicate output IDs.",
            "- Each diagnostic capture's aggregate recall and underfilled fraction agrees with its "
            "ordinary non-instrumented benchmark invocation within 0.0002.",
            "- The dedicated instrumentation is untimed; its QPS is intentionally invalid. Only "
            "the ordinary benchmark row is used for the equivalence check.",
            "",
            "* Reproduction",
            "",
            "#+begin_src sh",
            "ninja -C cpp/build -j8 cuvs CUVS_CAGRA_ANN_BENCH",
            "NAVIX_RESULT_ROOT=benchmarks/navix_single_cta/results_yfcc_diagnosis_$(date +%Y%m%d_%H%M%S) \\",
            "  benchmarks/navix_single_cta/run_experiment.sh diagnosis",
            "python benchmarks/navix_single_cta/diagnose_yfcc.py \\",
            f"  --result-root <result-root> --data-root datasets --oracle-sample 18 \\",
            f"  --oracle-max-expanded {oracle_max_expanded}",
            "#+end_src",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--oracle-sample", type=int, default=18)
    parser.add_argument("--oracle-max-expanded", type=int, default=2048)
    args = parser.parse_args()
    if args.oracle_sample < 0 or args.oracle_max_expanded <= 0:
        parser.error("oracle limits must be non-negative and positive, respectively")

    captures = [load_capture(args.result_root, name) for name in CAPTURES]
    validation = {
        capture.name: validate_capture(capture, args.result_root) for capture in captures
    }
    predicate_checked_seeds, predicate_checked_outputs = validate_predicates(
        captures, args.data_root
    )
    summaries = {capture.name: summarize(capture) for capture in captures}
    oracle_results = run_oracle(
        captures[-1], args.data_root, args.oracle_sample, args.oracle_max_expanded
    )

    plot_dir = args.result_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_failure_breakdown(captures, plot_dir / "yfcc_navix_failure_breakdown.png")
    plot_gt_stages(captures, plot_dir / "yfcc_navix_gt_stage_waterfall.png")
    plot_local_yield(captures[-1], plot_dir / "yfcc_navix_local_yield.png")
    plot_oracle(
        oracle_results, plot_dir / "yfcc_navix_counterfactual.png", args.oracle_max_expanded
    )

    (args.result_root / "diagnostic_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n"
    )
    (args.result_root / "validation.json").write_text(
        json.dumps(
            {
                "captures": validation,
                "predicate_checked_seed_slots": predicate_checked_seeds,
                "predicate_checked_output_slots": predicate_checked_outputs,
                "status": "pass",
            },
            indent=2,
        )
        + "\n"
    )
    with (args.result_root / "counterfactual_oracle.csv").open("w", newline="") as stream:
        if oracle_results:
            writer = csv.DictWriter(stream, fieldnames=list(oracle_results[0]))
            writer.writeheader()
            writer.writerows(oracle_results)
    write_report(
        args.result_root / "YFCC_NAVIX_ROOT_CAUSE_REPORT.org",
        captures,
        summaries,
        oracle_results,
        args.oracle_max_expanded,
        validation,
        predicate_checked_seeds,
        predicate_checked_outputs,
    )


if __name__ == "__main__":
    main()
