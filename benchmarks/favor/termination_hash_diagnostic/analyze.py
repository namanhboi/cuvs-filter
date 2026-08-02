#!/usr/bin/env python3
"""Evaluate global FAVOR termination policies from exact/forgetful shadow trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from generate_configs import DATASETS, HASHES


CHECKPOINT_DTYPE = np.dtype(
    [
        ("query_id", "<u4"),
        ("checkpoint", "<u4"),
        ("iteration", "<u4"),
        ("expanded_parents", "<u4"),
        ("prefix_valid", "<u4"),
        ("prefix_pass", "<u4"),
        ("passing_count", "<u4"),
        ("output_count", "<u4"),
        ("frontier_best", "<f4"),
        ("prefix_boundary", "<f4"),
        ("kth_passing_raw_distance", "<f4"),
        ("top_ids", "<u4", (10,)),
        ("top_distances", "<f4", (10,)),
    ]
)
assert CHECKPOINT_DTYPE.itemsize == 124


@dataclass(frozen=True)
class Policy:
    name: str
    fires: Callable[[np.void, int], bool]


def policies() -> list[Policy]:
    result = [
        Policy(
            "current_prefix16_of32",
            lambda record, _: record["prefix_valid"] == 32
            and record["prefix_pass"] >= 16
            and record["frontier_best"] > record["prefix_boundary"],
        )
    ]
    for passing in (10, 12, 14):
        result.append(
            Policy(
                f"prefix{passing}_of32",
                lambda record, _, passing=passing: record["prefix_valid"] == 32
                and record["prefix_pass"] >= passing
                and record["frontier_best"] > record["prefix_boundary"],
            )
        )
    for stable in (1, 2, 4, 8):
        result.append(
            Policy(
                f"top10_stable{stable}",
                lambda record, streak, stable=stable: record["output_count"] == 10
                and streak >= stable,
            )
        )
        for gap in (1.0, 1.05, 1.10):
            result.append(
                Policy(
                    f"gap{gap:.2f}_stable{stable}",
                    lambda record, streak, gap=gap, stable=stable: record["output_count"]
                    == 10
                    and streak >= stable
                    and record["frontier_best"]
                    > gap * record["kth_passing_raw_distance"],
                )
            )
    return result


def load_ibin(path: Path, rows: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if len(header) != 2 or header[0] < rows or header[1] < 10:
            raise ValueError(f"invalid nq x k ground truth: {path}")
        values = np.fromfile(stream, dtype="<i4", count=int(header[0] * header[1]))
    return values.reshape(int(header[0]), int(header[1]))[:rows, :10].astype(np.uint32)


def load_capture(directory: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[dict[str, str]]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    num_queries = int(manifest["num_queries"])
    stride = int(manifest["termination_checkpoint_stride"])
    if int(manifest["termination_checkpoint_record_size"]) != CHECKPOINT_DTYPE.itemsize:
        raise ValueError(f"checkpoint ABI mismatch: {directory}")
    checkpoints = np.fromfile(directory / "termination_checkpoints.bin", dtype=CHECKPOINT_DTYPE)
    if checkpoints.size != num_queries * stride:
        raise ValueError(f"checkpoint extent mismatch: {directory}")
    checkpoints = checkpoints.reshape(num_queries, stride)
    counts = np.fromfile(directory / "termination_checkpoint_counts.bin", dtype="<u4")
    if counts.size != num_queries or np.any(counts == 0) or np.any(counts > stride):
        raise ValueError(f"invalid checkpoint counts: {directory}")
    with (directory / "query_summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    if len(summaries) != num_queries:
        raise ValueError(f"summary extent mismatch: {directory}")
    return manifest, checkpoints, counts, summaries


def recall(ids: np.ndarray, truth: np.ndarray) -> float:
    return float(len(set(map(int, ids)) & set(map(int, truth))) / 10.0)


def accumulated_ids(records: np.ndarray) -> np.ndarray:
    best: dict[int, float] = {}
    for record in records:
        for node, distance in zip(record["top_ids"], record["top_distances"]):
            node_int = int(node)
            if node_int == 0xFFFFFFFF:
                continue
            best[node_int] = min(best.get(node_int, math.inf), float(distance))
    ordered = sorted(best, key=lambda node: (best[node], node))[:10]
    return np.asarray(ordered, dtype=np.uint32)


def stable_streaks(checkpoints: np.ndarray, valid_slots: np.ndarray) -> np.ndarray:
    complete = checkpoints["output_count"] == 10
    result = np.zeros(complete.shape, dtype=np.uint16)
    result[:, 0] = complete[:, 0] & valid_slots[:, 0]
    for slot in range(1, checkpoints.shape[1]):
        same = np.all(
            checkpoints["top_ids"][:, slot] == checkpoints["top_ids"][:, slot - 1], axis=1
        )
        result[:, slot] = np.where(
            complete[:, slot] & valid_slots[:, slot],
            np.where(same, result[:, slot - 1] + 1, 1),
            0,
        )
    return result


def policy_mask(
    name: str, checkpoints: np.ndarray, valid_slots: np.ndarray, streaks: np.ndarray
) -> np.ndarray:
    if name == "current_prefix16_of32":
        condition = checkpoints["prefix_pass"] >= 16
        condition &= checkpoints["prefix_valid"] == 32
        condition &= checkpoints["frontier_best"] > checkpoints["prefix_boundary"]
    elif name.startswith("prefix"):
        passing = int(name.removeprefix("prefix").removesuffix("_of32"))
        condition = checkpoints["prefix_pass"] >= passing
        condition &= checkpoints["prefix_valid"] == 32
        condition &= checkpoints["frontier_best"] > checkpoints["prefix_boundary"]
    else:
        stable = int(name.rsplit("stable", maxsplit=1)[1])
        condition = checkpoints["output_count"] == 10
        condition &= streaks >= stable
        if name.startswith("gap"):
            gap = float(name.split("_", maxsplit=1)[0].removeprefix("gap"))
            condition &= (
                checkpoints["frontier_best"]
                > gap * checkpoints["kth_passing_raw_distance"]
            )
    return condition & valid_slots


def selected_slots(
    name: str, checkpoints: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    slots = np.arange(checkpoints.shape[1])[None, :]
    valid_slots = slots < counts[:, None]
    streaks = stable_streaks(checkpoints, valid_slots)
    condition = policy_mask(name, checkpoints, valid_slots, streaks)
    fired = np.any(condition, axis=1)
    first = np.argmax(condition, axis=1)
    selected = np.where(fired, first, counts - 1).astype(np.int64)
    return selected, fired


def evaluate_policy(
    policy: Policy, checkpoints: np.ndarray, counts: np.ndarray, truth: np.ndarray
) -> dict[str, float | int | str | None]:
    selected, _ = selected_slots(policy.name, checkpoints, counts)
    chosen = checkpoints[np.arange(len(counts)), selected]
    matches = np.any(chosen["top_ids"][:, :, None] == truth[:, None, :], axis=2)
    instant_recalls = np.sum(matches, axis=1) / 10.0
    values = chosen["iteration"].astype(np.float64)
    return {
        "policy": policy.name,
        "recall": float(np.mean(instant_recalls)),
        "accumulated_recall": None,
        "mean_iterations": float(np.mean(values)),
        "median_iterations": float(np.median(values)),
        "p95_iterations": float(np.percentile(values, 95)),
        "cap_stop_fraction": float(np.mean(selected == counts - 1)),
    }


def populate_selected_accumulated_recall(
    rows: list[dict[str, object]],
    selected_policy: str,
    capture_root: Path,
    data_root: Path,
) -> None:
    for dataset, spec in DATASETS.items():
        gt_path = data_root / str(spec["directory"]) / "favor/groundtruth_s01.ibin"
        for hash_variant in HASHES:
            directory = capture_root / dataset / hash_variant
            manifest, checkpoints, counts, _ = load_capture(directory)
            truth = load_ibin(gt_path, int(manifest["num_queries"]))
            selected, _ = selected_slots(selected_policy, checkpoints, counts)
            recalls = [
                recall(
                    accumulated_ids(checkpoints[query, : int(selected[query]) + 1]),
                    truth[query],
                )
                for query in range(len(counts))
            ]
            for row in rows:
                if (
                    row["dataset"] == dataset
                    and row["hash"] == hash_variant
                    and row["policy"] == selected_policy
                ):
                    row["accumulated_recall"] = float(np.mean(recalls))
                    break


def evaluate_all(capture_root: Path, data_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    policy_rows: list[dict[str, object]] = []
    hash_rows: list[dict[str, object]] = []
    for dataset, spec in DATASETS.items():
        gt_path = data_root / str(spec["directory"]) / "favor/groundtruth_s01.ibin"
        for hash_variant in HASHES:
            directory = capture_root / dataset / hash_variant
            manifest, checkpoints, counts, summaries = load_capture(directory)
            truth = load_ibin(gt_path, int(manifest["num_queries"]))
            final_recall = float(
                np.mean(
                    [
                        recall(checkpoints[q, int(counts[q]) - 1]["top_ids"], truth[q])
                        for q in range(len(counts))
                    ]
                )
            )
            summary_recall = float(np.mean([float(row["recall"]) for row in summaries]))
            if abs(final_recall - summary_recall) > 1e-6:
                raise ValueError(
                    f"terminal checkpoint/output recall mismatch in {directory}: "
                    f"{final_recall} vs {summary_recall}"
                )
            b0_recalls = [
                recall(checkpoints[query, 0]["top_ids"], truth[query])
                for query in range(len(counts))
            ]
            b0_underfilled = [
                int(checkpoints[query, 0]["output_count"]) < 10
                for query in range(len(counts))
            ]
            hash_rows.append(
                {
                    "dataset": dataset,
                    "hash": hash_variant,
                    "b0_recall": float(np.mean(b0_recalls)),
                    "b0_underfilled_fraction": float(np.mean(b0_underfilled)),
                    "final_recall": final_recall,
                    "mean_candidate_evaluations": float(
                        np.mean([float(row["candidate_evaluations"]) for row in summaries])
                    ),
                    "mean_candidate_duplicates": float(
                        np.mean([float(row["candidate_duplicates"]) for row in summaries])
                    ),
                    "sum_candidate_hash_full": sum(
                        int(row["candidate_hash_full"]) for row in summaries
                    ),
                    "hash_bitlen": int(summaries[0]["hash_bitlen"]),
                    "small_hash_bitlen": int(summaries[0]["small_hash_bitlen"]),
                    "small_hash_reset_interval": int(
                        summaries[0]["small_hash_reset_interval"]
                    ),
                }
            )
            for policy in policies():
                row = evaluate_policy(policy, checkpoints, counts, truth)
                row.update(
                    {
                        "dataset": dataset,
                        "hash": hash_variant,
                        "mean_b0_multiple": float(row["mean_iterations"]) / float(spec["b0"]),
                    }
                )
                policy_rows.append(row)
    return policy_rows, hash_rows


def select_policy(
    rows: list[dict[str, object]], hash_rows: list[dict[str, object]]
) -> dict[str, object]:
    forgetful_hash_recall_safe = all(
        float(row["final_recall"]) >= 0.905
        for row in hash_rows
        if row["hash"] == "forgetful"
    )
    candidates: list[dict[str, object]] = []
    for name in {str(row["policy"]) for row in rows}:
        cells = [
            row for row in rows if row["hash"] == "forgetful" and row["policy"] == name
        ]
        if len(cells) != len(DATASETS):
            raise ValueError(f"missing forgetful cells for {name}")
        qualifies = forgetful_hash_recall_safe and all(
            float(row["recall"]) >= 0.905 for row in cells
        )
        geometric_mean_iterations = math.exp(
            sum(math.log(float(row["mean_iterations"])) for row in cells) / len(cells)
        )
        candidates.append(
            {
                "policy": name,
                "qualifies": qualifies,
                "geometric_mean_iterations": geometric_mean_iterations,
                "recall_by_dataset": {
                    str(row["dataset"]): float(row["recall"]) for row in cells
                },
            }
        )
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    selected = min(qualifying, key=lambda item: item["geometric_mean_iterations"]) if qualifying else None
    return {
        "recall_floor": 0.905,
        "selected": selected,
        "live_v2_justified": selected is not None,
        "forgetful_hash_recall_safe_all": forgetful_hash_recall_safe,
        "disposition": (
            "implement_live_v2"
            if selected
            else "reject_forgetful_hash_no_live_v2"
            if not forgetful_hash_recall_safe
            else "hash_only_no_live_termination"
        ),
        "candidates": sorted(candidates, key=lambda item: item["geometric_mean_iterations"]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_median(path: Path) -> dict[str, float]:
    benchmarks = json.loads(path.read_text())["benchmarks"]
    medians = [row for row in benchmarks if row.get("aggregate_name") == "median"]
    row = medians[-1] if medians else benchmarks[-1]
    return {
        "recall": float(row["Recall"]),
        "qps": float(row["items_per_second"]),
        "underfilled_fraction": float(row.get("UnderfilledQueries", 0.0)),
    }


def factorial_rows(result_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        exact = benchmark_median(result_dir / "raw" / f"{dataset}_exact_current.json")
        forgetful = benchmark_median(result_dir / "raw" / f"{dataset}_forgetful_current.json")
        for hash_variant, values in (("exact", exact), ("forgetful", forgetful)):
            rows.append(
                {
                    "dataset": dataset,
                    "hash": hash_variant,
                    "termination": "current_adaptive",
                    **values,
                    "qps_vs_exact_current": values["qps"] / exact["qps"],
                    "recall_passes_0_90": values["recall"] >= 0.90,
                    "accepted": values["recall"] >= 0.90
                    and values["qps"] >= 1.10 * exact["qps"],
                }
            )
        for hash_variant in HASHES:
            rows.append(
                {
                    "dataset": dataset,
                    "hash": hash_variant,
                    "termination": "v2",
                    "recall": None,
                    "qps": None,
                    "underfilled_fraction": None,
                    "qps_vs_exact_current": None,
                    "recall_passes_0_90": False,
                    "accepted": False,
                }
            )
    return rows


def write_report(
    path: Path,
    policy_rows: list[dict[str, object]],
    hash_rows: list[dict[str, object]],
    factorial: list[dict[str, object]],
    gate: dict[str, object],
) -> None:
    selected = gate["selected"]
    lines = [
        "# FAVOR termination + forgetful-hash shadow diagnostic",
        "",
        f"Disposition: **{gate['disposition']}**",
        "",
        "The policy gate uses instantaneous checkpoint top-10 recall. For a selected policy,",
        "accumulated checkpoint recall is computed afterward as a sensitivity diagnostic only.",
        "",
        "## Deep hash controls",
        "",
        "| Dataset | Hash | B0 recall | B0 underfill | Final recall | Candidate evals/query | Duplicates/query | Hash full | Hash bits | Small bits | Reset |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hash_rows:
        lines.append(
            f"| {row['dataset']} | {row['hash']} | {float(row['b0_recall']):.5f} | "
            f"{float(row['b0_underfilled_fraction']):.4f} | "
            f"{float(row['final_recall']):.5f} | "
            f"{float(row['mean_candidate_evaluations']):.1f} | "
            f"{float(row['mean_candidate_duplicates']):.1f} | "
            f"{int(row['sum_candidate_hash_full'])} | {int(row['hash_bitlen'])} | "
            f"{int(row['small_hash_bitlen'])} | {int(row['small_hash_reset_interval'])} |"
        )
    lines.extend(
        [
            "",
            "## Uninstrumented factorial edge",
            "",
            "The first repetition may include JIT loading, so the table uses the three-repetition",
            "median. V2 cells were not run because the shadow gate rejected implementation.",
            "",
            "| Dataset | Hash | Termination | Recall | QPS | QPS/exact-current | Underfill | Accepted |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in factorial:
        recall_value = "—" if row["recall"] is None else f"{float(row['recall']):.5f}"
        qps_value = "—" if row["qps"] is None else f"{float(row['qps']):.1f}"
        ratio_value = (
            "—"
            if row["qps_vs_exact_current"] is None
            else f"{float(row['qps_vs_exact_current']):.3f}"
        )
        underfill_value = (
            "—"
            if row["underfilled_fraction"] is None
            else f"{float(row['underfilled_fraction']):.4f}"
        )
        lines.append(
            f"| {row['dataset']} | {row['hash']} | {row['termination']} | {recall_value} | "
            f"{qps_value} | {ratio_value} | {underfill_value} | {row['accepted']} |"
        )
    if selected is not None:
        name = str(selected["policy"])
        lines.extend(
            [
                "",
                f"## Selected global policy: `{name}`",
                "",
                "| Dataset | Hash | Recall | Accumulated sensitivity | Mean iterations | B0 multiple | Cap stops |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in policy_rows:
            if row["policy"] != name:
                continue
            lines.append(
                f"| {row['dataset']} | {row['hash']} | {float(row['recall']):.5f} | "
                f"{float(row['accumulated_recall']):.5f} | "
                f"{float(row['mean_iterations']):.1f} | "
                f"{float(row['mean_b0_multiple']):.3f} | "
                f"{float(row['cap_stop_fraction']):.3f} |"
            )
    lines.extend(["", "## Gate", "", "```json", json.dumps(gate, indent=2), "```", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    policy_rows, hash_rows = evaluate_all(args.capture_root, args.data_root)
    gate = select_policy(policy_rows, hash_rows)
    if gate["selected"] is not None:
        populate_selected_accumulated_recall(
            policy_rows,
            str(gate["selected"]["policy"]),
            args.capture_root,
            args.data_root,
        )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    factorial = factorial_rows(args.result_dir)
    write_csv(args.result_dir / "policy_summary.csv", policy_rows)
    write_csv(args.result_dir / "hash_summary.csv", hash_rows)
    write_csv(args.result_dir / "factorial_summary.csv", factorial)
    (args.result_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    write_report(args.result_dir / "report.md", policy_rows, hash_rows, factorial, gate)
    print(f"disposition={gate['disposition']}; wrote {args.result_dir}")


if __name__ == "__main__":
    main()
