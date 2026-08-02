#!/usr/bin/env python3
"""Combine saved-state retry captures and apply the causal decision gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generate_configs import DATASETS, STRATEGIES


def read_rounds(capture_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            directory = capture_root / dataset / strategy
            manifest = json.loads((directory / "manifest.json").read_text())
            if not manifest.get("complete"):
                raise ValueError(f"incomplete capture: {directory}")
            if manifest["dataset"] != dataset or manifest["strategy"] != strategy:
                raise ValueError(f"capture identity mismatch: {directory}")
            with (directory / "round_metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != int(DATASETS[dataset]["rounds"]):
                raise ValueError(f"wrong round count in {directory}")
            for row in rows:
                records.append(
                    {
                        "dataset": dataset,
                        "strategy": strategy,
                        "round": int(row["round"]),
                        "budget": int(row["budget"]),
                        "individual_recall": float(row["individual_recall"]),
                        "accumulated_recall": float(row["accumulated_recall"]),
                        "mean_jaccard_previous": float(row["mean_jaccard_previous"]),
                        "mean_new_ids": float(row["mean_new_ids"]),
                        "mean_new_ground_truth": float(row["mean_new_ground_truth"]),
                        "median_frontier_unexpanded": float(
                            row["median_frontier_unexpanded"]
                        ),
                        "mean_frontier_unexpanded": float(row["mean_frontier_unexpanded"]),
                        "mean_frontier_unexpanded_pass": float(
                            row["mean_frontier_unexpanded_pass"]
                        ),
                        "mean_frontier_unexpanded_reject": float(
                            row["mean_frontier_unexpanded_reject"]
                        ),
                        "mean_retry_seed_count": float(row["mean_retry_seed_count"]),
                        "mean_candidate_evaluations": float(
                            row["mean_candidate_evaluations"]
                        ),
                        "mean_candidate_duplicates": float(
                            row["mean_candidate_duplicates"]
                        ),
                        "sum_candidate_hash_full": int(row["sum_candidate_hash_full"]),
                    }
                )
    return records


def decide(records: list[dict[str, object]]) -> dict[str, object]:
    final = {
        (str(row["dataset"]), str(row["strategy"])): row
        for row in records
        if int(row["round"]) == int(DATASETS[str(row["dataset"])]["rounds"])
    }
    first = {
        str(row["dataset"]): row
        for row in records
        if str(row["strategy"]) == "independent" and int(row["round"]) == 1
    }
    datasets = tuple(DATASETS)
    oracle_reaches_target = all(
        float(final[(dataset, "oracle")]["accumulated_recall"]) >= 0.90
        for dataset in datasets
    )
    target = {
        strategy: all(
            float(final[(dataset, strategy)]["accumulated_recall"]) >= 0.90
            for dataset in datasets
        )
        for strategy in ("passing", "frontier", "combined")
    }
    partial = {}
    for strategy in ("passing", "frontier", "combined"):
        details = []
        for dataset in datasets:
            baseline = float(first[dataset]["accumulated_recall"])
            candidate = float(final[(dataset, strategy)]["accumulated_recall"])
            independent = float(final[(dataset, "independent")]["accumulated_recall"])
            oracle = float(final[(dataset, "oracle")]["accumulated_recall"])
            oracle_gain = max(0.0, oracle - baseline)
            details.append(
                {
                    "dataset": dataset,
                    "candidate_gain": candidate - baseline,
                    "oracle_gain": oracle_gain,
                    "fraction_of_oracle_gain": (
                        (candidate - baseline) / oracle_gain if oracle_gain > 0 else 0.0
                    ),
                    "margin_vs_independent": candidate - independent,
                    "passes": oracle_gain > 0
                    and candidate - baseline >= 0.5 * oracle_gain
                    and candidate - independent > 0.01,
                }
            )
        partial[strategy] = {
            "passes": all(detail["passes"] for detail in details),
            "details": details,
        }

    if not oracle_reaches_target:
        conclusion = "depth_not_confirmed"
    elif target["passing"]:
        conclusion = "passing_accumulator_sufficient"
    elif target["frontier"]:
        conclusion = "frontier_reseed_sufficient"
    elif target["combined"]:
        conclusion = "combined_reseed_sufficient"
    elif any(value["passes"] for value in partial.values()):
        conclusion = "saved_state_reseed_partial"
    else:
        conclusion = "full_in_kernel_state_required"
    return {
        "conclusion": conclusion,
        "oracle_reaches_0_90_all": oracle_reaches_target,
        "reseed_reaches_0_90_all": target,
        "partial_gain_gate": partial,
    }


def write_summary(path: Path, records: list[dict[str, object]]) -> None:
    fields = list(records[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_report(
    path: Path, records: list[dict[str, object]], decision: dict[str, object]
) -> None:
    final = [
        row
        for row in records
        if int(row["round"]) == int(DATASETS[str(row["dataset"])]["rounds"])
    ]
    lines = [
        "# FAVOR saved-state retry diagnostic",
        "",
        f"Decision: **{decision['conclusion']}**",
        "",
        "These are diagnostic recalls, not benchmark throughput results. Every restart strategy",
        "uses a fresh hash table; the oracle instead reruns one uninterrupted traversal to the",
        "same cumulative iteration budget.",
        "",
        "| Dataset | Strategy | Rounds | Final recall | Last-round recall | New GT/query | Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in final:
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | {row['round']} | "
            f"{float(row['accumulated_recall']):.5f} | "
            f"{float(row['individual_recall']):.5f} | "
            f"{float(row['mean_new_ground_truth']):.4f} | "
            f"{float(row['mean_jaccard_previous']):.4f} |"
        )
    lines.extend(["", "```json", json.dumps(decision, indent=2), "```", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    records = read_rounds(args.capture_root)
    decision = decide(records)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.result_dir / "summary.csv", records)
    write_report(args.result_dir / "report.md", records, decision)
    (args.result_dir / "gate.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(f"decision={decision['conclusion']}; wrote {args.result_dir}")


if __name__ == "__main__":
    main()
