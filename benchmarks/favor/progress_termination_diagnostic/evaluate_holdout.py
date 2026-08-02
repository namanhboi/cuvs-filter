#!/usr/bin/env python3
"""Apply the frozen development rule to DEEP-image1M without retuning it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze
from generate_holdout_configs import B0, SEEDS


def load_seed(seed: int, capture_root: Path, data_root: Path) -> analyze.Capture:
    directory = capture_root / f"seed{seed}"
    manifest = json.loads((directory / "manifest.json").read_text())
    if int(manifest["schema_version"]) != 4:
        raise ValueError(f"holdout capture is not schema 4: {directory}")
    if int(manifest["termination_checkpoint_record_size"]) != 136:
        raise ValueError(f"holdout checkpoint ABI mismatch: {directory}")
    num_queries = int(manifest["num_queries"])
    stride = int(manifest["termination_checkpoint_stride"])
    checkpoints = np.fromfile(
        directory / "termination_checkpoints.bin", analyze.CHECKPOINT_DTYPE
    )
    if checkpoints.size != num_queries * stride:
        raise ValueError(f"holdout checkpoint extent mismatch: {directory}")
    checkpoints = checkpoints.reshape(num_queries, stride)
    counts = np.fromfile(directory / "termination_checkpoint_counts.bin", dtype="<u4")
    if counts.size != num_queries or np.any(counts == 0) or np.any(counts > stride):
        raise ValueError(f"invalid holdout checkpoint counts: {directory}")
    with (directory / "query_summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    truth = analyze.load_ibin(
        data_root
        / "deep-image-1M"
        / f"favor_seed{seed}/groundtruth_s01.ibin",
        num_queries,
    )
    return analyze.Capture(
        f"deep_image1m_seed{seed}", manifest, checkpoints, counts, truth, summaries
    )


def evaluate(
    capture_root: Path, data_root: Path, result_dir: Path, frozen_path: Path
) -> dict[str, object]:
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("holdout_seeds") != list(SEEDS) or frozen.get("retuning_after_holdout"):
        raise ValueError("frozen rule does not declare the expected untouched holdout")
    rule = analyze.Rule(
        int(frozen["evidence_multiple"]),
        None if frozen["frontier_gap"] is None else float(frozen["frontier_gap"]),
    )
    if rule.name != frozen["rule"]:
        raise ValueError("frozen rule name/parameters disagree")

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        capture = load_seed(seed, capture_root, data_root)
        selected, fired = analyze.selected_slots(capture, rule, b0_override=B0)
        chosen = capture.checkpoints[np.arange(len(selected)), selected]
        b0 = analyze.uniform_iteration_metrics(capture, B0)
        fixed = analyze.fixed_target_metrics(capture)
        recall = float(np.mean(analyze.recalls_for_records(chosen, capture.truth)))
        work = float(np.mean(chosen["cumulative_candidate_evaluations"]))
        limit = (
            1.10 * float(fixed["mean_work"])
            if float(b0["recall"]) < 0.90
            else 1.25 * float(b0["mean_work"])
        )
        rows.append(
            {
                "seed": seed,
                "rule": rule.name,
                "recall": recall,
                "mean_iterations": float(np.mean(chosen["iteration"])),
                "mean_work": work,
                "work_limit": limit,
                "b0_recall": b0["recall"],
                "fixed_target_iteration": fixed["iteration"],
                "fixed_target_recall": fixed["recall"],
                "fixed_target_work": fixed["mean_work"],
                "fired_fraction": float(np.mean(fired)),
                "underfilled_fraction": float(np.mean(chosen["output_count"] < 10)),
                "hash_full": sum(
                    int(summary["candidate_hash_full"])
                    for summary in capture.summaries
                ),
                "passed": recall >= 0.90
                and work <= limit
                and not np.any(chosen["output_count"] < 10),
            }
        )

    passed = all(bool(row["passed"]) and int(row["hash_full"]) == 0 for row in rows)
    gate: dict[str, object] = {
        "schema_version": 1,
        "frozen_rule": rule.name,
        "seeds": list(SEEDS),
        "holdout_passed": passed,
        "retuned": False,
        "disposition": (
            "holdout_pass_shadow_rule_justified"
            if passed
            else "holdout_fail_no_live_rule"
        ),
        "results": rows,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "holdout_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    with (result_dir / "holdout_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Frozen DEEP-image1M holdout",
        "",
        f"Disposition: **{gate['disposition']}**",
        "",
        f"Frozen rule: `{rule.name}`. No parameter was selected or changed using this data.",
        "",
        "| Seed | Recall | Mean iterations | Mean work / limit | B0 recall | Fixed >=.905 iteration | Fired | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {float(row['recall']):.5f} | "
            f"{float(row['mean_iterations']):.1f} | {float(row['mean_work']):.1f} / "
            f"{float(row['work_limit']):.1f} | {float(row['b0_recall']):.5f} | "
            f"{int(float(row['fixed_target_iteration']))} | "
            f"{float(row['fired_fraction']):.4f} | {row['passed']} |"
        )
    (result_dir / "holdout_report.md").write_text("\n".join(lines) + "\n")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    args = parser.parse_args()
    gate = evaluate(
        args.capture_root, args.data_root, args.result_dir, args.frozen_rule
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
