#!/usr/bin/env python3
"""Select and cross-validate a query-local FAVOR progress termination rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_configs import DATASETS, SELECTIVITY


CHECKPOINT_DTYPE = np.dtype(
    [
        ("query_id", "<u4"),
        ("checkpoint", "<u4"),
        ("iteration", "<u4"),
        ("expanded_parents", "<u4"),
        ("cumulative_candidate_evaluations", "<u4"),
        ("cumulative_passing_candidates", "<u4"),
        ("cumulative_candidate_duplicates", "<u4"),
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
assert CHECKPOINT_DTYPE.itemsize == 136


@dataclass(frozen=True, order=True)
class Rule:
    evidence_multiple: int
    frontier_gap: float | None

    @property
    def name(self) -> str:
        gap = "off" if self.frontier_gap is None else f"{self.frontier_gap:.2f}"
        return f"evidence{self.evidence_multiple}_gap{gap}"


@dataclass
class Capture:
    slug: str
    manifest: dict[str, object]
    checkpoints: np.ndarray
    counts: np.ndarray
    truth: np.ndarray
    summaries: list[dict[str, str]]


def rules() -> list[Rule]:
    return [
        Rule(multiple, gap)
        for multiple in (2, 4, 8, 12, 16, 24, 32, 48, 64)
        for gap in (None, 1.0, 1.05)
    ]


def load_ibin(path: Path, rows: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype="<i4", count=2)
        if len(header) != 2 or header[0] < rows or header[1] < 10:
            raise ValueError(f"invalid nq x k ground truth: {path}")
        values = np.fromfile(stream, dtype="<i4", count=int(header[0] * header[1]))
    return values.reshape(int(header[0]), int(header[1]))[:rows, :10].astype(np.uint32)


def load_capture(slug: str, capture_root: Path, data_root: Path) -> Capture:
    directory = capture_root / slug
    manifest = json.loads((directory / "manifest.json").read_text())
    if int(manifest["schema_version"]) not in (4, 5):
        raise ValueError(f"expected diagnostic schema 4 or 5: {directory}")
    if int(manifest["termination_checkpoint_record_size"]) != CHECKPOINT_DTYPE.itemsize:
        raise ValueError(f"checkpoint ABI mismatch: {directory}")
    num_queries = int(manifest["num_queries"])
    stride = int(manifest["termination_checkpoint_stride"])
    checkpoints = np.fromfile(directory / "termination_checkpoints.bin", CHECKPOINT_DTYPE)
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
    spec = DATASETS[slug]
    truth = load_ibin(
        data_root / str(spec["directory"]) / "favor/groundtruth_s01.ibin", num_queries
    )
    return Capture(slug, manifest, checkpoints, counts, truth, summaries)


def recalls_for_records(records: np.ndarray, truth: np.ndarray) -> np.ndarray:
    matches = np.any(records["top_ids"][:, :, None] == truth[:, None, :], axis=2)
    return np.sum(matches, axis=1).astype(np.float64) / 10.0


def slot_at_or_before(records: np.ndarray, iteration: int) -> int:
    eligible = np.flatnonzero(records["iteration"] <= iteration)
    if eligible.size == 0:
        raise ValueError(f"trajectory starts after requested iteration {iteration}")
    return int(eligible[-1])


def uniform_iteration_metrics(capture: Capture, iteration: int) -> dict[str, float]:
    slots = np.arange(capture.checkpoints.shape[1])[None, :]
    valid = slots < capture.counts[:, None]
    eligible = valid & (capture.checkpoints["iteration"] <= iteration)
    if np.any(~np.any(eligible, axis=1)):
        raise ValueError(f"at least one trajectory starts after iteration {iteration}")
    selected = np.sum(eligible, axis=1) - 1
    chosen = capture.checkpoints[np.arange(len(capture.counts)), selected]
    return {
        "recall": float(np.mean(recalls_for_records(chosen, capture.truth))),
        "mean_iterations": float(np.mean(chosen["iteration"])),
        "mean_work": float(np.mean(chosen["cumulative_candidate_evaluations"])),
    }


def fixed_target_metrics(capture: Capture, recall_floor: float = 0.905) -> dict[str, float]:
    iterations = sorted(
        {
            int(record["iteration"])
            for query, count in enumerate(capture.counts)
            for record in capture.checkpoints[query, : int(count)]
        }
    )
    for iteration in iterations:
        metrics = uniform_iteration_metrics(capture, iteration)
        if metrics["recall"] >= recall_floor:
            return {"iteration": float(iteration), **metrics}
    terminal = uniform_iteration_metrics(capture, iterations[-1])
    return {"iteration": float(iterations[-1]), **terminal, "target_unreachable": 1.0}


def stale_evidence(records: np.ndarray, selectivity: float) -> np.ndarray:
    """Return conservative evidence accumulated since the ordered passing top-10 changed."""
    result = np.zeros(len(records), dtype=np.float64)
    previous_top: np.ndarray | None = None
    anchor_evaluations = 0
    anchor_passing = 0
    for slot, record in enumerate(records):
        evaluations = int(record["cumulative_candidate_evaluations"])
        passing = int(record["cumulative_passing_candidates"])
        complete = int(record["output_count"]) == 10
        top = record["top_ids"]
        changed = previous_top is None or not np.array_equal(top, previous_top)
        if not complete or changed:
            anchor_evaluations = evaluations
            anchor_passing = passing
            result[slot] = 0.0
        else:
            stale_unique = max(0, evaluations - anchor_evaluations)
            stale_passing = max(0, passing - anchor_passing)
            result[slot] = min(float(stale_passing), selectivity * stale_unique)
        previous_top = top.copy() if complete else None
    return result


def evidence_matrix(capture: Capture, selectivity: float = SELECTIVITY) -> np.ndarray:
    result = np.zeros(capture.checkpoints.shape, dtype=np.float64)
    for query, count in enumerate(capture.counts):
        result[query, : int(count)] = stale_evidence(
            capture.checkpoints[query, : int(count)], selectivity
        )
    return result


def selected_slots(
    capture: Capture,
    rule: Rule,
    b0_override: int | None = None,
    selectivity: float = SELECTIVITY,
    evidence_cache: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    b0 = int(DATASETS[capture.slug]["b0"]) if b0_override is None else b0_override
    evidence = (
        evidence_matrix(capture, selectivity)
        if evidence_cache is None
        else evidence_cache
    )
    if evidence.shape != capture.checkpoints.shape:
        raise ValueError("evidence cache extent does not match checkpoints")
    threshold = 10.0 * rule.evidence_multiple
    slots = np.arange(capture.checkpoints.shape[1])[None, :]
    condition = slots < capture.counts[:, None]
    condition &= capture.checkpoints["iteration"] >= b0
    condition &= capture.checkpoints["output_count"] == 10
    condition &= evidence >= threshold
    if rule.frontier_gap is not None:
        condition &= capture.checkpoints["frontier_best"] > (
            rule.frontier_gap * capture.checkpoints["kth_passing_raw_distance"]
        )
    fired = np.any(condition, axis=1)
    first = np.argmax(condition, axis=1)
    selected = np.where(fired, first, capture.counts - 1).astype(np.int64)
    return selected, fired


def evaluate_rule(
    capture: Capture, rule: Rule, evidence_cache: np.ndarray | None = None
) -> dict[str, object]:
    selected, fired = selected_slots(capture, rule, evidence_cache=evidence_cache)
    chosen = capture.checkpoints[np.arange(len(selected)), selected]
    return {
        "dataset": capture.slug,
        "family": DATASETS[capture.slug]["family"],
        "rule": rule.name,
        "evidence_multiple": rule.evidence_multiple,
        "frontier_gap": rule.frontier_gap,
        "recall": float(np.mean(recalls_for_records(chosen, capture.truth))),
        "mean_iterations": float(np.mean(chosen["iteration"])),
        "mean_expanded_parents": float(np.mean(chosen["expanded_parents"])),
        "mean_work": float(np.mean(chosen["cumulative_candidate_evaluations"])),
        "p95_iterations": float(np.percentile(chosen["iteration"], 95)),
        "fired_fraction": float(np.mean(fired)),
        "cap_fraction": float(np.mean(~fired)),
        "underfilled_fraction": float(np.mean(chosen["output_count"] < 10)),
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def work_limit(control: dict[str, object]) -> float:
    if float(control["b0_recall"]) < 0.90:
        return 1.10 * float(control["fixed_target_work"])
    return 1.25 * float(control["b0_work"])


def qualifies(
    rule_name: str,
    slugs: list[str],
    rows_by_key: dict[tuple[str, str], dict[str, object]],
    controls: dict[str, dict[str, object]],
    recall_floor: float,
) -> bool:
    return all(
        float(rows_by_key[(slug, rule_name)]["recall"]) >= recall_floor
        and float(rows_by_key[(slug, rule_name)]["mean_work"]) <= work_limit(controls[slug])
        for slug in slugs
    )


def select_lowest_work(
    candidate_rules: list[Rule],
    slugs: list[str],
    rows_by_key: dict[tuple[str, str], dict[str, object]],
    controls: dict[str, dict[str, object]],
) -> Rule | None:
    eligible = [
        rule
        for rule in candidate_rules
        if qualifies(rule.name, slugs, rows_by_key, controls, 0.905)
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda rule: (
            geometric_mean(
                [
                    float(rows_by_key[(slug, rule.name)]["mean_work"])
                    / float(controls[slug]["fixed_target_work"])
                    for slug in slugs
                ]
            ),
            rule.evidence_multiple,
            -1.0 if rule.frontier_gap is None else rule.frontier_gap,
        ),
    )


def cross_validate(
    candidate_rules: list[Rule],
    rows_by_key: dict[tuple[str, str], dict[str, object]],
    controls: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    families = sorted({str(spec["family"]) for spec in DATASETS.values()})
    folds: list[dict[str, object]] = []
    for heldout_family in families:
        heldout = [
            slug for slug, spec in DATASETS.items() if spec["family"] == heldout_family
        ]
        training = [slug for slug in DATASETS if slug not in heldout]
        selected = select_lowest_work(candidate_rules, training, rows_by_key, controls)
        heldout_rows = [] if selected is None else [rows_by_key[(slug, selected.name)] for slug in heldout]
        passed = selected is not None and all(
            float(row["recall"]) >= 0.90
            and float(row["mean_work"]) <= work_limit(controls[str(row["dataset"])])
            for row in heldout_rows
        )
        folds.append(
            {
                "heldout_family": heldout_family,
                "training_datasets": training,
                "heldout_datasets": heldout,
                "selected_rule": None if selected is None else selected.name,
                "passed": passed,
                "heldout": [
                    {
                        "dataset": row["dataset"],
                        "recall": row["recall"],
                        "mean_work": row["mean_work"],
                        "work_limit": work_limit(controls[str(row["dataset"])]),
                    }
                    for row in heldout_rows
                ],
            }
        )
    return folds


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_rule(path: Path, selected: Rule, gate: dict[str, object]) -> None:
    payload = {
        "schema_version": 1,
        "rule": selected.name,
        "evidence_multiple": selected.evidence_multiple,
        "frontier_gap": selected.frontier_gap,
        "k": 10,
        "development_selectivity": SELECTIVITY,
        "development_datasets": list(DATASETS),
        "development_gate": gate["disposition"],
        "holdout": "deep-image-96-angular-1M",
        "holdout_seeds": [20260802, 20260803],
        "retuning_after_holdout": False,
    }
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise RuntimeError(
                f"frozen rule differs from the new selection; refusing to retune {path}"
            )
        return
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_report(
    path: Path,
    controls: dict[str, dict[str, object]],
    rows_by_key: dict[tuple[str, str], dict[str, object]],
    folds: list[dict[str, object]],
    gate: dict[str, object],
) -> None:
    lines = [
        "# Exact-state FAVOR progress termination diagnostic",
        "",
        f"Disposition: **{gate['disposition']}**",
        "",
        "The shadow rule never changes kernel termination. It measures evidence accumulated since",
        "the ordered passing top-10 last changed and is eligible only at or after B0.",
        "",
        "## Controls",
        "",
        "| Dataset | B0 recall | B0 work | Fixed >=.905 iteration | Fixed recall | Fixed work | Cap recall | Hash-full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, row in controls.items():
        lines.append(
            f"| {slug} | {float(row['b0_recall']):.5f} | {float(row['b0_work']):.1f} | "
            f"{int(float(row['fixed_target_iteration']))} | {float(row['fixed_target_recall']):.5f} | "
            f"{float(row['fixed_target_work']):.1f} | {float(row['cap_recall']):.5f} | "
            f"{int(row['hash_full'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-dataset admissible rules",
            "",
            "These are diagnostic optima, not deployable choices. Their disagreement is the reason",
            "the dataset-independent gate fails.",
            "",
            "| Dataset | Lowest-work admissible rule | Recall | Mean work / limit |",
            "|---|---|---:|---:|",
        ]
    )
    for slug, best in gate["per_dataset_best"].items():
        if best is None:
            lines.append(f"| {slug} | none | — | — |")
        else:
            lines.append(
                f"| {slug} | {best['rule']} | {float(best['recall']):.5f} | "
                f"{float(best['mean_work']):.1f} / {float(best['work_limit']):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Progress-signal dynamics",
            "",
            "| Dataset | B0 evidence p50 / p90 | Max evidence p50 | Top-10 changes p50 / p90 |",
            "|---|---:|---:|---:|",
        ]
    )
    for slug, row in controls.items():
        lines.append(
            f"| {slug} | {float(row['b0_evidence_p50']):.1f} / "
            f"{float(row['b0_evidence_p90']):.1f} | {float(row['max_evidence_p50']):.1f} | "
            f"{float(row['top10_changes_p50']):.0f} / {float(row['top10_changes_p90']):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-family-out validation",
            "",
            "| Held-out family | Selected on training families | Held-out result | Pass |",
            "|---|---|---|---|",
        ]
    )
    for fold in folds:
        heldout = ", ".join(
            f"{row['dataset']} r={float(row['recall']):.5f} "
            f"work={float(row['mean_work']):.1f}/{float(row['work_limit']):.1f}"
            for row in fold["heldout"]
        )
        lines.append(
            f"| {fold['heldout_family']} | {fold['selected_rule'] or 'none'} | "
            f"{heldout or 'no qualifying training rule'} | {fold['passed']} |"
        )
    selected = gate.get("selected_rule")
    if selected is not None:
        lines.extend(
            [
                "",
                "## Frozen development rule",
                "",
                f"Selected: `{selected}`.",
                "",
                "| Dataset | Recall | Mean work | Work limit | Fired |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for slug in DATASETS:
            row = rows_by_key[(slug, str(selected))]
            lines.append(
                f"| {slug} | {float(row['recall']):.5f} | {float(row['mean_work']):.1f} | "
                f"{work_limit(controls[slug]):.1f} | {float(row['fired_fraction']):.4f} |"
            )
    lines.extend(
        [
            "",
            "A DEEP-image1M holdout may be run only when the disposition is `run_frozen_holdout`.",
            "Failure leaves the rule shadow-only; it must not be retuned on the holdout.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def analyze(capture_root: Path, data_root: Path, result_dir: Path) -> dict[str, object]:
    captures = {
        slug: load_capture(slug, capture_root, data_root) for slug in DATASETS
    }
    evidence = {slug: evidence_matrix(capture) for slug, capture in captures.items()}
    controls: dict[str, dict[str, object]] = {}
    for slug, capture in captures.items():
        b0 = uniform_iteration_metrics(capture, int(DATASETS[slug]["b0"]))
        fixed = fixed_target_metrics(capture)
        cap_iteration = max(
            int(capture.checkpoints[q, int(count) - 1]["iteration"])
            for q, count in enumerate(capture.counts)
        )
        cap = uniform_iteration_metrics(capture, cap_iteration)
        b0_evidence = []
        max_evidence = []
        top10_changes = []
        for query, count_value in enumerate(capture.counts):
            count = int(count_value)
            records = capture.checkpoints[query, :count]
            b0_slot = slot_at_or_before(records, int(DATASETS[slug]["b0"]))
            b0_evidence.append(evidence[slug][query, b0_slot])
            max_evidence.append(np.max(evidence[slug][query, :count]))
            complete = records["output_count"] == 10
            changed = np.any(records["top_ids"][1:] != records["top_ids"][:-1], axis=1)
            top10_changes.append(int(np.sum(complete[1:] & changed)))
        controls[slug] = {
            "dataset": slug,
            "family": DATASETS[slug]["family"],
            "b0_recall": b0["recall"],
            "b0_work": b0["mean_work"],
            "fixed_target_iteration": fixed["iteration"],
            "fixed_target_recall": fixed["recall"],
            "fixed_target_work": fixed["mean_work"],
            "fixed_target_reachable": "target_unreachable" not in fixed,
            "cap_recall": cap["recall"],
            "cap_work": cap["mean_work"],
            "hash_full": sum(int(row["candidate_hash_full"]) for row in capture.summaries),
            "b0_evidence_p50": float(np.percentile(b0_evidence, 50)),
            "b0_evidence_p90": float(np.percentile(b0_evidence, 90)),
            "max_evidence_p50": float(np.percentile(max_evidence, 50)),
            "top10_changes_p50": float(np.percentile(top10_changes, 50)),
            "top10_changes_p90": float(np.percentile(top10_changes, 90)),
        }

    candidate_rules = rules()
    rows = [
        evaluate_rule(captures[slug], rule, evidence[slug])
        for slug in DATASETS
        for rule in candidate_rules
    ]
    rows_by_key = {(str(row["dataset"]), str(row["rule"])): row for row in rows}
    folds = cross_validate(candidate_rules, rows_by_key, controls)
    selected = select_lowest_work(candidate_rules, list(DATASETS), rows_by_key, controls)
    per_dataset_best: dict[str, dict[str, object] | None] = {}
    for slug in DATASETS:
        eligible = [
            rows_by_key[(slug, rule.name)]
            for rule in candidate_rules
            if float(rows_by_key[(slug, rule.name)]["recall"]) >= 0.905
            and float(rows_by_key[(slug, rule.name)]["mean_work"])
            <= work_limit(controls[slug])
        ]
        best = min(eligible, key=lambda row: float(row["mean_work"])) if eligible else None
        per_dataset_best[slug] = (
            None
            if best is None
            else {
                "rule": best["rule"],
                "recall": best["recall"],
                "mean_work": best["mean_work"],
                "work_limit": work_limit(controls[slug]),
            }
        )
    exact_state_safe = all(
        int(row["hash_full"]) == 0
        and bool(row["fixed_target_reachable"])
        and float(row["cap_recall"]) >= 0.905
        for row in controls.values()
    )
    cross_validation_passed = all(bool(fold["passed"]) for fold in folds)
    gate_passed = exact_state_safe and cross_validation_passed and selected is not None
    gate: dict[str, object] = {
        "schema_version": 1,
        "exact_state_safe": exact_state_safe,
        "cross_validation_passed": cross_validation_passed,
        "selected_rule": None if selected is None else selected.name,
        "development_gate_passed": gate_passed,
        "disposition": "run_frozen_holdout" if gate_passed else "reject_shadow_rule",
        "per_dataset_best": per_dataset_best,
        "folds": folds,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "controls.csv", list(controls.values()))
    write_csv(result_dir / "candidate_results.csv", rows)
    (result_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    write_report(result_dir / "report.md", controls, rows_by_key, folds, gate)
    if gate_passed and selected is not None:
        freeze_rule(result_dir / "frozen_rule.json", selected, gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    gate = analyze(args.capture_root, args.data_root, args.result_dir)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
