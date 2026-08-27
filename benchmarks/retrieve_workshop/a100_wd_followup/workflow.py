#!/usr/bin/env python3
"""Generate controls, analyze, and bundle the A100 W*D follow-up campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSHOP = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSHOP / "gpu_graph"))

import analyze_gpu_graph as gpu_graph_analysis
from generate_configs import (
    config_payload,
    dataset_paths,
    point_identity,
    search_point,
)

WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = (
    "default_cagra_seeded",
    "default_cagra_accumulator_seeded",
    "navix_reference",
)
METHOD_LABELS = {
    "default_cagra": "Base",
    "default_cagra_accumulator": "Retain",
    "default_cagra_seeded": "Base + $WD$ passing seeds",
    "default_cagra_accumulator_seeded": "Retain + $WD$ passing seeds",
    "navix_reference": "NaviX",
}
TARGETS = {"yfcc": 0.80, "em": 0.95, "emis": 0.95, "r": 0.95}
TARGET_WINDOW = 0.002
K10_REFERENCE_MEMBERS = {
    "k_selected": "seed_ablation/k_seed_selected_points.csv",
    "wd_selected": "seed_ablation/wd_all/matched_recall/analysis/selected_points.csv",
    "wd_b0": "seed_ablation/wd_all/frontier/analysis/summary_points.csv",
}
K100_REFERENCE_MEMBERS = {
    "selected": "matched_recall/selected_points.csv",
    "measurements": "matched_recall/measurements.csv",
    "final_summary": "matched_recall/final_summary.csv",
    "provenance": "matched_recall/provenance.json",
    "gpu_raw": "gpu_graph/raw_points.csv",
    "gpu_repetitions": "gpu_graph/repetition_aggregates.csv",
    "gpu_summary": "gpu_graph/summary_points.csv",
    "gpu_pareto": "gpu_graph/pareto_points.csv",
    "gpu_provenance": "gpu_graph/provenance.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes(rows))


class BundleReader:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _directory_match(self, suffix: str) -> Path:
        matches = [
            path
            for path in self.path.rglob("*")
            if path.is_file() and path.as_posix().endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one bundle member ending {suffix!r}, found {matches}"
            )
        return matches[0]

    def bytes(self, suffix: str) -> bytes:
        if self.path.is_dir():
            return self._directory_match(suffix).read_bytes()
        with tarfile.open(self.path, "r:gz") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(suffix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one archive member ending {suffix!r}, found {[row.name for row in matches]}"
                )
            source = archive.extractfile(matches[0])
            if source is None:
                raise ValueError(f"cannot read {matches[0].name}")
            return source.read()

    def csv(self, suffix: str) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.bytes(suffix).decode())))


def workload_rows(
    rows: list[dict[str, str]], method: str | None = None
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        workload = row.get("workload", "")
        if workload in output:
            raise ValueError(
                f"duplicate {method or 'selected'} row for {workload}"
            )
        output[workload] = row
    if set(output) != set(WORKLOADS):
        raise ValueError(f"incomplete workload rows: {sorted(output)}")
    return output


def bool_value(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def target_comparison(
    k_rows: list[dict[str, str]], wd_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    old = workload_rows(k_rows, "navix_reference")
    new = workload_rows(wd_rows, "navix_reference")
    output: list[dict[str, object]] = []
    for workload in WORKLOADS:
        k_row, wd_row = old[workload], new[workload]
        k_qps, wd_qps = float(k_row["qps_median"]), float(wd_row["qps_median"])
        output.append(
            {
                "workload": workload,
                "target_recall": TARGETS[workload],
                "k_seed_recall": float(k_row["recall_median"]),
                "k_seed_qps": k_qps,
                "wd_seed_recall": float(wd_row["recall_median"]),
                "wd_seed_qps": wd_qps,
                "wd_over_k_qps": wd_qps / k_qps,
                "k_seed_target_reached": bool_value(k_row["target_reached"]),
                "wd_seed_target_reached": bool_value(wd_row["target_reached"]),
            }
        )
    return output


def select_b0_target(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for workload in WORKLOADS:
        target = TARGETS[workload]
        for method in METHODS:
            candidates = [
                row
                for row in rows
                if row["workload"] == workload and row["method"] == method
            ]
            if len(candidates) != 6:
                raise ValueError(
                    f"expected six B0 candidates for {workload}/{method}, got {len(candidates)}"
                )
            qualifying = [
                row for row in candidates if float(row["recall_min"]) >= target
            ]
            within = [
                row
                for row in qualifying
                if float(row["recall_median"]) <= target + TARGET_WINDOW
            ]
            if within:
                winner = max(within, key=lambda row: float(row["qps_median"]))
                target_reached = True
                selection = "maximum B0 QPS inside the target window"
            elif qualifying:
                winner = min(
                    qualifying,
                    key=lambda row: (
                        float(row["recall_median"]) - target,
                        -float(row["qps_median"]),
                    ),
                )
                target_reached = True
                selection = "smallest measured B0 target overshoot"
            else:
                winner = max(
                    candidates,
                    key=lambda row: (
                        float(row["recall_median"]),
                        float(row["qps_median"]),
                    ),
                )
                target_reached = False
                selection = "target not reached; maximum observed B0 recall"
            selected.append(
                {
                    **winner,
                    "target_recall": target,
                    "target_window": TARGET_WINDOW,
                    "target_reached": target_reached,
                    "selection": selection,
                }
            )
    return selected


def write_k10_tex(
    output: Path,
    target_rows: list[dict[str, object]],
    controls: list[dict[str, object]],
    selected_b0: list[dict[str, object]],
) -> None:
    seed_lines = [
        "% Generated by a100_wd_followup/workflow.py; do not edit by hand.",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Workload & $k$-seed QPS & $WD$-seed QPS & Speedup \\\\",
        "\\midrule",
    ]
    for row in target_rows:
        seed_lines.append(
            f"{str(row['workload']).upper()} & {float(row['k_seed_qps']):,.0f} & "
            f"{float(row['wd_seed_qps']):,.0f} & {float(row['wd_over_k_qps']):.2f}$\\times$ \\\\"
        )
    seed_lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "navix_k_vs_wd_seeding.tex").write_text(
        "\n".join(seed_lines) + "\n"
    )

    control_lines = [
        "% Generated by a100_wd_followup/workflow.py; do not edit by hand.",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Workload & Method & Recall & QPS \\\\",
        "\\midrule",
    ]
    for row in controls:
        control_lines.append(
            f"{str(row['workload']).upper()} & {METHOD_LABELS[str(row['method'])]} & "
            f"{float(row['recall_median']):.4f} & {float(row['qps_median']):,.0f} \\\\"
        )
    control_lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "matched_wd_seed_controls.tex").write_text(
        "\n".join(control_lines) + "\n"
    )

    selected_lines = [
        "% Generated by a100_wd_followup/workflow.py; do not edit by hand.",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Workload & Method & Recall & QPS \\\\",
        "\\midrule",
    ]
    for row in selected_b0:
        prefix = "" if bool_value(row["target_reached"]) else "max "
        selected_lines.append(
            f"{str(row['workload']).upper()} & {METHOD_LABELS[str(row['method'])]} & "
            f"{prefix}{float(row['recall_median']):.4f} & {float(row['qps_median']):,.0f} \\\\"
        )
    selected_lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "matched_wd_seed_b0_results.tex").write_text(
        "\n".join(selected_lines) + "\n"
    )


def analyze_k10(root: Path, reference_bundle: Path) -> None:
    summary_path = root / "graph" / "analysis" / "summary_points.csv"
    rows = read_csv(summary_path)
    b0 = [
        row
        for row in rows
        if row["group"] == "b0" and row["phase"] == "throughput"
    ]
    correctness = [row for row in rows if row["group"] == "correctness"]
    if len(b0) != 4 * 6 * 3 or len(correctness) != 4 * 3:
        raise ValueError(
            f"unexpected k=10 matrix: b0={len(b0)}, correctness={len(correctness)}"
        )
    for row in rows:
        width = int(row["search_width"])
        if (
            row["method"] not in METHODS
            or row.get("seed_policy") != "wd"
            or int(row["seed_cap"]) != width * 64
        ):
            raise ValueError(f"invalid matched W*D row: {row}")

    high_resource = [
        {**row, "seed_cap": int(row["seed_cap"])}
        for row in b0
        if int(row["itopk"]) == 512 and int(row["search_width"]) == 2
    ]
    if len(high_resource) != 12:
        raise ValueError(
            f"expected 12 common L=512,W=2 controls, got {len(high_resource)}"
        )

    reference = BundleReader(reference_bundle)
    old_wd = reference.csv(K10_REFERENCE_MEMBERS["wd_b0"])
    old_index = {
        (
            row["workload"],
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
        ): row
        for row in old_wd
        if row["method"] == "navix_reference" and row["group"] == "b0"
    }
    replay: list[dict[str, object]] = []
    for row in b0:
        if row["method"] != "navix_reference":
            continue
        key = (
            row["workload"],
            int(row["itopk"]),
            int(row["search_width"]),
            int(row["max_iterations"]),
        )
        if key not in old_index:
            raise ValueError(f"missing reviewed W*D replay point {key}")
        old = old_index[key]
        recall_slots = (
            abs(float(row["recall_median"]) - float(old["recall_median"]))
            * 10_000
            * 10
        )
        qps_ratio = float(row["qps_median"]) / float(old["qps_median"])
        if recall_slots > 10.000001 or not 0.90 <= qps_ratio <= 1.10:
            raise ValueError(
                f"reviewed NaviX W*D replay drift for {key}: slots={recall_slots}, qps_ratio={qps_ratio}"
            )
        replay.append(
            {
                "workload": key[0],
                "itopk": key[1],
                "search_width": key[2],
                "max_iterations": key[3],
                "reviewed_recall": float(old["recall_median"]),
                "rerun_recall": float(row["recall_median"]),
                "recall_slot_drift": recall_slots,
                "reviewed_qps": float(old["qps_median"]),
                "rerun_qps": float(row["qps_median"]),
                "qps_ratio": qps_ratio,
            }
        )

    target_rows = target_comparison(
        reference.csv(K10_REFERENCE_MEMBERS["k_selected"]),
        reference.csv(K10_REFERENCE_MEMBERS["wd_selected"]),
    )
    selected_b0 = select_b0_target(b0)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "matched_wd_seed_controls.csv", high_resource)
    write_csv(output / "matched_wd_seed_b0_selected.csv", selected_b0)
    write_csv(output / "navix_k_vs_wd_target.csv", target_rows)
    write_csv(output / "navix_wd_replay.csv", replay)
    write_k10_tex(output, target_rows, high_resource, selected_b0)
    payload = {
        "schema_version": 1,
        "experiment": "a100_k10_matched_wd_seed_controls",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "k": 10,
        "max_queries": 2048,
        "native_initialization": "internal_topk + search_width * graph_degree random candidates",
        "passing_seed_rule": "search_width * graph_degree",
        "timing": "passing-seed prepass and seed distances are inside the timed search call",
        "b0_rows": len(b0),
        "control_rows": len(high_resource),
        "selected_b0_rows": len(selected_b0),
        "reference_bundle": str(reference_bundle.resolve()),
        "reference_sha256": sha256(reference_bundle)
        if reference_bundle.is_file()
        else None,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def selected_navix(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return workload_rows(rows, "navix_reference")


def merged_method_rows(
    old: list[dict[str, str]], new: list[dict[str, str]], method: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        dict(row) for row in old if row.get("method") != method
    ]
    rows.extend(dict(row) for row in new if row.get("method") == method)
    return rows


def summary_point(row: dict[str, object]) -> gpu_graph_analysis.SummaryPoint:
    method = str(row["method"])
    seed_policy = str(row.get("seed_policy", "")) or (
        "k" if method == "navix_reference" else "none"
    )
    seed_cap_value = str(row.get("seed_cap", ""))
    seed_cap = (
        int(seed_cap_value)
        if seed_cap_value
        else (100 if seed_policy == "k" else 0)
    )
    return gpu_graph_analysis.SummaryPoint(
        group=str(row["group"]),
        phase=str(row["phase"]),
        workload=str(row["workload"]),
        graph_degree=int(row["graph_degree"]),
        intermediate_graph_degree=int(row["intermediate_graph_degree"]),
        method=method,
        itopk=int(row["itopk"]),
        search_width=int(row["search_width"]),
        max_iterations=int(row["max_iterations"]),
        seed_policy=seed_policy,
        seed_cap=seed_cap,
        repetitions=int(row["repetitions"]),
        shards_per_repetition=int(row["shards_per_repetition"]),
        queries_per_repetition=int(row["queries_per_repetition"]),
        recall_median=float(row["recall_median"]),
        recall_min=float(row["recall_min"]),
        recall_max=float(row["recall_max"]),
        valid_gt_fraction_min=float(row["valid_gt_fraction_min"]),
        qps_median=float(row["qps_median"]),
        qps_min=float(row["qps_min"]),
        qps_max=float(row["qps_max"]),
        seconds_median=float(row["seconds_median"]),
        filter_violations=float(row["filter_violations"]),
        sentinel_errors=float(row["sentinel_errors"]),
        duplicate_output_query_rate_max=float(
            row["duplicate_output_query_rate_max"]
        ),
        underfilled_queries_max=float(row["underfilled_queries_max"]),
        missing_result_slots_max=float(row["missing_result_slots_max"]),
        paper_included=bool_value(row["paper_included"]),
    )


def write_composite_gpu_graph(
    root: Path, reference: BundleReader, output: Path
) -> None:
    """Combine reviewed Base/Retain rows with the new NaviX W*D frontier."""
    output.mkdir(parents=True, exist_ok=False)
    members = {
        "raw_points.csv": K100_REFERENCE_MEMBERS["gpu_raw"],
        "repetition_aggregates.csv": K100_REFERENCE_MEMBERS["gpu_repetitions"],
        "summary_points.csv": K100_REFERENCE_MEMBERS["gpu_summary"],
        "pareto_points.csv": K100_REFERENCE_MEMBERS["gpu_pareto"],
    }
    for filename, old_member in members.items():
        write_csv(
            output / filename,
            merged_method_rows(
                reference.csv(old_member),
                read_csv(root / "frontier" / "analysis" / filename),
                "navix_reference",
            ),
        )

    summary_rows = read_csv(output / "summary_points.csv")
    summaries = [summary_point(row) for row in summary_rows]
    for workload in WORKLOADS:
        gpu_graph_analysis.plot_workload(
            output,
            workload,
            summaries,
            gpu_graph_analysis.PRIMARY_METHODS,
            "gpu_graph_qps_recall",
            "GPU graph search (3-repetition median)",
            100,
        )
    (output / "deep_candidates.json").write_text(
        json.dumps(
            gpu_graph_analysis.deep_candidates(summaries, 0.90), indent=2
        )
        + "\n"
    )

    source_provenance = output / "source_provenance"
    source_provenance.mkdir()
    old_provenance = reference.bytes(K100_REFERENCE_MEMBERS["gpu_provenance"])
    new_provenance_path = root / "frontier" / "analysis" / "provenance.json"
    (source_provenance / "k_seed_base_retain.json").write_bytes(old_provenance)
    shutil.copy2(new_provenance_path, source_provenance / "wd_seed_navix.json")
    old_payload = json.loads(old_provenance)
    new_payload = json.loads(new_provenance_path.read_text())
    for contract in ("timing_contract", "recall_contract"):
        if old_payload.get(contract) != new_payload.get(contract):
            raise ValueError(
                f"GPU graph {contract} changed between composite sources"
            )
    provenance = {
        "schema_version": 1,
        "experiment": "retrieve_workshop_gpu_graph_composite",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "k": 100,
        "max_queries": 2048,
        "target_recall": 0.90,
        "timing_contract": new_payload.get("timing_contract"),
        "recall_contract": new_payload.get("recall_contract"),
        "passing_seed_policy_by_method": {
            "default_cagra": "native initialization (reviewed evidence)",
            "default_cagra_accumulator": "native initialization (reviewed evidence)",
            "navix_reference": "search_width * graph_degree passing IDs",
        },
        "sources": {
            "default_cagra": "source_provenance/k_seed_base_retain.json",
            "default_cagra_accumulator": "source_provenance/k_seed_base_retain.json",
            "navix_reference": "source_provenance/wd_seed_navix.json",
        },
        "composition": "method-wise replacement only; no synthetic measurements",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "retrieve_workshop_gpu_graph_composite",
                "raw_points": len(read_csv(output / "raw_points.csv")),
                "repetition_aggregates": len(
                    read_csv(output / "repetition_aggregates.csv")
                ),
                "summary_points": len(summary_rows),
                "pareto_points": len(read_csv(output / "pareto_points.csv")),
                "correctness_error_total": sum(
                    float(row["filter_violations"])
                    + float(row["sentinel_errors"])
                    for row in summary_rows
                ),
            },
            indent=2,
        )
        + "\n"
    )


def build_composite_gpu_graph(
    root: Path, reference: BundleReader, output: Path
) -> None:
    required = (
        "summary_points.csv",
        "pareto_points.csv",
        "provenance.json",
        "yfcc_gpu_graph_qps_recall.pdf",
    )
    if output.exists():
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise FileExistsError(
                f"incomplete immutable composite GPU graph {output}: missing {missing}"
            )
        return
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(partial)
    try:
        write_composite_gpu_graph(root, reference, partial)
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def create_k100_controls(
    root: Path, data_root: Path, reference_bundle: Path, new_selected: Path
) -> None:
    output = root / "controls"
    contract = output / "state" / "contract.json"
    if contract.exists():
        raise FileExistsError(
            f"immutable controls already initialized: {contract}"
        )
    old = selected_navix(
        BundleReader(reference_bundle).csv(K100_REFERENCE_MEMBERS["selected"])
    )
    new = selected_navix(read_csv(new_selected))
    for group, coordinates in (("paired_old", old), ("paired_new", new)):
        for workload in WORKLOADS:
            coordinate = coordinates[workload]
            itopk = int(coordinate["itopk"])
            width = int(coordinate["search_width"])
            iterations = int(coordinate["max_iterations"])
            paths = dataset_paths(data_root, workload, "throughput", 64)
            searches = [
                search_point(
                    "navix_reference",
                    itopk,
                    width,
                    iterations,
                    k=100,
                    max_queries=2048,
                    seed_policy=policy,
                    graph_degree=paths.graph_degree,
                )
                for policy in ("k", "wd")
            ]
            source = json.loads(paths.manifest.read_text())
            configs: list[dict[str, object]] = []
            workload_root = output / "configs" / group / workload
            workload_root.mkdir(parents=True, exist_ok=False)
            cursor = 0
            for shard_index, shard in enumerate(source["shards"]):
                first, count = (
                    int(shard["first_query"]),
                    int(shard["query_count"]),
                )
                if first != cursor or count <= 0:
                    raise ValueError(
                        f"invalid source shard for {workload}: {shard}"
                    )
                cursor += count
                config = workload_root / f"shard_{shard_index:02d}.json"
                config.write_text(
                    json.dumps(
                        config_payload(
                            workload=workload,
                            phase="throughput",
                            shard=shard,
                            paths=paths,
                            searches=searches,
                            k=100,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                configs.append(
                    {
                        "config": str(config.resolve()),
                        "shard_index": shard_index,
                        "first_query": first,
                        "query_count": count,
                    }
                )
            if cursor != 10_000:
                raise ValueError(f"{workload} controls cover {cursor} queries")
            manifest = {
                "schema_version": 1,
                "experiment": "a100_k100_navix_seed_policy_replay",
                "group": group,
                "workload": workload,
                "k": 100,
                "max_queries": 2048,
                "graph_degree": paths.graph_degree,
                "intermediate_graph_degree": paths.intermediate_graph_degree,
                "expected_queries": 10_000,
                "expected_shards": len(configs),
                "repetitions": 3,
                "source_bitmap_manifest": str(paths.manifest.resolve()),
                "coordinate_origin": "reviewed k-seed winner"
                if group == "paired_old"
                else "new W*D-seed winner",
                "search_points": [
                    {**point_identity(search), "seed_policy": policy}
                    for search, policy in zip(
                        searches, ("k", "wd"), strict=True
                    )
                ],
                "configs": configs,
            }
            (workload_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "k": 100,
                "max_queries": 2048,
                "policies": ["k", "wd"],
                "groups": ["paired_old", "paired_new"],
                "reference_bundle": str(reference_bundle.resolve()),
                "reference_sha256": sha256(reference_bundle)
                if reference_bundle.is_file()
                else None,
                "new_selected": str(new_selected.resolve()),
                "new_selected_sha256": sha256(new_selected),
            },
            indent=2,
        )
        + "\n"
    )


def label_value(label: str, key: str) -> str:
    prefix = f'{key}="'
    for field in label.split("#"):
        if field.startswith(prefix) and field.endswith('"'):
            return field[len(prefix) : -1]
    return ""


def finite(record: dict, key: str, path: Path) -> float:
    if key not in record:
        raise ValueError(f"missing {key} in {path}")
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {path}")
    return value


def analyze_k100_controls(root: Path) -> list[dict[str, object]]:
    control_root = root / "controls"
    repetitions: dict[tuple[object, ...], list[dict[str, float]]] = {}
    for manifest_path in sorted(
        (control_root / "configs").glob("*/*/manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text())
        group, workload = manifest["group"], manifest["workload"]
        expected = {
            (row["seed_policy"], int(row["navix_seed_cap"]))
            for row in manifest["search_points"]
        }
        if expected != {
            ("k", 100),
            ("wd", int(manifest["search_points"][0]["search_width"]) * 64),
        }:
            raise ValueError(
                f"invalid control seed matrix in {manifest_path}: {expected}"
            )
        by_repetition: dict[tuple[int, str, int], list[dict[str, float]]] = {}
        for shard in manifest["configs"]:
            raw_path = (
                control_root
                / "raw"
                / group
                / workload
                / f"shard_{int(shard['shard_index']):02d}.json"
            )
            payload = json.loads(raw_path.read_text())
            rows = [
                row
                for row in payload.get("benchmarks", [])
                if row.get("run_type") == "iteration"
            ]
            if len(rows) != 6 or any(
                row.get("error_occurred") or row.get("skipped") for row in rows
            ):
                raise ValueError(f"incomplete control output {raw_path}")
            for record in rows:
                if (
                    label_value(str(record.get("label", "")), "bitmap_method")
                    != "navix_reference"
                ):
                    raise ValueError(f"non-NaviX control in {raw_path}")
                cap = round(finite(record, "navix_seed_cap", raw_path))
                policy = "k" if cap == 100 else "wd"
                width = round(finite(record, "search_width", raw_path))
                if cap != (100 if policy == "k" else width * 64):
                    raise ValueError(
                        f"invalid runtime seed cap {cap} in {raw_path}"
                    )
                errors = sum(
                    finite(record, key, raw_path)
                    for key in (
                        "FilterViolations",
                        "InvalidSentinelErrors",
                        "SentinelOrderErrors",
                        "InvalidSentinelDistanceErrors",
                        "DuplicateOutputQueries",
                    )
                )
                if errors != 0:
                    raise ValueError(
                        f"correctness error total {errors} in {raw_path}"
                    )
                repetition = int(record["repetition_index"])
                queries = round(finite(record, "n_queries", raw_path))
                qps = finite(record, "items_per_second", raw_path)
                by_repetition.setdefault((repetition, policy, cap), []).append(
                    {
                        "queries": queries,
                        "seconds": queries / qps,
                        "matches": finite(record, "ValidGTRecall", raw_path)
                        * finite(record, "ValidGTFraction", raw_path)
                        * queries,
                        "valid": finite(record, "ValidGTFraction", raw_path)
                        * queries,
                    }
                )
        for (repetition, policy, cap), members in by_repetition.items():
            if (
                len(members) != int(manifest["expected_shards"])
                or sum(row["queries"] for row in members) != 10_000
            ):
                raise ValueError(
                    f"incomplete {group}/{workload}/{policy}/rep{repetition}"
                )
            repetitions.setdefault((group, workload, policy, cap), []).append(
                {
                    "recall": sum(row["matches"] for row in members)
                    / sum(row["valid"] for row in members),
                    "qps": 10_000 / sum(row["seconds"] for row in members),
                }
            )
    summaries: list[dict[str, object]] = []
    for (group, workload, policy, cap), members in sorted(repetitions.items()):
        if len(members) != 3:
            raise ValueError(
                f"expected three control repetitions for {group}/{workload}/{policy}"
            )
        summaries.append(
            {
                "group": group,
                "workload": workload,
                "seed_policy": policy,
                "seed_cap": cap,
                "recall_median": statistics.median(
                    row["recall"] for row in members
                ),
                "recall_min": min(row["recall"] for row in members),
                "qps_median": statistics.median(row["qps"] for row in members),
                "qps_min": min(row["qps"] for row in members),
                "qps_max": max(row["qps"] for row in members),
            }
        )
    if len(summaries) != 16:
        raise ValueError(
            f"expected 16 paired control rows, got {len(summaries)}"
        )
    return summaries


def write_k100_tex(output: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "% Generated by a100_wd_followup/workflow.py; do not edit by hand.",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Workload & Method & Recall & QPS \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        recall_prefix = "" if bool_value(row["target_reached"]) else "max "
        lines.append(
            f"{str(row['workload']).upper()} & {method} & {recall_prefix}{float(row['recall_median']):.4f} & "
            f"{float(row['qps_median']):,.0f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "fixed_recall_k100_wd_results.tex").write_text(
        "\n".join(lines) + "\n"
    )


def build_composite_matched_recall(
    root: Path,
    reference_bundle: Path,
    reference: BundleReader,
    combined: list[dict[str, object]],
    output: Path,
) -> None:
    required = (
        "selected_points.csv",
        "final_summary.csv",
        "measurements.csv",
        "provenance.json",
        "gpu_matched_recall_k100.pdf",
    )
    if output.exists():
        missing = [
            name
            for name in required
            if not (output / "analysis" / name).is_file()
        ]
        if missing:
            raise FileExistsError(
                f"incomplete immutable composite matched-recall view {output}: missing {missing}"
            )
        return
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(partial)
    composite = partial / "analysis"
    composite.mkdir(parents=True)
    try:
        write_csv(composite / "selected_points.csv", combined)
        write_csv(
            composite / "final_summary.csv",
            merged_method_rows(
                reference.csv(K100_REFERENCE_MEMBERS["final_summary"]),
                read_csv(
                    root / "matched_recall" / "analysis" / "final_summary.csv"
                ),
                "navix_reference",
            ),
        )
        write_csv(
            composite / "measurements.csv",
            merged_method_rows(
                reference.csv(K100_REFERENCE_MEMBERS["measurements"]),
                read_csv(
                    root / "matched_recall" / "analysis" / "measurements.csv"
                ),
                "navix_reference",
            ),
        )
        composite_provenance = json.loads(
            (
                root / "matched_recall" / "analysis" / "provenance.json"
            ).read_text()
        )
        composite_provenance["methods"] = [
            "default_cagra",
            "default_cagra_accumulator",
            "navix_reference",
        ]
        composite_provenance["composite_evidence"] = {
            "default_cagra": "reviewed k=100 reference bundle",
            "default_cagra_accumulator": "reviewed k=100 reference bundle",
            "navix_reference": "new W*D matched-recall run",
            "reference_bundle": str(reference_bundle.resolve()),
            "reference_sha256": sha256(reference_bundle)
            if reference_bundle.is_file()
            else None,
        }
        (composite / "provenance.json").write_text(
            json.dumps(composite_provenance, indent=2) + "\n"
        )
        subprocess.run(
            [
                sys.executable,
                str(WORKSHOP / "a100_k100" / "matched_table.py"),
                "--result-root",
                str(partial),
            ],
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def analyze_k100(root: Path, reference_bundle: Path) -> None:
    frontier = read_csv(root / "frontier" / "analysis" / "summary_points.csv")
    b0 = [
        row
        for row in frontier
        if row["group"] == "b0" and row["method"] == "navix_reference"
    ]
    correctness = [row for row in frontier if row["group"] == "correctness"]
    if len(b0) != 24 or len(correctness) != 4:
        raise ValueError(
            f"unexpected k=100 NaviX frontier: b0={len(b0)}, correctness={len(correctness)}"
        )
    for row in frontier:
        if (
            row.get("seed_policy") != "wd"
            or int(row["seed_cap"]) != int(row["search_width"]) * 64
        ):
            raise ValueError(f"invalid k=100 W*D frontier row: {row}")

    new_selected = selected_navix(
        read_csv(root / "matched_recall" / "analysis" / "selected_points.csv")
    )
    reference = BundleReader(reference_bundle)
    old_all = reference.csv(K100_REFERENCE_MEMBERS["selected"])
    old_navix = selected_navix(old_all)
    combined = [
        dict(row) for row in old_all if row["method"] != "navix_reference"
    ]
    comparison: list[dict[str, object]] = []
    for workload in WORKLOADS:
        old, new = old_navix[workload], new_selected[workload]
        combined.append(dict(new))
        comparison.append(
            {
                "workload": workload,
                "target_recall": TARGETS[workload],
                "k_seed_recall": float(old["recall_median"]),
                "k_seed_qps": float(old["qps_median"]),
                "wd_seed_recall": float(new["recall_median"]),
                "wd_seed_qps": float(new["qps_median"]),
                "wd_over_k_qps": float(new["qps_median"])
                / float(old["qps_median"]),
            }
        )
    order = {workload: index for index, workload in enumerate(WORKLOADS)}
    method_order = {
        "default_cagra": 0,
        "default_cagra_accumulator": 1,
        "navix_reference": 2,
    }
    combined.sort(
        key=lambda row: (order[row["workload"]], method_order[row["method"]])
    )
    controls = analyze_k100_controls(root)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "matched_recall_k100_wd_table.csv", combined)
    write_csv(output / "navix_k_vs_wd_target.csv", comparison)
    write_csv(output / "paired_seed_policy_controls.csv", controls)
    write_k100_tex(output, combined)

    # Rebuild paper-facing views from preserved Base/Retain evidence and the measured NaviX W*D
    # campaign, so no old k-seed plot remains under a current-looking filename.
    build_composite_matched_recall(
        root,
        reference_bundle,
        reference,
        combined,
        output / "composite_matched_recall",
    )
    build_composite_gpu_graph(root, reference, output / "composite_gpu_graph")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "a100_k100_navix_wd_frontier",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "k": 100,
                "max_queries": 2048,
                "passing_seed_rule": "search_width * graph_degree",
                "frontier_rows": len(b0),
                "selected_navix_rows": len(new_selected),
                "paired_control_rows": len(controls),
                "reference_bundle": str(reference_bundle.resolve()),
                "reference_sha256": sha256(reference_bundle)
                if reference_bundle.is_file()
                else None,
            },
            indent=2,
        )
        + "\n"
    )


def safe_extract_bundle(archive_path: Path, destination: Path) -> Path:
    if archive_path.is_dir():
        roots = [archive_path]
    else:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination, filter="data")
        roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError(f"expected one bundle root, got {roots}")
    return roots[0]


def replace_method_rows(old_path: Path, new_path: Path, method: str) -> None:
    write_csv(
        old_path,
        merged_method_rows(read_csv(old_path), read_csv(new_path), method),
    )


def write_hash_manifest(root: Path) -> None:
    manifest_path = root / "final_hash_manifest.json"
    rows = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": rows}, indent=2) + "\n"
    )


def bundle_k10(root: Path, reference_bundle: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable bundle {output}")
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    shutil.copytree(root / "graph", temporary / "graph")
    shutil.copytree(root / "analysis", temporary / "analysis")
    (temporary / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "a100_k10_matched_wd_seed_controls",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "reference_bundle": str(reference_bundle.resolve()),
                "reference_sha256": sha256(reference_bundle)
                if reference_bundle.is_file()
                else None,
            },
            indent=2,
        )
        + "\n"
    )
    write_hash_manifest(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)


def bundle_k100(root: Path, reference_bundle: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable bundle {output}")
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch_name:
        reference_root = safe_extract_bundle(
            reference_bundle, Path(scratch_name)
        )
        shutil.copytree(reference_root, temporary)
    seed_ablation = temporary / "seed_ablation"
    seed_ablation.mkdir(exist_ok=True)
    shutil.copytree(
        temporary / "matched_recall", seed_ablation / "k_seed_matched_recall"
    )
    shutil.copytree(
        temporary / "gpu_graph", seed_ablation / "k_seed_gpu_graph"
    )
    shutil.rmtree(temporary / "gpu_graph")
    shutil.copytree(
        root / "analysis" / "composite_gpu_graph", temporary / "gpu_graph"
    )
    shutil.rmtree(temporary / "matched_recall")
    shutil.copytree(
        root / "analysis" / "composite_matched_recall" / "analysis",
        temporary / "matched_recall",
    )
    shutil.copy2(
        root / "analysis" / "fixed_recall_k100_wd_results.tex",
        temporary / "matched_recall" / "fixed_recall_k100_wd_results.tex",
    )
    followup = seed_ablation / "wd_followup"
    followup.mkdir()
    for name in ("frontier", "matched_recall", "controls", "analysis"):
        source = root / name
        if source.exists():
            shutil.copytree(source, followup / name)
    manifest_path = temporary / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"schema_version": 1}
    )
    manifest["wd_followup"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "k": 100,
        "max_queries": 2048,
        "navix_seed_policy": "search_width * graph_degree",
        "reference_bundle": str(reference_bundle.resolve()),
        "reference_sha256": sha256(reference_bundle)
        if reference_bundle.is_file()
        else None,
        "replacement": "paper-facing graph and matched-recall views combine reviewed Base/Retain with new NaviX W*D evidence; original k-seed views and exact-search evidence are preserved",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_hash_manifest(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("analyze-k10", "analyze-k100"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--reference-bundle", type=Path, required=True)

    controls = subparsers.add_parser("create-k100-controls")
    controls.add_argument("--root", type=Path, required=True)
    controls.add_argument("--data-root", type=Path, required=True)
    controls.add_argument("--reference-bundle", type=Path, required=True)
    controls.add_argument("--new-selected", type=Path, required=True)

    for command in ("bundle-k10", "bundle-k100"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--reference-bundle", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "analyze-k10":
        analyze_k10(args.root.resolve(), args.reference_bundle.resolve())
    elif args.command == "analyze-k100":
        analyze_k100(args.root.resolve(), args.reference_bundle.resolve())
    elif args.command == "create-k100-controls":
        create_k100_controls(
            args.root.resolve(),
            args.data_root.resolve(),
            args.reference_bundle.resolve(),
            args.new_selected.resolve(),
        )
    elif args.command == "bundle-k10":
        bundle_k10(
            args.root.resolve(),
            args.reference_bundle.resolve(),
            args.output.resolve(),
        )
    elif args.command == "bundle-k100":
        bundle_k100(
            args.root.resolve(),
            args.reference_bundle.resolve(),
            args.output.resolve(),
        )


if __name__ == "__main__":
    main()
