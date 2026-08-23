#!/usr/bin/env python3
"""Validate and summarize the RETRIEVE GPU resource/dynamic-work experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")
RESOURCE_METHOD = {
    "default_cagra": "base",
    "default_cagra_accumulator": "retain",
    "navix_reference": "navix",
}
LATEX_METHOD = {
    "default_cagra": r"\methodbase{}",
    "default_cagra_accumulator": r"\methodretain{}",
    "navix_reference": r"\methodnavix{}",
}
WORKLOAD_LABEL = {"yfcc": "YFCC", "em": "EM", "emis": "EMIS", "r": "R"}
SCHEMA_VERSION = 9
RESOURCE_PREFIX = "CAGRA_KERNEL_RESOURCES "


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def load_resources(path: Path, graph_degree: int) -> dict[str, dict]:
    grouped: dict[str, set[tuple]] = {name: set() for name in RESOURCE_METHOD.values()}
    values: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        position = line.find(RESOURCE_PREFIX)
        if position < 0:
            continue
        record = json.loads(line[position + len(RESOURCE_PREFIX) :])
        if (
            bool(record.get("diagnostics"))
            or int(record.get("graph_degree", -1)) != graph_degree
            or int(record.get("itopk", -1)) != 64
            or int(record.get("search_width", -1)) != 1
        ):
            continue
        method = str(record.get("method"))
        if method not in grouped:
            continue
        key = (
            method,
            int(record["threads_per_cta"]),
            int(record["dynamic_smem_bytes"]),
            int(record["static_smem_bytes"]),
            int(record["registers_per_thread"]),
            int(record["active_ctas_per_sm"]),
        )
        grouped[method].add(key)
        values[key] = record
    result: dict[str, dict] = {}
    for method, keys in grouped.items():
        if len(keys) != 1:
            raise ValueError(f"expected one production resource tuple for {method} in {path}, got {keys}")
        result[method] = values[next(iter(keys))]
    return result


def load_raw_correctness(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    result: dict[str, dict] = {}
    for record in payload.get("benchmarks", []):
        if record.get("run_type") != "iteration":
            continue
        method = label_value(str(record.get("label", "")), "bitmap_method")
        if method not in METHODS:
            continue
        if method in result:
            raise ValueError(f"duplicate ordinary correctness record for {method} in {path}")
        if int(round(float(record.get("n_queries", -1)))) != 1_000:
            raise ValueError(f"{method} did not cover 1,000 queries in {path}")
        if int(round(float(record.get("itopk", -1)))) != 64:
            raise ValueError(f"{method} does not use L=64 in {path}")
        if int(round(float(record.get("search_width", -1)))) != 1:
            raise ValueError(f"{method} does not use W=1 in {path}")
        if int(round(float(record.get("max_iterations", -1)))) != 0:
            raise ValueError(f"{method} does not use B0 in {path}")
        sentinel_errors = sum(
            float(record.get(key, 0.0))
            for key in (
                "InvalidSentinelErrors",
                "SentinelOrderErrors",
            )
        )
        if (
            float(record.get("ValidGTFraction", 0.0)) != 1.0
            or float(record.get("FilterViolations", -1.0)) != 0.0
            or sentinel_errors != 0.0
        ):
            raise ValueError(f"correctness gate failed for {method} in {path}")
        if method != "default_cagra" and float(record.get("DuplicateOutputQueries", -1.0)) != 0.0:
            raise ValueError(f"duplicate outputs for {method} in {path}")
        result[method] = record
    if set(result) != set(METHODS):
        raise ValueError(f"missing ordinary correctness rows in {path}: {set(METHODS) - set(result)}")
    return result


def load_diagnostic(path: Path, manifest_path: Path, method: str) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    if (
        int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or int(manifest.get("num_queries", -1)) != 1_000
        or int(manifest.get("itopk", -1)) != 64
        or int(manifest.get("search_width", -1)) != 1
        or int(manifest.get("configured_max_iterations", -1)) != 0
        or bool(manifest.get("timing_valid", True))
        or str(manifest.get("variant")) != method
    ):
        raise ValueError(f"diagnostic manifest contract failed: {manifest_path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1_000 or [int(row["query_id"]) for row in rows] != list(range(1_000)):
        raise ValueError(f"diagnostic query coverage failed: {path}")
    required = {
        "graph_rows_read",
        "predicate_probes",
        "distance_evaluations",
        "passing_admissions",
        "seed_inspected_units",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"schema-9 work columns missing from {path}")
    return rows, manifest


def mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"invalid {field} values")
    return statistics.fmean(values)


def summarize(result_root: Path) -> list[dict]:
    rows_by_pair: dict[tuple[str, str], list[dict]] = {}
    manifests: dict[tuple[str, str], dict] = {}
    resources_by_workload: dict[str, dict[str, dict]] = {}
    raw_by_workload: dict[str, dict[str, dict]] = {}
    config_manifest_path = result_root / "configs" / "manifest.json"
    config_manifest = json.loads(config_manifest_path.read_text())
    contracts = config_manifest.get("dataset_contracts", {})
    if set(contracts) != set(WORKLOADS):
        raise ValueError(f"resource manifest lacks dataset contracts: {config_manifest_path}")
    for workload in WORKLOADS:
        graph_degree = int(contracts[workload]["graph_degree"])
        resources_by_workload[workload] = load_resources(
            result_root / "resources" / f"{workload}.log", graph_degree
        )
        raw_by_workload[workload] = load_raw_correctness(
            result_root / "raw" / "resources" / f"{workload}.json"
        )
        for method in METHODS:
            directory = result_root / "diagnostics" / workload / method
            diagnostic_rows, manifest = load_diagnostic(
                directory / "query_summary.csv", directory / "manifest.json", method
            )
            rows_by_pair[(workload, method)] = diagnostic_rows
            manifests[(workload, method)] = manifest

    for workload in WORKLOADS:
        base = rows_by_pair[(workload, "default_cagra")]
        retain = rows_by_pair[(workload, "default_cagra_accumulator")]
        for query, (base_row, retain_row) in enumerate(zip(base, retain, strict=True)):
            for field in ("graph_rows_read", "distance_evaluations", "passing_admissions"):
                if int(base_row[field]) != int(retain_row[field]):
                    raise ValueError(f"Base/Retain {field} mismatch for {workload} query {query}")

        navix = rows_by_pair[(workload, "navix_reference")]
        for query, row in enumerate(navix):
            expected_rows = (
                int(row["navix_one_hop_parents"])
                + int(row["navix_directed_parents"])
                + int(row["navix_blind_parents"])
                + int(row["navix_bridge_rows_loaded"])
            )
            expected_probes = int(row["navix_first_hop_checks"]) + int(
                row["navix_second_hop_checks"]
            )
            if int(row["graph_rows_read"]) != expected_rows:
                raise ValueError(f"NaviX graph-row invariant failed for {workload} query {query}")
            if int(row["predicate_probes"]) != expected_probes:
                raise ValueError(f"NaviX predicate-probe invariant failed for {workload} query {query}")
            if int(row["passing_admissions"]) != int(row["navix_admitted_candidates"]):
                raise ValueError(f"NaviX admission invariant failed for {workload} query {query}")

        base_resource = resources_by_workload[workload]["base"]
        retain_resource = resources_by_workload[workload]["retain"]
        if int(retain_resource["dynamic_smem_bytes"]) - int(base_resource["dynamic_smem_bytes"]) != 84:
            raise ValueError(f"Retain does not add exactly 84 dynamic-shared bytes for {workload}")

    summary: list[dict] = []
    for workload in WORKLOADS:
        base_smem = int(resources_by_workload[workload]["base"]["dynamic_smem_bytes"])
        for method in METHODS:
            diagnostic_rows = rows_by_pair[(workload, method)]
            resource = resources_by_workload[workload][RESOURCE_METHOD[method]]
            diagnostic_recall = mean(diagnostic_rows, "recall")
            raw_recall = float(raw_by_workload[workload][method]["ValidGTRecall"])
            if abs(diagnostic_recall - raw_recall) > 1e-5:
                raise ValueError(
                    f"diagnostic/ordinary recall mismatch for {workload}/{method}: "
                    f"{diagnostic_recall} vs {raw_recall}"
                )
            summary.append(
                {
                    "workload": workload,
                    "method": method,
                    "queries": 1_000,
                    "itopk": 64,
                    "search_width": 1,
                    "max_iterations": 0,
                    "recall": diagnostic_recall,
                    "graph_rows_per_query": mean(diagnostic_rows, "graph_rows_read"),
                    "seed_bitmap_words_per_query": mean(diagnostic_rows, "seed_inspected_units"),
                    "bitmap_probes_per_query": mean(diagnostic_rows, "predicate_probes"),
                    "distance_evaluations_per_query": mean(
                        diagnostic_rows, "distance_evaluations"
                    ),
                    "passing_admissions_per_query": mean(
                        diagnostic_rows, "passing_admissions"
                    ),
                    "threads_per_cta": int(resource["threads_per_cta"]),
                    "registers_per_thread": int(resource["registers_per_thread"]),
                    "dynamic_smem_bytes": int(resource["dynamic_smem_bytes"]),
                    "dynamic_smem_delta_from_base": int(resource["dynamic_smem_bytes"])
                    - base_smem,
                    "active_ctas_per_sm": int(resource["active_ctas_per_sm"]),
                }
            )
    return summary


def format_count(value: float) -> str:
    return f"{value:,.1f}"


def write_outputs(result_root: Path, output: Path, summary: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "gpu_resource_work.csv"
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    provenance = result_root / "provenance" / "run.json"
    if not provenance.is_file():
        raise FileNotFoundError(f"missing executable/source provenance: {provenance}")
    evidence_files = sorted(
        path
        for pattern in (
            "configs/**/*.json",
            "raw/resources/*.json",
            "resources/*.log",
            "diagnostics/*/*/manifest.json",
            "diagnostics/*/*/query_summary.csv",
            "provenance/run.json",
        )
        for path in result_root.glob(pattern)
        if path.is_file()
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "diagnostic_schema_version": SCHEMA_VERSION,
        "configuration": {"queries": 1_000, "itopk": 64, "search_width": 1, "max_iterations": 0},
        "rows": summary,
        "evidence": [
            {
                "path": str(path.resolve()),
                "relative_to_result_root": str(path.relative_to(result_root)),
                "sha256": sha256(path),
            }
            for path in evidence_files
        ],
    }
    (output / "gpu_resource_work.json").write_text(json.dumps(payload, indent=2) + "\n")

    latex_lines: list[str] = []
    for row in summary:
        delta = int(row["dynamic_smem_delta_from_base"])
        smem = f"{int(row['dynamic_smem_bytes']):,}"
        if delta:
            smem += f" ({delta:+,})"
        latex_lines.append(
            " & ".join(
                (
                    WORKLOAD_LABEL[str(row["workload"])],
                    LATEX_METHOD[str(row["method"])],
                    format_count(float(row["graph_rows_per_query"])),
                    format_count(float(row["seed_bitmap_words_per_query"])),
                    format_count(float(row["bitmap_probes_per_query"])),
                    format_count(float(row["distance_evaluations_per_query"])),
                    format_count(float(row["passing_admissions_per_query"])),
                    str(int(row["threads_per_cta"])),
                    str(int(row["registers_per_thread"])),
                    smem,
                    str(int(row["active_ctas_per_sm"])),
                )
            )
            + r" \\"
        )
    # booktabs rules must be seen directly by TeX's alignment scanner.  A
    # \bottomrule placed after an \input of row fragments is separated from
    # the final \cr by the input-group boundary and triggers "Misplaced
    # \noalign".  Keep the closing rule in the generated fragment instead.
    latex_lines.append(r"\bottomrule")
    (output / "gpu_resource_work.tex").write_text("\n".join(latex_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    output = args.output.resolve() if args.output else result_root / "analysis"
    summary = summarize(result_root)
    write_outputs(result_root, output, summary)
    print(output)


if __name__ == "__main__":
    main()
