#!/usr/bin/env python3
"""Create a hash-bound, GPU-only paper-results bundle from one A100 run root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree_files(source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    if not source.is_dir():
        return copied
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def json_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def validate_max_queries_contract(root: Path, profile: dict) -> tuple[int, list[Path]]:
    expected = int(profile.get("max_queries", 512))
    if expected != 2_048:
        raise ValueError(f"A100 bundle requires max_queries=2048, found {expected}")
    contracts = [
        root / "gpu_graph/provenance/run.json",
        root / "matched_recall/provenance/run.json",
        root / "resource_work/provenance/run.json",
        root / "mechanism_diagnostics/provenance/run.json",
        root / "maxq_gate/provenance/run.json",
        root / "per_query_latency/provenance/run.json",
    ]
    observed = (
        int(json_payload(contracts[0])["fixed_contract"]["max_queries"]),
        int(json_payload(contracts[1])["fixed_contract"]["max_queries"]),
        int(json_payload(contracts[2])["contract"]["max_queries"]),
        int(json_payload(contracts[3])["fixed_contract"]["max_queries"]),
        int(json_payload(contracts[4])["fixed_contract"]["max_queries"]),
    )
    if any(value != expected for value in observed):
        raise ValueError(f"mixed max_queries contracts in A100 run: {observed}")
    latency_provenance = json_payload(contracts[5])
    latency_contract = latency_provenance["contract"]
    if (
        int(latency_provenance.get("schema_version", -1)) != 2
        or int(latency_contract.get("k", -1)) != 10
        or int(latency_contract["graph_source_max_queries"]) != expected
        or int(latency_contract["serialized_max_queries"]) != 1
        or int(latency_contract["queries_per_call"]) != 1
        or latency_contract.get("output_set_semantics")
        != "distinct_valid_output_ids_v1"
    ):
        raise ValueError(f"invalid serialized-latency contract: {latency_contract}")
    resource = json_payload(root / "resource_work/analysis/gpu_resource_work.json")
    mechanism = json_payload(
        root / "mechanism_diagnostics/analysis/mechanism_summary.json"
    )
    gpu_analysis = json_payload(root / "gpu_graph/analysis/provenance.json")
    matched_analysis = json_payload(root / "matched_recall/analysis/provenance.json")
    latency_summary = json_payload(root / "per_query_latency/analysis/latency_summary.json")
    if (
        int(resource["configuration"]["max_queries"]) != expected
        or int(mechanism["max_queries"]) != expected
        or int(gpu_analysis["max_queries"]) != expected
        or int(matched_analysis["max_queries"]) != expected
    ):
        raise ValueError("analyzed evidence uses the wrong max_queries")
    latency_measurement = latency_summary.get("measurement_contract", {})
    if (
        latency_summary.get("status") != "PASS"
        or int(latency_measurement.get("k", 0)) != 10
        or int(latency_measurement.get("source_max_queries", 0)) != expected
        or int(latency_measurement.get("serialized_max_queries", 0)) != 1
        or int(latency_measurement.get("queries_per_search_call", 0)) != 1
        or int(latency_measurement.get("complete_passes", 0)) != 3
    ):
        raise ValueError("serialized-latency analysis did not pass its frozen contract")
    return expected, contracts


METHODS = {
    "default_cagra": ("CAGRA-Base", "#1f77b4", "o"),
    "default_cagra_accumulator": ("CAGRA-Retain", "#d62728", "s"),
    "navix_reference": ("CAGRA-NaviX", "#2ca02c", "^"),
}
WORKLOAD_LABEL = {
    "yfcc": "YFCC-10M",
    "em": "ArXiv-large EM",
    "emis": "ArXiv-large EMIS",
    "r": "ArXiv-large R",
}


def truth(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def pareto(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    best_qps = -1.0
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["recall_median"]),
            float(item["qps_median"]),
        ),
        reverse=True,
    ):
        if float(row["qps_median"]) > best_qps:
            result.append(row)
            best_qps = float(row["qps_median"])
    return sorted(result, key=lambda item: float(item["recall_median"]))


def write_gpu_plot(
    output: Path,
    b0: list[dict[str, str]],
    selected: list[dict[str, str]],
    exact: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(
        1, 4, figsize=(16.0, 3.7), constrained_layout=True
    )
    handles = []
    labels = []
    for axis, workload in zip(axes, ("yfcc", "em", "emis", "r"), strict=True):
        x_values: list[float] = []
        for method, (label, color, marker) in METHODS.items():
            local = pareto(
                [
                    row
                    for row in b0
                    if row["phase"] == "throughput"
                    and row["workload"] == workload
                    and row["method"] == method
                    and truth(row["paper_included"])
                ]
            )
            if local:
                xs = [float(row["recall_median"]) for row in local]
                ys = [float(row["qps_median"]) for row in local]
                line = axis.plot(
                    xs,
                    ys,
                    color=color,
                    marker=marker,
                    linewidth=1.4,
                    markersize=4,
                    label=label,
                )[0]
                if label not in labels:
                    handles.append(line)
                    labels.append(label)
                x_values.extend(xs)
            target_rows = [
                row
                for row in selected
                if row["workload"] == workload and row["method"] == method
            ]
            for row in target_rows:
                axis.scatter(
                    float(row["recall_median"]),
                    float(row["qps_median"]),
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    s=72,
                    linewidths=1.6,
                    zorder=5,
                )
                x_values.append(float(row["recall_median"]))
        exact_rows = [
            row
            for row in exact
            if row["workload"] == workload and row["phase"] == "throughput"
        ]
        for row in exact_rows:
            exact_recall = float(row["native_l2_cutoff_recall"])
            point = axis.scatter(
                exact_recall,
                float(row["median_qps"]),
                color="black",
                marker="x",
                s=46,
                label="Masked exact scan",
            )
            if "Masked exact scan" not in labels:
                handles.append(point)
                labels.append("Masked exact scan")
            x_values.append(exact_recall)
        if x_values:
            low = min(x_values)
            axis.set_xlim(max(0.0, low - max(0.01, 0.04 * (1.0 - low))), 1.005)
        axis.set_ylim(bottom=0)
        axis.set_title(WORKLOAD_LABEL[workload])
        axis.set_xlabel("Recall@10")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Queries/s")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"gpu_qps_recall_a100.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_headline_tex(output: Path, selected: list[dict[str, str]]) -> None:
    lines = []
    for workload in ("yfcc", "em", "emis", "r"):
        rows = {
            row["method"]: row
            for row in selected
            if row["workload"] == workload
        }
        for method in METHODS:
            if method not in rows:
                raise ValueError(
                    f"missing matched point for {workload}/{method}"
                )
        cells = []
        for method in METHODS:
            row = rows[method]
            cells.append(
                f"{float(row['recall_median']):.4f} ({float(row['qps_median']):,.0f}; "
                f"$L={int(row['itopk'])},W={int(row['search_width'])},I={int(row['max_iterations'])}$)"
            )
        lines.append(
            f"{WORKLOAD_LABEL[workload]} & " + " & ".join(cells) + r" \\"
        )
    (output / "gpu_matched_recall_rows.tex").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="immutable bundle directory; defaults to RUN_ROOT/paper_gpu_bundle",
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    profile_payload = json.loads(args.profile.read_text())
    max_queries, contract_paths = validate_max_queries_contract(root, profile_payload)
    output = args.output.resolve() if args.output else root / "paper_gpu_bundle"
    if output.exists():
        raise FileExistsError(
            f"refusing to replace immutable bundle: {output}"
        )

    inputs = {
        "b0": root / "gpu_graph" / "analysis",
        "matched_recall": root / "matched_recall" / "analysis",
        "exact_scan": root / "exact_bitmap" / "analysis",
        "resource_work": root / "resource_work" / "analysis",
        "mechanism_diagnostics": root / "mechanism_diagnostics" / "analysis",
        "dataset_stats": root / "dataset_stats",
        "maxq_gate": root / "maxq_gate" / "analysis",
        "preflight": root / "provenance",
        "per_query_latency": root / "per_query_latency" / "analysis",
    }
    required = (
        inputs["b0"] / "summary_points.csv",
        inputs["matched_recall"] / "selected_points.csv",
        inputs["exact_scan"] / "exact_summary.csv",
        inputs["resource_work"] / "gpu_resource_work.csv",
        inputs["mechanism_diagnostics"] / "mechanism_summary.json",
        inputs["dataset_stats"] / "workload_selectivity_summary.json",
        inputs["maxq_gate"] / "max_queries_gate_summary.json",
        inputs["preflight"] / "a100_preflight.json",
        inputs["per_query_latency"] / "latency_summary.csv",
        inputs["per_query_latency"] / "latency_summary.json",
        inputs["per_query_latency"] / "per_query_latency_cdf.pdf",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "bundle inputs are incomplete:\n" + "\n".join(missing)
        )
    output.mkdir(parents=True)

    copied: list[Path] = []
    for name, source in inputs.items():
        copied.extend(copy_tree_files(source, output / name))
    for path in contract_paths:
        target = output / "run_contracts" / path.parent.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)

    selected = csv_rows(required[1])
    exact = csv_rows(required[2])
    headline = {
        "schema_version": 1,
        "matched_recall_points": selected,
        "exact_scan_points": exact,
        "note": (
            "CPU FAISS-NaviX/ACORN measurements are deliberately excluded. ArXiv-large GPU "
            "numbers are not directly comparable with the paper's old ArXiv-medium CPU rows."
        ),
    }
    (output / "gpu_headline_inputs.json").write_text(
        json.dumps(headline, indent=2) + "\n"
    )
    copied.append(output / "gpu_headline_inputs.json")
    b0 = csv_rows(required[0])
    write_gpu_plot(output, b0, selected, exact)
    write_headline_tex(output, selected)
    copied.extend(
        output / name
        for name in (
            "gpu_qps_recall_a100.png",
            "gpu_qps_recall_a100.pdf",
            "gpu_matched_recall_rows.tex",
        )
    )

    claims = {
        "b0_qps_recall_and_seed_controls": "b0/summary_points.csv",
        "matched_recall_headlines_and_parameters": "matched_recall/selected_points.csv",
        "masked_exact_scan": "exact_scan/exact_summary.csv",
        "cuda_resources_and_dynamic_work": "resource_work/gpu_resource_work.csv",
        "yfcc_gt_seen_retention_and_navix_failure": "mechanism_diagnostics/mechanism_summary.json",
        "workload_cardinality_and_selectivity": "dataset_stats/workload_selectivity_summary.json",
        "max_queries_2048_memory_and_scheduling_gate": "maxq_gate/max_queries_gate_summary.json",
        "hardware_and_run_identity": "preflight/a100_preflight.json",
        "serialized_single_query_latency": "per_query_latency/latency_summary.csv",
    }
    (output / "claim_to_source.json").write_text(
        json.dumps(claims, indent=2) + "\n"
    )
    copied.append(output / "claim_to_source.json")
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_payload,
        "execution_contract": {
            "k": 10,
            "max_queries": max_queries,
            "throughput_shards": [2048, 2048, 2048, 2048, 1808],
            "throughput": {
                "max_queries": max_queries,
                "shards": [2048, 2048, 2048, 2048, 1808],
            },
            "serialized_latency": {
                "k": 10,
                "max_queries": 1,
                "queries_per_search_call": 1,
                "complete_passes": 3,
                "latency": "host API entry through synchronized GPU completion",
            },
        },
        "run_root": str(root),
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(copied))
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
