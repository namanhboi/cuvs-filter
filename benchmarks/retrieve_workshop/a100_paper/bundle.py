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
            point = axis.scatter(
                1.0,
                float(row["median_qps"]),
                color="black",
                marker="x",
                s=46,
                label="Masked exact scan",
            )
            if "Masked exact scan" not in labels:
                handles.append(point)
                labels.append("Masked exact scan")
            x_values.append(1.0)
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
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = root / "paper_gpu_bundle"
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
        "preflight": root / "provenance",
    }
    required = (
        inputs["b0"] / "summary_points.csv",
        inputs["matched_recall"] / "selected_points.csv",
        inputs["exact_scan"] / "exact_summary.csv",
        inputs["resource_work"] / "gpu_resource_work.csv",
        inputs["mechanism_diagnostics"] / "mechanism_summary.json",
        inputs["dataset_stats"] / "workload_selectivity_summary.json",
        inputs["preflight"] / "a100_preflight.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "bundle inputs are incomplete:\n" + "\n".join(missing)
        )

    copied: list[Path] = []
    for name, source in inputs.items():
        copied.extend(copy_tree_files(source, output / name))

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
        "hardware_and_run_identity": "preflight/a100_preflight.json",
    }
    (output / "claim_to_source.json").write_text(
        json.dumps(claims, indent=2) + "\n"
    )
    copied.append(output / "claim_to_source.json")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": json.loads(args.profile.read_text()),
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
