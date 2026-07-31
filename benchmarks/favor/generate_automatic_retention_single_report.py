#!/usr/bin/env python3
"""Generate the automatic-retention SINGLE_CTA comparison plots and summaries."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    key: str
    prefix: str
    title: str


DATASETS = (
    Dataset("sift", "sift", "SIFT-1M"),
    Dataset("gist", "gist", "GIST-1M"),
    Dataset("bigann1m", "bigann1m", "BIGANN-1M"),
    Dataset("bigann10m", "bigann10m", "BIGANN-10M"),
    Dataset("msturing1m", "msturing1m", "MSTuring-1M"),
    Dataset("msturing10m", "msturing10m", "MSTuring-10M"),
)
SELECTIVITIES = (1, 10, 50, 90)


def org_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a compact Org table."""
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "+".join("-" * (len(value) + 2) for value in headers) + "|"
    return [header, separator, *("| " + " | ".join(row) + " |" for row in rows)]


def render_report(
    output_path: Path,
    max_recall_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, str]],
) -> None:
    method_order = ("default", "favor_retention_safe", "automatic_retention")
    method_short = {
        "default": "Default",
        "favor_retention_safe": "Fixed rho=0.5",
        "automatic_retention": "Automatic rho",
    }
    max_lookup = {
        (row["dataset"], float(row["selectivity"]), row["workload"], row["series"]): float(
            row["max_recall"]
        )
        for row in max_recall_rows
    }
    target_lookup = {
        (row["dataset"], float(row["selectivity"]), row["workload"], row["series"]): float(
            row["value"]
        )
        for row in target_rows
    }

    lines = [
        "#+TITLE: Automatic-Retention FAVOR for CAGRA",
        "#+SUBTITLE: SINGLE_CTA Performance Evaluation",
        "#+AUTHOR: Nam Anh Dang",
        "#+DATE: 2026-07-31",
        "#+OPTIONS: toc:2 num:t",
        "",
        "* Scope",
        "",
        "This report compares three =SINGLE_CTA= bitset-filtered search methods: default CAGRA,",
        "fixed-retention FAVOR with =lambda=1 and =rho=0.5=, and automatic-retention FAVOR",
        "with =lambda=1 and =rho=auto=.  It reports the complete measured QPS--recall and",
        "latency--recall curves for six dataset/scale combinations at 1%, 10%, 50%, and 90%",
        "passing selectivity.  The document describes the methods and measurements only; it",
        "does not make a deployment recommendation.",
        "",
        "* Methods",
        "",
        "** Default filtered CAGRA",
        "",
        "Default CAGRA tests the bitset during traversal and invalidates a node that does not",
        "pass.  A rejected node therefore cannot remain in the candidate set as a route to",
        "other nodes.",
        "",
        "** Shared FAVOR penalty",
        "",
        "Both FAVOR variants keep rejected nodes available for traversal and add a penalty to",
        "their raw distance.  The batch-level reference penalty is",
        "",
        "#+begin_example",
        "C(s,B) = (1 - s)(B - s) / (2sB)",
        "D_reference = C(s,B) * delta_d",
        "#+end_example",
        "",
        "where =s= is the fraction of dataset rows that pass the bitset, =B= is CAGRA's",
        "internal top-k size (=itopk=), and =delta_d= is the graph-specific distance-spacing",
        "statistic loaded from the index sidecar.",
        "",
        "The query-local penalty is",
        "",
        "#+begin_example",
        "D_query = min(D_reference, lambda * g_q)",
        "#+end_example",
        "",
        "where =lambda= is a dimensionless multiplier (1 in this evaluation) and =g_q= is the",
        "average gap across the interquartile span of the query's sorted, finite initial",
        "candidate distances.  For fewer than four finite candidates, =g_q= uses the average",
        "gap over the full finite span.  A missing or degenerate span produces zero penalty.",
        "",
        "For a rejected candidate with raw distance =d=, let =tau= be the worst finite",
        "distance currently retained in the CTA and let =h=max(tau-d,0)= be the candidate's",
        "retention headroom.  Both variants use",
        "",
        "#+begin_example",
        "D_effective = min(D_query, rho * h)",
        "score = d + D_effective",
        "#+end_example",
        "",
        "while passing candidates retain =score=d=.  Because =rho<1=, the penalty alone cannot",
        "move a rejected candidate beyond the current retention boundary.",
        "",
        "** Fixed retention",
        "",
        "The previous retention method uses =rho=0.5= for every query, selectivity, and",
        "traversal size.  Thus a rejected candidate can consume at most half of its current",
        "retention headroom.",
        "",
        "** Automatic retention",
        "",
        "The automatic method derives =rho= once on the host from expected passing capacity:",
        "",
        "#+begin_example",
        "E = sB",
        "x = E / k",
        "P_short = clamp(2(1 - x), 0, 1)",
        "S(P_short) = P_short^2 * (3 - 2P_short)",
        "rho = 0.5 + 0.4 * S(P_short)",
        "#+end_example",
        "",
        "Here =k= is the requested result count (10), =E= is the expected number of passing",
        "entries in the internal top-k, =x= expresses that capacity relative to =k=,",
        "=P_short= is a bounded shortfall pressure, and =S= is a smoothstep function.  The",
        "rule gives =rho=0.5= when =E>=k= and increases it smoothly toward 0.9 when the",
        "expected passing capacity falls to =k/2= or below.  A larger =rho= preserves rejected",
        "bridge candidates more aggressively when the internal candidate set is expected to",
        "contain too few passing rows.",
        "",
        "** GPU path",
        "",
        "The filter check and penalty are fused into neighbor-distance computation, so each",
        "candidate is written once.  The automatic rule changes only the scalar =rho= passed to",
        "the existing retention-safe kernel.  It adds no candidate buffer and no per-candidate",
        "calculation beyond the fixed-retention path.  A zero retention fraction selects the",
        "automatic rule only for =SINGLE_CTA= FAVOR; default CAGRA behavior is unchanged.",
        "",
        "* Results",
        "",
        "Throughput uses 10,000 queries per batch and latency uses 10 queries per batch.",
        "Recall is Recall@10.  Target values at 0.90 recall use linear interpolation only when",
        "the measured Pareto frontier brackets 0.90.  If the fastest measured point already",
        "exceeds 0.90, that measured feasible point is reported.  =N/A= means the measured",
        "frontier did not reach the target; no extrapolation is used.",
        "",
        "** Maximum measured recall (throughput sweep)",
        "",
    ]

    max_rows: list[list[str]] = []
    qps_rows: list[list[str]] = []
    latency_rows: list[list[str]] = []
    for dataset in DATASETS:
        for selectivity in SELECTIVITIES:
            encoded = selectivity / 100
            prefix = [dataset.title, f"{selectivity}%"]
            max_rows.append(
                prefix
                + [
                    f"{max_lookup[(dataset.title, encoded, 'throughput', method)]:.3f}"
                    for method in method_order
                ]
            )
            qps_rows.append(
                prefix
                + [
                    f"{target_lookup[(dataset.title, encoded, 'throughput', method)]:,.0f}"
                    if (dataset.title, encoded, "throughput", method) in target_lookup
                    else "N/A"
                    for method in method_order
                ]
            )
            latency_rows.append(
                prefix
                + [
                    f"{1000 * target_lookup[(dataset.title, encoded, 'latency', method)]:.3f}"
                    if (dataset.title, encoded, "latency", method) in target_lookup
                    else "N/A"
                    for method in method_order
                ]
            )

    headers = ["Dataset", "Selectivity", *(method_short[m] for m in method_order)]
    lines.extend(org_table(headers, max_rows))
    lines.extend(
        [
            "",
            "** Throughput at Recall@10 = 0.90 (QPS)",
            "",
            *org_table(headers, qps_rows),
            "",
            "** Batch-10 latency at Recall@10 = 0.90 (ms)",
            "",
            *org_table(headers, latency_rows),
            "",
            "** Complete Pareto figures",
            "",
            "Each figure contains a throughput panel (QPS versus recall) and a batch-10 latency",
            "panel (latency versus recall) for all three methods.",
            "",
        ]
    )

    for dataset in DATASETS:
        lines.extend([f"*** {dataset.title}", ""])
        for selectivity in SELECTIVITIES:
            lines.extend(
                [
                    f"#+CAPTION: SINGLE_CTA on {dataset.title} at {selectivity}% passing selectivity.",
                    f"[[file:results_automatic_retention_single_comparison/{dataset.key}/plots/"
                    f"{dataset.prefix}_selectivity_{selectivity:02d}.png]]",
                    "",
                ]
            )

    lines.extend(
        [
            "* Appendix: benchmark parameters",
            "",
            "** Dataset and index inputs",
            "",
            *org_table(
                ["Dataset", "Rows", "Dim", "Type", "Query file", "Subset"],
                [
                    ["SIFT-1M", "1,000,000", "128", "float32", "query.fbin", "none"],
                    ["GIST-1M", "1,000,000", "960", "float32", "query_10000.fbin", "none"],
                    ["BIGANN-1M", "1,000,000", "128", "uint8", "query.public.10K.u8bin", "first 1M"],
                    ["BIGANN-10M", "10,000,000", "128", "uint8", "query.public.10K.u8bin", "none"],
                    ["MSTuring-1M", "1,000,000", "100", "float32", "query.fbin", "none"],
                    ["MSTuring-10M", "10,000,000", "100", "float32", "query.fbin", "none"],
                ],
            ),
            "",
            "Every dataset uses =cagra_g32_ig64.index= (maximum graph degree 32, intermediate",
            "graph degree 64) and its =cagra_g32_ig64.index.delta_d= sidecar.  Exact bitsets and",
            "precomputed exact filtered ground truth are used at every selectivity.",
            "",
            "** Common search and measurement parameters",
            "",
            *org_table(
                ["Parameter", "Value"],
                [
                    ["Search mode", "SINGLE_CTA"],
                    ["Result count k", "10"],
                    ["Passing selectivity", "1%, 10%, 50%, 90%"],
                    ["Throughput batch", "10,000 queries"],
                    ["Latency batch", "10 queries"],
                    ["Benchmark repetitions", "1"],
                    ["Minimum measured time", "0.2 s"],
                    ["Warm-up time", "0.1 s"],
                    ["FAVOR penalty mode", "cagra_retention_safe"],
                    ["Penalty lambda", "1"],
                    ["Fixed retention fraction", "0.5"],
                    ["Automatic retention selector", "favor_retention_fraction = 0"],
                    ["GPU", "NVIDIA A30, 24 GB"],
                ],
            ),
            "",
            "** Search cells",
            "",
            "Each tuple is =(itopk, search_width, max_iterations, thread_block_size)=.  Zero for",
            "=max_iterations= or =thread_block_size= requests the existing planner/default",
            "behavior.  The same cell set was used for batch 10 and batch 10,000 and for fixed",
            "and automatic retention.",
            "",
        ]
    )

    grouped_cells: dict[tuple[str, float], list[str]] = {}
    for row in parameter_rows:
        key = (row["dataset"], float(row["selectivity"]))
        grouped_cells.setdefault(key, []).append(
            "(" + ",".join(
                row[field]
                for field in ("itopk", "search_width", "max_iterations", "thread_block_size")
            ) + ")"
        )
    cell_rows = []
    for dataset in DATASETS:
        for selectivity in SELECTIVITIES:
            key = (dataset.title, selectivity / 100)
            cell_rows.append([dataset.title, f"{selectivity}%", "; ".join(grouped_cells[key])])
    lines.extend(org_table(["Dataset", "Selectivity", "Cells"], cell_rows))
    lines.extend(
        [
            "",
            "** Artifact locations and reproduction",
            "",
            "The prior default/fixed-retention measurements and configurations are in",
            "=results_retention_safe_single_full=.  Automatic-retention raw JSON and generated",
            "configurations are in =results_automatic_retention_single_full=.  Plot data, target",
            "summaries, the parameter-cell CSV, and figures are in",
            "=results_automatic_retention_single_comparison=.",
            "",
            "#+begin_src sh",
            "benchmarks/favor/run_automatic_retention_single_full.sh",
            "python benchmarks/favor/generate_automatic_retention_single_report.py",
            "cd benchmarks/favor",
            "pandoc -f org -t docx --standalone \\",
            "  -o AUTOMATIC_RETENTION_FAVOR_SINGLE_CTA_REPORT.docx \\",
            "  AUTOMATIC_RETENTION_FAVOR_SINGLE_CTA_REPORT.org",
            "#+end_src",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty summary {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--automatic-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    baseline_root = (args.baseline_root or root / "results_retention_safe_single_full").resolve()
    automatic_root = (
        args.automatic_root or root / "results_automatic_retention_single_full"
    ).resolve()
    output_root = (
        args.output_root or root / "results_automatic_retention_single_comparison"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    max_recall_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    parameter_rows: list[dict[str, str]] = []
    labels = {
        "default": "Default CAGRA",
        "favor_retention_safe": "Fixed retention (lambda=1, rho=0.5)",
        "automatic_retention": "Automatic retention (lambda=1, rho=auto)",
    }

    for dataset in DATASETS:
        dataset_output = output_root / dataset.key
        subprocess.run(
            [
                sys.executable,
                str(root / "plot_results.py"),
                "--result-dir",
                str(baseline_root / dataset.key),
                "--output-dir",
                str(dataset_output),
                "--result-prefix",
                dataset.prefix,
                "--plot-title",
                dataset.title,
                "--selectivities",
                *(str(value) for value in SELECTIVITIES),
                "--penalty-lambdas",
                "1",
                "--cta-mode",
                "SINGLE_CTA",
                "--target-recall",
                "0.90",
                "--overlay-series",
                "automatic_retention",
                "favor_retention_safe",
                "Automatic retention FAVOR",
                str(automatic_root / dataset.key),
                "--series-label",
                "favor_retention_safe",
                "Fixed retention FAVOR (rho=0.5)",
                "--series-color",
                "automatic_retention",
                "#2ca02c",
            ],
            check=True,
        )

        with (dataset_output / "favor_benchmark_summary.csv").open(newline="") as stream:
            points = list(csv.DictReader(stream))
        for selectivity in SELECTIVITIES:
            encoded = selectivity / 100
            for workload in ("throughput", "latency"):
                for series, label in labels.items():
                    matching = [
                        row
                        for row in points
                        if float(row["selectivity"]) == encoded
                        and row["workload"] == workload
                        and row["series"] == series
                    ]
                    if not matching:
                        raise ValueError(
                            f"missing {dataset.key} {selectivity}% {workload} {series} points"
                        )
                    max_recall_rows.append(
                        {
                            "dataset": dataset.title,
                            "selectivity": f"{encoded:g}",
                            "workload": workload,
                            "batch_size": matching[0]["batch_size"],
                            "series": series,
                            "method": label,
                            "max_recall": f"{max(float(row['recall']) for row in matching):.6f}",
                        }
                    )

        with (dataset_output / "target_recall_summary.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                target_rows.append(
                    {
                        "dataset": dataset.title,
                        "method": labels[row["series"]],
                        **row,
                    }
                )

        for selectivity in SELECTIVITIES:
            config_path = (
                baseline_root
                / dataset.key
                / "configs"
                / f"{dataset.prefix}_s{selectivity:02d}_nq10000.json"
            )
            payload = json.loads(config_path.read_text())
            cells = {
                (
                    int(row["itopk"]),
                    int(row["search_width"]),
                    int(row.get("max_iterations", 0)),
                    int(row.get("thread_block_size", 0)),
                )
                for row in payload["index"][0]["search_params"]
                if row.get("favor_penalty_mode") == "cagra_retention_safe"
            }
            for itopk, width, iterations, block_size in sorted(cells):
                parameter_rows.append(
                    {
                        "dataset": dataset.title,
                        "selectivity": f"{selectivity / 100:g}",
                        "itopk": str(itopk),
                        "search_width": str(width),
                        "max_iterations": str(iterations),
                        "thread_block_size": str(block_size),
                    }
                )

    write_csv(output_root / "combined_max_recall.csv", max_recall_rows)
    write_csv(output_root / "combined_target_recall_090.csv", target_rows)
    write_csv(output_root / "benchmark_parameter_cells.csv", parameter_rows)
    render_report(
        root / "AUTOMATIC_RETENTION_FAVOR_SINGLE_CTA_REPORT.org",
        max_recall_rows,
        target_rows,
        parameter_rows,
    )


if __name__ == "__main__":
    main()
