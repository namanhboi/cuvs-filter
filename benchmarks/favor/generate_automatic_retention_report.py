#!/usr/bin/env python3
"""Generate combined SINGLE_CTA/MULTI_CTA automatic-retention results and report."""

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


@dataclass(frozen=True)
class Evaluation:
    max_recall_rows: list[dict[str, str]]
    target_rows: list[dict[str, str]]
    parameter_rows: list[dict[str, str]]
    gpu_name: str


DATASETS = (
    Dataset("sift", "sift", "SIFT-1M"),
    Dataset("gist", "gist", "GIST-1M"),
    Dataset("bigann1m", "bigann1m", "BIGANN-1M"),
    Dataset("bigann10m", "bigann10m", "BIGANN-10M"),
    Dataset("msturing1m", "msturing1m", "MSTuring-1M"),
    Dataset("msturing10m", "msturing10m", "MSTuring-10M"),
)
SELECTIVITIES = (1, 10, 50, 90)
METHOD_ORDER = ("default", "favor_retention_safe", "automatic_retention")
METHOD_LABELS = {
    "default": "Default CAGRA",
    "favor_retention_safe": "Fixed retention (lambda=1, rho=0.5)",
    "automatic_retention": "Automatic retention (lambda=1, rho=auto)",
}
METHOD_SHORT = {
    "default": "Default",
    "favor_retention_safe": "Fixed rho=0.5",
    "automatic_retention": "Automatic rho",
}
MULTI_ITOPK_VALUES = {
    "sift": (32, 64, 128, 256, 512, 1024),
    "gist": (32, 64, 128, 256, 512, 640, 768, 1024),
    "bigann1m": (32, 64, 128, 256, 512, 1024),
    "bigann10m": (32, 64, 128, 256, 512, 1024),
    "msturing1m": (32, 64, 128, 256, 512, 640, 768, 1024),
    "msturing10m": (32, 64, 128, 256, 512, 1024, 1536, 2048),
}


def org_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a compact Org table."""
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "+".join("-" * (len(value) + 2) for value in headers) + "|"
    return [header, separator, *("| " + " | ".join(row) + " |" for row in rows)]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty summary {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def benchmark_gpu(path: Path) -> str:
    """Return the GPU recorded by one benchmark result, failing on missing provenance."""
    try:
        payload = json.loads(path.read_text())
        gpu_name = payload["context"]["gpu_name"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark GPU provenance from {path}: {error}") from error
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise ValueError(f"benchmark result has an invalid GPU name: {path}")
    return gpu_name.strip()


def validate_result_completeness(result_path: Path, config_path: Path) -> None:
    """Apply the runner's completeness gate again before using a result in a report."""
    try:
        result_rows = json.loads(result_path.read_text())["benchmarks"]
        search_params = json.loads(config_path.read_text())["index"][0]["search_params"]
    except (OSError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot validate benchmark result {result_path} against {config_path}: {error}"
        ) from error
    iterations = [
        row for row in result_rows if row.get("run_type", "iteration") == "iteration"
    ]
    run_names = {row.get("run_name") for row in iterations if row.get("run_name")}
    has_error = any(row.get("error_occurred", False) for row in result_rows)
    expected = len(search_params)
    if len(iterations) < expected or len(run_names) < expected or has_error:
        raise ValueError(
            f"incomplete benchmark result {result_path}: expected {expected} configurations, "
            f"found {len(iterations)} iterations and {len(run_names)} distinct runs, "
            f"error_occurred={has_error}"
        )


def config_method(row: dict) -> str | None:
    """Map one benchmark search parameter to a report method."""
    if row.get("filter_mode") == "default":
        return "default"
    if row.get("favor_penalty_mode") != "cagra_retention_safe":
        return None
    retention_fraction = float(row.get("favor_retention_fraction", 0.5))
    if retention_fraction == 0.0:
        return "automatic_retention"
    if retention_fraction == 0.5:
        return "favor_retention_safe"
    raise ValueError(
        "automatic-retention reports only support fixed rho=0.5 and automatic rho=0; "
        f"found rho={retention_fraction:g}"
    )


def load_config_cells(paths: list[Path]) -> set[tuple[int, int, int, int]]:
    """Validate that default, fixed, and automatic methods use identical search cells."""
    method_cells: dict[str, set[tuple[int, int, int, int]]] = {
        method: set() for method in METHOD_ORDER
    }
    method_counts = {method: 0 for method in METHOD_ORDER}
    for path in paths:
        try:
            payload = json.loads(path.read_text())
            search_params = payload["index"][0]["search_params"]
        except (OSError, KeyError, IndexError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read benchmark configuration {path}: {error}") from error
        for row in search_params:
            method = config_method(row)
            if method is None:
                continue
            method_counts[method] += 1
            method_cells[method].add(
                (
                    int(row["itopk"]),
                    int(row["search_width"]),
                    int(row.get("max_iterations", 0)),
                    int(row.get("thread_block_size", 0)),
                )
            )

    missing = [method for method, cells in method_cells.items() if not cells]
    if missing:
        raise ValueError(
            f"benchmark configuration is missing report methods {missing}: {paths}"
        )
    duplicated = {
        method: method_counts[method] - len(cells)
        for method, cells in method_cells.items()
        if method_counts[method] != len(cells)
    }
    if duplicated:
        raise ValueError(f"benchmark configuration contains duplicate cells: {duplicated}")
    reference = method_cells[METHOD_ORDER[0]]
    mismatched = {
        method: sorted(cells)
        for method, cells in method_cells.items()
        if cells != reference
    }
    if mismatched:
        counts = {method: len(cells) for method, cells in method_cells.items()}
        raise ValueError(
            "default, fixed, and automatic retention must use identical search cells; "
            f"counts={counts}, mismatched methods={sorted(mismatched)}: {paths}"
        )
    return reference


def build_plot_command(
    root: Path,
    result_root: Path,
    output_root: Path,
    dataset: Dataset,
    *,
    automatic_overlay_root: Path | None,
    cta_mode: str,
    batch_size: int,
    target_recall: float,
    latency_derived_qps: bool,
) -> list[str]:
    """Build a plot command, using native automatic results unless an overlay is explicit."""
    command = [
        sys.executable,
        str(root / "plot_results.py"),
        "--result-dir",
        str(result_root / dataset.key),
        "--output-dir",
        str(output_root / dataset.key),
        "--result-prefix",
        dataset.prefix,
        "--plot-title",
        dataset.title,
        "--selectivities",
        *(str(value) for value in SELECTIVITIES),
        "--penalty-lambdas",
        "1",
        "--cta-mode",
        cta_mode,
        "--target-recall",
        f"{target_recall:g}",
        "--series-label",
        "favor_retention_safe",
        "Fixed retention FAVOR",
        "--series-label",
        "automatic_retention",
        "Automatic retention FAVOR",
        "--series-color",
        "automatic_retention",
        "#2ca02c",
        "--latency-batch-size",
        str(batch_size),
    ]
    if automatic_overlay_root is not None:
        command.extend(
            (
                "--overlay-series",
                "automatic_retention",
                "automatic_retention",
                "Automatic retention FAVOR",
                str(automatic_overlay_root / dataset.key),
            )
        )
    if latency_derived_qps:
        command.extend(("--latency-derived-qps", "--latency-unit", "us"))
    else:
        command.extend(("--throughput-batch-size", "10000"))
    return command


def validate_evaluation_sources(
    result_root: Path,
    *,
    automatic_overlay_root: Path | None,
    cta_mode: str,
    batch_size: int,
    config_batch_size: int,
    expected_gpu: str | None,
) -> tuple[list[dict[str, str]], str]:
    """Preflight result provenance and matched parameter cells before writing outputs."""
    parameter_rows: list[dict[str, str]] = []
    observed_gpus: set[str] = set()
    result_batch_sizes = tuple(dict.fromkeys((batch_size, config_batch_size)))
    for dataset in DATASETS:
        for selectivity in SELECTIVITIES:
            reference_cells: set[tuple[int, int, int, int]] | None = None
            for result_batch_size in result_batch_sizes:
                raw_paths = [
                    result_root
                    / dataset.key
                    / "raw"
                    / f"{dataset.prefix}_s{selectivity:02d}_nq{result_batch_size}.json"
                ]
                config_paths = [
                    result_root
                    / dataset.key
                    / "configs"
                    / f"{dataset.prefix}_s{selectivity:02d}_nq{result_batch_size}.json"
                ]
                if automatic_overlay_root is not None:
                    raw_paths.append(
                        automatic_overlay_root
                        / dataset.key
                        / "raw"
                        / f"{dataset.prefix}_s{selectivity:02d}_nq{result_batch_size}.json"
                    )
                    config_paths.append(
                        automatic_overlay_root
                        / dataset.key
                        / "configs"
                        / f"{dataset.prefix}_s{selectivity:02d}_nq{result_batch_size}.json"
                    )
                for raw_path, config_path in zip(raw_paths, config_paths, strict=True):
                    observed_gpus.add(benchmark_gpu(raw_path))
                    validate_result_completeness(raw_path, config_path)
                cells = load_config_cells(config_paths)
                if reference_cells is None:
                    reference_cells = cells
                elif cells != reference_cells:
                    raise ValueError(
                        f"throughput and latency use different {cta_mode} cells for "
                        f"{dataset.key} at {selectivity}%"
                    )
            assert reference_cells is not None
            if cta_mode == "MULTI_CTA" and automatic_overlay_root is None:
                expected_cells = {
                    (itopk, 1, 0, 0) for itopk in MULTI_ITOPK_VALUES[dataset.key]
                }
                if reference_cells != expected_cells:
                    raise ValueError(
                        f"unexpected MULTI_CTA search cells for {dataset.key} at "
                        f"{selectivity}%: expected {sorted(expected_cells)}, "
                        f"found {sorted(reference_cells)}"
                    )
            for itopk, width, iterations, block_size in sorted(reference_cells):
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

    if len(observed_gpus) != 1:
        raise ValueError(
            f"refusing to mix benchmark hardware for {cta_mode}: {sorted(observed_gpus)}"
        )
    gpu_name = next(iter(observed_gpus))
    if expected_gpu and gpu_name != expected_gpu:
        raise ValueError(
            f"{cta_mode} results were measured on {gpu_name!r}; expected {expected_gpu!r}"
        )
    return parameter_rows, gpu_name


def collect_evaluation(
    root: Path,
    result_root: Path,
    output_root: Path,
    *,
    automatic_overlay_root: Path | None,
    cta_mode: str,
    batch_size: int,
    config_batch_size: int,
    target_recall: float,
    latency_derived_qps: bool,
    expected_gpu: str | None = None,
) -> Evaluation:
    """Plot one CTA mode and collect report tables from hardware-matched result roots."""
    parameter_rows, gpu_name = validate_evaluation_sources(
        result_root,
        automatic_overlay_root=automatic_overlay_root,
        cta_mode=cta_mode,
        batch_size=batch_size,
        config_batch_size=config_batch_size,
        expected_gpu=expected_gpu,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    max_recall_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []

    for dataset in DATASETS:
        dataset_output = output_root / dataset.key
        plot_command = build_plot_command(
            root,
            result_root,
            output_root,
            dataset,
            automatic_overlay_root=automatic_overlay_root,
            cta_mode=cta_mode,
            batch_size=batch_size,
            target_recall=target_recall,
            latency_derived_qps=latency_derived_qps,
        )
        subprocess.run(plot_command, check=True)

        with (dataset_output / "favor_benchmark_summary.csv").open(newline="") as stream:
            points = list(csv.DictReader(stream))
        for selectivity in SELECTIVITIES:
            encoded = selectivity / 100
            for workload in ("throughput", "latency"):
                for series, label in METHOD_LABELS.items():
                    matching = [
                        row
                        for row in points
                        if float(row["selectivity"]) == encoded
                        and row["workload"] == workload
                        and row["series"] == series
                    ]
                    if not matching:
                        raise ValueError(
                            f"missing {cta_mode} {dataset.key} {selectivity}% "
                            f"{workload} {series} points"
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
                        "method": METHOD_LABELS[row["series"]],
                        **row,
                    }
                )

    return Evaluation(max_recall_rows, target_rows, parameter_rows, gpu_name)


def result_tables(
    evaluation: Evaluation, *, target_recall: float, latency_scale: float
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    max_lookup = {
        (row["dataset"], float(row["selectivity"]), row["workload"], row["series"]): float(
            row["max_recall"]
        )
        for row in evaluation.max_recall_rows
    }
    target_lookup = {
        (row["dataset"], float(row["selectivity"]), row["workload"], row["series"]): float(
            row["value"]
        )
        for row in evaluation.target_rows
        if float(row["target_recall"]) == target_recall
    }
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
                    for method in METHOD_ORDER
                ]
            )
            qps_rows.append(
                prefix
                + [
                    f"{target_lookup[(dataset.title, encoded, 'throughput', method)]:,.0f}"
                    if (dataset.title, encoded, "throughput", method) in target_lookup
                    else "N/A"
                    for method in METHOD_ORDER
                ]
            )
            latency_rows.append(
                prefix
                + [
                    f"{latency_scale * target_lookup[(dataset.title, encoded, 'latency', method)]:.3f}"
                    if (dataset.title, encoded, "latency", method) in target_lookup
                    else "N/A"
                    for method in METHOD_ORDER
                ]
            )
    return max_rows, qps_rows, latency_rows


def parameter_table(parameter_rows: list[dict[str, str]]) -> list[list[str]]:
    grouped_cells: dict[tuple[str, float], list[str]] = {}
    for row in parameter_rows:
        key = (row["dataset"], float(row["selectivity"]))
        grouped_cells.setdefault(key, []).append(
            "("
            + ",".join(
                row[field]
                for field in ("itopk", "search_width", "max_iterations", "thread_block_size")
            )
            + ")"
        )
    return [
        [
            dataset.title,
            f"{selectivity}%",
            "; ".join(grouped_cells[(dataset.title, selectivity / 100)]),
        ]
        for dataset in DATASETS
        for selectivity in SELECTIVITIES
    ]


def append_figures(lines: list[str], *, cta_mode: str, result_root: str) -> None:
    for dataset in DATASETS:
        lines.extend([f"*** {dataset.title}", ""])
        for selectivity in SELECTIVITIES:
            lines.extend(
                [
                    f"#+CAPTION: {cta_mode} on {dataset.title} at {selectivity}% passing selectivity.",
                    (
                        f"[[file:{result_root}/{dataset.key}/plots/"
                        f"{dataset.prefix}_selectivity_{selectivity:02d}.png]]"
                    ),
                    "",
                ]
            )


def render_report(output_path: Path, single: Evaluation, multi: Evaluation) -> None:
    single_max, single_qps, single_latency = result_tables(
        single, target_recall=0.90, latency_scale=1_000
    )
    multi_max, multi_qps, multi_latency = result_tables(
        multi, target_recall=0.99, latency_scale=1_000_000
    )
    headers = ["Dataset", "Selectivity", *(METHOD_SHORT[m] for m in METHOD_ORDER)]

    lines = [
        "#+TITLE: Automatic-Retention FAVOR for CAGRA",
        "#+SUBTITLE: SINGLE_CTA and MULTI_CTA Performance Evaluation",
        "#+AUTHOR: Nam Anh Dang",
        "#+DATE: 2026-08-02",
        "#+OPTIONS: toc:2 num:t",
        "",
        "* Scope",
        "",
        "This report compares default CAGRA, fixed-retention FAVOR with =lambda=1 and",
        "=rho=0.5=, and automatic-retention FAVOR with =lambda=1 and =rho=auto= in both",
        "=SINGLE_CTA= and =MULTI_CTA=.  It reports measured QPS--recall and latency--recall",
        "curves for six dataset/scale combinations at 1%, 10%, 50%, and 90% passing",
        "selectivity.  The document describes the methods and measurements only; it does not",
        "make a deployment recommendation.",
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
        "where =s= is passing selectivity, =B= is CAGRA's configured internal top-k size, and",
        "=delta_d= is the graph-specific distance-spacing statistic loaded from the index",
        "sidecar.  The query-local penalty is",
        "",
        "#+begin_example",
        "D_query = min(D_reference, lambda * g_q)",
        "#+end_example",
        "",
        "where =lambda= is 1 in this evaluation and =g_q= is the average gap across the",
        "interquartile span of the CTA's sorted, finite initial candidate distances.  For a",
        "rejected candidate with raw distance =d=, let =tau= be the worst finite distance",
        "currently retained and =h=max(tau-d,0)= be its retention headroom.  Both variants use",
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
        "The fixed method uses =rho=0.5= for every query, selectivity, and traversal size.",
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
        "Here =k= is the requested result count (10).  The rule gives =rho=0.5= when =E>=k=",
        "and increases it smoothly toward 0.9 when expected passing capacity falls to =k/2= or",
        "below.  =SINGLE_CTA= uses its internal top-k for =B=.  =MULTI_CTA= uses the",
        "user-configured global =itopk=, matching its reference-penalty convention; the",
        "selectivity-inflated number of physical CTA result slots does not redefine this policy.",
        "",
        "** Control-region interpretation",
        "",
        "Whenever =sB>=k=, automatic retention resolves exactly to =rho=0.5= and therefore",
        "passes the same kernel policy parameter as fixed retention.  In the =MULTI_CTA= grid",
        "this control region includes =itopk>=1024= at 1%, =itopk>=128= at 10%, and every",
        "cell at 50% and 90%.  The configuration and plot label remain =rho=auto= because zero",
        "is the host-side selector, even though the resolved kernel value is 0.5.",
        "",
        "Batch-size-1 Google Benchmark rows may execute different iteration counts and query",
        "samples.  Small fixed-versus-automatic differences in recall, QPS, latency, or apparent",
        "Pareto crossover within this control region are sampling noise, not an automatic-policy",
        "effect.  Those cells are equivalence controls and must not be used to claim a win or loss.",
        "",
        "** GPU path",
        "",
        "The filter check and penalty are fused into neighbor-distance computation, so each",
        "candidate is written once.  The automatic rule changes only the scalar =rho= passed to",
        "the existing retention-safe kernel.  It adds no candidate buffer and no per-candidate",
        "calculation beyond the fixed-retention path.  A zero public retention fraction selects",
        "the automatic host rule in either CTA mode; default CAGRA remains unchanged.",
        "",
        "* SINGLE_CTA cutoff-indexing caveat",
        "",
        "The archived =SINGLE_CTA= measurements preserve the implementation under test.  Its query-gap and",
        "retention-cutoff helpers index the bitonic result buffer as though physical positions",
        "were logical sorted ranks, but CAGRA's bitonic layout is swizzled.  In the hard",
        "=SINGLE_CTA= =itopk=512= cells, physical tail slot 511 is logical rank 496 rather than",
        "rank 511; the sampled quartile positions likewise correspond to logical ranks 124 and",
        "372 rather than 127 and 383.  The resulting =D_query= and =tau= therefore differ from",
        "the intended logical-rank formula.  Fixed and automatic curves remain a controlled",
        "comparison because both use the same helper, but this report does not present either",
        "curve as a corrected-cutoff result.  =MULTI_CTA= uses a linear sorted CTA-local",
        "buffer and is not affected by this indexing caveat.",
        "",
        "* SINGLE_CTA results",
        "",
        "Throughput uses 10,000 queries per batch and latency uses 10 queries per batch.",
        "Recall is Recall@10.  Target values at 0.90 use interpolation only when the measured",
        "Pareto frontier brackets 0.90; otherwise the fastest measured feasible point is used.",
        "=N/A= means the measured frontier did not reach the target.  No extrapolation is used.",
        "",
        "** Maximum measured recall (throughput sweep)",
        "",
        *org_table(headers, single_max),
        "",
        "** Throughput at Recall@10 = 0.90 (QPS)",
        "",
        *org_table(headers, single_qps),
        "",
        "** Batch-10 latency at Recall@10 = 0.90 (ms)",
        "",
        *org_table(headers, single_latency),
        "",
        "** Complete Pareto figures",
        "",
        "Each figure contains a batch-10,000 QPS panel and a batch-10 latency panel for all",
        "three methods.  Both y-axes begin at zero.",
        "",
    ]
    append_figures(
        lines,
        cta_mode="SINGLE_CTA",
        result_root="results_automatic_retention_single_comparison",
    )

    lines.extend(
        [
            "* MULTI_CTA results",
            "",
            "=MULTI_CTA= uses a one-query batch.  Reported QPS is the reciprocal of",
            "single-query latency and is not saturated large-batch throughput.  Target values",
            "at Recall@10 = 0.99 follow the same measured-feasible/bracketed-interpolation rule",
            "as =SINGLE_CTA=, with no extrapolation.",
            "",
            "** Maximum measured recall (batch-size-1 sweep)",
            "",
            *org_table(headers, multi_max),
            "",
            "** Batch-size-1 QPS at Recall@10 = 0.99",
            "",
            *org_table(headers, multi_qps),
            "",
            "** Single-query latency at Recall@10 = 0.99 (microseconds)",
            "",
            *org_table(headers, multi_latency),
            "",
            "** Complete Pareto figures",
            "",
            "Each figure contains batch-size-1 reciprocal QPS and latency panels for all three",
            "methods.  Both y-axes begin at zero.",
            "",
        ]
    )
    append_figures(
        lines,
        cta_mode="MULTI_CTA",
        result_root="results_automatic_retention_multi_comparison",
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
            "Every dataset uses =cagra_g32_ig64.index= and its matching =.delta_d= sidecar.",
            "Exact bitsets and precomputed exact filtered ground truth are used throughout.",
            "All experiments use =k=10=, one benchmark repetition, minimum measured time 0.2 s,",
            "and warm-up 0.1 s.  FAVOR runs use =cagra_retention_safe= and =lambda=1=;",
            "default CAGRA has no FAVOR penalty.  The archived",
            f"=SINGLE_CTA= curves were measured on {single.gpu_name}; the new =MULTI_CTA=",
            f"default, fixed, and automatic curves were all measured on {multi.gpu_name}.",
            "Absolute timing is compared only within a CTA mode, never between these GPUs.",
            "",
            "** SINGLE_CTA search cells",
            "",
            "Each tuple is =(itopk, search_width, max_iterations, thread_block_size)=.  Zero",
            "requests existing planner/default behavior.  Default, fixed, and automatic",
            "retention use the same cells for batch 10 and batch 10,000.",
            "",
            *org_table(
                ["Dataset", "Selectivity", "Cells"],
                parameter_table(single.parameter_rows),
            ),
            "",
            "** MULTI_CTA search cells",
            "",
            "The tuple schema is identical.  Default, fixed, and automatic retention use the",
            "same batch-size-1 cells, including MSTuring-10M values through =itopk=2048=.",
            "",
            *org_table(
                ["Dataset", "Selectivity", "Cells"],
                parameter_table(multi.parameter_rows),
            ),
            "",
            "** Artifact locations and reproduction",
            "",
            "=SINGLE_CTA= fixed/default data is in =results_retention_safe_single_full=; its",
            "automatic data and combined outputs are in =results_automatic_retention_single_full=",
            "and =results_automatic_retention_single_comparison=.  All three fresh =MULTI_CTA=",
            "series are in =results_automatic_retention_multi_full=; processed tables and",
            "plots are in",
            "=results_automatic_retention_multi_comparison=.",
            "",
            "#+begin_src sh",
            "benchmarks/favor/run_automatic_retention_single_full.sh",
            "benchmarks/favor/run_automatic_retention_multi_full.sh",
            "python benchmarks/favor/generate_automatic_retention_report.py",
            "cd benchmarks/favor",
            "pandoc -f org -t docx --standalone \\",
            "  -o AUTOMATIC_RETENTION_FAVOR_SINGLE_MULTI_CTA_REPORT.docx \\",
            "  AUTOMATIC_RETENTION_FAVOR_SINGLE_MULTI_CTA_REPORT.org",
            "#+end_src",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path(__file__).resolve().parent
    )
    # Preserve the original option names as the SINGLE_CTA inputs.
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--automatic-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--multi-root",
        type=Path,
        help="unified MULTI_CTA root containing fresh default, fixed, and automatic runs",
    )
    parser.add_argument("--multi-output-root", type=Path)
    parser.add_argument(
        "--multi-expected-gpu",
        help=(
            "optionally require an exact MULTI_CTA GPU name; by default the report accepts "
            "one consistently recorded GPU and reports its observed name"
        ),
    )
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    single_baseline = (
        args.baseline_root or root / "results_retention_safe_single_full"
    ).resolve()
    single_automatic = (
        args.automatic_root or root / "results_automatic_retention_single_full"
    ).resolve()
    single_output = (
        args.output_root or root / "results_automatic_retention_single_comparison"
    ).resolve()
    multi_root = (
        args.multi_root or root / "results_automatic_retention_multi_full"
    ).resolve()
    multi_output = (
        args.multi_output_root or root / "results_automatic_retention_multi_comparison"
    ).resolve()
    report_output = (
        args.report_output
        or root / "AUTOMATIC_RETENTION_FAVOR_SINGLE_MULTI_CTA_REPORT.org"
    ).resolve()

    single = collect_evaluation(
        root,
        single_baseline,
        single_output,
        automatic_overlay_root=single_automatic,
        cta_mode="SINGLE_CTA",
        batch_size=10,
        config_batch_size=10000,
        target_recall=0.90,
        latency_derived_qps=False,
    )
    multi = collect_evaluation(
        root,
        multi_root,
        multi_output,
        automatic_overlay_root=None,
        cta_mode="MULTI_CTA",
        batch_size=1,
        config_batch_size=1,
        target_recall=0.99,
        latency_derived_qps=True,
        expected_gpu=args.multi_expected_gpu,
    )

    write_csv(single_output / "combined_max_recall.csv", single.max_recall_rows)
    write_csv(single_output / "combined_target_recall_090.csv", single.target_rows)
    write_csv(single_output / "benchmark_parameter_cells.csv", single.parameter_rows)
    write_csv(multi_output / "combined_max_recall.csv", multi.max_recall_rows)
    write_csv(multi_output / "combined_target_recall_099.csv", multi.target_rows)
    write_csv(multi_output / "benchmark_parameter_cells.csv", multi.parameter_rows)
    render_report(report_output, single, multi)


if __name__ == "__main__":
    main()
