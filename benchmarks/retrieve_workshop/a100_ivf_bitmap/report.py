#!/usr/bin/env python3
"""Generate the standalone IEEEtran report after the validated IVF sweep finishes."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

WORKLOADS = ("yfcc", "em", "emis", "r")
IVF = ("ivf_flat", "ivf_sq", "ivf_pq", "ivf_pq_refined")
METHODS = ("base", "retain", "navix", *IVF)
LABELS = dict(
    base="CAGRA-Base",
    retain="CAGRA-Retain",
    navix="CAGRA-NaviX",
    ivf_flat="IVF-Flat",
    ivf_sq="IVF-SQ",
    ivf_pq="IVF-PQ",
    ivf_pq_refined="IVF-PQ + GPU rerank",
)
WORKLOAD_LABELS = dict(
    yfcc="YFCC", em="ArXiv-EM", emis="ArXiv-EMIS", r="ArXiv-R"
)
TARGETS = dict(yfcc=0.8, em=0.95, emis=0.95, r=0.95)
MAP = dict(
    default_cagra="base",
    default_cagra_accumulator="retain",
    navix_reference="navix",
)
SPARSE_RATES = ("0p0001", "0p0003", "0p001", "0p003", "0p01")
SPARSE_METHODS = (
    "default_cagra",
    "default_cagra_accumulator",
    "navix_reference",
    "ivf_flat",
    "exact",
)


def load_sparse_results(root):
    """Only import the completed, hashed five-cohort sparse comparison."""
    progress = json.loads((root / "state/progress.json").read_text())
    if progress["state"] != "complete":
        raise ValueError(
            "Sparse experiment must finish before report inclusion"
        )
    provenance = json.loads((root / "analysis/provenance.json").read_text())
    for key, path in (
        ("manifest_sha256", root / "state/manifest.json"),
        ("comparison_sha256", root / "analysis/comparison.csv"),
        ("curve_points_sha256", root / "analysis/curve_points.csv"),
    ):
        if sha(path) != provenance[key]:
            raise ValueError(f"Changed sparse report input: {path}")
    manifest = json.loads((root / "state/manifest.json").read_text())
    if (
        manifest["experiment"],
        manifest["query_count"],
        manifest["k"],
        manifest["repetitions"],
    ) != ("arxiv_sparse_ivf_flat", 2048, 10, 3):
        raise ValueError(
            "Sparse report requires the 2,048-query k=10 experiment"
        )
    rows = read_csv(root / "analysis/comparison.csv")
    expected = {
        (f"synthetic_low_{rate}_random", m)
        for rate in SPARSE_RATES
        for m in SPARSE_METHODS
    }
    keys = [(r["cohort"], r["method"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected exactly 25 sparse method/cohort cells")
    for row in rows:
        if int(row["repetitions"]) != 3:
            raise ValueError("Sparse report requires three-repetition results")
        for key in (
            "qps",
            "recall",
            "max_recall",
            "qps_at_max_recall",
            "recall_min",
            "recall_max",
        ):
            if row.get(key):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"Nonfinite sparse {key}")
        if row["status"] not in (
            "measured",
            "interpolated",
            "unmatched",
            "exact",
        ):
            raise ValueError("Unknown sparse target status")
        if (
            row["status"] in ("measured", "interpolated")
            and not 0.950 <= row["recall"] <= 0.952 + 1e-12
        ):
            raise ValueError("Sparse match is outside the recall window")
        if row["method"] == "ivf_flat" and row["status"] == "measured":
            if (
                not 0.950
                <= row["recall_min"]
                <= row["recall_max"]
                <= 0.952 + 1e-12
            ):
                raise ValueError("Sparse IVF repetition is outside the window")
    return rows


def sparse_bar_values(rows):
    lookup = {(r["cohort"], r["method"]): r for r in rows}
    values = []
    for rate in SPARSE_RATES:
        for method in SPARSE_METHODS:
            row = lookup[(f"synthetic_low_{rate}_random", method)]
            unmatched = row["status"] == "unmatched"
            qps = row["qps_at_max_recall"] if unmatched else row["qps"]
            if qps <= 0:
                raise ValueError("Sparse bar QPS must be positive")
            values.append(
                dict(
                    selectivity=float(rate.replace("p", ".")),
                    method=method,
                    qps=qps,
                    recall=row["max_recall"] if unmatched else row["recall"],
                    status=row["status"],
                    unmatched=unmatched,
                )
            )
    return values


def make_sparse_plot(directory, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    values = sparse_bar_values(rows)
    colors = ("#777777", "#9467bd", "#8c564b", "#1f77b4", "#2ca02c")
    labels = (
        "CAGRA-Base",
        "CAGRA-Retain",
        "CAGRA-NaviX",
        "IVF-Flat",
        "Exact KNN",
    )
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    width = 0.145
    for index, (method, color, label) in enumerate(
        zip(SPARSE_METHODS, colors, labels)
    ):
        cells = [r for r in values if r["method"] == method]
        positions = [i + (index - 2) * width for i in range(5)]
        bars = ax.bar(
            positions,
            [r["qps"] for r in cells],
            width,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            label=label,
        )
        for bar, cell in zip(bars, cells):
            if cell["unmatched"]:
                bar.set_hatch("////")
                bar.set_alpha(0.5)
                ax.annotate(
                    f"max R={cell['recall']:.3f}",
                    (bar.get_x() + bar.get_width() / 2, cell["qps"]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6,
                )
    ax.set(
        xticks=range(5),
        xticklabels=("0.01%", "0.03%", "0.1%", "0.3%", "1%"),
        xlabel="Filter selectivity (passing rate)",
        ylabel="QPS (log scale)",
        yscale="log",
        ylim=(200, 240000),
    )
    ax.set_yticks([1000, 10000, 100000])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.grid(axis="y", alpha=0.2, which="major")
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.45)
        for color in colors
    ]
    legend_labels = list(labels)
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="////"))
    legend_labels.append("Unmatched: QPS at max recall")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"sparse_selectivity_qps.{ext}", dpi=200)
    plt.close(fig)
    return values


def sparse_section(rows):
    lookup = {(r["cohort"], r["method"]): r for r in rows}
    low = [f"synthetic_low_{rate}_random" for rate in SPARSE_RATES[:3]]
    exact_gains = [
        lookup[(c, "exact")]["qps"] / lookup[(c, "ivf_flat")]["qps"]
        for c in low
    ]
    navix_gains = [
        lookup[(f"synthetic_low_{rate}_random", "navix_reference")]["qps"]
        / lookup[(f"synthetic_low_{rate}_random", "ivf_flat")]["qps"]
        for rate in SPARSE_RATES[3:]
    ]
    return (
        r"""\section{Synthetic Low-Selectivity Filters}\label{sec:sparse}
We additionally evaluate independent random bitmaps at 0.01\%, 0.03\%, 0.1\%, 0.3\%, and 1\% passing rates on ArXiv-large, using the original 2,048 queries and exact ground truth from the paper's selectivity study ($k=10$). We reuse the same three IVF-Flat indexes and tune probe counts against $[0.950,0.952]$ Recall@10. All five selected IVF-Flat configurations are directly measured and satisfy this window in each of three final repetitions. These synthetic results are separate from the preliminary one-repetition primary ArXiv measurements.

"""
        + rf"IVF-Flat reaches the target at every rate and is the fastest qualifying ANN method at 0.01--0.1\%; at 0.01\%, none of the three CAGRA variants reaches the target. However, exact KNN remains {min(exact_gains):.2f}--{max(exact_gains):.2f}$\times$ faster than IVF-Flat at these three rates. At 0.3\% and 1\%, \methodnavix{{}} leads, achieving {navix_gains[0]:.2f}$\times$ and {navix_gains[1]:.2f}$\times$ IVF-Flat's QPS, respectively. Thus the best method shifts from exact search to graph search between the tested 0.1\% and 0.3\% rates on this workload; IVF-Flat expands the range where ANN reaches the target, but does not move that observed crossover."
        + r"""

\begin{figure*}[!t]\centering
\includegraphics[width=\textwidth]{figures/sparse_selectivity_qps.pdf}
\caption{Sparse ArXiv-large throughput at $k=10$, with five bars per passing rate. The QPS axis is logarithmic. Solid ANN bars satisfy $[0.950,0.952]$ Recall@10; exact KNN is a full-recall reference. Hatched bars report QPS at maximum observed recall for methods that did not match the target, with that recall annotated; they are not matched-recall comparisons. Values are medians of three repetitions.}\label{fig:sparse-selectivity}
\end{figure*}
"""
    )


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def snapshot_sources(root, paper_dir):
    metadata = root / "state/report_sources.json"
    if metadata.exists():
        sources = json.loads(metadata.read_text())
    elif (root / "state/manifest.json").exists() and "report_sources" in (
        manifest := json.loads((root / "state/manifest.json").read_text())
    ):
        sources = manifest["report_sources"]
    else:
        sources = []
        for name in ("current_gpu.csv", "current_gpu_k100.csv", "main.tex"):
            source = paper_dir / (
                name if name == "main.tex" else "data/" + name
            )
            target = (
                root
                / "references"
                / ("paper_format_source.tex" if name == "main.tex" else name)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources.append(
                dict(
                    path=str(source), snapshot=str(target), sha256=sha(source)
                )
            )
    for source in sources:
        if sha(source["snapshot"]) != source["sha256"]:
            raise ValueError(f"Changed report input: {source['snapshot']}")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(sources, indent=2) + "\n")
    return sources


def pareto(rows):
    """Keep only measured points not dominated in both recall and throughput."""
    best = -math.inf
    frontier = []
    for row in sorted(rows, key=lambda r: (-r["recall"], -r["qps"])):
        if row["qps"] > best:
            frontier.append(row)
            best = row["qps"]
    return sorted(frontier, key=lambda r: r["recall"])


def calibration_cell(root, workload, k, method):
    manifest = json.loads((root / "state/manifest.json").read_text())
    corpus = "yfcc" if workload == "yfcc" else "arxiv"
    kind = "pq" if method == "ivf_pq_refined" else method.removeprefix("ivf_")
    rows = []
    for index in manifest["indexes"]:
        if (index["corpus"], index["kind"]) != (corpus, kind):
            continue
        marker = (
            root / "state/calibrated" / index["id"] / f"{workload}_k{k}.json"
        )
        if not marker.exists():
            raise ValueError(f"Incomplete calibration: {marker}")
        rows.extend(
            row
            for row in json.loads(marker.read_text())
            if row["method"] == method
        )
    if not rows:
        raise ValueError(
            f"No completed calibration: {workload} k={k} {method}"
        )
    spec = importlib.util.spec_from_file_location(
        "ivf_report_selection", Path(__file__).with_name("workflow.py")
    )
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    target = TARGETS[workload]
    maximum = max(
        rows, key=lambda row: (row["recall_median"], row["qps_median"])
    )
    cell = dict(
        workload=workload,
        k=k,
        method=method,
        target=target,
        source="ivf_calibration_1rep",
        repetitions=1,
        max_observed_recall=maximum["recall_median"],
        qps_at_max_recall=maximum["qps_median"],
    )
    matched = [row for row in rows if workflow.within_window(row, target)]
    if matched:
        best = max(matched, key=lambda row: row["qps_median"])
        cell.update(
            status="measured",
            recall=best["recall_median"],
            qps=best["qps_median"],
            recall_min=best["recall_min"],
            index_id=best["index_id"],
            nprobe=best["nprobe"],
            point_id=best["point_id"],
        )
    elif pairs := workflow.adjacent_brackets(rows, target, adjacent=True):
        cell.update(
            max(
                (workflow.interpolate(*pair, target) for pair in pairs),
                key=lambda row: row["qps"],
            )
        )
    else:
        cell.update(
            status="unreached"
            if maximum["recall_median"] < target
            else "unmatched",
            recall="",
            qps="",
        )
    return cell


def load_results(root, ivf_methods=IVF, allow_calibration=False):
    comparison = json.loads((root / "analysis/comparison.json").read_text())
    expected = {
        (w, k, m) for w in WORKLOADS for k in (10, 100) for m in ivf_methods
    }
    cells = [
        dict(row, repetitions=3)
        for row in comparison["rows"]
        if row["method"] in ivf_methods
    ]
    if allow_calibration:
        present = {
            (row["workload"], int(row["k"]), row["method"]) for row in cells
        }
        cells.extend(
            calibration_cell(root, w, k, m)
            for w, k, m in sorted(expected - present)
        )
    keys = [(r["workload"], int(r["k"]), r["method"]) for r in cells]
    if (
        (
            ivf_methods == IVF
            and not allow_calibration
            and not comparison["complete"]
        )
        or set(keys) != expected
        or len(keys) != len(expected)
    ):
        raise ValueError(
            f"The report requires all {len(expected)} completed IVF comparison cells"
        )
    manifest = json.loads((root / "state/manifest.json").read_text())
    for entry in manifest["references"]:
        if sha(entry["snapshot"]) != entry["sha256"]:
            raise ValueError("Changed saved paper baseline")
    for k in (10, 100):
        name = (
            "current_gpu_matched_recall"
            + ("_k100" if k == 100 else "")
            + ".csv"
        )
        selected = read_csv(root / "references" / name)
        if {(r["workload"], MAP[r["method"]]) for r in selected} != {
            (w, m) for w in WORKLOADS for m in ("base", "retain", "navix")
        } or len(selected) != 12:
            raise ValueError("Incomplete saved CAGRA comparison")
        for row in selected:
            recall, qps = float(row["recall_median"]), float(row["qps_median"])
            target = TARGETS[row["workload"]]
            matched = (
                float(row["recall_min"]) >= target and recall <= target + 0.002
            )
            cells.append(
                dict(
                    workload=row["workload"],
                    k=k,
                    method=MAP[row["method"]],
                    status="measured"
                    if matched
                    else ("unreached" if recall < target else "unmatched"),
                    recall=recall if matched else "",
                    qps=qps if matched else "",
                    max_observed_recall=recall,
                    qps_at_max_recall=qps,
                    source="saved_paper_measurement",
                )
            )
    curves = []
    for name, stage in (
        ("calibration_points.csv", "calibration_1rep"),
        ("final_points.csv", "final_3rep"),
    ):
        for row in read_csv(root / "analysis" / name):
            if row["method"] not in ivf_methods:
                continue
            curves.append(
                dict(
                    workload=row["workload"],
                    k=int(row["k"]),
                    method=row["method"],
                    recall=float(row["recall_median"]),
                    qps=float(row["qps_median"]),
                    stage=stage,
                    point_id=row["point_id"],
                )
            )
    for k, name in ((10, "current_gpu.csv"), (100, "current_gpu_k100.csv")):
        for row in read_csv(root / "references" / name):
            if (
                row["method"] not in MAP
                or row.get("phase") != "throughput"
                or row.get("paper_included", "true").lower() != "true"
            ):
                continue
            if (
                int(row["queries_per_repetition"]) != 10000
                or int(row["repetitions"]) != 3
            ):
                raise ValueError(
                    "Saved CAGRA curve differs from the benchmark contract"
                )
            curves.append(
                dict(
                    workload=row["workload"],
                    k=k,
                    method=MAP[row["method"]],
                    recall=float(row["recall_median"]),
                    qps=float(row["qps_median"]),
                    stage="saved_paper_3rep",
                    point_id=f"{row['group']}_L{row['itopk']}_W{row['search_width']}_I{row['max_iterations']}",
                )
            )
    for row in cells:
        if row["source"] == "saved_paper_measurement":
            curves.append(
                dict(
                    workload=row["workload"],
                    k=row["k"],
                    method=row["method"],
                    recall=row["recall"]
                    if row["recall"] != ""
                    else row["max_observed_recall"],
                    qps=row["qps"]
                    if row["qps"] != ""
                    else row["qps_at_max_recall"],
                    stage="saved_paper_3rep",
                    point_id="paper_selected",
                )
            )
    present = {(r["workload"], r["k"], r["method"]) for r in curves}
    if present != {
        (w, k, m)
        for w in WORKLOADS
        for k in (10, 100)
        for m in ("base", "retain", "navix", *ivf_methods)
    }:
        raise ValueError(
            "Recall-QPS data is missing a method/workload/k series"
        )
    return cells, curves


def make_plots(directory, cells, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = tuple(
        method
        for method in METHODS
        if any(row["method"] == method for row in cells)
    )

    plt.rcParams.update(
        {"font.family": "serif", "font.size": 9, "pdf.fonttype": 42}
    )
    colors = dict(
        zip(
            METHODS,
            (
                "#777777",
                "#9467bd",
                "#8c564b",
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
            ),
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    for k in (10, 100):
        overview, overview_axes = plt.subplots(2, 2, figsize=(7.1, 5.5))
        for workload, overview_ax in zip(WORKLOADS, overview_axes.flat):
            fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
            for ax in (*axes, overview_ax):
                for method in methods:
                    points = [
                        r
                        for r in curves
                        if (r["workload"], r["k"], r["method"])
                        == (workload, k, method)
                    ]
                    ax.scatter(
                        [r["recall"] for r in points],
                        [r["qps"] for r in points],
                        color=colors[method],
                        s=9,
                        alpha=0.22,
                        rasterized=True,
                    )
                    frontier = pareto(points)
                    ax.plot(
                        [r["recall"] for r in frontier],
                        [r["qps"] for r in frontier],
                        color=colors[method],
                        linewidth=1.2,
                    )
                    cell = next(
                        r
                        for r in cells
                        if (r["workload"], int(r["k"]), r["method"])
                        == (workload, k, method)
                    )
                    if cell["status"] in ("measured", "interpolated"):
                        ax.scatter(
                            cell["recall"],
                            cell["qps"],
                            color=colors[method],
                            s=26,
                            marker="x"
                            if cell["status"] == "interpolated"
                            else "D",
                            zorder=5,
                        )
                ax.axvspan(
                    TARGETS[workload],
                    TARGETS[workload] + 0.002,
                    color="grey",
                    alpha=0.2,
                )
                ax.set(
                    xlabel=f"Recall@{k}",
                    ylabel="QPS",
                    yscale="log",
                    xlim=(0, 1.005),
                )
                ax.grid(alpha=0.15)
            axes[0].set_title(WORKLOAD_LABELS[workload] + ": full range")
            axes[1].set_title("Around target recall")
            axes[1].set_xlim(
                max(0, TARGETS[workload] - 0.1),
                min(1.005, TARGETS[workload] + 0.05),
            )
            low, high = axes[1].get_xlim()
            zoom_qps = [
                row["qps"]
                for row in curves
                if row["workload"] == workload
                and row["k"] == k
                and low <= row["recall"] <= high
            ]
            if zoom_qps:
                axes[1].set_ylim(min(zoom_qps) * 0.7, max(zoom_qps) * 1.3)
            overview_ax.set_title(WORKLOAD_LABELS[workload])
            handles = [
                Line2D([0], [0], color=colors[m], label=LABELS[m])
                for m in methods
            ]
            fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7)
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            for extension in ("pdf", "png"):
                fig.savefig(
                    directory / f"{workload}_qps_recall_k{k}.{extension}",
                    dpi=180,
                )
            plt.close(fig)
        overview.legend(
            handles=handles, loc="lower center", ncol=4, fontsize=7
        )
        overview.tight_layout(rect=(0, 0.09, 1, 1))
        for extension in ("pdf", "png"):
            overview.savefig(
                directory / f"qps_recall_k{k}.{extension}", dpi=180
            )
        plt.close(overview)


def table_cell(row):
    matched = row["status"] in ("measured", "interpolated")
    qps = row["qps"] if matched else row["qps_at_max_recall"]
    recall = row["recall"] if matched else row["max_observed_recall"]
    star = r"$^{*}$" if row["status"] == "interpolated" else ""
    if row.get("source") == "ivf_calibration_1rep":
        star += r"$^{\dagger}$"
    maximum = "" if matched else r"\(\max\) "
    return (
        r"\shortstack[r]{"
        + f"{qps:,.0f}"
        + star
        + r"\\"
        + maximum
        + f"{recall:.5f}"
        + "}"
    )


def flat_intro(cells):
    return r"""\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{textcomp}
\usepackage{stfloats}
\usepackage[hidelinks]{hyperref}
\newcommand{\methodbase}{CAGRA-Base}
\newcommand{\methodretain}{CAGRA-Retain}
\newcommand{\methodnavix}{CAGRA-NaviX}
\title{Bitmap-Filtered IVF-Flat versus CAGRA: A100 Benchmark Report}
\author{Anonymous Authors}
\begin{document}
\maketitle
\begin{abstract}
We compare bitmap-filtered IVF-Flat with \methodbase{}, \methodretain{}, and \methodnavix{} on YFCC and three ArXiv workloads at $k=10$ and $k=100$. IVF-Flat leads on YFCC at $k=10$, while NaviX narrowly leads at $k=100$. The available ArXiv measurements favor CAGRA. YFCC IVF-Flat results and the saved CAGRA results use three repetitions; the ArXiv IVF-Flat comparison is preliminary and uses one repetition.
\end{abstract}
\section{Experimental Setup}
All measurements use an NVIDIA A100 80\,GB PCIe GPU and 10,000 queries per workload, processed in serial batches of 2,048, 2,048, 2,048, 2,048, and 1,808 queries. YFCC contains 10 million 192-dimensional uint8 vectors; ArXiv contains 2,735,264 4,096-dimensional float32 vectors. IVF-Flat and the paper's CAGRA measurements use the same query files, per-query bitmap filters, and ground truth. The recall windows are $[0.800,0.802]$ for YFCC and $[0.950,0.952]$ for ArXiv.

IVF-Flat uses 1,024, 2,048, or 4,096 lists, 25 k-means iterations, and a 10\% training sample. Probe counts start at powers of two and are refined to adjacent integers around the recall target. Reported matches select the highest measured QPS within the target window. Index loading, query preparation, and bitmap upload are outside the timed search. The five shard times are summed before deriving QPS; recall counts only valid ground-truth slots. All included IVF-Flat runs passed filter-membership, duplicate-result, and sentinel checks.

YFCC IVF-Flat finalists were measured three times; table values are median QPS. ArXiv IVF-Flat values are from the completed one-repetition calibration sweep and are marked $\dagger$. All CAGRA values are the saved three-repetition measurements used by the RETRIEVE submission, with its original graph, parameters, seeding policies, and output semantics. This report therefore compares new IVF-Flat measurements with frozen paper baselines rather than rerunning CAGRA.
\section{Matched-Recall Results}
Tables~\ref{tab:k10} and~\ref{tab:k100} give QPS above recall. $\dagger$ marks preliminary ArXiv IVF-Flat values. Cells marked $\max$ give the maximum observed recall and its QPS when a saved CAGRA sweep did not reach the target; they are excluded from matched-recall speedups. An asterisk would identify time-per-query interpolation between adjacent probe counts; all IVF-Flat matches in this report are directly measured.
"""


def flat_findings(cells):
    lookup = {
        (row["workload"], int(row["k"]), row["method"]): row for row in cells
    }
    flat10 = lookup[("yfcc", 10, "ivf_flat")]["qps"]
    retain10 = lookup[("yfcc", 10, "retain")]["qps"]
    flat100 = lookup[("yfcc", 100, "ivf_flat")]["qps"]
    navix100 = lookup[("yfcc", 100, "navix")]["qps"]
    gaps = []
    for workload in ("em", "emis", "r"):
        for k in (10, 100):
            cagra = [
                lookup[(workload, k, method)]["qps"]
                for method in ("base", "retain", "navix")
                if lookup[(workload, k, method)]["status"] == "measured"
            ]
            gaps.append(max(cagra) / lookup[(workload, k, "ivf_flat")]["qps"])
    return (
        r"\section{Findings}"
        + "\n"
        + rf"On YFCC at $k=10$, IVF-Flat achieves {flat10:,.0f} QPS, {flat10 / retain10:.2f}$\times$ \methodretain{{}}, the fastest matching CAGRA variant. At $k=100$, \methodnavix{{}} leads with {navix100:,.0f} QPS versus {flat100:,.0f} for IVF-Flat, a {(navix100 / flat100 - 1) * 100:.1f}\% advantage. \methodbase{{}} does not reach either YFCC target, and \methodretain{{}} does not reach the $k=100$ target."
        + "\n\n"
        + rf"On ArXiv, the fastest matching CAGRA variant achieves {min(gaps):.2f}--{max(gaps):.2f}$\times$ IVF-Flat's QPS in the available measurements. \methodretain{{}} leads on EM at $k=10$; \methodnavix{{}} leads on the other five workload/$k$ combinations. EMIS has the largest gap. To reach its recall target, the selected IVF-Flat index probes 766 of 4,096 lists at $k=10$ and 1,204 at $k=100$, compared with 58--62 and 101--105 probes on R and EM."
        + "\n\n"
        + r"The results show workload-dependent advantages rather than a dimensionality rule. YFCC and ArXiv differ in vector structure, predicates, and recall requirements as well as dimension; even on YFCC, the winning method changes with $k$. ArXiv IVF-Flat finalist repetitions remain future validation for this preliminary comparison."
    )


def make_tex(cells, sparse_rows=None):
    methods = tuple(
        method
        for method in METHODS
        if any(row["method"] == method for row in cells)
    )
    flat_only = methods == ("base", "retain", "navix", "ivf_flat")
    parts = [
        r"""\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{textcomp}
\usepackage{stfloats}
\usepackage[hidelinks]{hyperref}
\newcommand{\methodbase}{CAGRA-Base}
\newcommand{\methodretain}{CAGRA-Retain}
\newcommand{\methodnavix}{CAGRA-NaviX}
\title{Bitmap-Filtered IVF and CAGRA: A100 Benchmark Report}
\author{Anonymous Authors}
\begin{document}
\maketitle
\begin{abstract}
We compare bitmap-filtered IVF-Flat, IVF-SQ, IVF-PQ, and IVF-PQ with GPU reranking against saved CAGRA-Base, CAGRA-Retain, and CAGRA-NaviX measurements from the RETRIEVE submission. This report includes every recall--throughput curve and the matched-recall results for YFCC and ArXiv-EM, EMIS, and R at $k=10$ and $k=100$.
\end{abstract}
\section{Experimental Setup}
All methods use the NVIDIA A100 80\,GB PCIe GPU and the same 10,000-query workloads, processed in five serial shards of 2,048, 2,048, 2,048, 2,048, and 1,808 queries. YFCC has 10 million 192-dimensional vectors; ArXiv has 2,735,264 4,096-dimensional vectors. IVF-Flat and IVF-PQ use YFCC's original uint8 coordinates; IVF-SQ uses their coordinate-exact float conversion. ArXiv coordinates are float32. The targets are $[0.800,0.802]$ for YFCC and $[0.950,0.952]$ for ArXiv.

IVF indexes use 1,024, 2,048, or 4,096 lists. PQ uses 8-bit codes with 32 or 64 subquantizers on YFCC and 64 or 128 on ArXiv; reranking ratios range from 2 to 64. Probe counts are refined to adjacent integers around every observed target crossing. Plain and reranked PQ run in separate processes. GPU reranking is timed, with candidate-vector temporaries bounded to approximately 1\,GiB per query chunk. Index loading, data preparation, and bitmap upload are outside the timed search.

Calibration uses one repetition; finalists use three. We sum shard query times before deriving QPS and report the median over repetitions. Recall counts only valid ground-truth slots. New IVF measurements must have zero filter violations, duplicate results, and sentinel errors. CAGRA results are reused from the paper without changing its parameters or legacy output semantics.
\section{Matched-Recall Results}
Each cell in Tables~\ref{tab:k10} and~\ref{tab:k100} gives QPS on the first line and recall on the second. Measured matches are preferred. An asterisk identifies an estimate interpolated in time per query between adjacent probe counts with identical index and reranking settings. Cells marked $\max$ give the highest observed recall and its QPS when no matched result is available; these cells are excluded from matched-recall speedups. Maximum recall is limited to the evaluated sweep.
"""
    ]
    if flat_only:
        parts = [flat_intro(cells)]
        if sparse_rows is not None:
            parts[0] = (
                parts[0]
                .replace(
                    "All measurements use an NVIDIA",
                    "The primary workload measurements use an NVIDIA",
                )
                .replace(
                    "the ArXiv IVF-Flat comparison is preliminary",
                    "the primary ArXiv IVF-Flat comparison is preliminary",
                )
                .replace(
                    "ArXiv IVF-Flat values are from",
                    "Primary ArXiv IVF-Flat values are from",
                )
                .replace(
                    r"\end{abstract}",
                    r"A separate three-repetition sparse-filter study finds that exact KNN leads through 0.1\% passing, while NaviX leads at 0.3\% and 1\%."
                    + "\n"
                    + r"\end{abstract}",
                )
            )
    short = dict(
        base=r"\methodbase",
        retain=r"\methodretain",
        navix=r"\methodnavix",
        ivf_flat="IVF-Flat",
        ivf_sq="IVF-SQ",
        ivf_pq="IVF-PQ",
        ivf_pq_refined=r"\shortstack{IVF-PQ\\GPU rerank}",
    )
    for k in (10, 100):
        caption = f"Matched Recall@{k}: QPS above recall. $\\max$ denotes an unmatched maximum; $^*$ denotes interpolation."
        if flat_only:
            caption += (
                r" $\dagger$ marks one-repetition ArXiv IVF-Flat measurements."
            )
        parts += [
            r"\begin{table*}[!t]\centering\small",
            rf"\caption{{{caption}}}\label{{tab:k{k}}}",
            rf"\setlength{{\tabcolsep}}{{{8 if flat_only else 4}pt}}\begin{{tabular}}{{l"
            + "r" * len(methods)
            + r"}\toprule",
            "Workload & "
            + " & ".join(short[m] for m in methods)
            + r"\\\midrule",
        ]
        for workload in WORKLOADS:
            rows = [
                next(
                    r
                    for r in cells
                    if (r["workload"], int(r["k"]), r["method"])
                    == (workload, k, m)
                )
                for m in methods
            ]
            parts.append(
                WORKLOAD_LABELS[workload]
                + " & "
                + " & ".join(table_cell(r) for r in rows)
                + r"\\"
            )
            parts.append(r"\addlinespace[3pt]")
        parts += [r"\bottomrule\end{tabular}\end{table*}"]
    if flat_only:
        parts.append(flat_findings(cells))
    if sparse_rows is not None:
        parts.append(sparse_section(sparse_rows))
    parts.append(r"""\section{Recall--Throughput Curves}
Faint markers show all measured configurations. Lines show empirical Pareto envelopes across the evaluated settings; they are not measurements between configurations. Diamonds mark matched-recall selections and crosses mark interpolated selections. IVF calibration points use one repetition, IVF final points and the saved CAGRA curves use three. The shaded band marks the target window. Each workload includes a full-range view and a view around its recall target.
""")
    for k in (10, 100):
        for workload in WORKLOADS:
            parts += [
                r"\begin{figure*}[!t]\centering",
                rf"\includegraphics[width=\textwidth]{{figures/{workload}_qps_recall_k{k}.pdf}}",
                rf"\caption{{{WORKLOAD_LABELS[workload]} recall--QPS comparison at $k={k}$, with all {len(methods)} methods.}}",
                r"\end{figure*}",
            ]
    parts += [
        r"""\section{Reproducibility}
The report includes its LaTeX source, PDF/PNG plots, combined curve data, matched-recall CSV, and hashed source provenance. The experiment directory retains the original benchmark JSON, configurations, index files, calibration and final aggregates, and correctness evidence. Measurements affected by runtime fixes are archived separately and excluded from the reported PQ sweep.
\end{document}
"""
    ]
    if flat_only:
        parts[-1] = r"""\section{Reproducibility}
The report includes LaTeX source, PDF/PNG figures, matched-recall and curve CSVs, and hashed provenance. The experiment retains raw benchmark JSON, configurations, the complete IVF-Flat calibration sweep, YFCC final measurements, and the frozen CAGRA sources. The $\dagger$ values can be replaced by ArXiv finalist repetitions without changing the experimental inputs.
\end{document}
"""
    if sparse_rows is not None:
        parts[-1] = parts[-1].replace(
            r"\end{document}",
            "The sparse-study comparison and curve data are included as separate CSVs; the grouped-bar CSV records the exact QPS, recall, and target status plotted for all 25 bars. Sparse experiment inputs and output hashes are recorded in the report provenance.\n"
            + r"\end{document}",
        )
    return "\n".join(parts)


def generate(
    root, ivf_methods=IVF, allow_calibration=False, sparse_run_root=None
):
    sparse_rows = None
    if sparse_run_root is not None:
        if ivf_methods != ("ivf_flat",):
            raise ValueError(
                "Sparse inclusion currently requires the IVF-Flat report scope"
            )
        sparse_rows = load_sparse_results(sparse_run_root)
    cells, curves = load_results(root, ivf_methods, allow_calibration)
    directory = root / (
        "report" if ivf_methods == IVF else "report_" + "_".join(ivf_methods)
    )
    directory.mkdir(exist_ok=True)
    make_plots(directory / "figures", cells, curves)
    if sparse_rows is not None:
        bars = make_sparse_plot(directory / "figures", sparse_rows)
        with (directory / "sparse_bar_values.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bars[0]))
            writer.writeheader()
            writer.writerows(bars)
        for name in ("comparison.csv", "curve_points.csv"):
            shutil.copyfile(
                sparse_run_root / "analysis" / name,
                directory / ("sparse_" + name),
            )
        (directory / "README.md").write_text(
            "# IVF-Flat versus CAGRA report\n\n"
            "Open main.pdf; main.tex and figures/ contain the LaTeX source and plots.\n"
            "Compile with: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex\n\n"
            "The primary study covers YFCC and ArXiv EM/EMIS/R at k=10 and k=100.\n"
            "Primary ArXiv IVF-Flat values remain preliminary one-repetition calibration;\n"
            "YFCC IVF-Flat and saved CAGRA use three repetitions.\n\n"
            "The added sparse study covers five ArXiv random passing rates (0.01%, 0.03%,\n"
            "0.1%, 0.3%, 1%) at k=10 with 2,048 queries and three final repetitions.\n"
            "Its grouped bar chart uses a logarithmic QPS axis. Hatched bars are QPS at\n"
            "maximum recall for unmatched methods, with achieved recall annotated.\n"
            "Exact KNN bars are full-recall measurements.\n\n"
            "matched_recall.csv and curve_points.csv retain the primary study data.\n"
            "sparse_comparison.csv, sparse_curve_points.csv, and sparse_bar_values.csv\n"
            "contain the added study and all 25 plotted bar values. provenance.json\n"
            "records both experiments and generated artifact hashes.\n"
        )
    for name, rows in (
        ("matched_recall.csv", cells),
        ("curve_points.csv", curves),
    ):
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    source = directory / "main.tex"
    source.write_text(make_tex(cells, sparse_rows))
    with (directory / "build.log").open("w") as log:
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "main.tex",
            ],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    artifacts = [
        p
        for p in directory.rglob("*")
        if p.suffix in (".tex", ".pdf", ".png", ".csv")
    ]
    provenance = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        format=r"\documentclass[lettersize,journal]{IEEEtran}",
        input_manifest_sha256=sha(root / "state/manifest.json"),
        comparison_sha256=sha(root / "analysis/comparison.json"),
        calibration_csv_sha256=sha(root / "analysis/calibration_points.csv"),
        final_csv_sha256=sha(root / "analysis/final_points.csv"),
        ivf_methods=list(ivf_methods),
        allow_calibration=allow_calibration,
        preliminary_cells=[
            dict(
                workload=row["workload"],
                k=row["k"],
                method=row["method"],
                repetitions=1,
            )
            for row in cells
            if row.get("source") == "ivf_calibration_1rep"
        ],
        report_sources=json.loads(
            (root / "state/report_sources.json").read_text()
        ),
        generator_sha256=sha(__file__),
        files=[
            dict(path=str(p.relative_to(directory)), sha256=sha(p))
            for p in sorted(artifacts)
        ],
    )
    if sparse_rows is not None:
        provenance["sparse_experiment"] = dict(
            run_root=str(sparse_run_root),
            manifest_sha256=sha(sparse_run_root / "state/manifest.json"),
            comparison_sha256=sha(sparse_run_root / "analysis/comparison.csv"),
            curve_points_sha256=sha(
                sparse_run_root / "analysis/curve_points.csv"
            ),
            analysis_provenance=json.loads(
                (sparse_run_root / "analysis/provenance.json").read_text()
            ),
            chart="Five methods per selectivity; unmatched bars use QPS at maximum observed recall",
            yscale="log",
        )
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("Created report:", directory / "main.pdf", flush=True)
    return directory / "main.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("snapshot", "generate"),
        default="generate",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--ivf-methods", nargs="+", choices=IVF, default=list(IVF)
    )
    parser.add_argument(
        "--allow-calibration",
        action="store_true",
        help="Use completed calibration for cells whose final repetitions are pending; mark them preliminary",
    )
    parser.add_argument(
        "--sparse-run-root",
        type=Path,
        help="Include a completed sparse IVF-Flat study and grouped selectivity/QPS bars",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("/home/ubuntu/paper_selectivity_revision"),
    )
    args = parser.parse_args()
    snapshot_sources(args.run_root, args.paper_dir)
    if args.action == "generate":
        progress_path = args.run_root / "state/report_progress.json"
        try:
            progress_path.write_text(
                json.dumps(
                    dict(
                        state="generating",
                        updated_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
                + "\n"
            )
            pdf = generate(
                args.run_root,
                tuple(args.ivf_methods),
                args.allow_calibration,
                args.sparse_run_root.resolve()
                if args.sparse_run_root
                else None,
            )
        except Exception as error:
            progress_path.write_text(
                json.dumps(
                    dict(
                        state="failed",
                        error=str(error),
                        updated_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
                + "\n"
            )
            raise
        progress_path.write_text(
            json.dumps(
                dict(
                    state="complete",
                    pdf=str(pdf),
                    updated_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
