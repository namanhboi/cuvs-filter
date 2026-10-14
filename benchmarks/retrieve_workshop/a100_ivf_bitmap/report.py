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


def make_tex(cells):
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
    return "\n".join(parts)


def generate(root, ivf_methods=IVF, allow_calibration=False):
    cells, curves = load_results(root, ivf_methods, allow_calibration)
    directory = root / (
        "report" if ivf_methods == IVF else "report_" + "_".join(ivf_methods)
    )
    directory.mkdir(exist_ok=True)
    make_plots(directory / "figures", cells, curves)
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
    source.write_text(make_tex(cells))
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
                args.run_root, tuple(args.ivf_methods), args.allow_calibration
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
