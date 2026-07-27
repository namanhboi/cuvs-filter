#!/usr/bin/env python3
"""Generate, validate, summarize, and plot static FAVOR penalty experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from statistics import median


DEFAULT_SELECTIVITIES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 50, 90)
DEFAULT_ITOPK_VALUES = (32, 64, 128, 256, 512)
DEFAULT_SEARCH_WIDTHS = (1, 2, 4)
DEFAULT_BATCH_SIZES = (10, 10_000)
ALL_FORMULAS = ("current", "zero", "hard05", "hard10", "smooth2", "smooth4")

DATASETS = {
    "sift": {
        "name": "sift-128-euclidean",
        "title": "SIFT-1M",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
    },
    "gist": {
        "name": "gist-960-euclidean",
        "title": "GIST-1M",
        "base": "base.fbin",
        "query": "query_10000.fbin",
        "dtype": "float",
    },
    "bigann1m": {
        "name": "bigann-1M",
        "title": "BIGANN-1M",
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
        "subset_size": 1_000_000,
    },
    "bigann10m": {
        "name": "bigann-10M",
        "title": "BIGANN-10M",
        "base": "base.10M.u8bin",
        "query": "query.public.10K.u8bin",
        "dtype": "uint8",
    },
    "msturing1m": {
        "name": "msturing-1M",
        "title": "MSTuring-1M",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
    },
    "msturing10m": {
        "name": "msturing-10M",
        "title": "MSTuring-10M",
        "base": "base.fbin",
        "query": "query.fbin",
        "dtype": "float",
    },
}

FORMULA_LABELS = {
    "default": "Default CAGRA",
    "current": "FAVOR: current penalty",
    "zero": "FAVOR: zero penalty",
    "hard05": "FAVOR: hard gate, epsilon=0.05",
    "hard10": "FAVOR: hard gate, epsilon=0.10",
    "smooth2": "FAVOR: smooth gate, r=2",
    "smooth4": "FAVOR: smooth gate, r=4",
}


def read_delta_d(path: Path) -> float:
    """Read delta-d from the validated cuVS sidecar layout."""
    payload = path.read_bytes()
    if len(payload) != 80 or payload[:8] != b"CUVSDD\r\n":
        raise ValueError(f"invalid delta-d sidecar: {path}")
    value = struct.unpack_from("<f", payload, 52)[0]
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid delta-d value {value!r} in {path}")
    return value


def binomial_shortage_probability(buffer_size: int, selectivity: float, k: int) -> float:
    """Return Pr[Binomial(buffer_size, selectivity) < k]."""
    if buffer_size < 0:
        raise ValueError("buffer_size must be nonnegative")
    if not 0.0 <= selectivity <= 1.0:
        raise ValueError("selectivity must be in [0, 1]")
    if k <= 0:
        return 0.0
    if k > buffer_size:
        return 1.0
    if selectivity == 0.0:
        return 1.0
    if selectivity == 1.0:
        return 0.0

    log_p = math.log(selectivity)
    log_q = math.log1p(-selectivity)
    log_n_factorial = math.lgamma(buffer_size + 1)
    terms = []
    for passing in range(k):
        log_probability = (
            log_n_factorial
            - math.lgamma(passing + 1)
            - math.lgamma(buffer_size - passing + 1)
            + passing * log_p
            + (buffer_size - passing) * log_q
        )
        terms.append(math.exp(log_probability))
    return min(1.0, max(0.0, math.fsum(terms)))


def formula_multiplier(formula: str, p_short: float) -> float:
    if formula == "current":
        return 1.0
    if formula == "zero":
        return 0.0
    if formula == "hard05":
        return 1.0 if p_short > 0.05 else 0.0
    if formula == "hard10":
        return 1.0 if p_short > 0.10 else 0.0
    if formula == "smooth2":
        return p_short**2
    if formula == "smooth4":
        return p_short**4
    raise ValueError(f"unknown formula: {formula}")


def current_penalty(delta_d: float, selectivity: float, buffer_size: int) -> float:
    if selectivity <= 0.0:
        return math.inf
    return (
        (1.0 - selectivity)
        * (float(buffer_size) - selectivity)
        * delta_d
        / (2.0 * selectivity * float(buffer_size))
    )


def dataset_config(metadata: dict, selectivity_percent: int) -> dict:
    name = metadata["name"]
    dataset = {
        "name": f"{name}-static-penalty-s{selectivity_percent:02d}",
        "base_file": f"{name}/{metadata['base']}",
        "query_file": f"{name}/{metadata['query']}",
        "groundtruth_neighbors_file": (
            f"{name}/favor/groundtruth_s{selectivity_percent:02d}.ibin"
        ),
        "filter_bitset_file": f"{name}/favor/filter_s{selectivity_percent:02d}.bin",
        "distance": "euclidean",
        "dtype": metadata["dtype"],
    }
    if "subset_size" in metadata:
        dataset["subset_size"] = metadata["subset_size"]
    return dataset


def generate(args: argparse.Namespace) -> None:
    config_dir = args.result_dir / "configs"
    raw_dir = args.result_dir / "raw"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    formulas = tuple(dict.fromkeys(args.formulas))
    unknown = sorted(set(formulas) - set(ALL_FORMULAS))
    if unknown:
        raise ValueError(f"unknown formulas: {', '.join(unknown)}")

    manifest = {
        "version": 1,
        "model": {
            "description": "X ~ Binomial(B, p); P_short = Pr[X < k]",
            "kernel_formula": (
                "D = (1-p) * (B-p) * favor_delta_d / (2*p*B)"
            ),
            "formulas": {name: FORMULA_LABELS[name] for name in formulas},
        },
        "configs": {},
    }

    for dataset_key in args.datasets:
        metadata = DATASETS[dataset_key]
        delta_path = (
            args.data_dir / metadata["name"] / "cagra_g32_ig64.index.delta_d"
        )
        delta_d = read_delta_d(delta_path)
        for selectivity_percent in args.selectivities:
            if not 1 <= selectivity_percent <= 100:
                raise ValueError("selectivities must be whole percentages in [1, 100]")
            selectivity = selectivity_percent / 100.0
            for batch_size in args.batch_sizes:
                params = []
                entries = []
                for buffer_size in args.itopk_values:
                    p_short = binomial_shortage_probability(
                        buffer_size, selectivity, args.k
                    )
                    base_penalty = current_penalty(
                        delta_d, selectivity, buffer_size
                    )
                    for width in args.search_widths:
                        params.append(
                            {
                                "algo": "single_cta",
                                "filter_mode": "default",
                                "itopk": buffer_size,
                                "search_width": width,
                            }
                        )
                        entries.append(
                            {
                                "param_index": len(params) - 1,
                                "formula": "default",
                                "formula_label": FORMULA_LABELS["default"],
                                "B": buffer_size,
                                "p": selectivity,
                                "k": args.k,
                                "P_short": p_short,
                                "multiplier": None,
                                "delta_d": delta_d,
                                "favor_delta_d": None,
                                "D_current": base_penalty,
                                "D_test": None,
                                "search_width": width,
                            }
                        )
                    for formula in formulas:
                        multiplier = formula_multiplier(formula, p_short)
                        scaled_delta = multiplier * delta_d
                        for width in args.search_widths:
                            params.append(
                                {
                                    "algo": "single_cta",
                                    "filter_mode": "favor",
                                    "favor_delta_d": scaled_delta,
                                    "itopk": buffer_size,
                                    "search_width": width,
                                }
                            )
                            entries.append(
                                {
                                    "param_index": len(params) - 1,
                                    "formula": formula,
                                    "formula_label": FORMULA_LABELS[formula],
                                    "B": buffer_size,
                                    "p": selectivity,
                                    "k": args.k,
                                    "P_short": p_short,
                                    "multiplier": multiplier,
                                    "delta_d": delta_d,
                                    "favor_delta_d": scaled_delta,
                                    "D_current": base_penalty,
                                    "D_test": multiplier * base_penalty,
                                    "search_width": width,
                                }
                            )

                config = {
                    "dataset": dataset_config(metadata, selectivity_percent),
                    "search_basic_param": {"batch_size": batch_size, "k": args.k},
                    "index": [
                        {
                            "name": "cagra-g32-ig64",
                            "algo": "cuvs_cagra",
                            "file": f"{metadata['name']}/cagra_g32_ig64.index",
                            "build_param": {
                                "graph_build_algo": "NN_DESCENT",
                                "graph_degree": 32,
                                "intermediate_graph_degree": 64,
                            },
                            "search_params": params,
                        }
                    ],
                }
                filename = (
                    f"{dataset_key}_s{selectivity_percent:02d}_nq{batch_size}.json"
                )
                (config_dir / filename).write_text(
                    json.dumps(config, indent=2) + "\n"
                )
                manifest["configs"][filename] = {
                    "dataset": dataset_key,
                    "dataset_name": metadata["name"],
                    "dataset_title": metadata["title"],
                    "selectivity_percent": selectivity_percent,
                    "batch_size": batch_size,
                    "k": args.k,
                    "delta_d_file": str(delta_path),
                    "entries": entries,
                }

    (args.result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"generated {len(manifest['configs'])} configurations in {config_dir}"
    )


def result_index(row: dict) -> int:
    match = re.search(r"/(\d+)/", row["run_name"])
    if not match:
        raise ValueError(f"cannot parse parameter index from {row['run_name']!r}")
    return int(match.group(1))


def summarize(args: argparse.Namespace) -> None:
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    rows = []
    missing = []
    for filename, config_record in manifest["configs"].items():
        result_path = args.result_dir / "raw" / filename
        if not result_path.exists():
            missing.append(filename)
            continue
        payload = json.loads(result_path.read_text())
        iterations = [
            row
            for row in payload["benchmarks"]
            if row.get("run_type", "iteration") == "iteration"
            and not row.get("error_occurred", False)
        ]
        grouped = defaultdict(list)
        for row in iterations:
            grouped[result_index(row)].append(row)
        expected = len(config_record["entries"])
        if len(grouped) != expected:
            raise ValueError(
                f"{result_path}: expected {expected} parameter results, "
                f"found {len(grouped)}"
            )

        workload = (
            "throughput" if config_record["batch_size"] == 10_000 else "latency"
        )
        metric = "items_per_second" if workload == "throughput" else "Latency"
        for entry in config_record["entries"]:
            repetitions = grouped[entry["param_index"]]
            rows.append(
                {
                    "dataset": config_record["dataset"],
                    "dataset_title": config_record["dataset_title"],
                    "selectivity_percent": config_record["selectivity_percent"],
                    "batch_size": config_record["batch_size"],
                    "workload": workload,
                    "formula": entry["formula"],
                    "formula_label": entry["formula_label"],
                    "B": entry["B"],
                    "p": entry["p"],
                    "k": entry["k"],
                    "P_short": entry["P_short"],
                    "multiplier": (
                        "" if entry["multiplier"] is None else entry["multiplier"]
                    ),
                    "delta_d": entry["delta_d"],
                    "favor_delta_d": (
                        ""
                        if entry["favor_delta_d"] is None
                        else entry["favor_delta_d"]
                    ),
                    "D_current": entry["D_current"],
                    "D_test": "" if entry["D_test"] is None else entry["D_test"],
                    "search_width": entry["search_width"],
                    "recall": median(float(row["Recall"]) for row in repetitions),
                    "value": median(float(row[metric]) for row in repetitions),
                }
            )

    if not rows:
        raise ValueError("no complete benchmark results found")
    output_path = args.result_dir / "summary.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")
    if missing:
        print(f"skipped {len(missing)} missing result files")


def pareto(points: list[dict], maximize: bool) -> list[dict]:
    ordered = sorted(points, key=lambda point: float(point["recall"]), reverse=True)
    frontier = []
    best = -math.inf if maximize else math.inf
    for point in ordered:
        value = float(point["value"])
        if (maximize and value > best) or (not maximize and value < best):
            frontier.append(point)
            best = value
    return sorted(frontier, key=lambda point: float(point["recall"]))


def plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    summary_path = args.result_dir / "summary.csv"
    with summary_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    plot_dir = args.result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["dataset_title"],
                int(row["selectivity_percent"]),
                row["workload"],
            )
        ].append(row)

    colors = {
        "default": "#1f77b4",
        "current": "#d62728",
        "zero": "#2ca02c",
        "hard05": "#9467bd",
        "hard10": "#8c564b",
        "smooth2": "#e377c2",
        "smooth4": "#ff7f0e",
    }
    for (dataset, title, selectivity, workload), group in grouped.items():
        fig, axis = plt.subplots(figsize=(8.5, 6.0))
        by_formula = defaultdict(list)
        for row in group:
            by_formula[row["formula"]].append(row)
        for formula in ("default", *ALL_FORMULAS):
            if formula not in by_formula:
                continue
            frontier = pareto(
                by_formula[formula], maximize=(workload == "throughput")
            )
            scale = 1.0 if workload == "throughput" else 1000.0
            axis.plot(
                [float(point["recall"]) for point in frontier],
                [scale * float(point["value"]) for point in frontier],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=colors[formula],
                label=FORMULA_LABELS[formula],
            )
        axis.set_xlabel("Recall@10")
        if workload == "throughput":
            axis.set_ylabel("Throughput (queries/second, higher is better)")
            workload_title = "Large batch: 10,000 queries"
        else:
            axis.set_ylabel("Batch latency (milliseconds, lower is better)")
            workload_title = "Low batch: 10 queries"
        axis.set_title(
            f"{title}: {selectivity}% selectivity\n"
            f"{workload_title}; graph degree 32, intermediate degree 64"
        )
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"{dataset}_s{selectivity:02d}_{workload}.png", dpi=180
        )
        plt.close(fig)
    print(f"wrote {len(grouped)} plots to {plot_dir}")


def compare(args: argparse.Namespace) -> None:
    with (args.result_dir / "summary.csv").open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["workload"] == "throughput"
        ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                int(row["selectivity_percent"]),
                row["formula"],
            )
        ].append(row)

    cell_records = {}
    for (dataset, selectivity, formula), points in grouped.items():
        max_recall = max(float(point["recall"]) for point in points)
        eligible90 = [
            float(point["value"])
            for point in points
            if float(point["recall"]) >= 0.90
        ]
        eligible95 = [
            float(point["value"])
            for point in points
            if float(point["recall"]) >= 0.95
        ]
        cell_records[(dataset, selectivity, formula)] = {
            "dataset": dataset,
            "selectivity_percent": selectivity,
            "formula": formula,
            "max_recall": max_recall,
            "best_qps_recall_90": max(eligible90) if eligible90 else None,
            "best_qps_recall_95": max(eligible95) if eligible95 else None,
        }

    cells = sorted({(key[0], key[1]) for key in cell_records})
    formulas = sorted({key[2] for key in cell_records})
    output = []
    for dataset, selectivity in cells:
        baseline_records = [
            cell_records[(dataset, selectivity, formula)]
            for formula in ("default", "current", "zero")
            if (dataset, selectivity, formula) in cell_records
        ]
        baseline_best_recall = max(
            record["max_recall"] for record in baseline_records
        )
        available = [
            cell_records[(dataset, selectivity, formula)]
            for formula in formulas
            if (dataset, selectivity, formula) in cell_records
        ]
        best_qps90 = max(
            (
                record["best_qps_recall_90"]
                for record in available
                if record["best_qps_recall_90"] is not None
            ),
            default=None,
        )
        best_qps95 = max(
            (
                record["best_qps_recall_95"]
                for record in available
                if record["best_qps_recall_95"] is not None
            ),
            default=None,
        )
        for record in available:
            qps90 = record["best_qps_recall_90"]
            qps95 = record["best_qps_recall_95"]
            output.append(
                {
                    **record,
                    "baseline_best_max_recall": baseline_best_recall,
                    "recall_shortfall": max(
                        0.0, baseline_best_recall - record["max_recall"]
                    ),
                    "passes_recall_constraint": (
                        baseline_best_recall < 0.90
                        or baseline_best_recall - record["max_recall"] <= 0.01
                    ),
                    "best_cell_qps_recall_90": (
                        "" if best_qps90 is None else best_qps90
                    ),
                    "qps_regret_recall_90": (
                        ""
                        if best_qps90 is None or qps90 is None
                        else (best_qps90 - qps90) / best_qps90
                    ),
                    "best_cell_qps_recall_95": (
                        "" if best_qps95 is None else best_qps95
                    ),
                    "qps_regret_recall_95": (
                        ""
                        if best_qps95 is None or qps95 is None
                        else (best_qps95 - qps95) / best_qps95
                    ),
                    "best_qps_recall_90": "" if qps90 is None else qps90,
                    "best_qps_recall_95": "" if qps95 is None else qps95,
                }
            )
    output.sort(
        key=lambda row: (
            row["dataset"],
            row["selectivity_percent"],
            row["formula"],
        )
    )
    path = args.result_dir / "formula_comparison.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {path}")

    candidate_formulas = [
        formula
        for formula in formulas
        if formula not in {"default", "current", "zero"}
    ]
    selection = {"recall_target": 0.90, "recall_tolerance": 0.01, "candidates": {}}
    for formula in candidate_formulas:
        records = [record for record in output if record["formula"] == formula]
        invalid_cells = [
            {
                "dataset": record["dataset"],
                "selectivity_percent": record["selectivity_percent"],
                "baseline_best_max_recall": record["baseline_best_max_recall"],
                "candidate_max_recall": record["max_recall"],
                "shortfall": record["recall_shortfall"],
            }
            for record in records
            if not record["passes_recall_constraint"]
        ]
        missing_qps_cells = [
            {
                "dataset": record["dataset"],
                "selectivity_percent": record["selectivity_percent"],
            }
            for record in records
            if record["best_cell_qps_recall_90"] != ""
            and record["best_qps_recall_90"] == ""
        ]
        regrets = [
            float(record["qps_regret_recall_90"])
            for record in records
            if record["qps_regret_recall_90"] != ""
        ]
        selection["candidates"][formula] = {
            "survives": not invalid_cells and not missing_qps_cells,
            "invalid_recall_cells": invalid_cells,
            "missing_qps_at_recall_90_cells": missing_qps_cells,
            "worst_qps_regret_recall_90": max(regrets) if regrets else None,
            "median_qps_regret_recall_90": median(regrets) if regrets else None,
        }

    survivors = [
        (formula, record)
        for formula, record in selection["candidates"].items()
        if record["survives"]
    ]
    if survivors:
        survivors.sort(
            key=lambda item: (
                item[1]["worst_qps_regret_recall_90"],
                item[1]["median_qps_regret_recall_90"],
                item[0],
            )
        )
        selection["winner"] = survivors[0][0]
        selection["conclusion"] = "static formula selected"
    else:
        selection["winner"] = None
        selection["conclusion"] = (
            "no static formula satisfies the recall and QPS coverage constraints"
        )
    selection_path = args.result_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"wrote {selection_path}")


def self_check(_: argparse.Namespace) -> None:
    expected = {
        0.01: 0.9643819583,
        0.02: 0.4267572771,
        0.03: 0.0563239237,
    }
    for selectivity, target in expected.items():
        actual = binomial_shortage_probability(512, selectivity, 10)
        if not math.isclose(actual, target, rel_tol=0.0, abs_tol=5e-10):
            raise AssertionError(
                f"B=512 p={selectivity} k=10: expected {target}, got {actual}"
            )
    if binomial_shortage_probability(512, 0.10, 10) >= 1e-12:
        raise AssertionError("10% shortage probability should be effectively zero")
    print("static penalty formula checks passed")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--result-dir", type=Path, required=True)
    generate_parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    generate_parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, required=True
    )
    generate_parser.add_argument(
        "--selectivities",
        type=int,
        nargs="+",
        default=DEFAULT_SELECTIVITIES,
    )
    generate_parser.add_argument(
        "--itopk-values", type=int, nargs="+", default=DEFAULT_ITOPK_VALUES
    )
    generate_parser.add_argument(
        "--search-widths", type=int, nargs="+", default=DEFAULT_SEARCH_WIDTHS
    )
    generate_parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES
    )
    generate_parser.add_argument(
        "--formulas", nargs="+", default=ALL_FORMULAS
    )
    generate_parser.add_argument("--k", type=int, default=10)
    generate_parser.set_defaults(function=generate)

    for name, function in (
        ("summarize", summarize),
        ("plot", plot),
        ("compare", compare),
    ):
        command = commands.add_parser(name)
        command.add_argument("--result-dir", type=Path, required=True)
        command.set_defaults(function=function)

    check_parser = commands.add_parser("check")
    check_parser.set_defaults(function=self_check)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
