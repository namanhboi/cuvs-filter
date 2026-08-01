#!/usr/bin/env python3
"""Run the bounded 1%-selectivity SINGLE_CTA diagnostic matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
BENCH = ROOT / "cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
TRACE_TOOL = ROOT / "benchmarks/favor/favor_trace.py"

DATASETS = {
    "sift": {
        "dir": "sift-128-euclidean", "base": "base.fbin", "query": "query.fbin",
        "dtype": "float", "width": 4,
    },
    "gist": {
        "dir": "gist-960-euclidean", "base": "base.fbin", "query": "query_10000.fbin",
        "dtype": "float", "width": 4, "expanded": 256,
    },
    "bigann1m": {
        "dir": "bigann-1M", "base": "base.10M.u8bin", "query": "query.public.10K.u8bin",
        "dtype": "uint8", "width": 1, "subset_size": 1_000_000,
    },
    "bigann10m": {
        "dir": "bigann-10M", "base": "base.10M.u8bin", "query": "query.public.10K.u8bin",
        "dtype": "uint8", "width": 1,
    },
    "msturing1m": {
        "dir": "msturing-1M", "base": "base.fbin", "query": "query.fbin",
        "dtype": "float", "width": 1, "expanded": 2048,
    },
    "msturing10m": {
        "dir": "msturing-10M", "base": "base.fbin", "query": "query.fbin",
        "dtype": "float", "width": 1, "expanded": 3584,
    },
}


def make_config(dataset_key: str, variant: str, output: Path, selected: Path | None = None) -> dict:
    info = DATASETS[dataset_key]
    dataset_dir = DATA / info["dir"]
    dataset = {
        "name": f"{dataset_key}-s01-diagnostic-{variant}",
        "base_file": f"{info['dir']}/{info['base']}",
        "query_file": f"{info['dir']}/{info['query']}",
        "groundtruth_neighbors_file": f"{info['dir']}/favor/groundtruth_s01.ibin",
        "filter_bitset_file": f"{info['dir']}/favor/filter_s01.bin",
        "distance": "euclidean",
        "dtype": info["dtype"],
    }
    if "subset_size" in info:
        dataset["subset_size"] = info["subset_size"]
    search = {
        "algo": "single_cta",
        "filter_mode": "default" if variant == "default" else "favor",
        "itopk": 512,
        "search_width": info["width"],
        # Bound the exact visited hash for adaptive ceilings while preserving the external
        # 10,000-query workload through CAGRA's existing internal chunking.
        "max_queries": 2048,
        "filtering_rate": 0.99,
        "favor_diagnostics_output": str(output),
        "favor_diagnostics_groundtruth": str(dataset_dir / "favor/groundtruth_s01.ibin"),
        "favor_diagnostics_dataset": dataset_key,
        "favor_diagnostics_variant": variant,
    }
    if variant != "default":
        search.update({
            "favor_penalty_mode": "cagra_retention_safe",
            "favor_penalty_lambda": 1.0,
            "favor_retention_fraction": 0.0,
        })
    if variant in ("current", "expanded"):
        search["favor_delta_d_file"] = str(dataset_dir / "cagra_g32_ig64.index.delta_d")
        search.update({"favor_delta_d_alpha": 10, "favor_delta_d_beta": 64, "favor_delta_d_bfs_depth": 2})
    elif variant == "zero":
        search["favor_delta_d"] = 0.0
    if variant == "expanded":
        search["max_iterations"] = info["expanded"]
        # The normal 0.5 fill target rounds a deep traversal to a 262,144-entry table per query
        # (10.5 GB for nq=10k). 0.89 keeps actual occupancy below about 88% and, through 3,584
        # iterations at width 1, selects the next smaller power-of-two table (5.2 GB).
        search["hashmap_max_fill_rate"] = 0.89
    if selected is not None:
        search["favor_diagnostics_selected_queries"] = str(selected)
        search["favor_diagnostics_max_trace_iterations"] = (
            info.get("expanded", 1024) if variant == "expanded" else 1024
        )
    return {
        "dataset": dataset,
        "search_basic_param": {"batch_size": 10000, "k": 10},
        "index": [{
            "name": "cagra-g32-ig64", "algo": "cuvs_cagra",
            "file": f"{info['dir']}/cagra_g32_ig64.index",
            "build_param": {"graph_build_algo": "NN_DESCENT", "graph_degree": 32, "intermediate_graph_degree": 64},
            "search_params": [search],
        }],
    }


def compress_traces(output: Path) -> None:
    if shutil.which("zstd") is None:
        return
    for name in ("iteration_trace.bin", "candidate_trace.bin"):
        path = output / name
        if path.exists() and path.stat().st_size:
            subprocess.run(["zstd", "-q", "-f", "-T0", "-3", "--rm", str(path)], check=True)


def run_one(
    dataset: str,
    variant: str,
    root: Path,
    selected: Path | None,
    dry_run: bool,
    resume: bool,
) -> None:
    suffix = "trace" if selected else "summary"
    run_dir = root / dataset / f"{variant}_{suffix}"
    config_path = root / "configs" / f"{dataset}_{variant}_{suffix}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(make_config(dataset, variant, run_dir, selected), indent=2) + "\n")
    if resume and (run_dir / "manifest.json").exists():
        subprocess.run([str(TRACE_TOOL), "validate", str(run_dir)], check=True)
        print(f"reuse {run_dir}", flush=True)
        return
    command = [
        str(BENCH), "--search", "--mode=latency", "--threads=1",
        f"--data_prefix={DATA}", f"--index_prefix={DATA}",
        "--benchmark_repetitions=1", "--benchmark_min_time=0.01s",
        "--benchmark_min_warmup_time=0", str(config_path),
    ]
    print(" ".join(command), flush=True)
    if dry_run:
        return
    log = run_dir / "bench.log"
    with log.open("w") as f:
        environment = os.environ.copy()
        environment["LD_PRELOAD"] = str(ROOT / "cpp/build/libcuvs.so")
        environment["CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION"] = (
            "0" if variant == "default" else "1"
        )
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=f, stderr=subprocess.STDOUT, check=False
        )
    if not (run_dir / "manifest.json").exists():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(
            f"benchmark exited {completed.returncode} and produced no diagnostic capture:\n{tail}"
        )
    if completed.returncode != 0:
        print(
            f"warning: benchmark exited {completed.returncode} after writing a complete capture; "
            f"see {log}",
            flush=True,
        )
    subprocess.run([str(TRACE_TOOL), "validate", str(run_dir)], check=True)
    compress_traces(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/favor/results_deep_traversal_diagnostics")
    parser.add_argument("--phase", choices=("summary", "deep", "all"), default="all")
    parser.add_argument("--cutoff-minutes", type=float, default=135.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse captures that validate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".gitignore").write_text(
        "**/candidate_trace.bin\n**/candidate_trace.bin.zst\n"
        "**/iteration_trace.bin\n**/iteration_trace.bin.zst\n**/bench.log\n"
    )
    start = time.monotonic()
    completed = False
    error = ""

    def check_cutoff() -> None:
        if not args.dry_run and time.monotonic() - start > args.cutoff_minutes * 60:
            raise TimeoutError("diagnostic cutoff reached; completed captures remain valid")

    try:
        if args.phase in ("summary", "all"):
            for dataset in DATASETS:
                for variant in ("default", "current", "zero"):
                    check_cutoff()
                    run_one(dataset, variant, args.output, None, args.dry_run, args.resume)
            for dataset in ("gist", "msturing1m", "msturing10m"):
                check_cutoff()
                run_one(dataset, "expanded", args.output, None, args.dry_run, args.resume)

        if args.phase in ("deep", "all") and not args.dry_run:
            selected_files: dict[str, Path] = {}
            for dataset in ("gist", "msturing1m", "msturing10m", "sift", "bigann10m"):
                summary = args.output / dataset / "current_summary"
                selected = args.output / dataset / "selected_queries.txt"
                subprocess.run([
                    str(TRACE_TOOL), "select-queries", str(summary), "--target", "0.9",
                    "--count", "16", "--output", str(selected),
                ], check=True)
                selected_files[dataset] = selected
            for dataset in ("gist", "msturing1m", "msturing10m"):
                for variant in ("current", "zero", "expanded"):
                    check_cutoff()
                    run_one(dataset, variant, args.output, selected_files[dataset], False, args.resume)
            for dataset in ("sift", "bigann10m"):
                check_cutoff()
                run_one(dataset, "current", args.output, selected_files[dataset], False, args.resume)
        completed = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.monotonic() - start
        (args.output / "run_manifest.json").write_text(json.dumps({
            "schema_version": 2,
            "elapsed_seconds": elapsed,
            "cutoff_minutes": args.cutoff_minutes,
            "completed": completed,
            "error": error,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
