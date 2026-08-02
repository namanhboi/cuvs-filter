#!/usr/bin/env python3
"""Compare CAGRA delta-d with the original FAVOR HNSW construction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DATASETS = {
    "gist": {
        "directory": "gist-960-euclidean",
        "name": "GIST",
        "query": "query.fbin",
    },
    "msturing1m": {
        "directory": "msturing-1M",
        "name": "MSTuring",
        "query": "query.fbin",
    },
}


def build_config(dataset: dict[str, str], rows: int, graph_file: str) -> dict[str, object]:
    directory = dataset["directory"]
    return {
        "dataset": {
            "name": f"{dataset['name']}-{rows}-FAVOR-reference",
            "base_file": f"{directory}/base.fbin",
            "query_file": f"{directory}/{dataset['query']}",
            "subset_size": rows,
            "distance": "euclidean",
            "dtype": "float",
        },
        "search_basic_param": {"batch_size": 10, "k": 10},
        "index": [
            {
                "name": "cagra-g32-ig64",
                "algo": "cuvs_cagra",
                "file": graph_file,
                "build_param": {
                    "graph_build_algo": "NN_DESCENT",
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                },
                "search_params": [
                    {"algo": "single_cta", "itopk": 64, "search_width": 1}
                ],
            }
        ],
    }


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=repo_dir / "datasets")
    parser.add_argument(
        "--reuse-production-graphs",
        action="store_true",
        help="compare against data-dir/<dataset>/cagra_g32_ig64.index instead of building a subset",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=script_dir / "results" / "reference_validation"
    )
    parser.add_argument("--favor-dir", type=Path, default=Path("/home/ubuntu/FAVOR"))
    args = parser.parse_args()
    if args.rows <= 64 or args.threads <= 0:
        raise ValueError("--rows must exceed 64 and --threads must be positive")

    bench_bin = repo_dir / "cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
    libcuvs = repo_dir / "cpp/build/libcuvs.so"
    compare_bin = repo_dir / "examples/cpp/build/CAGRA_FAVOR_COMPARE"
    source = script_dir / "favor_reference_delta.cpp"
    favor_bin = args.result_dir / "bin/FAVOR_REFERENCE_DELTA"
    for path in (bench_bin, libcuvs, compare_bin, source, args.favor_dir / "include/favor.h"):
        if not path.exists():
            raise FileNotFoundError(path)

    favor_bin.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++20",
            "-O3",
            "-march=native",
            "-mtune=native",
            "-mavx2",
            "-mfma",
            "-ffast-math",
            "-funroll-loops",
            "-ftree-vectorize",
            "-pthread",
            "-fopenmp",
            f"-I{args.favor_dir / 'include'}",
            f"-I{args.favor_dir / 'include/hnswlib'}",
            str(source),
            "-o",
            str(favor_bin),
        ],
        cwd=repo_dir,
    )

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(libcuvs) + (
        f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
    )
    records = []
    for slug in args.datasets:
        dataset = DATASETS[slug]
        output_dir = args.result_dir / f"{slug}_{args.rows}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_production_graphs:
            graph_path = args.data_dir / dataset["directory"] / "cagra_g32_ig64.index"
            if not graph_path.exists():
                raise FileNotFoundError(graph_path)
        else:
            graph_relative = f"{slug}_{args.rows}/cagra_g32_ig64.index"
            graph_path = args.result_dir / graph_relative
            config_path = output_dir / "cagra_build.json"
            config_path.write_text(
                json.dumps(build_config(dataset, args.rows, graph_relative), indent=2) + "\n"
            )
            if not graph_path.exists():
                run(
                    [
                        str(bench_bin),
                        "--build",
                        f"--data_prefix={args.data_dir}",
                        f"--index_prefix={args.result_dir}",
                        str(config_path),
                    ],
                    cwd=repo_dir,
                    env=env,
                )

        base_path = args.data_dir / dataset["directory"] / "base.fbin"
        favor_json = output_dir / "favor.json"
        if not favor_json.exists():
            run(
                [
                    str(favor_bin),
                    str(base_path),
                    str(favor_json),
                    str(args.rows),
                    str(args.threads),
                ],
                cwd=repo_dir,
            )
        comparison_json = output_dir / "comparison.json"
        run(
            [
                str(compare_bin),
                str(base_path),
                str(graph_path),
                str(favor_json),
                str(comparison_json),
                "2",
                str(args.rows),
            ],
            cwd=repo_dir,
            env=env,
        )
        comparison = json.loads(comparison_json.read_text())
        comparison["dataset"] = slug
        records.append(comparison)

    summary_path = args.result_dir / f"summary_{args.rows}.json"
    summary_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {summary_path}")
    for record in records:
        print(
            f"{record['dataset']}: FAVOR={record['favor_delta_d']:.8g} "
            f"CAGRA={record['cagra_delta_d']:.8g} "
            f"relative_difference={record['relative_percent_difference']:.3f}%"
        )


if __name__ == "__main__":
    main()
