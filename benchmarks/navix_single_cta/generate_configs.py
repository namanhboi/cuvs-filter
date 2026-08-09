#!/usr/bin/env python3
"""Generate compact, staged configs for the benchmark-only NaviX SINGLE_CTA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_QUERIES = 512
CORRECTNESS_BATCH_SIZE = 1_000
THROUGHPUT_BATCH_SIZE = 10_000
SWEEP_CELLS = ((64, 1), (128, 1), (128, 2), (256, 1), (256, 2), (512, 2))
# Policy isolation needs one representative B0 cell. Depth/L/W behavior is covered by the reduced
# two-method sweep, so repeating all five policies there only extends runtime without answering a
# new question.
CORRECTNESS_CELLS = ((64, 1, 0),)
POLICIES = ("one_hop", "directed_capped", "blind_capped", "adaptive_kuzu", "adaptive_paper")


def yfcc_dataset(*, throughput: bool = False) -> dict:
    root = (
        "yfcc-10M/workloads/throughput_10000"
        if throughput
        else "yfcc-10M/workloads/correctness_1000"
    )
    return {
        "name": f"yfcc-10M-navix-{'10000' if throughput else '1000'}",
        "base_file": "yfcc-10M/base.10M.u8bin",
        "query_file": f"{root}/query.u8bin",
        "groundtruth_neighbors_file": f"{root}/groundtruth.ibin",
        "distance": "euclidean",
        "dtype": "uint8",
        "filter": {
            "kind": "udf",
            "adapter": "spmat_contains_all",
            "base_metadata_file": "yfcc-10M/base.metadata.10M.spmat",
            "query_metadata_file": f"{root}/query.metadata.spmat",
        },
    }


def arxiv_dataset(predicate: str, *, throughput: bool = False) -> dict:
    workload = "throughput_10000" if throughput else "correctness_10000"
    root = f"arxiv-for-fanns-medium/{predicate}/{workload}"
    if predicate in {"em", "emis"}:
        filter_spec = {
            "kind": "udf",
            "adapter": "spmat_contains_all",
            "base_metadata_file": f"arxiv-for-fanns-medium/{predicate}/base_metadata.spmat",
            "query_metadata_file": f"{root}/query_metadata.spmat",
        }
    elif predicate == "r":
        filter_spec = {
            "kind": "udf",
            "adapter": "arxiv_range",
            "base_metadata_file": "arxiv-for-fanns-medium/r/base_metadata.rmeta",
            "query_metadata_file": f"{root}/query_metadata.rmeta",
        }
    else:
        raise ValueError(f"unknown ARXIV predicate: {predicate}")
    return {
        "name": f"arxiv-{predicate}-navix-{'10000' if throughput else '1000'}",
        "base_file": "arxiv-for-fanns-medium/base.fbin",
        "query_file": f"{root}/query.fbin",
        "groundtruth_neighbors_file": f"{root}/groundtruth.ibin",
        "distance": "euclidean",
        "dtype": "float",
        "filter": filter_spec,
    }


def default_accumulator(itopk: int, width: int, iterations: int, threads: int = 0) -> dict:
    result = {
        "algo": "single_cta",
        "filter_mode": "default",
        "max_queries": MAX_QUERIES,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": iterations,
        "favor_udf_passing_accumulator": True,
    }
    if threads:
        result["thread_block_size"] = threads
    return result


def navix(
    itopk: int,
    width: int,
    iterations: int,
    *,
    mode: str = "adaptive_kuzu",
    scheduler: str = "tiled",
    threads: int = 0,
) -> dict:
    result = {
        "algo": "single_cta",
        "filter_mode": "default",
        "max_queries": MAX_QUERIES,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": iterations,
        "favor_udf_passing_accumulator": False,
        "navix_mode": mode,
        "navix_scheduler": scheduler,
    }
    if threads:
        result["thread_block_size"] = threads
    return result


def navix_diagnostic(
    itopk: int,
    width: int,
    iterations: int,
    *,
    data_root: Path,
    output_directory: Path,
    variant: str,
) -> dict:
    """One untimed, benchmark-only capture of the current adaptive GPU implementation."""
    result = navix(itopk, width, iterations, mode="adaptive_kuzu")
    result.update(
        {
            "navix_diagnostics_output": str(output_directory.resolve()),
            "navix_diagnostics_groundtruth": str(
                (
                    data_root
                    / "yfcc-10M/workloads/correctness_1000/groundtruth.ibin"
                ).resolve()
            ),
            "navix_diagnostics_dataset": "yfcc-10M-correctness-1000",
            "navix_diagnostics_variant": variant,
        }
    )
    return result


def config(
    dataset: dict,
    index_file: str,
    searches: list[dict],
    batch_size: int = CORRECTNESS_BATCH_SIZE,
) -> dict:
    return {
        "dataset": dataset,
        "search_basic_param": {"batch_size": batch_size, "k": 10},
        "index": [
            {
                "name": "cagra-g32-ig64",
                "algo": "cuvs_cagra",
                "file": index_file,
                "build_param": {
                    "graph_build_algo": "NN_DESCENT",
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                },
                "search_params": searches,
            }
        ],
    }


def write(root: Path, name: str, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def correctness_searches() -> list[dict]:
    searches: list[dict] = []
    for itopk, width, iterations in CORRECTNESS_CELLS:
        searches.append(default_accumulator(itopk, width, iterations))
        searches.extend(navix(itopk, width, iterations, mode=policy) for policy in POLICIES)
    return searches


def sweep_searches(iterations: int) -> list[dict]:
    searches: list[dict] = []
    for itopk, width in SWEEP_CELLS:
        searches.append(default_accumulator(itopk, width, iterations))
        searches.append(navix(itopk, width, iterations))
    return searches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    yfcc_correctness = yfcc_dataset()
    emis_correctness = arxiv_dataset("emis")
    yfcc_throughput = yfcc_dataset(throughput=True)
    emis_throughput = arxiv_dataset("emis", throughput=True)
    write(
        args.output,
        "yfcc_smoke",
        config(
            yfcc_correctness,
            "yfcc-10M/cagra_g32_ig64.index",
            [default_accumulator(64, 1, 0), *(navix(64, 1, 0, mode=p) for p in POLICIES)],
            1,
        ),
    )
    write(
        args.output,
        "yfcc_correctness",
        config(
            yfcc_correctness,
            "yfcc-10M/cagra_g32_ig64.index",
            correctness_searches(),
        ),
    )
    write(
        args.output,
        "emis_correctness",
        config(
            emis_correctness,
            "arxiv-for-fanns-medium/cagra_g32_ig64.index",
            correctness_searches(),
        ),
    )

    scheduler_searches = [
        navix(128, 1, 0, mode=mode, scheduler=scheduler, threads=threads)
        for mode in ("directed_capped", "blind_capped")
        for scheduler in ("serial", "tiled")
        for threads in (64, 128)
    ]
    write(
        args.output,
        "emis_scheduler_gate",
        config(
            emis_correctness,
            "arxiv-for-fanns-medium/cagra_g32_ig64.index",
            scheduler_searches,
        ),
    )

    for name, dataset_spec, index_file in (
        ("yfcc", yfcc_throughput, "yfcc-10M/cagra_g32_ig64.index"),
        ("emis", emis_throughput, "arxiv-for-fanns-medium/cagra_g32_ig64.index"),
    ):
        for iterations in (0, 522, 1044):
            write(
                args.output,
                f"{name}_sweep_i{iterations}",
                config(
                    dataset_spec,
                    index_file,
                    sweep_searches(iterations),
                    THROUGHPUT_BATCH_SIZE,
                ),
            )

    for predicate in ("em", "r"):
        write(
            args.output,
            f"{predicate}_b0",
            config(
                arxiv_dataset(predicate, throughput=True),
                "arxiv-for-fanns-medium/cagra_g32_ig64.index",
                sweep_searches(0),
                THROUGHPUT_BATCH_SIZE,
            ),
        )

    # Three deliberately untimed captures isolate automatic B0, extra depth at fixed L/W, and a
    # substantially wider/deeper frontier.  Keeping one search parameter per file avoids any
    # ambiguity about which session produced an output directory.
    diagnostic_root = args.output.parent / "diagnostics"
    for name, itopk, width, iterations in (
        ("yfcc_navix_diag_b0_l64_w1", 64, 1, 0),
        ("yfcc_navix_diag_i1044_l64_w1", 64, 1, 1044),
        ("yfcc_navix_diag_i1044_l512_w2", 512, 2, 1044),
    ):
        write(
            args.output,
            name,
            config(
                yfcc_correctness,
                "yfcc-10M/cagra_g32_ig64.index",
                [
                    navix_diagnostic(
                        itopk,
                        width,
                        iterations,
                        data_root=args.data_root,
                        output_directory=diagnostic_root / name,
                        variant=name,
                    )
                ],
            ),
        )


if __name__ == "__main__":
    main()
