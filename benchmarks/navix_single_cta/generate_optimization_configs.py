#!/usr/bin/env python3
"""Generate strict-bitmap reference/optimized NaviX comparison configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_QUERIES = 512
K = 10
SWEEP_CELLS = ((64, 1), (128, 1), (128, 2), (256, 1), (256, 2), (512, 2))
METHODS = (
    "default_cagra",
    "default_cagra_accumulator",
    "navix_reference",
    "navix_optimized",
)
POLICIES = ("one_hop", "directed_capped", "blind_capped", "adaptive_kuzu", "adaptive_paper")


def search(
    method: str,
    itopk: int,
    width: int,
    iterations: int,
    *,
    policy: str = "adaptive_kuzu",
    scheduler: str = "tiled",
    threads: int = 0,
) -> dict:
    row: dict[str, object] = {
        "algo": "single_cta",
        "filter_mode": "default",
        "max_queries": MAX_QUERIES,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": iterations,
        "favor_udf_passing_accumulator": method == "default_cagra_accumulator",
        "bitmap_method": method,
    }
    if method.startswith("navix_"):
        row.update(
            {
                "navix_mode": policy,
                "navix_scheduler": scheduler,
                "navix_bitmap_seeds": True,
                "navix_kernel_variant": (
                    "optimized" if method == "navix_optimized" else "reference"
                ),
            }
        )
    if threads:
        row["thread_block_size"] = threads
    return row


def workload_paths(data_root: Path, workload: str, phase: str) -> tuple[Path, str, str, str]:
    suffix = "1000" if phase == "correctness" else "10000"
    if workload == "yfcc":
        manifest = data_root / "navix_bitmap" / "yfcc" / f"{phase}_{suffix}" / "manifest.json"
        return manifest, "yfcc-10M/base.10M.u8bin", "yfcc-10M/cagra_g32_ig64.index", "uint8"
    manifest = (
        data_root
        / "navix_bitmap"
        / "arxiv"
        / workload
        / f"{phase}_{suffix}"
        / "manifest.json"
    )
    return (
        manifest,
        "arxiv-for-fanns-medium/base.fbin",
        "arxiv-for-fanns-medium/cagra_g32_ig64.index",
        "float",
    )


def payload(
    *,
    workload: str,
    phase: str,
    shard: dict,
    base_file: str,
    index_file: str,
    dtype: str,
    searches: list[dict],
) -> dict:
    directory = Path(shard["directory"])
    count = int(shard["query_count"])
    return {
        "dataset": {
            "name": f"{workload}-bitmap-{phase}-q{shard['first_query']}-{count}",
            "base_file": base_file,
            "query_file": str(directory / "query.bin"),
            "groundtruth_neighbors_file": str(directory / "groundtruth.ibin"),
            "distance": "euclidean",
            "dtype": dtype,
            "filter": {"kind": "bitmap", "file": str(directory / "filter.bitmap")},
        },
        "search_basic_param": {"batch_size": count, "k": K},
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


def write_workload(
    output: Path,
    data_root: Path,
    workload: str,
    phase: str,
    name: str,
    searches: list[dict],
) -> None:
    manifest_path, base_file, index_file, dtype = workload_paths(data_root, workload, phase)
    manifest = json.loads(manifest_path.read_text())
    directory = output / name / workload
    directory.mkdir(parents=True, exist_ok=True)
    generated = []
    for number, shard in enumerate(manifest["shards"]):
        config_path = directory / f"shard_{number:02d}.json"
        config_path.write_text(
            json.dumps(
                payload(
                    workload=workload,
                    phase=phase,
                    shard=shard,
                    base_file=base_file,
                    index_file=index_file,
                    dtype=dtype,
                    searches=searches,
                ),
                indent=2,
            )
            + "\n"
        )
        generated.append(
            {
                "config": str(config_path),
                "first_query": int(shard["first_query"]),
                "query_count": int(shard["query_count"]),
            }
        )
    (directory / "manifest.json").write_text(json.dumps({"configs": generated}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    # Real-workload parity at one representative B0 cell, including every policy. Unit tests
    # cover packing/remapping edge cases; this catches data-scale regressions.
    correctness = [
        search(method, 64, 1, 0, policy=policy)
        for policy in POLICIES
        for method in ("navix_reference", "navix_optimized")
    ]
    for workload in ("yfcc", "em", "emis", "r"):
        write_workload(args.output, args.data_root, workload, "correctness", "correctness", correctness)

    # Matched end-to-end B0 frontier for every NaviX traversal policy.  Keep this on the
    # accepted reference kernel so the figure compares algorithms rather than the rejected
    # direct-lookup implementation variant.
    policy_sweep = [
        search("navix_reference", itopk, width, 0, policy=policy)
        for policy in POLICIES
        for itopk, width in SWEEP_CELLS
    ]
    for workload in ("yfcc", "em", "emis", "r"):
        write_workload(
            args.output,
            args.data_root,
            workload,
            "throughput",
            "navix_policy_b0",
            policy_sweep,
        )

    # Four-method B0 sweep. Deeper configs are generated separately and should only run when the
    # B0 output has not reached approximately 0.95 recall.
    for iterations in (0, 522, 1044):
        methods = METHODS if iterations == 0 else METHODS[:-1]
        searches = [
            search(method, itopk, width, iterations)
            for method in methods
            for itopk, width in SWEEP_CELLS
        ]
        for workload in ("yfcc", "em", "emis", "r"):
            write_workload(
                args.output,
                args.data_root,
                workload,
                "throughput",
                f"sweep_i{iterations}",
                searches,
            )

    # Scheduler/block-size gate on the hard workload, plus EM/R non-regression checks.
    gate = [
        search(method, 128, 1, 0, policy=policy, scheduler=scheduler, threads=threads)
        for method in ("navix_reference", "navix_optimized")
        for policy in ("directed_capped", "blind_capped", "adaptive_kuzu")
        for scheduler in ("serial", "tiled")
        for threads in (64, 128, 256)
    ]
    for workload in ("em", "emis", "r"):
        write_workload(args.output, args.data_root, workload, "correctness", "scheduler_gate", gate)

    # Small 10,000-query acceptance gate for implementation-only changes. Keep this separate from
    # the full frontier so a rejected optimization never forces another 96-point sweep.
    performance_gate = [
        search(method, itopk, 1, 0, threads=128)
        for itopk in (128, 256)
        for method in ("navix_reference", "navix_optimized")
    ]
    for workload in ("yfcc", "em", "emis", "r"):
        write_workload(
            args.output,
            args.data_root,
            workload,
            "throughput",
            "performance_gate",
            performance_gate,
        )


if __name__ == "__main__":
    main()
