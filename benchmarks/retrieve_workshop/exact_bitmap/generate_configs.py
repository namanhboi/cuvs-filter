#!/usr/bin/env python3
"""Generate one cuVS brute-force bitmap benchmark config per resident bitmap shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_memory_preflight import validate_estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-manifest", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--phase",
        choices=("correctness", "throughput", "smoke"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-marker", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k != 10:
        parser.error("the frozen exact-control contract requires --k=10")

    manifest = json.loads(args.exact_manifest.read_text())
    if (
        manifest.get("method") != "cuvs_brute_force_bitmap"
        or manifest.get("timed_invalid_sentinel_normalization") is not True
        or "invalid-sentinel normalization"
        not in str(manifest.get("timing_contract", ""))
    ):
        raise ValueError("unsupported or stale exact-workload manifest")
    if manifest.get("search_dtype") != "float32":
        raise ValueError(
            "cuVS exact-control configs require float32 search vectors"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    args.index_marker.parent.mkdir(parents=True, exist_ok=True)
    # The cuVS benchmark brute-force wrapper intentionally serializes an empty marker: its index is
    # reconstructed from the configured resident dataset before each search process.
    args.index_marker.touch(exist_ok=True)
    generated: list[dict[str, object]] = []
    for shard in manifest["shards"]:
        shard_number = int(shard["shard_number"])
        query_count = int(shard["query_count"])
        memory_estimate = shard.get("gpu_memory_estimate")
        if not isinstance(memory_estimate, dict):
            raise TypeError(
                f"exact shard {shard_number} lacks a GPU-memory estimate"
            )
        required_free_bytes = validate_estimate(memory_estimate)
        config = {
            "dataset": {
                "name": (
                    f"retrieve-exact-{args.workload}-{args.phase}-"
                    f"q{shard['first_query']}-{query_count}"
                ),
                "base_file": manifest["base_file"],
                "query_file": shard["query_file"],
                "groundtruth_neighbors_file": shard["groundtruth_file"],
                "distance": "euclidean",
                "dtype": "float",
                "filter": {"kind": "bitmap", "file": shard["bitmap_file"]},
            },
            "search_basic_param": {"batch_size": query_count, "k": args.k},
            "index": [
                {
                    "name": "cuvs-exact-bitmap",
                    "algo": "cuvs_brute_force",
                    "file": str(args.index_marker.resolve()),
                    "build_param": {},
                    "search_params": [
                        {
                            "exact_control": "bitmap_count_csr_search",
                            "native_l2_cutoff_validation": True,
                            "resident_bitmap": True,
                        }
                    ],
                }
            ],
        }
        output = args.output / f"shard_{shard_number:02d}.json"
        output.write_text(json.dumps(config, indent=2) + "\n")
        generated.append(
            {
                "config": str(output.resolve()),
                "shard_number": shard_number,
                "first_query": int(shard["first_query"]),
                "query_count": query_count,
                "required_free_device_bytes": required_free_bytes,
            }
        )

    output_manifest = {
        "schema_version": 1,
        "workload": args.workload,
        "phase": args.phase,
        "expected_repetitions": 3 if args.phase == "throughput" else 1,
        "expected_shards": len(generated),
        "expected_queries": sum(int(row["query_count"]) for row in generated),
        "exact_manifest": str(args.exact_manifest.resolve()),
        "gpu_memory_preflight": (
            "mandatory before every shard; required bytes include the prepared estimate's "
            "max(2 GiB,20%) safety margin"
        ),
        "configs": generated,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
