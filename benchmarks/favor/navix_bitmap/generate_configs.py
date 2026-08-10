#!/usr/bin/env python3
"""Generate matched default/accumulator/NaviX bitmap B0 sweep configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ITOPK = (64, 128, 256, 512)
WIDTHS = (1, 2)


def search(method: str, itopk: int, width: int) -> dict:
    row = {
        "algo": "single_cta",
        "filter_mode": "default",
        "max_queries": 512,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": 0,
    }
    if method == "default_cagra_accumulator":
        row["favor_udf_passing_accumulator"] = True
    if method in ("legacy_navix", "bitmap_seeded_navix"):
        row["navix_mode"] = "adaptive_kuzu"
        row["navix_scheduler"] = "tiled"
    if method == "bitmap_seeded_navix":
        row["navix_bitmap_seeds"] = True
    row["bitmap_method"] = method
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-manifest", type=Path, required=True)
    parser.add_argument("--base-file", required=True)
    parser.add_argument("--index-file", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--dtype", choices=("float", "half", "int8", "uint8"), required=True
    )
    parser.add_argument("--distance", default="euclidean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.bitmap_manifest.read_text())
    searches = [
        search(method, itopk, width)
        for method in (
            "default_cagra",
            "default_cagra_accumulator",
            "legacy_navix",
            "bitmap_seeded_navix",
        )
        for itopk in ITOPK
        for width in WIDTHS
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    generated = []
    for shard_number, shard in enumerate(manifest["shards"]):
        directory = Path(shard["directory"])
        count = int(shard["query_count"])
        config = {
            "dataset": {
                "name": f"{args.dataset_name}-q{shard['first_query']}-{count}",
                "base_file": args.base_file,
                "query_file": str(directory / "query.bin"),
                "groundtruth_neighbors_file": str(
                    directory / "groundtruth.ibin"
                ),
                "distance": args.distance,
                "dtype": args.dtype,
                "filter": {
                    "kind": "bitmap",
                    "file": str(directory / "filter.bitmap"),
                },
            },
            "search_basic_param": {"batch_size": count, "k": 10},
            "index": [
                {
                    "name": "cagra-g32-ig64",
                    "algo": "cuvs_cagra",
                    "file": args.index_file,
                    "build_param": {
                        "graph_build_algo": "NN_DESCENT",
                        "graph_degree": 32,
                        "intermediate_graph_degree": 64,
                    },
                    "search_params": searches,
                }
            ],
        }
        output = args.output / f"shard_{shard_number:02d}.json"
        output.write_text(json.dumps(config, indent=2) + "\n")
        generated.append(
            {
                "config": str(output),
                "first_query": shard["first_query"],
                "query_count": count,
            }
        )
    (args.output / "manifest.json").write_text(
        json.dumps({"configs": generated}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
