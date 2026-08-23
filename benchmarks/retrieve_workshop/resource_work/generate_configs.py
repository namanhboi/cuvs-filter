#!/usr/bin/env python3
"""Generate the fixed L=64/W=1/B0 resource-work diagnostic configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "gpu_graph"))

from generate_configs import (  # noqa: E402
    PRIMARY_METHODS,
    config_payload,
    dataset_paths,
    search_point,
)

WORKLOADS = ("yfcc", "em", "emis", "r")
ITOPK = 64
SEARCH_WIDTH = 1
MAX_ITERATIONS = 0
DIAGNOSTIC_SCHEMA = 9


def build(output: Path, data_root: Path, diagnostic_root: Path) -> dict:
    configs: dict[str, dict[str, str]] = {}
    for workload in WORKLOADS:
        paths = dataset_paths(data_root, workload, "correctness", 64)
        source_manifest = json.loads(paths.manifest.read_text())
        shards = source_manifest.get("shards", [])
        if len(shards) != 1 or int(shards[0].get("query_count", -1)) != 1_000:
            raise ValueError(f"{paths.manifest} must contain one 1,000-query shard")
        shard = shards[0]
        shard_directory = Path(shard["directory"])
        ground_truth = shard_directory / "groundtruth.ibin"
        if not ground_truth.is_file():
            raise FileNotFoundError(ground_truth)

        resource_searches = [
            search_point(method, ITOPK, SEARCH_WIDTH, MAX_ITERATIONS)
            for method in PRIMARY_METHODS
        ]
        diagnostic_searches: list[dict] = []
        for row in resource_searches:
            method = str(row["bitmap_method"])
            diagnostic = dict(row)
            diagnostic.update(
                {
                    # The diagnostic session writes one capture per internal query chunk. Keep the
                    # fixed 1,000-query sample in one untimed chunk; max_queries is only a host
                    # scheduling cap and does not change per-query traversal or kernel resources.
                    "max_queries": 1_024,
                    "favor_diagnostics_output": str(
                        (diagnostic_root / workload / method).resolve()
                    ),
                    "favor_diagnostics_groundtruth": str(ground_truth.resolve()),
                    "favor_diagnostics_dataset": workload,
                    "favor_diagnostics_variant": method,
                }
            )
            diagnostic_searches.append(diagnostic)

        workload_configs: dict[str, str] = {}
        for mode, searches in (
            ("resources", resource_searches),
            ("diagnostics", diagnostic_searches),
        ):
            path = output / mode / f"{workload}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = config_payload(
                workload=workload,
                phase="correctness",
                shard=shard,
                paths=paths,
                searches=searches,
            )
            path.write_text(json.dumps(payload, indent=2) + "\n")
            workload_configs[mode] = str(path.resolve())
        configs[workload] = workload_configs

    manifest = {
        "schema_version": 1,
        "experiment": "retrieve_gpu_resource_work",
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA,
        "workloads": list(WORKLOADS),
        "methods": list(PRIMARY_METHODS),
        "queries": 1_000,
        "itopk": ITOPK,
        "search_width": SEARCH_WIDTH,
        "max_iterations": MAX_ITERATIONS,
        "configs": configs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.data_root.resolve(), args.diagnostic_root.resolve())


if __name__ == "__main__":
    main()
