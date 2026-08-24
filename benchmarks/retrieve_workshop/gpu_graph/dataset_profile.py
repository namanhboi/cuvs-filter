#!/usr/bin/env python3
"""Dataset geometry and artifact routing for RETRIEVE GPU experiments.

The checked-in experiments keep their historical medium-ArXiv defaults.  A follow-up run can set
``RETRIEVE_DATASET_PROFILE`` to an explicit JSON file; every generator then resolves dataset size,
graph degree, and bitmap/index paths from that same immutable profile.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

WORKLOADS = ("yfcc", "em", "emis", "r")

DEFAULT_PROFILE: dict[str, Any] = {
    "schema_version": 1,
    "name": "legacy-yfcc10m-arxiv-medium",
    "max_queries": 512,
    "matched_widths": [1, 2, 4],
    "datasets": {
        "yfcc": {
            "bitmap_directory": "navix_bitmap/yfcc",
            "base_file": "yfcc-10M/base.10M.u8bin",
            "index_file": "yfcc-10M/cagra_g64_ig128.index",
            "dtype": "uint8",
            "dataset_size": 10_000_000,
            "dimension": 192,
            "graph_degree": 64,
            "intermediate_graph_degree": 128,
        },
        **{
            workload: {
                "bitmap_directory": f"navix_bitmap/arxiv/{workload}",
                "base_file": "arxiv-for-fanns-medium/base.fbin",
                "index_file": "arxiv-for-fanns-medium/cagra_g32_ig64.index",
                "dtype": "float",
                "dataset_size": 100_000,
                "dimension": 4096,
                "graph_degree": 32,
                "intermediate_graph_degree": 64,
            }
            for workload in ("em", "emis", "r")
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(profile: dict[str, Any]) -> dict[str, Any]:
    if int(profile.get("schema_version", -1)) != 1:
        raise ValueError("dataset profile must use schema_version=1")
    if not str(profile.get("name", "")).strip():
        raise ValueError("dataset profile requires a name")
    datasets = profile.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(WORKLOADS):
        raise ValueError(f"dataset profile must define exactly {WORKLOADS}")
    widths = tuple(int(value) for value in profile.get("matched_widths", []))
    if not widths or any(value not in (1, 2, 4) for value in widths):
        raise ValueError("matched_widths must be a nonempty subset of 1, 2, 4")
    if len(set(widths)) != len(widths):
        raise ValueError("matched_widths contains duplicates")
    max_queries = int(profile.get("max_queries", 512))
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    for workload, row in datasets.items():
        required = {
            "bitmap_directory",
            "base_file",
            "index_file",
            "dtype",
            "dataset_size",
            "dimension",
            "graph_degree",
            "intermediate_graph_degree",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{workload} profile is missing {sorted(missing)}"
            )
        if row["dtype"] not in ("float", "uint8", "int8", "half"):
            raise ValueError(
                f"unsupported dtype for {workload}: {row['dtype']}"
            )
        for key in (
            "dataset_size",
            "dimension",
            "graph_degree",
            "intermediate_graph_degree",
        ):
            if int(row[key]) <= 0:
                raise ValueError(f"{workload}.{key} must be positive")
        if int(row["intermediate_graph_degree"]) < int(row["graph_degree"]):
            raise ValueError(
                f"{workload} intermediate degree is smaller than graph degree"
            )
    profile["matched_widths"] = list(widths)
    profile["max_queries"] = max_queries
    return profile


def load_profile(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        configured = os.environ.get("RETRIEVE_DATASET_PROFILE", "").strip()
        path = Path(configured).resolve() if configured else None
    if path is None:
        profile = json.loads(json.dumps(DEFAULT_PROFILE))
        profile["source"] = "built-in"
        profile["sha256"] = None
        return _validate(profile)
    if not path.is_file():
        raise FileNotFoundError(path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["source"] = str(path.resolve())
    profile["sha256"] = sha256(path)
    return _validate(profile)


def workload_spec(
    workload: str, profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}")
    active = profile if profile is not None else load_profile()
    return dict(active["datasets"][workload])


def profile_record(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    active = profile if profile is not None else load_profile()
    return {
        "name": active["name"],
        "source": active.get("source", "built-in"),
        "sha256": active.get("sha256"),
        "max_queries": int(active["max_queries"]),
    }
