#!/usr/bin/env python3
"""Bind the resource-work evidence to the exact executable and instrumented sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


INSTRUMENTED_SOURCES = (
    "cpp/bench/ann/src/cuvs/cuvs_cagra_wrapper.h",
    "cpp/bench/ann/src/cuvs/favor_search_diagnostic_session.h",
    "cpp/src/neighbors/detail/cagra/jit_lto_kernels/favor_search_diagnostics.cuh",
    "cpp/src/neighbors/detail/cagra/jit_lto_kernels/navix_device.cuh",
    "cpp/src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_jit.cuh",
    "cpp/src/neighbors/detail/cagra/search_single_cta_kernel_launcher_jit.cuh",
)


def command(*argv: str) -> dict:
    completed = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bench-bin", type=Path, required=True)
    parser.add_argument("--libcuvs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    source_artifacts = {
        relative: artifact(repo / relative) for relative in INSTRUMENTED_SOURCES
    }
    payload = {
        "schema_version": 1,
        "experiment": "retrieve_gpu_resource_work_schema9",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": {
            "head": command("git", "-C", str(repo), "rev-parse", "HEAD"),
            "status": command("git", "-C", str(repo), "status", "--short"),
        },
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        "benchmark_binary": artifact(args.bench_bin),
        "libcuvs": artifact(args.libcuvs),
        "instrumented_sources": source_artifacts,
        "data_root": str(args.data_root.resolve()),
        "contract": {
            "diagnostic_schema_version": 9,
            "workloads": ["yfcc", "emis"],
            "methods": [
                "default_cagra",
                "default_cagra_accumulator",
                "navix_reference",
            ],
            "queries": 1000,
            "itopk": 64,
            "search_width": 1,
            "max_iterations": 0,
            "diagnostics_are_timed": False,
            "resources_from_production_kernel": True,
        },
    }
    destination = args.result_root.resolve() / "provenance" / "run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = json.loads(destination.read_text())
        for field in (
            "benchmark_binary",
            "libcuvs",
            "instrumented_sources",
            "gpu",
            "contract",
        ):
            if previous.get(field) != payload.get(field):
                raise ValueError(f"provenance changed within one result root: {field}")
        return
    destination.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
