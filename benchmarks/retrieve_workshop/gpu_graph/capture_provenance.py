#!/usr/bin/env python3
"""Capture append-only, machine-readable provenance for a staged benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*argv: str) -> dict:
    completed = subprocess.run(
        argv, text=True, capture_output=True, check=False
    )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--bench-bin", type=Path, required=True)
    parser.add_argument("--libcuvs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    destination = args.result_root / "provenance" / "run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    current_gpu = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    if destination.is_file():
        payload = json.loads(destination.read_text())
        old_hashes = (
            payload["benchmark_binary"]["sha256"],
            payload["libcuvs"]["sha256"],
        )
        new_hashes = (sha256(args.bench_bin), sha256(args.libcuvs))
        if old_hashes != new_hashes:
            raise ValueError(
                "benchmark binary/library changed within one result root"
            )
        if payload["data_root"] != str(args.data_root.resolve()):
            raise ValueError("data root changed within one result root")
        if payload["host"] != platform.node() or payload["gpu"] != current_gpu:
            raise ValueError("host/GPU identity changed within one result root")
    else:
        git_head = command("git", "-C", str(args.repo), "rev-parse", "HEAD")
        git_status = command("git", "-C", str(args.repo), "status", "--short")
        payload = {
            "schema_version": 1,
            "experiment": "retrieve_workshop_gpu_graph",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "git": {"head": git_head, "status": git_status},
            "gpu": current_gpu,
            "cpu": command("lscpu"),
            "benchmark_binary": {
                "path": str(args.bench_bin.resolve()),
                "sha256": sha256(args.bench_bin),
            },
            "libcuvs": {
                "path": str(args.libcuvs.resolve()),
                "sha256": sha256(args.libcuvs),
            },
            "data_root": str(args.data_root.resolve()),
            "fixed_contract": {
                "gpu_algo": "SINGLE_CTA",
                "k": 10,
                "max_queries": 512,
                "reported_throughput_repetitions": 3,
                "correctness_repetitions": 1,
                "throughput_queries": 10000,
                "correctness_queries": 1000,
                "timing": "complete cuVS-bench search call; setup and index loading excluded",
            },
            "events": [],
        }
    payload["events"].append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": args.stage,
            "argv": sys.argv,
            "relevant_environment": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith("RETRIEVE_")
                or key in {"CUDA_VISIBLE_DEVICES", "LD_PRELOAD"}
            },
        }
    )
    destination.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
