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

from dataset_profile import load_profile


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
    expected_max_queries = int(
        os.environ.get(
            "RETRIEVE_PROVENANCE_MAX_QUERIES",
            str(load_profile()["max_queries"]),
        )
    )
    if expected_max_queries <= 0:
        raise ValueError("provenance max_queries must be positive")

    destination = args.result_root / "provenance" / "run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    current_gpu = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    current_git = {
        "head": command("git", "-C", str(args.repo), "rev-parse", "HEAD"),
        "status": command("git", "-C", str(args.repo), "status", "--short"),
    }
    if destination.is_file():
        payload = json.loads(destination.read_text())
        if (
            payload.get("schema_version") != 2
            or payload.get("fixed_contract", {}).get("output_set_semantics")
            != "distinct_valid_output_ids_v1"
        ):
            raise ValueError("result root uses legacy output/recall semantics")
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
        if int(payload["fixed_contract"]["max_queries"]) != expected_max_queries:
            raise ValueError("max_queries changed within one result root")
    else:
        payload = {
            "schema_version": 2,
            "experiment": "retrieve_workshop_gpu_graph",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "git": current_git,
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
                "max_queries": expected_max_queries,
                "reported_throughput_repetitions": int(
                    os.environ.get("RETRIEVE_PROVENANCE_REPETITIONS", "3")
                ),
                "correctness_repetitions": 1,
                "throughput_queries": 10000,
                "correctness_queries": 1000,
                "timing": os.environ.get(
                    "RETRIEVE_PROVENANCE_TIMING",
                    "complete cuVS-bench search call; setup and index loading excluded",
                ),
                "output_set_semantics": "distinct_valid_output_ids_v1",
            },
            "events": [],
        }
    payload["events"].append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": args.stage,
            "argv": sys.argv,
            "git": current_git,
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
