#!/usr/bin/env python3
"""Capture append-only provenance for the exact bitmap control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(*argv: str) -> dict[str, object]:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {
        "argv": list(argv),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
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
    args = parser.parse_args()

    destination = args.result_root / "provenance" / "run.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    current_gpu = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    if destination.exists():
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
            raise ValueError("benchmark binary/library changed within one result root")
        if payload["host"] != platform.node() or payload["gpu"] != current_gpu:
            raise ValueError("host/GPU identity changed within one result root")
    else:
        payload = {
            "schema_version": 2,
            "experiment": "retrieve_workshop_exact_bitmap",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "git": {
                "head": command("git", "-C", str(args.repo), "rev-parse", "HEAD"),
                "status": command("git", "-C", str(args.repo), "status", "--short"),
            },
            "gpu": current_gpu,
            "platform": platform.platform(),
            "benchmark_binary": {
                "path": str(args.bench_bin.resolve()),
                "sha256": sha256(args.bench_bin),
            },
            "libcuvs": {
                "path": str(args.libcuvs.resolve()),
                "sha256": sha256(args.libcuvs),
            },
            "fixed_contract": {
                "output_set_semantics": "distinct_valid_output_ids_v1",
            },
            "events": [],
        }
    payload["events"].append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": args.stage,
            "environment": {
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
