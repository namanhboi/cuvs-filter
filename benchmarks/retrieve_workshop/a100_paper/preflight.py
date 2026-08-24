#!/usr/bin/env python3
"""Fail-fast machine, repository, storage, and input checks for an A100 paper run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=350.0)
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="testing only; never use for paper data",
    )
    args = parser.parse_args()

    failures: list[str] = []
    repo = args.repo.resolve()
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    status = command("git", "-C", str(repo), "status", "--porcelain")
    if status.returncode or status.stdout.strip():
        failures.append("repository is not clean")
    profile = json.loads(args.profile.read_text())

    python_packages: dict[str, str] = {}
    for package in ("numpy", "matplotlib"):
        probe = command(
            sys.executable,
            "-c",
            f"import {package}; print({package}.__version__)",
        )
        if probe.returncode:
            failures.append(
                f"Python package {package!r} is unavailable from {sys.executable}: "
                f"{probe.stderr.strip()}"
            )
        else:
            python_packages[package] = probe.stdout.strip()

    gpu = command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,mig.mode.current,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    )
    gpu_rows = [
        line.strip() for line in gpu.stdout.splitlines() if line.strip()
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu.returncode or len(gpu_rows) != 1:
        failures.append(
            "exactly one visible GPU is required; pin CUDA_VISIBLE_DEVICES"
        )
    elif not args.allow_non_a100:
        fields = [value.strip() for value in gpu_rows[0].split(",")]
        name, memory, capability, mig = (
            fields[0],
            float(fields[3]),
            fields[4],
            fields[5],
        )
        if "A100" not in name or memory < 79_000 or capability != "8.0":
            failures.append(
                f"expected a full A100 80GB (CC 8.0), got: {gpu_rows[0]}"
            )
        if mig.lower() not in ("disabled", "n/a", "[n/a]"):
            failures.append(f"MIG must be disabled, got {mig}")
    apps = command(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    if apps.returncode == 0 and apps.stdout.strip():
        failures.append(
            f"competing GPU compute processes are present: {apps.stdout.strip()}"
        )

    free_gib = shutil.disk_usage(data_root).free / 2**30
    if free_gib < args.minimum_free_gib:
        failures.append(
            f"only {free_gib:.1f} GiB free at {data_root}; require {args.minimum_free_gib:.1f}"
        )
    memory_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
    memory_gib = memory_kib / 2**20
    if memory_gib < 110:
        failures.append(
            f"only {memory_gib:.1f} GiB host RAM; require at least 110 GiB"
        )
    cpus = os.cpu_count() or 0
    if cpus < 12:
        failures.append(f"only {cpus} logical CPUs; require at least 12")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "repo": str(repo),
        "git_head": command(
            "git", "-C", str(repo), "rev-parse", "HEAD"
        ).stdout.strip(),
        "git_status": status.stdout,
        "profile": profile,
        "profile_path": str(args.profile.resolve()),
        "cuda_visible_devices": visible,
        "gpu_query": gpu_rows,
        "compute_apps": apps.stdout.strip().splitlines(),
        "disk_free_gib": free_gib,
        "minimum_free_gib": args.minimum_free_gib,
        "host_memory_gib": memory_gib,
        "logical_cpus": cpus,
        "python_executable": sys.executable,
        "python_packages": python_packages,
        "failures": failures,
    }
    destination = (
        args.run_root.resolve() / "provenance" / "a100_preflight.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("preflight failed:\n- " + "\n- ".join(failures))
    print(destination)


if __name__ == "__main__":
    main()
