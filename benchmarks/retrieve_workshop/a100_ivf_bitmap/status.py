#!/usr/bin/env python3
"""Read progress without taking the running benchmark's execution lock."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def calibration_eta(manifest, completed, timings):
    """Estimate only families with complete measured cases; never guess unseen costs."""
    specs = {spec["id"]: spec for spec in manifest["indexes"]}
    durations = defaultdict(float)
    epoch = manifest.get("runtime_revisions", [{}])[-1].get("captured_utc", "")
    for timing in timings:
        if timing["stage"] != "calibration" or timing["completed_utc"] < epoch:
            continue
        fields = timing["detail"].split()
        identity = (fields[0], fields[1], int(fields[2].split("=")[1]))
        durations[identity] += timing["elapsed_seconds"]
    samples = defaultdict(list)
    for identity in completed:
        spec = specs[identity[0]]
        if durations[identity]:
            samples[(spec["corpus"], spec["kind"])].append(durations[identity])
    known_seconds = 0
    unknown = set()
    pending = 0
    for spec in specs.values():
        family = (spec["corpus"], spec["kind"])
        for workload in (
            ("yfcc",) if spec["corpus"] == "yfcc" else ("em", "emis", "r")
        ):
            for k in (10, 100):
                if (spec["id"], workload, k) in completed:
                    continue
                pending += 1
                if samples[family]:
                    known_seconds += statistics.median(samples[family])
                else:
                    unknown.add("/".join(family))
    return dict(
        remaining_calibrations=pending,
        estimated_remaining_calibration_seconds=round(known_seconds)
        if not unknown
        else None,
        estimated_seconds_for_sampled_families=round(known_seconds),
        families_without_timing_samples=sorted(unknown),
        eta_scope="Calibration only; excludes future builds and final measurements",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root
    progress = json.loads((root / "state/progress.json").read_text())
    manifest = json.loads((root / "state/manifest.json").read_text())
    completed = set()
    for path in (root / "state/calibrated").glob("*/*.json"):
        workload, k = path.stem.rsplit("_k", 1)
        completed.add((path.parent.name, workload, int(k)))
    timings = []
    path = root / "logs/timing.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                timings.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A concurrent append can leave the final line incomplete.
    progress.update(calibration_eta(manifest, completed, timings))
    report_progress = root / "state/report_progress.json"
    if report_progress.exists():
        progress["report"] = json.loads(report_progress.read_text())
    pid_path = root / "state/orchestrator.pid"
    if pid_path.exists():
        pid = int(pid_path.read_text())
        command = Path(f"/proc/{pid}/cmdline")
        try:
            progress["process_alive"] = b"workflow.py" in command.read_bytes()
        except FileNotFoundError:
            progress["process_alive"] = False
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
