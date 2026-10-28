#!/usr/bin/env python3
"""Validate a completed batch-1 run and export compact, immutable review artifacts."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import primary_ivf as pipeline


def collect(root, output):
    root, output = root.resolve(), output.resolve()
    manifest = pipeline.read(root / "manifest.json")
    progress = pipeline.read(root / "progress.json")
    if progress["status"] != "complete":
        raise ValueError("Refusing to publish an incomplete experiment")
    summary = pipeline.read(root / "analysis/latency_summary.json")
    if (
        progress["status"] != "complete"
        or not summary["complete"]
        or len(summary["rows"]) != 32
    ):
        raise ValueError("Refusing to publish an incomplete experiment")
    if summary["recall_drift_flags"] or any(
        not r["recall_preserved"] for r in summary["rows"]
    ):
        raise ValueError("Recall drift needs investigation before publishing")
    if any(r["queries"] != 10000 or r["passes"] != 3 for r in summary["rows"]):
        raise ValueError("Wrong full-query contract")
    pipeline.check_inputs(manifest)
    signature = pipeline.sha(root / "manifest.json")
    for job in manifest["jobs"]:
        if not pipeline.completed(job, signature):
            raise ValueError(f"Missing verified job: {job['id']}")
    if output.exists():
        checks = pipeline.read(output / "SHA256SUMS.json")
        if pipeline.sha(output / "manifest.json") != signature or any(
            pipeline.sha(output / p) != h for p, h in checks.items()
        ):
            raise ValueError(
                "Existing snapshot differs; refusing to replace it"
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".batch1-export-", dir=output.parent
    ) as d:
        temp = Path(d) / output.name
        temp.mkdir()
        paths = [
            root / name
            for name in (
                "manifest.json",
                "configurations.csv",
                "configuration_audit.json",
                "build_provenance.json",
                "progress.json",
            )
        ]
        paths += list((root / "analysis").glob("*"))
        paths = [p for p in paths if p.name != "per_query_samples.csv"]
        paths += list((root / "configs").rglob("*.json"))
        paths += list((root / "raw").rglob("*.json"))
        checks = {}
        for source in sorted(set(paths)):
            destination = temp / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            checks[str(destination.relative_to(temp))] = pipeline.sha(
                destination
            )
        outputs = {
            p: h
            for job in manifest["jobs"]
            for p, h in pipeline.read(job["raw"] + ".complete.json")[
                "outputs"
            ].items()
        }
        pipeline.save(temp / "external-output-checksums.json", outputs)
        (temp / "README.md").write_text(
            "# Completed primary batch-1 latency\n\n"
            "See analysis/REPORT.md and analysis/latency_summary.csv. All four methods,\n"
            "four primary workloads, both k values, all 10,000 queries, three passes.\n"
            "CAGRA uses SINGLE_CTA; every method uses dense bitmaps and frozen QPS-recall\n"
            "settings. Both host completion and CUDA event timings are measured.\n\n"
            "Unmatched source-maximum and serialized-out-of-window rows are explicitly\n"
            "labeled and excluded from matched-recall comparisons. Base's historical\n"
            "duplicate-slot diagnostic remains visible under distinct-ID recall.\n\n"
            f"Full per-query samples, traces, neighbor outputs, logs, dataset/index\n"
            f"inputs, and runtime binaries remain at {root}. Their output checksums\n"
            "are included; these large files are not duplicated in Git.\n"
        )
        for name in ("external-output-checksums.json", "README.md"):
            checks[name] = pipeline.sha(temp / name)
        pipeline.save(temp / "SHA256SUMS.json", checks)
        temp.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(collect(args.root, args.output))


if __name__ == "__main__":
    main()
