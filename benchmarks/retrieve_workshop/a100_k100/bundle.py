#!/usr/bin/env python3
"""Create a compact, hash-addressed A100 k=100 paper bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_files(source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = (args.output or root / "paper_gpu_bundle_k100_matched").resolve()
    required = {
        "gpu_graph": root / "gpu_graph/analysis",
        "exact_bitmap": root / "exact_bitmap/analysis",
        "matched_recall": root / "matched_recall/analysis",
        "run_provenance": root / "provenance",
    }
    required_files = (
        required["gpu_graph"] / "summary_points.csv",
        required["gpu_graph"] / "provenance.json",
        required["exact_bitmap"] / "exact_results.json",
        required["matched_recall"] / "selected_points.csv",
        required["matched_recall"] / "measurements.csv",
        required["matched_recall"] / "provenance.json",
        required["matched_recall"] / "matched_recall_k100_table.csv",
        required["matched_recall"] / "fixed_recall_k100_results.tex",
        required["matched_recall"] / "gpu_matched_recall_k100.pdf",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing k=100 bundle inputs:\n" + "\n".join(missing)
        )

    matched_provenance = json.loads(
        (required["matched_recall"] / "provenance.json").read_text()
    )
    if (
        int(matched_provenance.get("k", -1)) != 100
        or int(matched_provenance.get("max_queries", -1)) != 2048
        or len(matched_provenance.get("selected_rows", [])) != 12
    ):
        raise ValueError(
            "matched-recall analysis did not satisfy the k=100 bundle contract"
        )

    latency_root = root / "per_query_latency"
    latency_present = any(
        path.exists()
        for path in (
            root / ".done/per_query_latency",
            latency_root / "analysis",
            latency_root / "provenance/run.json",
        )
    )
    latency_sources: list[tuple[Path, Path]] = []
    if latency_present:
        latency_summary_path = latency_root / "analysis/latency_summary.json"
        latency_provenance_path = latency_root / "provenance/run.json"
        latency_required = (
            latency_summary_path,
            latency_root / "analysis/latency_summary.csv",
            latency_root / "analysis/per_query_latency_cdf.pdf",
            latency_provenance_path,
        )
        missing_latency = [
            str(path) for path in latency_required if not path.is_file()
        ]
        if missing_latency:
            raise FileNotFoundError(
                "incomplete k=100 serialized-latency result:\n"
                + "\n".join(missing_latency)
            )
        latency_summary = json.loads(latency_summary_path.read_text())
        measurement = latency_summary.get("measurement_contract", {})
        latency_provenance = json.loads(latency_provenance_path.read_text())
        contract = latency_provenance.get("contract", {})
        if (
            latency_summary.get("status") != "PASS"
            or int(latency_summary.get("k", -1)) != 100
            or int(measurement.get("k", -1)) != 100
            or int(measurement.get("source_max_queries", -1)) != 2048
            or int(measurement.get("serialized_max_queries", -1)) != 1
            or int(measurement.get("queries_per_search_call", -1)) != 1
            or int(measurement.get("complete_passes", -1)) != 3
            or int(latency_summary.get("query_trace_rows", -1)) != 480_000
            or int(latency_provenance.get("schema_version", -1)) != 2
            or int(contract.get("k", -1)) != 100
            or int(contract.get("graph_source_max_queries", -1)) != 2048
            or int(contract.get("serialized_max_queries", -1)) != 1
            or int(contract.get("queries_per_call", -1)) != 1
        ):
            raise ValueError(
                "serialized k=100 latency did not satisfy its frozen contract"
            )
        latency_sources = [
            (latency_root / "analysis", Path("per_query_latency/analysis")),
            (
                latency_root / "provenance",
                Path("per_query_latency/provenance"),
            ),
        ]

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    copied: list[Path] = []
    for name, source in required.items():
        if source.is_dir():
            copied.extend(copy_files(source, temporary / name))
    for source, relative_destination in latency_sources:
        copied.extend(copy_files(source, temporary / relative_destination))
    profile = root / "state/dataset_profile.json"
    if profile.is_file():
        target = temporary / "dataset_profile.json"
        shutil.copy2(profile, target)
        copied.append(target)
    manifest = {
        "schema_version": 1,
        "experiment": "retrieve_workshop_a100_k100_matched_recall",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "k": 100,
        "max_queries": 2048,
        "serialized_latency_included": latency_present,
        "run_root": str(root),
        "files": [
            {
                "path": str(path.relative_to(temporary)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(copied)
        ],
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if output.exists():
        old_manifest = output / "manifest.json"
        if not old_manifest.is_file():
            raise ValueError(
                f"refusing to replace unrecognized bundle directory {output}"
            )
        old = json.loads(old_manifest.read_text())
        if old.get("experiment") != manifest["experiment"] or old.get(
            "run_root"
        ) != str(root):
            raise ValueError(
                f"refusing to replace incompatible bundle directory {output}"
            )
        shutil.rmtree(output)
    temporary.rename(output)
    print(output)


if __name__ == "__main__":
    main()
