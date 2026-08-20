#!/usr/bin/env python3
"""Freeze the duplicate-safe YFCC B0 retention diagnostic used by the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

VARIANTS = ("default_b0", "default_accumulator_b0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def variant_from_label(label: str) -> str:
    match = re.search(r'(?:^|#)favor_diagnostics_variant="([^"]+)"', label)
    return match.group(1) if match else ""


def label_value(label: str, key: str) -> str:
    match = re.search(rf'(?:^|#){re.escape(key)}="([^"]+)"', label)
    return match.group(1) if match else ""


def validate_config(
    path: Path, data_root: Path
) -> tuple[dict[str, dict], list[Path]]:
    payload = json.loads(path.read_text())
    dataset = payload.get("dataset", {})
    basic = payload.get("search_basic_param", {})
    indexes = payload.get("index", [])
    if (
        dataset.get("name") != "yfcc-10M-correctness_1000"
        or dataset.get("distance") != "euclidean"
        or dataset.get("dtype") != "uint8"
        or dataset.get("filter", {}).get("kind") != "udf"
        or dataset.get("filter", {}).get("adapter") != "spmat_contains_all"
        or basic != {"batch_size": 1000, "k": 10}
        or len(indexes) != 1
        or indexes[0].get("algo") != "cuvs_cagra"
    ):
        raise ValueError(f"frozen YFCC diagnostic config contract failure: {path}")
    searches = indexes[0].get("search_params", [])
    result: dict[str, dict] = {}
    for position, search in enumerate(searches):
        variant = str(search.get("favor_diagnostics_variant", ""))
        if variant not in VARIANTS:
            continue
        if variant in result:
            raise ValueError(f"duplicate config variant: {variant}")
        expected_accumulator = variant == "default_accumulator_b0"
        if (
            position not in (0, 1)
            or search.get("algo") != "single_cta"
            or search.get("filter_mode") != "default"
            or int(search.get("max_queries", -1)) != 512
            or int(search.get("itopk", -1)) != 512
            or int(search.get("search_width", -1)) != 2
            or int(search.get("max_iterations", -1)) != 0
            or bool(search.get("favor_udf_passing_accumulator"))
            != expected_accumulator
        ):
            raise ValueError(f"search config contract failure: {variant}")
        result[variant] = {"position": position, "search": search}
    if set(result) != set(VARIANTS):
        raise ValueError(f"wrong config variant set: {sorted(result)}")
    def data_path(value: object) -> Path:
        candidate = Path(str(value))
        return (candidate if candidate.is_absolute() else data_root / candidate).resolve()

    inputs = [
        data_path(dataset["base_file"]),
        data_path(dataset["query_file"]),
        data_path(dataset["groundtruth_neighbors_file"]),
        data_path(dataset["filter"]["base_metadata_file"]),
        data_path(dataset["filter"]["query_metadata_file"]),
        data_path(indexes[0]["file"]),
    ]
    missing = [str(candidate) for candidate in inputs if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(f"missing diagnostic inputs: {missing}")
    return result, inputs


def raw_records(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text())
    result: dict[str, dict[str, object]] = {}
    for row in payload.get("benchmarks", []):
        if row.get("run_type") != "iteration":
            continue
        variant = variant_from_label(str(row.get("label", "")))
        if variant not in VARIANTS:
            continue
        if variant in result:
            raise ValueError(f"duplicate raw iteration for {variant}")
        if row.get("error_occurred") or row.get("skipped"):
            raise ValueError(f"failed raw iteration for {variant}")
        result[variant] = row
    if set(result) != set(VARIANTS):
        raise ValueError(f"wrong raw variant set: {sorted(result)}")
    return result


def finite(row: dict[str, object], key: str) -> float:
    value = float(row.get(key, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"missing/nonfinite raw counter {key}")
    return value


def summarize_variant(
    result_root: Path,
    variant: str,
    raw: dict[str, object],
    configured: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    directory = result_root / "diagnostics" / "root_cause" / variant
    manifest_path = directory / "manifest.json"
    summary_path = directory / "query_summary.csv"
    result_ids_path = directory / "result_indices.i64bin"
    manifest = json.loads(manifest_path.read_text())
    search = configured["search"]
    position = int(configured["position"])
    if (
        int(manifest.get("schema_version", -1)) != 7
        or manifest.get("dataset") != "yfcc10m"
        or manifest.get("variant") != variant
        or int(manifest.get("num_queries", -1)) != 1000
        or int(manifest.get("topk", -1)) != 10
        or int(manifest.get("dataset_size", -1)) != 10_000_000
        or int(manifest.get("graph_degree", -1)) != 32
        or manifest.get("output_set_semantics")
        != "distinct_valid_output_ids_v1"
        or int(manifest.get("itopk", -1)) != int(search["itopk"])
        or int(manifest.get("search_width", -1))
        != int(search["search_width"])
        or int(manifest.get("configured_max_iterations", -1))
        != int(search["max_iterations"])
    ):
        raise ValueError(f"diagnostic manifest contract failure: {manifest_path}")
    rows = list(csv.DictReader(summary_path.open()))
    if len(rows) != 1000 or [int(row["query_id"]) for row in rows] != list(
        range(1000)
    ):
        raise ValueError(f"diagnostic query coverage failure: {summary_path}")

    recall = sum(float(row["recall"]) for row in rows) / len(rows)
    gt_seen = sum(int(row["gt_seen_mask"]).bit_count() / 10 for row in rows) / len(
        rows
    )
    raw_recall = finite(raw, "ValidGTRecall")
    label = str(raw.get("label", ""))
    name_match = re.search(r"/(\d+)/process_time/", str(raw.get("name", "")))
    if (
        int(raw.get("repetition_index", -1)) != 0
        or int(raw.get("n_queries", -1)) != int(manifest["num_queries"])
        or int(raw.get("k", -1)) != int(manifest["topk"])
        or name_match is None
        or int(name_match.group(1)) != position
        or label_value(label, "algo") != "single_cta"
        or label_value(label, "filter_mode") != "default"
        or label_value(label, "favor_diagnostics_variant") != variant
        or int(finite(raw, "itopk")) != int(search["itopk"])
        or int(finite(raw, "search_width")) != int(search["search_width"])
        or int(finite(raw, "max_iterations"))
        != int(search["max_iterations"])
        or int(finite(raw, "max_queries")) != int(search["max_queries"])
        or bool(int(finite(raw, "favor_udf_passing_accumulator")))
        != bool(search["favor_udf_passing_accumulator"])
        or Path(str(search["favor_diagnostics_output"])).resolve()
        != directory.resolve()
    ):
        raise ValueError(f"raw/config/manifest identity mismatch for {variant}")
    if not math.isclose(recall, raw_recall, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            f"raw/diagnostic recall mismatch for {variant}: {raw_recall} != {recall}"
        )
    semantics = finite(raw, "OutputSetSemanticsVersion")
    duplicates = finite(raw, "DuplicateOutputQueries")
    if (
        not math.isclose(semantics, 1.0, rel_tol=0.0, abs_tol=1e-12)
        or not 0.0 <= duplicates <= 1.0
        or finite(raw, "FilterViolations") != 0.0
        or finite(raw, "InvalidSentinelErrors") != 0.0
        or (variant == "default_accumulator_b0" and duplicates != 0.0)
    ):
        raise ValueError(f"raw correctness contract failure for {variant}")

    result = {
        "variant": variant,
        "queries": len(rows),
        "itopk": int(manifest["itopk"]),
        "search_width": int(manifest["search_width"]),
        "configured_max_iterations": int(manifest["configured_max_iterations"]),
        "distinct_output_recall": recall,
        "gt_seen_rate": gt_seen,
        "selection_loss": gt_seen - recall,
        "duplicate_output_query_rate": duplicates,
        "mean_unique_output_count": sum(float(row["output_count"]) for row in rows)
        / len(rows),
        "mean_distance_evaluations": sum(
            float(row["candidate_evaluations"]) for row in rows
        )
        / len(rows),
        "mean_passing_discoveries": sum(
            float(row["passing_candidates"]) for row in rows
        )
        / len(rows),
    }
    return result, [source(manifest_path), source(summary_path), source(result_ids_path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bench-bin", type=Path, required=True)
    parser.add_argument("--libcuvs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    repo = args.repo.resolve()
    raw_path = result_root / "raw" / "root_cause_b0_diagnostic.json"
    config_path = result_root / "configs" / "root_cause_b0_diagnostic.json"
    configured, input_paths = validate_config(
        config_path, args.data_root.resolve()
    )
    records = raw_records(raw_path)
    variants: list[dict[str, object]] = []
    variant_sources: list[dict[str, object]] = []
    for variant in VARIANTS:
        summary, current_sources = summarize_variant(
            result_root, variant, records[variant], configured[variant]
        )
        variants.append(summary)
        variant_sources.extend(current_sources)

    by_variant = {str(row["variant"]): row for row in variants}
    base = by_variant["default_b0"]
    retain = by_variant["default_accumulator_b0"]
    if not math.isclose(
        float(base["gt_seen_rate"]),
        float(retain["gt_seen_rate"]),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError("accumulator changed traversal GT-seen rate")
    if not math.isclose(
        float(retain["distinct_output_recall"]),
        float(retain["gt_seen_rate"]),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError("accumulator did not retain every seen GT neighbor")

    # Perform the expensive full-input hashing only after every structural and semantic gate has
    # passed.  Legacy traces should fail closed without reading multi-gigabyte dataset files.
    sources = [
        source(raw_path),
        source(config_path),
        source(args.bench_bin),
        source(args.libcuvs),
    ]
    sources.extend(source(path) for path in input_paths)
    sources.extend(variant_sources)

    payload = {
        "schema_version": 1,
        "experiment": "yfcc_1000_b0_duplicate_safe_retention_diagnostic",
        "output_set_semantics": "distinct_valid_output_ids_v1",
        "git": {
            "head": git(repo, "rev-parse", "HEAD"),
            "status": git(repo, "status", "--short"),
        },
        "variants": variants,
        "headline": {
            "gt_seen_rate": float(base["gt_seen_rate"]),
            "base_returned_recall": float(base["distinct_output_recall"]),
            "retain_returned_recall": float(retain["distinct_output_recall"]),
            "base_selection_loss_percentage_points": 100
            * float(base["selection_loss"]),
        },
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
