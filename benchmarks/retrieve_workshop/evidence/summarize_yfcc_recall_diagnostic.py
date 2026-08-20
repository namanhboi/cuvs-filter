#!/usr/bin/env python3
"""Freeze the duplicate-safe YFCC B0 retention diagnostic used by the paper."""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

VARIANTS = ("default_b0", "default_accumulator_b0")
MATRIX_HEADER = struct.Struct("<II")


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
    for configured in result.values():
        configured["groundtruth_path"] = inputs[2]
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


def little_endian_array(typecode: str, payload: bytes) -> array.array:
    values = array.array(typecode)
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def independently_verify_outputs(
    result_ids_path: Path,
    ground_truth_path: Path,
    rows: list[dict[str, str]],
    num_queries: int,
    topk: int,
    dataset_size: int,
) -> dict[str, object]:
    result_payload = result_ids_path.read_bytes()
    if len(result_payload) != num_queries * topk * 8:
        raise ValueError(f"result-ID array shape failure: {result_ids_path}")
    result_ids = little_endian_array("q", result_payload)

    with ground_truth_path.open("rb") as stream:
        header = stream.read(MATRIX_HEADER.size)
        payload = stream.read()
    if len(header) != MATRIX_HEADER.size:
        raise ValueError(f"truncated ground-truth header: {ground_truth_path}")
    gt_rows, gt_columns = MATRIX_HEADER.unpack(header)
    if (
        gt_rows != num_queries
        or gt_columns != topk
        or len(payload) != gt_rows * gt_columns * 4
    ):
        raise ValueError(f"ground-truth matrix shape failure: {ground_truth_path}")
    ground_truth = little_endian_array("I", payload)

    matches = 0
    unique_outputs = 0
    duplicate_queries = 0
    output_gt_masks: list[int] = []
    for query in range(num_queries):
        gt_order = tuple(ground_truth[query * topk : (query + 1) * topk])
        gt_row = set(gt_order)
        if len(gt_row) != topk or any(node >= dataset_size for node in gt_row):
            raise ValueError(f"YFCC diagnostic GT row is not ten unique legal IDs: {query}")
        candidates = result_ids[query * topk : (query + 1) * topk]
        legal = [node for node in candidates if 0 <= node < dataset_size]
        distinct = set(legal)
        query_matches = len(distinct & gt_row)
        output_gt_masks.append(
            sum(1 << rank for rank, node in enumerate(gt_order) if node in distinct)
        )
        matches += query_matches
        unique_outputs += len(distinct)
        duplicate_queries += len(legal) != len(distinct)
        if (
            not math.isclose(
                float(rows[query]["recall"]),
                query_matches / topk,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            or int(rows[query]["output_count"]) != len(distinct)
        ):
            raise ValueError(
                f"query summary disagrees with result IDs/GT for query {query}"
            )
    return {
        "match_count": matches,
        "recall": matches / (num_queries * topk),
        "unique_output_count": unique_outputs,
        "mean_unique_output_count": unique_outputs / num_queries,
        "duplicate_query_count": duplicate_queries,
        "duplicate_query_rate": duplicate_queries / num_queries,
        "output_gt_masks": tuple(output_gt_masks),
    }


def summarize_variant(
    result_root: Path,
    variant: str,
    raw: dict[str, object],
    configured: dict[str, object],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    tuple[int, ...],
    dict[str, Path],
]:
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
    gt_seen_masks = tuple(int(row["gt_seen_mask"]) for row in rows)
    gt_seen_count = sum(mask.bit_count() for mask in gt_seen_masks)
    gt_seen = gt_seen_count / (len(rows) * 10)
    independent = independently_verify_outputs(
        result_ids_path,
        Path(configured["groundtruth_path"]),
        rows,
        int(manifest["num_queries"]),
        int(manifest["topk"]),
        int(manifest["dataset_size"]),
    )
    output_gt_masks = tuple(int(mask) for mask in independent["output_gt_masks"])
    if any(
        output_mask & ~seen_mask
        for output_mask, seen_mask in zip(output_gt_masks, gt_seen_masks)
    ):
        raise ValueError(f"output contains a GT neighbor not marked seen for {variant}")
    if variant == "default_accumulator_b0" and output_gt_masks != gt_seen_masks:
        raise ValueError("accumulator did not retain every seen GT neighbor per query")
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
        or label_value(label, "favor_diagnostics_dataset") != "yfcc10m"
        or Path(label_value(label, "favor_diagnostics_groundtruth")).resolve()
        != Path(configured["groundtruth_path"]).resolve()
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
    if not math.isclose(
        recall, independent["recall"], rel_tol=0.0, abs_tol=1e-7
    ):
        raise ValueError(f"independent output recall mismatch for {variant}")
    semantics = finite(raw, "OutputSetSemanticsVersion")
    duplicates = finite(raw, "DuplicateOutputQueries")
    if (
        not math.isclose(semantics, 1.0, rel_tol=0.0, abs_tol=1e-12)
        or not 0.0 <= duplicates <= 1.0
        or finite(raw, "FilterViolations") != 0.0
        or finite(raw, "InvalidSentinelErrors") != 0.0
        or not math.isclose(
            duplicates,
            independent["duplicate_query_rate"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or (variant == "default_accumulator_b0" and duplicates != 0.0)
    ):
        raise ValueError(f"raw correctness contract failure for {variant}")

    result = {
        "variant": variant,
        "queries": len(rows),
        "itopk": int(manifest["itopk"]),
        "search_width": int(manifest["search_width"]),
        "configured_max_iterations": int(manifest["configured_max_iterations"]),
        "distinct_output_match_count": int(independent["match_count"]),
        "distinct_output_recall": independent["recall"],
        "gt_seen_count": gt_seen_count,
        "gt_seen_rate": gt_seen,
        "selection_loss": (
            gt_seen_count - int(independent["match_count"])
        )
        / (len(rows) * 10),
        "duplicate_output_query_rate": duplicates,
        "mean_unique_output_count": independent["mean_unique_output_count"],
        "mean_distance_evaluations": sum(
            float(row["candidate_evaluations"]) for row in rows
        )
        / len(rows),
        "mean_passing_discoveries": sum(
            float(row["passing_candidates"]) for row in rows
        )
        / len(rows),
    }
    return (
        result,
        [source(manifest_path), source(summary_path), source(result_ids_path)],
        gt_seen_masks,
        {
            f"diagnostics/{variant}/manifest.json": manifest_path,
            f"diagnostics/{variant}/query_summary.csv": summary_path,
            f"diagnostics/{variant}/result_indices.i64bin": result_ids_path,
        },
    )


def archive_compact_sources(
    archive_dir: Path, paths: dict[str, Path], reuse_existing: bool
) -> list[dict[str, object]]:
    archive_dir = archive_dir.resolve()
    partial = archive_dir.with_name(f"{archive_dir.name}.partial")
    if partial.exists():
        raise FileExistsError(
            f"incomplete diagnostic archive already exists: {partial}"
        )
    if archive_dir.exists() and not reuse_existing:
        raise FileExistsError(f"refusing to replace diagnostic archive: {archive_dir}")
    if not archive_dir.exists() and reuse_existing:
        raise FileNotFoundError(f"diagnostic archive does not exist: {archive_dir}")
    if archive_dir.exists():
        actual = {
            str(path.relative_to(archive_dir))
            for path in archive_dir.rglob("*")
            if path.is_file()
        }
        if actual != set(paths):
            raise ValueError(
                f"diagnostic archive membership mismatch: {archive_dir}"
            )
    else:
        partial.mkdir(parents=True)
    archived: list[dict[str, object]] = []
    for relative, original in sorted(paths.items()):
        destination = archive_dir / relative if archive_dir.exists() else partial / relative
        if not archive_dir.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        original_hash = sha256(original)
        archived_hash = sha256(destination)
        if original_hash != archived_hash:
            raise ValueError(f"diagnostic archive copy mismatch: {original}")
        archived.append(
            {
                "path": str((archive_dir / relative).resolve()),
                "relative_path": relative,
                "bytes": destination.stat().st_size,
                "sha256": archived_hash,
                "original_path": str(original.resolve()),
            }
        )
    if not archive_dir.exists():
        partial.rename(archive_dir)
    return archived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bench-bin", type=Path, required=True)
    parser.add_argument("--libcuvs", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--reuse-existing-archive", action="store_true")
    parser.add_argument("--require-clean-repo", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    repo = args.repo.resolve()
    git_provenance = {
        "head": git(repo, "rev-parse", "HEAD"),
        "status": git(repo, "status", "--short"),
    }
    if args.require_clean_repo and git_provenance["status"]:
        raise ValueError("diagnostic generation requires a clean source repository")
    raw_path = result_root / "raw" / "root_cause_b0_diagnostic.json"
    config_path = result_root / "configs" / "root_cause_b0_diagnostic.json"
    configured, input_paths = validate_config(
        config_path, args.data_root.resolve()
    )
    records = raw_records(raw_path)
    variants: list[dict[str, object]] = []
    variant_sources: list[dict[str, object]] = []
    gt_seen_masks: dict[str, tuple[int, ...]] = {}
    compact_paths: dict[str, Path] = {
        "raw/root_cause_b0_diagnostic.json": raw_path,
        "configs/root_cause_b0_diagnostic.json": config_path,
        "inputs/groundtruth.ibin": input_paths[2],
    }
    for variant in VARIANTS:
        summary, current_sources, current_masks, current_compact_paths = summarize_variant(
            result_root, variant, records[variant], configured[variant]
        )
        variants.append(summary)
        variant_sources.extend(current_sources)
        gt_seen_masks[variant] = current_masks
        compact_paths.update(current_compact_paths)

    by_variant = {str(row["variant"]): row for row in variants}
    base = by_variant["default_b0"]
    retain = by_variant["default_accumulator_b0"]
    if gt_seen_masks["default_b0"] != gt_seen_masks["default_accumulator_b0"]:
        raise ValueError("accumulator changed a per-query traversal GT-seen mask")
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
    archived_sources = (
        archive_compact_sources(
            args.archive_dir, compact_paths, args.reuse_existing_archive
        )
        if args.archive_dir is not None
        else []
    )

    payload = {
        "schema_version": 1,
        "experiment": "yfcc_1000_b0_duplicate_safe_retention_diagnostic",
        "output_set_semantics": "distinct_valid_output_ids_v1",
        "git": git_provenance,
        "variants": variants,
        "headline": {
            "gt_seen_rate": float(base["gt_seen_rate"]),
            "base_returned_recall": float(base["distinct_output_recall"]),
            "retain_returned_recall": float(retain["distinct_output_recall"]),
            "base_selection_loss_percentage_points": 100
            * float(base["selection_loss"]),
        },
        "sources": sources,
        "archived_sources": archived_sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
