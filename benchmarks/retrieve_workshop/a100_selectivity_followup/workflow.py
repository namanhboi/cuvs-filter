#!/usr/bin/env python3
"""Reproducible k=10 ArXiv selectivity and correlation follow-up.

The existing CAGRA benchmark and graph analyzer remain authoritative.  This module only
constructs new query/bitmap/GT cohorts, drives their configs, and combines validated output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSHOP = HERE.parent
REPO = WORKSHOP.parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "favor" / "navix_bitmap"))
sys.path.insert(0, str(WORKSHOP / "exact_bitmap"))
sys.path.insert(0, str(WORKSHOP / "gpu_graph"))

import analyze_gpu_graph as graph_analysis
import matplotlib.pyplot as plt
from generate_configs import (
    B0_CELLS,
    DEEP_CELLS,
    DEEP_ITERATIONS,
    DatasetPaths,
    config_payload,
    point_identity,
    search_point,
)
from gpu_memory_preflight import estimate_gpu_memory, selected_device
from prepare_bitmaps import HEADER as BITMAP_HEADER
from prepare_bitmaps import BitmapWriter, Matrix, write_matrix

BASE_ROWS = 2_735_264
DIM = 4096
K = 10
MAX_QUERIES = 2048
TARGET = 0.95
GRAPH_DEGREE = 64
# SINGLE_CTA filtered search uses a normal visited hash with at most 2^20 slots
# and the default 0.5 maximum fill rate (search_plan.cuh).
MAX_HASH_VISITS = (1 << 20) // 2
WORKLOADS = ("em", "emis", "r")
EXPECTED_REAL_LOW = {"em": 24, "emis": 197, "r": 14}
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")
MATCHED_METHODS = ("default_cagra_seeded", "default_cagra_accumulator_seeded")
LOW_SELECTIVITIES = ("0.0001", "0.0003", "0.001", "0.003", "0.01")
CORRELATION_SELECTIVITIES = ("0.10", "0.25", "0.50", "0.90", "0.99")
INVALID = np.iinfo(np.uint32).max
MATRIX_HEADER = struct.Struct("<II")
SEED = 20260928


def followup_search_point(
    method: str, itopk: int, width: int, max_iterations: int
) -> dict:
    return search_point(
        method,
        itopk,
        width,
        max_iterations,
        max_queries=MAX_QUERIES,
        seed_policy="wd",
        graph_degree=GRAPH_DEGREE,
    )


def legal_hash_iterations(method: str, itopk: int, width: int) -> int:
    if method not in METHODS + MATCHED_METHODS or itopk <= 0 or width <= 0:
        raise ValueError(f"invalid deep-search cell: {method}, L={itopk}, W={width}")
    rounded_itopk = ((itopk + 31) // 32) * 32
    visit_multiplier = 2 if method == "navix_reference" else 1
    return (MAX_HASH_VISITS - rounded_itopk) // (
        width * GRAPH_DEGREE * visit_multiplier
    )


def deep_search_points(
    method: str,
    requested_iterations: int,
    measured: set[tuple[int, int, int]],
) -> list[dict]:
    points = []
    for itopk, width in DEEP_CELLS:
        iterations = min(
            requested_iterations, legal_hash_iterations(method, itopk, width)
        )
        cell = (itopk, width, iterations)
        if iterations <= 0:
            raise ValueError(f"no legal deep iterations for {method} at L/W={cell}")
        if cell not in measured:
            points.append(followup_search_point(method, *cell))
    return points


def inspect_required_data(data_root: Path) -> tuple[list[str], list[str]]:
    """Check A100 inputs using headers and file sizes, without reading vector payloads."""
    summaries: list[str] = []
    errors: list[str] = []
    data_root = data_root.resolve()

    def check_matrix(path: Path, rows: int, cols: int, label: str) -> None:
        try:
            with path.open("rb") as source:
                header = source.read(MATRIX_HEADER.size)
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{label}: cannot read {path}: {exc}")
            return
        if len(header) != MATRIX_HEADER.size:
            errors.append(f"{label}: truncated matrix header: {path}")
            return
        actual_rows, actual_cols = MATRIX_HEADER.unpack(header)
        expected_size = MATRIX_HEADER.size + rows * cols * 4
        if (actual_rows, actual_cols) != (rows, cols) or size != expected_size:
            errors.append(
                f"{label}: expected {rows}x{cols}, {expected_size} bytes; "
                f"found {actual_rows}x{actual_cols}, {size} bytes: {path}"
            )

    def check_bitmap(path: Path, rows: int, label: str) -> None:
        try:
            with path.open("rb") as source:
                header = source.read(BITMAP_HEADER.size)
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{label}: cannot read {path}: {exc}")
            return
        if len(header) != BITMAP_HEADER.size:
            errors.append(f"{label}: truncated bitmap header: {path}")
            return
        words = rows * ((BASE_ROWS + 31) // 32)
        expected_header = (b"CUVSBMAP", 1, 32, rows, BASE_ROWS, words)
        expected_size = BITMAP_HEADER.size + 4 * words
        if BITMAP_HEADER.unpack(header) != expected_header or size != expected_size:
            errors.append(
                f"{label}: bitmap header or size mismatch "
                f"(expected {expected_header}, {expected_size} bytes): {path}"
            )

    def manifest_path(value: object, label: str) -> Path | None:
        if not isinstance(value, str) or not value:
            errors.append(f"{label}: missing path in manifest")
            return None
        path = Path(value)
        if not path.is_absolute() or not path.resolve().is_relative_to(data_root):
            errors.append(
                f"{label}: manifest path is outside {data_root}: {path}"
            )
            return None
        return path

    check_matrix(
        data_root / "arxiv-for-fanns-large/base.fbin", BASE_ROWS, DIM, "base"
    )
    index_path = data_root / "arxiv-for-fanns-large/cagra_g64_ig128.index"
    try:
        if not index_path.is_file() or index_path.stat().st_size == 0:
            errors.append(f"degree-64 index is missing or empty: {index_path}")
    except OSError as exc:
        errors.append(f"degree-64 index cannot be read: {index_path}: {exc}")

    for workload in WORKLOADS:
        source = (
            data_root
            / "navix_bitmap"
            / "arxiv-large"
            / workload
            / "throughput_10000"
            / "manifest.json"
        )
        try:
            manifest = json.loads(source.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f"{workload}: cannot read manifest {source}: {exc}")
            continue
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != expected
            for key, expected in (
                ("schema_version", 1),
                ("dataset", "SPCL/arxiv-for-fanns-large"),
                ("predicate", workload),
                ("base_rows", BASE_ROWS),
                ("query_rows", 10_000),
            )
        ):
            errors.append(f"{workload}: unexpected manifest geometry: {source}")
            continue
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            errors.append(f"{workload}: no shards in {source}")
            continue
        cursor = 0
        for number, shard in enumerate(shards):
            label = f"{workload} shard {number}"
            if not isinstance(shard, dict):
                errors.append(f"{label}: malformed manifest entry")
                continue
            try:
                first = int(shard["first_query"])
                count = int(shard["query_count"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{label}: invalid query range: {exc}")
                continue
            if first != cursor or not 0 < count <= MAX_QUERIES:
                errors.append(
                    f"{label}: expected first_query={cursor} and "
                    f"1..{MAX_QUERIES} queries, found {first} and {count}"
                )
            cursor = first + count
            directory = manifest_path(shard.get("directory"), f"{label} directory")
            bitmap = manifest_path(shard.get("bitmap"), f"{label} bitmap")
            if directory is not None:
                check_matrix(directory / "query.bin", count, DIM, f"{label} query")
                check_matrix(
                    directory / "groundtruth.ibin", count, K, f"{label} GT"
                )
            if bitmap is not None:
                check_bitmap(bitmap, count, label)
        if cursor != 10_000:
            errors.append(f"{workload}: shard sequence ends at {cursor}, not 10000")
        summaries.append(f"{workload}: {len(shards)} shards, {cursor} queries")
    return summaries, errors


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(path: Path, dtype: str) -> np.ndarray:
    return Matrix(path, np.dtype(dtype)).values


def source_shards(
    data_root: Path, workload: str
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    manifest_path = (
        data_root
        / "navix_bitmap"
        / "arxiv-large"
        / workload
        / "throughput_10000"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        int(manifest["base_rows"]) != BASE_ROWS
        or int(manifest["query_rows"]) != 10_000
    ):
        raise ValueError(f"unexpected ArXiv source geometry: {manifest_path}")
    output = []
    cursor = 0
    for shard in manifest["shards"]:
        first, count = int(shard["first_query"]), int(shard["query_count"])
        if first != cursor or not 0 < count <= MAX_QUERIES:
            raise ValueError(f"noncontiguous ArXiv shard in {manifest_path}")
        directory = Path(shard["directory"])
        query = matrix(directory / "query.bin", "<f4")
        gt = matrix(directory / "groundtruth.ibin", "<u4")
        bitmap = Path(shard["bitmap"])
        with bitmap.open("rb") as source:
            header = BITMAP_HEADER.unpack(source.read(BITMAP_HEADER.size))
        if header[:3] != (b"CUVSBMAP", 1, 32) or header[3:5] != (
            count,
            BASE_ROWS,
        ):
            raise ValueError(f"bad bitmap geometry: {bitmap}")
        words = np.memmap(
            bitmap,
            dtype="<u4",
            mode="r",
            offset=BITMAP_HEADER.size,
            shape=(count, BASE_ROWS // 32),
        )
        if query.shape != (count, DIM) or gt.shape != (count, K):
            raise ValueError(f"bad query/GT geometry in {directory}")
        output.append((first, query, gt, words))
        cursor += count
    if cursor != 10_000:
        raise ValueError(f"incomplete source shards: {manifest_path}")
    return output


def popcounts(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words)
    if words.ndim != 2:
        raise ValueError("popcounts expects bitmap rows")
    output = np.empty(words.shape[0], dtype=np.int64)
    # The full unpacked 2,048-row bitmap would occupy >5 GiB; bound the temporary.
    lookup = np.array(
        [value.bit_count() for value in range(256)], dtype=np.uint8
    )
    for first in range(0, len(words), 32):
        chunk = np.asarray(words[first : first + 32]).view(np.uint8)
        output[first : first + len(chunk)] = lookup[chunk].sum(
            axis=1, dtype=np.int64
        )
    return output


def validate_gt(ids: np.ndarray, words: np.ndarray, count: int) -> None:
    if (
        ids.shape != (K,)
        or np.any(ids >= BASE_ROWS)
        or len(set(ids.tolist())) != K
    ):
        raise ValueError("GT must contain ten distinct, in-range IDs")
    if count < K or not np.all(words[ids >> 5] & (np.uint32(1) << (ids & 31))):
        raise ValueError("GT ID fails the bitmap")


def write_cohort(
    root: Path,
    name: str,
    query_rows: np.ndarray,
    bitmap_rows: list[np.ndarray],
    original_ids: list[int],
    *,
    gt_rows: np.ndarray | None,
    metadata: dict,
) -> Path:
    target = root / "data" / name
    manifest_path = target / "manifest.json"
    if target.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"incomplete cohort; refusing overwrite: {target}"
            )
        record = json.loads(manifest_path.read_text())
        if (
            record.get("metadata") != metadata
            or record.get("original_query_ids") != original_ids
        ):
            raise ValueError(f"cohort contract drifted: {target}")
        validate_existing_cohort(manifest_path)
        return manifest_path
    if (
        not len(query_rows)
        or len(query_rows) != len(bitmap_rows)
        or len(query_rows) != len(original_ids)
    ):
        raise ValueError("cohort row count mismatch")
    if gt_rows is not None and gt_rows.shape != (len(query_rows), K):
        raise ValueError("cohort GT shape mismatch")
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    shard = temporary / f"shard_00000_{len(query_rows):05d}"
    shard.mkdir()
    writer = BitmapWriter(shard / "filter.bitmap", len(query_rows), BASE_ROWS)
    output_words = writer.words.reshape(len(query_rows), BASE_ROWS // 32)
    counts = []
    for row, words in enumerate(bitmap_rows):
        if words.shape != (BASE_ROWS // 32,):
            raise ValueError("bitmap word row shape mismatch")
        output_words[row] = words
        count = int(popcounts(words.reshape(1, -1))[0])
        counts.append(count)
        if gt_rows is not None:
            validate_gt(np.asarray(gt_rows[row]), words, count)
    writer.close()
    write_matrix(shard / "query.bin", query_rows)
    if gt_rows is not None:
        write_matrix(shard / "groundtruth.ibin", gt_rows)
    final_shard = target / shard.name
    record = {
        "schema_version": 1,
        "bitmap_schema": "CUVSBMAP/v1/u32/row-major",
        "dataset": "SPCL/arxiv-for-fanns-large",
        "predicate": name,
        "base_rows": BASE_ROWS,
        "query_rows": len(query_rows),
        "original_query_ids": original_ids,
        "metadata": metadata,
        "shards": [
            {
                "first_query": 0,
                "query_count": len(query_rows),
                "directory": str(final_shard.resolve()),
                "bitmap": str((final_shard / "filter.bitmap").resolve()),
                "min_passing": min(counts),
                "max_passing": max(counts),
                "mean_selectivity": float(np.mean(counts) / BASE_ROWS),
                "empty_queries": 0,
            }
        ],
    }
    atomic_json(temporary / "manifest.json", record)
    os.replace(temporary, target)
    return manifest_path


def validate_existing_cohort(manifest_path: Path) -> dict:
    source = json.loads(manifest_path.read_text())
    rows = int(source["query_rows"])
    if rows <= 0 or len(source["shards"]) != 1:
        raise ValueError(f"invalid cohort manifest: {manifest_path}")
    shard = source["shards"][0]
    directory = Path(shard["directory"])
    query = directory / "query.bin"
    bitmap = directory / "filter.bitmap"
    if matrix(query, "<f4").shape != (rows, DIM):
        raise ValueError(f"incomplete cohort query matrix: {query}")
    with bitmap.open("rb") as stream:
        header = BITMAP_HEADER.unpack(stream.read(BITMAP_HEADER.size))
    if (
        header[:3] != (b"CUVSBMAP", 1, 32)
        or header[3:5] != (rows, BASE_ROWS)
        or header[5] != rows * BASE_ROWS // 32
        or bitmap.stat().st_size != BITMAP_HEADER.size + 4 * header[5]
    ):
        raise ValueError(f"incomplete cohort bitmap: {bitmap}")
    gt = directory / "groundtruth.ibin"
    if gt.is_file() and matrix(gt, "<u4").shape != (rows, K):
        raise ValueError(f"invalid cohort GT: {gt}")
    return source


def prepare_real(root: Path, data_root: Path) -> None:
    for workload in WORKLOADS:
        source_manifest = (
            data_root
            / "navix_bitmap"
            / "arxiv-large"
            / workload
            / "throughput_10000"
            / "manifest.json"
        )
        source_hash = sha256(source_manifest)
        shards = source_shards(data_root, workload)
        index: list[tuple[int, int, int, int]] = []
        for shard_index, (first, _, _, words) in enumerate(shards):
            counts = popcounts(words)
            index.extend(
                (first + local, shard_index, local, int(counts[local]))
                for local in range(len(counts))
                if counts[local] < 0.03 * BASE_ROWS
            )
        low = [item for item in index if item[3] < 0.01 * BASE_ROWS]
        medium = [item for item in index if item[3] >= 0.01 * BASE_ROWS]
        if len(low) != EXPECTED_REAL_LOW[workload]:
            raise ValueError(
                f"{workload} low cohort changed: {len(low)} vs "
                f"{EXPECTED_REAL_LOW[workload]}"
            )
        if not low or len(medium) < len(low):
            raise ValueError(f"insufficient real cohorts for {workload}")
        rng = np.random.default_rng(SEED + WORKLOADS.index(workload))
        mid_indices = sorted(
            rng.choice(len(medium), size=len(low), replace=False).tolist()
        )
        for label, chosen in (
            ("low", low),
            ("mid", [medium[i] for i in mid_indices]),
        ):
            ids = [item[0] for item in chosen]
            queries = np.stack(
                [shards[s][1][local] for _, s, local, _ in chosen]
            )
            gt = np.stack([shards[s][2][local] for _, s, local, _ in chosen])
            bitmap = [
                np.asarray(shards[s][3][local]).copy()
                for _, s, local, _ in chosen
            ]
            write_cohort(
                root,
                f"real_{workload}_{label}",
                queries,
                bitmap,
                ids,
                gt_rows=gt,
                metadata={
                    "kind": "real",
                    "workload": workload,
                    "bin": "<1%" if label == "low" else "1-3%",
                    "source_manifest_sha256": source_hash,
                    "selection_seed": SEED + WORKLOADS.index(workload),
                },
            )


def bitwords_from_ids(ids: np.ndarray) -> np.ndarray:
    words = np.zeros(BASE_ROWS // 32, dtype="<u4")
    ids = np.asarray(ids, dtype=np.int64)
    np.bitwise_or.at(
        words,
        ids >> 5,
        np.left_shift(np.uint32(1), (ids & 31).astype(np.uint32)),
    )
    return words


def sampled_words(
    rng: np.random.Generator, passing: int, nearby: np.ndarray | None = None
) -> np.ndarray:
    if nearby is None:
        if passing <= BASE_ROWS // 2:
            return bitwords_from_ids(
                rng.choice(BASE_ROWS, passing, replace=False)
            )
        words = np.full(BASE_ROWS // 32, np.iinfo(np.uint32).max, dtype="<u4")
        failed = rng.choice(BASE_ROWS, BASE_ROWS - passing, replace=False)
        np.bitwise_and.at(
            words,
            failed >> 5,
            ~np.left_shift(np.uint32(1), (failed & 31).astype(np.uint32)),
        )
        return words
    nearby = np.unique(np.asarray(nearby, dtype=np.int64))
    if len(nearby) != 1024 or nearby[0] < 0 or nearby[-1] >= BASE_ROWS:
        raise ValueError(
            "correlation requires 1,024 distinct exact-nearest IDs"
        )
    selectivity = passing / BASE_ROWS
    near_passing = round((selectivity + 0.5 * (1 - selectivity)) * len(nearby))
    near_pass_ids = rng.choice(nearby, near_passing, replace=False)
    outside_total = BASE_ROWS - len(nearby)
    outside_passing = passing - near_passing
    if not 0 <= outside_passing <= outside_total:
        raise ValueError("invalid correlated-cardinality construction")

    # Map integers in [0, N-1024) to dataset IDs while excluding the exact-nearest IDs.
    def outside_ids(draw: np.ndarray) -> np.ndarray:
        return draw + np.searchsorted(
            nearby - np.arange(len(nearby)), draw, side="right"
        )

    if passing <= BASE_ROWS // 2:
        outside = outside_ids(
            rng.choice(outside_total, outside_passing, replace=False)
        )
        return bitwords_from_ids(np.concatenate((near_pass_ids, outside)))
    words = np.full(BASE_ROWS // 32, np.iinfo(np.uint32).max, dtype="<u4")
    near_failed = np.setdiff1d(nearby, near_pass_ids, assume_unique=False)
    outside_failed = outside_ids(
        rng.choice(
            outside_total, outside_total - outside_passing, replace=False
        )
    )
    failed = np.concatenate((near_failed, outside_failed))
    np.bitwise_and.at(
        words,
        failed >> 5,
        ~np.left_shift(np.uint32(1), (failed & 31).astype(np.uint32)),
    )
    return words


def prepare_synthetic(
    root: Path, data_root: Path, kind: str, nearest: Path | None
) -> None:
    query_manifest_hash = sha256(
        data_root
        / "navix_bitmap/arxiv-large/em/throughput_10000/manifest.json"
    )
    source = source_shards(data_root, "em")[0][1]
    query_ids = list(range(MAX_QUERIES))
    nearest_rows = None
    if kind == "correlation":
        if nearest is None or not nearest.is_file():
            raise FileNotFoundError(
                "positive correlation requires exact nearest-1024 output"
            )
        nearest_rows = matrix(nearest, "<u4")
        if nearest_rows.shape != (MAX_QUERIES, 1024):
            raise ValueError("unexpected exact-nearest matrix shape")
    nearest_hash = sha256(nearest) if nearest_rows is not None else None
    levels = LOW_SELECTIVITIES if kind == "low" else CORRELATION_SELECTIVITIES
    for level in levels:
        fraction = float(level)
        passing = round(BASE_ROWS * fraction)
        for relation in (
            ("random",) if kind == "low" else ("random", "positive")
        ):
            name = f"synthetic_{kind}_{level.replace('.', 'p')}_{relation}"
            existing = root / "data" / name / "manifest.json"
            if existing.is_file():
                record = validate_existing_cohort(existing)
                meta = record["metadata"]
                if (
                    meta.get("relation") != relation
                    or float(meta.get("global_selectivity", -1)) != fraction
                    or int(meta.get("exact_passing", -1)) != passing
                    or int(meta.get("selection_seed", -1)) != SEED
                    or meta.get("source_manifest_sha256")
                    != query_manifest_hash
                    or meta.get("nearest_1024_sha256") != nearest_hash
                    or record.get("original_query_ids") != query_ids
                ):
                    raise ValueError(
                        f"synthetic cohort contract drifted: {existing}"
                    )
                continue
            bitmap_rows = []
            observed_near = []
            for query in query_ids:
                rng = np.random.default_rng(
                    SEED
                    + 1_000_003 * query
                    + int(fraction * 1_000_000) * 13
                    + (1 if relation == "positive" else 0)
                )
                near = (
                    np.asarray(nearest_rows[query])
                    if relation == "positive"
                    else None
                )
                words = sampled_words(rng, passing, near)
                bitmap_rows.append(words)
                if nearest_rows is not None:
                    ids = np.asarray(nearest_rows[query])
                    observed_near.append(
                        float(
                            np.mean(
                                (
                                    words[ids >> 5]
                                    & (np.uint32(1) << (ids & 31))
                                )
                                != 0
                            )
                        )
                    )
            write_cohort(
                root,
                name,
                np.asarray(source),
                bitmap_rows,
                query_ids,
                gt_rows=None,
                metadata={
                    "kind": "synthetic",
                    "relation": relation,
                    "global_selectivity": fraction,
                    "exact_passing": passing,
                    "nearest_1024_passing_mean": (
                        float(np.mean(observed_near))
                        if observed_near
                        else None
                    ),
                    "source_manifest_sha256": query_manifest_hash,
                    "nearest_1024_sha256": nearest_hash,
                    "selection_seed": SEED,
                },
            )


def graph_manifest(
    root: Path,
    data_root: Path,
    cohort: str,
    group: str,
    searches: list[dict],
    repetitions: int,
) -> Path:
    if not searches or {
        int(point["max_queries"]) for point in searches
    } != {MAX_QUERIES}:
        raise ValueError(
            f"{cohort}/{group}: every search point must use max_queries={MAX_QUERIES}"
        )
    for point in searches:
        method = str(point["bitmap_method"])
        maximum = int(point["max_iterations"])
        legal = legal_hash_iterations(
            method, int(point["itopk"]), int(point["search_width"])
        )
        if maximum > legal:
            raise ValueError(
                f"{cohort}/{group}: {method} L={point['itopk']} "
                f"W={point['search_width']} max_iterations={maximum} "
                f"exceeds normal-hash limit {legal}"
            )
    source_path = root / "data" / cohort / "manifest.json"
    source = json.loads(source_path.read_text())
    paths = DatasetPaths(
        source_path,
        "arxiv-for-fanns-large/base.fbin",
        "arxiv-for-fanns-large/cagra_g64_ig128.index",
        "float",
        64,
        128,
        BASE_ROWS,
        DIM,
    )
    directory = root / "graph" / "configs" / group / cohort
    directory.mkdir(parents=True, exist_ok=True)
    configs = []
    for shard_index, shard in enumerate(source["shards"]):
        config = directory / f"shard_{shard_index:02d}.json"
        if not config.is_file():
            atomic_json(
                config,
                config_payload(
                    workload=cohort,
                    phase="throughput",
                    shard=shard,
                    paths=paths,
                    searches=searches,
                ),
            )
        configs.append(
            {
                "config": str(config.resolve()),
                "shard_index": shard_index,
                "first_query": shard["first_query"],
                "query_count": shard["query_count"],
            }
        )
    manifest = directory / "manifest.json"
    payload = {
        "schema_version": 1,
        "experiment": "arxiv_selectivity_followup",
        "group": group,
        "phase": "throughput",
        "workload": cohort,
        "k": K,
        "max_queries": MAX_QUERIES,
        "graph_degree": 64,
        "intermediate_graph_degree": 128,
        "dataset_size": BASE_ROWS,
        "dimension": DIM,
        "repetitions": repetitions,
        "expected_queries": source["query_rows"],
        "expected_shards": len(configs),
        "source_bitmap_manifest": str(source_path.resolve()),
        "passing_seed_policy": "wd",
        "navix_seed_policy": "wd",
        "search_points": [point_identity(row) for row in searches],
        "configs": configs,
    }
    if manifest.is_file() and json.loads(manifest.read_text()) != payload:
        raise ValueError(f"graph manifest contract drifted: {manifest}")
    if not manifest.is_file():
        atomic_json(manifest, payload)
    return manifest


def benchmark(
    config: Path,
    output: Path,
    binary: Path,
    library: Path,
    data_root: Path,
    repetitions: int,
    minimum_time: str,
) -> None:
    if output.is_file():
        payload = json.loads(output.read_text())
        expected = len(
            json.loads(config.read_text())["index"][0]["search_params"]
        )
        rows = [
            row
            for row in payload.get("benchmarks", [])
            if row.get("run_type") == "iteration"
        ]
        if len(rows) == repetitions * expected and all(
            not row.get("error_occurred") and not row.get("skipped")
            for row in rows
        ):
            print(f"reuse {output}", flush=True)
            return
        raise ValueError(
            f"incomplete raw result; refusing overwrite: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(library.resolve()) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    command = [
        str(binary.resolve()),
        "--search",
        "--mode=throughput",
        "--threads=1",
        f"--data_prefix={data_root.resolve()}",
        f"--index_prefix={data_root.resolve()}",
        f"--benchmark_repetitions={repetitions}",
        f"--benchmark_min_time={minimum_time}",
        "--benchmark_min_warmup_time=0.01",
        "--benchmark_enable_random_interleaving=true",
        "--benchmark_report_aggregates_only=false",
        "--benchmark_out_format=json",
        f"--benchmark_out={output}",
        str(config),
    ]
    subprocess.run(command, check=True, env=env)


def run_graph_group(
    root: Path,
    data_root: Path,
    cohort: str,
    group: str,
    searches: list[dict],
    binary: Path,
    library: Path,
    repetitions: int = 3,
) -> None:
    manifest = graph_manifest(
        root, data_root, cohort, group, searches, repetitions
    )
    for shard in json.loads(manifest.read_text())["configs"]:
        output = (
            root
            / "graph"
            / "raw"
            / group
            / cohort
            / f"shard_{shard['shard_index']:02d}.json"
        )
        benchmark(
            Path(shard["config"]),
            output,
            binary,
            library,
            data_root,
            repetitions,
            "0.10s",
        )


def analyzed_rows(root: Path) -> list[graph_analysis.SummaryPoint]:
    manifests = sorted((root / "graph" / "configs").glob("*/*/manifest.json"))
    raw = []
    for path in manifests:
        raw.extend(
            graph_analysis.load_group(
                path, json.loads(path.read_text()), root / "graph" / "raw"
            )
        )
    return graph_analysis.summarize(
        graph_analysis.aggregate_repetitions(raw), TARGET
    )


def preflight_graph(
    root: Path, data_root: Path, binary: Path, library: Path
) -> None:
    for path in (
        data_root / "arxiv-for-fanns-large/base.fbin",
        data_root / "arxiv-for-fanns-large/cagra_g64_ig128.index",
        binary,
        library,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = selected_device()
    if (
        "A100" not in str(device["name"])
        or int(device["total_bytes"]) < 75 * 1024**3
    ):
        raise ValueError(
            f"expected NVIDIA A100 80 GB for paper follow-up, got {device}"
        )
    atomic_json(root / "state" / "gpu_preflight.json", device)


def run_graph(
    root: Path,
    data_root: Path,
    binary: Path,
    library: Path,
    cohort_prefix: str,
) -> None:
    preflight_graph(root, data_root, binary, library)
    names = sorted(
        path.parent.name
        for path in (root / "data").glob(f"{cohort_prefix}*/manifest.json")
    )
    if not names:
        raise ValueError(f"no prepared cohorts matching {cohort_prefix!r}")
    b0 = [
        followup_search_point(method, l, w, 0)
        for method in METHODS
        for l, w in B0_CELLS
    ]
    for cohort in names:
        source = json.loads(
            (root / "data" / cohort / "manifest.json").read_text()
        )
        if not (
            Path(source["shards"][0]["directory"]) / "groundtruth.ibin"
        ).is_file():
            raise ValueError(f"ground truth not finalized for {cohort}")
        run_graph_group(root, data_root, cohort, "b0", b0, binary, library)
    for cohort in names:
        for method in METHODS:
            measured: set[tuple[int, int, int]] = set()
            for iterations in DEEP_ITERATIONS:
                rows = analyzed_rows(root)
                if any(
                    row.workload == cohort
                    and row.method == method
                    and row.recall_min >= TARGET
                    for row in rows
                ):
                    break
                searches = deep_search_points(method, iterations, measured)
                if not searches:
                    break
                run_graph_group(
                    root,
                    data_root,
                    cohort,
                    f"deep_i{iterations}_{cohort}_{method}",
                    searches,
                    binary,
                    library,
                )
                measured.update(
                    (int(point["itopk"]), int(point["search_width"]),
                     int(point["max_iterations"]))
                    for point in searches
                )


def run_matched_seed_control(
    root: Path, data_root: Path, binary: Path, library: Path
) -> None:
    preflight_graph(root, data_root, binary, library)
    names = sorted(
        path.parent.name
        for path in (root / "data").glob(
            "synthetic_correlation_*/manifest.json"
        )
    )
    if not names:
        raise ValueError("no correlation cohorts prepared")
    b0 = [
        followup_search_point(method, l, w, 0)
        for method in MATCHED_METHODS
        for l, w in B0_CELLS
    ]
    for cohort in names:
        run_graph_group(
            root, data_root, cohort, "matched_b0", b0, binary, library
        )
    for cohort in names:
        for method in MATCHED_METHODS:
            measured: set[tuple[int, int, int]] = set()
            for iterations in DEEP_ITERATIONS:
                rows = analyzed_rows(root)
                if any(
                    row.workload == cohort
                    and row.method == method
                    and row.recall_min >= TARGET
                    for row in rows
                ):
                    break
                searches = deep_search_points(method, iterations, measured)
                if not searches:
                    break
                run_graph_group(
                    root,
                    data_root,
                    cohort,
                    f"matched_deep_i{iterations}_{cohort}_{method}",
                    searches,
                    binary,
                    library,
                )
                measured.update(
                    (int(point["itopk"]), int(point["search_width"]),
                     int(point["max_iterations"]))
                    for point in searches
                )


def exact_generation_config(
    root: Path, data_root: Path, cohort: str, *, width: int, output: Path
) -> Path:
    source = json.loads((root / "data" / cohort / "manifest.json").read_text())
    shard = source["shards"][0]
    marker = root / "state" / "cuvs_brute_force.index"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    config = root / "exact" / "configs" / cohort / f"k{width}.json"
    payload = {
        "dataset": {
            "name": f"selectivity-{cohort}-gt{width}",
            "base_file": "arxiv-for-fanns-large/base.fbin",
            "query_file": str(Path(shard["directory"]) / "query.bin"),
            "distance": "euclidean",
            "dtype": "float",
            "filter": {"kind": "bitmap", "file": shard["bitmap"]},
        },
        "search_basic_param": {"batch_size": source["query_rows"], "k": width},
        "index": [
            {
                "name": "cuvs-exact-bitmap",
                "algo": "cuvs_brute_force",
                "file": str(marker.resolve()),
                "build_param": {},
                "search_params": [
                    {
                        "exact_control": "bitmap_count_csr_search",
                        "resident_bitmap": True,
                        "benchmark_output_neighbors_file": str(
                            output.resolve()
                        ),
                    }
                ],
            }
        ],
    }
    if config.is_file() and json.loads(config.read_text()) != payload:
        raise ValueError(f"exact config drifted: {config}")
    if not config.is_file():
        atomic_json(config, payload)
    return config


def preflight_exact(root: Path, cohort: str, width: int) -> None:
    source = json.loads((root / "data" / cohort / "manifest.json").read_text())
    shard = source["shards"][0]
    rows = int(source["query_rows"])
    bitmap = Path(shard["bitmap"])
    passing = round(float(shard["mean_selectivity"]) * rows * BASE_ROWS)
    estimate = estimate_gpu_memory(
        base_rows=BASE_ROWS,
        dim=DIM,
        query_rows=rows,
        k=width,
        bitmap_storage_bytes=bitmap.stat().st_size,
        passing_count=passing,
    )
    device = selected_device()
    if int(device["free_bytes"]) < int(estimate["required_free_device_bytes"]):
        raise MemoryError(
            f"{cohort} exact k={width} needs "
            f"{estimate['required_free_device_bytes']} free GPU bytes; "
            f"{device['free_bytes']} available"
        )
    atomic_json(
        root / "exact" / "preflight" / f"{cohort}_k{width}.json",
        {"estimate": estimate, "device": device},
    )


def generate_gt(
    root: Path,
    data_root: Path,
    binary: Path,
    library: Path,
    cohort_prefix: str,
) -> None:
    for path in sorted(
        (root / "data").glob(f"{cohort_prefix}*/manifest.json")
    ):
        source = json.loads(path.read_text())
        cohort = path.parent.name
        shard = Path(source["shards"][0]["directory"])
        gt = shard / "groundtruth.ibin"
        if gt.is_file():
            continue
        preflight_exact(root, cohort, K)
        config = exact_generation_config(
            root, data_root, cohort, width=K, output=gt
        )
        raw = root / "exact" / "raw_gt" / f"{cohort}.json"
        benchmark(config, raw, binary, library, data_root, 1, "0.001s")
        ids = matrix(gt, "<u4")
        if ids.shape != (source["query_rows"], K):
            raise ValueError(f"bad generated GT shape: {gt}")
        bitmap = np.memmap(
            shard / "filter.bitmap",
            dtype="<u4",
            mode="r",
            offset=BITMAP_HEADER.size,
            shape=(len(ids), BASE_ROWS // 32),
        )
        for row in range(len(ids)):
            validate_gt(
                np.asarray(ids[row]),
                bitmap[row],
                int(source["shards"][0]["min_passing"]),
            )


def prepare_nearest(
    root: Path, data_root: Path, binary: Path, library: Path
) -> Path:
    # An all-pass bitmap makes the same cuVS exact path generate unfiltered top-1024 IDs.
    cohort = "nearest_all_pass"
    if not (root / "data" / cohort / "manifest.json").is_file():
        queries = np.asarray(source_shards(data_root, "em")[0][1])
        words = [
            np.full(BASE_ROWS // 32, np.iinfo(np.uint32).max, dtype="<u4")
            for _ in range(MAX_QUERIES)
        ]
        write_cohort(
            root,
            cohort,
            queries,
            words,
            list(range(MAX_QUERIES)),
            gt_rows=None,
            metadata={
                "kind": "nearest_prepass",
                "all_pass": True,
                "source_manifest_sha256": sha256(
                    data_root
                    / "navix_bitmap/arxiv-large/em/throughput_10000/manifest.json"
                ),
            },
        )
    output = root / "state" / "nearest_1024.ibin"
    preflight_exact(root, cohort, 1024)
    config = exact_generation_config(
        root, data_root, cohort, width=1024, output=output
    )
    benchmark(
        config,
        root / "exact" / "raw_gt" / "nearest_1024.json",
        binary,
        library,
        data_root,
        1,
        "0.001s",
    )
    if matrix(output, "<u4").shape != (MAX_QUERIES, 1024):
        raise ValueError("bad nearest-1024 output")
    return output


def exact_control(
    root: Path,
    data_root: Path,
    binary: Path,
    library: Path,
    cohort_prefix: str,
) -> None:
    marker = root / "state" / "cuvs_brute_force.index"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    for path in sorted(
        (root / "data").glob(f"{cohort_prefix}*/manifest.json")
    ):
        cohort = path.parent.name
        source = json.loads(path.read_text())
        shard = Path(source["shards"][0]["directory"])
        config = root / "exact" / "configs" / cohort / "control.json"
        payload = {
            "dataset": {
                "name": f"selectivity-{cohort}-exact-control",
                "base_file": "arxiv-for-fanns-large/base.fbin",
                "query_file": str(shard / "query.bin"),
                "groundtruth_neighbors_file": str(shard / "groundtruth.ibin"),
                "distance": "euclidean",
                "dtype": "float",
                "filter": {
                    "kind": "bitmap",
                    "file": str(shard / "filter.bitmap"),
                },
            },
            "search_basic_param": {"batch_size": source["query_rows"], "k": K},
            "index": [
                {
                    "name": "cuvs-exact-bitmap",
                    "algo": "cuvs_brute_force",
                    "file": str(marker.resolve()),
                    "build_param": {},
                    "search_params": [
                        {
                            "exact_control": "bitmap_count_csr_search",
                            "native_l2_cutoff_validation": True,
                            "resident_bitmap": True,
                        }
                    ],
                }
            ],
        }
        if not config.is_file():
            atomic_json(config, payload)
        elif json.loads(config.read_text()) != payload:
            raise ValueError(f"exact-control config drifted: {config}")
        output = root / "exact" / "raw_control" / f"{cohort}.json"
        preflight_exact(root, cohort, K)
        benchmark(config, output, binary, library, data_root, 3, "0.001s")


def analyze(root: Path) -> None:
    rows = analyzed_rows(root)
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    graph_analysis.write_csv(root / "analysis" / "graph_summary.csv", rows)
    graph_analysis.write_csv(
        root / "analysis" / "graph_pareto.csv",
        [
            row
            for cohort in sorted({item.workload for item in rows})
            for method in METHODS + MATCHED_METHODS
            for row in graph_analysis.pareto(
                [
                    item
                    for item in rows
                    if item.workload == cohort
                    and item.method == method
                    and item.paper_included
                ]
            )
        ],
    )
    plot_dir = root / "analysis" / "frontiers"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "default_cagra": "#4c78a8",
        "default_cagra_accumulator": "#e45756",
        "navix_reference": "#f58518",
        "default_cagra_seeded": "#72a5cf",
        "default_cagra_accumulator_seeded": "#ed8b87",
    }
    for cohort in sorted({item.workload for item in rows}):
        fig, axis = plt.subplots(figsize=(6.2, 3.8))
        for method in METHODS + MATCHED_METHODS:
            members = [
                row
                for row in rows
                if row.workload == cohort
                and row.method == method
                and row.paper_included
            ]
            if not members:
                continue
            axis.scatter(
                [row.recall_median for row in members],
                [row.qps_median for row in members],
                s=16,
                color=colors[method],
                alpha=0.35,
            )
            frontier = graph_analysis.pareto(members)
            axis.plot(
                [row.recall_median for row in frontier],
                [row.qps_median for row in frontier],
                marker="o",
                markersize=3,
                color=colors[method],
                label=method,
            )
        axis.axvline(TARGET, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Valid-GT Recall@10")
        axis.set_ylabel("Queries/s")
        axis.set_title(cohort)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{cohort}.pdf")
        plt.close(fig)
    selected = []
    for cohort in sorted({row.workload for row in rows}):
        present_methods = METHODS + tuple(
            method
            for method in MATCHED_METHODS
            if any(
                row.workload == cohort and row.method == method for row in rows
            )
        )
        for method in present_methods:
            candidates = [
                row
                for row in rows
                if row.workload == cohort and row.method == method
            ]
            if not candidates:
                raise ValueError(f"missing graph method {cohort}/{method}")
            reached = [row for row in candidates if row.recall_min >= TARGET]
            winner = (
                max(reached, key=lambda row: row.qps_median)
                if reached
                else max(
                    candidates,
                    key=lambda row: (row.recall_median, row.qps_median),
                )
            )
            selected.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "target_recall": TARGET,
                    "target_reached": bool(reached),
                    "max_measured_recall": max(
                        row.recall_median for row in candidates
                    ),
                    "recall": winner.recall_median,
                    "qps": winner.qps_median,
                    "qps_min": winner.qps_min,
                    "qps_max": winner.qps_max,
                    "itopk": winner.itopk,
                    "search_width": winner.search_width,
                    "max_iterations": winner.max_iterations,
                    "queries": winner.queries_per_repetition,
                    "repetitions": winner.repetitions,
                    "duplicate_output_query_rate_max": winner.duplicate_output_query_rate_max,
                    "underfilled_queries_max": winner.underfilled_queries_max,
                    "exploratory_small_batch": winner.queries_per_repetition
                    < 256,
                }
            )
    output = root / "analysis" / "selected.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    exact_rows = []
    for path in sorted((root / "exact" / "raw_control").glob("*.json")):
        payload = json.loads(path.read_text())
        records = [
            row
            for row in payload.get("benchmarks", [])
            if row.get("run_type") == "iteration"
        ]
        expected_queries = int(
            json.loads(
                (root / "data" / path.stem / "manifest.json").read_text()
            )["query_rows"]
        )
        if len(records) != 3 or sorted(
            int(row.get("repetition_index", -1)) for row in records
        ) != [0, 1, 2]:
            raise ValueError(f"exact control lacks three repetitions: {path}")
        for row in records:
            if row.get("error_occurred") or row.get("skipped"):
                raise ValueError(f"exact control failed: {path}")
            if (
                int(row["n_queries"]) != expected_queries
                or int(row["k"]) != K
                or float(row["items_per_second"]) <= 0
            ):
                raise ValueError(
                    f"exact control query/rate contract failed: {path}"
                )
            for key in (
                "FilterViolations",
                "InvalidSentinelErrors",
                "SentinelOrderErrors",
                "InvalidSentinelDistanceErrors",
                "DuplicateOutputQueries",
                "NativeL2StrictPrefixErrors",
            ):
                if float(row[key]) != 0:
                    raise ValueError(
                        f"exact control {key}={row[key]} in {path}"
                    )
            if float(row["NativeL2CutoffRecall"]) < 0.9999:
                raise ValueError(
                    f"exact control native-L2 cutoff failed: {path}"
                )
            if (
                float(row["NativeL2CutoffErrors"]) > 0.0001
                or float(row["NativeL2CutoffValidated"]) != 1
                or float(row["OutputSetSemanticsVersion"]) != 1
            ):
                raise ValueError(
                    f"exact control numerical/set contract failed: {path}"
                )
            exact_rows.append(
                {
                    "cohort": path.stem,
                    "repetition": row["repetition_index"],
                    "qps": row["items_per_second"],
                    "native_l2_cutoff_recall": row["NativeL2CutoffRecall"],
                }
            )
    if {item["cohort"] for item in exact_rows} != {
        item["cohort"] for item in selected
    }:
        raise ValueError("exact controls do not cover all graph cohorts")
    with (root / "analysis" / "exact_control.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exact_rows[0]))
        writer.writeheader()
        writer.writerows(exact_rows)
    print(
        json.dumps(
            {
                "cohorts": len({item["cohort"] for item in selected}),
                "selected": len(selected),
                "exact_controls": len(exact_rows),
            },
            indent=2,
        )
    )


def bundle(root: Path, archive: Path) -> None:
    if not (root / "analysis" / "selected.csv").is_file():
        raise ValueError("analyze results before bundling")
    if archive.exists():
        raise FileExistsError(
            f"refusing to replace immutable bundle: {archive}"
        )
    fingerprints = {}
    for manifest in sorted((root / "data").glob("*/manifest.json")):
        cohort = manifest.parent.name
        source = json.loads(manifest.read_text())
        files = {}
        for shard in source["shards"]:
            directory = Path(shard["directory"])
            for name in ("query.bin", "filter.bitmap", "groundtruth.ibin"):
                path = directory / name
                if path.is_file():
                    files[name] = {
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
        fingerprints[cohort] = files
    atomic_json(root / "state" / "cohort_file_hashes.json", fingerprints)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp.{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as output:
        for directory_name in (
            "analysis",
            "graph/configs",
            "graph/raw",
            "exact/configs",
            "exact/preflight",
            "exact/raw_gt",
            "exact/raw_control",
            "state",
        ):
            directory = root / directory_name
            if directory.exists():
                output.add(
                    directory, arcname=f"selectivity_followup/{directory_name}"
                )
        for manifest in sorted((root / "data").glob("*/manifest.json")):
            cohort = manifest.parent.name
            output.add(
                manifest,
                arcname=f"selectivity_followup/data/{cohort}/manifest.json",
            )
            for gt in sorted(manifest.parent.glob("shard_*/groundtruth.ibin")):
                output.add(
                    gt,
                    arcname=f"selectivity_followup/data/{cohort}/{gt.parent.name}/groundtruth.ibin",
                )
    os.replace(temporary, archive)
    print(
        json.dumps(
            {"archive": str(archive), "sha256": sha256(archive)}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "check-data",
            "prepare-real",
            "prepare-synthetic-low",
            "prepare-nearest",
            "prepare-synthetic-correlation",
            "generate-gt",
            "graph",
            "matched-control",
            "exact-control",
            "analyze",
            "bundle",
        ),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/retrieve_data")
    )
    parser.add_argument("--cohort-prefix", default="")
    parser.add_argument(
        "--graph-binary",
        type=Path,
        default=REPO / "cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH",
    )
    parser.add_argument(
        "--exact-binary",
        type=Path,
        default=REPO / "cpp/build/bench/ann/CUVS_BRUTE_FORCE_ANN_BENCH",
    )
    parser.add_argument(
        "--library", type=Path, default=REPO / "cpp/build/libcuvs.so"
    )
    parser.add_argument("--bundle-path", type=Path)
    args = parser.parse_args()
    if args.stage == "check-data":
        summaries, errors = inspect_required_data(args.data_root)
        for summary in summaries:
            print(summary, flush=True)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(f"data preflight failed: {len(errors)} problem(s)")
        print(f"data preflight OK: {args.data_root.resolve()}")
        return
    if args.root is None:
        parser.error("--root is required for this stage")
    root, data_root = args.root.resolve(), args.data_root.resolve()
    if args.stage == "prepare-real":
        prepare_real(root, data_root)
    elif args.stage == "prepare-synthetic-low":
        prepare_synthetic(root, data_root, "low", None)
    elif args.stage == "prepare-nearest":
        prepare_nearest(root, data_root, args.exact_binary, args.library)
    elif args.stage == "prepare-synthetic-correlation":
        prepare_synthetic(
            root, data_root, "correlation", root / "state/nearest_1024.ibin"
        )
    elif args.stage == "generate-gt":
        generate_gt(
            root,
            data_root,
            args.exact_binary,
            args.library,
            args.cohort_prefix,
        )
    elif args.stage == "graph":
        run_graph(
            root,
            data_root,
            args.graph_binary,
            args.library,
            args.cohort_prefix,
        )
    elif args.stage == "matched-control":
        run_matched_seed_control(
            root, data_root, args.graph_binary, args.library
        )
    elif args.stage == "exact-control":
        exact_control(
            root,
            data_root,
            args.exact_binary,
            args.library,
            args.cohort_prefix,
        )
    elif args.stage == "bundle":
        if args.bundle_path is None:
            parser.error("bundle requires --bundle-path")
        bundle(root, args.bundle_path.resolve())
    else:
        analyze(root)


if __name__ == "__main__":
    main()
