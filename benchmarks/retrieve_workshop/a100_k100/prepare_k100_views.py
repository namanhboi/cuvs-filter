#!/usr/bin/env python3
"""Create lightweight k=100 workload views over the existing A100 bitmap shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np

K = 100
WORKLOADS = ("em", "emis", "r")
PHASES = ("correctness_1000", "throughput_10000")
MATRIX_HEADER = struct.Struct("<II")
BITMAP_HEADER = struct.Struct("<8sIIQQQ")
BITMAP_MAGIC = b"CUVSBMAP"
INVALID_U32 = int(np.iinfo(np.uint32).max)
INVALID_PADDING_START = INVALID_U32 - 999
VIEW_SCHEMA_VERSION = 3
VIEW_KIND = "symlinked_bitmap_query_with_k100_groundtruth_unique_padding_v1"
LEGACY_VIEW_KIND = "symlinked_bitmap_query_with_k100_groundtruth"


def matrix_info(path: Path, dtype: np.dtype) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(MATRIX_HEADER.size)
    if len(header) != MATRIX_HEADER.size:
        raise ValueError(f"truncated matrix: {path}")
    rows, cols = MATRIX_HEADER.unpack(header)
    expected = MATRIX_HEADER.size + rows * cols * dtype.itemsize
    if rows <= 0 or cols <= 0 or path.stat().st_size != expected:
        raise ValueError(f"matrix size mismatch: {path}")
    return rows, cols


def write_u32_matrix(path: Path, values: np.ndarray) -> None:
    matrix = np.ascontiguousarray(values, dtype="<u4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(MATRIX_HEADER.pack(*matrix.shape))
        matrix.tofile(output)


def resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def read_ivecs_k100(
    path: Path, query_count: int, base_rows: int
) -> np.ndarray:
    values = np.memmap(path, dtype="<u4", mode="r")
    rows = np.empty((query_count, K), dtype=np.uint32)
    cursor = 0
    for query in range(query_count):
        if cursor >= values.size:
            raise ValueError(f"truncated ivecs before query {query}: {path}")
        count = int(values[cursor])
        cursor += 1
        if count < K or cursor + count > values.size:
            raise ValueError(
                f"query {query} has invalid/count<{K} ivecs row: count={count}"
            )
        source = np.asarray(values[cursor : cursor + count], dtype=np.uint32)
        cursor += count
        selected: list[int] = []
        seen: set[int] = set()
        for raw in source:
            node = int(raw)
            if node >= INVALID_PADDING_START:
                continue
            if node >= base_rows:
                raise ValueError(
                    f"out-of-range GT ID: query={query}, node={node}"
                )
            if node not in seen:
                seen.add(node)
                selected.append(node)
            if len(selected) == K:
                break
        if len(selected) != K:
            raise ValueError(
                f"query {query} has only {len(selected)} distinct valid GT IDs; require {K}"
            )
        rows[query] = selected
    if cursor != values.size:
        raise ValueError(f"ivecs has trailing or excess rows: {path}")
    return rows


def bitmap_values(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as source:
        raw = source.read(BITMAP_HEADER.size)
    if len(raw) != BITMAP_HEADER.size:
        raise ValueError(f"truncated bitmap: {path}")
    magic, version, word_bits, rows, cols, words = BITMAP_HEADER.unpack(raw)
    expected_words = (rows * cols + 31) // 32
    if (
        magic != BITMAP_MAGIC
        or version != 1
        or word_bits != 32
        or words != expected_words
        or path.stat().st_size != BITMAP_HEADER.size + words * 4
    ):
        raise ValueError(f"unsupported bitmap: {path}")
    payload = np.memmap(
        path, dtype="<u4", mode="r", offset=BITMAP_HEADER.size, shape=(words,)
    )
    return payload, int(rows), int(cols)


def validate_gt_membership(
    bitmap: Path, gt: np.ndarray, first_query: int
) -> None:
    words, rows, cols = bitmap_values(bitmap)
    if rows != gt.shape[0]:
        raise ValueError(f"bitmap/GT row mismatch: {bitmap}")
    for local, ids in enumerate(gt):
        valid = ids < cols
        invalid_positions = np.flatnonzero(~valid)
        valid_count = (
            int(invalid_positions[0]) if invalid_positions.size else ids.size
        )
        expected_padding = np.uint32(INVALID_U32) - np.arange(
            ids.size - valid_count, dtype=np.uint32
        )
        if not np.array_equal(ids[valid_count:], expected_padding):
            raise ValueError(
                f"invalid GT sentinel ordering: query={first_query + local}"
            )
        legal = ids[:valid_count]
        if np.unique(legal).size != legal.size:
            raise ValueError(f"duplicate GT ID: query={first_query + local}")
        flat = np.uint64(local * cols) + legal.astype(np.uint64)
        passed = (
            words[flat >> np.uint64(5)]
            & np.left_shift(
                np.uint32(1), (flat & np.uint64(31)).astype(np.uint32)
            )
        ) != 0
        if not np.all(passed):
            bad = int(legal[np.flatnonzero(~passed)[0]])
            raise ValueError(
                f"GT predicate violation: query={first_query + local}, node={bad}"
            )


def make_loader_safe_padding(values: np.ndarray, base_rows: int) -> np.ndarray:
    """Replace a canonical repeated-invalid suffix with distinct invalid GT-map keys."""
    result = np.ascontiguousarray(values, dtype="<u4").copy()
    for query, ids in enumerate(result):
        invalid_positions = np.flatnonzero(ids >= base_rows)
        valid_count = (
            int(invalid_positions[0]) if invalid_positions.size else ids.size
        )
        if np.any(ids[:valid_count] >= base_rows) or np.any(
            ids[valid_count:] != INVALID_U32
        ):
            raise ValueError(
                f"generated YFCC GT has a malformed invalid suffix: query={query}"
            )
        ids[valid_count:] = np.uint32(INVALID_U32) - np.arange(
            ids.size - valid_count, dtype=np.uint32
        )
    return result


def read_generated_yfcc_gt(path: Path) -> np.ndarray:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("method") != "cuvs_brute_force_knn_masked_gt_generation"
        or int(manifest.get("k", -1)) != K
        or int(manifest.get("query_rows", -1)) != 10_000
        or manifest.get("complete") is not True
    ):
        raise ValueError(f"incomplete or stale YFCC k=100 GT manifest: {path}")
    result = np.empty((10_000, K), dtype=np.uint32)
    cursor = 0
    for shard in manifest["shards"]:
        first = int(shard["first_query"])
        count = int(shard["query_count"])
        if first != cursor:
            raise ValueError("YFCC k=100 GT shards are not contiguous")
        gt_path = Path(shard["groundtruth_file"]).resolve()
        if matrix_info(gt_path, np.dtype("<u4")) != (count, K):
            raise ValueError(f"YFCC generated GT shape changed: {gt_path}")
        result[first : first + count] = np.memmap(
            gt_path,
            dtype="<u4",
            mode="r",
            offset=MATRIX_HEADER.size,
            shape=(count, K),
        )
        cursor += count
    if cursor != result.shape[0]:
        raise ValueError("YFCC k=100 GT does not cover all queries")
    return make_loader_safe_padding(result, int(manifest["base_rows"]))


def symlink(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.symlink_to(source.resolve())


def provenance(path: Path) -> dict[str, object]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def valid_cached_view(
    target: Path,
    source_manifest: Path,
    ground_truth_source: dict[str, object],
) -> bool:
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if (
            int(manifest.get("schema_version", -1)) != VIEW_SCHEMA_VERSION
            or int(manifest.get("k", -1)) != K
            or manifest.get("view_kind") != VIEW_KIND
            or Path(manifest.get("source_bitmap_manifest", "")).resolve()
            != source_manifest.resolve()
            or manifest.get("ground_truth_source") != ground_truth_source
        ):
            return False
        cursor = 0
        for shard in manifest.get("shards", []):
            if int(shard["first_query"]) != cursor:
                return False
            count = int(shard["query_count"])
            directory = resolve(manifest_path, shard["directory"])
            if matrix_info(
                directory / "groundtruth.ibin", np.dtype("<u4")
            ) != (
                count,
                K,
            ):
                return False
            if (
                not (directory / "query.bin").is_file()
                or not (directory / "filter.bitmap").is_file()
            ):
                return False
            cursor += count
        return cursor == int(manifest["query_rows"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def create_view(
    *,
    source_manifest: Path,
    target: Path,
    ground_truth: np.ndarray,
    ground_truth_source: dict[str, object],
) -> None:
    if valid_cached_view(target, source_manifest, ground_truth_source):
        print(f"reuse {target / 'manifest.json'}")
        return
    if target.exists():
        manifest_path = target / "manifest.json"
        try:
            stale = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise FileExistsError(
                f"unrecognized stale k=100 view; refusing to replace: {target}"
            ) from error
        if int(stale.get("k", -1)) != K or stale.get("view_kind") not in {
            LEGACY_VIEW_KIND,
            VIEW_KIND,
        }:
            raise FileExistsError(
                f"unrecognized stale k=100 view; refusing to replace: {target}"
            )
        print(f"replace incompatible generated k=100 view {target}")
        shutil.rmtree(target)
    source = json.loads(source_manifest.read_text())
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    output = dict(source)
    output.update(
        {
            "schema_version": VIEW_SCHEMA_VERSION,
            "k": K,
            "view_kind": VIEW_KIND,
            "source_bitmap_manifest": str(source_manifest.resolve()),
            "ground_truth_source": ground_truth_source,
            "shards": [],
        }
    )
    try:
        for shard_number, shard in enumerate(source["shards"]):
            first = int(shard["first_query"])
            count = int(shard["query_count"])
            source_directory = resolve(source_manifest, shard["directory"])
            source_bitmap = resolve(source_manifest, shard["bitmap"])
            local = (
                temporary
                / f"shard_{shard_number:02d}_{first:05d}_{first + count:05d}"
            )
            local.mkdir()
            symlink(source_directory / "query.bin", local / "query.bin")
            symlink(source_bitmap, local / "filter.bitmap")
            gt_slice = ground_truth[first : first + count]
            if gt_slice.shape != (count, K):
                raise ValueError("k=100 GT slice does not cover source shard")
            write_u32_matrix(local / "groundtruth.ibin", gt_slice)
            validate_gt_membership(source_bitmap, np.asarray(gt_slice), first)
            final_directory = target / local.name
            row = dict(shard)
            row.update(
                {
                    "directory": str(final_directory.resolve()),
                    "bitmap": str(
                        (final_directory / "filter.bitmap").resolve()
                    ),
                }
            )
            output["shards"].append(row)
        output["query_rows"] = sum(
            int(row["query_count"]) for row in output["shards"]
        )
        (temporary / "manifest.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arxiv-raw", type=Path, required=True)
    parser.add_argument("--yfcc-gt-manifest", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    source_root = data_root / "navix_bitmap"
    target_root = data_root / "navix_bitmap_k100"

    yfcc_gt = read_generated_yfcc_gt(args.yfcc_gt_manifest.resolve())
    yfcc_gt_source = provenance(args.yfcc_gt_manifest)
    for phase in PHASES:
        create_view(
            source_manifest=source_root / "yfcc" / phase / "manifest.json",
            target=target_root / "yfcc" / phase,
            ground_truth=yfcc_gt,
            ground_truth_source=yfcc_gt_source,
        )

    base_rows, base_dim = matrix_info(
        data_root / "arxiv-for-fanns-large" / "base.fbin", np.dtype("<f4")
    )
    if (base_rows, base_dim) != (2_735_264, 4096):
        raise ValueError(
            f"unexpected ArXiv-large geometry: rows={base_rows}, dim={base_dim}"
        )
    for workload in WORKLOADS:
        raw_gt = args.arxiv_raw / f"ground_truth_{workload}.ivecs"
        gt = read_ivecs_k100(raw_gt, 10_000, base_rows)
        for phase in PHASES:
            create_view(
                source_manifest=(
                    source_root
                    / "arxiv-large"
                    / workload
                    / phase
                    / "manifest.json"
                ),
                target=target_root / "arxiv-large" / workload / phase,
                ground_truth=gt,
                ground_truth_source=provenance(raw_gt),
            )


if __name__ == "__main__":
    main()
