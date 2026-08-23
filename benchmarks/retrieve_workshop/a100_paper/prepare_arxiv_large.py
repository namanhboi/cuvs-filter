#!/usr/bin/env python3
"""Prepare bounded-memory ArXiv-large bitmap workloads for the A100 paper rerun.

The raw SPCL files are kept intact.  Vector conversion is streamed, base metadata is reduced to
compact numeric arrays in one JSONL pass, and resident bitmaps are emitted in independently usable
query shards.  Ground truth is parsed as standard count-prefixed ivecs and every retained ID is
checked against the corresponding predicate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RETRIEVE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(RETRIEVE_DIR / "gpu_graph"))
sys.path.insert(0, str(RETRIEVE_DIR.parent / "favor" / "navix_bitmap"))
sys.path.insert(0, str(RETRIEVE_DIR.parent / "favor" / "arxiv_udf"))

from convert_fvecs_to_fbin import convert, geometry
from prepare_bitmaps import BitmapWriter
from prepare_workloads import read_ivec_first_k

EXPECTED_ROWS = 2_735_264
EXPECTED_DIM = 4096
QUERY_ROWS = 10_000
K = 10
THROUGHPUT_SHARD = 2_048
INVALID_PADDING_START = int(np.iinfo(np.uint32).max) - 999
MATRIX_HEADER = struct.Struct("<II")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
            if limit is not None and len(rows) == limit:
                break
    if limit is not None and len(rows) != limit:
        raise ValueError(
            f"{path} has {len(rows)} rows, expected at least {limit}"
        )
    return rows


def fbin_values(path: Path) -> np.memmap:
    with path.open("rb") as source:
        header = source.read(MATRIX_HEADER.size)
    if len(header) != MATRIX_HEADER.size:
        raise ValueError(f"truncated fbin: {path}")
    rows, cols = MATRIX_HEADER.unpack(header)
    expected = MATRIX_HEADER.size + rows * cols * 4
    if path.stat().st_size != expected:
        raise ValueError(f"invalid fbin geometry: {path}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, cols))


def write_matrix(path: Path, rows: np.ndarray, dtype: np.dtype) -> None:
    values = np.asarray(rows, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(MATRIX_HEADER.pack(values.shape[0], values.shape[1]))
        values.tofile(output)


def compact_metadata(
    source: Path, rows: int, query_emis_labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_emis = list(dict.fromkeys(query_emis_labels))
    label_bits = {label: bit for bit, label in enumerate(unique_emis)}
    mask_words = (len(unique_emis) + 63) // 64
    em = np.empty(rows, dtype=np.int32)
    dates = np.empty(rows, dtype=np.int32)
    emis = np.zeros((rows, mask_words), dtype=np.uint64)
    count = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if count >= rows:
                raise ValueError(
                    f"{source} has more than {rows} nonempty rows"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {source}:{line_number}"
                ) from exc
            em[count] = int(record.get("number_of_sub_categories", 0))
            dates[count] = int(record["update_date"])
            for label in record.get("main_categories", []):
                bit = label_bits.get(label)
                if bit is not None:
                    emis[count, bit // 64] |= np.uint64(1) << np.uint64(
                        bit % 64
                    )
            count += 1
            if count % 250_000 == 0:
                print(f"metadata {count}/{rows}", flush=True)
    if count != rows:
        raise ValueError(f"{source} has {count} rows, expected {rows}")
    return em, dates, emis


def packed_words(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(np.asarray(mask, dtype=np.bool_), bitorder="little")
    if packed.size % 4:
        packed = np.pad(packed, (0, 4 - packed.size % 4))
    return packed.view("<u4")


def valid_gt_ids(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.uint32)
    return values[values < INVALID_PADDING_START].astype(np.int64, copy=False)


def materialize_phase(
    *,
    output: Path,
    phase: str,
    query_count: int,
    shard_size: int,
    base_rows: int,
    queries: np.ndarray,
    ground_truth: dict[str, np.ndarray],
    bitmap_row: dict[str, Callable[[int], tuple[np.ndarray, int]]],
    reuse_valid: bool,
) -> None:
    for workload in ("em", "emis", "r"):
        target = output / workload / f"{phase}_{query_count}"
        manifest_path = target / "manifest.json"
        if manifest_path.is_file() and reuse_valid:
            manifest = json.loads(manifest_path.read_text())
            if (
                int(manifest.get("base_rows", -1)) == base_rows
                and int(manifest.get("query_rows", -1)) == query_count
                and sum(
                    int(row["query_count"])
                    for row in manifest.get("shards", [])
                )
                == query_count
                and all(
                    Path(row["bitmap"]).is_file() for row in manifest["shards"]
                )
            ):
                print(f"reusing {manifest_path}")
                continue
            raise ValueError(f"stale prepared workload: {manifest_path}")
        if target.exists():
            raise FileExistsError(f"refusing to replace {target}")
        temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.mkdir(parents=True)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "bitmap_schema": "CUVSBMAP/v1/u32/row-major",
            "dataset": "SPCL/arxiv-for-fanns-large",
            "predicate": workload,
            "base_rows": base_rows,
            "query_rows": query_count,
            "shards": [],
        }
        try:
            for first in range(0, query_count, shard_size):
                end = min(query_count, first + shard_size)
                count = end - first
                shard = temporary / f"shard_{first:05d}_{end:05d}"
                shard.mkdir()
                writer = BitmapWriter(
                    shard / "filter.bitmap", count, base_rows
                )
                if base_rows % 32:
                    raise ValueError(
                        "A100 preparer requires bitmap rows aligned to 32 bits"
                    )
                words = writer.words.reshape(count, base_rows // 32)
                passing: list[int] = []
                for local, query_id in enumerate(range(first, end)):
                    row_words, passing_count = bitmap_row[workload](query_id)
                    if row_words.shape != (base_rows // 32,):
                        raise ValueError(
                            f"predicate {workload} returned {row_words.shape}"
                        )
                    ids = valid_gt_ids(ground_truth[workload][query_id])
                    gt_passes = (
                        row_words[ids >> 5]
                        & (np.uint32(1) << (ids & 31).astype(np.uint32))
                    ) != 0
                    if ids.size and not np.all(gt_passes):
                        bad = int(ids[np.flatnonzero(~gt_passes)[0]])
                        raise ValueError(
                            f"{workload} GT violation: query={query_id}, node={bad}"
                        )
                    words[local] = row_words
                    passing.append(passing_count)
                writer.close()
                write_matrix(
                    shard / "query.bin", queries[first:end], np.dtype("<f4")
                )
                write_matrix(
                    shard / "groundtruth.ibin",
                    ground_truth[workload][first:end],
                    np.dtype("<u4"),
                )
                resolved = target / shard.name
                manifest["shards"].append(
                    {
                        "first_query": first,
                        "query_count": count,
                        "directory": str(resolved.resolve()),
                        "bitmap": str((resolved / "filter.bitmap").resolve()),
                        "min_passing": min(passing),
                        "max_passing": max(passing),
                        "mean_selectivity": float(
                            np.mean(passing) / base_rows
                        ),
                        "empty_queries": int(
                            np.count_nonzero(np.asarray(passing) == 0)
                        ),
                    }
                )
                print(f"{workload} {phase}: {end}/{query_count}", flush=True)
            # Paths above intentionally name the final directory, not the temporary transaction.
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def prepare(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    dataset = args.data_root.resolve() / "arxiv-for-fanns-large"
    dataset.mkdir(parents=True, exist_ok=True)
    base_source = source / "database_vectors.fvecs"
    query_source = source / "query_vectors.fvecs"
    base_geometry = geometry(base_source)
    query_geometry = geometry(query_source)
    if base_geometry[:2] != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError(
            f"unexpected ArXiv-large base geometry: {base_geometry[:2]}"
        )
    if query_geometry[0] < QUERY_ROWS or query_geometry[1] != EXPECTED_DIM:
        raise ValueError(
            f"unexpected ArXiv-large query geometry: {query_geometry[:2]}"
        )
    convert(
        base_source,
        dataset / "base.fbin",
        args.vector_chunk_rows,
        args.reuse_valid,
    )
    convert(
        query_source,
        dataset / "query.fbin",
        args.vector_chunk_rows,
        args.reuse_valid,
    )
    queries = fbin_values(dataset / "query.fbin")[:QUERY_ROWS]

    em_query_rows = read_jsonl(
        source / "em_query_attributes.jsonl", QUERY_ROWS
    )
    emis_query_rows = read_jsonl(
        source / "emis_query_attributes.jsonl", QUERY_ROWS
    )
    r_query_rows = read_jsonl(source / "r_query_attributes.jsonl", QUERY_ROWS)
    em_query = np.asarray(
        [int(row["label"]) for row in em_query_rows], dtype=np.int32
    )
    emis_labels = [str(row["label"]) for row in emis_query_rows]
    r_query = np.asarray(
        [
            [int(row["range_start"]), int(row["range_end"])]
            for row in r_query_rows
        ],
        dtype=np.int32,
    )
    if np.any(r_query[:, 0] > r_query[:, 1]):
        raise ValueError("ArXiv range query has start > end")

    cache = dataset / "compact_filter_metadata.npz"
    if cache.is_file() and args.reuse_valid:
        with np.load(cache) as values:
            em_base = values["em"]
            dates = values["dates"]
            emis_masks = values["emis"]
            cached_labels = list(values["emis_labels"].astype(str))
        if cached_labels != list(
            dict.fromkeys(emis_labels)
        ) or em_base.shape != (EXPECTED_ROWS,):
            raise ValueError(f"stale compact metadata: {cache}")
        if (
            dates.shape != (EXPECTED_ROWS,)
            or emis_masks.shape[0] != EXPECTED_ROWS
        ):
            raise ValueError(f"stale compact metadata geometry: {cache}")
    else:
        if cache.exists():
            raise FileExistsError(cache)
        em_base, dates, emis_masks = compact_metadata(
            source / "database_attributes.jsonl", EXPECTED_ROWS, emis_labels
        )
        temporary_cache = cache.with_name(
            f".{cache.name}.partial.{os.getpid()}.npz"
        )
        np.savez(
            temporary_cache,
            em=em_base,
            dates=dates,
            emis=emis_masks,
            emis_labels=np.asarray(list(dict.fromkeys(emis_labels))),
        )
        os.replace(temporary_cache, cache)

    label_to_bit = {
        label: bit for bit, label in enumerate(dict.fromkeys(emis_labels))
    }
    ground_truth = {
        workload: read_ivec_first_k(
            source / f"ground_truth_{workload}.ivecs", QUERY_ROWS, K
        )
        for workload in ("em", "emis", "r")
    }

    em_cache: dict[int, tuple[np.ndarray, int]] = {}
    emis_cache: dict[str, tuple[np.ndarray, int]] = {}

    def encode(mask: np.ndarray) -> tuple[np.ndarray, int]:
        return packed_words(mask), int(np.count_nonzero(mask))

    def em_bitmap(query_id: int) -> tuple[np.ndarray, int]:
        label = int(em_query[query_id])
        if label not in em_cache:
            em_cache[label] = encode(em_base == label)
        return em_cache[label]

    def emis_bitmap(query_id: int) -> tuple[np.ndarray, int]:
        label = emis_labels[query_id]
        if label not in emis_cache:
            bit = label_to_bit[label]
            emis_cache[label] = encode(
                (
                    emis_masks[:, bit // 64]
                    & (np.uint64(1) << np.uint64(bit % 64))
                )
                != 0
            )
        return emis_cache[label]

    def range_bitmap(query_id: int) -> tuple[np.ndarray, int]:
        start, end = r_query[query_id]
        return encode((dates >= start) & (dates <= end))

    bitmap_rows = {"em": em_bitmap, "emis": emis_bitmap, "r": range_bitmap}
    bitmap_root = args.data_root.resolve() / "navix_bitmap" / "arxiv-large"
    materialize_phase(
        output=bitmap_root,
        phase="correctness",
        query_count=1_000,
        shard_size=1_000,
        base_rows=EXPECTED_ROWS,
        queries=queries,
        ground_truth=ground_truth,
        bitmap_row=bitmap_rows,
        reuse_valid=args.reuse_valid,
    )
    materialize_phase(
        output=bitmap_root,
        phase="throughput",
        query_count=QUERY_ROWS,
        shard_size=THROUGHPUT_SHARD,
        base_rows=EXPECTED_ROWS,
        queries=queries,
        ground_truth=ground_truth,
        bitmap_row=bitmap_rows,
        reuse_valid=args.reuse_valid,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory containing raw HF files",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vector-chunk-rows", type=int, default=1_024)
    parser.add_argument("--reuse-valid", action="store_true")
    args = parser.parse_args()
    if args.vector_chunk_rows <= 0:
        raise ValueError("--vector-chunk-rows must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
