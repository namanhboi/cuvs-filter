#!/usr/bin/env python3
"""Materialize exact per-query bitmaps for filtered CAGRA benchmark workloads.

The output is a little-endian, row-major bit matrix.  It deliberately contains no vector values
or filter implementation: the search path sees only a bitmap and therefore has exact per-query
membership without learning the user's predicate.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse

MAGIC = b"CUVSBMAP"
VERSION = 1
WORD_BITS = 32
HEADER = struct.Struct("<8sIIQQQ")
INVALID_U32 = np.iinfo(np.uint32).max
INVALID_PADDING_START = int(INVALID_U32) - 999


@dataclass(frozen=True)
class Matrix:
    path: Path
    dtype: np.dtype
    allow_trailing_float_distances: bool = False

    def __post_init__(self) -> None:
        with self.path.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated matrix header: {self.path}")
        rows, cols = struct.unpack("<II", header)
        expected = 8 + rows * cols * self.dtype.itemsize
        distances_expected = expected + rows * cols * np.dtype("<f4").itemsize
        actual = self.path.stat().st_size
        if actual != expected and not (
            self.allow_trailing_float_distances
            and actual == distances_expected
        ):
            raise ValueError(f"matrix size mismatch: {self.path}")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(
            self,
            "values",
            np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                offset=8,
                shape=(rows, cols),
            ),
        )


@dataclass(frozen=True)
class Spmat:
    path: Path

    def __post_init__(self) -> None:
        with self.path.open("rb") as stream:
            header = stream.read(24)
        if len(header) != 24:
            raise ValueError(f"truncated sparse-matrix header: {self.path}")
        rows, cols, nnz = struct.unpack("<qqq", header)
        expected = 24 + 8 * (rows + 1) + 8 * nnz
        if (
            rows <= 0
            or cols <= 0
            or nnz < 0
            or self.path.stat().st_size != expected
        ):
            raise ValueError(f"invalid sparse matrix: {self.path}")
        offsets = np.memmap(
            self.path, dtype="<i8", mode="r", offset=24, shape=(rows + 1,)
        )
        columns = np.memmap(
            self.path,
            dtype="<i4",
            mode="r",
            offset=24 + 8 * (rows + 1),
            shape=(nnz,),
        )
        if (
            offsets[0] != 0
            or offsets[-1] != nnz
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError(f"invalid sparse offsets: {self.path}")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "nnz", nnz)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "columns", columns)

    def row(self, row: int) -> np.ndarray:
        return np.asarray(
            self.columns[self.offsets[row] : self.offsets[row + 1]]
        )


class BitmapWriter:
    def __init__(self, path: Path, rows: int, cols: int):
        if rows <= 0 or cols <= 0:
            raise ValueError("bitmap dimensions must be positive")
        self.path = path
        self.rows = rows
        self.cols = cols
        self.word_count = (rows * cols + WORD_BITS - 1) // WORD_BITS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(
                HEADER.pack(
                    MAGIC, VERSION, WORD_BITS, rows, cols, self.word_count
                )
            )
            stream.truncate(HEADER.size + 4 * self.word_count)
        self.words = np.memmap(
            path,
            dtype="<u4",
            mode="r+",
            offset=HEADER.size,
            shape=(self.word_count,),
        )
        self.words[:] = 0

    def write_ids(self, row: int, ids: np.ndarray | None) -> int:
        if ids is None:
            ids = np.arange(self.cols, dtype=np.uint64)
        else:
            ids = np.asarray(ids, dtype=np.uint64)
        if ids.size and (ids[-1] >= self.cols or np.any(ids[1:] <= ids[:-1])):
            raise ValueError(
                f"candidate IDs for bitmap row {row} are not sorted unique in range"
            )
        flat = np.uint64(row * self.cols) + ids
        words = flat >> 5
        masks = np.left_shift(np.uint32(1), (flat & 31).astype(np.uint32))
        np.bitwise_or.at(self.words, words, masks)
        return int(ids.size)

    def test(self, row: int, col: int) -> bool:
        bit = row * self.cols + col
        return bool(
            self.words[bit >> 5] & (np.uint32(1) << np.uint32(bit & 31))
        )

    def close(self) -> None:
        used = (self.rows * self.cols) % WORD_BITS
        if used:
            self.words[-1] &= (np.uint32(1) << np.uint32(used)) - np.uint32(1)
        self.words.flush()
        del self.words


def write_matrix(path: Path, rows: np.ndarray) -> None:
    rows = np.asarray(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", rows.shape[0], rows.shape[1]))
        rows.tofile(stream)


def shards(total: int, shard_size: int) -> Iterator[tuple[int, int]]:
    if shard_size <= 0:
        shard_size = total
    for first in range(0, total, shard_size):
        yield first, min(total, first + shard_size)


def sparse_candidate_factory(base: Spmat):
    data = np.ones(base.nnz, dtype=np.bool_)
    # scipy may sort/deduplicate CSR storage in place.  The source arrays are read-only memmaps,
    # and YFCC is not already in scipy's canonical ordering, so give scipy writable index arrays.
    csr = scipy.sparse.csr_matrix(
        (
            data,
            np.asarray(base.columns).copy(),
            np.asarray(base.offsets).copy(),
        ),
        shape=(base.rows, base.cols),
        dtype=np.bool_,
    )
    csr.sum_duplicates()
    csc = csr.tocsc()
    csc.sort_indices()

    def candidates(tags: np.ndarray) -> np.ndarray | None:
        tags = np.unique(np.asarray(tags, dtype=np.int64))
        if tags.size == 0:
            return None  # contains-all over an empty predicate is true for every candidate
        if tags[0] < 0 or tags[-1] >= base.cols:
            raise ValueError("query sparse metadata contains an out-of-range column")
        result = np.asarray(
            csc.indices[csc.indptr[tags[0]] : csc.indptr[tags[0] + 1]]
        )
        for tag in tags[1:]:
            posting = np.asarray(
                csc.indices[csc.indptr[tag] : csc.indptr[tag + 1]]
            )
            result = np.intersect1d(result, posting, assume_unique=True)
            if result.size == 0:
                break
        return result

    return candidates


def validate_groundtruth(
    writer: BitmapWriter, gt: np.ndarray, first: int
) -> None:
    for local, row in enumerate(gt):
        for node in row:
            node = int(node)
            if node >= INVALID_PADDING_START:
                continue
            if node >= writer.cols or not writer.test(local, node):
                raise ValueError(
                    f"ground-truth predicate violation: query={first + local}, node={node}"
                )


def materialize(
    args: argparse.Namespace,
    base_rows: int,
    query_rows: int,
    candidate_for_query,
) -> None:
    vectors = Matrix(args.query_vectors, np.dtype(args.vector_dtype))
    # BIG-ann YFCC appends a float32 distance matrix after its uint32 neighbor-ID matrix. Other
    # prepared workloads contain IDs only. Both layouts share the same header and ID prefix.
    gt = Matrix(
        args.groundtruth,
        np.dtype("<u4"),
        allow_trailing_float_distances=True,
    )
    total = min(query_rows, vectors.rows, gt.rows)
    if args.limit:
        total = min(total, args.limit)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "bitmap_schema": "CUVSBMAP/v1/u32/row-major",
        "base_rows": base_rows,
        "query_rows": total,
        "shards": [],
    }
    for first, end in shards(total, args.shard_size):
        count = end - first
        target = args.output / f"shard_{first:05d}_{end:05d}"
        writer = BitmapWriter(target / "filter.bitmap", count, base_rows)
        passing_counts: list[int] = []
        for local, query in enumerate(range(first, end)):
            passing_counts.append(
                writer.write_ids(local, candidate_for_query(query))
            )
        validate_groundtruth(writer, np.asarray(gt.values[first:end]), first)
        writer.close()
        write_matrix(
            target / "query.bin", np.asarray(vectors.values[first:end])
        )
        write_matrix(
            target / "groundtruth.ibin", np.asarray(gt.values[first:end])
        )
        row = {
            "first_query": first,
            "query_count": count,
            "directory": str(target),
            "bitmap": str(target / "filter.bitmap"),
            "min_passing": min(passing_counts),
            "max_passing": max(passing_counts),
            "mean_selectivity": float(np.mean(passing_counts) / base_rows),
            "empty_queries": int(
                np.count_nonzero(np.asarray(passing_counts) == 0)
            ),
        }
        manifest["shards"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def sparse_command(args: argparse.Namespace) -> None:
    base = Spmat(args.base_metadata)
    queries = Spmat(args.query_metadata)
    if base.cols != queries.cols:
        raise ValueError("base/query sparse metadata column mismatch")
    find_candidates = sparse_candidate_factory(base)
    materialize(
        args,
        base.rows,
        queries.rows,
        lambda query: find_candidates(queries.row(query)),
    )


def read_range(path: Path, width: int) -> np.ndarray:
    with path.open("rb") as stream:
        raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError(f"truncated range metadata: {path}")
    (rows,) = struct.unpack("<I", raw)
    expected = 4 + rows * width * 4
    if rows <= 0 or path.stat().st_size != expected:
        raise ValueError(f"range metadata size mismatch: {path}")
    return np.memmap(
        path, dtype="<i4", mode="r", offset=4, shape=(rows, width)
    )


def range_command(args: argparse.Namespace) -> None:
    base = read_range(args.base_metadata, 1)[:, 0]
    queries = read_range(args.query_metadata, 2)

    def find_candidates(query: int) -> np.ndarray:
        start, end = queries[query]
        return np.flatnonzero((base >= start) & (base <= end))

    materialize(args, base.shape[0], queries.shape[0], find_candidates)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--query-metadata", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument(
        "--vector-dtype",
        choices=("uint8", "int8", "float16", "float32"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sparse = commands.add_parser(
        "sparse", help="contains-all sparse-tag predicate"
    )
    add_common(sparse)
    sparse.set_defaults(func=sparse_command)
    ranges = commands.add_parser(
        "range", help="inclusive scalar range predicate"
    )
    add_common(ranges)
    ranges.set_defaults(func=range_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
