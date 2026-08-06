#!/usr/bin/env python3
"""Prepare aligned YFCC filtered-search workloads without copying the 10M base artifacts."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_matrix(
    path: Path, dtype: np.dtype, *, allow_trailing_float_distances: bool = False
) -> np.memmap:
    with path.open("rb") as stream:
        rows, cols = struct.unpack("<II", stream.read(8))
    expected = 8 + rows * cols * np.dtype(dtype).itemsize
    distances_expected = expected + rows * cols * np.dtype("<f4").itemsize
    if path.stat().st_size != expected and not (
        allow_trailing_float_distances and path.stat().st_size == distances_expected
    ):
        raise ValueError(f"matrix size mismatch: {path}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, cols))


class Spmat:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            self.rows, self.cols, self.nnz = struct.unpack("<qqq", stream.read(24))
        expected = 24 + 8 * (self.rows + 1) + 4 * self.nnz + 4 * self.nnz
        if self.rows <= 0 or self.cols <= 0 or self.nnz < 0 or path.stat().st_size != expected:
            raise ValueError(f"invalid spmat: {path}")
        self.offsets = np.memmap(
            path, dtype="<i8", mode="r", offset=24, shape=(self.rows + 1,)
        )
        self.columns = np.memmap(
            path,
            dtype="<i4",
            mode="r",
            offset=24 + 8 * (self.rows + 1),
            shape=(self.nnz,),
        )
        self.values = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=24 + 8 * (self.rows + 1) + 4 * self.nnz,
            shape=(self.nnz,),
        )
        if self.offsets[0] != 0 or self.offsets[-1] != self.nnz:
            raise ValueError(f"invalid spmat offsets: {path}")


def write_matrix(path: Path, rows: np.ndarray) -> None:
    rows = np.asarray(rows)
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", rows.shape[0], rows.shape[1]))
        rows.tofile(stream)


def write_spmat_subset(path: Path, source: Spmat, query_ids: list[int]) -> None:
    lengths = np.asarray(
        [source.offsets[q + 1] - source.offsets[q] for q in query_ids], dtype="<i8"
    )
    offsets = np.empty(len(query_ids) + 1, dtype="<i8")
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    columns = np.empty(int(offsets[-1]), dtype="<i4")
    values = np.empty(int(offsets[-1]), dtype="<f4")
    output = 0
    for query_id, length in zip(query_ids, lengths, strict=True):
        begin = int(source.offsets[query_id])
        end = int(source.offsets[query_id + 1])
        columns[output : output + length] = source.columns[begin:end]
        values[output : output + length] = source.values[begin:end]
        output += int(length)
    with path.open("wb") as stream:
        stream.write(struct.pack("<qqq", len(query_ids), source.cols, len(columns)))
        offsets.tofile(stream)
        columns.tofile(stream)
        values.tofile(stream)


def write_subset(
    output_dir: Path,
    name: str,
    query_ids: list[int],
    queries: np.memmap,
    metadata: Spmat,
    groundtruth: np.memmap,
    annotations: dict[int, dict],
) -> dict:
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    write_matrix(target / "query.u8bin", queries[query_ids])
    write_matrix(target / "groundtruth.ibin", groundtruth[query_ids])
    write_spmat_subset(target / "query.metadata.spmat", metadata, query_ids)
    manifest = {
        "schema_version": 1,
        "source_query_rows": int(queries.shape[0]),
        "query_ids": query_ids,
        "groups": [annotations.get(query_id, {}) for query_id in query_ids],
    }
    (target / "query_ids.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"name": name, "queries": len(query_ids)}


def symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise FileExistsError(f"refusing to replace {link}")
    link.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--delta-d", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    links = {
        "base.10M.u8bin": args.source / "base.10M.u8bin",
        "base.metadata.10M.spmat": args.source / "base.metadata.10M.spmat",
        "query.public.100K.u8bin": args.source / "query.public.100K.u8bin",
        "query.metadata.public.100K.spmat": args.source / "query.metadata.public.100K.spmat",
        "GT.public.ibin": args.source / "GT.public.ibin",
        "cagra_g32_ig64.index": args.index,
        "cagra_g32_ig64.index.delta_d": args.delta_d,
    }
    for name, target in links.items():
        symlink(target, args.output / name)

    queries = read_matrix(links["query.public.100K.u8bin"], np.dtype("u1"))
    # BigANN distributes YFCC ground truth as IDs followed by an equally shaped float-distance
    # matrix.  cuVS-bench needs only the ID matrix, so generated subsets intentionally omit the
    # trailing distances.
    groundtruth = read_matrix(
        links["GT.public.ibin"],
        np.dtype("<u4"),
        allow_trailing_float_distances=True,
    )
    metadata = Spmat(links["query.metadata.public.100K.spmat"])
    if queries.shape[0] != groundtruth.shape[0] or queries.shape[0] != metadata.rows:
        raise ValueError("public query/vector/metadata/ground-truth row mismatch")

    selection = json.loads(args.selection_json.read_text())
    selected = selection["queries"]
    correctness_ids = [int(row["query_id"]) for row in selected]
    if len(correctness_ids) != 1000 or len(set(correctness_ids)) != 1000:
        raise ValueError("selection JSON must contain 1,000 unique queries")
    annotations = {
        int(row["query_id"]): {
            "arity": int(row["arity"]),
            "selectivity_decile": int(row["selectivity_decile"]),
        }
        for row in selected
    }

    workloads = [
        write_subset(
            args.output / "workloads",
            "correctness_1000",
            correctness_ids,
            queries,
            metadata,
            groundtruth,
            annotations,
        ),
        write_subset(
            args.output / "workloads",
            "throughput_10000",
            list(range(10_000)),
            queries,
            metadata,
            groundtruth,
            {},
        ),
    ]

    for arity in (1, 2):
        for decile in range(10):
            group = [
                query_id
                for query_id in correctness_ids
                if annotations[query_id]["arity"] == arity
                and annotations[query_id]["selectivity_decile"] == decile
            ][:10]
            if len(group) != 10:
                raise ValueError(f"missing latency group arity={arity}, decile={decile}")
            workloads.append(
                write_subset(
                    args.output / "workloads",
                    f"latency_a{arity}_d{decile + 1}",
                    group,
                    queries,
                    metadata,
                    groundtruth,
                    annotations,
                )
            )

    manifest = {
        "schema_version": 1,
        "predicate": "query_tags_subset_of_candidate_tags",
        "base_rows": 10_000_000,
        "workloads": workloads,
        "stores_selectivity": False,
    }
    (args.output / "workloads" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
