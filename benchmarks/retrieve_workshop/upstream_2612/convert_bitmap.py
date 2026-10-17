#!/usr/bin/env python3
"""Lossless, row-bounded interchange from CUVSBMAP to sorted Roaring input IDs.

This file stores sorted IDs, not portable Roaring bytes. cuVS constructs the
resident Roaring owners from these IDs with pre_sorted=True outside search timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8sIIQQQ")
BITMAP_MAGIC = b"CUVSBMAP"
ALLOWLIST_MAGIC = b"CUVSRIDS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Bitmap:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated bitmap header")
        magic, version, word_bits, self.rows, self.cols, words = HEADER.unpack(
            header
        )
        if magic != BITMAP_MAGIC or version != 1 or word_bits != 32:
            raise ValueError("unsupported bitmap format")
        if not 0 < self.rows <= 2**32 - 1 or not 0 < self.cols <= 2**32 - 1:
            raise ValueError("invalid bitmap dimensions")
        bits = self.rows * self.cols
        if (
            words != (bits + 31) // 32
            or self.path.stat().st_size != HEADER.size + words * 4
        ):
            raise ValueError("bitmap size does not match its header")
        self.payload = np.memmap(
            self.path, mode="r", dtype="u1", offset=HEADER.size
        )
        if bits % 32:
            last = int.from_bytes(self.payload[-4:].tobytes(), "little")
            if last >> (bits % 32):
                raise ValueError("nonzero bitmap padding")

    def row(self, query: int) -> np.ndarray:
        if not 0 <= query < self.rows:
            raise IndexError(query)
        start = query * self.cols
        end = start + self.cols
        bits = np.unpackbits(
            self.payload[start // 8 : (end + 7) // 8], bitorder="little"
        )
        return bits[start % 8 : start % 8 + self.cols]


class AllowlistRows:
    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated allowlist header")
        magic, version, id_bits, self.rows, self.cols, self.count = (
            HEADER.unpack(header)
        )
        if magic != ALLOWLIST_MAGIC or version != 1 or id_bits != 32:
            raise ValueError("unsupported allowlist format")
        if not 0 < self.rows <= 2**32 - 1 or not 0 < self.cols <= 2**32 - 1:
            raise ValueError("invalid allowlist dimensions")
        payload_start = HEADER.size + (self.rows + 1) * 8
        if self.path.stat().st_size != payload_start + self.count * 4:
            raise ValueError("allowlist size does not match its header")
        self.offsets = np.memmap(
            self.path,
            mode="r",
            dtype="<u8",
            offset=HEADER.size,
            shape=(self.rows + 1,),
        )
        if self.offsets[0] != 0 or self.offsets[-1] != self.count:
            raise ValueError("invalid terminal allowlist offsets")
        if np.any(self.offsets[1:] < self.offsets[:-1]):
            raise ValueError("allowlist offsets must be nondecreasing")
        if np.any(self.offsets[1:] - self.offsets[:-1] > self.cols):
            raise ValueError("allowlist row exceeds the dataset domain")
        self.ids = (
            np.memmap(
                self.path,
                mode="r",
                dtype="<u4",
                offset=payload_start,
                shape=(self.count,),
            )
            if self.count
            else np.empty(0, dtype="<u4")
        )

    def row(self, query: int) -> np.ndarray:
        if not 0 <= query < self.rows:
            raise IndexError(query)
        ids = self.ids[int(self.offsets[query]) : int(self.offsets[query + 1])]
        if ids.size and (ids[-1] >= self.cols or np.any(ids[1:] <= ids[:-1])):
            raise ValueError(
                "IDs must be unique, ascending, and within the dataset domain"
            )
        return ids


def verify(bitmap: Bitmap, archive_path: Path) -> None:
    archive = AllowlistRows(archive_path)
    if (bitmap.rows, bitmap.cols) != (archive.rows, archive.cols):
        raise ValueError("bitmap and allowlist dimensions differ")
    for query in range(bitmap.rows):
        # Reconstruct every row independently, including its rejected IDs.
        reconstructed = np.zeros(bitmap.cols, dtype="u1")
        reconstructed[archive.row(query)] = 1
        if not np.array_equal(reconstructed, bitmap.row(query)):
            raise ValueError(f"membership mismatch for query {query}")


def convert(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata_path = output.with_name(output.name + ".json")
    if source == output or output.exists() or metadata_path.exists():
        raise FileExistsError("conversion outputs must be new paths")
    bitmap = Bitmap(source)
    before = source.stat()
    source_hash = sha256(source)
    offsets = np.zeros(bitmap.rows + 1, dtype="<u8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=".allowlist-", dir=output.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "w+b") as stream:
            stream.write(
                HEADER.pack(
                    ALLOWLIST_MAGIC, 1, 32, bitmap.rows, bitmap.cols, 0
                )
            )
            stream.write(offsets.tobytes())
            for query in range(bitmap.rows):
                ids = np.flatnonzero(bitmap.row(query)).astype("<u4")
                stream.write(ids.tobytes())
                offsets[query + 1] = int(offsets[query]) + ids.size
            stream.seek(0)
            stream.write(
                HEADER.pack(
                    ALLOWLIST_MAGIC,
                    1,
                    32,
                    bitmap.rows,
                    bitmap.cols,
                    int(offsets[-1]),
                )
            )
            stream.write(offsets.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        verify(bitmap, temporary_path)
        after = source.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("source bitmap changed during conversion")
        metadata = {
            "format": "CUVSRIDS/v1",
            "source": str(source),
            "source_sha256": source_hash,
            "output": str(output),
            "output_sha256": sha256(temporary_path),
            "rows": bitmap.rows,
            "dataset_rows": bitmap.cols,
            "cardinalities": np.diff(offsets).tolist(),
            "verified_membership": True,
            "construction": "cuvs::core::roaring_allowlist::from_ids(pre_sorted=true)",
        }
        # Hard-link publication refuses to replace an existing result, including a race.
        os.link(temporary_path, output)
        try:
            with metadata_path.open("x") as stream:
                json.dump(metadata, stream, indent=2)
                stream.write("\n")
        except Exception:
            output.unlink()
            raise
        return metadata
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = convert(args.source, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("output", "rows", "verified_membership")
            }
        )
    )


if __name__ == "__main__":
    main()
