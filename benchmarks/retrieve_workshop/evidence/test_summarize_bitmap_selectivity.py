#!/usr/bin/env python3
"""Unit tests for exact per-query bitmap-population accounting."""

from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from summarize_bitmap_selectivity import BITMAP_HEADER, count_bitmap_rows


def write_bitmap(path: Path, rows: list[list[int]], columns: int) -> None:
    total_bits = len(rows) * columns
    words = np.zeros((total_bits + 31) // 32, dtype="<u4")
    for row, columns_set in enumerate(rows):
        for column in columns_set:
            bit = row * columns + column
            words[bit // 32] |= np.uint32(1) << np.uint32(bit % 32)
    with path.open("wb") as stream:
        stream.write(
            BITMAP_HEADER.pack(
                b"CUVSBMAP", 1, 32, len(rows), columns, words.size
            )
        )
        words.tofile(stream)


class BitmapSelectivityTest(unittest.TestCase):
    def test_unaligned_contiguous_rows_do_not_leak_boundary_bits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unaligned.bitmap"
            write_bitmap(path, [[0, 31, 32], [0, 1, 31, 32], [7, 25]], 33)
            counts, digest = count_bitmap_rows(path, 3, 33)
            self.assertEqual(counts.tolist(), [3, 4, 2])
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_nonzero_global_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "padding.bitmap"
            write_bitmap(path, [[0], [32]], 33)
            with path.open("r+b") as stream:
                stream.seek(-4, 2)
                word = struct.unpack("<I", stream.read(4))[0]
                stream.seek(-4, 2)
                stream.write(struct.pack("<I", word | (1 << 31)))
            with self.assertRaisesRegex(ValueError, "padding bits"):
                count_bitmap_rows(path, 2, 33)


if __name__ == "__main__":
    unittest.main()
