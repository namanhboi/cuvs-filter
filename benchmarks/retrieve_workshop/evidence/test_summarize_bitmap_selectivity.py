#!/usr/bin/env python3
"""Unit tests for exact per-query bitmap-population accounting."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from summarize_bitmap_selectivity import (
    BITMAP_HEADER,
    MATRIX_HEADER,
    count_bitmap_rows,
    summarize,
)


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


def write_ground_truth(path: Path, rows: list[list[int]]) -> None:
    values = np.asarray(rows, dtype="<u4")
    with path.open("wb") as stream:
        stream.write(MATRIX_HEADER.pack(*values.shape))
        values.tofile(stream)


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

    def test_summary_resolves_relative_sources_and_counts_valid_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bitmap = root / "filter.bitmap"
            ground_truth = root / "groundtruth.ibin"
            manifest = root / "manifest.json"
            write_bitmap(bitmap, [[0, 2], [1, 3, 4]], 33)
            write_ground_truth(
                ground_truth,
                [list(range(10)), [1] * 9 + [np.iinfo(np.uint32).max]],
            )
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "bitmap_schema": "CUVSBMAP/v1/u32/row-major",
                        "base_rows": 33,
                        "query_rows": 2,
                        "shards": [
                            {
                                "directory": ".",
                                "first_query": 0,
                                "query_count": 2,
                                "bitmap": "filter.bitmap",
                                "groundtruth": "groundtruth.ibin",
                                "min_passing": 2,
                                "max_passing": 3,
                                "mean_selectivity": 2.5 / 33,
                            }
                        ],
                    }
                )
            )
            result = summarize("fixture", manifest)
            self.assertEqual(result["valid_ground_truth"]["slots"], 19)
            self.assertEqual(result["valid_ground_truth"]["min_per_query"], 9)
            self.assertEqual(result["valid_ground_truth"]["max_per_query"], 10)
            self.assertEqual(result["valid_ground_truth"]["underfilled_queries"], 1)
            self.assertEqual(
                result["valid_ground_truth"]["per_query_count_histogram"],
                {"9": 1, "10": 1},
            )
            self.assertEqual(result["source_bitmaps"][0]["path"], str(bitmap))
            self.assertEqual(
                result["source_ground_truth"][0]["path"], str(ground_truth)
            )

    def test_summary_requires_manifest_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(
                json.dumps({"bitmap_schema": "CUVSBMAP/v1/u32/row-major"})
            )
            with self.assertRaisesRegex(ValueError, "unsupported bitmap schema"):
                summarize("fixture", manifest)


if __name__ == "__main__":
    unittest.main()
