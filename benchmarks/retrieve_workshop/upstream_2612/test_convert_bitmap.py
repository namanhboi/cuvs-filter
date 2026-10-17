# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convert_bitmap import (
    HEADER,
    AllowlistRows,
    Bitmap,
    convert,
    sha256,
    verify,
)


def write_bitmap(path: Path, values: np.ndarray) -> None:
    rows, cols = values.shape
    words = (rows * cols + 31) // 32
    packed = np.packbits(
        values.ravel().astype("u1"), bitorder="little"
    ).tobytes()
    path.write_bytes(
        HEADER.pack(b"CUVSBMAP", 1, 32, rows, cols, words)
        + packed.ljust(words * 4, b"\0")
    )


class ConversionTests(unittest.TestCase):
    def test_membership_and_provenance_across_word_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cols in (1, 7, 31, 32, 33, 769):
                with self.subTest(cols=cols):
                    values = np.random.default_rng(cols).integers(
                        0, 2, size=(7, cols), dtype="u1"
                    )
                    values[0] = 0
                    values[1] = 1
                    source, target = (
                        root / f"{cols}.bitmap",
                        root / f"{cols}.allowids",
                    )
                    write_bitmap(source, values)
                    original = source.read_bytes()
                    metadata = convert(source, target)
                    archive = AllowlistRows(target)
                    for query in range(7):
                        expected = [
                            col for col in range(cols) if values[query, col]
                        ]
                        self.assertEqual(archive.row(query).tolist(), expected)
                    self.assertEqual(source.read_bytes(), original)
                    self.assertEqual(metadata["source_sha256"], sha256(source))
                    self.assertEqual(metadata["output_sha256"], sha256(target))
                    saved = json.loads(
                        target.with_name(target.name + ".json").read_text()
                    )
                    self.assertEqual(saved, metadata)

    def test_empty_batch_of_allowlists(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (
                Path(directory) / "zero.bitmap",
                Path(directory) / "zero.allowids",
            )
            write_bitmap(source, np.zeros((3, 33), dtype="u1"))
            convert(source, target)
            archive = AllowlistRows(target)
            self.assertEqual(archive.count, 0)
            self.assertEqual(archive.row(2).tolist(), [])

    def test_conversion_never_replaces_inputs_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "input.bitmap", root / "output.allowids"
            write_bitmap(source, np.ones((2, 33), dtype="u1"))
            with self.assertRaises(FileExistsError):
                convert(source, source)
            target.write_bytes(b"existing result")
            with self.assertRaises(FileExistsError):
                convert(source, target)
            self.assertEqual(target.read_bytes(), b"existing result")

    def test_corrupted_offsets_and_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "input.bitmap", root / "output.allowids"
            write_bitmap(source, np.ones((2, 33), dtype="u1"))
            convert(source, target)
            valid = target.read_bytes()
            damaged = bytearray(valid)
            struct.pack_into("<Q", damaged, HEADER.size + 8, 70)
            target.write_bytes(damaged)
            with self.assertRaises(ValueError):
                AllowlistRows(target)
            damaged = bytearray(valid)
            struct.pack_into("<I", damaged, HEADER.size + 3 * 8, 1000)
            target.write_bytes(damaged)
            with self.assertRaises(ValueError):
                AllowlistRows(target).row(0)

    def test_verification_detects_a_different_bitmap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "input.bitmap", root / "output.allowids"
            values = np.ones((2, 33), dtype="u1")
            write_bitmap(source, values)
            convert(source, target)
            values[1, 17] = 0
            write_bitmap(source, values)
            with self.assertRaisesRegex(ValueError, "query 1"):
                verify(Bitmap(source), target)

    def test_malformed_bitmap_does_not_publish_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "input.bitmap", root / "output.allowids"
            source.write_bytes(
                HEADER.pack(b"CUVSBMAP", 1, 32, 1, 1, 1) + b"\xff\0\0\0"
            )
            with self.assertRaisesRegex(ValueError, "padding"):
                convert(source, target)
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".allowlist-*")))


if __name__ == "__main__":
    unittest.main()
