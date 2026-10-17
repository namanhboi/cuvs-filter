# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from convert_indexes import convert


def scalar(stream, value, dtype):
    np.save(stream, np.asarray(value, dtype=dtype))


def write_cagra(path, version=5, embedded=False):
    graph = np.asarray([[1, 2], [0, 2], [1, 3], [0, 2]], dtype="u4")
    mapping = np.asarray([3, 0, 2, 1], dtype="u4")
    with path.open("wb") as stream:
        stream.write(b"<f4\0")
        for value, dtype in [
            (version, "i4"),
            (4, "u4"),
            (0, "u4"),
            (2, "u4"),
            (0, "i4"),
        ]:
            scalar(stream, value, dtype)
        np.save(stream, graph)
        scalar(stream, 3 if embedded else 2, "u4")
        np.save(stream, mapping)
    return graph, mapping


def write_flat(path, dtype="f4", norms=True):
    sizes = (35, 0, 7)
    dim, rows = 5, sum(sizes)
    centers = np.arange(3 * dim, dtype="f4").reshape(3, dim)
    payloads = []
    with path.open("wb") as stream:
        stream.write(np.dtype(dtype).str.encode().ljust(4, b"\0"))
        fields = [
            (4, "i4"),
            (rows, "i8"),
            (dim, "u4"),
            (3, "u4"),
            (0, "i4"),
            (False, "u1"),
            (True, "u1"),
        ]
        for value, field_dtype in fields:
            scalar(stream, value, field_dtype)
        np.save(stream, centers)
        scalar(stream, norms, "u1")
        if norms:
            np.save(stream, np.asarray([1.0, 2.0, 3.0], dtype="f4"))
        np.save(stream, np.asarray(sizes, dtype="u4"))
        first_id = 0
        for logical in sizes:
            padded = (logical + 31) // 32 * 32
            scalar(stream, padded, "u4")
            if padded:
                vectors = np.arange(padded * dim, dtype=dtype).reshape(
                    padded, dim
                )
                ids = np.full(padded, np.iinfo("i8").max, dtype="i8")
                ids[:logical] = np.arange(first_id, first_id + logical)[::-1]
                np.save(stream, vectors)
                np.save(stream, ids)
                payloads.append((vectors, ids[:logical].copy()))
            first_id += logical
    return centers, payloads, sizes


class IndexConversionTests(unittest.TestCase):
    def test_cagra_keeps_graph_and_original_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "old.index", root / "new.index"
            graph, mapping = write_cagra(source)
            before = source.read_bytes()
            result = convert(source, target, "cagra")
            self.assertTrue(result["verified_components"])
            self.assertEqual(source.read_bytes(), before)
            with target.open("rb") as stream:
                self.assertEqual(stream.read(4), b"<f4\0")
                self.assertEqual(
                    [np.load(stream).item() for _ in range(6)],
                    [6, 0, 4, 0, 2, 0],
                )
                np.testing.assert_array_equal(np.load(stream), graph)
                self.assertEqual(np.load(stream).item(), 2)
                np.testing.assert_array_equal(np.load(stream), mapping)
                self.assertEqual(stream.read(), b"")

    def test_flat_keeps_centroids_vectors_list_assignments_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dtype, norms in (("f4", True), ("u1", False)):
                with self.subTest(dtype=dtype, norms=norms):
                    source, target = (
                        root / f"old-{dtype}.index",
                        root / f"new-{dtype}.index",
                    )
                    centers, payloads, sizes = write_flat(source, dtype, norms)
                    original = source.read_bytes()
                    result = convert(source, target, "ivf_flat")
                    self.assertTrue(result["verified_components"])
                    self.assertEqual(source.read_bytes(), original)
                    with target.open("rb") as stream:
                        stream.read(4)
                        fields = [np.load(stream).item() for _ in range(7)]
                        self.assertEqual(fields, [5, 42, 5, 3, 0, False, True])
                        np.testing.assert_array_equal(np.load(stream), centers)
                        self.assertEqual(np.load(stream).item(), norms)
                        if norms:
                            np.testing.assert_array_equal(
                                np.load(stream), [1, 2, 3]
                            )
                        np.testing.assert_array_equal(np.load(stream), sizes)
                        nonempty = iter(payloads)
                        for logical in sizes:
                            self.assertEqual(np.load(stream).item(), logical)
                            if logical:
                                vectors, ids = next(nonempty)
                                np.testing.assert_array_equal(
                                    np.load(stream), vectors
                                )
                                np.testing.assert_array_equal(
                                    np.load(stream), ids
                                )
                        self.assertEqual(stream.read(), b"")

    def test_unsupported_version_or_embedded_dataset_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, embedded in ((99, False), (5, True)):
                source, target = root / "old.index", root / "new.index"
                write_cagra(source, version=version, embedded=embedded)
                with self.assertRaises(ValueError):
                    convert(source, target, "cagra")
                self.assertFalse(target.exists())
                self.assertFalse(list(root.glob(".index-conversion-*")))

    def test_truncated_index_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "old.index", root / "new.index"
            write_flat(source)
            source.write_bytes(source.read_bytes()[:-9])
            with self.assertRaises(ValueError):
                convert(source, target, "ivf_flat")
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".index-conversion-*")))

    def test_existing_result_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "old.index", root / "new.index"
            write_cagra(source)
            target.write_bytes(b"keep this result")
            with self.assertRaises(FileExistsError):
                convert(source, target, "cagra")
            self.assertEqual(target.read_bytes(), b"keep this result")


if __name__ == "__main__":
    unittest.main()
