#!/usr/bin/env python3
"""CPU-side tests for the ArXiv selectivity follow-up."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import workflow as study


class SelectivityFollowupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_rows = study.BASE_ROWS
        self.old_dim = study.DIM
        study.BASE_ROWS = 8192
        study.DIM = 4

    def tearDown(self) -> None:
        study.BASE_ROWS = self.old_rows
        study.DIM = self.old_dim

    def make_data_fixture(self, root: Path) -> Path:
        data_root = root / "data"
        base = data_root / "arxiv-for-fanns-large"
        base.mkdir(parents=True)
        study.write_matrix(
            base / "base.fbin",
            np.zeros((study.BASE_ROWS, study.DIM), dtype=np.float32),
        )
        (base / "cagra_g64_ig128.index").write_bytes(b"fixture index")
        for workload in study.WORKLOADS:
            source = (
                data_root
                / "navix_bitmap"
                / "arxiv-large"
                / workload
                / "throughput_10000"
            )
            source.mkdir(parents=True)
            shards = []
            first = 0
            for count in (2048, 2048, 2048, 2048, 1808):
                directory = source / f"shard_{first:05d}_{first + count:05d}"
                directory.mkdir()
                study.write_matrix(
                    directory / "query.bin",
                    np.zeros((count, study.DIM), dtype=np.float32),
                )
                study.write_matrix(
                    directory / "groundtruth.ibin",
                    np.zeros((count, study.K), dtype=np.uint32),
                )
                bitmap = directory / "filter.bitmap"
                words = count * (study.BASE_ROWS // 32)
                with bitmap.open("wb") as output:
                    output.write(
                        study.BITMAP_HEADER.pack(
                            b"CUVSBMAP", 1, 32, count, study.BASE_ROWS, words
                        )
                    )
                    output.truncate(study.BITMAP_HEADER.size + 4 * words)
                shards.append(
                    {
                        "first_query": first,
                        "query_count": count,
                        "directory": str(directory),
                        "bitmap": str(bitmap),
                    }
                )
                first += count
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "SPCL/arxiv-for-fanns-large",
                        "predicate": workload,
                        "base_rows": study.BASE_ROWS,
                        "query_rows": 10_000,
                        "shards": shards,
                    }
                )
            )
        return data_root

    def test_data_preflight_checks_all_paths_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = self.make_data_fixture(root)
            summaries, errors = study.inspect_required_data(data_root)
            self.assertEqual(len(summaries), 3)
            self.assertEqual(errors, [])

            unused_root = root / "must_not_be_created"
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON": sys.executable,
                    "RETRIEVE_DATA_ROOT": str(data_root),
                    "RETRIEVE_SELECTIVITY_RUN_ROOT": str(unused_root),
                }
            )
            result = subprocess.run(
                [str(study.HERE / "run.sh"), "check-data"],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            # The subprocess uses production dimensions rather than this test's
            # reduced constants, but must still fail before creating a run root.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("data preflight failed", result.stderr)
            self.assertFalse(unused_root.exists())

            shard = (
                data_root
                / "navix_bitmap/arxiv-large/em/throughput_10000"
                / "shard_00000_02048"
            )
            gt = shard / "groundtruth.ibin"
            gt.unlink()
            self.assertTrue(
                any("groundtruth.ibin" in error for error in
                    study.inspect_required_data(data_root)[1])
            )
            study.write_matrix(gt, np.zeros((2048, study.K), dtype=np.uint32))

            query = shard / "query.bin"
            with query.open("r+b") as output:
                output.truncate(8)
            self.assertTrue(
                any("query.bin" in error for error in
                    study.inspect_required_data(data_root)[1])
            )
            study.write_matrix(
                query, np.zeros((2048, study.DIM), dtype=np.float32)
            )

            bitmap = shard / "filter.bitmap"
            with bitmap.open("r+b") as output:
                output.write(b"BAD")
            self.assertTrue(
                any(
                    "bitmap header or size mismatch" in error
                    for error in study.inspect_required_data(data_root)[1]
                )
            )
            with bitmap.open("r+b") as output:
                output.write(
                    study.BITMAP_HEADER.pack(
                        b"CUVSBMAP", 1, 32, 2048, study.BASE_ROWS,
                        2048 * (study.BASE_ROWS // 32),
                    )
                )

            source = data_root / "navix_bitmap/arxiv-large/em/throughput_10000"
            manifest = json.loads((source / "manifest.json").read_text())
            manifest["shards"][0]["bitmap"] = str(root / "outside.bitmap")
            (source / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(
                any("outside" in error for error in
                    study.inspect_required_data(data_root)[1])
            )

    def test_random_bitmap_has_exact_cardinality(self) -> None:
        for passing in (10, 819, 4096, 7373, 8110):
            words = study.sampled_words(np.random.default_rng(42), passing)
            self.assertEqual(
                int(study.popcounts(words.reshape(1, -1))[0]), passing
            )

    def test_correlated_bitmap_has_exact_cardinality_and_lift(self) -> None:
        nearest = np.random.default_rng(7).choice(
            study.BASE_ROWS, 1024, replace=False
        )
        for fraction in (0.10, 0.25, 0.50, 0.90, 0.99):
            passing = round(fraction * study.BASE_ROWS)
            words = study.sampled_words(
                np.random.default_rng(42), passing, nearest
            )
            self.assertEqual(
                int(study.popcounts(words.reshape(1, -1))[0]), passing
            )
            near_pass = int(
                np.count_nonzero(
                    words[nearest >> 5] & (np.uint32(1) << (nearest & 31))
                )
            )
            self.assertEqual(
                near_pass,
                round(
                    (
                        passing / study.BASE_ROWS
                        + 0.5 * (1 - passing / study.BASE_ROWS)
                    )
                    * 1024
                ),
            )

    def test_negative_bitmap_has_exact_cardinality_and_reciprocal_lift(self) -> None:
        nearest = np.random.default_rng(7).choice(
            study.BASE_ROWS, 1024, replace=False
        )
        for fraction in (0.10, 0.25, 0.50, 0.90, 0.99):
            passing = round(fraction * study.BASE_ROWS)
            selectivity = passing / study.BASE_ROWS
            first = study.sampled_words(
                np.random.default_rng(42), passing, nearest, relation="negative"
            )
            second = study.sampled_words(
                np.random.default_rng(42), passing, nearest, relation="negative"
            )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(int(study.popcounts(first.reshape(1, -1))[0]), passing)
            local_count = int(np.count_nonzero(
                first[nearest >> 5] & (np.uint32(1) << (nearest & 31))
            ))
            self.assertEqual(
                local_count,
                round(1024 * 2 * selectivity * selectivity / (1 + selectivity)),
            )
            self.assertGreaterEqual(local_count, study.K)
            self.assertLess(local_count / 1024, selectivity)

    def test_deep_sweep_respects_cagra_hash_capacity(self) -> None:
        self.assertEqual(
            study.legal_hash_iterations("default_cagra", 512, 2), 4092
        )
        self.assertEqual(
            study.legal_hash_iterations("default_cagra_accumulator", 64, 1),
            8191,
        )
        self.assertEqual(
            study.legal_hash_iterations("navix_reference", 512, 2), 2046
        )
        self.assertEqual(
            study.legal_hash_iterations("navix_reference", 64, 1), 4095
        )
        for method in study.METHODS + study.MATCHED_METHODS:
            measured: set[tuple[int, int, int]] = set()
            for requested in study.DEEP_ITERATIONS:
                points = study.deep_search_points(method, requested, measured)
                for point in points:
                    cell = (
                        int(point["itopk"]),
                        int(point["search_width"]),
                        int(point["max_iterations"]),
                    )
                    self.assertNotIn(cell, measured)
                    self.assertLessEqual(
                        cell[2], study.legal_hash_iterations(method, *cell[:2])
                    )
                    measured.add(cell)
            self.assertEqual(
                study.deep_search_points(method, study.DEEP_ITERATIONS[-1], measured),
                [],
            )

    def test_cohort_geometry_gt_and_immutable_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = np.arange(20, dtype=np.uint32)
            words = study.bitwords_from_ids(ids)
            queries = np.zeros((2, study.DIM), dtype=np.float32)
            gt = np.stack((ids[:10], ids[10:20]))
            metadata = {"kind": "test"}
            manifest = study.write_cohort(
                root,
                "fixture",
                queries,
                [words, words],
                [5, 9],
                gt_rows=gt,
                metadata=metadata,
            )
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["query_rows"], 2)
            self.assertEqual(payload["original_query_ids"], [5, 9])
            self.assertEqual(
                study.matrix(
                    Path(payload["shards"][0]["directory"])
                    / "groundtruth.ibin",
                    "<u4",
                ).shape,
                (2, 10),
            )
            self.assertEqual(
                study.write_cohort(
                    root,
                    "fixture",
                    queries,
                    [words, words],
                    [5, 9],
                    gt_rows=gt,
                    metadata=metadata,
                ),
                manifest,
            )
            points = [
                study.followup_search_point(method, 64, 2, 0)
                for method in study.METHODS
            ]
            self.assertEqual(
                {point["max_queries"] for point in points},
                {study.MAX_QUERIES},
            )
            graph_manifest = study.graph_manifest(
                root, root, "fixture", "b0", points, 3
            )
            graph = json.loads(graph_manifest.read_text())
            self.assertEqual(graph["expected_queries"], 2)
            self.assertEqual(graph["max_queries"], study.MAX_QUERIES)
            self.assertEqual(graph["search_points"][2]["navix_seed_cap"], 128)
            config = json.loads(
                Path(graph["configs"][0]["config"]).read_text()
            )
            self.assertEqual(
                {row["max_queries"] for row in config["index"][0]["search_params"]},
                {study.MAX_QUERIES},
            )
            self.assertEqual(
                config["dataset"]["filter"]["file"],
                str(Path(payload["shards"][0]["directory"]) / "filter.bitmap"),
            )
            with self.assertRaisesRegex(ValueError, "max_queries=2048"):
                study.graph_manifest(
                    root,
                    root,
                    "fixture",
                    "bad",
                    [dict(points[0], max_queries=512)],
                    3,
                )
            with self.assertRaisesRegex(ValueError, "normal-hash limit 4092"):
                study.graph_manifest(
                    root,
                    root,
                    "fixture",
                    "too_deep",
                    [study.followup_search_point("default_cagra", 512, 2, 4176)],
                    3,
                )
            raw = root / "graph/raw/b0/fixture/shard_00.json"
            raw.parent.mkdir(parents=True)
            records = []
            for repetition in range(3):
                for method in study.METHODS:
                    label = (
                        f'bitmap_method="{method}"#algo="single_cta"#'
                        'filter_mode="default"'
                    )
                    if method == "navix_reference":
                        label += (
                            '#navix_mode="adaptive_kuzu"#navix_scheduler="tiled"#'
                            'navix_kernel_variant="reference"'
                        )
                    record = {
                        "name": "synthetic",
                        "run_type": "iteration",
                        "repetition_index": repetition,
                        "n_queries": 2,
                        "k": 10,
                        "max_queries": 2048,
                        "ValidGTRecall": 0.96,
                        "ValidGTFraction": 1,
                        "items_per_second": 1000.0,
                        "itopk": 64,
                        "search_width": 2,
                        "max_iterations": 0,
                        "favor_udf_passing_accumulator": int(
                            "accumulator" in method
                        ),
                        "cagra_bitmap_seeds": 0,
                        "navix_bitmap_seeds": int(method == "navix_reference"),
                        "require_identity_source_indices": 1,
                        "FilterViolations": 0,
                        "InvalidSentinelErrors": 0,
                        "SentinelOrderErrors": 0,
                        "InvalidSentinelDistanceErrors": 0,
                        "DuplicateOutputQueries": 0,
                        "OutputSetSemanticsVersion": 1,
                        "UnderfilledQueries": 0,
                        "MissingResultSlots": 0,
                        "label": label,
                    }
                    if method == "navix_reference":
                        record["navix_seed_cap"] = 128
                    records.append(record)
            raw.write_text(json.dumps({"benchmarks": records}) + "\n")
            summary = study.analyzed_rows(root)
            self.assertEqual(len(summary), 3)
            self.assertTrue(
                all(row.queries_per_repetition == 2 for row in summary)
            )
            self.assertTrue(all(row.qps_median == 1000 for row in summary))
            exact_dir = root / "exact/raw_control"
            exact_dir.mkdir(parents=True)
            exact_rows = []
            for repetition in range(3):
                exact_rows.append(
                    {
                        "run_type": "iteration",
                        "repetition_index": repetition,
                        "n_queries": 2,
                        "k": 10,
                        "items_per_second": 900.0,
                        "FilterViolations": 0,
                        "InvalidSentinelErrors": 0,
                        "SentinelOrderErrors": 0,
                        "InvalidSentinelDistanceErrors": 0,
                        "DuplicateOutputQueries": 0,
                        "NativeL2StrictPrefixErrors": 0,
                        "NativeL2CutoffRecall": 1,
                        "NativeL2CutoffErrors": 0,
                        "NativeL2CutoffValidated": 1,
                        "OutputSetSemanticsVersion": 1,
                    }
                )
            (exact_dir / "fixture.json").write_text(
                json.dumps({"benchmarks": exact_rows})
            )
            study.analyze(root)
            self.assertTrue((root / "analysis/selected.csv").is_file())
            self.assertTrue(
                (root / "analysis/frontiers/fixture.pdf").is_file()
            )
            archive = root / "fixture_results.tar.gz"
            study.bundle(root, archive)
            self.assertTrue(archive.is_file())
            with self.assertRaises(FileExistsError):
                study.bundle(root, archive)
            with self.assertRaises(ValueError):
                study.write_cohort(
                    root,
                    "fixture",
                    queries,
                    [words, words],
                    [6, 9],
                    gt_rows=gt,
                    metadata=metadata,
                )

    def test_gt_violation_is_rejected(self) -> None:
        words = study.bitwords_from_ids(np.arange(10, dtype=np.uint32))
        with self.assertRaisesRegex(ValueError, "GT ID fails"):
            study.validate_gt(np.arange(1, 11, dtype=np.uint32), words, 10)


if __name__ == "__main__":
    unittest.main()
