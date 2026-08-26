#!/usr/bin/env python3
"""Synthetic contract tests for the A100 Recall@100 runner."""

from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RETRIEVE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(RETRIEVE_DIR / "exact_bitmap"))

MATRIX_HEADER = struct.Struct("<II")
BITMAP_HEADER = struct.Struct("<8sIIQQQ")
WORKLOADS = ("yfcc", "em", "emis", "r")
METHODS = ("default_cagra", "default_cagra_accumulator", "navix_reference")


def write_matrix(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(MATRIX_HEADER.pack(*values.shape))
        values.tofile(output)


def write_bitmap(path: Path, rows: list[np.ndarray], cols: int) -> None:
    words = (len(rows) * cols + 31) // 32
    payload = np.zeros(words, dtype="<u4")
    for row, ids in enumerate(rows):
        flat = row * cols + ids.astype(np.int64)
        np.bitwise_or.at(
            payload,
            flat >> 5,
            np.left_shift(np.uint32(1), (flat & 31).astype(np.uint32)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(
            BITMAP_HEADER.pack(b"CUVSBMAP", 1, 32, len(rows), cols, words)
        )
        payload.tofile(output)


def make_generator_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    for relative in (
        "yfcc-10M/base.10M.u8bin",
        "yfcc-10M/cagra_g64_ig128.index",
        "arxiv-for-fanns-large/base.fbin",
        "arxiv-for-fanns-large/cagra_g64_ig128.index",
    ):
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    datasets: dict[str, dict[str, object]] = {}
    for workload in WORKLOADS:
        bitmap_directory = (
            "navix_bitmap_k100/yfcc"
            if workload == "yfcc"
            else f"navix_bitmap_k100/arxiv-large/{workload}"
        )
        datasets[workload] = {
            "bitmap_directory": bitmap_directory,
            "base_file": (
                "yfcc-10M/base.10M.u8bin"
                if workload == "yfcc"
                else "arxiv-for-fanns-large/base.fbin"
            ),
            "index_file": (
                "yfcc-10M/cagra_g64_ig128.index"
                if workload == "yfcc"
                else "arxiv-for-fanns-large/cagra_g64_ig128.index"
            ),
            "dtype": "uint8" if workload == "yfcc" else "float",
            "dataset_size": 10_000_000 if workload == "yfcc" else 2_735_264,
            "dimension": 192 if workload == "yfcc" else 4096,
            "graph_degree": 64,
            "intermediate_graph_degree": 128,
        }
        for phase, count in (
            ("correctness_1000", 1000),
            ("throughput_10000", 10_000),
        ):
            directory = data / bitmap_directory / phase / "shard_00"
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": 2,
                "k": 100,
                "query_rows": count,
                "shards": [
                    {
                        "first_query": 0,
                        "query_count": count,
                        "directory": str(directory),
                    }
                ],
            }
            path = data / bitmap_directory / phase / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest) + "\n")
    profile = root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "k100-test",
                "max_queries": 2048,
                "matched_widths": [1, 2],
                "datasets": datasets,
            }
        )
        + "\n"
    )
    return data, profile


class K100PipelineTest(unittest.TestCase):
    def test_graph_generator_emits_complete_k100_cartesian_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, profile = make_generator_fixture(root)
            output = root / "configs"
            environment = dict(os.environ)
            environment["RETRIEVE_DATASET_PROFILE"] = str(profile)
            subprocess.run(
                [
                    sys.executable,
                    str(RETRIEVE_DIR / "gpu_graph/generate_configs.py"),
                    "--output",
                    str(output),
                    "--data-root",
                    str(data),
                    "--k",
                    "100",
                    "--primary-methods-only",
                    "--cartesian-b0",
                ],
                check=True,
                env=environment,
            )
            for workload in WORKLOADS:
                b0 = json.loads(
                    (output / "b0" / workload / "manifest.json").read_text()
                )
                correctness = json.loads(
                    (
                        output / "correctness" / workload / "manifest.json"
                    ).read_text()
                )
                self.assertEqual(b0["k"], 100)
                self.assertEqual(len(b0["search_points"]), 18)
                self.assertEqual(
                    {
                        (row["itopk"], row["search_width"])
                        for row in b0["search_points"]
                    },
                    {
                        (l_value, width)
                        for l_value in (128, 256, 512)
                        for width in (1, 2)
                    },
                )
                self.assertEqual(
                    {row["method"] for row in b0["search_points"]},
                    set(METHODS),
                )
                self.assertEqual(len(correctness["search_points"]), 3)
                self.assertEqual(
                    {row["itopk"] for row in correctness["search_points"]},
                    {128},
                )
                config = json.loads(
                    Path(b0["configs"][0]["config"]).read_text()
                )
                self.assertEqual(config["search_basic_param"]["k"], 100)
                self.assertTrue(
                    all(
                        row["k"] == 100
                        for row in config["index"][0]["search_params"]
                    )
                )

    def test_yfcc_generated_gt_validation_handles_unaligned_underfill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cols = 130
            bitmap = root / "filter.bitmap"
            query_count = 10_000
            passing = [
                np.arange(120, dtype=np.int64)
                if query % 2 == 0
                else np.arange(10, 70, dtype=np.int64)
                for query in range(query_count)
            ]
            write_bitmap(bitmap, passing, cols)
            gt = np.full(
                (query_count, 100), np.iinfo(np.uint32).max, dtype="<u4"
            )
            gt[0::2] = np.arange(100, dtype="<u4")
            gt[1::2, :60] = np.arange(10, 70, dtype="<u4")
            gt_path = root / "groundtruth.ibin"
            write_matrix(gt_path, gt)
            official = np.empty((query_count, 10), dtype="<u4")
            official[0::2] = np.arange(10, dtype="<u4")
            official[1::2] = np.arange(10, 20, dtype="<u4")
            official_path = root / "GT.public.ibin"
            write_matrix(official_path, official)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": "cuvs_brute_force_knn_masked_gt_generation",
                        "k": 100,
                        "base_rows": cols,
                        "query_rows": query_count,
                        "source_official_gt10": str(official_path),
                        "complete": False,
                        "shards": [
                            {
                                "first_query": 0,
                                "query_count": query_count,
                                "bitmap_file": str(bitmap),
                                "groundtruth_file": str(gt_path),
                            }
                        ],
                    }
                )
                + "\n"
            )
            from prepare_yfcc_gt100 import finalize

            finalize(Namespace(output=root))
            completed = json.loads(manifest_path.read_text())
            self.assertIs(completed["complete"], True)
            self.assertEqual(
                completed["validation"]["underfilled_queries"], 5000
            )
            first_mtime = manifest_path.stat().st_mtime_ns
            finalize(Namespace(output=root))
            self.assertEqual(manifest_path.stat().st_mtime_ns, first_mtime)

            from prepare_k100_views import (
                read_generated_yfcc_gt,
                validate_gt_membership,
            )

            loader_safe = read_generated_yfcc_gt(manifest_path)
            self.assertEqual(len(set(map(int, loader_safe[1]))), 100)
            np.testing.assert_array_equal(
                loader_safe[1, 60:64],
                np.asarray(
                    [
                        np.iinfo(np.uint32).max,
                        np.iinfo(np.uint32).max - 1,
                        np.iinfo(np.uint32).max - 2,
                        np.iinfo(np.uint32).max - 3,
                    ],
                    dtype="<u4",
                ),
            )
            validate_gt_membership(bitmap, loader_safe, 0)

    def test_combined_analyzer_writes_four_workload_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_analysis = root / "gpu_graph/analysis"
            exact_analysis = root / "exact_bitmap/analysis"
            graph_analysis.mkdir(parents=True)
            exact_analysis.mkdir(parents=True)
            (graph_analysis / "provenance.json").write_text(
                json.dumps({"k": 100}) + "\n"
            )
            fieldnames = [
                "phase",
                "workload",
                "method",
                "max_iterations",
                "paper_included",
                "itopk",
                "search_width",
                "recall_median",
                "recall_min",
                "recall_max",
                "qps_median",
                "qps_min",
                "qps_max",
                "repetitions",
            ]
            with (graph_analysis / "summary_points.csv").open(
                "w", newline=""
            ) as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for workload in WORKLOADS:
                    for method_number, method in enumerate(METHODS):
                        for l_value in (128, 256, 512):
                            for width in (1, 2):
                                recall = (
                                    0.55
                                    + 0.05 * method_number
                                    + 0.0002 * l_value
                                )
                                writer.writerow(
                                    {
                                        "phase": "throughput",
                                        "workload": workload,
                                        "method": method,
                                        "max_iterations": 0,
                                        "paper_included": True,
                                        "itopk": l_value,
                                        "search_width": width,
                                        "recall_median": recall,
                                        "recall_min": recall,
                                        "recall_max": recall,
                                        "qps_median": 100_000
                                        / l_value
                                        / width,
                                        "qps_min": 99_000 / l_value / width,
                                        "qps_max": 101_000 / l_value / width,
                                        "repetitions": 3,
                                    }
                                )
            (exact_analysis / "exact_results.json").write_text(
                json.dumps(
                    {
                        "k": 100,
                        "summaries": [
                            {
                                "workload": workload,
                                "phase": "throughput",
                                "correct": True,
                                "native_l2_cutoff_recall": 1.0,
                                "median_qps": 100.0,
                                "min_qps": 99.0,
                                "max_qps": 101.0,
                                "repetitions": 3,
                            }
                            for workload in WORKLOADS
                        ],
                    }
                )
                + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "analyze.py"),
                    "--run-root",
                    str(root),
                ],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            self.assertTrue(
                (root / "analysis/plots/gpu_qps_recall_k100.png").is_file()
            )
            for workload in WORKLOADS:
                self.assertTrue(
                    (
                        root / f"analysis/plots/{workload}_qps_recall_k100.pdf"
                    ).is_file()
                )

    def test_view_builder_replaces_legacy_repeated_padding_view(self) -> None:
        from prepare_k100_views import (
            VIEW_KIND,
            VIEW_SCHEMA_VERSION,
            create_view,
            make_loader_safe_padding,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cols = 130
            source_directory = root / "source/shard_00"
            source_directory.mkdir(parents=True)
            (source_directory / "query.bin").write_bytes(b"fixture")
            bitmap = source_directory / "filter.bitmap"
            write_bitmap(
                bitmap,
                [np.arange(120), np.arange(10, 70)],
                cols,
            )
            source_manifest = root / "source/manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "query_rows": 2,
                        "shards": [
                            {
                                "first_query": 0,
                                "query_count": 2,
                                "directory": str(source_directory),
                                "bitmap": str(bitmap),
                            }
                        ],
                    }
                )
                + "\n"
            )
            canonical = np.full((2, 100), np.iinfo(np.uint32).max, dtype="<u4")
            canonical[0] = np.arange(100, dtype="<u4")
            canonical[1, :60] = np.arange(10, 70, dtype="<u4")
            loader_safe = make_loader_safe_padding(canonical, cols)

            target = root / "view"
            target.mkdir()
            (target / "legacy-marker").write_text("stale\n")
            (target / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "k": 100,
                        "view_kind": "symlinked_bitmap_query_with_k100_groundtruth",
                    }
                )
                + "\n"
            )
            create_view(
                source_manifest=source_manifest,
                target=target,
                ground_truth=loader_safe,
                ground_truth_source={"fixture": True},
            )
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], VIEW_SCHEMA_VERSION)
            self.assertEqual(manifest["view_kind"], VIEW_KIND)
            self.assertFalse((target / "legacy-marker").exists())
            output_gt = np.memmap(
                target / "shard_00_00000_00002/groundtruth.ibin",
                dtype="<u4",
                mode="r",
                offset=MATRIX_HEADER.size,
                shape=(2, 100),
            )
            self.assertEqual(len(set(map(int, output_gt[1]))), 100)

    def test_matched_table_and_bundle_hide_configs_but_preserve_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matched = root / "matched_recall/analysis"
            graph = root / "gpu_graph/analysis"
            exact = root / "exact_bitmap/analysis"
            run_provenance = root / "provenance"
            for directory in (matched, graph, exact, run_provenance):
                directory.mkdir(parents=True)
            selected_rows = []
            measurement_rows = []
            for workload in WORKLOADS:
                target = 0.80 if workload == "yfcc" else 0.95
                for method in METHODS:
                    selected_rows.append(
                        {
                            "workload": workload,
                            "method": method,
                            "recall_median": target + 0.001,
                            "recall_min": target + 0.0005,
                            "within_target_window": True,
                            "target_reached": True,
                            "qps_median": 12_345,
                            "itopk": 100,
                            "search_width": 1,
                            "max_iterations": 7
                            if method == "navix_reference"
                            else 0,
                            "resolved_iterations": 7
                            if method == "navix_reference"
                            else 105,
                            "filter_violations": 0,
                            "sentinel_errors": 0,
                            "duplicate_output_query_rate_max": 0,
                        }
                    )
                    measurement_rows.append(
                        {
                            "workload": workload,
                            "method": method,
                            "recall": target + 0.001,
                            "qps": 12_345,
                        }
                    )
            with (matched / "selected_points.csv").open(
                "w", newline=""
            ) as output:
                writer = csv.DictWriter(
                    output, fieldnames=list(selected_rows[0])
                )
                writer.writeheader()
                writer.writerows(selected_rows)
            with (matched / "measurements.csv").open(
                "w", newline=""
            ) as output:
                writer = csv.DictWriter(
                    output, fieldnames=list(measurement_rows[0])
                )
                writer.writeheader()
                writer.writerows(measurement_rows)
            provenance = {
                "k": 100,
                "max_queries": 2048,
                "targets": {"yfcc": 0.80, "em": 0.95, "emis": 0.95, "r": 0.95},
                "target_window": 0.002,
                "selected_rows": selected_rows,
            }
            (matched / "provenance.json").write_text(
                json.dumps(provenance) + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "matched_table.py"),
                    "--result-root",
                    str(root / "matched_recall"),
                ],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            latex = (matched / "fixed_recall_k100_results.tex").read_text()
            self.assertIn(r"\FixedRecallKOneHundredRows", latex)
            self.assertNotIn("L=", latex)
            self.assertNotIn("W=", latex)

            (graph / "summary_points.csv").write_text("fixture\n")
            (graph / "provenance.json").write_text(
                json.dumps({"k": 100}) + "\n"
            )
            (exact / "exact_results.json").write_text(
                json.dumps({"k": 100}) + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "bundle.py"),
                    "--run-root",
                    str(root),
                ],
                check=True,
            )
            bundle = root / "paper_gpu_bundle_k100_matched"
            initial_manifest = json.loads(
                (bundle / "manifest.json").read_text()
            )
            self.assertFalse(initial_manifest["serialized_latency_included"])

            latency_analysis = root / "per_query_latency/analysis"
            latency_provenance = root / "per_query_latency/provenance"
            latency_analysis.mkdir(parents=True)
            latency_provenance.mkdir(parents=True)
            (latency_analysis / "latency_summary.csv").write_text("fixture\n")
            (latency_analysis / "per_query_latency_cdf.pdf").write_bytes(
                b"pdf"
            )
            latency_summary = {
                "k": 100,
                "status": "PASS",
                "query_trace_rows": 480_000,
                "measurement_contract": {
                    "k": 100,
                    "source_max_queries": 2048,
                    "serialized_max_queries": 1,
                    "queries_per_search_call": 1,
                    "complete_passes": 3,
                },
            }
            (latency_analysis / "latency_summary.json").write_text(
                json.dumps(latency_summary) + "\n"
            )
            (latency_provenance / "run.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "contract": {
                            "k": 100,
                            "graph_source_max_queries": 2048,
                            "serialized_max_queries": 1,
                            "queries_per_call": 1,
                        },
                    }
                )
                + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "bundle.py"),
                    "--run-root",
                    str(root),
                ],
                check=True,
            )
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(manifest["k"], 100)
            self.assertTrue(manifest["serialized_latency_included"])
            self.assertTrue(
                (
                    bundle / "matched_recall/fixed_recall_k100_results.tex"
                ).is_file()
            )
            self.assertTrue(
                (
                    bundle / "per_query_latency/analysis/latency_summary.csv"
                ).is_file()
            )
            latency_summary["status"] = "FAIL"
            (latency_analysis / "latency_summary.json").write_text(
                json.dumps(latency_summary) + "\n"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "bundle.py"),
                    "--run-root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("frozen contract", completed.stderr)


if __name__ == "__main__":
    unittest.main()
