#!/usr/bin/env python3
"""CPU-only tests for the A100 profile, large-data preparer, and result bundler."""

from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RETRIEVE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(RETRIEVE_DIR / "gpu_graph"))
sys.path.insert(0, str(RETRIEVE_DIR.parent / "favor" / "navix_bitmap"))

from dataset_profile import load_profile
from prepare_arxiv_large import materialize_phase, packed_words
from prepare_bitmaps import Matrix


class A100PipelineTest(unittest.TestCase):
    def test_bigann_groundtruth_accepts_trailing_distances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "GT.public.ibin"
            ids = np.arange(20, dtype="<u4").reshape(2, 10)
            distances = np.arange(20, dtype="<f4").reshape(2, 10)
            with path.open("wb") as stream:
                stream.write(struct.pack("<II", 2, 10))
                ids.tofile(stream)
                distances.tofile(stream)

            with self.assertRaises(ValueError):
                Matrix(path, np.dtype("<u4"))
            matrix = Matrix(
                path,
                np.dtype("<u4"),
                allow_trailing_float_distances=True,
            )
            np.testing.assert_array_equal(matrix.values, ids)

    def test_profile_contract(self) -> None:
        profile = load_profile(
            SCRIPT_DIR / "profiles" / "a100_yfcc10m_arxiv_large.json"
        )
        self.assertEqual(profile["matched_widths"], [1, 2])
        self.assertEqual(profile["datasets"]["yfcc"]["graph_degree"], 64)
        self.assertEqual(profile["datasets"]["em"]["dataset_size"], 2_735_264)
        self.assertEqual(profile["datasets"]["emis"]["dimension"], 4096)

    def test_packed_words_are_little_endian_bitmap_words(self) -> None:
        mask = np.zeros(64, dtype=np.bool_)
        mask[[0, 1, 31, 32, 63]] = True
        words = packed_words(mask)
        self.assertEqual(words.tolist(), [0x80000003, 0x80000001])

    def test_small_materialization_validates_gt_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries = np.arange(12, dtype=np.float32).reshape(6, 2)
            masks: dict[str, list[np.ndarray]] = {}
            ground_truth: dict[str, np.ndarray] = {}
            for method, offset in (("em", 0), ("emis", 1), ("r", 2)):
                local: list[np.ndarray] = []
                gt = np.full((6, 10), np.iinfo(np.uint32).max, dtype=np.uint32)
                for query in range(6):
                    mask = np.zeros(32, dtype=np.bool_)
                    mask[(query + offset) % 32] = True
                    gt[query, 0] = (query + offset) % 32
                    local.append(mask)
                masks[method] = local
                ground_truth[method] = gt

            def row(method: str):
                return lambda query: (
                    packed_words(masks[method][query]),
                    int(np.count_nonzero(masks[method][query])),
                )

            materialize_phase(
                output=root,
                phase="throughput",
                query_count=6,
                shard_size=2,
                base_rows=32,
                queries=queries,
                ground_truth=ground_truth,
                bitmap_row={method: row(method) for method in masks},
                reuse_valid=False,
            )
            for method in masks:
                manifest = json.loads(
                    (
                        root / method / "throughput_6" / "manifest.json"
                    ).read_text()
                )
                self.assertEqual(
                    [item["query_count"] for item in manifest["shards"]],
                    [2, 2, 2],
                )
                self.assertTrue(
                    all(
                        item["mean_selectivity"] == 1 / 32
                        for item in manifest["shards"]
                    )
                )

    def test_generator_uses_large_profile_for_all_arxiv_predicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            for workload in ("yfcc", "em", "emis", "r"):
                directory = (
                    "navix_bitmap/yfcc"
                    if workload == "yfcc"
                    else f"navix_bitmap/arxiv-large/{workload}"
                )
                for phase, count, shard_size in (
                    ("correctness", 1000, 1000),
                    ("throughput", 10000, 2048),
                ):
                    shards = []
                    for first in range(0, count, shard_size):
                        current = min(shard_size, count - first)
                        shard = (
                            data
                            / directory
                            / f"{phase}_{count}"
                            / f"shard_{first:05d}"
                        )
                        shard.mkdir(parents=True)
                        shards.append(
                            {
                                "first_query": first,
                                "query_count": current,
                                "directory": str(shard),
                            }
                        )
                    manifest = (
                        data / directory / f"{phase}_{count}" / "manifest.json"
                    )
                    manifest.write_text(json.dumps({"shards": shards}) + "\n")
            output = root / "configs"
            environment = os.environ.copy()
            environment["RETRIEVE_DATASET_PROFILE"] = str(
                SCRIPT_DIR / "profiles" / "a100_yfcc10m_arxiv_large.json"
            )
            subprocess.run(
                (
                    sys.executable,
                    str(RETRIEVE_DIR / "gpu_graph" / "generate_configs.py"),
                    "--output",
                    str(output),
                    "--data-root",
                    str(data),
                ),
                env=environment,
                check=True,
            )
            manifest = json.loads(
                (output / "b0" / "emis" / "manifest.json").read_text()
            )
            config = json.loads(
                Path(manifest["configs"][0]["config"]).read_text()
            )
            self.assertEqual(manifest["dataset_size"], 2_735_264)
            self.assertEqual(manifest["graph_degree"], 64)
            self.assertEqual(
                config["dataset"]["base_file"],
                "arxiv-for-fanns-large/base.fbin",
            )
            self.assertEqual(
                config["index"][0]["file"],
                "arxiv-for-fanns-large/cagra_g64_ig128.index",
            )

    def test_bundle_is_hash_bound_and_contains_all_four_workloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (
                root / "gpu_graph/analysis",
                root / "matched_recall/analysis",
                root / "exact_bitmap/analysis",
                root / "resource_work/analysis",
                root / "mechanism_diagnostics/analysis",
                root / "dataset_stats",
                root / "provenance",
            ):
                directory.mkdir(parents=True)

            fields = [
                "group",
                "phase",
                "workload",
                "method",
                "itopk",
                "search_width",
                "max_iterations",
                "recall_median",
                "qps_median",
                "paper_included",
            ]
            rows = []
            for workload in ("yfcc", "em", "emis", "r"):
                for method in (
                    "default_cagra",
                    "default_cagra_accumulator",
                    "navix_reference",
                ):
                    rows.append(
                        {
                            "group": "b0",
                            "phase": "throughput",
                            "workload": workload,
                            "method": method,
                            "itopk": 64,
                            "search_width": 1,
                            "max_iterations": 0,
                            "recall_median": 0.95
                            if workload != "yfcc"
                            else 0.8,
                            "qps_median": 1000,
                            "paper_included": True,
                        }
                    )
            for path in (
                root / "gpu_graph/analysis/summary_points.csv",
                root / "matched_recall/analysis/selected_points.csv",
            ):
                with path.open("w", newline="") as destination:
                    writer = csv.DictWriter(destination, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            with (root / "exact_bitmap/analysis/exact_summary.csv").open(
                "w", newline=""
            ) as destination:
                fields_exact = ["workload", "phase", "median_qps"]
                writer = csv.DictWriter(destination, fieldnames=fields_exact)
                writer.writeheader()
                writer.writerows(
                    {
                        "workload": workload,
                        "phase": "throughput",
                        "median_qps": 100,
                    }
                    for workload in ("yfcc", "em", "emis", "r")
                )
            (root / "resource_work/analysis/gpu_resource_work.csv").write_text(
                "workload,method\nyfcc,default_cagra\n"
            )
            for path in (
                root / "mechanism_diagnostics/analysis/mechanism_summary.json",
                root / "dataset_stats/workload_selectivity_summary.json",
                root / "provenance/a100_preflight.json",
            ):
                path.write_text("{}\n")
            subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT_DIR / "bundle.py"),
                    "--run-root",
                    str(root),
                    "--profile",
                    str(SCRIPT_DIR / "profiles/a100_yfcc10m_arxiv_large.json"),
                ),
                check=True,
            )
            manifest = json.loads(
                (root / "paper_gpu_bundle/manifest.json").read_text()
            )
            self.assertGreater(len(manifest["files"]), 8)
            self.assertTrue(
                (root / "paper_gpu_bundle/gpu_qps_recall_a100.pdf").is_file()
            )


if __name__ == "__main__":
    unittest.main()
