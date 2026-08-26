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

from bundle import validate_max_queries_contract
from dataset_profile import load_profile
from mechanism_diagnostics import validate_base_retain_traversal
from prepare_arxiv_large import materialize_phase, packed_words
from prepare_bitmaps import Matrix


class A100PipelineTest(unittest.TestCase):
    def test_retain_predicate_probes_are_measured_not_equalized(self) -> None:
        base = [
            {
                "iterations": "69",
                "graph_rows_read": "68",
                "predicate_probes": "163",
                "distance_evaluations": "2687",
                "passing_admissions": "11",
                "gt_seen_mask": "31",
            }
        ]
        retain = [dict(base[0], predicate_probes="2755")]
        validate_base_retain_traversal(base, retain)

        changed_traversal = [dict(retain[0], graph_rows_read="69")]
        with self.assertRaisesRegex(ValueError, "graph_rows_read"):
            validate_base_retain_traversal(base, changed_traversal)

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
        self.assertEqual(profile["max_queries"], 2_048)
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
                        (shard / "groundtruth.ibin").write_bytes(b"fixture")
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
            self.assertEqual(manifest["max_queries"], 2_048)
            self.assertTrue(
                all(
                    int(search["max_queries"]) == 2_048
                    for search in config["index"][0]["search_params"]
                )
            )
            self.assertEqual(
                config["dataset"]["base_file"],
                "arxiv-for-fanns-large/base.fbin",
            )
            self.assertEqual(
                config["index"][0]["file"],
                "arxiv-for-fanns-large/cagra_g64_ig128.index",
            )
            for group_manifest_path in output.glob("*/*/manifest.json"):
                group_manifest = json.loads(group_manifest_path.read_text())
                self.assertEqual(group_manifest["max_queries"], 2_048)
                for shard in group_manifest["configs"]:
                    shard_config = json.loads(Path(shard["config"]).read_text())
                    self.assertEqual(
                        {
                            int(row["max_queries"])
                            for row in shard_config["index"][0]["search_params"]
                        },
                        {2_048},
                    )

            matched_root = root / "matched"
            matched_points = root / "matched_points.json"
            matched_points.write_text(
                json.dumps(
                    {
                        "points": [
                            {
                                "workload": "emis",
                                "method": "default_cagra_accumulator",
                                "itopk": 64,
                                "search_width": 1,
                                "max_iterations": 0,
                            }
                        ]
                    }
                )
                + "\n"
            )
            subprocess.run(
                (
                    sys.executable,
                    str(RETRIEVE_DIR / "matched_recall" / "matched_recall.py"),
                    "generate-group",
                    "--result-root",
                    str(matched_root),
                    "--data-root",
                    str(data),
                    "--group",
                    "synthetic",
                    "--repetitions",
                    "1",
                    "--points",
                    str(matched_points),
                ),
                env=environment,
                check=True,
            )
            matched_manifest = json.loads(
                (matched_root / "configs/synthetic/emis/manifest.json").read_text()
            )
            self.assertEqual(matched_manifest["max_queries"], 2_048)
            for shard in matched_manifest["configs"]:
                matched_config = json.loads(Path(shard["config"]).read_text())
                self.assertEqual(
                    matched_config["index"][0]["search_params"][0]["max_queries"],
                    2_048,
                )

            gate = root / "maxq_gate"
            subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT_DIR / "max_queries_gate.py"),
                    "generate",
                    "--data-root",
                    str(data),
                    "--output",
                    str(gate),
                ),
                env=environment,
                check=True,
            )
            gate_manifest = json.loads((gate / "manifest.json").read_text())
            self.assertEqual(gate_manifest["caps"], [512, 1024, 2048])
            self.assertEqual(gate_manifest["production_cap"], 2_048)
            self.assertEqual(len(gate_manifest["records"]), 6)
            for record in gate_manifest["records"]:
                gate_config = json.loads(Path(record["config"]).read_text())
                self.assertEqual(
                    {int(row["max_queries"]) for row in gate_config["index"][0]["search_params"]},
                    {int(record["max_queries"])},
                )
                benchmarks = []
                for point in record["search_points"]:
                    method = point["method"]
                    label = (
                        f'bitmap_method="{method}"#algo="single_cta"#'
                        'filter_mode="default"'
                    )
                    if method == "navix_reference":
                        label += (
                            '#navix_mode="adaptive_kuzu"#navix_scheduler="tiled"'
                            '#navix_kernel_variant="reference"'
                        )
                    benchmarks.append(
                        {
                            "run_type": "iteration",
                            "label": label,
                            "itopk": point["itopk"],
                            "search_width": point["search_width"],
                            "max_iterations": point["max_iterations"],
                            "max_queries": record["max_queries"],
                            "n_queries": 2_048,
                            "k": 10,
                            "favor_udf_passing_accumulator": int(
                                "accumulator" in method
                            ),
                            "navix_bitmap_seeds": int(
                                method == "navix_reference"
                            ),
                            "require_identity_source_indices": 1,
                            "FilterViolations": 0,
                            "InvalidSentinelErrors": 0,
                            "SentinelOrderErrors": 0,
                            # Native graph output may leave non-canonical
                            # distances in invalid underfilled slots.  This is
                            # reported but is not a hard ID-set correctness
                            # error for Base or Retain.
                            "InvalidSentinelDistanceErrors": (
                                0.2
                                if method
                                in (
                                    "default_cagra",
                                    "default_cagra_accumulator",
                                )
                                else 0
                            ),
                            "DuplicateOutputQueries": (
                                0.01 if method == "default_cagra" else 0
                            ),
                            "UnderfilledQueries": (
                                0.25
                                if method
                                in (
                                    "default_cagra",
                                    "default_cagra_accumulator",
                                )
                                else 0
                            ),
                            "MissingResultSlots": (
                                0.2
                                if method
                                in (
                                    "default_cagra",
                                    "default_cagra_accumulator",
                                )
                                else 0
                            ),
                            "OutputSetSemanticsVersion": 1,
                            "ValidGTFraction": 1,
                            "items_per_second": 10_000,
                            "ValidGTRecall": 0.8,
                        }
                    )
                raw = (
                    gate
                    / "raw"
                    / f"maxq_{record['max_queries']}"
                    / f"{record['workload']}.json"
                )
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_text(json.dumps({"benchmarks": benchmarks}) + "\n")
            subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT_DIR / "max_queries_gate.py"),
                    "analyze",
                    "--root",
                    str(gate),
                ),
                env=environment,
                check=True,
            )
            gate_summary = json.loads(
                (gate / "analysis/max_queries_gate_summary.json").read_text()
            )
            self.assertEqual(gate_summary["status"], "PASS")
            self.assertEqual(gate_summary["production_max_queries"], 2_048)
            self.assertEqual(len(gate_summary["sensitivity"]), 6)

            resource_root = root / "resource"
            subprocess.run(
                (
                    sys.executable,
                    str(RETRIEVE_DIR / "resource_work/generate_configs.py"),
                    "--output",
                    str(resource_root / "configs"),
                    "--data-root",
                    str(data),
                    "--diagnostic-root",
                    str(resource_root / "captures"),
                ),
                env=environment,
                check=True,
            )
            resource_manifest = json.loads(
                (resource_root / "configs/manifest.json").read_text()
            )
            self.assertEqual(resource_manifest["max_queries"], 2_048)
            for workload in ("yfcc", "em", "emis", "r"):
                for mode in ("resources", "diagnostics"):
                    resource_config = json.loads(
                        (
                            resource_root / f"configs/{mode}/{workload}.json"
                        ).read_text()
                    )
                    self.assertEqual(
                        {
                            int(row["max_queries"])
                            for row in resource_config["index"][0]["search_params"]
                        },
                        {2_048},
                    )

            mechanism_config = root / "mechanism.json"
            subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT_DIR / "mechanism_diagnostics.py"),
                    "generate",
                    "--data-root",
                    str(data),
                    "--output",
                    str(mechanism_config),
                    "--diagnostics",
                    str(root / "mechanism_captures"),
                ),
                env=environment,
                check=True,
            )
            mechanism = json.loads(mechanism_config.read_text())
            self.assertEqual(
                {
                    int(row["max_queries"])
                    for row in mechanism["index"][0]["search_params"]
                },
                {2_048},
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
                root / "maxq_gate/analysis",
                root / "per_query_latency/analysis",
                root / "dataset_stats",
                root / "provenance",
                root / "gpu_graph/provenance",
                root / "matched_recall/provenance",
                root / "resource_work/provenance",
                root / "mechanism_diagnostics/provenance",
                root / "maxq_gate/provenance",
                root / "per_query_latency/provenance",
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
                fields_exact = ["workload", "phase", "median_qps", "native_l2_cutoff_recall"]
                writer = csv.DictWriter(destination, fieldnames=fields_exact)
                writer.writeheader()
                writer.writerows(
                    {
                        "workload": workload,
                        "phase": "throughput",
                        "median_qps": 100,
                        "native_l2_cutoff_recall": 0.99998,
                    }
                    for workload in ("yfcc", "em", "emis", "r")
                )
            (root / "resource_work/analysis/gpu_resource_work.csv").write_text(
                "workload,method\nyfcc,default_cagra\n"
            )
            (root / "resource_work/analysis/gpu_resource_work.json").write_text(
                json.dumps({"configuration": {"max_queries": 2_048}}) + "\n"
            )
            for experiment in ("gpu_graph", "matched_recall"):
                (root / experiment / "analysis/provenance.json").write_text(
                    json.dumps({"max_queries": 2_048}) + "\n"
                )
            (root / "mechanism_diagnostics/analysis/mechanism_summary.json").write_text(
                json.dumps({"max_queries": 2_048}) + "\n"
            )
            (root / "maxq_gate/analysis/max_queries_gate_summary.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "production_max_queries": 2_048,
                    }
                )
                + "\n"
            )
            (root / "per_query_latency/analysis/latency_summary.csv").write_text(
                "workload,method,mean_us,p50_us,p95_us,p99_us\n"
                "yfcc,default_cagra,100,90,150,200\n"
            )
            (root / "per_query_latency/analysis/latency_summary.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "measurement_contract": {
                            "source_max_queries": 2_048,
                            "serialized_max_queries": 1,
                            "queries_per_search_call": 1,
                            "complete_passes": 3,
                        },
                    }
                )
                + "\n"
            )
            (root / "per_query_latency/analysis/per_query_latency_cdf.pdf").write_bytes(
                b"fixture"
            )
            for path in (
                root / "dataset_stats/workload_selectivity_summary.json",
                root / "provenance/a100_preflight.json",
            ):
                path.write_text("{}\n")
            for experiment in (
                "gpu_graph",
                "matched_recall",
                "mechanism_diagnostics",
                "maxq_gate",
            ):
                (root / experiment / "provenance/run.json").write_text(
                    json.dumps({"fixed_contract": {"max_queries": 2_048}}) + "\n"
                )
            (root / "resource_work/provenance/run.json").write_text(
                json.dumps({"contract": {"max_queries": 2_048}}) + "\n"
            )
            (root / "per_query_latency/provenance/run.json").write_text(
                json.dumps(
                    {
                        "contract": {
                            "graph_source_max_queries": 2_048,
                            "serialized_max_queries": 1,
                            "queries_per_call": 1,
                        }
                    }
                )
                + "\n"
            )
            profile = json.loads(
                (SCRIPT_DIR / "profiles/a100_yfcc10m_arxiv_large.json").read_text()
            )
            observed, contracts = validate_max_queries_contract(root, profile)
            self.assertEqual(observed, 2_048)
            self.assertEqual(len(contracts), 6)
            latency_summary_path = (
                root / "per_query_latency/analysis/latency_summary.json"
            )
            latency_summary = json.loads(latency_summary_path.read_text())
            latency_summary_path.write_text(
                json.dumps({**latency_summary, "status": "FAIL"}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "serialized-latency"):
                validate_max_queries_contract(root, profile)
            latency_summary_path.write_text(json.dumps(latency_summary) + "\n")
            (root / "mechanism_diagnostics/provenance/run.json").write_text(
                json.dumps({"fixed_contract": {"max_queries": 1_024}}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "mixed max_queries"):
                validate_max_queries_contract(root, profile)
            (root / "mechanism_diagnostics/provenance/run.json").write_text(
                json.dumps({"fixed_contract": {"max_queries": 2_048}}) + "\n"
            )
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
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["execution_contract"]["max_queries"], 2_048
            )
            self.assertEqual(
                manifest["execution_contract"]["serialized_latency"]["max_queries"], 1
            )
            self.assertGreater(len(manifest["files"]), 8)
            self.assertTrue(
                (root / "paper_gpu_bundle/gpu_qps_recall_a100.pdf").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "paper_gpu_bundle/per_query_latency/per_query_latency_cdf.pdf"
                ).is_file()
            )
            original_manifest_hash = (
                root / "paper_gpu_bundle/manifest.json"
            ).read_bytes()
            refined = root / "navix_refined_bundle/paper_gpu_bundle"
            subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT_DIR / "bundle.py"),
                    "--run-root",
                    str(root),
                    "--profile",
                    str(SCRIPT_DIR / "profiles/a100_yfcc10m_arxiv_large.json"),
                    "--output",
                    str(refined),
                ),
                check=True,
            )
            self.assertTrue((refined / "manifest.json").is_file())
            self.assertEqual(
                (root / "paper_gpu_bundle/manifest.json").read_bytes(),
                original_manifest_hash,
            )


if __name__ == "__main__":
    unittest.main()
