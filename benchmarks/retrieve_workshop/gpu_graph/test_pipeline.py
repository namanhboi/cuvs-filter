#!/usr/bin/env python3
"""Synthetic coverage tests for the frozen GPU graph benchmark pipeline."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
WORKLOADS = ("yfcc", "em", "emis", "r")


def source_manifest(root: Path, workload: str, phase: str) -> Path:
    count = "1000" if phase == "correctness" else "10000"
    if workload == "yfcc":
        return (
            root
            / "navix_bitmap"
            / "yfcc"
            / f"{phase}_{count}"
            / "manifest.json"
        )
    return (
        root
        / "navix_bitmap"
        / "arxiv"
        / workload
        / f"{phase}_{count}"
        / "manifest.json"
    )


def make_source_manifests(root: Path) -> None:
    for relative in (
        "yfcc-10M/base.10M.u8bin",
        "yfcc-10M/cagra_g32_ig64.index",
        "yfcc-10M/cagra_g64_ig128.index",
        "arxiv-for-fanns-medium/base.fbin",
        "arxiv-for-fanns-medium/cagra_g32_ig64.index",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    for phase in ("correctness", "throughput"):
        for workload in WORKLOADS:
            if phase == "throughput":
                counts = (2048, 2048, 2048, 2048, 1808)
            else:
                counts = (1000 if phase == "correctness" else 10000,)
            first = 0
            shards = []
            for count in counts:
                directory = (
                    root
                    / "fake"
                    / workload
                    / phase
                    / f"q{first}_{first + count}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                for filename in ("query.bin", "groundtruth.ibin", "filter.bitmap"):
                    (directory / filename).write_bytes(
                        f"{workload}:{phase}:{first}:{filename}".encode()
                    )
                shards.append(
                    {
                        "first_query": first,
                        "query_count": count,
                        "directory": str(directory),
                    }
                )
                first += count
            path = source_manifest(root, workload, phase)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"schema_version": 1, "shards": shards}) + "\n"
            )


def generate(root: Path, data: Path, *extra: str) -> None:
    subprocess.run(
        [
            PYTHON,
            str(SCRIPT_DIR / "generate_configs.py"),
            "--output",
            str(root / "configs"),
            "--data-root",
            str(data),
            *extra,
        ],
        check=True,
    )
    provenance = root / "provenance" / "run.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "data_root": str(data.resolve()),
                "fixed_contract": {
                    "output_set_semantics": "distinct_valid_output_ids_v1",
                    "max_queries": 512,
                    "k": 10,
                },
            }
        )
        + "\n"
    )


def benchmark_label(method: str) -> str:
    fields = [
        f'bitmap_method="{method}"',
        'algo="single_cta"',
        'filter_mode="default"',
    ]
    if method == "navix_reference":
        fields.extend(
            (
                'navix_mode="adaptive_kuzu"',
                'navix_scheduler="tiled"',
                'navix_kernel_variant="reference"',
            )
        )
    return "#".join(fields)


def write_synthetic_raw(result_root: Path, group: str) -> None:
    for manifest_path in sorted(
        (result_root / "configs" / group).glob("*/manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text())
        workload = manifest["workload"]
        raw_dir = result_root / "raw" / group / workload
        raw_dir.mkdir(parents=True, exist_ok=True)
        for shard in manifest["configs"]:
            rows = []
            repetitions = int(manifest["repetitions"])
            for repetition in range(repetitions):
                for point in manifest["search_points"]:
                    method = point["method"]
                    method_number = [
                        "default_cagra",
                        "default_cagra_accumulator",
                        "navix_reference",
                        "default_cagra_seeded",
                        "default_cagra_accumulator_seeded",
                    ].index(method)
                    recall = min(
                        0.99,
                        0.40
                        + 0.05 * method_number
                        + 0.0001 * point["itopk"]
                        + 0.01 * (point["max_iterations"] > 0),
                    )
                    qps = (
                        20_000.0
                        - 1_000.0 * method_number
                        - 4.0 * point["itopk"]
                        - 100.0 * shard["shard_index"]
                    ) * (1.0 + 0.01 * (repetition - 1))
                    rows.append(
                        {
                            "name": "synthetic",
                            "run_type": "iteration",
                            "repetitions": repetitions,
                            "repetition_index": repetition,
                            "n_queries": shard["query_count"],
                            "k": 10,
                            "max_queries": 512,
                            "Recall": recall,
                            "ValidGTRecall": recall,
                            "ValidGTFraction": 1.0,
                            "items_per_second": qps,
                            "itopk": point["itopk"],
                            "search_width": point["search_width"],
                            "max_iterations": point["max_iterations"],
                            "favor_udf_passing_accumulator": float(
                                "accumulator" in method
                            ),
                            "cagra_bitmap_seeds": float(
                                method.endswith("_seeded")
                            ),
                            "navix_bitmap_seeds": float(
                                method == "navix_reference"
                            ),
                            "require_identity_source_indices": 1.0,
                            "FilterViolations": 0,
                            "InvalidSentinelErrors": 0,
                            "DuplicateOutputQueries": 0,
                            "OutputSetSemanticsVersion": 1,
                            "UnderfilledQueries": 0,
                            "MissingResultSlots": 0,
                            "label": benchmark_label(method),
                        }
                    )
            destination = raw_dir / f"shard_{shard['shard_index']:02d}.json"
            destination.write_text(json.dumps({"benchmarks": rows}) + "\n")


class PipelineTest(unittest.TestCase):
    def test_fvecs_converter_streams_exact_rows_and_reuses_only_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tiny.fvecs"
            output = root / "tiny.fbin"
            rows = (
                (1.0, 2.0, 3.0),
                (-4.5, 5.25, 6.0),
                (7.0, 8.0, 9.5),
            )
            source.write_bytes(
                b"".join(struct.pack("<I3f", 3, *row) for row in rows)
            )
            command = (
                PYTHON,
                str(SCRIPT_DIR / "convert_fvecs_to_fbin.py"),
                str(source),
                str(output),
                "--chunk-rows",
                "2",
            )
            subprocess.run(command, check=True, capture_output=True, text=True)
            payload = output.read_bytes()
            self.assertEqual(struct.unpack("<II", payload[:8]), (3, 3))
            self.assertEqual(
                struct.unpack("<9f", payload[8:]),
                tuple(value for row in rows for value in row),
            )
            reused = subprocess.run(
                (*command[:4], "--reuse-valid"),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("reusing valid", reused.stdout)

    def test_generator_freezes_cells_methods_and_deep_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            make_source_manifests(data)
            generate(
                root,
                data,
                "--deep-pair",
                "yfcc:default_cagra",
                "--deep-pair",
                "emis:default_cagra_accumulator",
            )
            b0 = json.loads(
                (
                    root / "configs" / "b0" / "yfcc" / "manifest.json"
                ).read_text()
            )
            self.assertEqual(b0["expected_queries"], 10_000)
            self.assertEqual(b0["expected_shards"], 5)
            self.assertEqual(b0["repetitions"], 3)
            self.assertEqual(b0["graph_degree"], 64)
            self.assertEqual(b0["intermediate_graph_degree"], 128)
            self.assertEqual(len(b0["search_points"]), 30)
            self.assertEqual(
                {row["method"] for row in b0["search_points"]},
                {
                    "default_cagra",
                    "default_cagra_accumulator",
                    "navix_reference",
                    "default_cagra_seeded",
                    "default_cagra_accumulator_seeded",
                },
            )
            config = json.loads(Path(b0["configs"][0]["config"]).read_text())
            self.assertEqual(config["index"][0]["name"], "cagra-g64-ig128")
            self.assertEqual(
                config["index"][0]["file"],
                "yfcc-10M/cagra_g64_ig128.index",
            )
            searches = config["index"][0]["search_params"]
            navix = [
                row
                for row in searches
                if row["bitmap_method"] == "navix_reference"
            ]
            self.assertTrue(all(row["navix_bitmap_seeds"] for row in navix))
            self.assertTrue(
                all(row["navix_mode"] == "adaptive_kuzu" for row in navix)
            )
            seeded = [
                row
                for row in searches
                if row["bitmap_method"].endswith("_seeded")
            ]
            self.assertTrue(all(row["cagra_bitmap_seeds"] for row in seeded))
            correctness = json.loads(
                (
                    root / "configs" / "correctness" / "yfcc" / "manifest.json"
                ).read_text()
            )
            self.assertEqual(correctness["repetitions"], 1)
            self.assertEqual(len(correctness["search_points"]), 5)
            self.assertEqual(
                {row["itopk"] for row in correctness["search_points"]}, {64}
            )
            self.assertEqual(
                {row["search_width"] for row in correctness["search_points"]},
                {1},
            )
            deep = json.loads(
                (
                    root
                    / "configs"
                    / "deep_i522_yfcc_default_cagra"
                    / "yfcc"
                    / "manifest.json"
                ).read_text()
            )
            self.assertEqual(len(deep["search_points"]), 2)
            self.assertEqual(
                {row["itopk"] for row in deep["search_points"]}, {64, 512}
            )
            self.assertEqual(
                {row["max_iterations"] for row in deep["search_points"]}, {522}
            )
            deep_plan = json.loads(
                (root / "configs" / "deep_plan.json").read_text()
            )
            self.assertEqual(
                deep_plan["pairs"],
                [
                    {
                        "workload": "emis",
                        "method": "default_cagra_accumulator",
                    },
                    {"workload": "yfcc", "method": "default_cagra"},
                ],
            )

            d32_root = root / "d32"
            generate(d32_root, data, "--yfcc-graph-degree", "32")
            d32_manifest = json.loads(
                (
                    d32_root
                    / "configs"
                    / "b0"
                    / "yfcc"
                    / "manifest.json"
                ).read_text()
            )
            self.assertEqual(d32_manifest["graph_degree"], 32)
            d32_config = json.loads(
                Path(d32_manifest["configs"][0]["config"]).read_text()
            )
            self.assertEqual(d32_config["index"][0]["name"], "cagra-g32-ig64")

    def test_analyzer_keeps_repetitions_separate_before_serial_shard_sum(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            make_source_manifests(data)
            generate(root, data)
            write_synthetic_raw(root, "correctness")
            write_synthetic_raw(root, "b0")
            # Give the five YFCC shards unequal valid-GT densities and underfill rates.  The
            # aggregate must weight recall by valid GT slots, while query-rate counters are
            # weighted by shard query count; simply summing shard fractions is incorrect.
            gt_fractions = (1.0, 0.8, 0.6, 0.4, 0.2)
            gt_recalls = (0.1, 0.2, 0.3, 0.4, 0.5)
            underfill_rates = (0.05, 0.1, 0.2, 0.3, 0.4)
            duplicate_rates = (0.01, 0.02, 0.03, 0.04, 0.05)
            for shard_index in range(5):
                raw_path = (
                    root
                    / "raw"
                    / "b0"
                    / "yfcc"
                    / f"shard_{shard_index:02d}.json"
                )
                payload = json.loads(raw_path.read_text())
                for row in payload["benchmarks"]:
                    if (
                        'bitmap_method="default_cagra"' in row["label"]
                        and row["itopk"] == 64
                        and row["search_width"] == 1
                        and row["max_iterations"] == 0
                    ):
                        row["ValidGTRecall"] = gt_recalls[shard_index]
                        row["ValidGTFraction"] = gt_fractions[shard_index]
                        row["UnderfilledQueries"] = underfill_rates[shard_index]
                        row["DuplicateOutputQueries"] = duplicate_rates[
                            shard_index
                        ]
                raw_path.write_text(json.dumps(payload) + "\n")
            env = dict(os.environ, MPLBACKEND="Agg")
            subprocess.run(
                [
                    PYTHON,
                    str(SCRIPT_DIR / "analyze_gpu_graph.py"),
                    "--result-root",
                    str(root),
                    "--require-group",
                    "correctness",
                    "--require-group",
                    "b0",
                    "--no-plots",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            with (root / "analysis" / "summary_points.csv").open() as source:
                rows = list(csv.DictReader(source))
            selected = next(
                row
                for row in rows
                if row["group"] == "b0"
                and row["workload"] == "yfcc"
                and row["method"] == "default_cagra"
                and row["itopk"] == "64"
                and row["search_width"] == "1"
            )
            shard_counts = (2048, 2048, 2048, 2048, 1808)
            base_rates = [
                20_000.0 - 4.0 * 64 - 100.0 * index for index in range(5)
            ]
            expected = 10_000 / sum(
                count / rate for count, rate in zip(shard_counts, base_rates)
            )
            self.assertTrue(
                math.isclose(
                    float(selected["qps_median"]), expected, rel_tol=1e-12
                )
            )
            expected_recall = sum(
                count * fraction * recall
                for count, fraction, recall in zip(
                    shard_counts, gt_fractions, gt_recalls
                )
            ) / sum(
                count * fraction
                for count, fraction in zip(shard_counts, gt_fractions)
            )
            expected_underfill = sum(
                count * rate
                for count, rate in zip(shard_counts, underfill_rates)
            ) / sum(shard_counts)
            expected_duplicate_rate = sum(
                count * rate
                for count, rate in zip(shard_counts, duplicate_rates)
            ) / sum(shard_counts)
            self.assertTrue(
                math.isclose(
                    float(selected["recall_median"]),
                    expected_recall,
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(selected["underfilled_queries_max"]),
                    expected_underfill,
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(selected["duplicate_output_query_rate_max"]),
                    expected_duplicate_rate,
                    rel_tol=1e-12,
                )
            )
            summary = json.loads(
                (root / "analysis" / "summary.json").read_text()
            )
            self.assertEqual(summary["correctness_error_total"], 0)
            self.assertEqual(summary["groups"], ["b0", "correctness"])
            self.assertEqual(summary["max_queries"], 512)
            provenance = json.loads(
                (root / "analysis" / "provenance.json").read_text()
            )
            self.assertEqual(provenance["max_queries"], 512)
            self.assertIn(
                "QPS=10000/sum(shard_seconds)", provenance["timing_contract"]
            )
            self.assertIn("distinct output ID", provenance["recall_contract"])

    def test_analyzer_rejects_filter_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            make_source_manifests(data)
            generate(root, data)
            write_synthetic_raw(root, "b0")
            target = root / "raw" / "b0" / "em" / "shard_00.json"
            payload = json.loads(target.read_text())
            payload["benchmarks"][0]["FilterViolations"] = 1
            target.write_text(json.dumps(payload) + "\n")
            completed = subprocess.run(
                [
                    PYTHON,
                    str(SCRIPT_DIR / "analyze_gpu_graph.py"),
                    "--result-root",
                    str(root),
                    "--require-group",
                    "b0",
                    "--no-plots",
                ],
                env=dict(os.environ, MPLBACKEND="Agg"),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("correctness failure", completed.stderr)

    def test_analyzer_requires_duplicate_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            make_source_manifests(data)
            generate(root, data)
            write_synthetic_raw(root, "b0")
            target = root / "raw" / "b0" / "r" / "shard_00.json"
            payload = json.loads(target.read_text())
            del payload["benchmarks"][0]["DuplicateOutputQueries"]
            target.write_text(json.dumps(payload) + "\n")
            completed = subprocess.run(
                [
                    PYTHON,
                    str(SCRIPT_DIR / "analyze_gpu_graph.py"),
                    "--result-root",
                    str(root),
                    "--require-group",
                    "b0",
                    "--no-plots",
                ],
                env=dict(os.environ, MPLBACKEND="Agg"),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing DuplicateOutputQueries", completed.stderr)

    def test_analyzer_requires_and_validates_distinct_underfill_counters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            make_source_manifests(data)
            generate(root, data)
            write_synthetic_raw(root, "b0")
            target = root / "raw" / "b0" / "r" / "shard_00.json"
            original = json.loads(target.read_text())
            command = [
                PYTHON,
                str(SCRIPT_DIR / "analyze_gpu_graph.py"),
                "--result-root",
                str(root),
                "--require-group",
                "b0",
                "--no-plots",
            ]
            env = dict(os.environ, MPLBACKEND="Agg")

            for key in ("UnderfilledQueries", "MissingResultSlots"):
                payload = copy.deepcopy(original)
                del payload["benchmarks"][0][key]
                target.write_text(json.dumps(payload) + "\n")
                completed = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(f"missing {key}", completed.stderr)

            for key, value in (
                ("UnderfilledQueries", 1.01),
                ("MissingResultSlots", -0.01),
            ):
                payload = copy.deepcopy(original)
                payload["benchmarks"][0][key] = value
                target.write_text(json.dumps(payload) + "\n")
                completed = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("invalid duplicate/underfill", completed.stderr)

            payload = copy.deepcopy(original)
            base = next(
                row
                for row in payload["benchmarks"]
                if 'bitmap_method="default_cagra"' in row["label"]
            )
            base["DuplicateOutputQueries"] = 0.2
            base["UnderfilledQueries"] = 0.1
            target.write_text(json.dumps(payload) + "\n")
            completed = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exceeds unique-underfill", completed.stderr)

    def test_analyzer_reports_native_base_duplicates_but_rejects_new_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            make_source_manifests(data)
            generate(root, data)
            write_synthetic_raw(root, "b0")
            target = root / "raw" / "b0" / "emis" / "shard_00.json"
            payload = json.loads(target.read_text())
            base = next(
                row
                for row in payload["benchmarks"]
                if 'bitmap_method="default_cagra"' in row["label"]
            )
            base["DuplicateOutputQueries"] = 0.031
            base["UnderfilledQueries"] = 0.04
            seeded_base = next(
                row
                for row in payload["benchmarks"]
                if 'bitmap_method="default_cagra_seeded"' in row["label"]
            )
            seeded_base["DuplicateOutputQueries"] = 0.021
            seeded_base["UnderfilledQueries"] = 0.03
            target.write_text(json.dumps(payload) + "\n")
            command = [
                PYTHON,
                str(SCRIPT_DIR / "analyze_gpu_graph.py"),
                "--result-root",
                str(root),
                "--require-group",
                "b0",
                "--no-plots",
            ]
            completed = subprocess.run(
                command,
                env=dict(os.environ, MPLBACKEND="Agg"),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "analysis" / "summary.json").read_text())
            self.assertEqual(summary["correctness_error_total"], 0)
            self.assertGreater(summary["native_base_duplicate_query_rate_max"], 0)

            for method in (
                "default_cagra_accumulator",
                "default_cagra_accumulator_seeded",
                "navix_reference",
            ):
                invalid = copy.deepcopy(payload)
                row = next(
                    item
                    for item in invalid["benchmarks"]
                    if f'bitmap_method="{method}"' in item["label"]
                )
                row["DuplicateOutputQueries"] = 0.001
                row["UnderfilledQueries"] = 0.001
                row["MissingResultSlots"] = 0.001
                target.write_text(json.dumps(invalid) + "\n")
                completed = subprocess.run(
                    command,
                    env=dict(os.environ, MPLBACKEND="Agg"),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("correctness failure", completed.stderr)

            invalid_rate = copy.deepcopy(payload)
            next(
                item
                for item in invalid_rate["benchmarks"]
                if 'bitmap_method="default_cagra"' in item["label"]
            )["DuplicateOutputQueries"] = -0.01
            target.write_text(json.dumps(invalid_rate) + "\n")
            completed = subprocess.run(
                command,
                env=dict(os.environ, MPLBACKEND="Agg"),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid duplicate/underfill", completed.stderr)


if __name__ == "__main__":
    unittest.main()
