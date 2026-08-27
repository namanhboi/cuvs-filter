#!/usr/bin/env python3
"""Synthetic contract tests for the matched W*D follow-up campaigns."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSHOP = SCRIPT_DIR.parent
GENERATOR = WORKSHOP / "gpu_graph" / "generate_configs.py"
WORKFLOW = SCRIPT_DIR / "workflow.py"
WORKLOADS = ("yfcc", "em", "emis", "r")

SPEC = importlib.util.spec_from_file_location(
    "a100_wd_followup_workflow", WORKFLOW
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PipelineTest(unittest.TestCase):
    def fixture(self, base: Path) -> tuple[Path, Path]:
        data = base / "data"
        profile = base / "profile.json"
        datasets: dict[str, dict[str, object]] = {}
        for workload in WORKLOADS:
            bitmap = f"bitmap/{workload}"
            spec = {
                "bitmap_directory": bitmap,
                "base_file": f"datasets/{workload}/base.bin",
                "index_file": f"datasets/{workload}/graph.index",
                "dtype": "uint8" if workload == "yfcc" else "float",
                "dataset_size": 10_000_000
                if workload == "yfcc"
                else 2_735_264,
                "dimension": 192 if workload == "yfcc" else 4_096,
                "graph_degree": 64,
                "intermediate_graph_degree": 128,
            }
            datasets[workload] = spec
            for key in ("base_file", "index_file"):
                path = data / str(spec[key])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            for phase, counts in (
                ("correctness_1000", (1_000,)),
                ("throughput_10000", (2_048, 2_048, 2_048, 2_048, 1_808)),
            ):
                manifest_root = data / bitmap / phase
                shards = []
                first = 0
                for index, count in enumerate(counts):
                    shard = manifest_root / f"shard_{index:02d}"
                    shard.mkdir(parents=True, exist_ok=True)
                    for name in (
                        "query.bin",
                        "groundtruth.ibin",
                        "filter.bitmap",
                    ):
                        (shard / name).write_bytes(b"fixture")
                    shards.append(
                        {
                            "shard_index": index,
                            "first_query": first,
                            "query_count": count,
                            "directory": str(shard.resolve()),
                        }
                    )
                    first += count
                (manifest_root / "manifest.json").write_text(
                    json.dumps({"query_rows": first, "shards": shards}) + "\n"
                )
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "synthetic-a100-wd-followup",
                    "max_queries": 2_048,
                    "matched_widths": [1, 2],
                    "datasets": datasets,
                },
                indent=2,
            )
            + "\n"
        )
        return data, profile

    @staticmethod
    def environment(profile: Path) -> dict[str, str]:
        return {**os.environ, "RETRIEVE_DATASET_PROFILE": str(profile)}

    def test_generator_builds_exact_wd_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, profile = self.fixture(base)
            k10 = base / "k10"
            subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--output",
                    str(k10),
                    "--data-root",
                    str(data),
                    "--methods",
                    "default_cagra_seeded,default_cagra_accumulator_seeded,navix_reference",
                    "--seed-policy",
                    "wd",
                ],
                check=True,
                env=self.environment(profile),
            )
            for workload in WORKLOADS:
                manifest = json.loads(
                    (k10 / "b0" / workload / "manifest.json").read_text()
                )
                self.assertEqual(manifest["passing_seed_policy"], "wd")
                self.assertEqual(len(manifest["search_points"]), 18)
                self.assertEqual(
                    {row["method"] for row in manifest["search_points"]},
                    {
                        "default_cagra_seeded",
                        "default_cagra_accumulator_seeded",
                        "navix_reference",
                    },
                )
                for row in manifest["search_points"]:
                    cap_key = (
                        "navix_seed_cap"
                        if row["method"] == "navix_reference"
                        else "cagra_seed_cap"
                    )
                    self.assertEqual(
                        int(row[cap_key]), int(row["search_width"]) * 64
                    )

            k100 = base / "k100"
            subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--output",
                    str(k100),
                    "--data-root",
                    str(data),
                    "--k",
                    "100",
                    "--cartesian-b0",
                    "--methods",
                    "navix_reference",
                    "--seed-policy",
                    "wd",
                ],
                check=True,
                env=self.environment(profile),
            )
            for workload in WORKLOADS:
                manifest = json.loads(
                    (k100 / "b0" / workload / "manifest.json").read_text()
                )
                self.assertEqual(len(manifest["search_points"]), 6)
                self.assertEqual(
                    {
                        (int(row["itopk"]), int(row["search_width"]))
                        for row in manifest["search_points"]
                    },
                    {
                        (128, 1),
                        (128, 2),
                        (256, 1),
                        (256, 2),
                        (512, 1),
                        (512, 2),
                    },
                )
                self.assertEqual(
                    {
                        int(row["navix_seed_cap"])
                        for row in manifest["search_points"]
                    },
                    {64, 128},
                )

    def write_selected(
        self, path: Path, coordinates: dict[str, tuple[int, int]]
    ) -> None:
        fields = [
            "workload",
            "method",
            "itopk",
            "search_width",
            "max_iterations",
            "recall_median",
            "recall_min",
            "qps_median",
            "target_reached",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            for workload in WORKLOADS:
                itopk, width = coordinates[workload]
                writer.writerow(
                    {
                        "workload": workload,
                        "method": "navix_reference",
                        "itopk": itopk,
                        "search_width": width,
                        "max_iterations": 0,
                        "recall_median": 0.951,
                        "recall_min": 0.950,
                        "qps_median": 10_000,
                        "target_reached": True,
                    }
                )

    def test_k100_controls_pair_old_and_new_coordinates_immutably(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data, profile = self.fixture(base)
            reference_tree = base / "reference/paper_gpu_bundle_k100_matched"
            old_selected = (
                reference_tree / "matched_recall/selected_points.csv"
            )
            self.write_selected(
                old_selected,
                {
                    "yfcc": (128, 1),
                    "em": (256, 1),
                    "emis": (256, 2),
                    "r": (512, 2),
                },
            )
            archive = base / "reference.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(reference_tree, arcname=reference_tree.name)
            new_selected = base / "new_selected.csv"
            self.write_selected(
                new_selected,
                {
                    "yfcc": (128, 2),
                    "em": (256, 2),
                    "emis": (512, 1),
                    "r": (512, 1),
                },
            )
            root = base / "run"
            command = [
                sys.executable,
                str(WORKFLOW),
                "create-k100-controls",
                "--root",
                str(root),
                "--data-root",
                str(data),
                "--reference-bundle",
                str(archive),
                "--new-selected",
                str(new_selected),
            ]
            subprocess.run(command, check=True, env=self.environment(profile))
            manifests = sorted(
                (root / "controls/configs").glob("*/*/manifest.json")
            )
            self.assertEqual(len(manifests), 8)
            for path in manifests:
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest["repetitions"], 3)
                self.assertEqual(manifest["expected_queries"], 10_000)
                self.assertEqual(len(manifest["search_points"]), 2)
                width = int(manifest["search_points"][0]["search_width"])
                self.assertEqual(
                    {
                        (row["seed_policy"], int(row["navix_seed_cap"]))
                        for row in manifest["search_points"]
                    },
                    {("k", 100), ("wd", width * 64)},
                )

                destination = (
                    root
                    / "controls/raw"
                    / manifest["group"]
                    / manifest["workload"]
                )
                destination.mkdir(parents=True, exist_ok=True)
                for shard in manifest["configs"]:
                    rows = []
                    for repetition in range(3):
                        for point in manifest["search_points"]:
                            rows.append(
                                {
                                    "name": "synthetic",
                                    "run_type": "iteration",
                                    "repetition_index": repetition,
                                    "n_queries": int(shard["query_count"]),
                                    "items_per_second": 10_000.0,
                                    "ValidGTRecall": 0.951,
                                    "ValidGTFraction": 1.0,
                                    "search_width": int(point["search_width"]),
                                    "navix_seed_cap": int(
                                        point["navix_seed_cap"]
                                    ),
                                    "FilterViolations": 0,
                                    "InvalidSentinelErrors": 0,
                                    "SentinelOrderErrors": 0,
                                    "InvalidSentinelDistanceErrors": 0,
                                    "DuplicateOutputQueries": 0,
                                    "label": 'bitmap_method="navix_reference"',
                                }
                            )
                    raw = (
                        destination
                        / f"shard_{int(shard['shard_index']):02d}.json"
                    )
                    raw.write_text(json.dumps({"benchmarks": rows}) + "\n")
            summaries = MODULE.analyze_k100_controls(root)
            self.assertEqual(len(summaries), 16)
            self.assertEqual(
                {
                    (row["seed_policy"], int(row["seed_cap"]))
                    for row in summaries
                },
                {("k", 100), ("wd", 64), ("wd", 128)},
            )

            broken = root / "controls/raw/paired_old/yfcc/shard_00.json"
            payload = json.loads(broken.read_text())
            payload["benchmarks"][0]["SentinelOrderErrors"] = 1
            broken.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "correctness error total"):
                MODULE.analyze_k100_controls(root)
            failed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=self.environment(profile),
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(
                "immutable controls already initialized", failed.stderr
            )


if __name__ == "__main__":
    unittest.main()
