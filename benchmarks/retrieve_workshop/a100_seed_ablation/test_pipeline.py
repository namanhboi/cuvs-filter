#!/usr/bin/env python3
"""Focused tests for the adaptive A100 NaviX seed-count ablation."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "seed_ablation.py"


class SeedAblationPipelineTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict[str, str], Path, Path, Path]:
        data = root / "data"
        for phase, counts in (
            ("correctness_1000", [1000]),
            ("throughput_10000", [2048, 2048, 2048, 2048, 1808]),
        ):
            manifest = (
                data / "navix_bitmap_k100/yfcc" / phase / "manifest.json"
            )
            manifest.parent.mkdir(parents=True)
            cursor = 0
            shards = []
            for index, count in enumerate(counts):
                directory = (
                    manifest.parent
                    / f"shard_{index:02d}_{cursor:05d}_{cursor + count:05d}"
                )
                directory.mkdir()
                shards.append(
                    {
                        "directory": str(directory.resolve()),
                        "first_query": cursor,
                        "query_count": count,
                    }
                )
                cursor += count
            manifest.write_text(
                json.dumps({"query_rows": cursor, "shards": shards}) + "\n"
            )

        profile = root / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "seed-test",
                    "max_queries": 2048,
                    "matched_widths": [1, 2],
                    "datasets": {
                        "yfcc": {
                            "bitmap_directory": "navix_bitmap_k100/yfcc",
                            "base_file": "yfcc-10M/base.10M.u8bin",
                            "index_file": "yfcc-10M/cagra_g64_ig128.index",
                            "dtype": "uint8",
                            "dataset_size": 10_000_000,
                            "dimension": 192,
                            "graph_degree": 64,
                            "intermediate_graph_degree": 128,
                        },
                        **{
                            workload: {
                                "bitmap_directory": f"navix_bitmap_k100/arxiv-large/{workload}",
                                "base_file": "arxiv-for-fanns-large/base.fbin",
                                "index_file": "arxiv-for-fanns-large/cagra_g64_ig128.index",
                                "dtype": "float",
                                "dataset_size": 2_735_264,
                                "dimension": 4096,
                                "graph_degree": 64,
                                "intermediate_graph_degree": 128,
                            }
                            for workload in ("em", "emis", "r")
                        },
                    },
                }
            )
            + "\n"
        )
        selected = root / "selected.csv"
        with selected.open("w", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "workload",
                    "method",
                    "selected",
                    "itopk",
                    "search_width",
                    "max_iterations",
                    "recall_median",
                    "qps_median",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "workload": "yfcc",
                    "method": "navix_reference",
                    "selected": True,
                    "itopk": 100,
                    "search_width": 1,
                    "max_iterations": 56,
                    "recall_median": 0.8005,
                    "qps_median": 10_000,
                }
            )
        summary = root / "summary.csv"
        summary.write_text(
            "group,phase,workload,method,max_iterations,recall_median,qps_median\n"
            "b0,throughput,yfcc,navix_reference,0,0.8500,9000\n"
        )
        provenance = root / "provenance.json"
        provenance.write_text(
            json.dumps({"k": 100, "max_queries": 2048}) + "\n"
        )
        env = {
            **os.environ,
            "RETRIEVE_DATASET_PROFILE": str(profile),
            "MPLBACKEND": "Agg",
        }
        return env, data, selected, summary, provenance

    def initialize(self, root: Path) -> tuple[dict[str, str], Path, Path]:
        env, data, selected, summary, provenance = self.fixture(root)
        result = root / "result"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "initialize",
                "--root",
                str(result),
                "--data-root",
                str(data),
                "--reference-selected",
                str(selected),
                "--reference-summary",
                str(summary),
                "--reference-provenance",
                str(provenance),
            ],
            check=True,
            env=env,
        )
        return env, data, result

    @staticmethod
    def write_raw(result: Path, group: str) -> None:
        manifest_path = result / "configs" / group / "yfcc" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        cap = int(manifest["navix_seed_cap"])
        raw = result / "raw" / group / "yfcc"
        raw.mkdir(parents=True)
        for shard in manifest["configs"]:
            config = json.loads(Path(shard["config"]).read_text())
            rows = []
            for repetition in range(int(manifest["repetitions"])):
                for family, point in enumerate(
                    config["index"][0]["search_params"]
                ):
                    itopk = int(point["itopk"])
                    width = int(point["search_width"])
                    iterations = int(point["max_iterations"])
                    if group == "control_s100":
                        recall = 0.8005
                    elif manifest["phase"] == "correctness":
                        recall = 0.75
                    else:
                        recall = (
                            0.79 + (itopk - 128) / 6_400
                            if width == 1
                            else 0.795 + (itopk - 128) / 6_400
                        )
                    rows.append(
                        {
                            "name": f"cagra/{family}",
                            "family_index": family,
                            "run_type": "iteration",
                            "repetition_index": repetition,
                            "itopk": itopk,
                            "search_width": width,
                            "max_iterations": iterations,
                            "k": 100,
                            "max_queries": 2048,
                            "n_queries": int(shard["query_count"]),
                            "navix_bitmap_seeds": 1,
                            "navix_seed_cap": cap,
                            "ValidGTRecall": recall,
                            "ValidGTFraction": 1,
                            "items_per_second": 10_000 + itopk,
                            "FilterViolations": 0,
                            "InvalidSentinelErrors": 0,
                            "SentinelOrderErrors": 0,
                            "InvalidSentinelDistanceErrors": 0,
                            "DuplicateOutputQueries": 0,
                            "UnderfilledQueries": 0,
                            "MissingResultSlots": 0,
                            "label": (
                                'algo="single_cta"#bitmap_method="navix_reference"'
                                '#filter_mode="default"#navix_kernel_variant="reference"'
                                '#navix_mode="adaptive_kuzu"#navix_scheduler="tiled"'
                            ),
                        }
                    )
            output = raw / f"shard_{int(shard['shard_index']):02d}.json"
            output.write_text(json.dumps({"benchmarks": rows}) + "\n")

    def test_generator_separates_seed_cap_from_k_and_plans_integer_midpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, data, result = self.initialize(root)
            cap10 = json.loads(
                (result / "configs/anchors_s10/yfcc/shard_00.json").read_text()
            )["index"][0]["search_params"]
            cap100 = json.loads(
                (
                    result / "configs/control_s100/yfcc/shard_00.json"
                ).read_text()
            )["index"][0]["search_params"]
            self.assertTrue(
                all(int(row["k"]) == 100 for row in cap10 + cap100)
            )
            self.assertTrue(
                all(int(row["navix_seed_cap"]) == 10 for row in cap10)
            )
            self.assertEqual(int(cap100[0]["navix_seed_cap"]), 100)

            for group in (
                "correctness_s10",
                "correctness_s100",
                "control_s100",
                "anchors_s10",
            ):
                self.write_raw(result, group)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "plan-next",
                    "--root",
                    str(result),
                    "--data-root",
                    str(data),
                ],
                check=True,
                env=env,
            )
            state = json.loads(
                (result / "state/calibration_state.json").read_text()
            )
            self.assertFalse(state["complete"])
            self.assertEqual(
                {
                    (row["itopk"], row["search_width"], row["max_iterations"])
                    for row in state["next_points"]
                },
                {(192, 1, 0), (192, 2, 0)},
            )
            manifest = json.loads(
                (
                    result / "configs/calibration_r00/yfcc/manifest.json"
                ).read_text()
            )
            self.assertEqual(manifest["navix_seed_cap"], 10)
            self.assertTrue(
                all(
                    row["navix_seed_cap"] == 10
                    for row in manifest["search_points"]
                )
            )

    def test_raw_validation_rejects_a_mislabeled_seed_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, data, result = self.initialize(root)
            for group in (
                "correctness_s10",
                "correctness_s100",
                "control_s100",
                "anchors_s10",
            ):
                self.write_raw(result, group)
            path = result / "raw/anchors_s10/yfcc/shard_00.json"
            payload = json.loads(path.read_text())
            payload["benchmarks"][0]["navix_seed_cap"] = 100
            path.write_text(json.dumps(payload) + "\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "plan-next",
                    "--root",
                    str(result),
                    "--data-root",
                    str(data),
                ],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "seed-width/runtime contract failed", completed.stderr
            )

    def test_deepening_starts_only_after_l512_b0_fails_and_then_doubles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, data, result = self.initialize(root)
            for group in (
                "correctness_s10",
                "correctness_s100",
                "control_s100",
                "anchors_s10",
            ):
                self.write_raw(result, group)
            for path in (result / "raw/anchors_s10/yfcc").glob("*.json"):
                payload = json.loads(path.read_text())
                for row in payload["benchmarks"]:
                    row["ValidGTRecall"] = 0.7
                path.write_text(json.dumps(payload) + "\n")

            command = [
                sys.executable,
                str(SCRIPT),
                "plan-next",
                "--root",
                str(result),
                "--data-root",
                str(data),
            ]
            subprocess.run(command, check=True, env=env)
            state = json.loads(
                (result / "state/calibration_state.json").read_text()
            )
            self.assertEqual(
                state["next_points"],
                [{"itopk": 512, "search_width": 2, "max_iterations": 522}],
            )

            self.write_raw(result, "calibration_r00")
            for path in (result / "raw/calibration_r00/yfcc").glob("*.json"):
                payload = json.loads(path.read_text())
                for row in payload["benchmarks"]:
                    row["ValidGTRecall"] = 0.75
                path.write_text(json.dumps(payload) + "\n")
            subprocess.run(command, check=True, env=env)
            state = json.loads(
                (result / "state/calibration_state.json").read_text()
            )
            self.assertEqual(
                state["next_points"],
                [
                    {
                        "itopk": 512,
                        "search_width": 2,
                        "max_iterations": 1044,
                    }
                ],
            )

    def test_complete_adaptive_fixture_analyzes_to_two_selected_arms(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, data, result = self.initialize(root)
            for group in (
                "correctness_s10",
                "correctness_s100",
                "control_s100",
                "anchors_s10",
            ):
                self.write_raw(result, group)
            command = [
                sys.executable,
                str(SCRIPT),
                "plan-next",
                "--root",
                str(result),
                "--data-root",
                str(data),
            ]
            for _ in range(16):
                subprocess.run(command, check=True, env=env)
                state = json.loads(
                    (result / "state/calibration_state.json").read_text()
                )
                if state["complete"]:
                    break
                self.write_raw(result, state["next_group"])
            else:
                self.fail("synthetic integer calibration did not converge")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "analyze",
                    "--root",
                    str(result),
                ],
                check=True,
                env=env,
            )
            payload = json.loads(
                (result / "analysis/seed_ablation_results.json").read_text()
            )
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(
                {int(row["seed_cap"]) for row in payload["selected"]},
                {10, 100},
            )
            self.assertEqual(
                next(
                    row["status"]
                    for row in payload["selected"]
                    if int(row["seed_cap"]) == 10
                ),
                "matched",
            )


if __name__ == "__main__":
    unittest.main()
