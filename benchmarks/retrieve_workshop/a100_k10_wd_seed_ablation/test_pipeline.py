#!/usr/bin/env python3
"""Focused tests for the k=10 YFCC W*D seed-ablation pipeline."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "wd_seed_ablation.py"
REFERENCE_QPS = 23_486.56676830371


class WDSeedAblationPipelineTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict[str, str], Path, Path, Path]:
        data = root / "data"
        for phase, counts in (
            ("correctness_1000", [1000]),
            ("throughput_10000", [2048, 2048, 2048, 2048, 1808]),
        ):
            manifest = data / "navix_bitmap/yfcc" / phase / "manifest.json"
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
                    "name": "k10-wd-seed-test",
                    "max_queries": 2048,
                    "matched_widths": [1, 2],
                    "datasets": {
                        "yfcc": {
                            "bitmap_directory": "navix_bitmap/yfcc",
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
                                "bitmap_directory": f"navix_bitmap/arxiv-large/{workload}",
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
                    "recall_min",
                    "qps_median",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "workload": "yfcc",
                    "method": "navix_reference",
                    "selected": True,
                    "itopk": 129,
                    "search_width": 2,
                    "max_iterations": 0,
                    "recall_median": 0.80005,
                    "recall_min": 0.8,
                    "qps_median": REFERENCE_QPS,
                }
            )
        provenance = root / "provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "experiment": "retrieve_workshop_matched_recall",
                    "max_queries": 2048,
                    "targets": {
                        "yfcc": 0.8,
                        "em": 0.95,
                        "emis": 0.95,
                        "r": 0.95,
                    },
                }
            )
            + "\n"
        )
        env = {
            **os.environ,
            "RETRIEVE_DATASET_PROFILE": str(profile),
            "MPLBACKEND": "Agg",
        }
        return env, data, selected, provenance

    def initialize(self, root: Path) -> tuple[dict[str, str], Path, Path]:
        env, data, selected, provenance = self.fixture(root)
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
                "--reference-provenance",
                str(provenance),
            ],
            check=True,
            env=env,
        )
        return env, data, result

    @staticmethod
    def recall(
        group: str, cap: int, itopk: int, width: int, iterations: int
    ) -> float:
        if group == "paired_incumbent":
            return 0.80005 if cap == 10 else 0.8200
        if group == "correctness":
            return 0.75
        if cap == 10:
            return max(0.0, 0.7200 + itopk / 10_000)
        base = (
            0.749 + itopk * 0.0004 if width == 1 else 0.759 + itopk * 0.00032
        )
        if iterations > 0:
            resolved = itopk // width + 5
            base *= min(1.0, iterations / resolved)
        return min(base, 0.999)

    @classmethod
    def write_raw(
        cls, result: Path, group: str, forced_recall: float | None = None
    ) -> None:
        manifest_path = result / "configs" / group / "yfcc" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
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
                    cap = int(point["navix_seed_cap"])
                    recall = (
                        forced_recall
                        if forced_recall is not None
                        else cls.recall(group, cap, itopk, width, iterations)
                    )
                    qps = (
                        REFERENCE_QPS
                        if group == "paired_incumbent" and cap == 10
                        else 60_000
                        - itopk * 100
                        - iterations
                        + (2_000 if width == 1 else 0)
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
                            "k": 10,
                            "max_queries": 2048,
                            "n_queries": int(shard["query_count"]),
                            "navix_bitmap_seeds": 1,
                            "navix_seed_cap": cap,
                            "ValidGTRecall": recall,
                            "ValidGTFraction": 1,
                            "items_per_second": qps,
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
            (raw / f"shard_{int(shard['shard_index']):02d}.json").write_text(
                json.dumps({"benchmarks": rows}) + "\n"
            )

    def command(
        self, env: dict[str, str], data: Path, result: Path, name: str
    ) -> None:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                name,
                "--root",
                str(result),
                "--data-root",
                str(data),
            ],
            check=True,
            env=env,
        )

    def test_generator_uses_k10_and_width_dependent_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env, data, result = self.initialize(Path(temporary))
            del env, data
            config = json.loads(
                (result / "configs/anchors_wd/yfcc/shard_00.json").read_text()
            )
            rows = config["index"][0]["search_params"]
            self.assertEqual(int(config["search_basic_param"]["k"]), 10)
            self.assertTrue(all(int(row.get("k", 10)) == 10 for row in rows))
            self.assertEqual(
                {
                    int(row["navix_seed_cap"])
                    for row in rows
                    if int(row["search_width"]) == 1
                },
                {64},
            )
            self.assertEqual(
                {
                    int(row["navix_seed_cap"])
                    for row in rows
                    if int(row["search_width"]) == 2
                },
                {128},
            )

    def test_deep_fallback_is_planned_when_b0_cannot_reach_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env, data, result = self.initialize(Path(temporary))
            for group in ("correctness", "paired_incumbent"):
                self.write_raw(result, group)
            self.write_raw(result, "anchors_wd", forced_recall=0.7)
            self.command(env, data, result, "plan-next")
            state = json.loads(
                (result / "state/calibration_state.json").read_text()
            )
            self.assertEqual(
                {
                    (row["search_width"], row["itopk"])
                    for row in state["next_points"]
                },
                {(1, 512), (2, 512)},
            )
            self.assertTrue(
                all(row["max_iterations"] > 0 for row in state["next_points"])
            )

    def test_adaptive_fixture_produces_matched_target_and_paired_controls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env, data, result = self.initialize(Path(temporary))
            for group in ("correctness", "paired_incumbent", "anchors_wd"):
                self.write_raw(result, group)
            for _ in range(24):
                self.command(env, data, result, "plan-next")
                state = json.loads(
                    (result / "state/calibration_state.json").read_text()
                )
                if state["complete"]:
                    break
                self.write_raw(result, state["next_group"])
            else:
                self.fail("adaptive calibration did not converge")
            self.command(env, data, result, "prepare-finalists")
            self.write_raw(result, "finalists_wd")
            self.command(env, data, result, "prepare-controls")
            self.write_raw(result, "paired_winner")
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
                (result / "analysis/results.json").read_text()
            )
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(
                payload["target_comparison"][1]["status"], "matched"
            )
            self.assertEqual(len(payload["paired_controls"]), 2)
            self.assertFalse(payload["automatic_promotion"])

    def test_raw_validation_rejects_mislabeled_seed_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env, data, result = self.initialize(Path(temporary))
            del data
            for group in ("correctness", "paired_incumbent", "anchors_wd"):
                self.write_raw(result, group)
            path = result / "raw/anchors_wd/yfcc/shard_00.json"
            payload = json.loads(path.read_text())
            payload["benchmarks"][0]["navix_seed_cap"] = 63
            path.write_text(json.dumps(payload) + "\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "plan-next",
                    "--root",
                    str(result),
                    "--data-root",
                    str(result.parent / "data"),
                ],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unexpected/duplicate point", completed.stderr)


if __name__ == "__main__":
    unittest.main()
