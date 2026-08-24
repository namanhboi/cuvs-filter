#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("matched_recall.py")
SPEC = importlib.util.spec_from_file_location("matched_recall", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MatchedRecallTest(unittest.TestCase):
    def test_b0_formula_and_l_limit(self) -> None:
        self.assertEqual(MODULE.resolved_b0(32, 1, 10_000_000, 64), 37)
        self.assertEqual(MODULE.resolved_b0(512, 4, 10_000_000, 64), 133)
        self.assertEqual(MODULE.resolved_b0(512, 1, 100_000, 32), 517)
        with self.assertRaises(ValueError):
            MODULE.resolved_b0(544, 1, 100_000, 32)

    def test_requested_l_is_rounded_only_for_capacity(self) -> None:
        self.assertEqual(MODULE.internal_itopk(10), 32)
        self.assertEqual(MODULE.internal_itopk(65), 96)
        self.assertEqual(MODULE.resolved_b0(65, 1, 100_000, 32), 70)
        self.assertEqual(MODULE.resolved_b0(65, 2, 100_000, 32), 37)
        self.assertEqual(MODULE.hash_nodes("em", "default_cagra", 65, 1, 0), 96 + 32 * 70)
        MODULE.normalize_point(
            {
                "workload": "em",
                "method": "default_cagra",
                "itopk": 65,
                "search_width": 1,
                "max_iterations": 0,
            }
        )

    def test_tight_refinement_uses_only_w1_w2_and_deduplicates_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "baseline.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("stage", "workload", "method", "itopk", "search_width", "max_iterations", "recall"),
                )
                writer.writeheader()
                for workload, method in MODULE.TIGHT_PAIRS:
                    goal = MODULE.target(workload)
                    for width in MODULE.TUNING_WIDTHS:
                        writer.writerow(
                            {
                                "stage": "calibration",
                                "workload": workload,
                                "method": method,
                                "itopk": 32,
                                "search_width": width,
                                "max_iterations": 0,
                                "recall": goal - 0.01,
                            }
                        )
                        writer.writerow(
                            {
                                "stage": "calibration",
                                "workload": workload,
                                "method": method,
                                "itopk": 64,
                                "search_width": width,
                                "max_iterations": 0,
                                "recall": goal + 0.01,
                            }
                        )
            payload = MODULE.tight_refinement_points(path)
        self.assertTrue(payload["points"])
        self.assertEqual({point["search_width"] for point in payload["points"]}, {1, 2})
        fingerprints = [
            (
                point["workload"],
                point["method"],
                point["search_width"],
                MODULE.execution_fingerprint(
                    point["workload"], point["itopk"], point["search_width"]
                ),
            )
            for point in payload["points"]
        ]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_navix_refinement_exhausts_distinct_below_l32_b0_executions(self) -> None:
        payload = MODULE.navix_refinement_points()
        points = payload["points"]
        self.assertEqual(len(points), 66)
        self.assertEqual(
            {(point["workload"], point["method"]) for point in points},
            {("em", "navix_reference"), ("r", "navix_reference")},
        )
        self.assertTrue(all(point["max_iterations"] == 0 for point in points))
        self.assertTrue(all(10 <= point["itopk"] <= 31 for point in points))
        fingerprints = [
            (
                point["workload"],
                point["search_width"],
                MODULE.execution_fingerprint(
                    point["workload"], point["itopk"], point["search_width"]
                ),
            )
            for point in points
        ]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_navix_refinement_selects_only_fast_in_window_finalists(self) -> None:
        rows = []
        for point in MODULE.navix_refinement_points()["points"]:
            recall = 0.94
            if point["search_width"] == 1 and point["itopk"] in (30, 31):
                recall = 0.9505 + (point["itopk"] - 30) * 0.0005
            rows.append(
                {
                    **point,
                    "recall_min": recall,
                    "recall_median": recall,
                    "qps_median": 100_000 - point["itopk"],
                }
            )
        payload = MODULE.select_navix_refinement_finalists(rows)
        self.assertEqual(len(payload["points"]), 4)
        self.assertEqual(
            {(point["workload"], point["itopk"]) for point in payload["points"]},
            {("em", 30), ("em", 31), ("r", 30), ("r", 31)},
        )

        with self.assertRaisesRegex(ValueError, "missing"):
            MODULE.select_navix_refinement_finalists(rows[:-1])

        for row in rows:
            row["recall_min"] = 0.94
            row["recall_median"] = 0.94
        with self.assertRaisesRegex(ValueError, "no em NaviX calibration point"):
            MODULE.select_navix_refinement_finalists(rows)

    def test_raw_output_validator_rejects_incomplete_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            raw = root / "raw.json"
            config.write_text(
                json.dumps(
                    {
                        "index": [
                            {"search_params": [{"itopk": 30}, {"itopk": 31}]}
                        ]
                    }
                )
            )
            observations = []
            for name in ("benchmark/0", "benchmark/1"):
                for repetition in range(3):
                    observations.append(
                        {
                            "name": name,
                            "run_type": "iteration",
                            "repetition_index": repetition,
                        }
                    )
            observations.append(
                {"name": "benchmark/0_mean", "run_type": "aggregate"}
            )
            raw.write_text(json.dumps({"benchmarks": observations}))
            self.assertEqual(
                MODULE.validate_raw_output(raw, config, 3)["iteration_rows"], 6
            )
            raw.write_text(json.dumps({"benchmarks": observations[:-2]}))
            with self.assertRaisesRegex(ValueError, "repetitions"):
                MODULE.validate_raw_output(raw, config, 3)

    def test_navix_refinement_final_validation_is_strict(self) -> None:
        selected = []
        for workload in MODULE.NAVIX_REFINEMENT_WORKLOADS:
            selected.append(
                {
                    "workload": workload,
                    "method": "navix_reference",
                    "target_reached": True,
                    "within_target_window": True,
                    "itopk": 30,
                    "search_width": 1,
                    "max_iterations": 0,
                    "filter_violations": 0,
                    "sentinel_errors": 0,
                    "duplicate_output_query_rate_max": 0,
                }
            )
        with patch.object(MODULE, "summarize_final", return_value=([], selected)):
            payload = MODULE.validate_navix_refinement(Path("unused"))
        self.assertEqual(len(payload["selected"]), 2)

        selected[0]["within_target_window"] = False
        with patch.object(MODULE, "summarize_final", return_value=([], selected)):
            with self.assertRaisesRegex(ValueError, "strict, correct B0 match"):
                MODULE.validate_navix_refinement(Path("unused"))

    def test_next_starts_with_complete_b0_anchor_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reason, points = MODULE.next_points(Path(temporary))
        self.assertEqual(reason, "b0_anchors")
        self.assertEqual(
            len(points),
            len(MODULE.WORKLOADS)
            * len(MODULE.METHODS)
            * len(MODULE.WIDTHS)
            * len(MODULE.ANCHOR_L),
        )
        self.assertTrue(all(point["max_iterations"] == 0 for point in points))

    def test_explicit_depth_must_exceed_b0(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.normalize_point(
                {
                    "workload": "yfcc",
                    "method": "default_cagra",
                    "itopk": 64,
                    "search_width": 1,
                    "max_iterations": 69,
                }
            )

    def test_explicit_depth_respects_single_cta_hash_capacity(self) -> None:
        # 512 + 4 * 32 * 4092 = 524288: exactly the 20-bit table's 0.5-fill limit.
        MODULE.normalize_point(
            {
                "workload": "emis",
                "method": "default_cagra",
                "itopk": 512,
                "search_width": 4,
                "max_iterations": 4092,
            }
        )
        with self.assertRaises(ValueError):
            MODULE.normalize_point(
                {
                    "workload": "emis",
                    "method": "default_cagra",
                    "itopk": 512,
                    "search_width": 4,
                    "max_iterations": 4093,
                }
            )

        # Degree 64 halves the legal W=4 continuation depth on YFCC.
        MODULE.normalize_point(
            {
                "workload": "yfcc",
                "method": "default_cagra_accumulator",
                "itopk": 512,
                "search_width": 4,
                "max_iterations": 2046,
            }
        )
        with self.assertRaises(ValueError):
            MODULE.normalize_point(
                {
                    "workload": "yfcc",
                    "method": "default_cagra_accumulator",
                    "itopk": 512,
                    "search_width": 4,
                    "max_iterations": 2047,
                }
            )


if __name__ == "__main__":
    unittest.main()
