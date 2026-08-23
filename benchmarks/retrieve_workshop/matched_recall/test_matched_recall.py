#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

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
