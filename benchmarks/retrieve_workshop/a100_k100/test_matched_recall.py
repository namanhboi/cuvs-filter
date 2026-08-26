#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR.parent / "matched_recall" / "matched_recall.py"
PROFILE = SCRIPT_DIR / "profiles" / "a100_yfcc10m_arxiv_large_k100.json"

with patch.dict(
    os.environ,
    {
        "RETRIEVE_DATASET_PROFILE": str(PROFILE),
        "RETRIEVE_MATCHED_K": "100",
        "RETRIEVE_MATCHED_ALLOW_SHALLOW_NAVIX": "1",
    },
):
    SPEC = importlib.util.spec_from_file_location("matched_recall_k100", MODULE_PATH)
    assert SPEC and SPEC.loader
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)


def calibration_row(
    workload: str,
    method: str,
    itopk: int,
    width: int,
    maximum: int,
    recall: float,
    qps: float = 1000.0,
) -> dict[str, object]:
    return {
        "group": "synthetic",
        "stage": "calibration",
        "workload": workload,
        "graph_degree": 64,
        "intermediate_graph_degree": 128,
        "method": method,
        "itopk": itopk,
        "search_width": width,
        "max_iterations": maximum,
        "resolved_iterations": MODULE.resolved_iterations(
            workload, itopk, width, maximum
        ),
        "repetition_index": 0,
        "shards": 5,
        "queries": 10_000,
        "recall": recall,
        "valid_gt_fraction": 1.0,
        "qps": qps,
        "seconds": 10.0,
        "filter_violations": 0.0,
        "sentinel_errors": 0.0,
        "duplicate_output_query_rate": 0.0,
        "underfilled_queries": 0.0,
        "missing_result_slots": 0.0,
    }


class K100MatchedRecallTest(unittest.TestCase):
    def test_k100_contract_and_shallow_validation(self) -> None:
        self.assertEqual(MODULE.RESULT_K, 100)
        self.assertEqual(MODULE.MIN_L, 100)
        self.assertEqual(MODULE.ANCHOR_L, (100, 128, 256, 512))
        self.assertEqual(MODULE.resolved_b0(100, 1, 10_000_000, 64), 105)
        self.assertEqual(MODULE.resolved_b0(100, 2, 2_735_264, 64), 55)
        MODULE.normalize_point(
            {
                "workload": "yfcc",
                "method": "navix_reference",
                "itopk": 100,
                "search_width": 1,
                "max_iterations": 1,
            }
        )
        for method, itopk in (("default_cagra", 100), ("navix_reference", 128)):
            with self.assertRaisesRegex(ValueError, "reserved for minimum-L NaviX"):
                MODULE.normalize_point(
                    {
                        "workload": "yfcc",
                        "method": method,
                        "itopk": itopk,
                        "search_width": 1,
                        "max_iterations": 1,
                    }
                )

    def test_imported_b0_anchors_leave_only_l100_missing(self) -> None:
        fields = (
            "group",
            "phase",
            "workload",
            "graph_degree",
            "intermediate_graph_degree",
            "method",
            "itopk",
            "search_width",
            "max_iterations",
            "repetitions",
            "shards_per_repetition",
            "queries_per_repetition",
            "recall_median",
            "recall_min",
            "recall_max",
            "valid_gt_fraction_min",
            "qps_median",
            "qps_min",
            "qps_max",
            "seconds_median",
            "filter_violations",
            "sentinel_errors",
            "duplicate_output_query_rate_max",
            "underfilled_queries_max",
            "missing_result_slots_max",
            "paper_included",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            provenance = root / "provenance.json"
            with summary.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for workload in MODULE.WORKLOADS:
                    for method in MODULE.METHODS:
                        for itopk in (128, 256, 512):
                            for width in MODULE.WIDTHS:
                                writer.writerow(
                                    {
                                        "group": "b0",
                                        "phase": "throughput",
                                        "workload": workload,
                                        "graph_degree": 64,
                                        "intermediate_graph_degree": 128,
                                        "method": method,
                                        "itopk": itopk,
                                        "search_width": width,
                                        "max_iterations": 0,
                                        "repetitions": 3,
                                        "shards_per_repetition": 5,
                                        "queries_per_repetition": 10_000,
                                        "recall_median": 0.5,
                                        "recall_min": 0.5,
                                        "recall_max": 0.5,
                                        "valid_gt_fraction_min": 1.0,
                                        "qps_median": 1000,
                                        "qps_min": 990,
                                        "qps_max": 1010,
                                        "seconds_median": 10,
                                        "filter_violations": 0,
                                        "sentinel_errors": 0,
                                        "duplicate_output_query_rate_max": 0,
                                        "underfilled_queries_max": 0,
                                        "missing_result_slots_max": 0,
                                        "paper_included": True,
                                    }
                                )
            provenance.write_text(
                json.dumps(
                    {
                        "fixed_contract": {
                            "k": 100,
                            "max_queries": 2048,
                            "output_set_semantics": "distinct_valid_output_ids_v1",
                        }
                    }
                )
            )
            payload = MODULE.import_baseline(root, summary, provenance)
            self.assertEqual(payload["points"], 72)
            reason, points = MODULE.next_points(root)
        self.assertEqual(reason, "b0_anchors")
        self.assertEqual(len(points), 24)
        self.assertEqual({point["itopk"] for point in points}, {100})

    def test_shallow_navix_refinement_precedes_deep_search(self) -> None:
        rows: list[dict[str, object]] = []
        for workload in MODULE.WORKLOADS:
            goal = MODULE.target(workload)
            for method in MODULE.METHODS:
                for width in MODULE.WIDTHS:
                    for itopk in MODULE.ANCHOR_L:
                        recall = goal + 0.03 if method == "navix_reference" else goal - 0.20
                        rows.append(
                            calibration_row(
                                workload, method, itopk, width, 0, recall
                            )
                        )
        with patch.object(MODULE, "calibration_rows", return_value=rows):
            reason, points = MODULE.next_points(Path("unused"))
        self.assertEqual(reason, "shallow_navix_refinement")
        self.assertEqual(len(points), 8)
        self.assertTrue(all(point["itopk"] == 100 for point in points))
        self.assertTrue(all(point["max_iterations"] == 1 for point in points))

    def test_finalist_selection_prefers_strict_shallow_match(self) -> None:
        rows: list[dict[str, object]] = []
        for workload in MODULE.WORKLOADS:
            goal = MODULE.target(workload)
            for method in MODULE.METHODS:
                for width in MODULE.WIDTHS:
                    rows.append(
                        calibration_row(
                            workload,
                            method,
                            100,
                            width,
                            0,
                            goal + (0.03 if method == "navix_reference" else 0.001),
                            1000 + width,
                        )
                    )
                    if method == "navix_reference":
                        rows.append(
                            calibration_row(
                                workload,
                                method,
                                100,
                                width,
                                7,
                                goal + 0.001,
                                2000 + width,
                            )
                        )
        with patch.object(MODULE, "calibration_rows", return_value=rows):
            payload = MODULE.candidate_selection(Path("unused"))
        navix = [
            point
            for point in payload["points"]
            if point["method"] == "navix_reference"
        ]
        self.assertEqual(len(navix), 8)
        self.assertTrue(all(point["max_iterations"] == 7 for point in navix))
        decisions = [
            row
            for row in payload["decisions"]
            if row["method"] == "navix_reference"
        ]
        self.assertTrue(
            all("shallow iteration tuning" in row["eligibility_rule"] for row in decisions)
        )


if __name__ == "__main__":
    unittest.main()
