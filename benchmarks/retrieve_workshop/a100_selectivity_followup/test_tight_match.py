#!/usr/bin/env python3
"""CPU checks for strict selectivity matched-recall tuning."""

from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

import tight_match as match


class TightMatchTest(unittest.TestCase):
    def test_five_negative_cohorts_are_an_isolated_matched_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {
                f"synthetic_correlation_{level.replace('.', 'p')}_negative"
                for level in match.study.CORRELATION_SELECTIVITIES
            }
            for cohort in expected:
                path = root / "data" / cohort
                path.mkdir(parents=True)
                (path / "manifest.json").write_text("{}")
            self.assertEqual(set(match.cohorts(root)), expected)
            extra = root / "data/synthetic_correlation_0p10_positive"
            extra.mkdir()
            (extra / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "expected 21 legacy cohorts or five negative"):
                match.cohorts(root)

    def test_every_final_repetition_must_be_inside_closed_window(self) -> None:
        self.assertTrue(match.in_band([0.950, 0.9505, 0.951]))
        self.assertTrue(match.in_band([0.9499999999999998, 0.9505, 0.951]))
        self.assertFalse(match.in_band([0.9499, 0.9505, 0.9505]))
        self.assertFalse(match.in_band([0.950, 0.9505, 0.95101]))
        self.assertFalse(match.in_band([0.950, 0.951]))
        with patch.object(match, "HIGH", 0.952):
            self.assertTrue(match.in_band([0.950, 0.9515, 0.952]))
            self.assertFalse(match.in_band([0.950, 0.95201, 0.952]))

    def test_external_calibration_rows_are_used_for_new_result_root(self) -> None:
        root, reference, calibration = Path('/result'), Path('/reference'), Path('/calibration')
        old = {('cohort', 'default_cagra', 10, 1, 0): {'recall': 0.9515}}
        with patch.object(match, 'baseline_rows', return_value={}), patch.object(
            match, 'new_rows', side_effect=[old, {}]
        ) as load:
            self.assertEqual(match.observations(root, reference, calibration), old)
        self.assertEqual([call.args[0] for call in load.call_args_list], [calibration, root])

    def test_requested_l_changes_auto_budget_before_capacity_rounding(self) -> None:
        self.assertEqual(match.automatic_iterations(65, 1), 70)
        self.assertEqual(match.fingerprint(65, 1, 0), (96, 1, 70))
        self.assertEqual(match.fingerprint(66, 1, 0), (96, 1, 71))
        self.assertEqual(match.fingerprint(66, 2, 0), match.fingerprint(67, 2, 0))
        self.assertFalse(match.legal("navix_reference", 512, 2, 4092))

    def test_l_refinement_covers_distinct_executions_in_target_bracket(self) -> None:
        name, method = "synthetic_correlation_0p10_random", "default_cagra"
        rows = {
            (name, method, 64, 1, 0): {"recall": 0.867},
            (name, method, 128, 1, 0): {"recall": 0.976},
        }
        points = match.refine_l([name], rows)
        self.assertIn((name, method, 65, 1, 0), points)
        self.assertIn((name, method, 127, 1, 0), points)
        self.assertEqual(len([p for p in points if p[1:3] == (method, 65)]), 1)

    def test_shallow_iteration_sweep_handles_overshooting_minimum_l(self) -> None:
        name, method = "synthetic_correlation_0p99_positive", "navix_reference"
        rows = {(name, method, 10, 1, 0): {"recall": 0.985}}
        points = match.iteration_seeds([name], rows)
        self.assertEqual(
            {p[4] for p in points if p[:4] == (name, method, 10, 1)},
            set(range(1, 15)),
        )

    def test_iteration_refinement_exhausts_small_crossing_gap(self) -> None:
        name, method = "synthetic_correlation_0p10_positive", "navix_reference"
        rows = {
            (name, method, 10, 1, 1): {"recall": 0.94},
            (name, method, 10, 1, 15): {"recall": 0.98},
        }
        points = match.refine_iterations([name], rows)
        self.assertEqual({p[4] for p in points}, set(range(2, 15)))

    def test_iteration_refinement_stops_after_one_in_band_candidate(self) -> None:
        name, method = "real_em_low", "navix_reference"
        rows = {
            (name, method, 10, 1, 1): {"recall": 0.94, "recalls": [0.94]},
            (name, method, 10, 1, 15): {"recall": 0.98, "recalls": [0.98]},
        }
        rows[(name, method, 20, 1, 0)] = {
            "recall": 0.9505, "recalls": [0.9505]
        }
        self.assertEqual(match.refine_iterations([name], rows), set())

    def test_only_in_band_calibration_points_become_finalists(self) -> None:
        name, method = "synthetic_correlation_0p10_random", "default_cagra"
        rows = {
            (name, method, 70, 1, 0): {"recall": 0.9504, "recalls": [0.9504], "qps": 100.0},
            (name, method, 71, 1, 0): {"recall": 0.9505, "recalls": [0.9505], "qps": 200.0},
            (name, method, 72, 1, 0): {"recall": 0.9511, "recalls": [0.9511], "qps": 300.0},
        }
        self.assertEqual(match.finalists([name], rows), set(list(rows)[:2]))

    def test_finalist_retries_continue_until_a_three_repeat_match(self) -> None:
        name, method = "synthetic_correlation_0p10_random", "default_cagra"
        rows = {
            (name, method, 70 + i, 1, 0): {
                "recall": 0.9505, "recalls": [0.9505], "qps": 100.0 - i
            }
            for i in range(6)
        }
        first = match.finalists([name], rows)
        self.assertEqual(len(first), 4)
        confirmed = {
            key: {"recall": 0.952, "recalls": [0.9505, 0.9505, 0.952]}
            for key in first
        }
        retry = match.finalist_retries([name], rows, confirmed)
        self.assertEqual(retry, set(rows) - first)
        winner = next(iter(retry))
        confirmed[winner] = {"recall": 0.9505, "recalls": [0.9504, 0.9505, 0.9506]}
        self.assertEqual(match.finalist_retries([name], rows, confirmed), set())

    def test_rescue_l_is_limited_to_overshoot_pairs_and_widths(self) -> None:
        target = next(iter(match.RESCUE_PAIRS))
        rows = {
            (*target, 10, 1, 14): {"recall": 0.947},
            (*target, 10, 1, 15): {"recall": 0.954},
            ("synthetic_correlation_0p50_random", "navix_reference", 10, 1, 0): {
                "recall": 0.947,
            },
        }
        with patch.object(match, "RESCUE_L_RADIUS", 1):
            points = match.rescue_l_candidates(rows)
        self.assertTrue(points)
        self.assertTrue(all(point[:2] == target for point in points))
        self.assertTrue(all(point[3] in match.WIDTHS and point[4] == 0 for point in points))
        self.assertIn((*target, 11, 1, 0), points)

    def test_rescue_iteration_refines_integer_caps_around_crossing(self) -> None:
        target = ("synthetic_correlation_0p10_positive", "navix_reference")
        rows = {
            (*target, 10, 1, 14): {"recall": 0.949},
            (*target, 10, 1, 15): {"recall": 0.954},
        }
        points = match.rescue_iteration_candidates(rows)
        self.assertTrue(points)
        self.assertIn((*target, 10, 1, 13), points)
        self.assertTrue(all(point[:2] == target for point in points))
        self.assertTrue(all(point[4] != match.automatic_iterations(point[2], point[3]) for point in points))

    def test_rescue_finalists_only_select_target_pairs(self) -> None:
        target = ("synthetic_correlation_0p90_random", "navix_reference")
        other = ("synthetic_correlation_0p50_random", "navix_reference")
        rows = {
            (*target, 34, 1, 0): {
                "recall": 0.951,
                "recalls": [0.9505, 0.9508, 0.951],
                "qps": 100.0,
            },
            (*other, 34, 1, 0): {
                "recall": 0.9505,
                "recalls": [0.9505],
                "qps": 200.0,
            },
        }
        self.assertEqual(match.rescue_finalists(rows), {(*target, 34, 1, 0)})

    def test_rescue_retries_are_restricted_to_target_pairs(self) -> None:
        target = next(iter(match.RESCUE_PAIRS))
        other = ("synthetic_correlation_0p50_random", "navix_reference")
        rows = {
            (*target, 10, 1, 0): {
                "recall": 0.9505,
                "recalls": [0.9505],
                "qps": 100.0,
            },
            (*other, 10, 1, 0): {
                "recall": 0.9505,
                "recalls": [0.9505],
                "qps": 200.0,
            },
        }
        self.assertEqual(
            match.finalist_retries(
                [target[0], other[0]], rows, {}, {target}
            ),
            {(*target, 10, 1, 0)},
        )

    def test_frozen_stage_plan_is_reused_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = ("cohort", "default_cagra", 64, 1, 0)
            self.assertEqual(match.planned(root, "anchors", {row}), [row])
            self.assertEqual(match.planned(root, "anchors", set()), [row])

    def test_unmatched_method_has_no_comparable_qps(self) -> None:
        name = "synthetic_correlation_0p10_random"
        base = (name, "default_cagra", 70, 1, 0)
        retain = (name, "default_cagra_accumulator", 70, 1, 0)
        navix = (name, "navix_reference", 10, 1, 3)
        calibration = {
            base: {"recall": 0.9505, "qps": 100.0, "recalls": [0.9505], "source": "fixture"},
            retain: {"recall": 0.9505, "qps": 90.0, "recalls": [0.9505], "source": "fixture"},
            navix: {"recall": 0.952, "qps": 110.0, "recalls": [0.952], "source": "fixture"},
        }
        final = {
            base: {**calibration[base], "recalls": [0.95, 0.9505, 0.951]},
            navix: {**calibration[navix], "recalls": [0.95, 0.9505, 0.952]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state").mkdir()
            with patch.object(match, "observations", return_value=calibration), patch.object(
                match, "new_rows", return_value=final
            ):
                match.write_analysis(root, root, [name])
            with (root / "analysis/matched_selected.csv").open(newline="") as stream:
                rows = {row["method"]: row for row in csv.DictReader(stream)}
            self.assertEqual(rows["default_cagra"]["status"], "matched")
            self.assertEqual(rows["navix_reference"]["status"], "unmatched")
            self.assertEqual(rows["navix_reference"]["qps_median"], "")
            with (root / "analysis/matched_comparisons.csv").open(newline="") as stream:
                pairs = list(csv.DictReader(stream))
            self.assertTrue(all(row["status"] == "unmatched" and not row["right_over_left"]
                                for row in pairs))


if __name__ == "__main__":
    unittest.main()
