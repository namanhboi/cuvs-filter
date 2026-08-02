#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import plot_results


class PlotResultsTest(unittest.TestCase):
    def test_overlay_mode_is_loaded_without_primary_namespace_collision(self) -> None:
        rows = [
            {
                "run_type": "iteration",
                "label": 'filter_mode="default"',
                "itopk": 64,
                "search_width": 1,
                "Recall": 0.8,
                "items_per_second": 900.0,
            },
            {
                "run_type": "iteration",
                "label": 'filter_mode="favor"',
                "itopk": 64,
                "search_width": 1,
                "Recall": 0.95,
                "items_per_second": 1200.0,
            },
        ]
        overlay = plot_results.OverlaySeries(
            "committed_static",
            "favor",
            "Committed static FAVOR (old report)",
            Path("/unused"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_overlay_points(
                path, "items_per_second", overlay
            )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["recall"], 0.95)
        self.assertEqual(points[0]["value"], 1200.0)

    def test_overlay_missing_requested_mode_fails_clearly(self) -> None:
        rows = [
            {
                "run_type": "iteration",
                "label": 'filter_mode="default"',
                "itopk": 64,
                "search_width": 1,
                "Recall": 0.8,
                "items_per_second": 900.0,
            }
        ]
        overlay = plot_results.OverlaySeries(
            "committed_static",
            "favor",
            "Committed static FAVOR (old report)",
            Path("/unused"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            with self.assertRaisesRegex(ValueError, "requested overlay mode"):
                plot_results.load_overlay_points(
                    path, "items_per_second", overlay
                )

    def test_penalty_lambdas_are_separate_series(self) -> None:
        rows = []
        for penalty_lambda, recall in ((0.25, 0.91), (1.0, 0.99)):
            rows.append(
                {
                    "run_type": "iteration",
                    "label": (
                        'favor_penalty_mode="cagra_retention_safe"'
                        '#filter_mode="favor"'
                    ),
                    "favor_penalty_lambda": penalty_lambda,
                    "itopk": 64,
                    "search_width": 1,
                    "Recall": recall,
                    "items_per_second": 1000.0,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_points(path, "items_per_second")

        self.assertEqual(
            set(points),
            {
                "favor_retention_safe:lambda=0.25",
                "favor_retention_safe:lambda=1",
            },
        )
        self.assertEqual(
            points["favor_retention_safe:lambda=1"][0]["recall"], 0.99
        )

    def test_retention_fractions_are_separate_series(self) -> None:
        rows = []
        for fraction, recall in ((0.25, 0.91), (0.75, 0.97)):
            rows.append(
                {
                    "run_type": "iteration",
                    "label": (
                        'favor_penalty_mode="cagra_retention_safe"'
                        '#filter_mode="favor"'
                    ),
                    "favor_penalty_lambda": 1.0,
                    "favor_retention_fraction": fraction,
                    "itopk": 64,
                    "search_width": 1,
                    "Recall": recall,
                    "items_per_second": 1000.0,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_points(path, "items_per_second")

        self.assertEqual(
            set(points),
            {
                "favor_retention_safe:lambda=1:rho=0.25",
                "favor_retention_safe:lambda=1:rho=0.75",
            },
        )

    def test_zero_retention_fraction_is_automatic_series(self) -> None:
        rows = []
        for fraction, recall in ((0.0, 0.97), (0.5, 0.91)):
            rows.append(
                {
                    "run_type": "iteration",
                    "label": (
                        'favor_penalty_mode="cagra_retention_safe"'
                        '#filter_mode="favor"'
                    ),
                    "favor_penalty_lambda": 1.0,
                    "favor_retention_fraction": fraction,
                    "itopk": 64,
                    "search_width": 1,
                    "Recall": recall,
                    "items_per_second": 1000.0,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_points(path, "items_per_second")

        self.assertEqual(
            set(points),
            {
                "automatic_retention:lambda=1:rho=0",
                "favor_retention_safe:lambda=1:rho=0.5",
            },
        )
        self.assertEqual(
            points["automatic_retention:lambda=1:rho=0"][0]["recall"], 0.97
        )

    def test_legacy_retention_safe_overlay_loads_automatic_only_results(self) -> None:
        rows = [
            {
                "run_type": "iteration",
                "label": (
                    'favor_penalty_mode="cagra_retention_safe"'
                    '#filter_mode="favor"'
                ),
                "favor_penalty_lambda": 1.0,
                "favor_retention_fraction": 0.0,
                "itopk": 64,
                "search_width": 1,
                "Recall": 0.97,
                "items_per_second": 1000.0,
            }
        ]
        overlay = plot_results.OverlaySeries(
            "automatic_retention",
            "favor_retention_safe",
            "Automatic-retention FAVOR",
            Path("/unused"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_overlay_points(
                path, "items_per_second", overlay
            )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["retention_fraction"], 0.0)

    def test_lambda_filter_keeps_baselines(self) -> None:
        rows = [
            {
                "run_type": "iteration",
                "label": 'filter_mode="default"',
                "itopk": 64,
                "search_width": 1,
                "Recall": 0.9,
                "items_per_second": 1000.0,
            },
            {
                "run_type": "iteration",
                "label": (
                    'favor_penalty_mode="cagra_query_local"'
                    '#filter_mode="favor"'
                ),
                "favor_penalty_lambda": 0.25,
                "itopk": 64,
                "search_width": 1,
                "Recall": 0.9,
                "items_per_second": 1000.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_points(
                path, "items_per_second", selected_lambdas={1.0}
            )

        self.assertEqual(set(points), {"default"})

    def test_iteration_and_block_size_configs_are_not_combined(self) -> None:
        rows = []
        for max_iterations, thread_block_size, recall in (
            (56, 128, 0.90),
            (64, 256, 0.92),
        ):
            rows.append(
                {
                    "run_type": "iteration",
                    "label": 'filter_mode="default"',
                    "itopk": 128,
                    "search_width": 2,
                    "max_iterations": max_iterations,
                    "thread_block_size": thread_block_size,
                    "Recall": recall,
                    "items_per_second": 1000.0,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"benchmarks": rows}))
            points = plot_results.load_points(path, "items_per_second")

        self.assertEqual(len(points["default"]), 2)
        self.assertEqual(
            {point["max_iterations"] for point in points["default"]}, {56, 64}
        )
        self.assertEqual(
            {point["thread_block_size"] for point in points["default"]},
            {128, 256},
        )

    def test_interpolate_at_recall_uses_bracketing_frontier_points(self) -> None:
        result = plot_results.interpolate_at_recall(
            [
                {"recall": 0.89, "value": 600.0},
                {"recall": 0.91, "value": 500.0},
            ],
            0.90,
            maximize=True,
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["value"], 550.0)
        self.assertEqual(result["lower_recall"], 0.89)
        self.assertEqual(result["upper_recall"], 0.91)
        self.assertEqual(result["target_method"], "interpolated")

    def test_interpolate_at_recall_does_not_extrapolate(self) -> None:
        result = plot_results.interpolate_at_recall(
            [{"recall": 0.89, "value": 600.0}], 0.90, maximize=True
        )

        self.assertIsNone(result)

    def test_target_uses_best_measured_feasible_point_without_lower_bracket(self) -> None:
        result = plot_results.interpolate_at_recall(
            [
                {"recall": 0.91, "value": 500.0},
                {"recall": 0.95, "value": 600.0},
            ],
            0.90,
            maximize=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 600.0)
        self.assertEqual(result["point_recall"], 0.95)
        self.assertEqual(result["target_method"], "measured_feasible")


if __name__ == "__main__":
    unittest.main()
