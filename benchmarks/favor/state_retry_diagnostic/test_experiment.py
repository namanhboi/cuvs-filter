#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze
import generate_configs


class ConfigTest(unittest.TestCase):
    def test_fixed_matrix_and_cumulative_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            written = generate_configs.write_configs(
                root / "configs", root / "data", root / "captures"
            )
            self.assertEqual(len(written), 15)
            for slug, dataset in generate_configs.DATASETS.items():
                for strategy in generate_configs.STRATEGIES:
                    config = json.loads((root / "configs" / f"{slug}_{strategy}.json").read_text())
                    param = config["index"][0]["search_params"][0]
                    self.assertEqual(param["favor_retry_strategy"], strategy)
                    self.assertEqual(param["favor_retry_rounds"], dataset["rounds"])
                    self.assertEqual(param["favor_retry_b0"], dataset["b0"])
                    self.assertEqual(param["max_iterations"], 0)
                    self.assertEqual(param["favor_retention_fraction"], 0.0)
                    self.assertEqual(config["search_basic_param"]["batch_size"], 10000)


def synthetic_records(
    saved_recall: float, independent_recall: float = 0.84, oracle_recall: float = 0.94
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, spec in generate_configs.DATASETS.items():
        final_round = int(spec["rounds"])
        rows.append(
            {
                "dataset": dataset,
                "strategy": "independent",
                "round": 1,
                "accumulated_recall": 0.80,
            }
        )
        values = {
            "independent": independent_recall,
            "passing": saved_recall,
            "frontier": saved_recall,
            "combined": saved_recall,
            "oracle": oracle_recall,
        }
        for strategy, recall in values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "round": final_round,
                    "accumulated_recall": recall,
                }
            )
    return rows


class DecisionTest(unittest.TestCase):
    def test_passing_accumulator_can_be_sufficient(self) -> None:
        decision = analyze.decide(synthetic_records(saved_recall=0.91))
        self.assertEqual(decision["conclusion"], "passing_accumulator_sufficient")

    def test_only_oracle_implies_full_state(self) -> None:
        decision = analyze.decide(
            synthetic_records(saved_recall=0.83, independent_recall=0.82)
        )
        self.assertEqual(decision["conclusion"], "full_in_kernel_state_required")

    def test_half_oracle_gain_and_independent_margin_is_partial(self) -> None:
        decision = analyze.decide(
            synthetic_records(saved_recall=0.88, independent_recall=0.85, oracle_recall=0.94)
        )
        self.assertEqual(decision["conclusion"], "saved_state_reseed_partial")

    def test_oracle_below_target_does_not_confirm_depth(self) -> None:
        decision = analyze.decide(
            synthetic_records(saved_recall=0.84, independent_recall=0.83, oracle_recall=0.89)
        )
        self.assertEqual(decision["conclusion"], "depth_not_confirmed")


if __name__ == "__main__":
    unittest.main()
