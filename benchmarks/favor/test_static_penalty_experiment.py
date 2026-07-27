#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import static_penalty_experiment as experiment


class StaticPenaltyFormulaTest(unittest.TestCase):
    def test_known_shortage_probabilities(self) -> None:
        expected = {
            0.01: 0.9643819583,
            0.02: 0.4267572771,
            0.03: 0.0563239237,
        }
        for selectivity, target in expected.items():
            actual = experiment.binomial_shortage_probability(
                512, selectivity, 10
            )
            self.assertAlmostEqual(actual, target, places=9)
        self.assertLess(
            experiment.binomial_shortage_probability(512, 0.10, 10),
            1e-12,
        )

    def test_probability_boundaries(self) -> None:
        self.assertEqual(
            experiment.binomial_shortage_probability(512, 0.0, 10), 1.0
        )
        self.assertEqual(
            experiment.binomial_shortage_probability(512, 1.0, 10), 0.0
        )
        self.assertEqual(
            experiment.binomial_shortage_probability(5, 0.5, 10), 1.0
        )
        self.assertEqual(
            experiment.binomial_shortage_probability(512, 0.5, 0), 0.0
        )

    def test_formula_multipliers(self) -> None:
        p_short = 0.0563239237
        self.assertEqual(experiment.formula_multiplier("current", p_short), 1.0)
        self.assertEqual(experiment.formula_multiplier("zero", p_short), 0.0)
        self.assertEqual(experiment.formula_multiplier("hard05", p_short), 1.0)
        self.assertEqual(experiment.formula_multiplier("hard10", p_short), 0.0)
        self.assertAlmostEqual(
            experiment.formula_multiplier("smooth2", p_short), p_short**2
        )
        self.assertAlmostEqual(
            experiment.formula_multiplier("smooth4", p_short), p_short**4
        )

    def test_generated_configs_use_direct_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "datasets"
            dataset_dir = data_dir / "sift-128-euclidean"
            dataset_dir.mkdir(parents=True)
            payload = bytearray(80)
            payload[:8] = b"CUVSDD\r\n"
            struct.pack_into("<f", payload, 52, 2.5)
            (dataset_dir / "cagra_g32_ig64.index.delta_d").write_bytes(payload)

            result_dir = root / "results"
            args = argparse.Namespace(
                result_dir=result_dir,
                data_dir=data_dir,
                datasets=["sift"],
                selectivities=[3],
                itopk_values=[512],
                search_widths=[1],
                batch_sizes=[10_000],
                formulas=list(experiment.ALL_FORMULAS),
                k=10,
            )
            experiment.generate(args)

            config_path = result_dir / "configs" / "sift_s03_nq10000.json"
            config = json.loads(config_path.read_text())
            params = config["index"][0]["search_params"]
            self.assertEqual(len(params), 7)
            self.assertNotIn("favor_delta_d", params[0])
            for param in params[1:]:
                self.assertIn("favor_delta_d", param)
                self.assertNotIn("favor_delta_d_file", param)

            manifest = json.loads((result_dir / "manifest.json").read_text())
            entries = manifest["configs"][config_path.name]["entries"]
            self.assertEqual(entries[0]["formula"], "default")
            self.assertTrue(
                math.isclose(entries[1]["favor_delta_d"], 2.5)
            )
            self.assertEqual(entries[2]["favor_delta_d"], 0.0)


if __name__ == "__main__":
    unittest.main()
