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
    def test_fixed_cells_and_cumulative_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            delta_root = Path("/tmp/favor-test-data")
            paths = generate_configs.write_configs(Path(directory), delta_root)
            self.assertEqual(len(paths), 21)
            for slug, dataset in generate_configs.DATASETS.items():
                for batch in (10, 10000):
                    path = Path(directory) / f"{slug}_nq{batch}_multiseed.json"
                    params = json.loads(path.read_text())["index"][0]["search_params"]
                    self.assertEqual(
                        [len(param["favor_seed_masks"]) for param in params],
                        [1, 2, 3],
                    )
                    for rounds, param in enumerate(params, 1):
                        self.assertEqual(
                            param["favor_seed_masks"],
                            list(generate_configs.SEED_MASKS[:rounds]),
                        )
                        self.assertEqual(param["itopk"], 512)
                        self.assertEqual(param["search_width"], dataset["search_width"])
                        self.assertEqual(param["max_iterations"], 0)
                        self.assertEqual(param["favor_retention_fraction"], 0.0)
                        self.assertEqual(
                            param["favor_delta_d_file"],
                            str(
                                delta_root
                                / str(dataset["directory"])
                                / "cagra_g32_ig64.index.delta_d"
                            ),
                        )


class GateTest(unittest.TestCase):
    def test_gate_accepts_three_rounds_only(self) -> None:
        records = []
        for dataset in generate_configs.DATASETS:
            records.extend(
                [
                    {
                        "dataset": dataset,
                        "batch_size": 10000,
                        "variant": "adaptive_termination",
                        "recall": 0.94,
                        "qps": 100.0,
                    },
                    {
                        "dataset": dataset,
                        "batch_size": 10000,
                        "variant": "multi_seed_2",
                        "recall": 0.89,
                        "qps": 120.0,
                    },
                    {
                        "dataset": dataset,
                        "batch_size": 10000,
                        "variant": "multi_seed_3",
                        "recall": 0.91,
                        "qps": 110.0,
                    },
                ]
            )
        result = analyze.gate(records)
        self.assertTrue(result["pass"])
        self.assertFalse(result["rounds"][2]["pass"])
        self.assertTrue(result["rounds"][3]["pass"])


if __name__ == "__main__":
    unittest.main()
