#!/usr/bin/env python3
"""Focused tests for the termination/hash experiment configuration and policy gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze
import generate_configs


class ExperimentTest(unittest.TestCase):
    def test_configs_are_fixed_to_the_declared_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "configs"
            data = Path(temporary) / "data"
            captures = Path(temporary) / "captures"
            for dataset, spec in generate_configs.DATASETS.items():
                for hash_variant in generate_configs.HASHES:
                    config = generate_configs.make_config(
                        dataset, spec, hash_variant, data, captures
                    )
                    search = config["index"][0]["search_params"][0]
                    self.assertEqual(search["max_iterations"], spec["cap"])
                    self.assertEqual(
                        search["favor_termination_shadow_start_iteration"], spec["b0"]
                    )
                    self.assertEqual(search["favor_termination_shadow_parent_interval"], 32)
                    self.assertEqual(search["favor_retention_fraction"], 0.0)
                    self.assertEqual(search["itopk"], 512)
                current = generate_configs.make_current_config(dataset, spec, data)
                search = current["index"][0]["search_params"][0]
                self.assertEqual(search["max_iterations"], 0)
                self.assertFalse(any("diagnostics" in key for key in search))

    def test_checkpoint_abi_and_first_fire_selection(self) -> None:
        checkpoints = np.zeros((2, 4), dtype=analyze.CHECKPOINT_DTYPE)
        checkpoints["iteration"][:] = np.asarray([10, 20, 30, 40])
        checkpoints["output_count"][:] = 10
        checkpoints["top_ids"][:] = np.arange(10, dtype=np.uint32)
        counts = np.asarray([4, 3], dtype=np.uint32)
        selected, fired = analyze.selected_slots("top10_stable2", checkpoints, counts)
        np.testing.assert_array_equal(selected, np.asarray([1, 1]))
        np.testing.assert_array_equal(fired, np.asarray([True, True]))
        self.assertEqual(analyze.CHECKPOINT_DTYPE.itemsize, 124)

    def test_gate_rejects_recall_unsafe_forgetful_hash(self) -> None:
        rows = []
        hashes = []
        for dataset in generate_configs.DATASETS:
            rows.append(
                {
                    "dataset": dataset,
                    "hash": "forgetful",
                    "policy": "top10_stable1",
                    "recall": 0.99,
                    "mean_iterations": 100.0,
                }
            )
            hashes.append({"dataset": dataset, "hash": "forgetful", "final_recall": 0.5})
        gate = analyze.select_policy(rows, hashes)
        self.assertIsNone(gate["selected"])
        self.assertFalse(gate["forgetful_hash_recall_safe_all"])
        self.assertEqual(gate["disposition"], "reject_forgetful_hash_no_live_v2")


if __name__ == "__main__":
    unittest.main()
