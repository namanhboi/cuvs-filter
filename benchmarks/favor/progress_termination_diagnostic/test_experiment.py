#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402


def records(
    tops: list[list[int]],
    evaluations: list[int],
    passing: list[int],
    iterations: list[int] | None = None,
) -> np.ndarray:
    result = np.zeros(len(tops), dtype=analyze.CHECKPOINT_DTYPE)
    result["iteration"] = iterations or list(range(1, len(tops) + 1))
    result["cumulative_candidate_evaluations"] = evaluations
    result["cumulative_passing_candidates"] = passing
    result["output_count"] = [10 if len(top) == 10 else len(top) for top in tops]
    result["top_ids"][:] = 0xFFFFFFFF
    result["kth_passing_raw_distance"] = 1.0
    result["frontier_best"] = 2.0
    for slot, top in enumerate(tops):
        result[slot]["top_ids"][: len(top)] = top
    return result


class ProgressRuleTest(unittest.TestCase):
    def test_checkpoint_abi(self) -> None:
        self.assertEqual(analyze.CHECKPOINT_DTYPE.itemsize, 136)

    def test_evidence_resets_when_top10_changes(self) -> None:
        first = list(range(10))
        second = list(range(1, 11))
        trajectory = records(
            [first, first, second, second],
            [100, 1100, 2100, 3100],
            [1, 21, 41, 61],
        )
        evidence = analyze.stale_evidence(trajectory, 0.01)
        np.testing.assert_allclose(evidence, [0.0, 10.0, 0.0, 10.0])

    def test_incomplete_top10_cannot_accumulate_evidence(self) -> None:
        incomplete = list(range(9))
        complete = list(range(10))
        trajectory = records(
            [incomplete, incomplete, complete, complete],
            [100, 2100, 2200, 4200],
            [1, 21, 22, 42],
        )
        evidence = analyze.stale_evidence(trajectory, 0.01)
        np.testing.assert_allclose(evidence, [0.0, 0.0, 0.0, 20.0])

    def test_gap_is_an_additional_gate(self) -> None:
        trajectory = records(
            [list(range(10)), list(range(10))],
            [0, 3000],
            [0, 30],
            iterations=[133, 141],
        )
        trajectory[1]["frontier_best"] = 1.02
        capture = analyze.Capture(
            "gist",
            {},
            trajectory.reshape(1, 2),
            np.asarray([2], dtype=np.uint32),
            np.arange(10, dtype=np.uint32).reshape(1, 10),
            [],
        )
        selected, fired = analyze.selected_slots(capture, analyze.Rule(2, 1.05))
        self.assertFalse(bool(fired[0]))
        self.assertEqual(int(selected[0]), 1)


if __name__ == "__main__":
    unittest.main()
