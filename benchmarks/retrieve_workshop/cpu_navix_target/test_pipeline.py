#!/usr/bin/env python3
"""Synthetic unit tests for the CPU NaviX target-selection helpers."""

from __future__ import annotations

import importlib.util
import pathlib


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cpu_navix_target", HERE / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import run.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(workload: str, ef: int, recall: float, qps: float = 1.0) -> dict[str, object]:
    return {"workload": workload, "ef_search": ef, "recall": recall, "qps": qps}


def main() -> None:
    ef, status = MODULE.choose_ef([
        row("em", 21, 0.949), row("em", 22, 0.9505), row("em", 23, 0.9515),
    ], "em")
    assert (ef, status) == (22, "inside_window")

    ef, status = MODULE.choose_ef([
        row("em", 21, 0.949), row("em", 22, 0.953),
    ], "em")
    assert (ef, status) == (22, "closest_above")

    ef, status = MODULE.choose_ef([
        row("yfcc", 750, 0.796), row("yfcc", 8192, 0.799),
    ], "yfcc")
    assert (ef, status) == (8192, "closest_below_within_tolerance")

    ef, status = MODULE.choose_ef([
        row("yfcc", 750, 0.786), row("yfcc", 8192, 0.789),
    ], "yfcc")
    assert ef is None and status == "unreached"
    print("cpu_navix_target pipeline tests: PASS")


if __name__ == "__main__":
    main()
