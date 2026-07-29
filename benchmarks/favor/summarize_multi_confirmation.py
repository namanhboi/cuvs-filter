#!/usr/bin/env python3
"""Validate and combine the controlled retention-safe MULTI_CTA results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATASETS = ("sift", "gist", "bigann1m", "bigann10m", "msturing1m", "msturing10m")
SELECTIVITIES = (0.01, 0.10, 0.50, 0.90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    combined = []
    failures = []
    for dataset in DATASETS:
        path = args.result_root / dataset / "target_recall_summary.csv"
        rows = list(csv.DictReader(path.open()))
        qps = {
            (float(row["selectivity"]), row["series"]): row
            for row in rows
            if row["workload"] == "throughput"
        }
        for selectivity in SELECTIVITIES:
            default = qps.get((selectivity, "default"))
            favor = qps.get((selectivity, "favor_retention_safe"))
            if default is None or favor is None:
                failures.append(f"{dataset} {selectivity:.0%}: missing target result")
                continue
            default_qps = float(default["value"])
            favor_qps = float(favor["value"])
            ratio = favor_qps / default_qps
            row = {
                "dataset": dataset,
                "selectivity": selectivity,
                "target_recall": 0.99,
                "default_qps": default_qps,
                "favor_qps": favor_qps,
                "favor_over_default": ratio,
                "default_method": default["target_method"],
                "favor_method": favor["target_method"],
                "needs_three_run_tiebreaker": ratio <= 1.03,
            }
            combined.append(row)
            if ratio < 0.97:
                failures.append(
                    f"{dataset} {selectivity:.0%}: FAVOR/default={ratio:.3f} < 0.97"
                )
            if dataset.startswith("bigann") and selectivity == 0.10 and ratio <= 1.10:
                failures.append(
                    f"{dataset} 10%: FAVOR/default={ratio:.3f} does not exceed 1.10"
                )

    output = args.result_root / "controlled_target_recall_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=combined[0].keys())
        writer.writeheader()
        writer.writerows(combined)

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(combined)} target cells; wrote {output}")


if __name__ == "__main__":
    main()
