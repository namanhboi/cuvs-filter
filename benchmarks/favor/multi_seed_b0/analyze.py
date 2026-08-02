#!/usr/bin/env python3
"""Summarize the fixed-cell multi-seed experiment and evaluate its go/no-go gate."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


CONFIG_RE = re.compile(r"^(gist|msturing1m|msturing10m)_nq(10|10000)_(\w+)\.json$")


def iteration_index(run_name: str, parameter_count: int) -> int:
    if parameter_count == 1:
        return 0
    match = re.search(r"/(\d+)/process_time/", run_name)
    if not match:
        raise ValueError(f"cannot recover search-parameter index from {run_name!r}")
    return int(match.group(1))


def median_metrics(raw_path: Path, params: list[dict[str, object]]) -> list[dict[str, object]]:
    raw = json.loads(raw_path.read_text())
    grouped: dict[int, list[dict[str, object]]] = {index: [] for index in range(len(params))}
    for row in raw["benchmarks"]:
        if row.get("run_type") != "iteration" or row.get("error_occurred", False):
            continue
        grouped[iteration_index(row["run_name"], len(params))].append(row)

    summaries = []
    for index, param in enumerate(params):
        rows = grouped[index]
        if not rows:
            raise ValueError(f"no successful iteration rows for parameter {index} in {raw_path}")
        variant = str(param["experiment_variant"])
        masks = param.get("favor_seed_masks", [])
        summaries.append(
            {
                "variant": variant,
                "rounds": len(masks) if masks else (1 if variant != "default_cagra" else 0),
                "recall": statistics.median(float(row["Recall"]) for row in rows),
                "qps": statistics.median(float(row["items_per_second"]) for row in rows),
                "latency_seconds": statistics.median(float(row["Latency"]) for row in rows),
                "underfilled_queries": statistics.median(
                    float(row.get("UnderfilledQueries", 0.0)) for row in rows
                ),
                "missing_result_slots": statistics.median(
                    float(row.get("MissingResultSlots", 0.0)) for row in rows
                ),
                "repetitions": len(rows),
            }
        )
    return summaries


def collect(config_dir: Path, result_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for config_path in sorted(config_dir.glob("*_nq*_*.json")):
        match = CONFIG_RE.match(config_path.name)
        if not match:
            continue
        dataset, batch_text, group = match.groups()
        raw_path = result_dir / "raw" / config_path.name
        if not raw_path.exists():
            raise FileNotFoundError(f"missing result for {config_path.name}: {raw_path}")
        config = json.loads(config_path.read_text())
        params = config["index"][0]["search_params"]
        for summary in median_metrics(raw_path, params):
            summary.update(dataset=dataset, batch_size=int(batch_text), group=group)
            records.append(summary)

    by_key = {
        (str(row["dataset"]), int(row["batch_size"]), str(row["variant"])): row
        for row in records
    }
    for row in records:
        dataset = str(row["dataset"])
        batch = int(row["batch_size"])
        baseline = by_key.get((dataset, batch, "automatic_retention"))
        adaptive = by_key.get((dataset, batch, "adaptive_termination"))
        row["qps_ratio_vs_baseline"] = (
            float(row["qps"]) / float(baseline["qps"]) if baseline else None
        )
        row["qps_ratio_vs_adaptive"] = (
            float(row["qps"]) / float(adaptive["qps"]) if adaptive else None
        )
        if str(row["variant"]).startswith("multi_seed_"):
            rounds = int(row["rounds"])
            previous_variant = "automatic_retention" if rounds == 1 else f"multi_seed_{rounds - 1}"
            previous = by_key.get((dataset, batch, previous_variant))
            row["marginal_recall"] = (
                float(row["recall"]) - float(previous["recall"]) if previous else None
            )
        else:
            row["marginal_recall"] = None
    return records


def gate(records: list[dict[str, object]]) -> dict[str, object]:
    datasets = ("gist", "msturing1m", "msturing10m")
    by_key = {
        (str(row["dataset"]), int(row["batch_size"]), str(row["variant"])): row
        for row in records
    }
    per_round: dict[int, dict[str, object]] = {}
    for rounds in (2, 3):
        details = []
        for dataset in datasets:
            candidate = by_key[(dataset, 10000, f"multi_seed_{rounds}")]
            adaptive = by_key[(dataset, 10000, "adaptive_termination")]
            details.append(
                {
                    "dataset": dataset,
                    "recall_ok": float(candidate["recall"]) >= 0.90,
                    "qps_ok": float(candidate["qps"]) > float(adaptive["qps"]),
                }
            )
        per_round[rounds] = {
            "pass": all(item["recall_ok"] and item["qps_ok"] for item in details),
            "details": details,
        }
    return {"pass": any(value["pass"] for value in per_round.values()), "rounds": per_round}


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "dataset",
        "batch_size",
        "group",
        "variant",
        "rounds",
        "recall",
        "marginal_recall",
        "qps",
        "qps_ratio_vs_baseline",
        "qps_ratio_vs_adaptive",
        "latency_seconds",
        "underfilled_queries",
        "missing_result_slots",
        "repetitions",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in records)


def write_report(path: Path, records: list[dict[str, object]], result: dict[str, object]) -> None:
    lines = [
        "# Independent multi-seed B0 result",
        "",
        f"Overall gate: **{'PASS' if result['pass'] else 'FAIL'}**",
        "",
        "The gate requires either the two- or three-mask variant to reach recall ≥ 0.90 on all",
        "three 10,000-query workloads and exceed the same-machine adaptive-termination QPS.",
        "",
        "| Dataset | Variant | Recall | QPS | QPS/adaptive | Marginal recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        if int(row["batch_size"]) != 10000 or str(row["variant"]) not in {
            "automatic_retention",
            "multi_seed_1",
            "multi_seed_2",
            "multi_seed_3",
            "adaptive_termination",
        }:
            continue
        marginal = row["marginal_recall"]
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {float(row['recall']):.5f} | "
            f"{float(row['qps']):,.0f} | {float(row['qps_ratio_vs_adaptive']):.3f} | "
            f"{'' if marginal is None else f'{float(marginal):+.5f}'} |"
        )
    lines.extend(["", "```json", json.dumps(result, indent=2), "```", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    records = collect(args.config_dir, args.result_dir)
    result = gate(records)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.result_dir / "summary.csv", records)
    write_report(args.result_dir / "report.md", records, result)
    (args.result_dir / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"gate={'PASS' if result['pass'] else 'FAIL'}; wrote {args.result_dir}")


if __name__ == "__main__":
    main()
