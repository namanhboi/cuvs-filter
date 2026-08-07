#!/usr/bin/env python3
"""Analyze ARXIV UDF benchmark outputs and produce recall/QPS summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def safe_int(value: Any, default: int = 0) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def read_rows(name: Path) -> list[dict]:
  payload = json.loads(name.read_text())
  return [row for row in payload.get("benchmarks", []) if row.get("run_type") == "iteration"]


def config_variant(search: dict) -> str:
  if search.get("filter_mode", "default") == "default":
    return "default_cagra"
  if search.get("favor_udf_passing_accumulator", True):
    return "automatic_accumulator"
  return "automatic_legacy"


def parse_predicate_and_workload(config_name: str) -> tuple[str, str]:
  stem = config_name
  if stem.endswith(".json"):
    stem = stem[:-5]
  if "_" in stem:
    predicate, workload = stem.split("_", 1)
  else:
    predicate, workload = "unknown", stem
  return predicate, workload


def write_csv(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def analyze_file(
  config_name: str,
  raw_root: Path,
  config_root: Path,
  source: str,
) -> list[dict]:
  raw_path = raw_root / f"{config_name}.json"
  if not raw_path.exists():
    raise FileNotFoundError(raw_path)
  config_path = config_root / f"{config_name}.json"
  if not config_path.exists():
    raise FileNotFoundError(config_path)

  rows = read_rows(raw_path)
  config = json.loads(config_path.read_text())
  searches = config["index"][0]["search_params"]

  by_family: dict[int, dict] = {}
  for row in rows:
    by_family[safe_int(row.get("family_index"), -1)] = row
  if len(by_family) != len(searches) or set(by_family) != set(range(len(searches))):
    raise RuntimeError(f"incomplete result file: {config_name}")

  predicate, workload = parse_predicate_and_workload(config_name)
  result_rows = []
  for i, search in enumerate(searches):
    row = by_family[i]
    result_rows.append(
      {
        "predicate": predicate,
        "workload": workload,
        "metric": config["dataset"]["name"],
        "source": source,
        "variant": config_variant(search),
        "itopk": safe_int(search.get("itopk")),
        "search_width": safe_int(search.get("search_width")),
        "max_iterations": safe_int(search.get("max_iterations")),
        "recall": safe_float(row.get("Recall"), safe_float(row.get("recall", 0.0))),
        "qps": safe_float(row.get("items_per_second", row.get("qps", 0.0))),
        "latency_seconds": safe_float(row.get("Latency", 0.0)),
        "filter_violations": safe_float(row.get("FilterViolations", 0.0)),
        "invalid_sentinel_errors": safe_float(row.get("InvalidSentinelErrors", 0.0)),
        "underfilled_queries": safe_float(row.get("UnderfilledQueries", 0.0)),
      }
    )

  return result_rows


def pareto_frontier(points: list[dict], maximize_y: bool = True) -> list[dict]:
  if not points:
    return []
  candidates = sorted(points, key=lambda r: (safe_float(r["recall"]), safe_float(r["qps"])), reverse=True)
  frontier: list[dict] = []
  best_qps = float("-inf") if maximize_y else float("inf")
  for row in candidates:
    y = safe_float(row["qps"], 0.0)
    if maximize_y:
      if y >= best_qps:
        frontier.append(row)
        best_qps = y
    else:
      if y <= best_qps or best_qps == float("inf"):
        frontier.append(row)
        best_qps = y
  frontier.sort(key=lambda r: safe_float(r["recall"]))
  return frontier


def _plot_empty_message(plot_path: Path, text: str) -> None:
  try:
    import matplotlib.pyplot as plt
  except Exception:
    return
  fig, ax = plt.subplots(figsize=(7, 2.5))
  ax.axis("off")
  ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
  fig.tight_layout()
  fig.savefig(plot_path, dpi=180)
  plt.close(fig)


def plot_sweep(rows: list[dict], output_root: Path, *, plot_name: str, title: str) -> None:
  try:
    import matplotlib.pyplot as plt
  except Exception:
    return

  if not rows:
    _plot_empty_message(
      output_root / "plots" / plot_name,
      "No throughput rows available for this plot.\nRun the corresponding benchmark stage and rerun analyze.",
    )
    return

  output_root.mkdir(parents=True, exist_ok=True)
  plot_path = output_root / "plots"
  plot_path.mkdir(parents=True, exist_ok=True)

  # Global sweep plot with method+predicate keys.
  points_by_key: dict[str, list[dict]] = {}
  for row in rows:
    key = f"{row['predicate']}::{row['variant']}"
    points_by_key.setdefault(key, []).append(row)

  fig, ax = plt.subplots(figsize=(7, 4.2))
  for key, rows in sorted(points_by_key.items()):
    frontier = pareto_frontier(rows)
    if not frontier:
      continue
      ax.plot(
        [r["recall"] for r in frontier],
        [r["qps"] for r in frontier],
        marker="o",
        label=key,
      )
  recalls = [
    safe_float(row.get("recall"), 0.0)
    for row in rows
    if safe_float(row.get("recall"), 0.0) > 0.0 and safe_float(row.get("qps"), 0.0) > 0.0
  ]
  if recalls:
    min_recall = min(recalls)
  else:
    min_recall = 0.0
  x_min = max(0.0, min_recall - 0.02)
  ax.set(
    xlabel="Recall@10",
    ylabel="QPS",
    title=title,
    xlim=(x_min, 1.0),
    ylim=(0, None),
  )
  ax.set_ylim(bottom=0.0)
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(plot_path / plot_name, dpi=180)
  plt.close(fig)


def plot_sweep_by_predicate(
  rows: list[dict],
  output_root: Path,
  *,
  base_name: str,
  title_prefix: str,
) -> None:
  try:
    import matplotlib.pyplot as plt
  except Exception:
    return

  if not rows:
    output_root.mkdir(parents=True, exist_ok=True)
    plot_path = output_root / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    _plot_empty_message(
      plot_path / f"{base_name}_em.png",
      "No throughput rows available for this plot.\nRun the corresponding benchmark stage and rerun analyze.",
    )
    _plot_empty_message(
      plot_path / f"{base_name}_emis.png",
      "No throughput rows available for this plot.\nRun the corresponding benchmark stage and rerun analyze.",
    )
    _plot_empty_message(
      plot_path / f"{base_name}_r.png",
      "No throughput rows available for this plot.\nRun the corresponding benchmark stage and rerun analyze.",
    )
    return

  output_root.mkdir(parents=True, exist_ok=True)
  plot_path = output_root / "plots"
  plot_path.mkdir(parents=True, exist_ok=True)

  for predicate in ("em", "emis", "r"):
    subset = [row for row in rows if str(row.get("predicate")) == predicate]
    if not subset:
      _plot_empty_message(
        plot_path / f"{base_name}_{predicate}.png",
        "No throughput rows available for this predicate.\nRun the corresponding benchmark stage and rerun analyze.",
      )
      continue

    points_by_key: dict[str, list[dict]] = {}
    for row in subset:
      key = str(row["variant"])
      points_by_key.setdefault(key, []).append(row)
    recalls = [
      safe_float(row.get("recall"), 0.0)
      for row in subset
      if safe_float(row.get("recall"), 0.0) > 0.0 and safe_float(row.get("qps"), 0.0) > 0.0
    ]
    if recalls:
      min_recall = min(recalls)
    else:
      min_recall = 0.0
    x_min = max(0.0, min_recall - 0.02)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, rows_ in sorted(points_by_key.items()):
      frontier = pareto_frontier(rows_)
      if not frontier:
        continue
      ax.plot(
        [r["recall"] for r in frontier],
        [r["qps"] for r in frontier],
        marker="o",
        label=key,
      )
    ax.set(
      xlabel="Recall@10",
      ylabel="QPS",
      title=f"{title_prefix} ({predicate})",
      xlim=(x_min, 1.0),
      ylim=(0, None),
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path / f"{base_name}_{predicate}.png", dpi=180)
    plt.close(fig)

def write_report(
  result_root: Path,
  report_path: Path,
  correctness_rows: list[dict],
  throughput_rows: list[dict],
  b0_rows: list[dict],
  b0_rows_by_pred_variant: list[str],
) -> None:
  best_by_mode: dict[tuple[str, str], dict] = {}
  for row in correctness_rows:
    key = (row["predicate"], row["variant"])
    current = best_by_mode.get(key)
    if current is None or row["recall"] > current["recall"]:
      best_by_mode[key] = row

  best_by_mode_text = []
  for predicate in ("em", "emis", "r"):
    for variant in ("default_cagra", "automatic_legacy", "automatic_accumulator"):
      row = best_by_mode.get((predicate, variant))
      if row is not None:
        best_by_mode_text.append(
          f"| {predicate} | {variant} | {row['itopk']} | {row['search_width']} | {row['max_iterations']} | {row['recall']:.4f} | {row['qps']:.1f} |"
        )

  b0_table = "\n".join(b0_rows_by_pred_variant) if b0_rows_by_pred_variant else "| No B0 throughput rows found. |"

  throughput_availability = (
    "available (full sweep completed)"
    if throughput_rows
    else "not available (throughput workloads were not run in this directory)"
  )
  throughput_file = (
    f"{result_root}/throughput_summary.csv"
    if throughput_rows
    else "not generated (missing full throughput raw files)"
  )

  report_body = f"""#+title: ARXIV-For-FANNs MEDIUM Single-CTA UDF Report

* Configurations

Benchmarks are run with three UDF predicates: =em=, =emis=, =r=.

* Data status

Correctness: complete for all 3 predicates in `raw/*_correctness.json`.

Throughput full sweep: {throughput_availability}.

B0 throughput: available from `raw_b0/*_b0_throughput.json`.

* Sweep summary

The full matrix varies:
- =itopk in {{64,128,256,512}}=
- =search_width in {{1,2,4}}=
- =max_iterations in {{0,522,1044,2088,4176,7569}}=

For each (predicate, workload), =default_cagra= and both FAVOR variants were benchmarked with
end-to-end timing (including per-query selectivity sampling when FILTER_MODE=favor).

| Predicate | Variant | Best recall row (itopk,width,max_iterations,recall,qps) |
|-
{chr(10).join(best_by_mode_text) if best_by_mode_text else "| unavailable | | | | | |"}

* B0 frontier (recall and QPS)

| Predicate | Variant | itopk | width | max_iter | Recall@10 | QPS |
|-
{b0_table}

* Outputs

- {result_root}/correctness_summary.csv
- {throughput_file}
- {result_root}/b0_summary.csv
- {result_root}/plots/qps_recall_sweep.png
- {result_root}/plots/qps_recall_b0.png
- {result_root}/plots/qps_recall_sweep_em.png
- {result_root}/plots/qps_recall_sweep_emis.png
- {result_root}/plots/qps_recall_sweep_r.png
- {result_root}/plots/qps_recall_b0_em.png
- {result_root}/plots/qps_recall_b0_emis.png
- {result_root}/plots/qps_recall_b0_r.png
"""

  report_path.write_text(report_body + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--result-root", type=Path, required=True)
  parser.add_argument("--report", type=Path, required=True)
  args = parser.parse_args()

  raw_root = args.result_root / "raw"
  raw_b0_root = args.result_root / "raw_b0"
  config_b0_root = args.result_root / "configs_b0"
  config_root = args.result_root / "configs"

  predicates = ("em", "emis", "r")
  correctness_rows: list[dict] = []
  throughput_rows: list[dict] = []
  b0_rows: list[dict] = []
  all_results: list[dict] = []

  for predicate in predicates:
    correctness_name = f"{predicate}_correctness"
    throughput_name = f"{predicate}_throughput"
    b0_name = f"{predicate}_b0_throughput"

    correctness_file = raw_root / f"{correctness_name}.json"
    if correctness_file.exists():
      rows = analyze_file(correctness_name, raw_root, config_root, "correctness")
      correctness_rows.extend(rows)
      all_results.extend(rows)

    throughput_file = raw_root / f"{throughput_name}.json"
    if throughput_file.exists():
      rows = analyze_file(throughput_name, raw_root, config_root, "throughput")
      throughput_rows.extend(rows)
      all_results.extend(rows)

    b0_file = raw_b0_root / f"{b0_name}.json"
    if b0_file.exists():
      rows = analyze_file(b0_name, raw_b0_root, config_b0_root, "throughput_b0")
      b0_rows.extend(rows)
      all_results.extend(rows)

  write_csv(args.result_root / "correctness_summary.csv", correctness_rows)
  if throughput_rows:
    write_csv(args.result_root / "throughput_summary.csv", throughput_rows)
  else:
    old = args.result_root / "throughput_summary.csv"
    if old.exists():
      old.unlink()
  write_csv(args.result_root / "all_results_summary.csv", all_results)
  write_csv(args.result_root / "b0_summary.csv", b0_rows)

  plot_sweep(throughput_rows, args.result_root, plot_name="qps_recall_sweep.png", title="ARXIV Single-CTA UDF sweep")
  plot_sweep_by_predicate(
    throughput_rows,
    args.result_root,
    base_name="qps_recall_sweep",
    title_prefix="ARXIV Single-CTA UDF sweep",
  )
  plot_sweep(b0_rows, args.result_root, plot_name="qps_recall_b0.png", title="ARXIV Single-CTA UDF B0 frontier")
  plot_sweep_by_predicate(
    b0_rows,
    args.result_root,
    base_name="qps_recall_b0",
    title_prefix="ARXIV Single-CTA UDF B0 frontier",
  )

  b0_rows_by_pred_variant: list[str] = []
  if b0_rows:
    b0_sorted = [r for r in b0_rows if safe_int(r["max_iterations"]) == 0]
    for predicate in ("em", "emis", "r"):
      for variant in ("default_cagra", "automatic_legacy", "automatic_accumulator"):
        best = max(
          [r for r in b0_sorted if r["predicate"] == predicate and r["variant"] == variant],
          key=lambda r: (safe_float(r["recall"]), safe_float(r["qps"])),
          default=None,
        )
        if best is not None:
          b0_rows_by_pred_variant.append(
            f"| {predicate} | {variant} | {best['itopk']} | {best['search_width']} | {best['max_iterations']} | {best['recall']:.4f} | {best['qps']:.1f} |"
          )
  write_report(
    args.result_root,
    args.report,
    correctness_rows,
    throughput_rows,
    b0_rows,
    b0_rows_by_pred_variant,
  )


if __name__ == "__main__":
  main()
