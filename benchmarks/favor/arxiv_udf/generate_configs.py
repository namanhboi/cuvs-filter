#!/usr/bin/env python3
"""Generate benchmark configs for the ARXIV UDF experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SWEEP_CELLS = [(l, w) for l in (64, 128, 256, 512) for w in (1, 2, 4)]
MAX_ITERATIONS = (0, 522, 1044, 2088, 4176, 7569)
MAX_QUERIES = 512
GRAPH_DEGREE = 32
MAX_HASH_SLOTS = 1 << 20
HASHMAP_MAX_FILL_RATE = 0.5
CORRECTNESS_WORKLOAD = "correctness_10000"
THROUGHPUT_WORKLOAD = "throughput_10000"
DATA_ROOT = Path(".")


def dataset(workload: str, predicate: str) -> dict:
  root = f"arxiv-for-fanns-medium/{predicate}/{workload}"
  if predicate in {"em", "emis"}:
    filter_spec = {
      "kind": "udf",
      "adapter": "spmat_contains_all",
      "base_metadata_file": f"{root}/../base_metadata.spmat",
      "query_metadata_file": f"{root}/query_metadata.spmat",
    }
  elif predicate == "r":
    filter_spec = {
      "kind": "udf",
      "adapter": "arxiv_range",
      "base_metadata_file": f"{root}/../base_metadata.rmeta",
      "query_metadata_file": f"{root}/query_metadata.rmeta",
    }
  else:
    raise ValueError(f"unknown predicate: {predicate}")

  return {
    "name": f"arxiv-{predicate}-{workload}",
    "base_file": "arxiv-for-fanns-medium/base.fbin",
    "query_file": f"{root}/query.fbin",
    "groundtruth_neighbors_file": f"{root}/groundtruth.ibin",
    "distance": "euclidean",
    "dtype": "float",
    "filter": filter_spec,
  }


def default_search(
  itopk: int, width: int, *, accumulator: bool = False, max_iterations: int = 0
) -> dict:
  return {
    "algo": "single_cta",
    "filter_mode": "default",
    "max_queries": MAX_QUERIES,
    "itopk": itopk,
    "search_width": width,
    "max_iterations": max_iterations,
    "favor_udf_passing_accumulator": accumulator,
  }


def favor_search(
  itopk: int,
  width: int,
  *,
  accumulator: bool = True,
  max_iterations: int = 0,
  include_sampling: bool = True,
) -> dict:
  return {
    "algo": "single_cta",
    "filter_mode": "favor",
    "max_queries": MAX_QUERIES,
    "itopk": itopk,
    "search_width": width,
    "max_iterations": max_iterations,
    "favor_delta_d_file": str(
      (DATA_ROOT / "arxiv-for-fanns-medium/cagra_g32_ig64.index.delta_d").resolve()
    ),
    "favor_delta_d_alpha": 10,
    "favor_delta_d_beta": 64,
    "favor_delta_d_bfs_depth": 2,
    "favor_penalty_mode": "cagra_retention_safe",
    "favor_penalty_lambda": 1.0,
    "favor_retention_fraction": 0.0,
    "favor_udf_include_sampling": include_sampling,
    "favor_udf_passing_accumulator": accumulator,
  }


def config(workload: str, predicate: str, batch_size: int, searches: list[dict]) -> dict:
  return {
    "dataset": dataset(workload, predicate),
    "search_basic_param": {"batch_size": batch_size, "k": 10},
    "index": [
      {
        "name": "cagra-g32-ig64",
        "algo": "cuvs_cagra",
        "file": "arxiv-for-fanns-medium/cagra_g32_ig64.index",
        "build_param": {
          "graph_build_algo": "NN_DESCENT",
          "graph_degree": 32,
          "intermediate_graph_degree": 64,
        },
        "search_params": searches,
      }
    ],
  }


def write(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2) + "\n")


def supported_search(itopk: int, width: int, max_iterations: int) -> bool:
  if max_iterations == 0:
    return True
  max_visited_nodes = itopk + width * GRAPH_DEGREE * max_iterations
  return max_visited_nodes <= MAX_HASH_SLOTS * HASHMAP_MAX_FILL_RATE


def main() -> None:
  global DATA_ROOT
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--result-root", type=Path, required=True)
  parser.add_argument("--data-root", type=Path, required=True)
  args = parser.parse_args()
  DATA_ROOT = args.data_root

  predicates = ("em", "emis", "r")
  # Use 10k-query workloads prepared by arxiv_udf/prepare_workloads.py.
  query_count = 10_000

  for predicate in predicates:
    # Single-query smoke runs against the full workload with batch_size=1.
    write(
      args.output / f"{predicate}_smoke.json",
      config(
        THROUGHPUT_WORKLOAD,
        predicate,
        1,
        [
          default_search(64, 1),
          default_search(64, 1, accumulator=True),
          favor_search(64, 1, accumulator=False),
          favor_search(64, 1),
        ],
      ),
    )
    write(
      args.output / f"{predicate}_accumulator_gate.json",
      config(
        THROUGHPUT_WORKLOAD,
        predicate,
        query_count,
        [default_search(512, 2), default_search(512, 2, accumulator=True)],
      ),
    )

    correctness_searches: list[dict] = []
    throughput_searches: list[dict] = []
    for itopk, width in SWEEP_CELLS:
      for max_iterations in MAX_ITERATIONS:
        if not supported_search(itopk, width, max_iterations):
          continue
        correctness_searches.append(default_search(itopk, width, max_iterations=max_iterations))
        correctness_searches.append(
          favor_search(
            itopk, width, accumulator=False, max_iterations=max_iterations, include_sampling=True
          )
        )
        correctness_searches.append(
          favor_search(
            itopk, width, accumulator=True, max_iterations=max_iterations, include_sampling=True
          )
        )
        throughput_searches.append(default_search(itopk, width, max_iterations=max_iterations))
        throughput_searches.append(
          favor_search(
            itopk,
            width,
            accumulator=False,
            max_iterations=max_iterations,
            include_sampling=True,
          )
        )
        throughput_searches.append(
          favor_search(
            itopk, width, accumulator=True, max_iterations=max_iterations, include_sampling=True
          )
        )

    write(
      args.output / f"{predicate}_correctness.json",
      config(CORRECTNESS_WORKLOAD, predicate, query_count, correctness_searches),
    )
    write(
      args.output / f"{predicate}_throughput.json",
      config(THROUGHPUT_WORKLOAD, predicate, query_count, throughput_searches),
    )

  # NOTE: sampling-only comparisons are intentionally omitted in this ARXIV harness.


if __name__ == "__main__":
  main()
