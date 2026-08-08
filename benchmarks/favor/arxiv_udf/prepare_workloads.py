#!/usr/bin/env python3
"""Prepare ArXiv UDF benchmark workloads from raw metadata files.

The script materializes the ArXiv MEDIUM dataset into cuVS-bench layout without copying the raw
files.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QUERY_COUNT = 10_000
GT_K = 10
INVALID_SENTINEL = np.iinfo(np.uint32).max
MAX_PADDING_SENTINELS = min(1_000, GT_K)


def _is_synthetic_padding(node_id: int) -> bool:
  start = int(INVALID_SENTINEL - (MAX_PADDING_SENTINELS - 1))
  return node_id >= start


def read_fvec_matrix(path: Path) -> np.ndarray:
  """Read ANN-style ``.fvecs`` where each row stores ``dim`` then coordinates."""
  raw = np.memmap(path, dtype="<i4", mode="r")
  if raw.size == 0:
    raise ValueError(f"Fvecs file is empty: {path}")
  dim = int(raw[0])
  if dim <= 0:
    raise ValueError(f"Invalid fvecs dimension in {path}")
  row_width = dim + 1
  if raw.size % row_width != 0:
    raise ValueError(
      f"Invalid fvecs geometry in {path}: dim={dim}, total_int32={raw.size}, row_width={row_width}"
    )
  rows = raw.size // row_width
  vectors_i32 = raw.reshape(rows, row_width)[:, 1:]
  vectors = vectors_i32.view("<f4")
  return np.asarray(vectors, dtype=np.float32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  values: list[dict[str, Any]] = []
  with path.open() as stream:
    for line_no, line in enumerate(stream, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        values.append(json.loads(line))
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at {path}:{line_no}") from e
  return values


def write_matrix_u32(path: Path, rows: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  rows = np.asarray(rows, dtype=np.uint32)
  with path.open("wb") as stream:
    stream.write(struct.pack("<II", int(rows.shape[0]), int(rows.shape[1])))
    rows.tofile(stream)


def write_fbin(path: Path, rows: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  rows = np.asarray(rows, dtype=np.float32)
  with path.open("wb") as stream:
    stream.write(struct.pack("<II", int(rows.shape[0]), int(rows.shape[1])))
    rows.astype("<f4", copy=False).tofile(stream)


def write_spmat(path: Path, rows: list[list[int]], cols: int) -> None:
  if cols <= 0:
    raise ValueError("spmat columns must be positive")
  nnz = sum(len(row) for row in rows)
  offsets = np.zeros(len(rows) + 1, dtype=np.int64)
  for i, row in enumerate(rows):
    if any(label < 0 for label in row):
      raise ValueError(f"Negative label in sparse metadata: {path}")
    offsets[i + 1] = offsets[i] + len(row)
  columns = np.empty(int(offsets[-1]), dtype=np.int32)
  cursor = 0
  for row in rows:
    columns[cursor : cursor + len(row)] = np.asarray(row, dtype=np.int32)
    cursor += len(row)
  values = np.ones(int(offsets[-1]), dtype=np.float32)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as stream:
    stream.write(struct.pack("<qqq", int(len(rows)), int(cols), int(nnz)))
    offsets.astype("<i8", copy=False).tofile(stream)
    columns.astype("<i4", copy=False).tofile(stream)
    values.astype("<f4", copy=False).tofile(stream)


def write_range_metadata(path: Path, values: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  values = np.asarray(values, dtype=np.int32)
  values = np.ascontiguousarray(values)
  with path.open("wb") as stream:
    stream.write(struct.pack("<I", int(values.shape[0])))
    values.astype("<i4", copy=False).tofile(stream)


def _dedupe_to_k(row: np.ndarray, k: int) -> np.ndarray:
  seen = set()
  values: list[int] = []
  for value in row:
    v = int(value)
    if _is_synthetic_padding(v):
      continue
    if v not in seen:
      seen.add(v)
      values.append(v)
    if len(values) == k:
      break
  if len(values) < k:
    for pad in range(k - len(values)):
      values.append(int(INVALID_SENTINEL - pad))
  return np.asarray(values, dtype=np.uint32)


def _read_ivec_prefixed_count(ivec: np.ndarray, query_count: int, k: int) -> np.ndarray | None:
  cursor = 0
  rows: list[np.ndarray] = []
  while cursor < ivec.size:
    count = int(ivec[cursor])
    cursor += 1
    if count <= 0 or cursor + count > ivec.size:
      return None
    if len(rows) < query_count:
      rows.append(_dedupe_to_k(ivec[cursor : cursor + count], k))
    cursor += count
  if len(rows) < query_count:
    return None
  return np.stack(rows, axis=0)


def read_ivec_first_k(ivec_path: Path, query_count: int, k: int = GT_K) -> np.ndarray:
  ivec = np.memmap(ivec_path, dtype="<u4", mode="r")
  if ivec.size == 0:
    raise ValueError(f"Ground-truth file is empty: {ivec_path}")
  values = np.asarray(ivec, dtype=np.uint32)
  if query_count <= 0:
    raise ValueError("query_count must be positive")

  parsed = _read_ivec_prefixed_count(values, query_count=query_count, k=k)
  if parsed is not None and parsed.shape[0] >= query_count:
    return parsed

  cursor = 0
  parsed_rows = 0
  while cursor < values.size:
    count = int(values[cursor])
    cursor += 1
    if count <= 0 or cursor + count > values.size:
      break
    cursor += count
    parsed_rows += 1

  raise ValueError(
    f"Could not parse {ivec_path}: first_values={values[:8].tolist()}, query_count={query_count}, "
    f"k={k}, prefixed_rows={parsed_rows}, consumed_values={cursor}, total_values={values.size}"
  )


def read_ivec_row_count(ivec_path: Path) -> int:
  ivec = np.memmap(ivec_path, dtype="<u4", mode="r")
  if ivec.size == 0:
    return 0
  values = np.asarray(ivec, dtype=np.uint32)

  cursor = 0
  rows = 0
  while cursor < values.size:
    count = int(values[cursor])
    cursor += 1
    if count <= 0 or cursor + count > values.size:
      break
    cursor += count
    rows += 1
  if rows > 0 and cursor == values.size:
    return rows
  return 0

def write_ivec_to_ibin(path: Path, ivec_path: Path, query_count: int, k: int = GT_K) -> None:
  neighbors = read_ivec_first_k(ivec_path, query_count=query_count, k=k)
  write_matrix_u32(path, neighbors)


def _validate_em_ground_truth(
  ground_truth: np.ndarray, query_labels: np.ndarray, base_labels: np.ndarray
) -> None:
  for query_id in range(ground_truth.shape[0]):
    label = int(query_labels[query_id])
    base_limit = base_labels.shape[0]
    for rank, node_id in enumerate(ground_truth[query_id], start=1):
      node_id = int(node_id)
      if _is_synthetic_padding(node_id):
        continue
      if node_id >= base_limit:
        raise ValueError(
          f"em predicate violation: out-of-range node id for query {query_id}, rank {rank}: node={node_id}, base_rows={base_limit}"
        )
      if int(base_labels[node_id]) != label:
        raise ValueError(
          f"em predicate violation: query {query_id}, rank {rank}, node={node_id}, base={int(base_labels[node_id])}, query={label}"
        )


def _validate_emis_ground_truth(
  ground_truth: np.ndarray, query_labels: np.ndarray, base_label_sets: list[set[int]]
) -> None:
  for query_id in range(ground_truth.shape[0]):
    label = int(query_labels[query_id])
    base_limit = len(base_label_sets)
    for rank, node_id in enumerate(ground_truth[query_id], start=1):
      node_id = int(node_id)
      if _is_synthetic_padding(node_id):
        continue
      if node_id >= base_limit:
        raise ValueError(
          f"emis predicate violation: out-of-range node id for query {query_id}, rank {rank}: node={node_id}, base_rows={base_limit}"
        )
      if label not in base_label_sets[node_id]:
        raise ValueError(
          f"emis predicate violation: query {query_id}, rank {rank}, node={node_id}, query label={label}"
        )


def _validate_r_ground_truth(
  ground_truth: np.ndarray, query_ranges: np.ndarray, base_update_dates: np.ndarray
) -> None:
  for query_id in range(ground_truth.shape[0]):
    start, end = map(int, query_ranges[query_id])
    base_limit = base_update_dates.shape[0]
    for rank, node_id in enumerate(ground_truth[query_id], start=1):
      node_id = int(node_id)
      if _is_synthetic_padding(node_id):
        continue
      if node_id >= base_limit:
        raise ValueError(
          f"r predicate violation: out-of-range node id for query {query_id}, rank {rank}: node={node_id}, base_rows={base_limit}"
        )
      date = int(base_update_dates[node_id])
      if date < start or date > end:
        raise ValueError(
          f"r predicate violation: query {query_id}, rank {rank}, node={node_id}, date={date}, range=[{start}, {end}]"
        )


def select_rows(matrix: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
  return np.asarray(matrix[query_ids])


def write_subset(
  output: Path,
  name: str,
  query_ids: list[int],
  queries: np.ndarray,
  query_metadata: list[list[int]] | np.ndarray,
  groundtruth: np.ndarray,
  query_metadata_file_name: str,
  predicate: str,
  spmat_cols: int,
) -> dict[str, Any]:
  target = output / name
  target.mkdir(parents=True, exist_ok=True)
  query_ids_arr = np.asarray(query_ids, dtype=np.int64)
  write_fbin(target / "query.fbin", select_rows(queries, query_ids_arr))
  write_matrix_u32(target / "groundtruth.ibin", groundtruth[query_ids_arr])
  if isinstance(query_metadata, list):
    write_spmat(target / query_metadata_file_name, [query_metadata[i] for i in query_ids], spmat_cols)
  else:
    write_range_metadata(target / query_metadata_file_name, query_metadata[query_ids_arr])
  return {
    "name": name,
    "queries": len(query_ids),
    "predicate": predicate,
  }


def build_workloads(output: Path, base_vectors: np.ndarray, queries: np.ndarray, source_root: Path) -> None:
  query_count = queries.shape[0]
  em_queries_all = read_jsonl(source_root / "em_query_attributes.jsonl")
  emis_queries_all = read_jsonl(source_root / "emis_query_attributes.jsonl")
  r_queries_all = read_jsonl(source_root / "r_query_attributes.jsonl")
  em_gt_rows = read_ivec_row_count(source_root / "ground_truth_em.ivecs")
  emis_gt_rows = read_ivec_row_count(source_root / "ground_truth_emis.ivecs")
  r_gt_rows = read_ivec_row_count(source_root / "ground_truth_r.ivecs")

  effective_query_count = min(
    query_count,
    len(em_queries_all),
    len(emis_queries_all),
    len(r_queries_all),
    em_gt_rows,
    emis_gt_rows,
    r_gt_rows,
  )
  if effective_query_count <= 0:
    raise ValueError("No query rows available from provided metadata/ground-truth files")
  if effective_query_count != query_count:
    print(
      f"query_count mismatch: requested={query_count}, using available={effective_query_count} "
      f"(em={len(em_queries_all)}, emis={len(emis_queries_all)}, r={len(r_queries_all)}, "
      f"gt_rows={em_gt_rows}/{emis_gt_rows}/{r_gt_rows})"
    )
  query_count = effective_query_count
  query_ids = list(range(query_count))

  # Build metadata once from raw sources.
  em_queries = em_queries_all[:query_count]
  emis_queries = emis_queries_all[:query_count]
  r_queries = r_queries_all[:query_count]
  if (
    len(em_queries) < query_count or len(emis_queries) < query_count or len(r_queries) < query_count
  ):
    raise ValueError(f"query metadata files must include at least {query_count} rows")

  base_attrs = read_jsonl(source_root / "database_attributes.jsonl")
  if len(base_attrs) != base_vectors.shape[0]:
    raise ValueError(f"base attributes rows mismatch: {len(base_attrs)} vs {base_vectors.shape[0]}")

  # Shared dataset-level outputs.
  write_fbin(output / "base.fbin", base_vectors)
  write_fbin(output / "query.fbin", queries)

  # EM predicate: candidate label is number_of_sub_categories (exact match).
  em_dir = output / "em"
  em_dir.mkdir(parents=True, exist_ok=True)
  em_base_labels = [int(row.get("number_of_sub_categories", 0)) for row in base_attrs]
  em_query_labels = [int(row["label"]) for row in em_queries]
  em_col_max = max(max(em_base_labels), max(em_query_labels))
  em_base_metadata = [[label] for label in em_base_labels]
  em_query_metadata = [[label] for label in em_query_labels]
  write_spmat(em_dir / "base_metadata.spmat", em_base_metadata, em_col_max + 1)
  write_spmat(em_dir / "query_metadata.spmat", em_query_metadata, em_col_max + 1)
  em_ground_truth = read_ivec_first_k(
    source_root / "ground_truth_em.ivecs", query_count=query_count, k=GT_K
  )
  _validate_em_ground_truth(
    em_ground_truth, np.asarray(em_query_labels, dtype=np.int64), np.asarray(em_base_labels, dtype=np.int64)
  )
  write_matrix_u32(em_dir / "groundtruth.ibin", em_ground_truth)
  gt_em = np.memmap(
    em_dir / "groundtruth.ibin", dtype=np.uint32, mode="r", offset=8, shape=(query_count, GT_K)
  )
  correctness_name = f"correctness_{query_count}"
  throughput_name = f"throughput_{query_count}"

  workloads = [
    write_subset(
      em_dir,
      correctness_name,
      query_ids,
      queries,
      em_query_metadata,
      gt_em,
      "query_metadata.spmat",
      "em",
      em_col_max + 1,
    ),
    write_subset(
      em_dir,
      throughput_name,
      query_ids,
      queries,
      em_query_metadata,
      gt_em,
      "query_metadata.spmat",
      "em",
      em_col_max + 1,
    ),
  ]
  (em_dir / "manifest.json").write_text(
    json.dumps(
      {"predicate": "em", "workloads": workloads, "query_rows": query_count}
    )
    + "\n"
  )

  # EMIS predicate: each query specifies one main category (string), mapped by vocabulary.
  emis_dir = output / "emis"
  emis_dir.mkdir(parents=True, exist_ok=True)
  main_vocab: dict[str, int] = {}
  next_id = 0
  for row in base_attrs:
    for label in row.get("main_categories", []):
      if label not in main_vocab:
        main_vocab[label] = next_id
        next_id += 1
  for row in emis_queries:
    label = row["label"]
    if label not in main_vocab:
      raise ValueError(f"EMIS label not found in base categories: {label}")
  num_main = len(main_vocab)
  emis_base_metadata: list[list[int]] = []
  for row in base_attrs:
    labels = {main_vocab[label] for label in row.get("main_categories", [])}
    emis_base_metadata.append(sorted(labels))
  emis_query_metadata: list[list[int]] = [[main_vocab[row["label"]]] for row in emis_queries]
  emis_base_category_sets: list[set[int]] = [
    {main_vocab[label] for label in row.get("main_categories", [])} for row in base_attrs
  ]
  write_spmat(emis_dir / "base_metadata.spmat", emis_base_metadata, num_main)
  write_spmat(emis_dir / "query_metadata.spmat", emis_query_metadata, num_main)
  emis_query_labels = np.asarray([main_vocab[row["label"]] for row in emis_queries], dtype=np.int64)
  emis_ground_truth = read_ivec_first_k(
    source_root / "ground_truth_emis.ivecs", query_count=query_count, k=GT_K
  )
  _validate_emis_ground_truth(emis_ground_truth, emis_query_labels, emis_base_category_sets)
  write_matrix_u32(emis_dir / "groundtruth.ibin", emis_ground_truth)
  gt_emis = np.memmap(
    emis_dir / "groundtruth.ibin", dtype=np.uint32, mode="r", offset=8, shape=(query_count, GT_K)
  )
  workloads = [
    write_subset(
      emis_dir,
      correctness_name,
      query_ids,
      queries,
      emis_query_metadata,
      gt_emis,
      "query_metadata.spmat",
      "emis",
      num_main,
    ),
    write_subset(
      emis_dir,
      throughput_name,
      query_ids,
      queries,
      emis_query_metadata,
      gt_emis,
      "query_metadata.spmat",
      "emis",
      num_main,
    ),
  ]
  (emis_dir / "manifest.json").write_text(
    json.dumps(
      {"predicate": "emis", "workloads": workloads, "query_rows": query_count}
    )
    + "\n"
  )

  # R predicate: query defines [range_start, range_end] over candidate update_date.
  r_dir = output / "r"
  r_dir.mkdir(parents=True, exist_ok=True)
  r_base_metadata = np.asarray([int(row["update_date"]) for row in base_attrs], dtype=np.int32)
  write_range_metadata(r_dir / "base_metadata.rmeta", r_base_metadata)
  r_query_metadata = np.asarray(
    [[int(row["range_start"]), int(row["range_end"])] for row in r_queries],
    dtype=np.int32,
  )
  if r_query_metadata.ndim != 2 or r_query_metadata.shape[1] != 2:
    raise ValueError("invalid r range metadata")
  if np.any(r_query_metadata[:, 0] > r_query_metadata[:, 1]):
    raise ValueError("invalid r query range (start > end)")
  write_range_metadata(r_dir / "query_metadata.rmeta", r_query_metadata)
  r_ground_truth = read_ivec_first_k(
    source_root / "ground_truth_r.ivecs", query_count=query_count, k=GT_K
  )
  _validate_r_ground_truth(r_ground_truth, r_query_metadata, r_base_metadata)
  write_matrix_u32(r_dir / "groundtruth.ibin", r_ground_truth)
  gt_r = np.memmap(
    r_dir / "groundtruth.ibin", dtype=np.uint32, mode="r", offset=8, shape=(query_count, GT_K)
  )
  workloads = [
    write_subset(
      r_dir,
      correctness_name,
      query_ids,
      queries,
      r_query_metadata,
      gt_r,
      "query_metadata.rmeta",
      "r",
      2,
    ),
    write_subset(
      r_dir,
      throughput_name,
      query_ids,
      queries,
      r_query_metadata,
      gt_r,
      "query_metadata.rmeta",
      "r",
      2,
    ),
  ]
  (r_dir / "manifest.json").write_text(
    json.dumps({"predicate": "r", "workloads": workloads, "query_rows": query_count}) + "\n"
  )

  (output / "manifest.json").write_text(
    json.dumps(
      {
        "name": output.name,
        "base_file": "base.fbin",
        "query_file": "query.fbin",
        "queries": len(queries),
        "predicates": ["em", "emis", "r"],
      }
    )
    + "\n"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--dataset-name",
    type=str,
    default="arxiv-for-fanns-medium",
    help="Dataset directory under output",
  )
  parser.add_argument(
    "--query-count",
    type=int,
    default=DEFAULT_QUERY_COUNT,
    help="Number of queries to prepare.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  source_root = args.source
  dataset_dir = args.output / args.dataset_name
  dataset_dir.mkdir(parents=True, exist_ok=True)

  base_vectors = read_fvec_matrix(source_root / "database_vectors.fvecs")
  query_vectors = read_fvec_matrix(source_root / "query_vectors.fvecs")
  if query_vectors.shape[1] != base_vectors.shape[1]:
    raise ValueError("query and base dimensions differ")
  if args.query_count <= 0:
    raise ValueError("query-count must be positive")
  if query_vectors.shape[0] < args.query_count:
    raise ValueError(
      f"query_vectors has only {query_vectors.shape[0]} rows, expected >= {args.query_count}"
    )

  build_workloads(
    dataset_dir, base_vectors, query_vectors[: args.query_count], source_root
  )


if __name__ == "__main__":
  main()
