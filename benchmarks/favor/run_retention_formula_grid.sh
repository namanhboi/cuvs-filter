#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
result_root=${1:-"${repo_dir}/benchmarks/favor/results_retention_formula_grid"}
source_root="${repo_dir}/benchmarks/favor/results_retention_safe_single_full"
penalty_lambdas=${FAVOR_PENALTY_LAMBDAS:-"0.5 1 2"}
retention_fractions=${FAVOR_RETENTION_FRACTIONS:-"0.25 0.5 0.75 0.9"}
all_cells=${FAVOR_RETENTION_ALL_CELLS:-0}

extract_cells() {
  python - "$1" "$2" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [
    row
    for row in payload["benchmarks"]
    if 'favor_penalty_mode="cagra_retention_safe"' in row.get("label", "")
]
if not rows:
    raise SystemExit("source result has no retention-safe measurements")

def cell(row):
    return (
        int(row["itopk"]),
        int(row["search_width"]),
        int(row.get("max_iterations", 0)),
        int(row.get("thread_block_size", 0)),
    )

if int(sys.argv[2]):
    cells = list(dict.fromkeys(cell(row) for row in rows))
    print(" ".join(":".join(map(str, value)) for value in cells))
    raise SystemExit

# Reuse exactly one historical traversal cell: the fastest point at or above
# the SINGLE_CTA target, or the highest-recall point if the target was missed.
target_rows = [row for row in rows if float(row.get("Recall", 0.0)) >= 0.90]
if target_rows:
    chosen = max(target_rows, key=lambda row: float(row.get("items_per_second", 0.0)))
else:
    chosen = max(rows, key=lambda row: float(row.get("Recall", 0.0)))

print(":".join(map(str, cell(chosen))))
PY
}

run_cell() {
  local source_key=$1
  local dataset_name=$2
  local result_prefix=$3
  local plot_title=$4
  local query_file=$5
  local base_file=$6
  local dtype=$7
  local subset_size=$8
  local selectivity=$9
  local source_result=
  source_result="${source_root}/${source_key}/raw/${result_prefix}_s$(printf '%02d' "${selectivity}")_nq10000.json"
  local cells=
  cells=$(extract_cells "${source_result}" "${all_cells}")

  FAVOR_BATCH_SIZES=10000 \
  FAVOR_SELECTIVITIES="${selectivity}" \
  FAVOR_SEARCH_CELLS="${cells}" \
  FAVOR_SEARCH_ALGO=single_cta \
  FAVOR_BENCHMARK_MODES="default favor_retention_safe" \
  FAVOR_PENALTY_LAMBDAS="${penalty_lambdas}" \
  FAVOR_RETENTION_FRACTIONS="${retention_fractions}" \
  FAVOR_BENCHMARK_REPETITIONS=1 \
  FAVOR_BENCHMARK_MIN_TIME=0.2s \
  FAVOR_BENCHMARK_WARMUP_TIME=0.1 \
    "${repo_dir}/benchmarks/favor/run_benchmarks.sh" \
      "${repo_dir}/datasets" \
      "${result_root}/${source_key}" \
      "${dataset_name}" \
      "${result_prefix}" \
      "${plot_title}" \
      "${query_file}" \
      "${base_file}" \
      "${dtype}" \
      "${subset_size}"
}

plot_dataset() {
  local source_key=$1
  local result_prefix=$2
  local plot_title=$3
  shift 3
  python "${repo_dir}/benchmarks/favor/plot_results.py" \
    --result-dir "${result_root}/${source_key}" \
    --result-prefix "${result_prefix}" \
    --plot-title "${plot_title}" \
    --selectivities "$@" \
    --latency-derived-qps \
    --latency-batch-size 10000 \
    --latency-unit us \
    --cta-mode SINGLE_CTA \
    --target-recall 0.90 \
    --qps-only \
    --zero-y
}

run_cell sift sift-128-euclidean sift SIFT-1M query.fbin base.fbin float 0 1
run_cell sift sift-128-euclidean sift SIFT-1M query.fbin base.fbin float 0 90
run_cell bigann1m bigann-1M bigann1m BIGANN-1M query.public.10K.u8bin base.10M.u8bin uint8 1000000 1
run_cell bigann1m bigann-1M bigann1m BIGANN-1M query.public.10K.u8bin base.10M.u8bin uint8 1000000 10
run_cell bigann10m bigann-10M bigann10m BIGANN-10M query.public.10K.u8bin base.10M.u8bin uint8 0 1
run_cell bigann10m bigann-10M bigann10m BIGANN-10M query.public.10K.u8bin base.10M.u8bin uint8 0 10
run_cell msturing10m msturing-10M msturing10m MSTuring-10M query.fbin base.fbin float 0 10
run_cell msturing10m msturing-10M msturing10m MSTuring-10M query.fbin base.fbin float 0 90
run_cell gist gist-960-euclidean gist GIST-1M query_10000.fbin base.fbin float 0 50
run_cell msturing1m msturing-1M msturing1m MSTuring-1M query.fbin base.fbin float 0 50

plot_dataset sift sift SIFT-1M 1 90
plot_dataset bigann1m bigann1m BIGANN-1M 1 10
plot_dataset bigann10m bigann10m BIGANN-10M 1 10
plot_dataset msturing10m msturing10m MSTuring-10M 10 90
plot_dataset gist gist GIST-1M 50
plot_dataset msturing1m msturing1m MSTuring-1M 50
