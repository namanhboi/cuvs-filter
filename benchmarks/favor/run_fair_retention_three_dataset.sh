#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
single_root=${1:-"${repo_dir}/benchmarks/favor/results_fair_retention_single_full"}
multi_root=${2:-"${repo_dir}/benchmarks/favor/results_fair_retention_multi_full"}
single_comparison_root=${3:-"${repo_dir}/benchmarks/favor/results_fair_retention_single_comparison"}
multi_comparison_root=${4:-"${repo_dir}/benchmarks/favor/results_fair_retention_multi_comparison"}
single_cell_source="${repo_dir}/benchmarks/favor/results_retention_safe_single_full"

check_existing_result_hardware() {
  local result_root=$1
  [[ -d "${result_root}" ]] || return 0
  local current_gpu
  current_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
  python - "${result_root}" "${current_gpu}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
current_gpu = sys.argv[2]
mismatches = []
for path in sorted(root.glob("*/raw/*.json")):
    try:
        recorded_gpu = json.loads(path.read_text())["context"]["gpu_name"]
    except (OSError, KeyError, json.JSONDecodeError):
        continue
    if recorded_gpu != current_gpu:
        mismatches.append(f"{path}: {recorded_gpu}")
if mismatches:
    details = "\n".join(f"  {value}" for value in mismatches)
    raise SystemExit(
        f"refusing to mix benchmark hardware; current GPU is {current_gpu}:\n{details}"
    )
PY
}

extract_single_cells() {
  python - "$1" "$2" <<'PY'
import json
import sys

def cells(path):
    rows = json.load(open(path, encoding="utf-8"))["index"][0]["search_params"]
    return {
        (
            int(row["itopk"]),
            int(row["search_width"]),
            int(row.get("max_iterations", 0)),
            int(row.get("thread_block_size", 0)),
        )
        for row in rows
        if row.get("favor_penalty_mode") == "cagra_retention_safe"
    }

latency = cells(sys.argv[1])
throughput = cells(sys.argv[2])
if not latency or latency != throughput:
    raise SystemExit("SINGLE_CTA source tuning cells are missing or mismatched")
print(" ".join(":".join(map(str, cell)) for cell in sorted(latency)))
PY
}

run_single_dataset() {
  local key=$1
  local dataset_name=$2
  local result_prefix=$3
  local plot_title=$4
  local query_file=$5
  local base_file=$6
  local dtype=$7
  local subset_size=$8
  local dataset_result="${single_root}/${key}"

  for selectivity in 1 10 50 90; do
    local encoded
    printf -v encoded "%02d" "${selectivity}"
    local source_dir="${single_cell_source}/${key}/configs"
    local cells
    cells=$(extract_single_cells \
      "${source_dir}/${result_prefix}_s${encoded}_nq10.json" \
      "${source_dir}/${result_prefix}_s${encoded}_nq10000.json")

    FAVOR_BATCH_SIZES="10 10000" \
    FAVOR_SELECTIVITIES="${selectivity}" \
    FAVOR_SEARCH_CELLS="${cells}" \
    FAVOR_SEARCH_ALGO=single_cta \
    FAVOR_BENCHMARK_MODES="default favor_retention_safe" \
    FAVOR_PENALTY_LAMBDAS=1 \
    FAVOR_RETENTION_FRACTIONS="0.5 0" \
    FAVOR_BENCHMARK_REPETITIONS=1 \
    FAVOR_BENCHMARK_MIN_TIME=0.2s \
    FAVOR_BENCHMARK_WARMUP_TIME=0.1 \
      "${repo_dir}/benchmarks/favor/run_benchmarks.sh" \
        "${repo_dir}/datasets" \
        "${dataset_result}" \
        "${dataset_name}" \
        "${result_prefix}" \
        "${plot_title}" \
        "${query_file}" \
        "${base_file}" \
        "${dtype}" \
        "${subset_size}"
  done

  python "${repo_dir}/benchmarks/favor/plot_results.py" \
    --result-dir "${dataset_result}" \
    --output-dir "${single_comparison_root}/${key}" \
    --result-prefix "${result_prefix}" \
    --plot-title "${plot_title}" \
    --selectivities 1 10 50 90 \
    --penalty-lambdas 1 \
    --cta-mode SINGLE_CTA \
    --target-recall 0.90
}

run_multi_dataset() {
  local key=$1
  local dataset_name=$2
  local result_prefix=$3
  local plot_title=$4
  local query_file=$5
  local base_file=$6
  local dtype=$7
  local subset_size=$8
  local itopk_values=$9
  local dataset_result="${multi_root}/${key}"

  FAVOR_BATCH_SIZES=1 \
  FAVOR_SELECTIVITIES="1 10 50 90" \
  FAVOR_ITOPK_VALUES="${itopk_values}" \
  FAVOR_SEARCH_WIDTHS=1 \
  FAVOR_SEARCH_CELLS= \
  FAVOR_MAX_ITERATIONS=0 \
  FAVOR_THREAD_BLOCK_SIZES=0 \
  FAVOR_SEARCH_ALGO=multi_cta \
  FAVOR_DEDUPLICATE_MULTI_CTA=0 \
  FAVOR_BENCHMARK_MODES="default favor_retention_safe" \
  FAVOR_PENALTY_LAMBDAS=1 \
  FAVOR_RETENTION_FRACTIONS="0.5 0" \
  FAVOR_BENCHMARK_REPETITIONS=1 \
  FAVOR_BENCHMARK_MIN_TIME=0.2s \
  FAVOR_BENCHMARK_WARMUP_TIME=0.1 \
    "${repo_dir}/benchmarks/favor/run_benchmarks.sh" \
      "${repo_dir}/datasets" \
      "${dataset_result}" \
      "${dataset_name}" \
      "${result_prefix}" \
      "${plot_title}" \
      "${query_file}" \
      "${base_file}" \
      "${dtype}" \
      "${subset_size}"

  python "${repo_dir}/benchmarks/favor/plot_results.py" \
    --result-dir "${dataset_result}" \
    --output-dir "${multi_comparison_root}/${key}" \
    --result-prefix "${result_prefix}" \
    --plot-title "${plot_title}" \
    --selectivities 1 10 50 90 \
    --latency-derived-qps \
    --latency-batch-size 1 \
    --latency-unit us \
    --penalty-lambdas 1 \
    --cta-mode MULTI_CTA \
    --target-recall 0.99
}

check_existing_result_hardware "${single_root}"
check_existing_result_hardware "${multi_root}"

run_single_dataset gist gist-960-euclidean gist GIST-1M \
  query_10000.fbin base.fbin float 0
run_single_dataset bigann1m bigann-1M bigann1m BIGANN-1M \
  query.public.10K.u8bin base.10M.u8bin uint8 1000000
run_single_dataset msturing1m msturing-1M msturing1m MSTuring-1M \
  query.fbin base.fbin float 0

run_multi_dataset gist gist-960-euclidean gist GIST-1M \
  query_10000.fbin base.fbin float 0 "32 64 128 256 512 640 768 1024"
run_multi_dataset bigann1m bigann-1M bigann1m BIGANN-1M \
  query.public.10K.u8bin base.10M.u8bin uint8 1000000 "32 64 128 256 512 1024"
run_multi_dataset msturing1m msturing-1M msturing1m MSTuring-1M \
  query.fbin base.fbin float 0 "32 64 128 256 512 640 768 1024"
