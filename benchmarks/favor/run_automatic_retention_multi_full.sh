#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
result_root=${1:-"${repo_dir}/benchmarks/favor/results_automatic_retention_multi_full"}

check_dataset_inputs() {
  local dataset_name=$1
  local query_file=$2
  local base_file=$3
  local dataset_dir="${repo_dir}/datasets/${dataset_name}"
  local missing=()

  for path in \
    "${dataset_dir}/${base_file}" \
    "${dataset_dir}/${query_file}" \
    "${dataset_dir}/cagra_g32_ig64.index" \
    "${dataset_dir}/cagra_g32_ig64.index.delta_d"; do
    [[ -f "${path}" ]] || missing+=("${path}")
  done
  for selectivity in 01 10 50 90; do
    for artifact in \
      "filter_s${selectivity}.bin" \
      "groundtruth_s${selectivity}.ibin"; do
      local path="${dataset_dir}/favor/${artifact}"
      [[ -f "${path}" ]] || missing+=("${path}")
    done
  done

  if (( ${#missing[@]} > 0 )); then
    echo "Missing required inputs for ${dataset_name}:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "Run benchmarks/favor/prepare_automatic_retention_multi_data.sh first." >&2
    return 1
  fi
}

check_existing_result_hardware() {
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
        recorded_gpu = json.loads(path.read_text(encoding="utf-8"))["context"]["gpu_name"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        # An interrupted benchmark may leave a partial JSON file. The underlying runner's
        # completeness check will replace it, so it must not prevent restart here.
        print(f"ignoring incomplete result during hardware preflight: {path}: {error}", file=sys.stderr)
        continue
    if recorded_gpu != current_gpu:
        mismatches.append(f"{path}: {recorded_gpu}")

if mismatches:
    details = "\n".join(f"  {value}" for value in mismatches)
    raise SystemExit(
        f"refusing to mix benchmark hardware; current GPU is {current_gpu}:\n{details}\n"
        "Use a fresh result root or explicitly remove the stale result directory."
    )
PY
}

run_dataset() {
  local dataset_name=$1
  local result_prefix=$2
  local plot_title=$3
  local query_file=$4
  local base_file=$5
  local dtype=$6
  local subset_size=$7
  local itopk_values=$8
  local dataset_result="${result_root}/${result_prefix}"

  check_dataset_inputs "${dataset_name}" "${query_file}" "${base_file}"

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
  FAVOR_DELTA_D= \
  FAVOR_MAX_QUERIES=0 \
  FAVOR_ADAPTIVE_TERMINATION=0 \
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
    --result-prefix "${result_prefix}" \
    --plot-title "${plot_title}" \
    --selectivities 1 10 50 90 \
    --latency-derived-qps \
    --latency-batch-size 1 \
    --latency-unit us \
    --penalty-lambdas 1 \
    --cta-mode MULTI_CTA \
    --target-recall 0.99 \
    --zero-y
}

check_existing_result_hardware

run_dataset sift-128-euclidean sift SIFT-1M query.fbin base.fbin float 0 \
  "32 64 128 256 512 1024"
run_dataset gist-960-euclidean gist GIST-1M query_10000.fbin base.fbin float 0 \
  "32 64 128 256 512 640 768 1024"
run_dataset bigann-1M bigann1m BIGANN-1M \
  query.public.10K.u8bin base.10M.u8bin uint8 1000000 \
  "32 64 128 256 512 1024"
run_dataset bigann-10M bigann10m BIGANN-10M \
  query.public.10K.u8bin base.10M.u8bin uint8 0 \
  "32 64 128 256 512 1024"
run_dataset msturing-1M msturing1m MSTuring-1M query.fbin base.fbin float 0 \
  "32 64 128 256 512 640 768 1024"
run_dataset msturing-10M msturing10m MSTuring-10M query.fbin base.fbin float 0 \
  "32 64 128 256 512 1024 1536 2048"
