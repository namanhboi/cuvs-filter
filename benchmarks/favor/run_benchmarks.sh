#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
result_dir=${2:-"${repo_dir}/benchmarks/favor/results"}
dataset_name=${3:-"sift-128-euclidean"}
result_prefix=${4:-"sift"}
plot_title=${5:-"SIFT-1M"}
query_file=${6:-"query.fbin"}
base_file=${7:-"base.fbin"}
dtype=${8:-"float"}
subset_size=${9:-0}
force_rerun=${FAVOR_FORCE_RERUN:-0}
favor_delta_d=${FAVOR_DELTA_D:-}
selectivities=${FAVOR_SELECTIVITIES:-"1 10 50 90"}
itopk_values=${FAVOR_ITOPK_VALUES:-"32 64 128 256 512"}
search_widths=${FAVOR_SEARCH_WIDTHS:-"1 2 4"}
batch_sizes=${FAVOR_BATCH_SIZES:-"10 10000"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
config_dir="${result_dir}/configs"
mkdir -p "${config_dir}" "${result_dir}/raw"

if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
  echo "Build artifacts are missing; run ./build.sh libcuvs bench-ann -n first." >&2
  exit 1
fi

generate_args=(
  --output-dir "${config_dir}"
  --dataset-name="${dataset_name}"
  --result-prefix="${result_prefix}"
  --base-file="${base_file}"
  --query-file="${query_file}"
  --dtype="${dtype}"
  --delta-d-file="${data_dir}/${dataset_name}/cagra_g32_ig64.index.delta_d"
  --selectivities ${selectivities}
  --itopk-values ${itopk_values}
  --search-widths ${search_widths}
)
if [[ "${subset_size}" -gt 0 ]]; then
  generate_args+=(--subset-size="${subset_size}")
fi
if [[ -n "${favor_delta_d}" ]]; then
  generate_args+=(--favor-delta-d="${favor_delta_d}")
fi
python "${repo_dir}/benchmarks/favor/generate_configs.py" "${generate_args[@]}"

is_complete_result() {
  python - "$1" "$2" <<'PY'
import json
import sys

try:
    rows = json.load(open(sys.argv[1], encoding="utf-8"))["benchmarks"]
    expected = len(json.load(open(sys.argv[2], encoding="utf-8"))["index"][0]["search_params"])
except (OSError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)

iterations = [row for row in rows if row.get("run_type") == "iteration"]
configs = {row["run_name"] for row in iterations}
has_error = any(row.get("error_occurred", False) for row in rows)
raise SystemExit(
    0 if len(iterations) >= expected and len(configs) >= expected and not has_error else 1
)
PY
}

for selectivity_value in ${selectivities}; do
  printf -v selectivity "%02d" "${selectivity_value}"
  for batch_size in ${batch_sizes}; do
    if [[ "${batch_size}" == 10000 ]]; then
      mode=throughput
    else
      mode=latency
    fi
    result_file="${result_dir}/raw/${result_prefix}_s${selectivity}_nq${batch_size}.json"
    config_file="${config_dir}/${result_prefix}_s${selectivity}_nq${batch_size}.json"
    if [[ "${force_rerun}" != 1 ]] && is_complete_result "${result_file}" "${config_file}"; then
      echo "Reusing complete result ${result_file}"
      continue
    fi
    set +e
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" "${bench_bin}" \
      --search \
      --mode="${mode}" \
      --threads=1 \
      --data_prefix="${data_dir}" \
      --index_prefix="${data_dir}" \
      --benchmark_repetitions=1 \
      --benchmark_min_time=0.2s \
      --benchmark_min_warmup_time=0.1 \
      --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json \
      --benchmark_out="${result_file}" \
      "${config_file}"
    status=$?
    set -e
    if ! is_complete_result "${result_file}" "${config_file}"; then
      echo "Benchmark failed with status ${status} and did not produce a complete result." >&2
      [[ "${status}" -ne 0 ]] && exit "${status}"
      exit 1
    fi
    if [[ "${status}" -ne 0 ]]; then
      echo "Benchmark data is complete; ignoring teardown-only status ${status}." >&2
    fi
  done
done

if [[ " ${batch_sizes} " == *" 10 "* && " ${batch_sizes} " == *" 10000 "* ]]; then
  python "${repo_dir}/benchmarks/favor/plot_results.py" \
    --result-dir "${result_dir}" \
    --result-prefix="${result_prefix}" \
    --plot-title="${plot_title}" \
    --selectivities ${selectivities}
fi
