#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
result_dir=${2:-"${script_dir}/results"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
config_dir="${result_dir}/configs"
repetitions=${FAVOR_MULTI_SEED_REPETITIONS:-3}
min_time=${FAVOR_MULTI_SEED_MIN_TIME:-0.2s}
warmup_time=${FAVOR_MULTI_SEED_WARMUP_TIME:-0.1}
force_rerun=${FAVOR_MULTI_SEED_FORCE_RERUN:-0}

if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
  echo "Build artifacts are missing; build CUVS_CAGRA_ANN_BENCH first." >&2
  exit 1
fi
if [[ ! "${repetitions}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FAVOR_MULTI_SEED_REPETITIONS must be a positive integer." >&2
  exit 1
fi

python "${script_dir}/generate_configs.py" \
  --output-dir "${config_dir}" \
  --delta-root "${data_dir}"
mkdir -p "${result_dir}/raw"

is_complete_result() {
  python - "$1" "$2" "${repetitions}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as result_stream:
        rows = json.load(result_stream)["benchmarks"]
    with open(sys.argv[2], encoding="utf-8") as config_stream:
        parameter_count = len(json.load(config_stream)["index"][0]["search_params"])
    repetitions = int(sys.argv[3])
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)

iterations = [row for row in rows if row.get("run_type") == "iteration"]
completed = {
    (row.get("run_name"), row.get("repetition_index")) for row in iterations
}
has_error = any(row.get("error_occurred", False) for row in rows)
expected = parameter_count * repetitions
raise SystemExit(0 if len(iterations) >= expected and len(completed) >= expected and not has_error else 1)
PY
}

run_config() {
  local dataset=$1
  local batch_size=$2
  local group=$3
  local adaptive=$4
  local mode=latency
  if [[ "${batch_size}" == 10000 ]]; then mode=throughput; fi
  local config="${config_dir}/${dataset}_nq${batch_size}_${group}.json"
  local output="${result_dir}/raw/${dataset}_nq${batch_size}_${group}.json"
  if [[ "${force_rerun}" != 1 ]] && is_complete_result "${output}" "${config}"; then
    echo "Reusing complete result ${output}"
    return
  fi
  set +e
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION="${adaptive}" \
    "${bench_bin}" \
    --search \
    --mode="${mode}" \
    --threads=1 \
    --data_prefix="${data_dir}" \
    --index_prefix="${data_dir}" \
    --benchmark_repetitions="${repetitions}" \
    --benchmark_min_time="${min_time}" \
    --benchmark_min_warmup_time="${warmup_time}" \
    --benchmark_report_aggregates_only=false \
    --benchmark_out_format=json \
    --benchmark_out="${output}" \
    "${config}"
  local status=$?
  set -e
  if ! is_complete_result "${output}" "${config}"; then
    echo "Benchmark failed with status ${status} and did not produce a complete result." >&2
    if [[ "${status}" -ne 0 ]]; then return "${status}"; fi
    return 1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Benchmark data is complete; ignoring teardown-only status ${status}." >&2
  fi
}

cd "${repo_dir}"
for dataset in gist msturing1m msturing10m; do
  for batch_size in 10 10000; do
    run_config "${dataset}" "${batch_size}" controls 0
    run_config "${dataset}" "${batch_size}" multiseed 0
    run_config "${dataset}" "${batch_size}" adaptive 1
  done
done

python "${script_dir}/analyze.py" \
  --config-dir "${config_dir}" \
  --result-dir "${result_dir}"
