#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../../.."; pwd)
data_root=${ARXIV_DATA_ROOT:-"${repo_dir}/datasets"}
result_root=${ARXIV_RESULT_ROOT:-"${repo_dir}/benchmarks/favor/arxiv_udf/results"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
stage=${1:-all}

mkdir -p "${result_root}/configs" "${result_root}/raw" "${result_root}/results"
"${python_bin}" "${repo_dir}/benchmarks/favor/arxiv_udf/generate_configs.py" \
  --output "${result_root}/configs" \
  --result-root "${result_root}" \
  --data-root "${data_root}"

run_one() {
  local name=$1
  local mode=$2
  local min_time=${3:-0.1s}
  local warmup_time=${4:-0.05}
  local config="${result_root}/configs/${name}.json"
  local output="${result_root}/raw/${name}.json"
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${bench_bin}" --search --mode="${mode}" --threads=1 \
    --data_prefix="${data_root}" --index_prefix="${data_root}" \
    --benchmark_repetitions=1 --benchmark_min_time="${min_time}" \
    --benchmark_min_warmup_time="${warmup_time}" \
    --benchmark_report_aggregates_only=false \
    --benchmark_out_format=json --benchmark_out="${output}" "${config}"
}

case "${stage}" in
  smoke)
    for predicate in em emis r; do
      run_one "${predicate}_smoke" latency 0.01s
    done
    ;;
  correctness)
    for predicate in em emis r; do
      run_one "${predicate}_correctness" throughput 0.01s
    done
    ;;
  accumulator_gate)
    for predicate in em emis r; do
      run_one "${predicate}_accumulator_gate" throughput 0.05s
    done
    ;;
  throughput)
    for predicate in em emis r; do
      run_one "${predicate}_throughput" throughput 0.2s
    done
    ;;
  all)
    for predicate in em emis r; do
      run_one "${predicate}_smoke" latency 0.01s
      run_one "${predicate}_accumulator_gate" throughput 0.05s
      run_one "${predicate}_correctness" throughput 0.01s
      run_one "${predicate}_throughput" throughput 0.2s
    done
    ;;
  *)
    echo "usage: $0 {smoke|accumulator_gate|correctness|throughput|all}" >&2
    exit 2
    ;;
esac
