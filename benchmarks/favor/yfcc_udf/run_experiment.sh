#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../../.."; pwd)
data_root=${YFCC_DATA_ROOT:-"${repo_dir}/datasets"}
result_root=${YFCC_RESULT_ROOT:-"${repo_dir}/benchmarks/favor/yfcc_udf/results"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
stage=${1:-all}

mkdir -p "${result_root}/configs" "${result_root}/raw"
"${python_bin}" "${repo_dir}/benchmarks/favor/yfcc_udf/generate_configs.py" \
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
  smoke) run_one smoke latency 0.01s ;;
  correctness) run_one correctness throughput 0.01s ;;
  throughput) run_one throughput throughput 0.2s ;;
  latency)
    for arity in 1 2; do
      for decile in $(seq 1 10); do
        run_one "latency_a${arity}_d${decile}" latency 0.05s
      done
    done
    ;;
  diagnostic) run_one diagnostic throughput 0.001s 0 ;;
  diagnostic_groups)
    for arity in 1 2; do
      for decile in $(seq 1 10); do
        run_one "diagnostic_latency_a${arity}_d${decile}" throughput 0.001s 0
      done
    done
    ;;
  all)
    run_one smoke latency 0.01s
    run_one correctness throughput 0.01s
    run_one throughput throughput 0.2s
    for arity in 1 2; do
      for decile in $(seq 1 10); do
        run_one "latency_a${arity}_d${decile}" latency 0.05s
      done
    done
    run_one diagnostic throughput 0.001s 0
    for arity in 1 2; do
      for decile in $(seq 1 10); do
        run_one "diagnostic_latency_a${arity}_d${decile}" throughput 0.001s 0
      done
    done
    ;;
  *) echo "usage: $0 {smoke|correctness|throughput|latency|diagnostic|diagnostic_groups|all}" >&2; exit 2 ;;
esac
