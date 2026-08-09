#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
data_root=${NAVIX_DATA_ROOT:-"${repo_dir}/datasets"}
result_root=${NAVIX_RESULT_ROOT:-"${repo_dir}/benchmarks/navix_single_cta/results_in_kernel_seed_20260809"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
stage=${1:-all}

mkdir -p "${result_root}/configs" "${result_root}/raw"
"${python_bin}" "${repo_dir}/benchmarks/navix_single_cta/generate_configs.py" \
  --output "${result_root}/configs" --data-root "${data_root}"

run_one() {
  local name=$1
  local mode=${2:-throughput}
  local min_time=${3:-0.05s}
  local output_name=${4:-${name}}
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    CUVS_NAVIX_LOG_RESOURCES="${CUVS_NAVIX_LOG_RESOURCES:-0}" \
    "${bench_bin}" --search --mode="${mode}" --threads=1 \
    --data_prefix="${data_root}" --index_prefix="${data_root}" \
    --benchmark_repetitions=1 --benchmark_min_time="${min_time}" \
    --benchmark_min_warmup_time=0.01 --benchmark_report_aggregates_only=false \
    --benchmark_out_format=json --benchmark_out="${result_root}/raw/${output_name}.json" \
    "${result_root}/configs/${name}.json"
}

run_reduced_sweep() {
  local workload=$1
  run_one "${workload}_sweep_i0"
  if ! "${python_bin}" "${repo_dir}/benchmarks/navix_single_cta/reached_target.py" \
      "${result_root}/raw/${workload}_sweep_i0.json"; then
    run_one "${workload}_sweep_i522"
    if ! "${python_bin}" "${repo_dir}/benchmarks/navix_single_cta/reached_target.py" \
        "${result_root}/raw/${workload}_sweep_i522.json"; then
      run_one "${workload}_sweep_i1044"
    fi
  fi
}

case "${stage}" in
  smoke) run_one yfcc_smoke latency 0.01s ;;
  correctness)
    run_one yfcc_correctness throughput 0.01s
    run_one emis_correctness throughput 0.01s
    ;;
  scheduler_gate) run_one emis_scheduler_gate ;;
  resources)
    CUVS_NAVIX_LOG_RESOURCES=1 run_one \
      emis_scheduler_gate throughput 0.01s emis_scheduler_resources \
      2>&1 | tee "${result_root}/navix_resources.log"
    ;;
  sweep)
    run_reduced_sweep yfcc
    run_reduced_sweep emis
    run_one em_b0
    run_one r_b0
    ;;
  arxiv_b0)
    run_one em_b0
    run_one r_b0
    ;;
  diagnosis)
    run_one yfcc_navix_diag_b0_l64_w1 throughput 0.01s
    run_one yfcc_navix_diag_i1044_l64_w1 throughput 0.01s
    run_one yfcc_navix_diag_i1044_l512_w2 throughput 0.01s
    ;;
  all)
    run_one yfcc_smoke latency 0.01s
    run_one yfcc_correctness throughput 0.01s
    run_one emis_correctness throughput 0.01s
    run_one emis_scheduler_gate
    run_reduced_sweep yfcc
    run_reduced_sweep emis
    run_one em_b0
    run_one r_b0
    ;;
  *)
    echo "usage: $0 {smoke|correctness|scheduler_gate|resources|sweep|arxiv_b0|diagnosis|all}" >&2
    exit 2
    ;;
esac
