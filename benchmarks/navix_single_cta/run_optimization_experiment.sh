#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
data_root=${NAVIX_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
result_root=${NAVIX_RESULT_ROOT:-"${repo_dir}/benchmarks/navix_single_cta/results_gpu_opt_$(date -u +%Y%m%d_%H%M%S)"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
stage=${1:-all}

mkdir -p "${result_root}/configs" "${result_root}/raw"
"${python_bin}" "${repo_dir}/benchmarks/navix_single_cta/generate_optimization_configs.py" \
  --output "${result_root}/configs" --data-root "${data_root}"

run_group() {
  local group=$1
  local workload=$2
  local mode=${3:-throughput}
  local min_time=${4:-0.05s}
  local input="${result_root}/configs/${group}/${workload}/manifest.json"
  local destination="${result_root}/raw/${group}/${workload}"
  mkdir -p "${destination}"
  "${python_bin}" - "${input}" <<'PY' | while IFS= read -r config; do
import json, pathlib, sys
for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["configs"]:
    print(row["config"])
PY
    name=$(basename "${config}" .json)
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      CUVS_NAVIX_LOG_RESOURCES="${CUVS_NAVIX_LOG_RESOURCES:-0}" \
      "${bench_bin}" --search --mode="${mode}" --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions=1 --benchmark_min_time="${min_time}" \
      --benchmark_min_warmup_time=0.01 --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json --benchmark_out="${destination}/${name}.json" "${config}"
  done
}

run_b0() {
  for workload in yfcc em emis r; do run_group sweep_i0 "${workload}"; done
}

run_policy_b0() {
  for workload in yfcc em emis r; do run_group navix_policy_b0 "${workload}"; done
}

case "${stage}" in
  smoke) run_group correctness yfcc latency 0.01s ;;
  correctness)
    for workload in yfcc em emis r; do run_group correctness "${workload}" throughput 0.01s; done
    ;;
  scheduler_gate)
    for workload in em emis r; do run_group scheduler_gate "${workload}" throughput 0.05s; done
    ;;
  performance_gate)
    for workload in yfcc em emis r; do
      run_group performance_gate "${workload}" throughput 0.10s
    done
    ;;
  policy_b0) run_policy_b0 ;;
  resources)
    CUVS_NAVIX_LOG_RESOURCES=1 run_group scheduler_gate emis throughput 0.01s \
      2>&1 | tee "${result_root}/navix_resources.log"
    ;;
  b0) run_b0 ;;
  deeper)
    for iterations in 522 1044; do
      run_group "sweep_i${iterations}" yfcc
    done
    ;;
  all)
    run_group correctness yfcc latency 0.01s
    for workload in yfcc em emis r; do run_group correctness "${workload}" throughput 0.01s; done
    for workload in em emis r; do run_group scheduler_gate "${workload}" throughput 0.05s; done
    run_b0
    ;;
  *) echo "usage: $0 {smoke|correctness|scheduler_gate|performance_gate|policy_b0|resources|b0|deeper|all}" >&2; exit 2 ;;
esac

printf '%s\n' "${result_root}"
