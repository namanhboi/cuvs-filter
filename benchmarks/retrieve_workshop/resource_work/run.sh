#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
result_root=${RETRIEVE_RESOURCE_WORK_ROOT:-"${script_dir}/results/resource_work_$(date -u +%Y%m%d_%H%M%S)"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
build_libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
stage=${1:-all}

generate() {
  "${python_bin}" "${script_dir}/generate_configs.py" \
    --output "${result_root}/configs" \
    --data-root "${data_root}" \
    --diagnostic-root "${result_root}/diagnostics"
}

require_runtime() {
  test -x "${bench_bin}" || { echo "missing benchmark binary: ${bench_bin}" >&2; exit 2; }
  test -f "${build_libcuvs}" || { echo "missing libcuvs: ${build_libcuvs}" >&2; exit 2; }
}

capture_provenance() {
  "${python_bin}" "${script_dir}/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" \
    --bench-bin "${bench_bin}" --libcuvs "${build_libcuvs}" \
    --data-root "${data_root}"
}

run_mode() {
  local mode=$1
  mkdir -p "${result_root}/raw/${mode}" "${result_root}/resources"
  while IFS= read -r workload; do
    local config="${result_root}/configs/${mode}/${workload}.json"
    local raw="${result_root}/raw/${mode}/${workload}.json"
    local log="${result_root}/resources/${workload}.log"
    test -f "${config}" || { echo "missing config: ${config}" >&2; exit 2; }
    if test -e "${raw}" && test "${RETRIEVE_ALLOW_OVERWRITE:-0}" != 1; then
      echo "refusing to overwrite ${raw}" >&2
      exit 2
    fi
    if test "${mode}" = resources; then
      env CUVS_CAGRA_LOG_RESOURCES=1 \
        LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${bench_bin}" --search --mode=throughput --threads=1 \
        --data_prefix="${data_root}" --index_prefix="${data_root}" \
        --benchmark_repetitions=1 --benchmark_min_time=0.001s \
        --benchmark_min_warmup_time=0.001 \
        --benchmark_report_aggregates_only=false \
        --benchmark_out_format=json --benchmark_out="${raw}" "${config}" 2>"${log}"
    else
      env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${bench_bin}" --search --mode=throughput --threads=1 \
        --data_prefix="${data_root}" --index_prefix="${data_root}" \
        --benchmark_repetitions=1 --benchmark_min_time=0.001s \
        --benchmark_min_warmup_time=0.001 \
        --benchmark_report_aggregates_only=false \
        --benchmark_out_format=json --benchmark_out="${raw}" "${config}"
    fi
  done < <("${python_bin}" -c \
    'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["workloads"]))' \
    "${result_root}/configs/manifest.json")
}

analyze() {
  "${python_bin}" "${script_dir}/analyze.py" --result-root "${result_root}"
}

case "${stage}" in
  generate)
    generate
    ;;
  resources)
    require_runtime
    capture_provenance
    run_mode resources
    ;;
  diagnostics)
    require_runtime
    capture_provenance
    run_mode diagnostics
    ;;
  analyze)
    analyze
    ;;
  all)
    generate
    require_runtime
    capture_provenance
    run_mode resources
    run_mode diagnostics
    analyze
    ;;
  *)
    echo "usage: $0 {generate|resources|diagnostics|analyze|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
