#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
profile=${RETRIEVE_DATASET_PROFILE:-"${repo_dir}/benchmarks/retrieve_workshop/a100_k100/profiles/a100_yfcc10m_arxiv_large_k100.json"}
source_run_root=${RETRIEVE_A100_K100_RUN_ROOT:-/data/retrieve_workshop_runs/a100_k100_20260825T232747Z}
source_config_root=${RETRIEVE_A100_K100_CONFIG_ROOT:-"${source_run_root}/gpu_graph/configs/b0"}
reference_bundle=${RETRIEVE_A100_K100_REFERENCE_BUNDLE:-"${HOME}/a100_k100_matched_paper_gpu_bundle.tar.gz"}
run_tag=${RETRIEVE_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
result_root=${RETRIEVE_A100_K100_OCCUPANCY_ROOT:-"/data/retrieve_workshop_runs/a100_k100_retain_occupancy_${run_tag}"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
helper="${script_dir}/k100_retain_occupancy.py"
stage=${1:-all}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON="${python_bin}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export RETRIEVE_DATA_ROOT="${data_root}"
export RETRIEVE_DATASET_PROFILE="${profile}"
export RETRIEVE_PROVENANCE_K=100
export RETRIEVE_PROVENANCE_MAX_QUERIES=2048
export RETRIEVE_PROVENANCE_REPETITIONS=1

mkdir -p "$(dirname "${result_root}")"
exec 9>"${result_root}.lock"
flock -n 9 || {
  echo "another occupancy verifier owns ${result_root}.lock" >&2
  exit 2
}
mkdir -p "${result_root}/logs"
exec > >(tee -a "${result_root}/logs/orchestrator.log") 2>&1

require_generation_inputs() {
  test -f "${reference_bundle}" || {
    echo "missing k=100 reference bundle: ${reference_bundle}" >&2
    exit 2
  }
  test -f "${profile}" || { echo "missing profile: ${profile}" >&2; exit 2; }
  local workload
  for workload in yfcc em emis r; do
    test -f "${source_config_root}/${workload}/shard_00.json" || {
      echo "missing source configuration: ${source_config_root}/${workload}/shard_00.json" >&2
      exit 2
    }
  done
}

require_runtime() {
  test -x "${bench_bin}" || { echo "missing benchmark binary: ${bench_bin}" >&2; exit 2; }
  test -f "${libcuvs}" || { echo "missing libcuvs: ${libcuvs}" >&2; exit 2; }
}

generate() {
  require_generation_inputs
  "${python_bin}" "${helper}" generate \
    --source-config-root "${source_config_root}" \
    --reference-bundle "${reference_bundle}" \
    --output "${result_root}/configs"
}

build() {
  if test -f "${result_root}/.done/build"; then
    echo "resume: build already completed"
    return
  fi
  (cd "${repo_dir}" && env PARALLEL_LEVEL=${PARALLEL_LEVEL:-12} \
    ./build.sh libcuvs bench-ann -n \
      --limit-bench-ann=CUVS_CAGRA_ANN_BENCH --gpu-arch=80-real)
  require_runtime
  mkdir -p "${result_root}/.done"
  date -u +%Y-%m-%dT%H:%M:%SZ >"${result_root}/.done/build"
}

capture_provenance() {
  "${python_bin}" \
    "${repo_dir}/benchmarks/retrieve_workshop/gpu_graph/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" \
    --stage k100_retain_occupancy --bench-bin "${bench_bin}" \
    --libcuvs "${libcuvs}" --data-root "${data_root}"
}

run_cases() {
  require_runtime
  test -f "${result_root}/configs/manifest.json" || {
    echo "missing generated manifest; run the generate stage first" >&2
    exit 2
  }
  capture_provenance
  mkdir -p "${result_root}/raw" "${result_root}/resources"
  while IFS=$'\t' read -r workload config; do
    local raw="${result_root}/raw/${workload}.json"
    local log="${result_root}/resources/${workload}.log"
    if test -f "${raw}" && test -f "${log}"; then
      echo "resume: resource capture already exists for ${workload}"
      continue
    fi
    if test -e "${raw}" || test -e "${log}"; then
      echo "incomplete ${workload} capture in ${result_root}; use a new result root" >&2
      exit 2
    fi
    local raw_tmp="${raw}.tmp.$$"
    local log_tmp="${log}.tmp.$$"
    echo "capturing ${workload} Base/Retain launch resources"
    env CUVS_CAGRA_LOG_RESOURCES=1 \
      LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions=1 --benchmark_min_time=0.001s \
      --benchmark_min_warmup_time=0.001 \
      --benchmark_enable_random_interleaving=false \
      --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json --benchmark_out="${raw_tmp}" "${config}" \
      2>"${log_tmp}"
    mv "${raw_tmp}" "${raw}"
    mv "${log_tmp}" "${log}"
  done < <("${python_bin}" - "${result_root}/configs/manifest.json" <<'PY'
import json
import pathlib
import sys

for case in json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]:
    print(f"{case['workload']}\t{case['config']}")
PY
  )
}

analyze() {
  local status=0
  "${python_bin}" "${helper}" analyze --result-root "${result_root}" || status=$?
  printf '\n'
  sed -n '1,120p' "${result_root}/analysis/k100_retain_occupancy.txt"
  return "${status}"
}

case "${stage}" in
  build)
    build
    ;;
  generate)
    generate
    ;;
  run)
    run_cases
    ;;
  analyze)
    analyze
    ;;
  all)
    generate
    build
    run_cases
    analyze
    ;;
  *)
    echo "usage: $0 {build|generate|run|analyze|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
