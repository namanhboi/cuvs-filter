#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
latency_k=${RETRIEVE_LATENCY_K:-10}
case "${latency_k}" in
  10)
    default_profile="${repo_dir}/benchmarks/retrieve_workshop/a100_paper/profiles/a100_yfcc10m_arxiv_large.json"
    default_exact_root="${data_root}/retrieve_workshop/exact_bitmap_a100"
    default_source_run_root=${RETRIEVE_A100_RUN_ROOT:-}
    ;;
  100)
    default_profile="${repo_dir}/benchmarks/retrieve_workshop/a100_k100/profiles/a100_yfcc10m_arxiv_large_k100.json"
    default_exact_root="${data_root}/retrieve_workshop/exact_bitmap_a100_k100"
    default_source_run_root=${RETRIEVE_A100_K100_RUN_ROOT:-}
    ;;
  *)
    echo "RETRIEVE_LATENCY_K must be 10 or 100, found ${latency_k}" >&2
    exit 2
    ;;
esac
profile=${RETRIEVE_DATASET_PROFILE:-"${default_profile}"}
result_root=${RETRIEVE_LATENCY_RESULT_ROOT:-"${repo_dir}/benchmarks/retrieve_workshop/per_query_latency/results_k${latency_k}"}
selected_points=${RETRIEVE_LATENCY_SELECTED_POINTS:-"${default_source_run_root}/matched_recall/analysis/selected_points.csv"}
selected_provenance=${RETRIEVE_LATENCY_SELECTED_PROVENANCE:-}
if test -z "${selected_provenance}"; then
  selected_provenance="$(dirname "${selected_points}")/provenance.json"
fi
exact_data_root=${RETRIEVE_LATENCY_EXACT_DATA_ROOT:-"${default_exact_root}"}
graph_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
exact_bin=${RETRIEVE_EXACT_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_BRUTE_FORCE_ANN_BENCH"}
libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
stage=${1:-all}

if test -z "${selected_points}" || ! test -f "${selected_points}"; then
  echo "missing selected points: ${selected_points}" >&2
  echo "set RETRIEVE_LATENCY_SELECTED_POINTS to the A100 matched-recall selected_points.csv" >&2
  exit 2
fi
if ! test -f "${selected_provenance}"; then
  echo "missing selected-point provenance: ${selected_provenance}" >&2
  echo "set RETRIEVE_LATENCY_SELECTED_PROVENANCE to matched_recall/analysis/provenance.json" >&2
  exit 2
fi
for path in "${profile}" "${graph_bin}" "${exact_bin}" "${libcuvs}"; do
  test -e "${path}" || { echo "missing latency input: ${path}" >&2; exit 2; }
done
for binary in "${graph_bin}" "${exact_bin}"; do
  grep -a -q benchmark_latency_trace_file "${binary}" || {
    echo "${binary} predates serialized-latency support; rebuild the benchmark binaries" >&2
    exit 2
  }
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON="${python_bin}"
export RETRIEVE_DATASET_PROFILE="${profile}"

mkdir -p "${result_root}/logs" "${result_root}/.done"
exec 9>"${result_root}.lock"
flock -n 9 || { echo "another latency stage owns ${result_root}.lock" >&2; exit 2; }

done_marker() { printf '%s/.done/%s\n' "${result_root}" "$1"; }
is_done() { test -f "$(done_marker "$1")"; }
mark_done() { date -u +%Y-%m-%dT%H:%M:%SZ >"$(done_marker "$1")"; }

validate_contract() {
  "${python_bin}" "${script_dir}/latency.py" validate-contract \
    --root "${result_root}" --selected-points "${selected_points}" \
    --selected-provenance "${selected_provenance}" --k "${latency_k}"
}

generate_configs() {
  if is_done configs; then
    validate_contract
    return
  fi
  if test -e "${result_root}/manifest.json"; then
    echo "refusing to replace an unmarked latency manifest: ${result_root}/manifest.json" >&2
    exit 2
  fi
  "${python_bin}" "${script_dir}/latency.py" generate \
    --root "${result_root}" --data-root "${data_root}" \
    --exact-data-root "${exact_data_root}" --selected-points "${selected_points}" \
    --selected-provenance "${selected_provenance}" \
    --profile "${profile}" --k "${latency_k}"
  mark_done configs
  validate_contract
}

capture_provenance() {
  if is_done provenance; then
    test -f "${result_root}/provenance/run.json" || {
      echo "latency provenance marker exists without provenance/run.json" >&2
      exit 2
    }
    validate_contract
    return
  fi
  "${python_bin}" "${script_dir}/latency.py" provenance \
    --root "${result_root}" --repo "${repo_dir}" \
    --selected-points "${selected_points}" --profile "${profile}" \
    --selected-provenance "${selected_provenance}" \
    --graph-binary "${graph_bin}" --exact-binary "${exact_bin}" --libcuvs "${libcuvs}" \
    --k "${latency_k}"
  mark_done provenance
  validate_contract
}

latency_cpu=${RETRIEVE_LATENCY_CPU:-$("${python_bin}" - <<'PY'
import os
print(min(os.sched_getaffinity(0)))
PY
)}

run_records() {
  local marker=$1
  shift
  if is_done "${marker}"; then
    return
  fi
  local listing="${result_root}/state_${marker}.tsv"
  local stage_args=()
  for requested in "$@"; do
    stage_args+=(--stage "${requested}")
  done
  "${python_bin}" "${script_dir}/latency.py" list-records \
    --manifest "${result_root}/manifest.json" "${stage_args[@]}" >"${listing}"
  while IFS=$'\t' read -r engine mode repetitions min_time config raw; do
    test -n "${config}" || continue
    mkdir -p "$(dirname "${raw}")" "${result_root}/logs/${marker}"
    if test -e "${raw}"; then
      if "${python_bin}" "${script_dir}/latency.py" validate-one \
        --manifest "${result_root}/manifest.json" --raw "${raw}" >/dev/null; then
        echo "resume: retaining complete latency output ${raw}"
        continue
      fi
      echo "refusing incomplete latency output ${raw}; remove that file and its matching trace files before retrying" >&2
      exit 2
    fi
    local binary="${graph_bin}"
    local data_prefix="${data_root}"
    local index_prefix="${data_root}"
    if test "${engine}" = exact; then
      binary="${exact_bin}"
      data_prefix=/
      index_prefix=/
    fi
    local random_interleaving=false
    test "${repetitions}" -gt 1 && random_interleaving=true
    local tag
    tag=$(printf '%s_%s' "$(basename "$(dirname "${config}")")" "$(basename "${config}" .json)")
    taskset -c "${latency_cpu}" env \
      LD_PRELOAD="${libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${binary}" --search --mode="${mode}" --threads=1 \
      --data_prefix="${data_prefix}" --index_prefix="${index_prefix}" \
      --benchmark_repetitions="${repetitions}" --benchmark_min_time="${min_time}" \
      --benchmark_min_warmup_time=0.001 \
      --benchmark_enable_random_interleaving="${random_interleaving}" \
      --benchmark_report_aggregates_only=false --benchmark_out_format=json \
      --benchmark_out="${raw}" "${config}" \
      2>&1 | tee "${result_root}/logs/${marker}/${tag}.log"
    "${python_bin}" "${script_dir}/latency.py" validate-one \
      --manifest "${result_root}/manifest.json" --raw "${raw}"
  done <"${listing}"
  mark_done "${marker}"
}

run_gate() {
  generate_configs
  capture_provenance
  run_records gate \
    gate_batch_graph gate_batch_exact gate_serial_graph gate_serial_exact
  "${python_bin}" "${script_dir}/latency.py" analyze-gate --root "${result_root}"
}

run_throughput_gate() {
  generate_configs
  capture_provenance
  run_records throughput_gate throughput_gate
  "${python_bin}" "${script_dir}/latency.py" analyze-throughput-gate --root "${result_root}"
}

run_trace() {
  generate_configs
  capture_provenance
  test -f "${result_root}/gate/analysis/batch_vs_serial_gate.json" || run_gate
  test -f "${result_root}/throughput_gate/analysis/throughput_regression_gate.json" || \
    run_throughput_gate
  run_records trace trace_graph trace_exact
}

analyze() {
  validate_contract
  "${python_bin}" "${script_dir}/latency.py" analyze-gate --root "${result_root}"
  "${python_bin}" "${script_dir}/latency.py" analyze-throughput-gate --root "${result_root}"
  "${python_bin}" "${script_dir}/latency.py" analyze --root "${result_root}"
}

case "${stage}" in
  configs) generate_configs ;;
  validate) validate_contract ;;
  gate) run_gate ;;
  throughput-gate) run_throughput_gate ;;
  trace) run_trace ;;
  analyze) analyze ;;
  all)
    generate_configs
    capture_provenance
    run_gate
    run_throughput_gate
    run_trace
    analyze
    ;;
  *)
    echo "usage: $0 {configs|validate|gate|throughput-gate|trace|analyze|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
