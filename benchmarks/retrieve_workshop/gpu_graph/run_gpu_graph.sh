#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
result_root=${RETRIEVE_RESULT_ROOT:-"${script_dir}/results/gpu_graph_$(date -u +%Y%m%d_%H%M%S)"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
build_libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
throughput_min_time=${RETRIEVE_BENCH_MIN_TIME:-0.10s}
correctness_min_time=${RETRIEVE_CORRECTNESS_MIN_TIME:-0.01s}
stage=${1:-all}
shift || true

generate_configs() {
  mkdir -p "${result_root}/configs" "${result_root}/raw"
  "${python_bin}" "${script_dir}/generate_configs.py" \
    --output "${result_root}/configs" --data-root "${data_root}" "$@"
}

require_runtime() {
  test -x "${bench_bin}" || {
    echo "missing benchmark binary: ${bench_bin}" >&2
    exit 2
  }
  test -f "${build_libcuvs}" || {
    echo "missing libcuvs: ${build_libcuvs}" >&2
    exit 2
  }
  "${python_bin}" "${script_dir}/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" --stage "$1" \
    --bench-bin "${bench_bin}" --libcuvs "${build_libcuvs}" --data-root "${data_root}"
}

ensure_configs() {
  if ! test -f "${result_root}/configs/b0/yfcc/manifest.json"; then
    generate_configs
  fi
}

run_group_workload() {
  local group=$1
  local workload=$2
  local min_time=$3
  local resume_complete=${4:-0}
  local repetitions=${5:-3}
  local manifest="${result_root}/configs/${group}/${workload}/manifest.json"
  test -f "${manifest}" || return 0
  local destination="${result_root}/raw/${group}/${workload}"
  mkdir -p "${destination}"
  while IFS= read -r config; do
    local name
    name=$(basename "${config}" .json)
    local output="${destination}/${name}.json"
    if test -e "${output}" && test "${RETRIEVE_ALLOW_OVERWRITE:-0}" != 1; then
      if test "${resume_complete}" = 1 && test -s "${output}" && \
        "${python_bin}" - "${output}" "${config}" "${repetitions}" <<'PY'
import collections
import json
import pathlib
import sys

raw = json.loads(pathlib.Path(sys.argv[1]).read_text())
config = json.loads(pathlib.Path(sys.argv[2]).read_text())
repetitions = int(sys.argv[3])
searches = sum(len(index["search_params"]) for index in config["index"])
rows = [row for row in raw.get("benchmarks", []) if row.get("run_type") == "iteration"]
counts = collections.Counter(int(row.get("repetition_index", -1)) for row in rows)
valid = (
    len(rows) == searches * repetitions
    and counts == collections.Counter({repetition: searches for repetition in range(repetitions)})
    and all(not row.get("error_occurred") and not row.get("skipped") for row in rows)
)
raise SystemExit(0 if valid else 1)
PY
      then
        echo "resume: retain complete ${output}" >&2
        continue
      else
        echo "refusing to overwrite incomplete/unchecked ${output}; remove only this file and retry" >&2
        exit 2
      fi
    fi
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --search --mode=throughput --threads=1 \
      --data_prefix="${data_root}" --index_prefix="${data_root}" \
      --benchmark_repetitions="${repetitions}" --benchmark_min_time="${min_time}" \
      --benchmark_min_warmup_time=0.01 \
      --benchmark_enable_random_interleaving=true \
      --benchmark_report_aggregates_only=false \
      --benchmark_out_format=json --benchmark_out="${output}" "${config}"
  done < <(
    "${python_bin}" - "${manifest}" <<'PY'
import json
import pathlib
import sys

for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["configs"]:
    print(row["config"])
PY
  )
}

run_group() {
  local group=$1
  local min_time=$2
  local repetitions=${3:-3}
  local resume_complete=${RETRIEVE_RESUME_COMPLETE:-0}
  for workload in yfcc em emis r; do
    run_group_workload \
      "${group}" "${workload}" "${min_time}" "${resume_complete}" "${repetitions}"
  done
}

analyze_without_plots() {
  "${python_bin}" "${script_dir}/analyze_gpu_graph.py" \
    --result-root "${result_root}" --target-recall="${RETRIEVE_TARGET_RECALL:-0.90}" \
    --no-plots --skip-full-input-hashes >/dev/null
}

pair_reached_target() {
  local workload=$1
  local method=$2
  "${python_bin}" - "${result_root}/analysis/summary_points.csv" \
    "${workload}" "${method}" "${RETRIEVE_TARGET_RECALL:-0.90}" <<'PY'
import csv
import pathlib
import sys

path, workload, method, target = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
with path.open() as source:
    reached = any(
        row["phase"] == "throughput"
        and row["workload"] == workload
        and row["method"] == method
        and float(row["recall_median"]) >= target
        for row in csv.DictReader(source)
    )
raise SystemExit(0 if reached else 1)
PY
}

case "${stage}" in
  generate)
    generate_configs "$@"
    ;;
  correctness)
    ensure_configs
    require_runtime correctness
    run_group correctness "${correctness_min_time}" 1
    ;;
  b0)
    ensure_configs
    require_runtime b0
    run_group b0 "${throughput_min_time}" 3
    ;;
  deep)
    ensure_configs
    require_runtime deep
    analyze_without_plots
    while IFS=$'\t' read -r workload method; do
        test -f "${result_root}/configs/deep_i522_${workload}_${method}/${workload}/manifest.json" || continue
        if pair_reached_target "${workload}" "${method}"; then
          echo "skip deep ${workload}/${method}: B0 already reaches target" >&2
          continue
        fi
        for iterations in 522 1044 2088 4176 7569; do
          if test "${iterations}" -gt "${RETRIEVE_DEEP_MAX_ITERATION:-7569}"; then
            break
          fi
          run_group_workload "deep_i${iterations}_${workload}_${method}" "${workload}" \
            "${throughput_min_time}" 1 3
          analyze_without_plots
          if pair_reached_target "${workload}" "${method}"; then
            echo "stop deep ${workload}/${method} after i${iterations}: target reached" >&2
            break
          fi
        done
    done < <(
      "${python_bin}" - "${result_root}/configs/deep_plan.json" <<'PY'
import json
import pathlib
import sys

for pair in json.loads(pathlib.Path(sys.argv[1]).read_text())["pairs"]:
    print(f"{pair['workload']}\t{pair['method']}")
PY
    )
    ;;
  analyze)
    "${python_bin}" "${script_dir}/analyze_gpu_graph.py" \
      --result-root "${result_root}" --require-group=correctness --require-group=b0 "$@"
    ;;
  all)
    generate_configs "$@"
    require_runtime all
    run_group correctness "${correctness_min_time}" 1
    run_group b0 "${throughput_min_time}" 3
    "${python_bin}" "${script_dir}/analyze_gpu_graph.py" \
      --result-root "${result_root}" --require-group=correctness --require-group=b0
    ;;
  *)
    echo "usage: $0 {generate|correctness|b0|deep|analyze|all} [generator/analyzer args]" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
