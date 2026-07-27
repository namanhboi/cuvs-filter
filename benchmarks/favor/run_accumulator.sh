#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
result_dir=${1:?usage: run_accumulator.sh RESULT_DIR [DATA_DIR]}
data_dir=${2:-"${repo_dir}/datasets"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
force_rerun=${FAVOR_FORCE_RERUN:-0}

if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
  echo "Build artifacts are missing; run ./build.sh libcuvs bench-ann -n first." >&2
  exit 1
fi
if [[ ! -f "${result_dir}/manifest.json" ]]; then
  echo "Missing ${result_dir}/manifest.json; generate configurations first." >&2
  exit 1
fi
mkdir -p "${result_dir}/raw"

is_complete_result() {
  python - "$1" "$2" <<'PY'
import json
import re
import sys

try:
    rows = json.load(open(sys.argv[1], encoding="utf-8"))["benchmarks"]
    expected = len(json.load(open(sys.argv[2], encoding="utf-8"))["index"][0]["search_params"])
except (OSError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
indices = set()
for row in rows:
    if row.get("run_type", "iteration") != "iteration" or row.get("error_occurred", False):
        continue
    match = re.search(r"/(\d+)/", row.get("run_name", ""))
    if match:
        indices.add(int(match.group(1)))
raise SystemExit(0 if len(indices) == expected else 1)
PY
}

shopt -s nullglob
config_files=("${result_dir}"/configs/*.json)
if [[ ${#config_files[@]} -eq 0 ]]; then
  echo "No configurations found in ${result_dir}/configs." >&2
  exit 1
fi

for config_file in "${config_files[@]}"; do
  filename=$(basename "${config_file}")
  result_file="${result_dir}/raw/${filename}"
  batch_size=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["search_basic_param"]["batch_size"])' \
    "${config_file}")
  if [[ "${batch_size}" -eq 10000 ]]; then
    mode=throughput
  else
    mode=latency
  fi
  if [[ "${force_rerun}" != 1 ]] && is_complete_result "${result_file}" "${config_file}"; then
    echo "Reusing complete result ${result_file}"
    continue
  fi

  echo "Running ${filename}"
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
    echo "Benchmark failed with status ${status} and incomplete output." >&2
    [[ "${status}" -ne 0 ]] && exit "${status}"
    exit 1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Benchmark data is complete; ignoring teardown-only status ${status}." >&2
  fi
done

python "${repo_dir}/benchmarks/favor/accumulator_experiment.py" \
  summarize --result-dir "${result_dir}"
python "${repo_dir}/benchmarks/favor/accumulator_experiment.py" \
  compare --result-dir "${result_dir}"
python "${repo_dir}/benchmarks/favor/accumulator_experiment.py" \
  plot --result-dir "${result_dir}"
