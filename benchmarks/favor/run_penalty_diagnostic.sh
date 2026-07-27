#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
result_dir=${2:-"${repo_dir}/benchmarks/favor/results_penalty_diagnostic"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"

python "${repo_dir}/benchmarks/favor/diagnose_penalty.py" \
  --data-dir="${data_dir}" --result-dir="${result_dir}"

is_complete_result() {
  python - "$1" "$2" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))["benchmarks"]
expected = len(json.load(open(sys.argv[2], encoding="utf-8"))["index"][0]["search_params"])
iterations = [row for row in result if row.get("run_type") == "iteration"]
has_error = any(row.get("error_occurred", False) for row in result)
raise SystemExit(0 if len(iterations) == expected and not has_error else 1)
PY
}

for config_file in "${result_dir}"/configs/*.json; do
  result_file="${result_dir}/raw/$(basename "${config_file}")"
  if is_complete_result "${result_file}" "${config_file}" 2>/dev/null; then
    echo "Reusing complete result ${result_file}"
    continue
  fi
  set +e
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" "${bench_bin}" \
    --search \
    --mode=throughput \
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
    echo "Incomplete diagnostic result (status ${status}): ${result_file}" >&2
    exit 1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Result complete; ignoring teardown-only status ${status}." >&2
  fi
done

python "${repo_dir}/benchmarks/favor/diagnose_penalty.py" \
  --result-dir="${result_dir}" --summarize
