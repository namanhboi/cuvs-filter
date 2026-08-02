#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
result_dir=${2:-"${script_dir}/results"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
config_dir="${result_dir}/configs"
capture_dir="${result_dir}/captures"
raw_dir="${result_dir}/raw"
force_rerun=${FAVOR_STATE_RETRY_FORCE_RERUN:-0}

if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
  echo "Build artifacts are missing; build CUVS_CAGRA_ANN_BENCH first." >&2
  exit 1
fi

python "${script_dir}/generate_configs.py" \
  --output-dir "${config_dir}" \
  --data-root "${data_dir}" \
  --capture-root "${capture_dir}"
mkdir -p "${raw_dir}"

is_complete_capture() {
  python - "$1" "$2" "$3" <<'PY'
import json
import sys
from pathlib import Path

try:
    manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
ok = (
    manifest.get("complete") is True
    and manifest.get("dataset") == sys.argv[2]
    and manifest.get("strategy") == sys.argv[3]
)
raise SystemExit(0 if ok else 1)
PY
}

run_config() {
  local dataset=$1
  local strategy=$2
  local config="${config_dir}/${dataset}_${strategy}.json"
  local capture="${capture_dir}/${dataset}/${strategy}"
  local output="${raw_dir}/${dataset}_${strategy}.json"
  if [[ "${force_rerun}" != 1 ]] && is_complete_capture "${capture}" "${dataset}" "${strategy}"; then
    echo "Reusing complete capture ${capture}"
    return
  fi
  set +e
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION=0 \
    "${bench_bin}" \
    --search \
    --mode=throughput \
    --threads=1 \
    --data_prefix="${data_dir}" \
    --index_prefix="${data_dir}" \
    --benchmark_repetitions=1 \
    --benchmark_min_time=0.01s \
    --benchmark_min_warmup_time=0 \
    --benchmark_out_format=json \
    --benchmark_out="${output}" \
    "${config}"
  local status=$?
  set -e
  if ! is_complete_capture "${capture}" "${dataset}" "${strategy}"; then
    echo "Diagnostic failed with status ${status}; capture is incomplete: ${capture}" >&2
    return 1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Capture is complete; ignoring teardown-only status ${status}." >&2
  fi
}

cd "${repo_dir}"
for dataset in gist msturing1m msturing10m; do
  for strategy in independent passing frontier combined oracle; do
    run_config "${dataset}" "${strategy}"
  done
done

python "${script_dir}/analyze.py" \
  --capture-root "${capture_dir}" \
  --result-dir "${result_dir}"
