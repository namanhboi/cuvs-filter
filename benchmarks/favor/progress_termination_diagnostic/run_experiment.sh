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
force_rerun=${FAVOR_PROGRESS_FORCE_RERUN:-0}

"${script_dir}/setup_data.sh" "${data_dir}"
mkdir -p "${raw_dir}"

is_complete_capture() {
  python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
try:
    manifest = json.loads((directory / "manifest.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
required = ("query_summary.csv", "termination_checkpoints.bin", "termination_checkpoint_counts.bin")
ok = (
    manifest.get("schema_version") == 4
    and manifest.get("dataset") == sys.argv[2]
    and manifest.get("variant") == "exact_progress_v4"
    and manifest.get("termination_record_start_iteration") == 1
    and manifest.get("termination_checkpoint_record_size") == 136
    and all((directory / name).is_file() for name in required)
)
raise SystemExit(0 if ok else 1)
PY
}

for dataset in sift gist bigann1m bigann10m msturing1m msturing10m; do
  capture="${capture_dir}/${dataset}"
  output="${raw_dir}/${dataset}.json"
  if [[ "${force_rerun}" != 1 ]] && is_complete_capture "${capture}" "${dataset}"; then
    echo "Reusing complete capture ${capture}"
    continue
  fi
  set +e
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION=0 \
    "${bench_bin}" \
    --search --mode=throughput --threads=1 \
    --data_prefix="${data_dir}" --index_prefix="${data_dir}" \
    --benchmark_repetitions=1 \
    --benchmark_min_time=0.01s --benchmark_min_warmup_time=0 \
    --benchmark_out_format=json --benchmark_out="${output}" \
    "${config_dir}/${dataset}_shadow.json"
  status=$?
  set -e
  if ! is_complete_capture "${capture}" "${dataset}"; then
    echo "${dataset} failed with status ${status} and did not produce a valid capture" >&2
    exit 1
  fi
  if [[ "${status}" != 0 ]]; then
    echo "warning: ${dataset} exited ${status} after writing a complete capture" >&2
  fi
done

python "${script_dir}/analyze.py" \
  --capture-root "${capture_dir}" \
  --data-root "${data_dir}" \
  --result-dir "${result_dir}"

disposition=$(python -c \
  "import json; print(json.load(open('${result_dir}/gate.json'))['disposition'])")
if [[ "${disposition}" == run_frozen_holdout ]]; then
  "${script_dir}/run_holdout.sh" "${data_dir}" "${result_dir}"
else
  echo "Development shadow gate failed; DEEP holdout and live implementation are intentionally skipped."
fi
