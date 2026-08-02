#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
result_dir=${2:-"${script_dir}/results"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
config_dir="${result_dir}/holdout_configs"
capture_dir="${result_dir}/holdout_captures"
raw_dir="${result_dir}/holdout_raw"

if [[ ! -f "${result_dir}/frozen_rule.json" ]]; then
  echo "No frozen development rule; refusing to inspect the holdout." >&2
  exit 1
fi
"${script_dir}/setup_holdout.sh" "${data_dir}" "${result_dir}"
mkdir -p "${raw_dir}"

for seed in 20260802 20260803; do
  capture="${capture_dir}/seed${seed}"
  if python - "${capture}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    m = json.loads((p / "manifest.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
ok = (
    m.get("schema_version") == 4
    and m.get("dataset") == f"deep_image1m_seed{sys.argv[2]}"
    and m.get("variant") == "frozen_progress_v4"
    and m.get("termination_checkpoint_record_size") == 136
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "Reusing frozen holdout capture ${capture}"
    continue
  fi
  set +e
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION=0 \
    "${bench_bin}" \
      --search --mode=throughput --threads=1 \
      --data_prefix="${data_dir}" --index_prefix="${data_dir}" \
      --benchmark_repetitions=1 --benchmark_min_time=0.01s \
      --benchmark_min_warmup_time=0 \
      --benchmark_out_format=json \
      --benchmark_out="${raw_dir}/seed${seed}.json" \
      "${config_dir}/deep_seed${seed}_shadow.json"
  status=$?
  set -e
  if [[ ! -f "${capture}/manifest.json" ]]; then
    echo "holdout seed ${seed} failed with status ${status}" >&2
    exit 1
  fi
done

python "${script_dir}/evaluate_holdout.py" \
  --capture-root "${capture_dir}" \
  --data-root "${data_dir}" \
  --result-dir "${result_dir}" \
  --frozen-rule "${result_dir}/frozen_rule.json"
