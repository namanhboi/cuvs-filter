#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
run_root=${RETRIEVE_IVF_RUN_ROOT:-/data/retrieve_workshop_runs/a100_ivf_bitmap_$(date -u +%Y%m%dT%H%M%SZ)}
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
if [[ ${1:-all} == status ]]; then
  exec "${python_bin}" "${script_dir}/status.py" --run-root "${RETRIEVE_IVF_RUN_ROOT:?set RETRIEVE_IVF_RUN_ROOT to the running experiment}"
fi
if [[ ${1:-all} == report ]]; then
  exec "${python_bin}" "${script_dir}/report.py" --run-root "${run_root}" "${@:2}"
fi
if [[ ${1:-all} == all ]]; then
  "${python_bin}" "${script_dir}/report.py" snapshot --run-root "${run_root}"
  "${python_bin}" "${script_dir}/workflow.py" all --run-root "${run_root}" "${@:2}"
  exec "${python_bin}" "${script_dir}/report.py" --run-root "${run_root}"
fi
exec "${python_bin}" "${script_dir}/workflow.py" "${1:-all}" --run-root "${run_root}" "${@:2}"
