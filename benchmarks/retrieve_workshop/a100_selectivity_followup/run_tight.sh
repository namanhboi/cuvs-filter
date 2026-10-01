#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
reference_root=${RETRIEVE_SELECTIVITY_REFERENCE_ROOT:-${RETRIEVE_SELECTIVITY_RUN_ROOT:?set RETRIEVE_SELECTIVITY_REFERENCE_ROOT to the completed selectivity run}}
result_root=${RETRIEVE_SELECTIVITY_MATCHED_ROOT:-"${HOME}/a100_selectivity_matched_$(date -u +%Y%m%dT%H%M%SZ)"}
data_root=${RETRIEVE_DATA_ROOT:-/data/retrieve_data}
stage=${1:-all}
upper=${RETRIEVE_SELECTIVITY_MATCHED_UPPER:-}
calibration_root=${RETRIEVE_SELECTIVITY_CALIBRATION_ROOT:-}
prior_finalists_root=${RETRIEVE_SELECTIVITY_PRIOR_FINALISTS_ROOT:-}

if [[ -z "${upper}" ]]; then
  if [[ "${stage}" == rescue ]]; then
    upper=0.952
  else
    upper=0.951
  fi
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}
unset RETRIEVE_DATASET_PROFILE
mkdir -p "${result_root}/logs" "${result_root}/state"
exec 9>"${result_root}.lock"
flock -n 9 || { echo "another process owns ${result_root}.lock" >&2; exit 2; }
exec > >(tee -a "${result_root}/logs/orchestrator.log") 2>&1

if [[ "${stage}" == all || "${stage}" == calibrate || "${stage}" == rescue ]]; then
  "${python_bin}" "${script_dir}/test_tight_match.py"
fi

arguments=("${stage}" --reference-root "${reference_root}" --root "${result_root}"
  --data-root "${data_root}" --upper "${upper}")
if [[ -n "${calibration_root}" ]]; then
  arguments+=(--calibration-root "${calibration_root}")
fi
if [[ -n "${prior_finalists_root}" ]]; then
  arguments+=(--prior-finalists-root "${prior_finalists_root}")
fi
"${python_bin}" "${script_dir}/tight_match.py" "${arguments[@]}"
