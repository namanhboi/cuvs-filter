#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
prepare_script="${repo_dir}/benchmarks/favor/prepare_sift_filters.py"

missing_selectivities() {
  local dataset_dir=$1
  local missing=()
  local selectivity encoded fraction
  for selectivity in 1 10 50 90; do
    printf -v encoded '%02d' "${selectivity}"
    if [[ ! -f "${dataset_dir}/favor/filter_s${encoded}.bin" ||
          ! -f "${dataset_dir}/favor/groundtruth_s${encoded}.ibin" ||
          ! -f "${dataset_dir}/favor/groundtruth_s${encoded}.fbin" ]]; then
      printf -v fraction '0.%02d' "${selectivity}"
      missing+=("${fraction}")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    printf '%s\n' "${missing[@]}"
  fi
}

prepare_dataset() {
  local dataset_name=$1
  shift
  local dataset_dir="${data_dir}/${dataset_name}"
  local -a selectivities=()
  mapfile -t selectivities < <(missing_selectivities "${dataset_dir}")
  if (( ${#selectivities[@]} == 0 )); then
    echo "Reusing complete FAVOR data for ${dataset_name}"
    return
  fi

  echo "Preparing ${dataset_name} selectivities: ${selectivities[*]}"
  micromamba run -n cuvs python "${prepare_script}" \
    --dataset-dir "${dataset_dir}" \
    --output-dir "${dataset_dir}/favor" \
    --selectivities "${selectivities[@]}" \
    "$@"
}

# The generator is deterministic (seed 20260724 by default) and merges compatible
# manifest entries. Complete selectivities are never requested again.
prepare_dataset sift-128-euclidean
prepare_dataset gist-960-euclidean \
  --benchmark-query-count 10000 \
  --benchmark-query-file query_10000.fbin
prepare_dataset bigann-1M \
  --base-file base.10M.u8bin \
  --query-file query.public.10K.u8bin \
  --dtype uint8 \
  --subset-size 1000000
prepare_dataset bigann-10M \
  --base-file base.10M.u8bin \
  --query-file query.public.10K.u8bin \
  --dtype uint8
prepare_dataset msturing-1M --dtype float32
prepare_dataset msturing-10M --dtype float32
