#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
bigann_source=${2:-"/home/ubuntu/big-ann-benchmarks/data/bigann"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
delta_bin="${repo_dir}/examples/cpp/build/CAGRA_FAVOR_COMPARE"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
config_dir="${script_dir}/results/configs"
capture_dir="${script_dir}/results/captures"

link_input() {
  local source=$1
  local destination=$2
  if [[ ! -f "${source}" ]]; then
    echo "Missing source data: ${source}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${destination}")"
  if [[ -L "${destination}" ]]; then
    if [[ "$(readlink -f "${destination}")" == "$(readlink -f "${source}")" ]]; then return; fi
    echo "Existing symlink has the wrong target: ${destination}" >&2
    exit 1
  fi
  if [[ -e "${destination}" ]]; then return; fi
  ln -s "${source}" "${destination}"
}

if [[ ! -f "${data_dir}/sift-128-euclidean/base.fbin" || \
      ! -f "${data_dir}/sift-128-euclidean/query.fbin" ]]; then
  PYTHONPATH="${repo_dir}/python/cuvs_bench${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m cuvs_bench.get_dataset \
      --dataset sift-128-euclidean \
      --dataset-path "${data_dir}"
fi

for directory in bigann-1M bigann-10M; do
  link_input "${bigann_source}/base.1B.u8bin.crop_nb_10000000" \
    "${data_dir}/${directory}/base.10M.u8bin"
  link_input "${bigann_source}/query.public.10K.u8bin" \
    "${data_dir}/${directory}/query.public.10K.u8bin"
done

prepare_filter() {
  local directory=$1
  shift
  if [[ -f "${directory}/favor/filter_s01.bin" && \
        -f "${directory}/favor/groundtruth_s01.ibin" ]]; then
    return
  fi
  python "${repo_dir}/benchmarks/favor/prepare_sift_filters.py" \
    --dataset-dir "${directory}" \
    --output-dir "${directory}/favor" \
    --selectivities 0.01 \
    "$@"
}

prepare_filter "${data_dir}/sift-128-euclidean" --dtype float32
prepare_filter "${data_dir}/bigann-1M" \
  --base-file base.10M.u8bin --query-file query.public.10K.u8bin \
  --dtype uint8 --subset-size 1000000
prepare_filter "${data_dir}/bigann-10M" \
  --base-file base.10M.u8bin --query-file query.public.10K.u8bin --dtype uint8

python "${script_dir}/generate_configs.py" \
  --output-dir "${config_dir}" \
  --data-root "${data_dir}" \
  --capture-root "${capture_dir}"

if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
  echo "Build artifacts are missing; build CUVS_CAGRA_ANN_BENCH first." >&2
  exit 1
fi

for dataset in sift gist bigann1m bigann10m msturing1m msturing10m; do
  directory=$(python - "${script_dir}" "${dataset}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from generate_configs import DATASETS
print(DATASETS[sys.argv[2]]["directory"])
PY
)
  graph="${data_dir}/${directory}/cagra_g32_ig64.index"
  if [[ ! -f "${graph}" ]]; then
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --build \
      --data_prefix="${data_dir}" --index_prefix="${data_dir}" \
      "${config_dir}/${dataset}_build.json"
  fi
done

if [[ ! -x "${delta_bin}" ]]; then
  echo "Missing ${delta_bin}; build CAGRA_FAVOR_COMPARE first." >&2
  exit 1
fi
for dataset in sift gist bigann1m bigann10m msturing1m msturing10m; do
  readarray -t values < <(python - "${script_dir}" "${dataset}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from generate_configs import DATASETS
s = DATASETS[sys.argv[2]]
print(s["directory"])
print(s["base"])
print(s.get("subset_size", 0))
PY
)
  directory=${values[0]}
  base=${values[1]}
  subset=${values[2]}
  graph="${data_dir}/${directory}/cagra_g32_ig64.index"
  if [[ ! -f "${graph}.delta_d" ]]; then
    if [[ "${subset}" == 0 ]]; then
      env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${delta_bin}" "${data_dir}/${directory}/${base}" "${graph}" 2
    else
      env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${delta_bin}" "${data_dir}/${directory}/${base}" "${graph}" 2 "${subset}"
    fi
  fi
done

echo "Six development families/scales, 1% filters, graphs, and delta-d sidecars are ready."
