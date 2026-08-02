#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
source_dir=${2:-"/home/ubuntu/big-ann-benchmarks/data/MSTuringANNS"}
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
delta_bin="${repo_dir}/examples/cpp/build/CAGRA_FAVOR_COMPARE"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"

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

link_input "${source_dir}/base1b.fbin.crop_nb_1000000" "${data_dir}/msturing-1M/base.fbin"
link_input "${source_dir}/testQuery10K.fbin" "${data_dir}/msturing-1M/query.fbin"
link_input "${source_dir}/base1b.fbin.crop_nb_10000000" "${data_dir}/msturing-10M/base.fbin"
link_input "${source_dir}/testQuery10K.fbin" "${data_dir}/msturing-10M/query.fbin"

if [[ ! -f "${data_dir}/gist-960-euclidean/base.fbin" || \
      ! -f "${data_dir}/gist-960-euclidean/query.fbin" ]]; then
  if ! PYTHONPATH="${repo_dir}/python/cuvs_bench${PYTHONPATH:+:${PYTHONPATH}}" \
    python -c "import h5py, requests, scipy, sklearn"; then
    echo "Missing Python dependencies for cuvs_bench dataset download." >&2
    echo "Install the cuvs_bench environment before rerunning setup_data.sh." >&2
    exit 1
  fi
  PYTHONPATH="${repo_dir}/python/cuvs_bench${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m cuvs_bench.get_dataset \
    --dataset gist-960-euclidean \
    --dataset-path "${data_dir}"
fi

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
    --dtype float32 \
    --selectivities 0.01 \
    "$@"
}

prepare_filter "${data_dir}/gist-960-euclidean" \
  --benchmark-query-count 10000 --benchmark-query-file query_10000.fbin
prepare_filter "${data_dir}/msturing-1M"
prepare_filter "${data_dir}/msturing-10M"

python "${script_dir}/generate_configs.py"
if [[ ! -x "${bench_bin}" || ! -f "${build_libcuvs}" ]]; then
  echo "Build artifacts are missing; build CUVS_CAGRA_ANN_BENCH first." >&2
  exit 1
fi

cd "${repo_dir}"
for dataset in gist msturing1m msturing10m; do
  directory=$(python -c \
    "import sys; sys.path.insert(0, '${script_dir}'); from generate_configs import DATASETS; print(DATASETS['${dataset}']['directory'])")
  graph="${data_dir}/${directory}/cagra_g32_ig64.index"
  if [[ ! -f "${graph}" ]]; then
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${bench_bin}" --build \
      --data_prefix="${data_dir}" --index_prefix="${data_dir}" \
      "${script_dir}/configs/${dataset}_build.json"
  fi
done

if [[ ! -x "${delta_bin}" ]]; then
  echo "Missing ${delta_bin}; build the CAGRA_FAVOR_COMPARE example first." >&2
  exit 1
fi
for directory in gist-960-euclidean msturing-1M msturing-10M; do
  graph="${data_dir}/${directory}/cagra_g32_ig64.index"
  if [[ ! -f "${graph}.delta_d" ]]; then
    env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
      "${delta_bin}" "${data_dir}/${directory}/base.fbin" "${graph}" 2
  fi
done

echo "All three datasets, 1% filters/ground truth, graphs, and delta-d sidecars are ready."
