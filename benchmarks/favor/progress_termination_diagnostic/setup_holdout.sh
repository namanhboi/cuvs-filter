#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
data_dir=${1:-"${repo_dir}/datasets"}
result_dir=${2:-"${script_dir}/results"}
source_dir="${data_dir}/deep-image-96-inner"
holdout_dir="${data_dir}/deep-image-1M"
bench_bin="${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"
delta_bin="${repo_dir}/examples/cpp/build/CAGRA_FAVOR_COMPARE"
build_libcuvs="${repo_dir}/cpp/build/libcuvs.so"
config_dir="${result_dir}/holdout_configs"
capture_dir="${result_dir}/holdout_captures"

if [[ ! -f "${source_dir}/base.fbin" || ! -f "${source_dir}/query.fbin" ]]; then
  PYTHONPATH="${repo_dir}/python/cuvs_bench${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m cuvs_bench.get_dataset \
      --dataset deep-image-96-angular \
      --normalize \
      --dataset-path "${data_dir}"
fi

if [[ ! -f "${holdout_dir}/base.fbin" ]]; then
  python "${script_dir}/crop_fbin.py" \
    "${source_dir}/base.fbin" "${holdout_dir}/base.fbin" 1000000
fi
if [[ ! -e "${holdout_dir}/query.fbin" ]]; then
  ln -s "${source_dir}/query.fbin" "${holdout_dir}/query.fbin"
fi

for seed in 20260802 20260803; do
  output="${holdout_dir}/favor_seed${seed}"
  if [[ ! -f "${output}/filter_s01.bin" || \
        ! -f "${output}/groundtruth_s01.ibin" ]]; then
    python "${repo_dir}/benchmarks/favor/prepare_sift_filters.py" \
      --dataset-dir "${holdout_dir}" \
      --output-dir "${output}" \
      --dtype float32 --selectivities 0.01 --seed "${seed}"
  fi
done

python "${script_dir}/generate_holdout_configs.py" \
  --output-dir "${config_dir}" \
  --data-root "${data_dir}" \
  --capture-root "${capture_dir}"

graph="${holdout_dir}/cagra_g32_ig64.index"
if [[ ! -f "${graph}" ]]; then
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${bench_bin}" --build \
    --data_prefix="${data_dir}" --index_prefix="${data_dir}" \
    "${config_dir}/deep_build.json"
fi
if [[ ! -f "${graph}.delta_d" ]]; then
  env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${delta_bin}" "${holdout_dir}/base.fbin" "${graph}" 2
fi

echo "Frozen DEEP-image1M holdout data and index are ready."
