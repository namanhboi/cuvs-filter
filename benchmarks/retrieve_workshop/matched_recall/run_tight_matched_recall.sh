#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
result_root=${RETRIEVE_RESULT_ROOT:-"/home/ubuntu/retrieve_workshop_runs/matched_recall_tight_$(date -u +%Y%m%d_%H%M%S)"}
baseline=${RETRIEVE_MATCHED_BASELINE:-/home/ubuntu/retrieve_workshop_runs/matched_recall_b0first_20260821_v1/analysis/measurements.csv}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
build_libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
min_time=${RETRIEVE_BENCH_MIN_TIME:-0.10s}
stage=${1:-all}

require_runtime() {
  test -x "${bench_bin}" || { echo "missing benchmark binary: ${bench_bin}" >&2; exit 2; }
  test -f "${build_libcuvs}" || { echo "missing libcuvs: ${build_libcuvs}" >&2; exit 2; }
  test -f "${baseline}" || { echo "missing baseline measurements: ${baseline}" >&2; exit 2; }
  "${python_bin}" "${script_dir}/../gpu_graph/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" --stage "$1" \
    --bench-bin "${bench_bin}" --libcuvs "${build_libcuvs}" --data-root "${data_root}"
}

run_group() {
  local group=$1
  while IFS= read -r manifest; do
    local workload destination repetitions
    workload=$(basename "$(dirname "${manifest}")")
    destination="${result_root}/raw/${group}/${workload}"
    repetitions=$("${python_bin}" -c "import json; print(json.load(open('${manifest}'))['repetitions'])")
    mkdir -p "${destination}"
    while IFS= read -r config; do
      local name output
      name=$(basename "${config}" .json)
      output="${destination}/${name}.json"
      if test -e "${output}"; then
        echo "resume: retain complete ${output}" >&2
        continue
      fi
      env LD_PRELOAD="${build_libcuvs}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${bench_bin}" --search --mode=throughput --threads=1 \
        --data_prefix="${data_root}" --index_prefix="${data_root}" \
        --benchmark_repetitions="${repetitions}" \
        --benchmark_min_time="${min_time}" --benchmark_min_warmup_time=0.01 \
        --benchmark_enable_random_interleaving=true \
        --benchmark_report_aggregates_only=false \
        --benchmark_out_format=json --benchmark_out="${output}" "${config}"
    done < <(
      "${python_bin}" - "${manifest}" <<'PY'
import json, pathlib, sys
for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["configs"]:
    print(row["config"])
PY
    )
  done < <(find "${result_root}/configs/${group}" -mindepth 2 -maxdepth 2 -name manifest.json | sort)
}

calibrate() {
  local points="${result_root}/state/tight_refinement_points.json"
  mkdir -p "${result_root}/state"
  if ! test -f "${points}"; then
    "${python_bin}" "${script_dir}/matched_recall.py" tight-refinement-points \
      --baseline-measurements "${baseline}" --output "${points}"
  fi
  if ! test -f "${result_root}/configs/tight_calibration/manifest.json"; then
    "${python_bin}" "${script_dir}/matched_recall.py" generate-group \
      --result-root "${result_root}" --data-root "${data_root}" \
      --group tight_calibration --stage calibration --repetitions 1 --points "${points}"
  fi
  run_group tight_calibration
  "${python_bin}" "${script_dir}/matched_recall.py" analyze --result-root "${result_root}"
}

run_finalists() {
  local points="${result_root}/state/tight_finalists.json"
  "${python_bin}" "${script_dir}/matched_recall.py" tight-finalists \
    --result-root "${result_root}" --output "${points}"
  if ! test -f "${result_root}/configs/tight_finalists/manifest.json"; then
    "${python_bin}" "${script_dir}/matched_recall.py" generate-group \
      --result-root "${result_root}" --data-root "${data_root}" \
      --group tight_finalists --stage finalist --repetitions 3 --points "${points}"
  fi
  run_group tight_finalists
  "${python_bin}" "${script_dir}/matched_recall.py" analyze --result-root "${result_root}"
}

run_paper_final() {
  local points="${result_root}/state/tight_paper_points.json"
  "${python_bin}" "${script_dir}/matched_recall.py" tight-paper-points \
    --result-root "${result_root}" --output "${points}"
  if ! test -f "${result_root}/configs/paper_final/manifest.json"; then
    "${python_bin}" "${script_dir}/matched_recall.py" generate-group \
      --result-root "${result_root}" --data-root "${data_root}" \
      --group paper_final --stage final --repetitions 3 --points "${points}"
  fi
  run_group paper_final
  "${python_bin}" "${script_dir}/matched_recall.py" finalize --result-root "${result_root}"
}

case "${stage}" in
  calibrate)
    require_runtime tight-calibrate
    calibrate
    ;;
  finalists)
    require_runtime tight-finalists
    run_finalists
    ;;
  final)
    require_runtime tight-final
    run_paper_final
    ;;
  analyze)
    "${python_bin}" "${script_dir}/matched_recall.py" finalize --result-root "${result_root}"
    ;;
  all)
    require_runtime tight-all
    calibrate
    run_finalists
    run_paper_final
    ;;
  *)
    echo "usage: $0 {calibrate|finalists|final|analyze|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
