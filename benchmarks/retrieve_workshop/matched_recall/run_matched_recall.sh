#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")"; pwd)
repo_dir=$(cd "${script_dir}/../../.."; pwd)
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
data_root=${RETRIEVE_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
result_root=${RETRIEVE_RESULT_ROOT:-"/home/ubuntu/retrieve_workshop_runs/matched_recall_$(date -u +%Y%m%d_%H%M%S)"}
bench_bin=${RETRIEVE_BENCH_BIN:-"${repo_dir}/cpp/build/bench/ann/CUVS_CAGRA_ANN_BENCH"}
build_libcuvs=${RETRIEVE_LIBCUVS:-"${repo_dir}/cpp/build/libcuvs.so"}
min_time=${RETRIEVE_BENCH_MIN_TIME:-0.10s}
stage=${1:-all}

require_runtime() {
  test -x "${bench_bin}" || { echo "missing benchmark binary: ${bench_bin}" >&2; exit 2; }
  test -f "${build_libcuvs}" || { echo "missing libcuvs: ${build_libcuvs}" >&2; exit 2; }
  "${python_bin}" "${script_dir}/../gpu_graph/capture_provenance.py" \
    --result-root "${result_root}" --repo "${repo_dir}" --stage "$1" \
    --bench-bin "${bench_bin}" --libcuvs "${build_libcuvs}" --data-root "${data_root}"
}

run_group() {
  local group=$1
  while IFS= read -r manifest; do
    local workload destination
    workload=$(basename "$(dirname "${manifest}")")
    destination="${result_root}/raw/${group}/${workload}"
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
        --benchmark_repetitions="$("${python_bin}" -c "import json; print(json.load(open('${manifest}'))['repetitions'])")" \
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
  local round=0
  while true; do
    local proposal group count
    group="calibration_r$(printf '%02d' "${round}")"
    if test -d "${result_root}/configs/${group}"; then
      test -f "${result_root}/configs/${group}/manifest.json" || {
        echo "incomplete existing calibration config group: ${group}" >&2
        exit 2
      }
      echo "resume: validate existing ${group}" >&2
      run_group "${group}"
      "${python_bin}" "${script_dir}/matched_recall.py" analyze --result-root "${result_root}"
      round=$((round + 1))
      continue
    fi
    proposal="${result_root}/state/next_$(printf '%02d' "${round}").json"
    mkdir -p "$(dirname "${proposal}")"
    "${python_bin}" "${script_dir}/matched_recall.py" next \
      --result-root "${result_root}" --output "${proposal}"
    count=$("${python_bin}" -c "import json; print(len(json.load(open('${proposal}'))['points']))")
    if test "${count}" -eq 0; then
      echo "calibration complete after ${round} rounds" >&2
      break
    fi
    "${python_bin}" "${script_dir}/matched_recall.py" generate-group \
      --result-root "${result_root}" --data-root "${data_root}" \
      --group "${group}" --repetitions 1 --points "${proposal}"
    run_group "${group}"
    "${python_bin}" "${script_dir}/matched_recall.py" analyze --result-root "${result_root}"
    round=$((round + 1))
  done
}

run_final() {
  local selection="${result_root}/state/final_selection.json"
  "${python_bin}" "${script_dir}/matched_recall.py" select-final \
    --result-root "${result_root}" --output "${selection}"
  if ! test -f "${result_root}/configs/final_candidates/manifest.json"; then
    "${python_bin}" "${script_dir}/matched_recall.py" generate-group \
      --result-root "${result_root}" --data-root "${data_root}" \
      --group final_candidates --repetitions 3 --points "${selection}"
  fi
  run_group final_candidates
  "${python_bin}" "${script_dir}/matched_recall.py" finalize --result-root "${result_root}"
}

case "${stage}" in
  calibrate)
    require_runtime calibrate
    calibrate
    ;;
  final)
    require_runtime final
    run_final
    ;;
  analyze)
    "${python_bin}" "${script_dir}/matched_recall.py" finalize --result-root "${result_root}"
    ;;
  all)
    require_runtime all
    calibrate
    run_final
    ;;
  *)
    echo "usage: $0 {calibrate|final|analyze|all}" >&2
    exit 2
    ;;
esac

printf '%s\n' "${result_root}"
