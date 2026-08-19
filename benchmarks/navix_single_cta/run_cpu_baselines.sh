#!/bin/bash
set -euo pipefail

data_root=${NAVIX_DATA_ROOT:-/home/ubuntu/cuvs-filter/datasets}
artifact_root=${NAVIX_CPU_ARTIFACT_ROOT:-/home/ubuntu/navix_cpu_artifacts}
result_root=${NAVIX_CPU_RESULT_ROOT:-${artifact_root}/results}
faiss_repo=${FAISS_NAVIX_REPO:-/home/ubuntu/faiss-navix-native-benchmark}
acorn_repo=${ACORN_REPO:-/home/ubuntu/ACORN-gamma-benchmark}
python_bin=${PYTHON:-/home/ubuntu/micromamba/envs/cuvs/bin/python}
# Eight threads is the lower endpoint of this study's local 8/16/32 scaling experiment.  The
# arxiv-for-fanns paper used a single query-execution thread on an 8-hardware-thread host, so these
# multi-threaded QPS measurements are not presented as a reproduction of that paper's throughput.
cpu_threads=${NAVIX_CPU_THREADS:-8}
# Graph construction is offline and is not part of search QPS.  Use 24 of the 32 logical CPUs for
# the expensive YFCC builds while leaving headroom for the OS and benchmark orchestration.
build_threads=${NAVIX_CPU_BUILD_THREADS:-24}
result_suffix=${NAVIX_CPU_RESULT_SUFFIX:-_t${cpu_threads}}
export OMP_NUM_THREADS="${cpu_threads}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=${OMP_PLACES:-cores}
stage=${1:-all}
# The study stops once the frontier reaches approximately 0.95 recall. Values above 100 add
# substantial 4,096-D CPU work without affecting that decision boundary on the ArXiv workloads.
ef_values=${NAVIX_CPU_EF_VALUES:-10,15,20,25,30,50,100}

mkdir -p "${artifact_root}/graphs" "${result_root}"

manifest_path() {
  local workload=$1
  if [[ "${workload}" == yfcc ]]; then
    echo "${data_root}/navix_bitmap/yfcc/throughput_10000/manifest.json"
  else
    echo "${data_root}/navix_bitmap/arxiv/${workload}/throughput_10000/manifest.json"
  fi
}

base_args() {
  local workload=$1
  if [[ "${workload}" == yfcc ]]; then
    echo "${data_root}/yfcc-10M/base.10M.u8bin u8 u8"
  else
    echo "${data_root}/arxiv-for-fanns-medium/base.fbin f32 f32"
  fi
}

shard_rows() {
  "${python_bin}" - "$(manifest_path "$1")" <<'PY'
import json, pathlib, sys
for i, row in enumerate(json.loads(pathlib.Path(sys.argv[1]).read_text())["shards"]):
    directory = pathlib.Path(row["directory"])
    print(i, directory / "query.bin", directory / "groundtruth.ibin", directory / "filter.bitmap")
PY
}

run_faiss() {
  local workload=$1 chunk=$2 ef=${3:-${ef_values}} suffix=${4:-}
  read -r base dtype qtype <<<"$(base_args "${workload}")"
  local base_key=arxiv
  local M=32 efc=200
  if [[ "${workload}" == yfcc ]]; then
    base_key=yfcc
    # JAG Appendix D.5 NaviX setting for the YFCC subset workload.
    M=64
  else
    # The FAISS-NaviX repository example and this runner's original baseline use M=32 and
    # efConstruction=200.  arXiv:2507.21989 does not specify a NaviX graph for
    # arxiv-for-fanns-medium, so do not attribute this ArXiv choice to that paper.
    M=32
  fi
  local graph="${artifact_root}/graphs/faiss_hnsw_${base_key}_M${M}_efc${efc}.index"
  local output="${result_root}/faiss_navix/${workload}"
  mkdir -p "${output}"
  local first=1
  while read -r shard query gt bitmap; do
    local build_args=()
    if [[ ! -f "${graph}" && ${first} -eq 1 ]]; then build_args=(--build); fi
    "${faiss_repo}/build/benchs/bench_navix_bitmap" \
      --base "${base}" --dtype "${dtype}" --queries "${query}" --qtype "${qtype}" \
      --ground-truth "${gt}" --bitmap "${bitmap}" --index "${graph}" \
      --csv "${output}/shard_${shard}${suffix}${result_suffix}.csv" \
      --chunk "${chunk}" --ef-search "${ef}" --M "${M}" \
      --ef-construction "${efc}" --threads "${cpu_threads}" \
      --build-threads "${build_threads}" "${build_args[@]}"
    first=0
  done < <(shard_rows "${workload}")
}

acorn_parameters() {
  local method=$1 workload=$2
  # ACORN Section 5.3 defines ACORN-1 as gamma=1 and M_beta=M.  M=32 and
  # efConstruction=200 are this benchmark's fixed HNSW-like control values.
  if [[ "${method}" == acorn_1 ]]; then echo "32 1 32 200"; return; fi
  case "${workload}" in
    # M, M_beta, and gamma below are the arxiv-for-fanns-medium values from Table 4 of
    # arXiv:2507.21989.  That table does not state efConstruction; the fourth value is the
    # ACORN fork's M*gamma constructor convention and is recorded as an implementation setting.
    em) echo "16 10 24 160" ;;
    r) echo "32 12 24 384" ;;
    emis) echo "16 15 24 240" ;;
    # JAG Appendix D.5 specifies M=64, gamma=30, and M_beta=64 for YFCC.  Its
    # table does not specify ACORN efConstruction; 1920 is the fork's native
    # constructor default M*gamma and is recorded separately in every artifact.
    yfcc) echo "64 30 64 1920" ;;
    *) return 2 ;;
  esac
}

run_acorn() {
  local method=$1 workload=$2 chunk=$3 ef=${4:-${ef_values}} suffix=${5:-}
  local filtered_seeds=${6:-0}
  read -r base dtype qtype <<<"$(base_args "${workload}")"
  read -r M gamma M_beta efc <<<"$(acorn_parameters "${method}" "${workload}")"
  local graph="${artifact_root}/graphs/${method}_${workload}_M${M}_g${gamma}_b${M_beta}_efc${efc}.index"
  if [[ "${method}" == acorn_1 ]]; then
    local base_key=arxiv
    [[ "${workload}" == yfcc ]] && base_key=yfcc
    graph="${artifact_root}/graphs/acorn_1_${base_key}_M32_efc200.index"
  fi
  local result_method=${method}
  if (( filtered_seeds > 0 )); then result_method="${method}_navix_seeded"; fi
  local output="${result_root}/${result_method}/${workload}"
  mkdir -p "${output}"
  local first=1
  while read -r shard query gt bitmap; do
    local build_args=()
    if [[ ! -f "${graph}" && ${first} -eq 1 ]]; then build_args=(--build); fi
    "${acorn_repo}/build/benchs/bench_acorn_bitmap" \
      --base "${base}" --dtype "${dtype}" --queries "${query}" --qtype "${qtype}" \
      --ground-truth "${gt}" --bitmap "${bitmap}" --index "${graph}" \
      --csv "${output}/shard_${shard}${suffix}${result_suffix}.csv" \
      --chunk "${chunk}" --ef-search "${ef}" \
      --method "${method}" --M "${M}" --gamma "${gamma}" --M-beta "${M_beta}" \
      --ef-construction "${efc}" --threads "${cpu_threads}" \
      --build-threads "${build_threads}" --filtered-seeds "${filtered_seeds}" \
      "${build_args[@]}"
    first=0
  done < <(shard_rows "${workload}")
}

run_chunk_gate() {
  local graph="${artifact_root}/graphs/faiss_hnsw_yfcc_M64_efc200.index"
  local first_line
  first_line=$(shard_rows yfcc | head -n 1)
  read -r shard query gt bitmap <<<"${first_line}"
  mkdir -p "${result_root}/chunk_gate"
  for chunk in 128 256 512; do
    local build_args=()
    if [[ ! -f "${graph}" ]]; then build_args=(--build); fi
    "${faiss_repo}/build/benchs/bench_navix_bitmap" \
      --base "${data_root}/yfcc-10M/base.10M.u8bin" --dtype u8 \
      --queries "${query}" --qtype u8 --ground-truth "${gt}" --bitmap "${bitmap}" \
      --index "${graph}" --csv "${result_root}/chunk_gate/chunk_${chunk}${result_suffix}.csv" \
      --chunk "${chunk}" --ef-search 100 --M 64 --ef-construction 200 \
      --threads "${cpu_threads}" --build-threads "${build_threads}" "${build_args[@]}"
  done
  "${python_bin}" - "${result_root}/chunk_gate" "${result_suffix}" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
suffix = sys.argv[2]
rows = []
for path in sorted(root.glob(f"chunk_*{suffix}.csv")):
    row = next(csv.DictReader(path.open()))
    rows.append((int(row["chunk"]), float(row["qps"])))
if not rows:
    raise SystemExit(f"no chunk-gate rows found for suffix {suffix!r}")
best = max(qps for _, qps in rows)
chosen = min(chunk for chunk, qps in rows if qps >= 0.98 * best)
(root / "selected_chunk.txt").write_text(f"{chosen}\n")
print(f"YFCC byte-mask chunk gate: {rows}; selected {chosen}")
PY
}

selected_yfcc_chunk() {
  local file="${result_root}/chunk_gate/selected_chunk.txt"
  [[ -f "${file}" ]] && cat "${file}" || echo 128
}

case "${stage}" in
  self_test)
    "${faiss_repo}/build/benchs/bench_navix_bitmap" --self-test
    "${acorn_repo}/build/benchs/bench_acorn_bitmap" --self-test
    ;;
  chunk_gate) run_chunk_gate ;;
  arxiv)
    for workload in em emis r; do
      run_faiss "${workload}" 10000
      run_acorn acorn_1 "${workload}" 10000
      run_acorn acorn_gamma "${workload}" 10000
    done
    ;;
  arxiv_deeper)
    deeper_ef=${NAVIX_CPU_DEEPER_EF:-250}
    for workload in em emis r; do
      run_acorn acorn_1 "${workload}" 10000 "${deeper_ef}" "_ef${deeper_ef}"
    done
    run_acorn acorn_gamma emis 10000 "${deeper_ef}" "_ef${deeper_ef}"
    ;;
  thread_sweep)
    # Evaluate the complete base frontier at one physical-half socket, the full physical socket,
    # and the full SMT socket.  The Pareto analysis then selects the best measured CPU resource
    # setting per method rather than assuming a CUDA/CPU thread equivalence.
    NAVIX_CPU_THREADS=8 NAVIX_CPU_BUILD_THREADS="${build_threads}" \
      NAVIX_CPU_RESULT_ROOT="${result_root}" "$0" chunk_gate
    for threads in 8 16 32; do
      NAVIX_CPU_THREADS="${threads}" NAVIX_CPU_BUILD_THREADS="${build_threads}" \
        NAVIX_CPU_RESULT_ROOT="${result_root}" "$0" arxiv
      NAVIX_CPU_THREADS="${threads}" NAVIX_CPU_BUILD_THREADS="${build_threads}" \
        NAVIX_CPU_RESULT_ROOT="${result_root}" "$0" yfcc
    done
    ;;
  yfcc)
    chunk=$(selected_yfcc_chunk)
    run_faiss yfcc "${chunk}"
    run_acorn acorn_1 yfcc "${chunk}"
    run_acorn acorn_gamma yfcc "${chunk}"
    ;;
  yfcc_deeper)
    chunk=$(selected_yfcc_chunk)
    deeper_ef=${NAVIX_CPU_DEEPER_EF:-250,500,750}
    run_faiss yfcc "${chunk}" "${deeper_ef}" "_deeper"
    run_acorn acorn_1 yfcc "${chunk}" "${deeper_ef}" "_deeper"
    run_acorn acorn_gamma yfcc "${chunk}" "${deeper_ef}" "_deeper"
    ;;
  acorn_seed_ablation)
    # Paired reference/seeded runs share graphs, query chunks, efSearch values, and CPU resources.
    for workload in em emis r; do
      run_acorn acorn_1 "${workload}" 10000
      run_acorn acorn_1 "${workload}" 10000 "${ef_values}" "" 10
      run_acorn acorn_gamma "${workload}" 10000
      run_acorn acorn_gamma "${workload}" 10000 "${ef_values}" "" 10
    done
    chunk=$(selected_yfcc_chunk)
    run_acorn acorn_1 yfcc "${chunk}"
    run_acorn acorn_1 yfcc "${chunk}" "${ef_values}" "" 10
    run_acorn acorn_gamma yfcc "${chunk}"
    run_acorn acorn_gamma yfcc "${chunk}" "${ef_values}" "" 10
    ;;
  acorn_seeded_deeper)
    deeper_ef=${NAVIX_CPU_DEEPER_EF:-250,500,750}
    # ACORN-1 remains below 0.95 recall on every workload at efSearch=100.
    for workload in em emis r; do
      run_acorn acorn_1 "${workload}" 10000 "${deeper_ef}" "_seeded_deeper" 10
    done
    # Seeded ACORN-gamma already exceeds 0.95 on EM and R at efSearch=100;
    # only EMIS and YFCC require deeper points.
    run_acorn acorn_gamma emis 10000 "${deeper_ef}" "_seeded_deeper" 10
    chunk=$(selected_yfcc_chunk)
    run_acorn acorn_1 yfcc "${chunk}" "${deeper_ef}" "_seeded_deeper" 10
    run_acorn acorn_gamma yfcc "${chunk}" "${deeper_ef}" "_seeded_deeper" 10
    ;;
  all)
    "$0" self_test
    "$0" chunk_gate
    "$0" arxiv
    "$0" yfcc
    ;;
  *) echo "usage: $0 {self_test|chunk_gate|arxiv|arxiv_deeper|thread_sweep|yfcc|yfcc_deeper|acorn_seed_ablation|acorn_seeded_deeper|all}" >&2; exit 2 ;;
esac

printf '%s\n' "${artifact_root}"
