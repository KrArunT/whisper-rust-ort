#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-whisper-rust-ort:4c4g}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.container}"
CORES_LIST="${CORES_LIST:-4}"
MEMORY_GB="${MEMORY_GB:-4}"
CPUSET_START="${CPUSET_START:-0}"
CPUSET_LIST="${CPUSET_LIST:-}"
SUT_NAME="${SUT_NAME:-$(hostname -s)}"
OUT_ROOT="${OUT_ROOT:-results/benchmarks}"
RESULTS_MD="${RESULTS_MD:-RESULTS.md}"
RESULTS_CSV="${RESULTS_CSV:-RESULTS.csv}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
MERGE_ONLY="${MERGE_ONLY:-0}"

if [[ "${BUILD_IMAGE}" != "0" ]]; then
  docker build -t "${IMAGE}" -f "${ROOT_DIR}/${DOCKERFILE}" "${ROOT_DIR}"
fi

cpu_range() {
  local cores="$1"
  local start="$2"
  local end=$((start + cores - 1))
  echo "${start}-${end}"
}

merge_existing_tables() {
  local tables
  tables="$(find "${OUT_ROOT}" -type f -name summary_table.md -path '*/container_*/*' 2>/dev/null || true)"
  if [[ -z "${tables}" ]]; then
    echo "No summary_table.md files found under ${OUT_ROOT}"
    return 0
  fi
  while IFS= read -r table; do
    local sut_dir container_dir container_name sut_name cores mem summary_csv
    sut_dir="$(dirname "${table}")"
    container_dir="$(dirname "${sut_dir}")"
    container_name="$(basename "${container_dir}")"
    sut_name="$(basename "${sut_dir}")"

    if [[ "${container_name}" == container_* ]]; then
      if [[ "${sut_name}" == container_* ]]; then
        sut_name="default"
      fi
    else
      container_name="$(basename "${sut_dir}")"
      sut_name="default"
    fi

    if [[ "${container_name}" =~ ^container_([0-9]+)c([0-9]+)g$ ]]; then
      cores="${BASH_REMATCH[1]}"
      mem="${BASH_REMATCH[2]}"
    elif [[ "${container_name}" == "container_4c4g" ]]; then
      cores="4"
      mem="4"
    else
      echo "Skip (unrecognized container dir): ${container_name}"
      continue
    fi

    summary_csv="${table%.md}.csv"
    uv run update_results_md.py \
      --results-md "${RESULTS_MD}" \
      --summary-table "${table}" \
      --summary-csv "${summary_csv}" \
      --sut-name "${sut_name}" \
      --core-count "${cores}" \
      --memory-gb "${mem}" \
      --results-csv "${RESULTS_CSV}"
  done <<< "${tables}"
}

if [[ "${MERGE_ONLY}" != "0" ]]; then
  merge_existing_tables
  exit 0
fi

for cores in ${CORES_LIST}; do
  if [[ -n "${CPUSET_LIST}" ]]; then
    cpuset="${CPUSET_LIST}"
  else
    cpuset="${CPUSET:-$(cpu_range "${cores}" "${CPUSET_START}")}"
  fi
  if [[ "${cores}" == "4" && "${MEMORY_GB}" == "4" ]]; then
    out_base="${OUT_ROOT}/container_4c4g"
  else
    out_base="${OUT_ROOT}/container_${cores}c${MEMORY_GB}g"
  fi
  if [[ -n "${SUT_NAME}" ]]; then
    out_base="${out_base}/${SUT_NAME}"
  fi

  docker run --rm \
    --cpuset-cpus "${cpuset}" \
    --memory "${MEMORY_GB}g" \
    --memory-swap "${MEMORY_GB}g" \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    -e CORE_COUNT="${cores}" \
    -e MEMORY_GB="${MEMORY_GB}" \
    -e SUT_NAME="${SUT_NAME}" \
    -e OUT_ROOT="${OUT_ROOT}" \
    -e OUT_BASE="${out_base}" \
    -e AUDIO_DIR \
    -e ONNX_DIR \
    -e ONNX_INT8_DIR \
    -e LANGUAGE \
    -e TASK \
    -e MAX_NEW_TOKENS \
    -e NUM_BEAMS \
    -e WARMUP \
    -e WRITE_TXT \
    -e HF_MODEL_ID \
    -e FW_MODEL_ID \
    -e UV_CACHE_DIR \
    -e ORT_INTRA_OP \
    -e ORT_INTER_OP \
    "${IMAGE}" \
    bash -lc "scripts/run_container_benchmarks_inner.sh"

  uv run update_results_md.py \
    --results-md "${RESULTS_MD}" \
    --summary-table "${out_base}/summary_table.md" \
    --summary-csv "${out_base}/summary_table.csv" \
    --sut-name "${SUT_NAME:-default}" \
    --core-count "${cores}" \
    --memory-gb "${MEMORY_GB}" \
    --results-csv "${RESULTS_CSV}"
done
