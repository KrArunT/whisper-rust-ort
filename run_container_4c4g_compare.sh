#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-whisper-rust-ort:4c4g}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.container}"
CORES_LIST="${CORES_LIST:-4}"
MEMORY_GB="${MEMORY_GB:-4}"
CPUSET_START="${CPUSET_START:-0}"
SUT_NAME="${SUT_NAME:-$(hostname -s)}"
OUT_ROOT="${OUT_ROOT:-results/benchmarks}"
RESULTS_MD="${RESULTS_MD:-RESULTS.md}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"

if [[ "${BUILD_IMAGE}" != "0" ]]; then
  docker build -t "${IMAGE}" -f "${ROOT_DIR}/${DOCKERFILE}" "${ROOT_DIR}"
fi

cpu_range() {
  local cores="$1"
  local start="$2"
  local end=$((start + cores - 1))
  echo "${start}-${end}"
}

for cores in ${CORES_LIST}; do
  cpuset="${CPUSET:-$(cpu_range "${cores}" "${CPUSET_START}")}"
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
    "${IMAGE}" \
    bash -lc "scripts/run_container_4c4g_inner.sh"

  uv run update_results_md.py \
    --results-md "${RESULTS_MD}" \
    --summary-table "${out_base}/summary_table.md" \
    --sut-name "${SUT_NAME:-default}" \
    --core-count "${cores}" \
    --memory-gb "${MEMORY_GB}"
done
