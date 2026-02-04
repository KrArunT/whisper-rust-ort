#!/usr/bin/env bash
set -euo pipefail

CORE_COUNT="${CORE_COUNT:-4}"
MEMORY_GB="${MEMORY_GB:-4}"
SUT_NAME="${SUT_NAME:-}"
OUT_ROOT="${OUT_ROOT:-results/benchmarks}"
OUT_BASE="${OUT_BASE:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
AUDIO_DIR="${AUDIO_DIR:-audio}"
ONNX_DIR="${ONNX_DIR:-whisper-base-with-past}"
ONNX_INT8_DIR="${ONNX_INT8_DIR:-whisper-base-with-past-int8}"
LANGUAGE="${LANGUAGE:-en}"
TASK="${TASK:-transcribe}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_BEAMS="${NUM_BEAMS:-1}"
WARMUP="${WARMUP:-1}"
WRITE_TXT="${WRITE_TXT:-1}"
HF_MODEL_ID="${HF_MODEL_ID:-openai/whisper-base}"
FW_MODEL_ID="${FW_MODEL_ID:-Systran/faster-whisper-base}"
RUST_BIN="${RUST_BIN:-/usr/local/bin/whisper_ort_bench}"
FORCE_BUILD_RUST="${FORCE_BUILD_RUST:-0}"
EXPORT_ONNX="${EXPORT_ONNX:-1}"

WRITE_TXT_FLAG_PY=()
WRITE_TXT_FLAG_RS=()
if [[ "${WRITE_TXT}" != "0" ]]; then
  WRITE_TXT_FLAG_PY=(--write_txt)
  WRITE_TXT_FLAG_RS=(--write-txt)
fi

if [[ -z "${OUT_BASE}" ]]; then
  if [[ "${CORE_COUNT}" == "4" && "${MEMORY_GB}" == "4" ]]; then
    OUT_BASE="${OUT_ROOT}/container_4c4g"
  else
    OUT_BASE="${OUT_ROOT}/container_${CORE_COUNT}c${MEMORY_GB}g"
  fi
  if [[ -n "${SUT_NAME}" ]]; then
    OUT_BASE="${OUT_BASE}/${SUT_NAME}"
  fi
fi

LOG_DIR="${OUT_BASE}/logs"

export UV_CACHE_DIR

if [[ "${CORE_COUNT}" -le 16 ]]; then
  ORT_INTER_OP="${ORT_INTER_OP:-1}"
elif [[ "${CORE_COUNT}" -le 32 ]]; then
  ORT_INTER_OP="${ORT_INTER_OP:-2}"
else
  ORT_INTER_OP="${ORT_INTER_OP:-4}"
fi
ORT_INTRA_OP="${ORT_INTRA_OP:-${CORE_COUNT}}"

export OMP_NUM_THREADS="${CORE_COUNT}"
export ORT_INTRA_OP
export ORT_INTER_OP
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export RAYON_NUM_THREADS="${CORE_COUNT}"

mkdir -p "${LOG_DIR}"

run_timed() {
  local label="$1"
  shift
  /usr/bin/time -v "$@" 2> "${LOG_DIR}/${label}.time.txt"
}

if [[ "${EXPORT_ONNX}" != "0" ]]; then
  if [[ ! -f "${ONNX_DIR}/encoder_model.onnx" || ! -f "${ONNX_DIR}/decoder_model.onnx" ]]; then
    run_timed "export_onnx" uv run python scripts/export_onnx_whisper.py \
      --model_id "${HF_MODEL_ID}" \
      --onnx_dir "${ONNX_DIR}"
  fi
fi

if [[ ! -d "${ONNX_INT8_DIR}" ]]; then
  run_timed "quantize_int8" uv run python quantize_onnx_int8.py \
    --src_dir "${ONNX_DIR}" \
    --dst_dir "${ONNX_INT8_DIR}"
fi

if [[ "${FORCE_BUILD_RUST}" != "0" || ! -x "${RUST_BIN}" ]]; then
  cargo build --release
  RUST_BIN="./target/release/whisper_ort_bench"
fi

run_timed "with_hf_pipeline" \
  uv run python benchmark_with_hf_pipeline.py \
  --model_id "${HF_MODEL_ID}" \
  --audio_dir "${AUDIO_DIR}" \
  --onnx_dir "${ONNX_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --intra_op "${ORT_INTRA_OP}" \
  --inter_op "${ORT_INTER_OP}" \
  --num_beams "${NUM_BEAMS}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG_PY[@]}" \
  --out_csv "${OUT_BASE}/with_hf_pipeline/inference_per_file.csv" \
  --out_json "${OUT_BASE}/with_hf_pipeline/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/with_hf_pipeline/inference_summary.json"

run_timed "without_hf_pipeline_py" \
  uv run python benchmark_without_hf_pipeline.py \
  --model_id "${HF_MODEL_ID}" \
  --audio_dir "${AUDIO_DIR}" \
  --onnx_dir "${ONNX_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --intra_op "${ORT_INTRA_OP}" \
  --inter_op "${ORT_INTER_OP}" \
  --num_beams "${NUM_BEAMS}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG_PY[@]}" \
  --out_csv "${OUT_BASE}/without_hf_pipeline_py/inference_per_file.csv" \
  --out_json "${OUT_BASE}/without_hf_pipeline_py/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/without_hf_pipeline_py/inference_summary.json"

run_timed "without_hf_pipeline_rust" \
  "${RUST_BIN}" \
  --audio-dir "${AUDIO_DIR}" \
  --onnx-dir "${ONNX_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --intra-op "${ORT_INTRA_OP}" \
  --inter-op "${ORT_INTER_OP}" \
  --chunk-parallelism "${CORE_COUNT}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG_RS[@]}" \
  --out-csv "${OUT_BASE}/without_hf_pipeline_rust/inference_per_file.csv" \
  --out-json "${OUT_BASE}/without_hf_pipeline_rust/inference_per_file.json" \
  --out-summary-json "${OUT_BASE}/without_hf_pipeline_rust/inference_summary.json"

run_timed "without_hf_pipeline_rust_int8" \
  "${RUST_BIN}" \
  --audio-dir "${AUDIO_DIR}" \
  --onnx-dir "${ONNX_INT8_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --intra-op "${ORT_INTRA_OP}" \
  --inter-op "${ORT_INTER_OP}" \
  --chunk-parallelism "${CORE_COUNT}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG_RS[@]}" \
  --out-csv "${OUT_BASE}/without_hf_pipeline_rust_int8/inference_per_file.csv" \
  --out-json "${OUT_BASE}/without_hf_pipeline_rust_int8/inference_per_file.json" \
  --out-summary-json "${OUT_BASE}/without_hf_pipeline_rust_int8/inference_summary.json"

run_timed "faster_whisper_fp32" \
  uv run python benchmark_faster_whisper.py \
  --model_id "${FW_MODEL_ID}" \
  --audio_dir "${AUDIO_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --cpu_threads "${CORE_COUNT}" \
  --num_workers 1 \
  --compute_type float32 \
  --beam_size "${NUM_BEAMS}" \
  --best_of "${NUM_BEAMS}" \
  "${WRITE_TXT_FLAG_PY[@]}" \
  --out_csv "${OUT_BASE}/faster_whisper_fp32/inference_per_file.csv" \
  --out_json "${OUT_BASE}/faster_whisper_fp32/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/faster_whisper_fp32/inference_summary.json"

run_timed "faster_whisper_int8" \
  uv run python benchmark_faster_whisper.py \
  --model_id "${FW_MODEL_ID}" \
  --audio_dir "${AUDIO_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --cpu_threads "${CORE_COUNT}" \
  --num_workers 1 \
  --compute_type int8 \
  --beam_size "${NUM_BEAMS}" \
  --best_of "${NUM_BEAMS}" \
  "${WRITE_TXT_FLAG_PY[@]}" \
  --out_csv "${OUT_BASE}/faster_whisper_int8/inference_per_file.csv" \
  --out_json "${OUT_BASE}/faster_whisper_int8/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/faster_whisper_int8/inference_summary.json"

uv run python compare_container_benchmarks.py \
  --results-dir "${OUT_BASE}" \
  --log-dir "${LOG_DIR}" \
  --out-md "${OUT_BASE}/summary_table.md" \
  --out-csv "${OUT_BASE}/summary_table.csv"
