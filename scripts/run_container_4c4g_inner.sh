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

WRITE_TXT_FLAG=()
if [[ "${WRITE_TXT}" != "0" ]]; then
  WRITE_TXT_FLAG=(--write-txt)
fi

if [[ -z "${OUT_BASE}" ]]; then
  if [[ -z "${SUT_NAME}" && "${CORE_COUNT}" == "4" && "${MEMORY_GB}" == "4" ]]; then
    OUT_BASE="${OUT_ROOT}/container_4c4g"
  else
    OUT_BASE="${OUT_ROOT}/container_${CORE_COUNT}c_${MEMORY_GB}g"
    if [[ -n "${SUT_NAME}" ]]; then
      OUT_BASE="${OUT_BASE}/${SUT_NAME}"
    fi
  fi
fi

LOG_DIR="${OUT_BASE}/logs"

export UV_CACHE_DIR
export OMP_NUM_THREADS="${CORE_COUNT}"
export ORT_INTRA_OP="${CORE_COUNT}"
export ORT_INTER_OP=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export RAYON_NUM_THREADS="${CORE_COUNT}"

mkdir -p "${LOG_DIR}"

run_timed() {
  local label="$1"
  shift
  /usr/bin/time -v "$@" 2> "${LOG_DIR}/${label}.time.txt"
}

if [[ ! -d whisper-base-with-past-int8 ]]; then
  run_timed "quantize_int8" uv run python quantize_onnx_int8.py
fi

cargo build --release

run_timed "with_hf_pipeline" \
  uv run python benchmark_with_hf_pipeline.py \
  --model_id "${HF_MODEL_ID}" \
  --audio_dir "${AUDIO_DIR}" \
  --onnx_dir "${ONNX_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --intra_op "${CORE_COUNT}" \
  --inter_op 1 \
  --num_beams "${NUM_BEAMS}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG[@]}" \
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
  --intra_op "${CORE_COUNT}" \
  --inter_op 1 \
  --num_beams "${NUM_BEAMS}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG[@]}" \
  --out_csv "${OUT_BASE}/without_hf_pipeline_py/inference_per_file.csv" \
  --out_json "${OUT_BASE}/without_hf_pipeline_py/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/without_hf_pipeline_py/inference_summary.json"

run_timed "without_hf_pipeline_rust" \
  ./target/release/whisper_ort_bench \
  --audio-dir "${AUDIO_DIR}" \
  --onnx-dir "${ONNX_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --intra-op 1 \
  --inter-op 1 \
  --chunk-parallelism "${CORE_COUNT}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG[@]}" \
  --out-csv "${OUT_BASE}/without_hf_pipeline_rust/inference_per_file.csv" \
  --out-json "${OUT_BASE}/without_hf_pipeline_rust/inference_per_file.json" \
  --out-summary-json "${OUT_BASE}/without_hf_pipeline_rust/inference_summary.json"

run_timed "without_hf_pipeline_rust_int8" \
  ./target/release/whisper_ort_bench \
  --audio-dir "${AUDIO_DIR}" \
  --onnx-dir "${ONNX_INT8_DIR}" \
  --language "${LANGUAGE}" \
  --task "${TASK}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --intra-op 1 \
  --inter-op 1 \
  --chunk-parallelism "${CORE_COUNT}" \
  --warmup "${WARMUP}" \
  "${WRITE_TXT_FLAG[@]}" \
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
  --best_of 1 \
  "${WRITE_TXT_FLAG[@]}" \
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
  --best_of 1 \
  "${WRITE_TXT_FLAG[@]}" \
  --out_csv "${OUT_BASE}/faster_whisper_int8/inference_per_file.csv" \
  --out_json "${OUT_BASE}/faster_whisper_int8/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/faster_whisper_int8/inference_summary.json"

uv run python compare_container_benchmarks.py \
  --results-dir "${OUT_BASE}" \
  --log-dir "${LOG_DIR}" \
  --out-md "${OUT_BASE}/summary_table.md" \
  --out-csv "${OUT_BASE}/summary_table.csv"
