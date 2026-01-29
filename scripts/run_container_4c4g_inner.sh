#!/usr/bin/env bash
set -euo pipefail

OUT_BASE="${OUT_BASE:-results/benchmarks/container_4c4g}"
LOG_DIR="${OUT_BASE}/logs"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

export UV_CACHE_DIR
export OMP_NUM_THREADS=4
export ORT_INTRA_OP=4
export ORT_INTER_OP=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export RAYON_NUM_THREADS=4

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
  --intra_op 4 \
  --inter_op 1 \
  --num_beams 1 \
  --out_csv "${OUT_BASE}/with_hf_pipeline/inference_per_file.csv" \
  --out_json "${OUT_BASE}/with_hf_pipeline/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/with_hf_pipeline/inference_summary.json"

run_timed "without_hf_pipeline_py" \
  uv run python benchmark_without_hf_pipeline.py \
  --intra_op 4 \
  --inter_op 1 \
  --num_beams 1 \
  --out_csv "${OUT_BASE}/without_hf_pipeline_py/inference_per_file.csv" \
  --out_json "${OUT_BASE}/without_hf_pipeline_py/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/without_hf_pipeline_py/inference_summary.json"

run_timed "without_hf_pipeline_rust" \
  ./target/release/whisper_ort_bench \
  --audio-dir audio \
  --onnx-dir whisper-base-with-past \
  --language en \
  --task transcribe \
  --max-new-tokens 128 \
  --intra-op 1 \
  --inter-op 1 \
  --chunk-parallelism 4 \
  --warmup 1 \
  --write-txt \
  --out-csv "${OUT_BASE}/without_hf_pipeline_rust/inference_per_file.csv" \
  --out-json "${OUT_BASE}/without_hf_pipeline_rust/inference_per_file.json" \
  --out-summary-json "${OUT_BASE}/without_hf_pipeline_rust/inference_summary.json"

run_timed "without_hf_pipeline_rust_int8" \
  ./target/release/whisper_ort_bench \
  --audio-dir audio \
  --onnx-dir whisper-base-with-past-int8 \
  --language en \
  --task transcribe \
  --max-new-tokens 128 \
  --intra-op 1 \
  --inter-op 1 \
  --chunk-parallelism 4 \
  --warmup 1 \
  --write-txt \
  --out-csv "${OUT_BASE}/without_hf_pipeline_rust_int8/inference_per_file.csv" \
  --out-json "${OUT_BASE}/without_hf_pipeline_rust_int8/inference_per_file.json" \
  --out-summary-json "${OUT_BASE}/without_hf_pipeline_rust_int8/inference_summary.json"

run_timed "faster_whisper_fp32" \
  uv run python benchmark_faster_whisper.py \
  --cpu_threads 4 \
  --num_workers 1 \
  --compute_type float32 \
  --beam_size 1 \
  --best_of 1 \
  --out_csv "${OUT_BASE}/faster_whisper_fp32/inference_per_file.csv" \
  --out_json "${OUT_BASE}/faster_whisper_fp32/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/faster_whisper_fp32/inference_summary.json"

run_timed "faster_whisper_int8" \
  uv run python benchmark_faster_whisper.py \
  --cpu_threads 4 \
  --num_workers 1 \
  --compute_type int8 \
  --beam_size 1 \
  --best_of 1 \
  --out_csv "${OUT_BASE}/faster_whisper_int8/inference_per_file.csv" \
  --out_json "${OUT_BASE}/faster_whisper_int8/inference_per_file.json" \
  --out_summary_json "${OUT_BASE}/faster_whisper_int8/inference_summary.json"

uv run python compare_container_benchmarks.py \
  --results-dir "${OUT_BASE}" \
  --log-dir "${LOG_DIR}" \
  --out-md "${OUT_BASE}/summary_table.md" \
  --out-csv "${OUT_BASE}/summary_table.csv"
