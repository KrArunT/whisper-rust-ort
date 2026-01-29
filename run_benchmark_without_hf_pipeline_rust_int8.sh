#!/usr/bin/env bash
set -euo pipefail

uv run python quantize_onnx_int8.py \
  --src_dir whisper-base-with-past \
  --dst_dir whisper-base-with-past-int8

cargo run --release -- \
    --audio-dir audio \
    --onnx-dir whisper-base-with-past-int8 \
    --language en \
    --task transcribe \
    --max-new-tokens 128 \
    --intra-op 1 \
    --inter-op 1 \
    --chunk-parallelism 8 \
    --warmup 1 \
    --write-txt \
    --out-csv results/benchmarks/without_hf_pipeline_rust_int8/inference_per_file.csv \
    --out-json results/benchmarks/without_hf_pipeline_rust_int8/inference_per_file.json \
    --out-summary-json results/benchmarks/without_hf_pipeline_rust_int8/inference_summary.json
