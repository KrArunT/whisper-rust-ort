#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

uv run python benchmark_with_hf_pipeline.py --intra_op 8 --inter_op 1
./benchmark_without_hf_pipeline.sh
./run_benchmark_without_hf_pipeline_rust.sh
./run_benchmark_without_hf_pipeline_rust_int8.sh
uv run python benchmark_faster_whisper.py --cpu_threads 8 --num_workers 1 --model_id Systran/faster-whisper-base
uv run python compare_end_to_end_latencies.py
