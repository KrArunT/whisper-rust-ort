# Repository Guidelines

## Project Overview
This repo benchmarks `openai/whisper-base` on CPU using ONNX Runtime (Rust + Python). The goal is to reduce end‑to‑end transcription latency ensuring no drop in accuracy and compare: HF pipeline, Python without HF pipeline, Rust without HF pipeline, and faster‑whisper.

## Project Structure & Module Organization
- `src/main.rs`: Rust benchmark CLI (audio decode → log‑mel → encoder/decoder → decode → metrics).
- `benchmark_with_hf_pipeline.py`: HF pipeline benchmark outputting summary JSON.
- `benchmark_without_hf_pipeline.py`: Python no‑HF pipeline benchmark with detailed timing.
- `benchmark_faster_whisper.py`: faster‑whisper benchmark.
- `compare_container_benchmarks.py`: summarize container runs into a table.
- `update_results_md.py`: sync summary tables into `RESULTS.md`.
- Shell runners: `benchmark_without_hf_pipeline.sh`, `run_benchmark_without_hf_pipeline_rust.sh`, `run_all_and_compare.sh`.
- Container scripts: `Dockerfile.container`, `run_container_4c4g_compare.sh`, `scripts/run_container_4c4g_inner.sh`.
- Assets: `audio/` input files, `whisper-base-with-past/` ONNX models + tokenizer.
- Outputs: `results/benchmarks/**`, `results_py/benchmarks/**`, `RESULTS.md`.

## Build, Test, and Development Commands
- `cargo build --release`: build Rust benchmark.
- `cargo run --release -- --audio-dir audio --onnx-dir whisper-base-with-past ...`: run Rust benchmark.
- `uv run python benchmark_with_hf_pipeline.py`: run HF pipeline benchmark.
- `./benchmark_without_hf_pipeline.sh`: run Python no‑HF benchmark.
- `./run_benchmark_without_hf_pipeline_rust.sh`: run Rust no‑HF benchmark.
- `uv run python benchmark_faster_whisper.py --cpu_threads 8`: run faster‑whisper benchmark.
- `uv run python compare_end_to_end_latencies.py`: compare end‑to‑end p95 latencies.
- `./run_all_and_compare.sh`: run all benchmarks + compare.
- `./run_container_4c4g_compare.sh`: run container benchmarks (4 cores/4GB) + generate summary table.
- `CORES_LIST="4 8 16 32 64" ./run_container_4c4g_compare.sh`: run multiple core counts.

## Coding Style & Naming Conventions
- Python: 4‑space indentation, snake_case functions/files, UPPER_CASE constants.
- Rust: `rustfmt` default style (4‑space indent), snake_case.
- Output files live under `results/benchmarks/` with descriptive names (e.g., `inference_summary.json`).

## Testing Guidelines
- No automated test suite currently. If you add tests, place them under `tests/` and use `test_*.py` (Python) or standard Rust test modules.

## Commit & Pull Request Guidelines
- No established commit history in this repo. Use short imperative subjects (e.g., “Add faster‑whisper benchmark”).
- PRs should include: summary, performance impact, and example outputs/commands.

## Configuration Tips
- Set CPU threads explicitly for fair comparisons (`--intra_op 8` / `--inter_op 1` or Rust `--intra-op` / `--inter-op`).
- Ensure `whisper-base-with-past/tokenizer.json` exists for Rust decoding.

## Documentation 
- Ensure README is always updated.

## Reproducibility
- Always log the experiments we did for improvements to README.md and record the impact on latency.
- Always ensure a working version of the code in a backup folder.

## Benchmark with containers
- Always ensure a docker/conatainer deployment and benchmark with defined number of cores and Memory.
- Try to exploit the hardware architecture features.

## Results
- Log results of different experiments in a seperate RESULTS.md too.
- Capture container summary tables under `results/benchmarks/container_4c4g/summary_table.md`.
- Use `RESULTS.md` markers to keep per‑SUT, per‑core results in one file.
- For other core counts, use `results/benchmarks/container_<cores>c<memory>g/<SUT_NAME>/`.
