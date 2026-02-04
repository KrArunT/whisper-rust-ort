# Whisper ORT Benchmarks (Rust + Python)

This repo benchmarks `openai/whisper-base` on CPU with ONNX Runtime and compares
HF pipeline vs no‑HF Python vs Rust vs faster‑whisper. It focuses on end‑to‑end
latency and reproducible comparisons across systems under test (SUTs).

## Repository layout

- `src/main.rs`: Rust benchmark CLI (decode → log‑mel → encoder/decoder → decode).
- `benchmark_with_hf_pipeline.py`: HF pipeline benchmark.
- `benchmark_without_hf_pipeline.py`: Python no‑HF benchmark.
- `benchmark_faster_whisper.py`: faster‑whisper benchmark.
- `run_container_benchmarks.sh`: container runner and results updater.
- `update_results_md.py`: append results into `RESULTS.md` and `RESULTS.csv`.
- `whisper-base-with-past/`: ONNX export + `tokenizer.json`.
- `audio/`: input samples.
- `results/`: benchmark outputs.
- `RESULTS.md` / `RESULTS.csv`: consolidated run history.

## Local quick start

Build Rust:

```bash
cargo build --release
```

Ensure `tokenizer.json` exists:

```bash
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
cache_path = Path(hf_hub_download(repo_id="openai/whisper-base", filename="tokenizer.json"))
Path("whisper-base-with-past/tokenizer.json").write_bytes(cache_path.read_bytes())
print("Wrote tokenizer.json")
PY
```

Run the Rust benchmark:

```bash
cargo run --release -- \
  --audio-dir audio \
  --onnx-dir whisper-base-with-past \
  --language en \
  --task transcribe \
  --max-new-tokens 128 \
  --warmup 1 \
  --write-txt
```

Outputs land under `results/benchmarks/`.

## Container benchmarks (4 cores / 4GB RAM by default)

Run all benchmarks and write summary tables:

```bash
./run_container_benchmarks.sh
```

Run multiple core counts:

```bash
CORES_LIST="4 8 16 32 64" ./run_container_benchmarks.sh
```

Merge existing summary tables only (no benchmarks):

```bash
MERGE_ONLY=1 ./run_container_benchmarks.sh
```

Tag results per SUT:

```bash
SUT_NAME=epyc-9654 ./run_container_benchmarks.sh
```

Outputs:

- `results/benchmarks/container_4c4g/<SUT_NAME>/summary_table.md`
- `results/benchmarks/container_4c4g/<SUT_NAME>/summary_table.csv`
- `results/benchmarks/container_4c4g/<SUT_NAME>/logs/*.time.txt`

Other core counts are written under:

- `results/benchmarks/container_<cores>c<memory>g/<SUT_NAME>/`

Notes:

- The script exports `openai/whisper-base` to ONNX if missing.
- The summary table time is end‑to‑end p95 from each `inference_summary.json`.
- If files are root‑owned after Docker runs:
  ```bash
  sudo chown -R $USER:$USER results/benchmarks
  ```
- The container image prebuilds the Rust binary. Use `FORCE_BUILD_RUST=1` to
  rebuild from the mounted workspace.
- Use `CPUSET_LIST=0-3,8-11` to pin a specific core list (overrides `CPUSET_START`).

## Results

- `RESULTS.md` is appended per run (per SUT/core/memory).
- `RESULTS.csv` aggregates rows across runs with timestamp, SUT, cores, memory.
