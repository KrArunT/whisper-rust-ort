# Whisper ORT (Rust) Benchmark

Optimize end-to-end transcription latency for FP32 `openai/whisper-base` on CPU using ONNX Runtime. This Rust binary mirrors the Python benchmark in `infer_without_hf_pipeline.py` and writes per-file and summary metrics.

## What’s in here

- `src/main.rs`: Rust benchmark CLI (audio decode → log-mel → encoder → decoder → decode → metrics).
- `whisper-base-with-past/`: ONNX export (encoder / decoder / decoder_with_past) + generation config.
- `audio/`: input samples.
- `benchmark_with_hf_pipeline.py`: HF pipeline benchmark (summary JSON).
- `benchmark_without_hf_pipeline.py`: Python no‑HF pipeline benchmark.
- `benchmark_faster_whisper.py`: faster‑whisper benchmark.
- `compare_container_benchmarks.py`: build a summary table from container runs.
- `run_container_4c4g_compare.sh`: run all benchmarks in a 4c/4GB container.
- `results/`: benchmark outputs (including container summary tables).
- `results_py/`: baseline outputs from Python for comparison.
- `RESULTS.md`: experiment log.

## Quick start

### 1) Build

```bash
cargo build --release
```

### 2) Ensure tokenizer.json exists

The Rust decoder uses `tokenizers` and needs a Whisper tokenizer. Place it here:

```
whisper-base-with-past/tokenizer.json
```

If you have Hugging Face cache, you can copy it from there. Example:

```bash
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "openai/whisper-base"
filename = "tokenizer.json"

cache_path = Path(hf_hub_download(repo_id=repo, filename=filename))
Path("whisper-base-with-past/tokenizer.json").write_bytes(cache_path.read_bytes())
print("Wrote tokenizer.json")
PY
```

### 3) Run the benchmark

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

Outputs:

- `results/benchmarks/inference_per_file.csv`
- `results/benchmarks/inference_per_file.json`
- `results/benchmarks/inference_summary.json`
- `results/benchmarks/*.transcript.txt`

## Comparing with Python baseline

`results_py/` contains outputs from `infer_without_hf_pipeline.py` using HF’s processor + generate. To compare transcripts:

```bash
python3 - <<'PY'
import json, difflib
from pathlib import Path

py = Path('results_py/benchmarks/inference_per_file.json')
rs = Path('results/benchmarks/inference_per_file.json')

py_data = json.loads(py.read_text())
rs_data = json.loads(rs.read_text())

py_map = {d['file']: d for d in py_data}
rs_map = {d['file']: d for d in rs_data}

common = sorted(set(py_map) & set(rs_map))
for f in common:
    py_text = py_map[f]['text']
    rs_text = rs_map[f]['text']
    if py_text != rs_text:
        print('---', f)
        diff = list(difflib.unified_diff(py_text.splitlines(), rs_text.splitlines(), lineterm='', fromfile='py', tofile='rs'))
        print('\n'.join(diff[:80]))
PY
```

## Notes on decoding

- The first decoding step uses `decoder_model.onnx` (needs `encoder_hidden_states`) to seed past KV.
- Subsequent steps use `decoder_with_past_model.onnx` with cached KV.
- `<|notimestamps|>` is added by default unless `--timestamps` is set.
- `generation_config.json` is used to suppress special tokens.
- Chunked decoding is stitched with overlap de-duplication.

## Threading controls

- ORT settings are applied in Rust via `SessionBuilder`.
- You can also set `OMP_NUM_THREADS` externally to reduce variance.

## CLI flags (common)

- `--audio-dir` (default: `audio`)
- `--onnx-dir` (default: `whisper-base-with-past`)
- `--language`, `--task`
- `--max-new-tokens`
- `--warmup`
- `--write-txt`
- `--tokenizer-json` (optional override)
- `--timestamps` (emit timestamps if the model supports them)

## Reproducibility

- Keep `whisper-base-with-past/` unchanged between runs.
- Use the same `audio/` files and the same `tokenizer.json`.
- Record CPU model and thread settings along with `results/benchmarks/inference_summary.json`.
- If you run the container workflow, record the container limits and CPU pinning.

## Results

The latest consolidated results live in `RESULTS.md`. The container summary table
is generated at `results/benchmarks/container_4c4g/summary_table.md`.

## Container comparison (4 cores / 4GB RAM)

Run all implementations inside a pinned container (4 cores, 4GB) and write a summary table:

```bash
./run_container_4c4g_compare.sh
```

Run multiple core counts (default is 4 only):

```bash
CORES_LIST="4 8 16 32 64" ./run_container_4c4g_compare.sh
```

Tag results per system-under-test (SUT):

```bash
SUT_NAME=epyc-9654 ./run_container_4c4g_compare.sh
```

Outputs:

- `results/benchmarks/container_4c4g/summary_table.md`
- `results/benchmarks/container_4c4g/summary_table.csv`
- `results/benchmarks/container_4c4g/logs/*.time.txt` (RSS and wall time)

For other core counts or SUT names, outputs are written under:

- `results/benchmarks/container_<cores>c<memory>g/<SUT_NAME>/`

Optional overrides:

- `CPUSET=0-3` to pick specific CPU cores.
- `IMAGE=whisper-rust-ort:4c4g` to reuse a prebuilt image.

Note: the Time column in `summary_table.md` uses end‑to‑end p95 latency from each
`inference_summary.json` (falls back to wall time if missing).

If the container runs as root, output files may be owned by root on the host.
Fix with:

```bash
sudo chown -R $USER:$USER results/benchmarks/container_4c4g
```

`RESULTS.md` is updated automatically by `run_container_4c4g_compare.sh`. You can
also update it manually with `update_results_md.py`.

Note: the container image now prebuilds the Rust binary to avoid downloading
crates on every run. Set `FORCE_BUILD_RUST=1` if you need to rebuild from the
mounted workspace sources.
