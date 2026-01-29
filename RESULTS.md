# Results

This file tracks benchmark outcomes. Update it via `update_results_md.py` after
each container run. Each section is keyed by SUT name and core/memory limits,
and new runs append to the existing section.

## How to update

Run the container workflow and it will update this file automatically:

```bash
CORES_LIST="4 8 16 32 64" ./run_container_4c4g_compare.sh
```

Manual update example:

```bash
python3 update_results_md.py \
  --results-md RESULTS.md \
  --summary-table results/benchmarks/container_4c4g/summary_table.md \
  --summary-csv results/benchmarks/container_4c4g/summary_table.csv \
  --sut-name default \
  --core-count 4 \
  --memory-gb 4 \
  --results-csv RESULTS.csv
```

<!-- RESULTS:default:4c:4g START -->
## default - 4 cores / 4GB RAM
Updated: not recorded

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m21s | 1730MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m23s | 1680MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 12s | 2130MB |
| onnxruntime rust (int8) | int8 | 1 | 8s | 1231MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1136MB |
| faster-whisper (int8) | int8 | 1 | 29s | 905MB |

<!-- RESULTS:default:4c:4g END -->

<!-- RESULTS:BLR-L-ARTIWARY:4c:4g START -->
## BLR-L-ARTIWARY - 4 cores / 4GB RAM
Updated: 2026-01-29T17:20:43

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m16s | 1724MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m16s | 1910MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | n/a | 2MB |
| onnxruntime rust (int8) | int8 | 1 | n/a | 2MB |
| faster-whisper (fp32) | fp32 | 1 | 23s | 1142MB |
| faster-whisper (int8) | int8 | 1 | 29s | 908MB |

### Run 2026-01-29T17:56:05

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m16s | 1724MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m16s | 1910MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | n/a | 2MB |
| onnxruntime rust (int8) | int8 | 1 | n/a | 2MB |
| faster-whisper (fp32) | fp32 | 1 | 23s | 1142MB |
| faster-whisper (int8) | int8 | 1 | 29s | 908MB |

### Run 2026-01-29T18:00:42

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m16s | 1724MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m16s | 1910MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | n/a | 2MB |
| onnxruntime rust (int8) | int8 | 1 | n/a | 2MB |
| faster-whisper (fp32) | fp32 | 1 | 23s | 1142MB |
| faster-whisper (int8) | int8 | 1 | 29s | 908MB |
<!-- RESULTS:BLR-L-ARTIWARY:4c:4g END -->

<!-- RESULTS:epyc-9654:4c:4g START -->
## epyc-9654 - 4 cores / 4GB RAM
Updated: 2026-01-29T17:33:37

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m13s | 1724MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m25s | 1906MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 15s | 2126MB |
| onnxruntime rust (int8) | int8 | 1 | 9s | 1206MB |
| faster-whisper (fp32) | fp32 | 1 | 23s | 1133MB |
| faster-whisper (int8) | int8 | 1 | 34s | 905MB |

### Run 2026-01-29T18:00:42

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m13s | 1724MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m25s | 1906MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 15s | 2126MB |
| onnxruntime rust (int8) | int8 | 1 | 9s | 1206MB |
| faster-whisper (fp32) | fp32 | 1 | 23s | 1133MB |
| faster-whisper (int8) | int8 | 1 | 34s | 905MB |
<!-- RESULTS:epyc-9654:4c:4g END -->

<!-- RESULTS:epyc-9654:8c:4g START -->
## epyc-9654 - 8 cores / 4GB RAM
Updated: 2026-01-29T17:49:42

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 2m52s | 1719MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 3m00s | 1917MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 17s | 3210MB |
| onnxruntime rust (int8) | int8 | 1 | 7s | 1790MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1136MB |
| faster-whisper (int8) | int8 | 1 | 43s | 908MB |

### Run 2026-01-29T18:00:42

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 2m52s | 1719MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 3m00s | 1917MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 17s | 3210MB |
| onnxruntime rust (int8) | int8 | 1 | 7s | 1790MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1136MB |
| faster-whisper (int8) | int8 | 1 | 43s | 908MB |
<!-- RESULTS:epyc-9654:8c:4g END -->
