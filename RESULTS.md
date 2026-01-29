# Results

This file tracks benchmark outcomes. Update it via `update_results_md.py` after
each container run. Each section is keyed by SUT name and core/memory limits.

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
  --sut-name default \
  --core-count 4 \
  --memory-gb 4
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
