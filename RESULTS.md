# Results

This file tracks benchmark outcomes. Update with each new experiment, and include the environment details (CPU model, threads, container limits).

## Container: 4 cores / 4GB RAM

Run the container comparison:

```bash
./run_container_4c4g_compare.sh
```

The latest table is written to:

- `results/benchmarks/container_4c4g/summary_table.md`
- `results/benchmarks/container_4c4g/summary_table.csv`

### Latest results (end-to-end p95 time)

Environment: container limited to 4 cores / 4GB RAM. CPU model not recorded.

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m21s | 1730MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m23s | 1680MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 12s | 2130MB |
| onnxruntime rust (int8) | int8 | 1 | 8s | 1231MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1136MB |
| faster-whisper (int8) | int8 | 1 | 29s | 905MB |
