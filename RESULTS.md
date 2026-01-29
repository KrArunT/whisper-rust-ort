# Results

<!-- RESULTS:epyc-9654:4c:4g START -->
## epyc-9654 - 4 cores / 4GB RAM

### Run 2026-01-29T18:16:06

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

### Run 2026-01-29T18:16:06

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 2m52s | 1719MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 3m00s | 1917MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 17s | 3210MB |
| onnxruntime rust (int8) | int8 | 1 | 7s | 1790MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1136MB |
| faster-whisper (int8) | int8 | 1 | 43s | 908MB |

<!-- RESULTS:epyc-9654:8c:4g END -->
