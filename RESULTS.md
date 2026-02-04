# Results

<!-- RESULTS:BLR-L-ARTIWARY:4c:4g START -->
## BLR-L-ARTIWARY - 4 cores / 4GB RAM

### Run 2026-02-04T13:06:21

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m29s | 1703MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m36s | 1945MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 23s | 2188MB |
| onnxruntime rust (int8) | int8 | 1 | 16s | 1316MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1131MB |
| faster-whisper (int8) | int8 | 1 | 29s | 905MB |

<!-- RESULTS:BLR-L-ARTIWARY:4c:4g END -->
