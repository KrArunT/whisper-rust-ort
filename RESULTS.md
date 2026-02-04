# Results

<!-- RESULTS:BLR-L-ARTIWARY:4c:4g START -->
## BLR-L-ARTIWARY - 4 cores / 4GB RAM

### Run 2026-02-04T12:19:53

| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m16s | 1727MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m17s | 1893MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 13s | 2131MB |
| onnxruntime rust (int8) | int8 | 1 | 8s | 1225MB |
| faster-whisper (fp32) | fp32 | 1 | 22s | 1128MB |
| faster-whisper (int8) | int8 | 1 | 32s | 902MB |

<!-- RESULTS:BLR-L-ARTIWARY:4c:4g END -->
