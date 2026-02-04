| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 2m52s | 1719MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 3m00s | 1917MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 17s | 3210MB |
| onnxruntime rust (int8) | int8 | 1 | 7s | 1790MB |
| faster-whisper (fp32) | fp32 | 1 | 21s | 1136MB |
| faster-whisper (int8) | int8 | 1 | 43s | 908MB |
