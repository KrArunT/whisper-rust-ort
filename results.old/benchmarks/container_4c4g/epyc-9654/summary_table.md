| Implementation | Precision | Beam size | Time | RAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper (HF pipeline) | fp32 | 1 | 1m13s | 1724MB |
| onnxruntime python (no HF pipeline) | fp32 | 1 | 1m25s | 1906MB |
| onnxruntime rust (no HF pipeline) | fp32 | 1 | 15s | 2126MB |
| onnxruntime rust (int8) | int8 | 1 | 9s | 1206MB |
| faster-whisper (fp32) | fp32 | 1 | 23s | 1133MB |
| faster-whisper (int8) | int8 | 1 | 34s | 905MB |
