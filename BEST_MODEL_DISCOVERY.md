# ⚡ Whisper ONNX Inference Benchmark

| Implementation | Precision | Optimization | Beam size | Time | RAM Usage |
|---------------|-----------|--------------|-----------|------|-----------|
| onnxruntime rust | fp32 | fp32 | 1 | 6s | 2991MB |
| onnxruntime rust | int8 | int8-avx2 | 1 | 4s | 1942MB |
| onnxruntime rust | int8 | int8-avx512 | 1 | 4s | 1880MB |
| onnxruntime rust | int8 | int8-vnni | 1 | 4s | 1936MB |
| onnxruntime rust | fp32 | fp32 | 1 | 7s | 2992MB |
| onnxruntime rust | int8 | int8-avx2 | 1 | 4s | 2493MB |
| onnxruntime rust | int8 | int8-avx512 | 1 | 4s | 2475MB |
| onnxruntime rust | int8 | int8-vnni | 1 | 4s | 2517MB |
| onnxruntime rust | fp32 | fp32 | 1 | 6s | 2998MB |
| onnxruntime rust | int8 | int8-avx2 | 1 | 4s | 2523MB |
| onnxruntime rust | int8 | int8-avx512 | 1 | 4s | 2499MB |
| onnxruntime rust | int8 | int8-vnni | 1 | 4s | 2487MB |
| onnxruntime rust | fp32 | fp32 | 1 | 6s | 2986MB |
| onnxruntime rust | int8 | int8-avx2 | 1 | 4s | 2475MB |
| onnxruntime rust | int8 | int8-avx512 | 1 | 4s | 2505MB |
| onnxruntime rust | int8 | int8-vnni | 1 | 4s | 2487MB |

## 🏎 Lowest Latency
- **onnxruntime rust**
- Optimization: **int8-avx512**
- Time: **4s**

## 🧠 Lowest Memory
- **onnxruntime rust**
- Optimization: **int8-avx512**
- RAM: **1880MB**
