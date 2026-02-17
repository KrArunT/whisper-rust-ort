# ⚡ Whisper ONNX Inference Benchmark

**Baseline (accuracy reference):** OpenAI Whisper `base` via python `whisper` library
**CPU pinning:** `explicit(PIN_CPUS=0-7,16-23) selected_cores=8 cpus=0,1,2,3,4,5,6,7`
**Time column:** average end-to-end latency per audio from `inference_per_file.csv`

| Implementation | Precision | Optimization | Instruction Set | Beam size | Time | RAM Usage | WER | CER |
|---------------|-----------|--------------|-----------------|-----------|------|-----------|-----|-----|
| onnxruntime rust | fp32 | o1 |  | 1 | 5.98s | 3024MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o2 |  | 1 | 6.11s | 3060MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o3 |  | 1 | 6.03s | 3034MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o4 |  | 1 | 6.07s | 3049MB | 13.60% | 12.58% |
| onnxruntime rust | int8 | o1 | avx2 | 1 | 3.47s | 1815MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o1 | avx512 | 1 | 3.46s | 1735MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o1 | vnni | 1 | 3.51s | 1719MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o2 | avx2 | 1 | 3.55s | 2564MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o2 | avx512 | 1 | 3.59s | 2494MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o2 | vnni | 1 | 3.61s | 2540MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o3 | avx2 | 1 | 3.61s | 2482MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o3 | avx512 | 1 | 3.55s | 2517MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o3 | vnni | 1 | 3.58s | 2523MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o4 | avx2 | 1 | 3.62s | 2493MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o4 | avx512 | 1 | 3.58s | 2517MB | 18.68% | 15.67% |
| onnxruntime rust | int8 | o4 | vnni | 1 | 3.63s | 2512MB | 18.68% | 15.67% |

## 🏎 Lowest Latency
- **onnxruntime rust**
- Optimization: **o1**
- Instruction set: **avx512**
- Time: **3.46s**
- WER/CER: **18.68%** / **15.67%**

## 🧠 Lowest Memory
- **onnxruntime rust**
- Optimization: **o1**
- Instruction set: **vnni**
- RAM: **1719MB**
- WER/CER: **18.68%** / **15.67%**

## 🎯 Best Accuracy
- Lowest WER Optimization: **o1** on **** (WER **13.60%**)
- Lowest CER Optimization: **o1** on **** (CER **12.58%**)


# ⚡ Whisper ONNX Inference Benchmark

**Baseline (accuracy reference):** OpenAI Whisper `base` via python `whisper` library
**CPU pinning:** `explicit(PIN_CPUS=0-7,16-23) selected_cores=16 cpus=0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23`
**Time column:** average end-to-end latency per audio from `inference_per_file.csv`

| Implementation | Precision | Optimization | Instruction Set | Beam size | Time | RAM Usage | WER | CER |
|---------------|-----------|--------------|-----------------|-----------|------|-----------|-----|-----|
| onnxruntime rust | fp32 | o1 |  | 1 | 3.52s | 3892MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o2 |  | 1 | 3.60s | 3904MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o3 |  | 1 | 3.62s | 3858MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o4 |  | 1 | 3.53s | 3876MB | 13.60% | 12.58% |
| onnxruntime rust | int8 | o1 | avx2 | 1 | 2.72s | 2284MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o1 | avx512 | 1 | 2.69s | 2302MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o1 | vnni | 1 | 2.64s | 2314MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o2 | avx2 | 1 | 2.81s | 3374MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o2 | avx512 | 1 | 2.74s | 3435MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o2 | vnni | 1 | 2.64s | 3357MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o3 | avx2 | 1 | 2.68s | 3411MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o3 | avx512 | 1 | 2.71s | 3385MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o3 | vnni | 1 | 2.69s | 3411MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o4 | avx2 | 1 | 2.67s | 3379MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o4 | avx512 | 1 | 2.69s | 3397MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o4 | vnni | 1 | 2.71s | 3392MB | 19.64% | 16.93% |

## 🏎 Lowest Latency
- **onnxruntime rust**
- Optimization: **o1**
- Instruction set: **vnni**
- Time: **2.64s**
- WER/CER: **19.64%** / **16.93%**

## 🧠 Lowest Memory
- **onnxruntime rust**
- Optimization: **o1**
- Instruction set: **avx2**
- RAM: **2284MB**
- WER/CER: **19.64%** / **16.93%**

## 🎯 Best Accuracy
- Lowest WER Optimization: **o1** on **** (WER **13.60%**)
- Lowest CER Optimization: **o1** on **** (CER **12.58%**)

# ⚡ Whisper ONNX Inference Benchmark

**Baseline (accuracy reference):** OpenAI Whisper `base` via python `whisper` library
**CPU pinning:** `explicit(PIN_CPUS=0-31) selected_cores=32 cpus=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31`
**Time column:** average end-to-end latency per audio from `inference_per_file.csv`

| Implementation | Precision | Optimization | Instruction Set | Beam size | Time | RAM Usage | WER | CER |
|---------------|-----------|--------------|-----------------|-----------|------|-----------|-----|-----|
| onnxruntime rust | fp32 | o1 |  | 1 | 3.46s | 3864MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o2 |  | 1 | 3.57s | 3870MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o3 |  | 1 | 3.31s | 3945MB | 13.60% | 12.58% |
| onnxruntime rust | fp32 | o4 |  | 1 | 3.41s | 3919MB | 13.60% | 12.58% |
| onnxruntime rust | int8 | o1 | avx2 | 1 | 2.59s | 2252MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o1 | avx512 | 1 | 2.59s | 2264MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o1 | vnni | 1 | 2.59s | 2246MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o2 | avx2 | 1 | 2.53s | 3369MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o2 | avx512 | 1 | 2.63s | 3375MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o2 | vnni | 1 | 2.58s | 3388MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o3 | avx2 | 1 | 2.60s | 3321MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o3 | avx512 | 1 | 2.50s | 3410MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o3 | vnni | 1 | 2.54s | 3352MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o4 | avx2 | 1 | 2.56s | 3363MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o4 | avx512 | 1 | 2.62s | 3352MB | 19.64% | 16.93% |
| onnxruntime rust | int8 | o4 | vnni | 1 | 2.57s | 3328MB | 19.64% | 16.93% |

## 🏎 Lowest Latency
- **onnxruntime rust**
- Optimization: **o3**
- Instruction set: **avx512**
- Time: **2.50s**
- WER/CER: **19.64%** / **16.93%**

## 🧠 Lowest Memory
- **onnxruntime rust**
- Optimization: **o1**
- Instruction set: **vnni**
- RAM: **2246MB**
- WER/CER: **19.64%** / **16.93%**

## 🎯 Best Accuracy
- Lowest WER Optimization: **o1** on **** (WER **13.60%**)
- Lowest CER Optimization: **o1** on **** (CER **12.58%**)
