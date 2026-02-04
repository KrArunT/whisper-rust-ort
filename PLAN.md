## ToDo
- Write an independent rust only inference engine/inference 
  server and modes as inference/benchmark with live streaming 
  and batch inference support for both f32 and int8.
- Investigate the impact of AVX-512 VNNI with int8 ORT.
- Add WER and CER based on whisper-base model transcription as base as 
  a column to the RESULTS.md/RESULTS.csv.
- Reduce Memory footprint by Onnx graph optimizations (o1/o2/o3/o4)
- Benchmark other models with optimized engine and optimized model.
- Refactor the Rust engine code as an library.
 
