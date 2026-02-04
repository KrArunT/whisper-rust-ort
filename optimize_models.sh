uv run python scripts/optimize_onnx_whisper.py \
  --export_onnx \
  --quantize \
  --opt_levels o1,o2,o3,o4 \
  --isas avx2,avx512,vnni \
  --onnx_dir whisper-base-with-past \
  --out_dir models/whisper-base-optimized

