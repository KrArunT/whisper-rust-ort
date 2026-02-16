#!/usr/bin/env bash
set -euo pipefail

##################################################
# CONFIG
##################################################

MODELS_ROOT="${MODELS_ROOT:-models/whisper-base-optimized}"
AUDIO_DIR="${AUDIO_DIR:-audio}"
RESULTS_ROOT="${RESULTS_ROOT:-results/benchmarks/without_hf_pipeline_rust}"

NUMA_NODE="${NUMA_NODE:-auto}"
PIN_STRATEGY="${PIN_STRATEGY:-ccd}"   # ccd | numa | flat
MAX_CORES_PER_RUN="${MAX_CORES_PER_RUN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"

##################################################
# Helpers
##################################################

time_to_seconds() {
  awk -F: '{
    if (NF==3)      { printf "%.3f", ($1*3600)+($2*60)+$3 }
    else if (NF==2) { printf "%.3f", ($1*60)+$2 }
    else            { printf "%.3f", $1 }
  }' <<< "$1"
}

pretty_time() {
  awk -v t="$1" 'BEGIN{
    if (t < 60) printf "%.0fs", t
    else printf "%dm%ds", int(t/60), int(t%60)
  }'
}

precision_from_model() {
  [[ "$1" == *int8* ]] && echo "int8" || echo "fp32"
}

optimization_from_model() {
  case "$1" in
    *int8_vnni*)   echo "int8-vnni" ;;
    *int8_avx512*) echo "int8-avx512" ;;
    *int8_avx2*)   echo "int8-avx2" ;;
    *fp32*)        echo "fp32" ;;
    *)             echo "unknown" ;;
  esac
}

implementation_name() {
  echo "onnxruntime rust"
}

##################################################
# Setup
##################################################

mkdir -p "$RESULTS_ROOT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MERGED_MODEL_CSV="$RESULTS_ROOT/merged_inference_results.csv"
REPORT_MD="$RESULTS_ROOT/BENCHMARK_REPORT.md"

##################################################
# CSV init
##################################################

echo "implementation,precision,optimization,beam_size,time_sec,ram_mb" \
  > "$MERGED_MODEL_CSV"

##################################################
# Benchmark loop
##################################################

for onnx_dir in "$MODELS_ROOT"/*; do
  [[ -d "$onnx_dir" ]] || continue
  model_name="$(basename "$onnx_dir")"

  echo "🚀 Benchmarking $model_name"

  TIME_LOG="$TMP_DIR/time_${model_name}.txt"

  /usr/bin/time -v -o "$TIME_LOG" \
    cargo run --release -- \
      --audio-dir "$AUDIO_DIR" \
      --onnx-dir "$onnx_dir" \
      --language en \
      --task transcribe \
      --max-new-tokens 128 \
      --num-beams "$NUM_BEAMS" \
      --intra-op 1 \
      --inter-op 1 \
      --chunk-parallelism 8 \
      --warmup 1

  RAW_TIME=$(grep "Elapsed (wall clock) time" "$TIME_LOG" | awk '{print $NF}')
  TIME_SEC=$(time_to_seconds "$RAW_TIME")

  PEAK_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
  PEAK_MB=$(awk "BEGIN { printf \"%.0f\", $PEAK_KB / 1024 }")

  PRECISION=$(precision_from_model "$model_name")
  OPT_TYPE=$(optimization_from_model "$model_name")
  IMPL=$(implementation_name)

  echo "$IMPL,$PRECISION,$OPT_TYPE,$NUM_BEAMS,$TIME_SEC,$PEAK_MB" \
    >> "$MERGED_MODEL_CSV"
done

##################################################
# Best models
##################################################

BEST_LATENCY=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k5 -n | head -n1)
BEST_MEMORY=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k6 -n | head -n1)

##################################################
# Markdown Report (comparison-table style)
##################################################

{
echo "# ⚡ Whisper ONNX Inference Benchmark"
echo
echo "| Implementation | Precision | Optimization | Beam size | Time | RAM Usage |"
echo "|---------------|-----------|--------------|-----------|------|-----------|"

tail -n +2 "$MERGED_MODEL_CSV" | \
while IFS=, read impl prec opt beam t ram; do
  printf "| %s | %s | %s | %s | %s | %sMB |\n" \
    "$impl" "$prec" "$opt" "$beam" "$(pretty_time "$t")" "$ram"
done

echo
echo "## 🏎 Lowest Latency"
echo "- **$(cut -d, -f1 <<<"$BEST_LATENCY")**"
echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_LATENCY")**"
echo "- Time: **$(pretty_time "$(cut -d, -f5 <<<"$BEST_LATENCY")")**"

echo
echo "## 🧠 Lowest Memory"
echo "- **$(cut -d, -f1 <<<"$BEST_MEMORY")**"
echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_MEMORY")**"
echo "- RAM: **$(cut -d, -f6 <<<"$BEST_MEMORY")MB**"
} > "$REPORT_MD"

##################################################
# Done
##################################################

echo "✅ Benchmark completed"
echo "📄 CSV   : $MERGED_MODEL_CSV"
echo "📄 Report: $REPORT_MD"
