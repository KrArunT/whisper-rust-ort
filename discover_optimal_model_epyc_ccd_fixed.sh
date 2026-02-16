#!/usr/bin/env bash
set -euo pipefail

##################################################
# CONFIG
##################################################

MODELS_ROOT="${MODELS_ROOT:-models/whisper-base-optimized}"
AUDIO_DIR="${AUDIO_DIR:-audio}"
RESULTS_ROOT="${RESULTS_ROOT:-results/benchmarks/without_hf_pipeline_rust}"

NUMA_NODE="${NUMA_NODE:-auto}"
PIN_STRATEGY="${PIN_STRATEGY:-ccd}"     # ccd | numa | flat
MAX_CORES_PER_RUN="${MAX_CORES_PER_RUN:-8}"

##################################################
# Helpers
##################################################

# Convert H:MM:SS / M:SS / SS → seconds
time_to_seconds() {
  awk -F: '{
    if (NF==3)      { printf "%.6f", ($1*3600)+($2*60)+$3 }
    else if (NF==2) { printf "%.6f", ($1*60)+$2 }
    else            { printf "%.6f", $1 }
  }' <<< "$1"
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
# CPU / NUMA detection
##################################################

CPU_MODEL="$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
IS_EPYC=0
grep -qi "epyc" <<<"$CPU_MODEL" && IS_EPYC=1

NUMA_NODES="$(lscpu | awk '/NUMA node\(s\)/ {print $3}')"
[[ "$NUMA_NODE" == "auto" ]] && NUMA_NODE=0

##################################################
# CPU feature detection
##################################################

CPU_FLAGS="$(lscpu | grep -i flags | tr 'A-Z' 'a-z')"
HAS_AVX2=$(grep -qw avx2 <<<"$CPU_FLAGS" && echo 1 || echo 0)
HAS_AVX512=$(grep -qw avx512f <<<"$CPU_FLAGS" && echo 1 || echo 0)
HAS_VNNI=$(grep -qw vnni <<<"$CPU_FLAGS" && echo 1 || echo 0)

##################################################
# Model compatibility
##################################################

is_model_compatible() {
  [[ "$1" == *avx512* && $HAS_AVX512 -ne 1 ]] && return 1
  [[ "$1" == *vnni*   && $HAS_VNNI   -ne 1 ]] && return 1
  [[ "$1" == *avx2*   && $HAS_AVX2   -ne 1 ]] && return 1
  return 0
}

##################################################
# NUMA + EPYC CCD-aware pinning (portable)
##################################################

TASKSET_CMD=()
NUMACTL_CMD=()

if command -v numactl >/dev/null; then
  NUMACTL_CMD=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
fi

if [[ "$PIN_STRATEGY" == "ccd" && $IS_EPYC -eq 1 ]]; then
  echo "🧠 Using EPYC CCD-aware pinning (core-group based)"

  CORE_LIST=$(lscpu -p=cpu,node,core | grep -v '^#' | \
    awk -F, -v node="$NUMA_NODE" '$2==node {print $1, $3}' | \
    sort -k2,2 -k1,1 | \
    head -n "$MAX_CORES_PER_RUN" | \
    awk '{print $1}' | paste -sd,)

  if [[ -z "$CORE_LIST" ]]; then
    echo "⚠️ CCD detection failed → falling back to NUMA-wide pinning"
    CORE_LIST=$(lscpu -p=cpu,node | grep -v '^#' | \
      awk -F, -v node="$NUMA_NODE" '$2==node {print $1}' | paste -sd,)
  fi

  TASKSET_CMD=(taskset -c "$CORE_LIST")

elif [[ "$PIN_STRATEGY" == "numa" ]]; then
  echo "📍 Using NUMA-wide pinning"

  CORE_LIST=$(lscpu -p=cpu,node | grep -v '^#' | \
    awk -F, -v node="$NUMA_NODE" '$2==node {print $1}' | paste -sd,)

  TASKSET_CMD=(taskset -c "$CORE_LIST")

else
  echo "📦 Using flat pinning"
  TASKSET_CMD=()
fi

##################################################
# CSV init (only executed models)
##################################################

echo "model,wall_time_sec,user_time,sys_time,peak_mem_mb" \
  > "$MERGED_MODEL_CSV"

##################################################
# Run benchmarks
##################################################

for onnx_dir in "$MODELS_ROOT"/*; do
  [[ -d "$onnx_dir" ]] || continue
  model_name="$(basename "$onnx_dir")"

  is_model_compatible "$model_name" || continue

  echo "🚀 Benchmarking $model_name"

  TIME_LOG="$TMP_DIR/time_${model_name}.txt"

  /usr/bin/time -v -o "$TIME_LOG" \
    "${NUMACTL_CMD[@]}" \
    "${TASKSET_CMD[@]}" \
    cargo run --release -- \
      --audio-dir "$AUDIO_DIR" \
      --onnx-dir "$onnx_dir" \
      --language en \
      --task transcribe \
      --max-new-tokens 128 \
      --intra-op 1 \
      --inter-op 1 \
      --chunk-parallelism 8 \
      --warmup 1 \
      --write-txt \
      --out-summary-json "$RESULTS_ROOT/inference_summary_${model_name}.json"

  PEAK_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
  PEAK_MB=$(awk "BEGIN { printf \"%.2f\", $PEAK_KB / 1024 }")

  RAW_WALL_TIME=$(grep "Elapsed (wall clock) time" "$TIME_LOG" | awk '{print $NF}')
  WALL_TIME_SEC=$(time_to_seconds "$RAW_WALL_TIME")

  USER_TIME=$(grep "User time (seconds)" "$TIME_LOG" | awk '{print $4}')
  SYS_TIME=$(grep "System time (seconds)" "$TIME_LOG" | awk '{print $4}')

  echo "$model_name,$WALL_TIME_SEC,$USER_TIME,$SYS_TIME,$PEAK_MB" \
    >> "$MERGED_MODEL_CSV"
done

##################################################
# Best models
##################################################

BEST_LATENCY_LINE=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k2 -n | head -n1)
BEST_MEMORY_LINE=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k5 -n | head -n1)

BEST_LATENCY_MODEL=$(cut -d, -f1 <<<"$BEST_LATENCY_LINE")
BEST_LATENCY_TIME=$(cut -d, -f2 <<<"$BEST_LATENCY_LINE")

BEST_MEMORY_MODEL=$(cut -d, -f1 <<<"$BEST_MEMORY_LINE")
BEST_MEMORY_MB=$(cut -d, -f5 <<<"$BEST_MEMORY_LINE")

##################################################
# Markdown report
##################################################

{
echo "# 🧪 Whisper ONNX Benchmark Report"
echo
echo "## 🖥 System"
echo "- CPU: $CPU_MODEL"
echo "- NUMA nodes: $NUMA_NODES"
echo "- NUMA node used: $NUMA_NODE"
echo "- Pinning strategy: $PIN_STRATEGY"
echo "- CPU cores pinned: ${TASKSET_CMD[*]}"
echo
echo "## 📊 Results"
echo "| Model | Wall Time (s) | User | Sys | Peak MB |"
echo "|------|---------------|------|-----|---------|"
tail -n +2 "$MERGED_MODEL_CSV" | \
  awk -F, '{printf "| %s | %s | %s | %s | %s |\n",$1,$2,$3,$4,$5}'
echo
echo "## 🏎 Lowest Latency Model"
echo "- **Model:** \`$BEST_LATENCY_MODEL\`"
echo "- **Wall time:** ${BEST_LATENCY_TIME}s"
echo
echo "## 🧠 Lowest Memory Model"
echo "- **Model:** \`$BEST_MEMORY_MODEL\`"
echo "- **Peak memory:** ${BEST_MEMORY_MB} MB"
echo
echo "## ✅ Deployment Guidance"
echo "- Use **$BEST_LATENCY_MODEL** for real-time inference"
echo "- Use **$BEST_MEMORY_MODEL** for memory-constrained environments"
} > "$REPORT_MD"

##################################################
# Done
##################################################

echo "🎉 Benchmark completed"
echo "📄 CSV: $MERGED_MODEL_CSV"
echo "📄 Report: $REPORT_MD"
