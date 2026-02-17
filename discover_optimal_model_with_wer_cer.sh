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
ENABLE_TASKSET_PINNING="${ENABLE_TASKSET_PINNING:-1}"  # 1 | 0
PIN_CCDS="${PIN_CCDS:-}"        # e.g. "0" or "0,1" (L3/CCD ids)
PIN_CPUS="${PIN_CPUS:-}"        # explicit cpu list, e.g. "0-7,16-23" (overrides strategy)
CORE_LIST="${CORE_LIST:-}"      # alias for PIN_CPUS
CPUSET_LIST="${CPUSET_LIST:-}"  # alias for PIN_CPUS
NUM_BEAMS="${NUM_BEAMS:-1}"

# Backward-compatible aliases for explicit CPU lists.
if [[ -z "$PIN_CPUS" ]]; then
  if [[ -n "$CORE_LIST" ]]; then
    PIN_CPUS="$CORE_LIST"
  elif [[ -n "$CPUSET_LIST" ]]; then
    PIN_CPUS="$CPUSET_LIST"
  fi
fi

# Whisper baseline settings
BASELINE_MODEL="${BASELINE_MODEL:-base}"   # whisper model name: tiny/base/small/...
BASELINE_LANG="${BASELINE_LANG:-en}"

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
    if (t < 60) printf "%.2fs", t
    else printf "%dm%.2fs", int(t/60), (t - (int(t/60) * 60))
  }'
}

pretty_score() {
  # format float like 0.1234 -> 12.34%
  awk -v x="$1" 'BEGIN{
    if (x=="" || x=="NA") { printf "NA" }
    else { printf "%.2f%%", (x*100.0) }
  }'
}

precision_from_model() {
  [[ "$1" == *int8* ]] && echo "int8" || echo "fp32"
}

# ✅ Optimization column must keep o1/o2/... label from folder name
optimization_from_model() {
  local name="$1"
  if [[ "$name" =~ (o[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

implementation_name() {
  echo "onnxruntime rust"
}

instruction_set_from_model() {
  local name="$1"
  if [[ "$name" =~ _int8_([^_]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

backup_old_results() {
  local ts backup_dir copied=0
  ts="$(date +%Y%m%d_%H%M%S)"
  backup_dir="$RESULTS_ROOT/backups/$ts"
  mkdir -p "$backup_dir"

  if [[ -f "$MERGED_MODEL_CSV" ]]; then
    cp -a "$MERGED_MODEL_CSV" "$backup_dir/"
    copied=1
  fi
  if [[ -f "$REPORT_MD" ]]; then
    cp -a "$REPORT_MD" "$backup_dir/"
    copied=1
  fi
  if [[ -d "$PER_MODEL_RESULTS_DIR" ]]; then
    cp -a "$PER_MODEL_RESULTS_DIR" "$backup_dir/"
    copied=1
  fi
  if [[ -d "$BASELINE_DIR" ]]; then
    cp -a "$BASELINE_DIR" "$backup_dir/"
    copied=1
  fi

  if [[ "$copied" -eq 1 ]]; then
    echo "💾 Backed up previous benchmark results to: $backup_dir"
  else
    rmdir "$backup_dir" 2>/dev/null || true
  fi
}

expand_number_list() {
  local spec="${1//[[:space:]]/}"
  local part start end
  [[ -z "$spec" ]] && return 0
  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( start <= end )); then
        seq "$start" "$end"
      else
        seq "$end" "$start"
      fi
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      echo "$part"
    fi
  done
}

get_online_cpus() {
  if [[ -r /sys/devices/system/cpu/online ]]; then
    expand_number_list "$(cat /sys/devices/system/cpu/online)" | sort -n -u
  else
    seq 0 $(( $(nproc) - 1 ))
  fi
}

get_allowed_cpus() {
  local affinity
  if command -v taskset >/dev/null 2>&1; then
    affinity="$(taskset -pc $$ 2>/dev/null | sed -E 's/.*: *//')"
    if [[ -n "$affinity" ]]; then
      expand_number_list "$affinity" | sort -n -u
      return 0
    fi
  fi
  get_online_cpus
}

default_core_count() {
  local host_cores
  host_cores="$(nproc 2>/dev/null || echo 1)"
  if [[ "$MAX_CORES_PER_RUN" =~ ^[0-9]+$ ]] && (( MAX_CORES_PER_RUN > 0 && MAX_CORES_PER_RUN < host_cores )); then
    echo "$MAX_CORES_PER_RUN"
  else
    echo "$host_cores"
  fi
}

build_numa_cpu_candidates() {
  local out_file="$1"
  local lscpu_file="$TMP_DIR/lscpu_cpu_node.csv"
  local target_node="${NUMA_NODE}"

  if ! lscpu -p=CPU,NODE > "$lscpu_file" 2>/dev/null; then
    return 1
  fi

  if [[ "$target_node" == "auto" ]]; then
    target_node="$(awk -F, '$1 !~ /^#/ && $2 >= 0 {print $2; exit}' "$lscpu_file")"
  fi
  [[ -z "$target_node" ]] && return 1

  awk -F, -v n="$target_node" '$1 !~ /^#/ && $2 == n {print $1}' "$lscpu_file" | sort -n -u > "$out_file"
  [[ -s "$out_file" ]]
}

build_ccd_cpu_candidates() {
  local out_file="$1"
  local map_file="$TMP_DIR/cpu_l3_map.txt"
  local ccd_ids_file="$TMP_DIR/ccd_ids.txt"
  local selected_ccd_file="$TMP_DIR/selected_ccd_ids.txt"

  : > "$map_file"
  while IFS= read -r cpu; do
    local l3_file="/sys/devices/system/cpu/cpu${cpu}/cache/index3/id"
    if [[ -r "$l3_file" ]]; then
      local l3_id
      l3_id="$(cat "$l3_file" 2>/dev/null || true)"
      [[ -n "$l3_id" ]] && echo "$l3_id $cpu" >> "$map_file"
    fi
  done < <(get_online_cpus)

  [[ -s "$map_file" ]] || return 1

  awk '{print $1}' "$map_file" | sort -n -u > "$ccd_ids_file"
  [[ -s "$ccd_ids_file" ]] || return 1

  if [[ -n "$PIN_CCDS" ]]; then
    expand_number_list "$PIN_CCDS" | sort -n -u > "$selected_ccd_file"
  else
    cp "$ccd_ids_file" "$selected_ccd_file"
  fi
  [[ -s "$selected_ccd_file" ]] || return 1

  : > "$out_file"
  while IFS= read -r ccd_id; do
    awk -v c="$ccd_id" '$1 == c {print $2}' "$map_file" | sort -n -u >> "$out_file"
  done < "$selected_ccd_file"

  [[ -s "$out_file" ]]
}

filter_and_limit_cpus() {
  local online_file="$1"
  local candidates_file="$2"
  local out_file="$3"
  local max_cores="$4"

  awk -v max="$max_cores" '
    FNR==NR { online[$1]=1; next }
    online[$1] && !seen[$1] {
      print $1
      seen[$1]=1
      n++
      if (max > 0 && n >= max) exit
    }
  ' "$online_file" "$candidates_file" > "$out_file"
}

prepare_taskset_pinning() {
  local allowed_file="$TMP_DIR/allowed_cpus.txt"
  local candidates_file="$TMP_DIR/pin_candidates.txt"
  local selected_file="$TMP_DIR/pin_selected.txt"
  local strategy="${PIN_STRATEGY}"
  local max_cores="$MAX_CORES_PER_RUN"
  local pin_desc=""

  TASKSET_CPU_LIST=""
  PINNING_DESC="disabled"
  RUN_CORE_COUNT="$(default_core_count)"

  if [[ ! "$max_cores" =~ ^[0-9]+$ ]]; then
    max_cores=0
  fi

  if [[ "$ENABLE_TASKSET_PINNING" != "1" ]]; then
    echo "📌 CPU pinning disabled (ENABLE_TASKSET_PINNING=$ENABLE_TASKSET_PINNING)"
    return 0
  fi
  if ! command -v taskset >/dev/null 2>&1; then
    echo "⚠️  'taskset' not found; running without CPU pinning."
    return 0
  fi

  get_allowed_cpus > "$allowed_file"
  if [[ ! -s "$allowed_file" ]]; then
    echo "⚠️  Could not detect allowed CPUs; running without CPU pinning."
    return 0
  fi

  if [[ -n "$PIN_CPUS" ]]; then
    # Keep user order from explicit core list; de-dup/validity handled later.
    expand_number_list "$PIN_CPUS" > "$candidates_file"
    pin_desc="explicit(PIN_CPUS=$PIN_CPUS)"
  else
    case "$strategy" in
      ccd)
        if build_ccd_cpu_candidates "$candidates_file"; then
          if [[ -n "$PIN_CCDS" ]]; then
            pin_desc="ccd(PIN_CCDS=$PIN_CCDS)"
          else
            pin_desc="ccd(auto)"
          fi
        else
          echo "⚠️  Could not build CCD-aware CPU set; falling back to flat."
          cat "$allowed_file" > "$candidates_file"
          pin_desc="flat(fallback)"
        fi
        ;;
      numa)
        if build_numa_cpu_candidates "$candidates_file"; then
          pin_desc="numa(node=$NUMA_NODE)"
        else
          echo "⚠️  Could not build NUMA CPU set; falling back to flat."
          cat "$allowed_file" > "$candidates_file"
          pin_desc="flat(fallback)"
        fi
        ;;
      flat|*)
        cat "$allowed_file" > "$candidates_file"
        pin_desc="flat"
        ;;
    esac
  fi

  filter_and_limit_cpus "$allowed_file" "$candidates_file" "$selected_file" "$max_cores"
  if [[ ! -s "$selected_file" ]]; then
    echo "⚠️  Selected CPU set is empty; running without CPU pinning."
    return 0
  fi

  TASKSET_CPU_LIST="$(paste -sd, "$selected_file")"
  RUN_CORE_COUNT="$(wc -l < "$selected_file" | tr -d ' ')"
  if [[ ! "$RUN_CORE_COUNT" =~ ^[0-9]+$ ]] || (( RUN_CORE_COUNT <= 0 )); then
    RUN_CORE_COUNT=1
  fi
  PINNING_DESC="$pin_desc selected_cores=${RUN_CORE_COUNT} cpus=${TASKSET_CPU_LIST}"
  echo "📌 CPU pinning enabled: $PINNING_DESC"
}

##################################################
# Setup
##################################################

mkdir -p "$RESULTS_ROOT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MERGED_MODEL_CSV="$RESULTS_ROOT/merged_inference_results.csv"
REPORT_MD="$RESULTS_ROOT/BENCHMARK_REPORT.md"
PER_MODEL_RESULTS_DIR="$RESULTS_ROOT/per_model"

BASELINE_DIR="$RESULTS_ROOT/baseline_whisper_${BASELINE_MODEL}"
BASELINE_ALL_TXT="$BASELINE_DIR/baseline_all.txt"

# Preserve previous run outputs before creating new ones.
backup_old_results
mkdir -p "$BASELINE_DIR"
mkdir -p "$PER_MODEL_RESULTS_DIR"

# Build taskset pinning plan once for all model runs.
prepare_taskset_pinning

##################################################
# Python helpers (baseline + metrics)
##################################################

PY_HELPER="$TMP_DIR/metrics_helper.py"
cat > "$PY_HELPER" <<'PY'
import csv, os, re, sys, json
from pathlib import Path

AUDIO_EXTS = {".wav",".mp3",".m4a",".flac",".ogg",".webm",".aac",".wma",".opus"}

def list_audio(audio_dir: str):
    p = Path(audio_dir)
    files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in AUDIO_EXTS]
    files.sort(key=lambda x: str(x).lower())
    return files

def ensure_whisper():
    try:
        import whisper  # type: ignore
        return whisper
    except Exception as e:
        print("ERROR: python package 'whisper' not found (OpenAI Whisper).", file=sys.stderr)
        print("Install it, e.g.: pip install -U openai-whisper", file=sys.stderr)
        raise

def baseline_transcribe(audio_dir: str, out_dir: str, model_name: str, language: str):
    whisper = ensure_whisper()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    files = list_audio(audio_dir)
    if not files:
        raise SystemExit(f"No audio files found under: {audio_dir}")

    model = whisper.load_model(model_name)
    all_text_parts = []

    for f in files:
        # CPU-friendly defaults
        res = model.transcribe(
            str(f),
            language=language,
            task="transcribe",
            fp16=False,
            verbose=False
        )
        text = (res.get("text") or "").strip()
        # store per-file transcript
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f.name)
        (outp / f"{safe_name}.txt").write_text(text + "\n", encoding="utf-8")
        all_text_parts.append(text)

    (outp / "baseline_all.txt").write_text("\n".join(all_text_parts).strip() + "\n", encoding="utf-8")

def normalize_for_words(s: str) -> str:
    s = s.lower()
    # keep apostrophes inside words, drop most punctuation
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_for_chars(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", "", s)  # remove all whitespace for CER
    return s

def levenshtein(a, b) -> int:
    # a, b are sequences (list of tokens or string)
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    # DP with 2 rows
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev = cur
    return prev[m]

def wer_cer(ref_text: str, hyp_text: str):
    ref_w = normalize_for_words(ref_text).split()
    hyp_w = normalize_for_words(hyp_text).split()
    if len(ref_w) == 0:
        wer = 0.0 if len(hyp_w) == 0 else 1.0
    else:
        wer = levenshtein(ref_w, hyp_w) / float(len(ref_w))

    ref_c = list(normalize_for_chars(ref_text))
    hyp_c = list(normalize_for_chars(hyp_text))
    if len(ref_c) == 0:
        cer = 0.0 if len(hyp_c) == 0 else 1.0
    else:
        cer = levenshtein(ref_c, hyp_c) / float(len(ref_c))

    return wer, cer

def extract_text(raw: str) -> str:
    """
    Heuristic cleaner for Rust stdout logs:
    - drops common log prefixes (INFO/WARN/DEBUG/TRACE)
    - drops timing/benchmark lines
    - drops lines that look like timestamps
    - keeps remaining lines as transcript
    """
    lines = raw.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # drop common log prefixes
        if re.match(r"^(INFO|WARN|WARNING|DEBUG|TRACE)\b", s):
            continue
        # drop 'Elapsed'/'time'/'warmup' etc
        if re.search(r"\b(elapsed|wall clock|warmup|max resident|tokens\/s|rtf|throughput)\b", s, re.I):
            continue
        # drop timestamp-like prefixes: [00:00.000 --> 00:01.000] or 00:00:01
        if re.match(r"^\[?\d{1,2}:\d{2}(:\d{2})?(\.\d+)?", s):
            continue
        # drop "file:" headers if any (but keep the rest)
        s = re.sub(r"^(file|audio|input)\s*:\s*", "", s, flags=re.I)
        kept.append(s)
    return "\n".join(kept).strip()

def extract_text_from_csv(csv_path: str) -> str:
    """
    Read transcription text from Rust benchmark CSV.
    Supports common column names:
      - transcription
      - trascription (legacy typo)
      - text
    """
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [x.strip() for x in (reader.fieldnames or [])]
        if not fieldnames:
            raise SystemExit(f"CSV has no header: {csv_path}")

        candidates = ["transcription", "trascription", "text"]
        text_col = None
        for c in candidates:
            if c in fieldnames:
                text_col = c
                break
        if text_col is None:
            raise SystemExit(
                f"No transcription column found in {csv_path}. "
                f"Expected one of: {', '.join(candidates)}. "
                f"Found: {', '.join(fieldnames)}"
            )

        rows = list(reader)
        rows.sort(key=lambda r: (r.get("file") or "").lower())
        parts = []
        for row in rows:
            t = (row.get(text_col) or "").strip()
            parts.append(t)
        return "\n".join(parts).strip()

def average_end_to_end_latency(csv_path: str) -> float:
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            v = (row.get("end_to_end_s") or "").strip()
            if not v:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    if not vals:
        raise SystemExit(f"No valid end_to_end_s values in: {csv_path}")
    return sum(vals) / len(vals)

def main():
    cmd = sys.argv[1]
    if cmd == "baseline":
        audio_dir, out_dir, model_name, lang = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        baseline_transcribe(audio_dir, out_dir, model_name, lang)
        return
    if cmd == "metrics":
        ref_path, hyp_raw_path, hyp_clean_out = sys.argv[2], sys.argv[3], sys.argv[4]
        ref = Path(ref_path).read_text(encoding="utf-8", errors="ignore")
        hyp_raw = Path(hyp_raw_path).read_text(encoding="utf-8", errors="ignore")
        hyp = extract_text(hyp_raw)
        Path(hyp_clean_out).write_text(hyp + "\n", encoding="utf-8")
        wer, cer = wer_cer(ref, hyp)
        print(f"{wer:.6f},{cer:.6f}")
        return
    if cmd == "metrics_csv":
        ref_path, hyp_csv_path, hyp_clean_out = sys.argv[2], sys.argv[3], sys.argv[4]
        ref = Path(ref_path).read_text(encoding="utf-8", errors="ignore")
        hyp = extract_text_from_csv(hyp_csv_path)
        Path(hyp_clean_out).write_text(hyp + "\n", encoding="utf-8")
        wer, cer = wer_cer(ref, hyp)
        print(f"{wer:.6f},{cer:.6f}")
        return
    if cmd == "avg_latency_csv":
        csv_path = sys.argv[2]
        avg = average_end_to_end_latency(csv_path)
        print(f"{avg:.6f}")
        return
    raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
PY

##################################################
# CSV init
##################################################

echo "implementation,precision,optimization,instruction_set,beam_size,time_sec,ram_mb,wer,cer" \
  > "$MERGED_MODEL_CSV"

##################################################
# Baseline generation (OpenAI Whisper base)
##################################################

echo "🎯 Generating baseline transcripts with OpenAI Whisper '${BASELINE_MODEL}'..."
if [[ -n "$TASKSET_CPU_LIST" ]]; then
  taskset -c "$TASKSET_CPU_LIST" \
    uv run python3 "$PY_HELPER" baseline "$AUDIO_DIR" "$BASELINE_DIR" "$BASELINE_MODEL" "$BASELINE_LANG"
else
  uv run python3 "$PY_HELPER" baseline "$AUDIO_DIR" "$BASELINE_DIR" "$BASELINE_MODEL" "$BASELINE_LANG"
fi
# baseline_all.txt will exist inside BASELINE_DIR
if [[ ! -f "$BASELINE_ALL_TXT" ]]; then
  echo "❌ Baseline transcript missing: $BASELINE_ALL_TXT"
  exit 1
fi
echo "✅ Baseline ready: $BASELINE_ALL_TXT"

##################################################
# Benchmark loop
##################################################

for onnx_dir in "$MODELS_ROOT"/*; do
  [[ -d "$onnx_dir" ]] || continue
  model_name="$(basename "$onnx_dir")"

  if [[ -n "$TASKSET_CPU_LIST" ]]; then
    echo "🚀 Benchmarking $model_name (pinned: $TASKSET_CPU_LIST)"
  else
    echo "🚀 Benchmarking $model_name"
  fi

  TIME_LOG="$TMP_DIR/time_${model_name}.txt"
  HYP_CLEAN="$TMP_DIR/hyp_clean_${model_name}.txt"
  RUN_ERR="$TMP_DIR/run_stderr_${model_name}.txt"
  MODEL_TMP_DIR="$PER_MODEL_RESULTS_DIR/$model_name"
  MODEL_CSV="$MODEL_TMP_DIR/inference_per_file.csv"
  MODEL_JSON="$MODEL_TMP_DIR/inference_per_file.json"
  MODEL_SUMMARY="$MODEL_TMP_DIR/inference_summary.json"
  mkdir -p "$MODEL_TMP_DIR"

  # Run Rust benchmark and persist per-file transcripts into CSV.
  if [[ -n "$TASKSET_CPU_LIST" ]]; then
    /usr/bin/time -v -o "$TIME_LOG" \
      taskset -c "$TASKSET_CPU_LIST" \
      cargo run --release -- \
        --audio-dir "$AUDIO_DIR" \
        --onnx-dir "$onnx_dir" \
        --language en \
        --task transcribe \
        --max-new-tokens 128 \
        --num-beams "$NUM_BEAMS" \
        --intra-op "$RUN_CORE_COUNT" \
        --inter-op 1 \
        --chunk-parallelism "$RUN_CORE_COUNT" \
        --warmup 1 \
        --out-csv "$MODEL_CSV" \
        --out-json "$MODEL_JSON" \
        --out-summary-json "$MODEL_SUMMARY" \
      1>/dev/null 2> "$RUN_ERR"
  else
    /usr/bin/time -v -o "$TIME_LOG" \
      cargo run --release -- \
        --audio-dir "$AUDIO_DIR" \
        --onnx-dir "$onnx_dir" \
        --language en \
        --task transcribe \
        --max-new-tokens 128 \
        --num-beams "$NUM_BEAMS" \
        --intra-op "$RUN_CORE_COUNT" \
        --inter-op 1 \
        --chunk-parallelism "$RUN_CORE_COUNT" \
        --warmup 1 \
        --out-csv "$MODEL_CSV" \
        --out-json "$MODEL_JSON" \
        --out-summary-json "$MODEL_SUMMARY" \
      1>/dev/null 2> "$RUN_ERR"
  fi

  RAW_TIME=$(grep "Elapsed (wall clock) time" "$TIME_LOG" | awk '{print $NF}')
  WALL_TIME_SEC=$(time_to_seconds "$RAW_TIME")
  AVG_TIME_SEC="$(python3 "$PY_HELPER" avg_latency_csv "$MODEL_CSV" || true)"
  if [[ -n "${AVG_TIME_SEC:-}" ]]; then
    TIME_SEC="$AVG_TIME_SEC"
  else
    TIME_SEC="$WALL_TIME_SEC"
  fi

  PEAK_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
  PEAK_MB=$(awk "BEGIN { printf \"%.0f\", $PEAK_KB / 1024 }")

  PRECISION=$(precision_from_model "$model_name")
  OPT_LABEL=$(optimization_from_model "$model_name")
  ISA_LABEL=$(instruction_set_from_model "$model_name")
  IMPL=$(implementation_name)

  # Compute WER/CER vs baseline using Rust benchmark CSV transcription field.
  METRICS=$(python3 "$PY_HELPER" metrics_csv "$BASELINE_ALL_TXT" "$MODEL_CSV" "$HYP_CLEAN" || true)
  if [[ -z "${METRICS:-}" ]]; then
    WER="NA"
    CER="NA"
  else
    WER="$(cut -d, -f1 <<<"$METRICS")"
    CER="$(cut -d, -f2 <<<"$METRICS")"
  fi

  echo "$IMPL,$PRECISION,$OPT_LABEL,$ISA_LABEL,$NUM_BEAMS,$TIME_SEC,$PEAK_MB,$WER,$CER" \
    >> "$MERGED_MODEL_CSV"
done

##################################################
# Best models (Latency / Memory / Accuracy)
##################################################

BEST_LATENCY=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k6 -n | head -n1)
BEST_MEMORY=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k7 -n | head -n1)

# Best accuracy = lowest WER (ignore NA)
BEST_WER=$(tail -n +2 "$MERGED_MODEL_CSV" | awk -F, '$8!="NA"{print}' | sort -t, -k8 -n | head -n1)
BEST_CER=$(tail -n +2 "$MERGED_MODEL_CSV" | awk -F, '$9!="NA"{print}' | sort -t, -k9 -n | head -n1)

##################################################
# Markdown Report (comparison-table style)
##################################################

{
echo "# ⚡ Whisper ONNX Inference Benchmark"
echo
echo "**Baseline (accuracy reference):** OpenAI Whisper \`$BASELINE_MODEL\` via python \`whisper\` library"
echo "**CPU pinning:** \`$PINNING_DESC\`"
echo "**Time column:** average end-to-end latency per audio from \`inference_per_file.csv\`"
echo
echo "| Implementation | Precision | Optimization | Instruction Set | Beam size | Time | RAM Usage | WER | CER |"
echo "|---------------|-----------|--------------|-----------------|-----------|------|-----------|-----|-----|"

tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k2,2 -k3,3V -k4,4 | \
while IFS=, read -r impl prec opt isa beam t ram wer cer; do
  printf "| %s | %s | %s | %s | %s | %s | %sMB | %s | %s |\n" \
    "$impl" "$prec" "$opt" "$isa" "$beam" "$(pretty_time "$t")" "$ram" "$(pretty_score "$wer")" "$(pretty_score "$cer")"
done

echo
echo "## 🏎 Lowest Latency"
echo "- **$(cut -d, -f1 <<<"$BEST_LATENCY")**"
echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_LATENCY")**"
echo "- Instruction set: **$(cut -d, -f4 <<<"$BEST_LATENCY")**"
echo "- Time: **$(pretty_time "$(cut -d, -f6 <<<"$BEST_LATENCY")")**"
echo "- WER/CER: **$(pretty_score "$(cut -d, -f8 <<<"$BEST_LATENCY")")** / **$(pretty_score "$(cut -d, -f9 <<<"$BEST_LATENCY")")**"

echo
echo "## 🧠 Lowest Memory"
echo "- **$(cut -d, -f1 <<<"$BEST_MEMORY")**"
echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_MEMORY")**"
echo "- Instruction set: **$(cut -d, -f4 <<<"$BEST_MEMORY")**"
echo "- RAM: **$(cut -d, -f7 <<<"$BEST_MEMORY")MB**"
echo "- WER/CER: **$(pretty_score "$(cut -d, -f8 <<<"$BEST_MEMORY")")** / **$(pretty_score "$(cut -d, -f9 <<<"$BEST_MEMORY")")**"

echo
echo "## 🎯 Best Accuracy"
if [[ -n "${BEST_WER:-}" ]]; then
  echo "- Lowest WER Optimization: **$(cut -d, -f3 <<<"$BEST_WER")** on **$(cut -d, -f4 <<<"$BEST_WER")** (WER **$(pretty_score "$(cut -d, -f8 <<<"$BEST_WER")")**) "
else
  echo "- Lowest WER: **NA** (no valid WER computed)"
fi
if [[ -n "${BEST_CER:-}" ]]; then
  echo "- Lowest CER Optimization: **$(cut -d, -f3 <<<"$BEST_CER")** on **$(cut -d, -f4 <<<"$BEST_CER")** (CER **$(pretty_score "$(cut -d, -f9 <<<"$BEST_CER")")**) "
else
  echo "- Lowest CER: **NA** (no valid CER computed)"
fi
} > "$REPORT_MD"

##################################################
# Done
##################################################

echo "✅ Benchmark completed"
echo "📄 CSV   : $MERGED_MODEL_CSV"
echo "📄 Report: $REPORT_MD"
echo "📁 Baseline transcripts: $BASELINE_DIR"
echo "📌 Pinning: $PINNING_DESC"
