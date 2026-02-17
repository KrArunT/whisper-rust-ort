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
    if (t < 60) printf "%.0fs", t
    else printf "%dm%ds", int(t/60), int(t%60)
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

##################################################
# Setup
##################################################

mkdir -p "$RESULTS_ROOT"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MERGED_MODEL_CSV="$RESULTS_ROOT/merged_inference_results.csv"
REPORT_MD="$RESULTS_ROOT/BENCHMARK_REPORT.md"

BASELINE_DIR="$RESULTS_ROOT/baseline_whisper_${BASELINE_MODEL}"
mkdir -p "$BASELINE_DIR"
BASELINE_ALL_TXT="$BASELINE_DIR/baseline_all.txt"

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
    raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
PY

##################################################
# CSV init
##################################################

echo "implementation,precision,optimization,beam_size,time_sec,ram_mb,wer,cer" \
  > "$MERGED_MODEL_CSV"

##################################################
# Baseline generation (OpenAI Whisper base)
##################################################

echo "🎯 Generating baseline transcripts with OpenAI Whisper '${BASELINE_MODEL}'..."
uv run python3 "$PY_HELPER" baseline "$AUDIO_DIR" "$BASELINE_DIR" "$BASELINE_MODEL" "$BASELINE_LANG"
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

  echo "🚀 Benchmarking $model_name"

  TIME_LOG="$TMP_DIR/time_${model_name}.txt"
  HYP_CLEAN="$TMP_DIR/hyp_clean_${model_name}.txt"
  RUN_ERR="$TMP_DIR/run_stderr_${model_name}.txt"
  MODEL_TMP_DIR="$TMP_DIR/model_${model_name}"
  MODEL_CSV="$MODEL_TMP_DIR/inference_per_file.csv"
  MODEL_JSON="$MODEL_TMP_DIR/inference_per_file.json"
  MODEL_SUMMARY="$MODEL_TMP_DIR/inference_summary.json"
  mkdir -p "$MODEL_TMP_DIR"

  # Run Rust benchmark and persist per-file transcripts into CSV.
  /usr/bin/time -v -o "$TIME_LOG" \
    bash -c '
      set -euo pipefail
      cargo run --release -- \
        --audio-dir "'"$AUDIO_DIR"'" \
        --onnx-dir "'"$onnx_dir"'" \
        --language en \
        --task transcribe \
        --max-new-tokens 128 \
        --num-beams "'"$NUM_BEAMS"'" \
        --intra-op 1 \
        --inter-op 1 \
        --chunk-parallelism 8 \
        --warmup 1 \
        --out-csv "'"$MODEL_CSV"'" \
        --out-json "'"$MODEL_JSON"'" \
        --out-summary-json "'"$MODEL_SUMMARY"'"
    ' 1>/dev/null 2> "$RUN_ERR"

  RAW_TIME=$(grep "Elapsed (wall clock) time" "$TIME_LOG" | awk '{print $NF}')
  TIME_SEC=$(time_to_seconds "$RAW_TIME")

  PEAK_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $6}')
  PEAK_MB=$(awk "BEGIN { printf \"%.0f\", $PEAK_KB / 1024 }")

  PRECISION=$(precision_from_model "$model_name")
  OPT_LABEL=$(optimization_from_model "$model_name")
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

  echo "$IMPL,$PRECISION,$OPT_LABEL,$NUM_BEAMS,$TIME_SEC,$PEAK_MB,$WER,$CER" \
    >> "$MERGED_MODEL_CSV"
done

##################################################
# Best models (Latency / Memory / Accuracy)
##################################################

BEST_LATENCY=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k5 -n | head -n1)
BEST_MEMORY=$(tail -n +2 "$MERGED_MODEL_CSV" | sort -t, -k6 -n | head -n1)

# Best accuracy = lowest WER (ignore NA)
BEST_WER=$(tail -n +2 "$MERGED_MODEL_CSV" | awk -F, '$7!="NA"{print}' | sort -t, -k7 -n | head -n1)
BEST_CER=$(tail -n +2 "$MERGED_MODEL_CSV" | awk -F, '$8!="NA"{print}' | sort -t, -k8 -n | head -n1)

##################################################
# Markdown Report (comparison-table style)
##################################################

{
echo "# ⚡ Whisper ONNX Inference Benchmark"
echo
echo "**Baseline (accuracy reference):** OpenAI Whisper \`$BASELINE_MODEL\` via python \`whisper\` library"
echo
echo "| Implementation | Precision | Optimization | Beam size | Time | RAM Usage | WER | CER |"
echo "|---------------|-----------|--------------|-----------|------|-----------|-----|-----|"

tail -n +2 "$MERGED_MODEL_CSV" | \
while IFS=, read -r impl prec opt beam t ram wer cer; do
  printf "| %s | %s | %s | %s | %s | %sMB | %s | %s |\n" \
    "$impl" "$prec" "$opt" "$beam" "$(pretty_time "$t")" "$ram" "$(pretty_score "$wer")" "$(pretty_score "$cer")"
done

echo
echo "## 🏎 Lowest Latency"
echo "- **$(cut -d, -f1 <<<"$BEST_LATENCY")**"
echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_LATENCY")**"
echo "- Time: **$(pretty_time "$(cut -d, -f5 <<<"$BEST_LATENCY")")**"
echo "- WER/CER: **$(pretty_score "$(cut -d, -f7 <<<"$BEST_LATENCY")")** / **$(pretty_score "$(cut -d, -f8 <<<"$BEST_LATENCY")")**"

echo
echo "## 🧠 Lowest Memory"
echo "- **$(cut -d, -f1 <<<"$BEST_MEMORY")**"
echo "- Optimization: **$(cut -d, -f3 <<<"$BEST_MEMORY")**"
echo "- RAM: **$(cut -d, -f6 <<<"$BEST_MEMORY")MB**"
echo "- WER/CER: **$(pretty_score "$(cut -d, -f7 <<<"$BEST_MEMORY")")** / **$(pretty_score "$(cut -d, -f8 <<<"$BEST_MEMORY")")**"

echo
echo "## 🎯 Best Accuracy"
if [[ -n "${BEST_WER:-}" ]]; then
  echo "- Lowest WER Optimization: **$(cut -d, -f3 <<<"$BEST_WER")** (WER **$(pretty_score "$(cut -d, -f7 <<<"$BEST_WER")")**) "
else
  echo "- Lowest WER: **NA** (no valid WER computed)"
fi
if [[ -n "${BEST_CER:-}" ]]; then
  echo "- Lowest CER Optimization: **$(cut -d, -f3 <<<"$BEST_CER")** (CER **$(pretty_score "$(cut -d, -f8 <<<"$BEST_CER")")**) "
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
