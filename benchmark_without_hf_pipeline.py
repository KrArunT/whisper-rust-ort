#!/usr/bin/env python3
"""
Whisper Optimum-ORT long-form transcription + benchmarking.

Why your transcript was partial:
- Whisper feature extractor defaults to chunk_length=30 seconds and truncation=True,
  so processor(...) only feeds first 30s unless you set truncation=False and padding="longest".
- For long-form, Transformers recommends return_attention_mask=True and generate(..., return_timestamps=True).
  (Long-form uses timestamp-based sequential decoding/heuristics.)

This script:
- Loads your exported ONNX files with explicit names:
  encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx
- Runs long-form transcription (FULL) on each audio:
  processor(..., truncation=False, padding="longest", return_attention_mask=True)
  model.generate(..., return_timestamps=True)
- Benchmarks:
  * model_only_s: time spent only inside model.generate
  * end_to_end_s: load+resample + preprocess + model + decode
  * also reports load_s, preprocess_s, decode_s
- Outputs CSV + summary JSON + optional per-file .txt transcripts
"""

import os
import time
import json
import csv
import argparse
import statistics
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf
import onnxruntime as ort
import torch

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor


# -------------------------
# Audio helpers
# -------------------------
def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    ratio = sr_out / sr_in
    n_out = int(round(len(x) * ratio))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def load_audio_16k_mono(path: str) -> Tuple[np.ndarray, int, float]:
    """
    Returns: (audio_float32_mono_16k, sr, duration_seconds)
    NOTE: soundfile may not decode mp3 unless libsndfile supports it in your image.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype("float32")
    if sr != 16000:
        audio = resample_linear(audio, sr, 16000).astype("float32")
        sr = 16000
    dur = float(len(audio)) / float(sr)
    return audio, sr, dur


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


# -------------------------
# ORT config helpers
# -------------------------
def suggested_optimum_cfg() -> Dict[str, Any]:
    cpu = os.cpu_count() or 8
    intra = min(cpu, 8)
    return {
        "intra_op": intra,
        "inter_op": 1,
        "execution_mode": "SEQUENTIAL",
        "graph_opt": "ENABLE_ALL",
        "cpu_mem_arena": True,
        "mem_pattern": True,
        "allow_spinning": True,
    }


def load_best_cfg_from_discovery(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    best = j.get("best", {}) or {}

    def _bool(x, default=False):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() in ("1", "true", "yes", "y", "on")
        if isinstance(x, (int, float)):
            return bool(x)
        return default

    return {
        "intra_op": int(best.get("intra_op", suggested_optimum_cfg()["intra_op"])),
        "inter_op": int(best.get("inter_op", 1)),
        "execution_mode": str(best.get("execution_mode", "SEQUENTIAL")),
        "graph_opt": str(best.get("graph_opt", "ENABLE_ALL")),
        "cpu_mem_arena": _bool(best.get("cpu_mem_arena", True), True),
        "mem_pattern": _bool(best.get("mem_pattern", True), True),
        "allow_spinning": _bool(best.get("allow_spinning", True), True),
    }


def make_session_options(cfg: Dict[str, Any]) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(cfg["intra_op"])
    so.inter_op_num_threads = int(cfg["inter_op"])

    exec_mode = str(cfg["execution_mode"]).upper()
    so.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
        if exec_mode == "SEQUENTIAL"
        else ort.ExecutionMode.ORT_PARALLEL
    )

    gopt = str(cfg["graph_opt"]).upper()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if gopt == "ENABLE_ALL"
        else ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    so.enable_cpu_mem_arena = bool(cfg["cpu_mem_arena"])
    so.enable_mem_pattern = bool(cfg["mem_pattern"])

    so.add_session_config_entry(
        "session.intra_op.allow_spinning", "1" if cfg["allow_spinning"] else "0"
    )
    so.add_session_config_entry(
        "session.inter_op.allow_spinning", "1" if cfg["allow_spinning"] else "0"
    )
    return so


def build_model(onnx_dir: str, model_id: str, session_options: ort.SessionOptions):
    """
    Explicit ONNX filenames for your export layout.
    """
    src = onnx_dir if os.path.isdir(onnx_dir) else model_id

    kwargs = dict(
        export=not os.path.isdir(onnx_dir),
        use_cache=True,
        provider="CPUExecutionProvider",
        session_options=session_options,
    )

    if os.path.isdir(onnx_dir):
        kwargs.update(
            encoder_file_name="encoder_model.onnx",
            decoder_file_name="decoder_model.onnx",
            decoder_with_past_file_name="decoder_with_past_model.onnx",
        )

    model = ORTModelForSpeechSeq2Seq.from_pretrained(src, **kwargs)

    # Best-effort: normalize any internal "device-like" members
    for attr in ("_device", "_execution_device", "execution_device"):
        if hasattr(model, attr):
            try:
                v = getattr(model, attr)
                if isinstance(v, str):
                    setattr(model, attr, torch.device(v))
            except Exception:
                pass

    return model


# -------------------------
# Long-form transcription (FULL)
# -------------------------
def generate_longform_full(
    model,
    processor,
    audio_16k: np.ndarray,
    language: str,
    task: str,
    max_new_tokens: int,
    forced_decoder_ids,
) -> Tuple[str, Dict[str, float]]:
    """
    Returns (full_text, timing_breakdown)

    Timing breakdown:
      load/preprocess handled outside
      preprocess_s: processor call
      model_only_s: model.generate only
      decode_s: batch_decode only
      end_to_end_s: preprocess + model + decode (audio load excluded here)
    """
    # Preprocess for long-form (FULL):
    # - truncation=False
    # - padding="longest"
    # - return_attention_mask=True
    t_pre0 = time.perf_counter()
    inputs = processor(
        audio_16k,
        sampling_rate=16000,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    # Ensure CPU float32
    inputs = inputs.to("cpu")
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=torch.float32)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
    preprocess_s = time.perf_counter() - t_pre0

    # Model-only time
    t_m0 = time.perf_counter()
    try:
        generated = model.generate(
            **inputs,
            return_timestamps=True,  # required for long-form path
            language=language,
            task=task,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    except TypeError:
        # Older stacks: fall back to forced_decoder_ids
        generated = model.generate(
            **inputs,
            return_timestamps=True,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    model_only_s = time.perf_counter() - t_m0

    # Decode time
    t_d0 = time.perf_counter()
    # generated may be tensor or dict depending on versions/flags.
    if isinstance(generated, dict) and "sequences" in generated:
        sequences = generated["sequences"]
    else:
        sequences = generated
    text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    decode_s = time.perf_counter() - t_d0

    end_to_end_s = preprocess_s + model_only_s + decode_s

    return text, {
        "preprocess_s": preprocess_s,
        "model_only_s": model_only_s,
        "decode_s": decode_s,
        "end_to_end_s": end_to_end_s,
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="audio")
    ap.add_argument("--model_id", default="openai/whisper-base")
    ap.add_argument("--onnx_dir", default="whisper-base-with-past")

    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")

    # IMPORTANT: max_new_tokens=256 can be too small depending on speech density;
    # long-form usually doesn’t need huge values, but you can raise it.
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--intra_op", type=int, default=0)
    ap.add_argument("--inter_op", type=int, default=0)
    ap.add_argument("--limit_files", type=int, default=0, help="0 = all files")

    ap.add_argument("--discovery_best_json", default="", help="Optional path to discovery_best.json")

    ap.add_argument("--out_csv", default="results/benchmarks/inference_per_file.csv")
    ap.add_argument("--out_json", default="results/benchmarks/inference_per_file.json")
    ap.add_argument("--out_summary_json", default="results/benchmarks/inference_summary.json")
    ap.add_argument("--write_txt", action="store_true", help="Also write transcript .txt per audio next to CSV dir")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary_json), exist_ok=True)

    cfg = load_best_cfg_from_discovery(args.discovery_best_json) if args.discovery_best_json else suggested_optimum_cfg()
    if args.intra_op and args.intra_op > 0:
        cfg["intra_op"] = int(args.intra_op)
    if args.inter_op and args.inter_op > 0:
        cfg["inter_op"] = int(args.inter_op)

    # Optional: keep torch thread count aligned (reduces variance)
    try:
        torch.set_num_threads(int(cfg["intra_op"]))
    except Exception:
        pass

    so = make_session_options(cfg)
    processor = AutoProcessor.from_pretrained(args.model_id)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model = build_model(args.onnx_dir, args.model_id, so)

    files = sorted(
        f for f in os.listdir(args.audio_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    )
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    # Warmup (optional): run on first file (small overhead stabilization)
    if args.warmup > 0:
        fn0 = files[0]
        path0 = os.path.join(args.audio_dir, fn0)
        a0, sr0, _ = load_audio_16k_mono(path0)
        if sr0 != 16000:
            raise RuntimeError("Unexpected sampling rate after resample")
        for _ in range(args.warmup):
            _ = generate_longform_full(
                model=model,
                processor=processor,
                audio_16k=a0,
                language=args.language,
                task=args.task,
                max_new_tokens=args.max_new_tokens,
                forced_decoder_ids=forced_decoder_ids,
            )

    # Per-file benchmark
    rows = []

    model_only_list = []
    end2end_list = []
    load_list = []
    preprocess_list = []
    decode_list = []
    rtf_model_list = []
    rtf_end2end_list = []

    txt_dir = os.path.dirname(args.out_csv)

    for fn in files:
        path = os.path.join(args.audio_dir, fn)

        # Load+resample time (part of end-to-end)
        t_l0 = time.perf_counter()
        audio, sr, dur = load_audio_16k_mono(path)
        load_s = time.perf_counter() - t_l0
        if sr != 16000:
            raise RuntimeError("Unexpected sampling rate after resample")

        # Model/pre/post
        text, t = generate_longform_full(
            model=model,
            processor=processor,
            audio_16k=audio,
            language=args.language,
            task=args.task,
            max_new_tokens=args.max_new_tokens,
            forced_decoder_ids=forced_decoder_ids,
        )

        model_only_s = t["model_only_s"]
        preprocess_s = t["preprocess_s"]
        decode_s = t["decode_s"]
        end_to_end_s = load_s + t["end_to_end_s"]  # include audio I/O + resample

        # RTFs
        rtf_model = model_only_s / max(dur, 1e-9)
        rtf_end2end = end_to_end_s / max(dur, 1e-9)

        rows.append({
            "file": fn,
            "duration_s": round(dur, 3),

            "end_to_end_s": round(end_to_end_s, 4),

            "rtf": round(rtf_end2end, 6),

            "text": text,
        })

        load_list.append(load_s)
        preprocess_list.append(preprocess_s)
        model_only_list.append(model_only_s)
        decode_list.append(decode_s)
        end2end_list.append(end_to_end_s)
        rtf_model_list.append(rtf_model)
        rtf_end2end_list.append(rtf_end2end)

        if args.write_txt:
            base = os.path.splitext(os.path.basename(fn))[0]
            out_txt = os.path.join(txt_dir, f"{base}.transcript.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(text.strip() + "\n")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "file", "duration_s",
            "end_to_end_s",
            "rtf",
            "text",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Write JSON (per file)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    def stat_block(xs: List[float]) -> Dict[str, float]:
        return {
            "min": round(min(xs), 6),
            "median": round(statistics.median(xs), 6),
            "p90": round(percentile(xs, 90), 6),
            "p95": round(percentile(xs, 95), 6),
            "max": round(max(xs), 6),
            "mean": round(statistics.mean(xs), 6),
        }

    summary = {
        "config_used": cfg,
        "n_files": len(rows),

        "latency_model_only_s": stat_block(model_only_list),
        "latency_end_to_end_s": stat_block(end2end_list),

        "breakdown_s": {
            "load_s": stat_block(load_list),
            "preprocess_s": stat_block(preprocess_list),
            "decode_s": stat_block(decode_list),
        },

        "rtf_model": stat_block(rtf_model_list),
        "rtf_end_to_end": stat_block(rtf_end2end_list),

        "model_id": args.model_id,
        "onnx_dir": args.onnx_dir,
        "language": args.language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "notes": {
            "longform": "processor(truncation=False, padding='longest', return_attention_mask=True) + generate(return_timestamps=True)",
        },
    }

    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("DONE")
    print("Config used:\n", json.dumps(cfg, indent=2))
    print("Per-file CSV:", args.out_csv)
    print("Per-file JSON:", args.out_json)
    print("Summary JSON:", args.out_summary_json)
    print("Model-only p95(s):", summary["latency_model_only_s"]["p95"], "| End-to-end p95(s):", summary["latency_end_to_end_s"]["p95"])


if __name__ == "__main__":
    main()
