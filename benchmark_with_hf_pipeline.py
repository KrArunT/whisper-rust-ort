#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import time
from typing import Any, Dict, List

import onnxruntime as ort
import soundfile as sf
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline


def audio_duration(path: str) -> float:
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = int(k + 0.999999)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def stat_block(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "min": round(min(xs), 6),
        "median": round(statistics.median(xs), 6),
        "p90": round(percentile(xs, 90), 6),
        "p95": round(percentile(xs, 95), 6),
        "max": round(max(xs), 6),
        "mean": round(statistics.mean(xs), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="audio")
    ap.add_argument("--model_id", default="openai/whisper-base")
    ap.add_argument("--onnx_dir", default="whisper-base-with-past")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--intra_op", type=int, default=8)
    ap.add_argument("--inter_op", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--limit_files", type=int, default=0)

    ap.add_argument("--out_csv", default="results/benchmarks/with_hf_pipeline/inference_per_file.csv")
    ap.add_argument("--out_json", default="results/benchmarks/with_hf_pipeline/inference_per_file.json")
    ap.add_argument("--out_summary_json", default="results/benchmarks/with_hf_pipeline/inference_summary.json")
    ap.add_argument("--write_txt", action="store_true")
    args = ap.parse_args()

    # Thread control
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = int(args.intra_op)
    sess_opts.inter_op_num_threads = int(args.inter_op)

    # Model (export once, then load from ONNX dir)
    processor = AutoProcessor.from_pretrained(args.model_id)

    if not os.path.isdir(args.onnx_dir):
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.model_id,
            export=True,
            use_cache=True,
            provider="CPUExecutionProvider",
            session_options=sess_opts,
        )
        model.save_pretrained(args.onnx_dir)
    else:
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.onnx_dir,
            provider="CPUExecutionProvider",
            session_options=sess_opts,
        )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=(1, 1),
        generate_kwargs={"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams, "do_sample": False},
    )

    files = sorted(
        f for f in os.listdir(args.audio_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    )
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary_json), exist_ok=True)

    # Warmup
    if args.warmup > 0 and files:
        pipe(os.path.join(args.audio_dir, files[0]), return_timestamps=False)

    rows = []
    end2end_list = []
    rtf_list = []

    print("file,duration_s,end_to_end_s,rtf,text")

    for f in files:
        path = os.path.join(args.audio_dir, f)
        dur = audio_duration(path)

        t0 = time.perf_counter()
        result = pipe(path, return_timestamps=False)
        latency = time.perf_counter() - t0
        rtf = latency / max(dur, 1e-9)

        print(f"{f},{dur:.3f},{latency:.3f},{rtf:.3f}")

        text = result.get("text", "").strip()

        rows.append(
            {
                "file": f,
                "duration_s": round(dur, 3),
                "end_to_end_s": round(latency, 4),
                "rtf": round(rtf, 6),
                "text": text,
            }
        )

        end2end_list.append(latency)
        rtf_list.append(rtf)

        if args.write_txt:
            base = os.path.splitext(os.path.basename(f))[0]
            transcript_path = os.path.join(os.path.dirname(args.out_csv), f"{base}.transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")

    # Write CSV/JSON
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "duration_s", "end_to_end_s", "rtf", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    summary = {
        "config_used": {
            "ORT_INTRA_OP": int(args.intra_op),
            "ORT_INTER_OP": int(args.inter_op),
            "num_beams": args.num_beams,
        },
        "n_files": len(rows),
        "latency_end_to_end_s": stat_block(end2end_list),
        "rtf_end_to_end": stat_block(rtf_list),
        "model_id": args.model_id,
        "onnx_dir": args.onnx_dir,
        "language": args.language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "notes": {
            "pipeline": "HF pipeline with chunk_length_s=30, stride_length_s=(1,1)",
        },
    }

    with open(args.out_summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("DONE")
    print("Per-file CSV:", args.out_csv)
    print("Per-file JSON:", args.out_json)
    print("Summary JSON:", args.out_summary_json)
    print("End-to-end p95(s):", summary["latency_end_to_end_s"]["p95"])


if __name__ == "__main__":
    main()
