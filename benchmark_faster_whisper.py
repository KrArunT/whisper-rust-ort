#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import time
from typing import Any, Dict, List

import soundfile as sf
from faster_whisper import WhisperModel


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
    ap.add_argument("--model_id", default="Systran/faster-whisper-base")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--limit_files", type=int, default=0)
    ap.add_argument("--cpu_threads", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--compute_type", default="float32")

    ap.add_argument("--out_csv", default="results/benchmarks/faster_whisper/inference_per_file.csv")
    ap.add_argument("--out_json", default="results/benchmarks/faster_whisper/inference_per_file.json")
    ap.add_argument("--out_summary_json", default="results/benchmarks/faster_whisper/inference_summary.json")
    ap.add_argument("--write_txt", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary_json), exist_ok=True)

    model = WhisperModel(
        args.model_id,
        device="cpu",
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )

    files = sorted(
        f for f in os.listdir(args.audio_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    )
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No audio files found in {args.audio_dir}")

    if args.warmup > 0 and files:
        _ = list(
            model.transcribe(
                os.path.join(args.audio_dir, files[0]),
                language=args.language,
                task=args.task,
                beam_size=1,
                best_of=1,
                temperature=0.0,
            )[0]
        )

    rows = []
    end2end_list = []
    rtf_list = []

    for f in files:
        path = os.path.join(args.audio_dir, f)
        dur = audio_duration(path)

        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            path,
            language=args.language,
            task=args.task,
            beam_size=1,
            best_of=1,
            temperature=0.0,
        )
        text = "".join(seg.text for seg in segments).strip()
        latency = time.perf_counter() - t0

        rtf = latency / max(dur, 1e-9)

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
            out_txt = os.path.join(os.path.dirname(args.out_csv), f"{base}.transcript.txt")
            with open(out_txt, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "duration_s", "end_to_end_s", "rtf", "text"])
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    summary = {
        "config_used": {
            "cpu_threads": args.cpu_threads,
            "num_workers": args.num_workers,
            "compute_type": args.compute_type,
        },
        "n_files": len(rows),
        "latency_end_to_end_s": stat_block(end2end_list),
        "rtf_end_to_end": stat_block(rtf_list),
        "model_id": args.model_id,
        "language": args.language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "notes": {
            "engine": "faster-whisper (CTranslate2)",
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
