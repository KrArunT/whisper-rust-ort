#!/usr/bin/env python3
"""Run Rust + Python Whisper ORT benchmarks and compare latencies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_nested(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def fmt_stat(name: str, stat: Dict[str, Any]) -> str:
    return (
        f"{name:<22} min={stat.get('min', float('nan')):>8.4f} "
        f"med={stat.get('median', float('nan')):>8.4f} "
        f"p90={stat.get('p90', float('nan')):>8.4f} "
        f"p95={stat.get('p95', float('nan')):>8.4f} "
        f"max={stat.get('max', float('nan')):>8.4f} "
        f"mean={stat.get('mean', float('nan')):>8.4f}"
    )


def compare_summaries(rust_summary: Dict[str, Any], py_summary: Dict[str, Any]) -> None:
    metrics = [
        ("latency_end_to_end_s", ["latency_end_to_end_s"]),
        ("rtf_end_to_end", ["rtf_end_to_end"]),
        ("load_s", ["breakdown_s", "load_s"]),
        ("preprocess_s", ["breakdown_s", "preprocess_s"]),
        ("model_only_s", ["breakdown_s", "model_only_s"]),
        ("decode_s", ["breakdown_s", "decode_s"]),
    ]

    print("\nSummary comparison")
    print("-" * 80)
    for label, path in metrics:
        r = get_nested(rust_summary, path)
        p = get_nested(py_summary, path)
        if isinstance(r, dict):
            print("RUST ", fmt_stat(label, r))
        else:
            print(f"RUST {label:<22} (missing)")
        if isinstance(p, dict):
            print("PY   ", fmt_stat(label, p))
        else:
            print(f"PY   {label:<22} (missing)")
        print("-")


def compare_per_file(rust_rows: List[Dict[str, Any]], py_rows: List[Dict[str, Any]]) -> None:
    rust_map = {r["file"]: r for r in rust_rows}
    py_map = {r["file"]: r for r in py_rows}
    common = sorted(set(rust_map) & set(py_map))
    if not common:
        print("No common files to compare.")
        return

    print("\nPer-file deltas (rust - py)")
    print("-" * 80)
    for fn in common:
        r = rust_map[fn]
        p = py_map[fn]
        dt = r.get("end_to_end_s", 0.0) - p.get("end_to_end_s", 0.0)
        dr = r.get("rtf", 0.0) - p.get("rtf", 0.0)
        print(f"{fn:<24} end_to_end_s={dt:+.4f}  rtf={dr:+.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--onnx-dir", default="whisper-base-with-past")
    ap.add_argument("--model-id", default="openai/whisper-base")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--tokenizer-json", default="")
    ap.add_argument("--timestamps", action="store_true")
    ap.add_argument("--write-txt", action="store_true")

    ap.add_argument("--rust-out-dir", default="results/benchmarks")
    ap.add_argument("--py-out-dir", default="results_py/benchmarks")

    ap.add_argument("--skip-rust", action="store_true")
    ap.add_argument("--skip-py", action="store_true")

    args = ap.parse_args()

    repo = Path(__file__).resolve().parent

    rust_out = repo / args.rust_out_dir
    py_out = repo / args.py_out_dir
    ensure_dir(rust_out)
    ensure_dir(py_out)

    rust_csv = rust_out / "inference_per_file.csv"
    rust_json = rust_out / "inference_per_file.json"
    rust_summary = rust_out / "inference_summary.json"

    py_csv = py_out / "inference_per_file.csv"
    py_json = py_out / "inference_per_file.json"
    py_summary = py_out / "inference_summary.json"

    if not args.skip_rust:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--",
            "--audio-dir",
            args.audio_dir,
            "--onnx-dir",
            args.onnx_dir,
            "--language",
            args.language,
            "--task",
            args.task,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--warmup",
            str(args.warmup),
            "--out-csv",
            str(rust_csv),
            "--out-json",
            str(rust_json),
            "--out-summary-json",
            str(rust_summary),
        ]
        if args.limit_files:
            cmd += ["--limit-files", str(args.limit_files)]
        if args.write_txt:
            cmd.append("--write-txt")
        if args.tokenizer_json:
            cmd += ["--tokenizer-json", args.tokenizer_json]
        if args.timestamps:
            cmd.append("--timestamps")
        run_cmd(cmd, repo)

    if not args.skip_py:
        cmd = [
            sys.executable,
            "infer_without_hf_pipeline.py",
            "--audio_dir",
            args.audio_dir,
            "--onnx_dir",
            args.onnx_dir,
            "--model_id",
            args.model_id,
            "--language",
            args.language,
            "--task",
            args.task,
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--warmup",
            str(args.warmup),
            "--out_csv",
            str(py_csv),
            "--out_json",
            str(py_json),
            "--out_summary_json",
            str(py_summary),
        ]
        if args.limit_files:
            cmd += ["--limit_files", str(args.limit_files)]
        if args.write_txt:
            cmd.append("--write_txt")
        run_cmd(cmd, repo)

    if not rust_summary.exists() or not py_summary.exists():
        print("Missing summary JSONs; skipping comparison.")
        return 1

    rust_sum = read_json(rust_summary)
    py_sum = read_json(py_summary)
    compare_summaries(rust_sum, py_sum)

    if rust_json.exists() and py_json.exists():
        rust_rows = read_json(rust_json)
        py_rows = read_json(py_json)
        if isinstance(rust_rows, list) and isinstance(py_rows, list):
            compare_per_file(rust_rows, py_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
