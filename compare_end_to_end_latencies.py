#!/usr/bin/env python3
"""Compare end-to-end latency stats across HF pipeline, Python no-HF, and Rust no-HF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def stat_line(label: str, stat: Dict[str, Any]) -> str:
    return (
        f"{label:<26} min={stat.get('min', float('nan')):>8.4f} "
        f"med={stat.get('median', float('nan')):>8.4f} "
        f"p90={stat.get('p90', float('nan')):>8.4f} "
        f"p95={stat.get('p95', float('nan')):>8.4f} "
        f"max={stat.get('max', float('nan')):>8.4f} "
        f"mean={stat.get('mean', float('nan')):>8.4f}"
    )


def get_latency(summary: Dict[str, Any]) -> Dict[str, Any] | None:
    return summary.get("latency_end_to_end_s") or summary.get("latency_end_to_end")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-hf", default="results/benchmarks/with_hf_pipeline/inference_summary.json")
    ap.add_argument("--without-hf", default="results/benchmarks/without_hf_pipeline_py/inference_summary.json")
    ap.add_argument("--rust", default="results/benchmarks/without_hf_pipeline_rust/inference_summary.json")
    ap.add_argument("--faster", default="results/benchmarks/faster_whisper/inference_summary.json")
    ap.add_argument("--rust_int8", default="results/benchmarks/without_hf_pipeline_rust_int8/inference_summary.json")
    args = ap.parse_args()

    paths = {
        "HF pipeline": Path(args.with_hf),
        "Python (no HF pipeline)": Path(args.without_hf),
        "Rust (no HF pipeline)": Path(args.rust),
        "Faster-Whisper": Path(args.faster),
        "Rust (no HF, int8)": Path(args.rust_int8),
    }

    print("End-to-end latency comparison")
    print("-" * 80)

    for label, path in paths.items():
        if not path.exists():
            print(f"{label:<26} MISSING {path}")
            continue
        summary = read_json(path)
        stat = get_latency(summary)
        if not isinstance(stat, dict):
            print(f"{label:<26} missing latency_end_to_end_s in {path}")
            continue
        print(stat_line(label, stat))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
