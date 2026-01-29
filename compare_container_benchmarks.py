#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_elapsed_time(raw: str) -> Optional[float]:
    raw = raw.strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        elif len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        else:
            hours = 0
            minutes = 0
            seconds = float(parts[0])
    except ValueError:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_time_log(path: Path) -> Tuple[Optional[float], Optional[int]]:
    elapsed_s = None
    max_rss_kb = None
    if not path.is_file():
        return None, None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Elapsed (wall clock) time" in line:
            _, value = line.split(":", 1)
            elapsed_s = parse_elapsed_time(value)
        elif "Maximum resident set size" in line:
            _, value = line.split(":", 1)
            try:
                max_rss_kb = int(value.strip())
            except ValueError:
                pass
    return elapsed_s, max_rss_kb


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_ram(max_rss_kb: Optional[int]) -> str:
    if max_rss_kb is None:
        return "n/a"
    mb = int(round(max_rss_kb / 1024.0))
    return f"{mb}MB"


def load_summary(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_beam_size(summary: Dict[str, object], fallback: int) -> int:
    config = summary.get("config_used", {}) if isinstance(summary, dict) else {}
    for key in ("num_beams", "beam_size"):
        value = config.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


def extract_precision(summary: Dict[str, object], fallback: str) -> str:
    config = summary.get("config_used", {}) if isinstance(summary, dict) else {}
    compute_type = config.get("compute_type")
    if isinstance(compute_type, str):
        lowered = compute_type.strip().lower()
        if lowered in ("float32", "fp32"):
            return "fp32"
        if lowered in ("int8", "qint8"):
            return "int8"
        return compute_type
    return fallback


def extract_end_to_end_p95(summary: Dict[str, object]) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    block = summary.get("latency_end_to_end_s")
    if not isinstance(block, dict):
        return None
    for key in ("p95", "p90", "median", "mean", "max", "min"):
        value = block.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/benchmarks/container_4c4g")
    ap.add_argument("--log-dir", default="results/benchmarks/container_4c4g/logs")
    ap.add_argument("--out-md", default="results/benchmarks/container_4c4g/summary_table.md")
    ap.add_argument("--out-csv", default="results/benchmarks/container_4c4g/summary_table.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    log_dir = Path(args.log_dir)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    benches = [
        {
            "label": "openai/whisper (HF pipeline)",
            "precision": "fp32",
            "beam_fallback": 1,
            "summary": results_dir / "with_hf_pipeline" / "inference_summary.json",
            "time_log": log_dir / "with_hf_pipeline.time.txt",
        },
        {
            "label": "onnxruntime python (no HF pipeline)",
            "precision": "fp32",
            "beam_fallback": 1,
            "summary": results_dir / "without_hf_pipeline_py" / "inference_summary.json",
            "time_log": log_dir / "without_hf_pipeline_py.time.txt",
        },
        {
            "label": "onnxruntime rust (no HF pipeline)",
            "precision": "fp32",
            "beam_fallback": 1,
            "summary": results_dir / "without_hf_pipeline_rust" / "inference_summary.json",
            "time_log": log_dir / "without_hf_pipeline_rust.time.txt",
        },
        {
            "label": "onnxruntime rust (int8)",
            "precision": "int8",
            "beam_fallback": 1,
            "summary": results_dir / "without_hf_pipeline_rust_int8" / "inference_summary.json",
            "time_log": log_dir / "without_hf_pipeline_rust_int8.time.txt",
        },
        {
            "label": "faster-whisper (fp32)",
            "precision": "float32",
            "beam_fallback": 1,
            "summary": results_dir / "faster_whisper_fp32" / "inference_summary.json",
            "time_log": log_dir / "faster_whisper_fp32.time.txt",
        },
        {
            "label": "faster-whisper (int8)",
            "precision": "int8",
            "beam_fallback": 1,
            "summary": results_dir / "faster_whisper_int8" / "inference_summary.json",
            "time_log": log_dir / "faster_whisper_int8.time.txt",
        },
    ]

    rows = []
    for bench in benches:
        summary = load_summary(bench["summary"])
        elapsed_s, max_rss_kb = parse_time_log(bench["time_log"])
        end_to_end_s = extract_end_to_end_p95(summary)
        beam_size = extract_beam_size(summary, bench["beam_fallback"])
        precision = extract_precision(summary, bench["precision"])

        time_s = end_to_end_s if end_to_end_s is not None else elapsed_s
        rows.append(
            {
                "implementation": bench["label"],
                "precision": precision,
                "beam_size": beam_size,
                "time_s": None if time_s is None else round(time_s, 3),
                "time": format_duration(time_s),
                "ram_mb": None if max_rss_kb is None else int(round(max_rss_kb / 1024.0)),
                "ram": format_ram(max_rss_kb),
            }
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("| Implementation | Precision | Beam size | Time | RAM Usage |\n")
        handle.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['implementation']} | {row['precision']} | {row['beam_size']} | {row['time']} | {row['ram']} |\n"
            )

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["implementation", "precision", "beam_size", "time_s", "ram_mb"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "implementation": row["implementation"],
                    "precision": row["precision"],
                    "beam_size": row["beam_size"],
                    "time_s": row["time_s"],
                    "ram_mb": row["ram_mb"],
                }
            )

    print("Wrote summary table:", out_md)
    print("Wrote summary csv:", out_csv)


if __name__ == "__main__":
    main()
