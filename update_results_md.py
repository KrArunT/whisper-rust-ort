#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path


def ensure_header(contents: str) -> str:
    if contents.strip():
        return contents
    return "# Results\n\n"


def build_section_header(
    sut_name: str,
    core_count: int,
    memory_gb: int,
) -> str:
    return f"## {sut_name} - {core_count} cores / {memory_gb}GB RAM"


def build_run_entry(timestamp: str, summary_table: Path) -> str:
    table_text = summary_table.read_text(encoding="utf-8").strip()
    lines = [
        f"### Run {timestamp}",
        "",
        table_text,
        "",
    ]
    return "\n".join(lines)


def append_section(contents: str, marker_key: str, header: str, entry: str) -> str:
    start = f"<!-- RESULTS:{marker_key} START -->"
    end = f"<!-- RESULTS:{marker_key} END -->"
    if start in contents and end in contents:
        pre, rest = contents.split(start, 1)
        body, post = rest.split(end, 1)
        body = body.strip()
        if header not in body:
            body = f"{header}\n\n{entry}\n{body}".strip()
        else:
            body = f"{body}\n\n{entry}".strip()
        block = f"{start}\n{body}\n{end}"
        return f"{pre}{block}{post}"
    block = f"{start}\n{header}\n\n{entry}\n{end}"
    return contents.rstrip() + "\n\n" + block + "\n"


def append_results_csv(
    results_csv: Path,
    summary_csv: Path,
    timestamp: str,
    sut_name: str,
    core_count: int,
    memory_gb: int,
) -> None:
    if not summary_csv.is_file():
        print(f"Missing summary CSV: {summary_csv}")
        return
    rows = []
    with summary_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.is_file()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "timestamp",
            "sut_name",
            "core_count",
            "memory_gb",
            "implementation",
            "precision",
            "beam_size",
            "time_s",
            "ram_mb",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "sut_name": sut_name,
                    "core_count": core_count,
                    "memory_gb": memory_gb,
                    "implementation": row.get("implementation", ""),
                    "precision": row.get("precision", ""),
                    "beam_size": row.get("beam_size", ""),
                    "time_s": row.get("time_s", ""),
                    "ram_mb": row.get("ram_mb", ""),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-md", default="RESULTS.md")
    ap.add_argument("--summary-table", required=True)
    ap.add_argument("--summary-csv", default="")
    ap.add_argument("--sut-name", default="default")
    ap.add_argument("--core-count", type=int, required=True)
    ap.add_argument("--memory-gb", type=int, required=True)
    ap.add_argument("--results-csv", default="RESULTS.csv")
    args = ap.parse_args()

    results_path = Path(args.results_md)
    summary_table = Path(args.summary_table)
    if not summary_table.is_file():
        raise SystemExit(f"Missing summary table: {summary_table}")

    contents = ""
    if results_path.is_file():
        contents = results_path.read_text(encoding="utf-8")

    contents = ensure_header(contents)

    timestamp = datetime.now().isoformat(timespec="seconds")
    marker_key = f"{args.sut_name}:{args.core_count}c:{args.memory_gb}g"
    header = build_section_header(
        sut_name=args.sut_name,
        core_count=args.core_count,
        memory_gb=args.memory_gb,
    )
    entry = build_run_entry(timestamp, summary_table)
    updated = append_section(contents, marker_key, header, entry)
    results_path.write_text(updated, encoding="utf-8")
    print(f"Updated {results_path}")

    summary_csv = Path(args.summary_csv) if args.summary_csv else None
    if summary_csv:
        append_results_csv(
            results_csv=Path(args.results_csv),
            summary_csv=summary_csv,
            timestamp=timestamp,
            sut_name=args.sut_name,
            core_count=args.core_count,
            memory_gb=args.memory_gb,
        )


if __name__ == "__main__":
    main()
