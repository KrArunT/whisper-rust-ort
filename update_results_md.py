#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path


def ensure_header(contents: str) -> str:
    if contents.strip():
        return contents
    return "# Results\n\n"


def build_section(
    sut_name: str,
    core_count: int,
    memory_gb: int,
    summary_table: Path,
) -> str:
    table_text = summary_table.read_text(encoding="utf-8").strip()
    timestamp = datetime.now().isoformat(timespec="seconds")
    header = f"## {sut_name} - {core_count} cores / {memory_gb}GB RAM"
    lines = [
        header,
        f"Updated: {timestamp}",
        "",
        table_text,
        "",
    ]
    return "\n".join(lines)


def upsert_section(contents: str, marker_key: str, section_body: str) -> str:
    start = f"<!-- RESULTS:{marker_key} START -->"
    end = f"<!-- RESULTS:{marker_key} END -->"
    block = f"{start}\n{section_body}\n{end}"
    if start in contents and end in contents:
        pre, _ = contents.split(start, 1)
        _, post = contents.split(end, 1)
        return f"{pre}{block}{post}"
    return contents.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-md", default="RESULTS.md")
    ap.add_argument("--summary-table", required=True)
    ap.add_argument("--sut-name", default="default")
    ap.add_argument("--core-count", type=int, required=True)
    ap.add_argument("--memory-gb", type=int, required=True)
    args = ap.parse_args()

    results_path = Path(args.results_md)
    summary_table = Path(args.summary_table)
    if not summary_table.is_file():
        raise SystemExit(f"Missing summary table: {summary_table}")

    contents = ""
    if results_path.is_file():
        contents = results_path.read_text(encoding="utf-8")

    contents = ensure_header(contents)

    marker_key = f"{args.sut_name}:{args.core_count}c:{args.memory_gb}g"
    section_body = build_section(
        sut_name=args.sut_name,
        core_count=args.core_count,
        memory_gb=args.memory_gb,
        summary_table=summary_table,
    )
    updated = upsert_section(contents, marker_key, section_body)
    results_path.write_text(updated, encoding="utf-8")
    print(f"Updated {results_path}")


if __name__ == "__main__":
    main()
