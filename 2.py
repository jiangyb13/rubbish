#!/usr/bin/env python3
"""Merge scattered one_shot_process JSONL files into one recovery JSONL."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge outputs/<video>/one_shot_process/output.jsonl files into one global JSONL."
    )
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory that contains per-video folders.",
    )
    parser.add_argument(
        "--pattern",
        default="*/one_shot_process/output.jsonl",
        help="Glob pattern under --root for one-shot JSONL files.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/one_shot_process_merged.jsonl",
        help="Merged output JSONL path.",
    )
    parser.add_argument(
        "--dedup-fields",
        default="source_shot_path,obj_id",
        help="Comma-separated fields used for deduplication.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return repo_root() / value


def iter_input_paths(root: Path, pattern: str, output_path: Path) -> Iterable[Path]:
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if path.resolve() == output_path.resolve():
            continue
        yield path


def record_key(record: dict, fields: Tuple[str, ...], raw_line: str):
    values = tuple(record.get(field) for field in fields)
    if any(value is not None for value in values):
        return values
    return ("line", raw_line)


def main() -> None:
    args = parse_args()
    root = resolve_path(args.root)
    output_path = resolve_path(args.output_jsonl)
    fields = tuple(field.strip() for field in args.dedup_fields.split(",") if field.strip())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    input_files = 0
    read_rows = 0
    written_rows = 0
    invalid_rows = 0
    duplicate_rows = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for input_path in iter_input_paths(root, args.pattern, output_path):
            input_files += 1
            with input_path.open("r", encoding="utf-8", errors="ignore") as fin:
                for line_number, line in enumerate(fin, 1):
                    raw = line.strip()
                    if not raw:
                        continue
                    read_rows += 1
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        invalid_rows += 1
                        continue
                    key = record_key(record, fields, raw)
                    if key in seen:
                        duplicate_rows += 1
                        continue
                    seen.add(key)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written_rows += 1

    print(json.dumps({
        "root": str(root),
        "pattern": args.pattern,
        "output_jsonl": str(output_path),
        "dedup_fields": fields,
        "input_files": input_files,
        "read_rows": read_rows,
        "written_rows": written_rows,
        "duplicate_rows": duplicate_rows,
        "invalid_rows": invalid_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
