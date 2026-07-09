#!/usr/bin/env python3
"""Sort a TSV word list by Relative Frequency and write a sorted copy.

This script reads the source file as tab-separated values, preserves all original
columns, and writes a new file with rows ordered by the "Relative Frequency"
column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


REQUIRED_COLUMN = "Relative Frequency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sorted copy of the word list by Relative Frequency."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default="items-words.txt",
        help="Path to input TSV file (default: items-words.txt)",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="data/items-words.sorted-by-relative-frequency.tsv",
        help=(
            "Path to output file (default: "
            "data/items-words.sorted-by-relative-frequency.tsv)"
        ),
    )
    parser.add_argument(
        "--order",
        choices=["desc", "asc"],
        default="desc",
        help="Sort direction: desc for highest frequency first, asc for lowest first.",
    )
    return parser.parse_args()


def parse_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        raise ValueError("empty Relative Frequency")
    return float(text)


def read_rows(input_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if REQUIRED_COLUMN not in fieldnames:
            raise ValueError(
                f"Missing required column '{REQUIRED_COLUMN}' in {input_path}"
            )

        rows: list[dict[str, Any]] = []
        for line_no, row in enumerate(reader, start=2):
            freq_text = row.get(REQUIRED_COLUMN, "")
            try:
                parsed = parse_float(freq_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {REQUIRED_COLUMN} at line {line_no}: {freq_text!r}"
                ) from exc

            row["_relative_frequency_numeric"] = parsed
            rows.append(row)

    return rows, fieldnames


def write_rows(output_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            row.pop("_relative_frequency_numeric", None)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows, fieldnames = read_rows(input_path)
    reverse = args.order == "desc"
    rows.sort(key=lambda r: r["_relative_frequency_numeric"], reverse=reverse)

    write_rows(output_path, fieldnames, rows)

    print(f"Input rows: {len(rows)}")
    print(f"Sort order: {args.order}")
    print(f"Wrote sorted file: {output_path}")


if __name__ == "__main__":
    main()
