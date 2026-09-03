#!/usr/bin/env python3
"""Compare the first execution of each PRINT id in a run-kernel JSON to run-gdb."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


GDB_VALUE_RE = re.compile(r"^[^=\n]*=\s*(.*)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check run-kernel PRINT executions against run-gdb. "
            "Only the first run-kernel occurrence of each print id is used."
        )
    )
    parser.add_argument(
        "run_json",
        type=Path,
        help="JSON written by run_kernel.py",
    )
    parser.add_argument(
        "gdb_json",
        type=Path,
        help="JSON written by run-gdb.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="JSON array of {print_id, run_value, gdb_value, gdb_status, "
        "sizeof} for each print id exercised in run-kernel",
    )
    parser.add_argument(
        "--unexpected",
        type=Path,
        required=True,
        help="JSON array of print ids where run-gdb stopped (status printed) "
        "but run-kernel never executed that id",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def first_run_prints(run_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    first: dict[int, dict[str, Any]] = {}
    for rec in run_report.get("prints") or []:
        if "id" not in rec:
            raise SystemExit("error: run-kernel JSON print is missing id")
        print_id = int(rec["id"])
        if print_id in first:
            continue
        if "value" not in rec:
            raise SystemExit(f"error: run-kernel JSON print id {print_id} is missing value")
        first[print_id] = rec
    return first


def load_gdb_records(gdb_report: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(gdb_report, dict):
        raise SystemExit("error: run-gdb JSON must be an object keyed by print id")
    records: dict[int, dict[str, Any]] = {}
    for key, rec in gdb_report.items():
        print_id = int(rec.get("id", key))
        records[print_id] = rec
    return records


def gdb_value(gdb_print: Any) -> str | None:
    if gdb_print is None:
        return None
    text = str(gdb_print).strip()
    match = GDB_VALUE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_first = first_run_prints(load_json(args.run_json))
    gdb_records = load_gdb_records(load_json(args.gdb_json))

    exercised: list[dict[str, Any]] = []
    for print_id, run_rec in sorted(run_first.items()):
        row: dict[str, Any] = {
            "print_id": print_id,
            "run_value": run_rec["value"],
            "gdb_value": None,
            "gdb_status": None,
        }
        if "sizeof" in run_rec:
            row["sizeof"] = run_rec["sizeof"]
        gdb_rec = gdb_records.get(print_id)
        if gdb_rec is not None:
            row["gdb_value"] = gdb_value(gdb_rec.get("gdb_print"))
            row["gdb_status"] = gdb_rec.get("status")
        exercised.append(row)

    unexpected: list[int] = []
    for print_id, gdb_rec in sorted(gdb_records.items()):
        if gdb_rec.get("status") != "printed":
            continue
        if print_id not in run_first:
            unexpected.append(print_id)

    write_json(args.output, exercised)
    write_json(args.unexpected, unexpected)
    print(f"wrote {args.output} ({len(exercised)} exercised print ids)")
    print(f"wrote {args.unexpected} ({len(unexpected)} gdb-only stops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
