#!/usr/bin/env python3
"""Interestingness test: a reduced crosscheck.json still shows the same divergences.

Exit 0 if CURRENT is interesting relative to INITIAL, 1 if not, 2 on usage error.

By default, concrete run vs gdb values are compared as bit-patterns of width
8 * sizeof (from the PRINT kernel), so signedness-only differences match.
Use --no-bit-equality for exact integer equality instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


CHAR_SUFFIX_RE = re.compile(r"""^(.*\S)\s+'.*'$""")
SIZEOF_BYTES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether a reduced run-vs-gdb crosscheck is still interesting. "
            "Exit 0 if yes, 1 if not."
        )
    )
    parser.add_argument(
        "initial",
        type=Path,
        help="Original crosscheck JSON (from crosscheck-run-vs-gdb.py -o)",
    )
    parser.add_argument(
        "current",
        type=Path,
        help="Crosscheck JSON of this reduction attempt",
    )
    parser.add_argument(
        "--ids",
        metavar="ID[,ID...]",
        help="Targeted print ids (default: every print_id in INITIAL)",
    )
    parser.add_argument(
        "--no-bit-equality",
        action="store_true",
        help=(
            "Compare concrete values with exact integer equality instead of "
            "masking to 8 * sizeof bits"
        ),
    )
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        die(f"no such file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        die(f"{path} is not a JSON array")
    rows: dict[int, dict[str, Any]] = {}
    for rec in payload:
        if "print_id" not in rec:
            die(f"{path} has a row with no print_id")
        rows[int(rec["print_id"])] = rec
    return rows


def parse_ids(text: str | None, initial_ids: list[int]) -> list[int]:
    if text is None:
        return initial_ids
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part, 10))
    if not ids:
        die("--ids is empty")
    unknown = [i for i in ids if i not in set(initial_ids)]
    if unknown:
        die(f"--ids not present in INITIAL: {unknown}")
    return ids


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = CHAR_SUFFIX_RE.match(text)
    if match:
        text = match.group(1).strip()
    try:
        return int(text, 0)
    except ValueError:
        return None


def is_concrete(value: Any) -> bool:
    return parse_int(value) is not None


def row_sizeof(rec: dict[str, Any], print_id: int, which: str) -> int:
    if "sizeof" not in rec:
        die(f"id {print_id}: {which} row has no sizeof (needed for bit-equality)")
    try:
        size = int(rec["sizeof"])
    except (TypeError, ValueError):
        die(f"id {print_id}: {which} sizeof {rec['sizeof']!r} is not an integer")
    if size not in SIZEOF_BYTES:
        die(f"id {print_id}: {which} sizeof {size} is not 1, 2, 4, or 8")
    return size


def values_match(
    run_value: Any,
    gdb_value: Any,
    rec: dict[str, Any],
    print_id: int,
    which: str,
    bit_equality: bool,
) -> bool:
    run_n = parse_int(run_value)
    gdb_n = parse_int(gdb_value)
    if run_n is None or gdb_n is None:
        return False
    if not bit_equality:
        return run_n == gdb_n
    mask = (1 << (row_sizeof(rec, print_id, which) * 8)) - 1
    return (run_n & mask) == (gdb_n & mask)


def same_gdb_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return str(left).strip() == str(right).strip()


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def fail(message: str) -> None:
    print(f"not interesting: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_id(
    print_id: int,
    initial: dict[str, Any],
    current: dict[str, Any],
    bit_equality: bool,
) -> None:
    if initial.get("gdb_status") != current.get("gdb_status"):
        fail(
            f"id {print_id}: gdb_status {initial.get('gdb_status')!r} -> "
            f"{current.get('gdb_status')!r}"
        )

    initial_match = values_match(
        initial.get("run_value"),
        initial.get("gdb_value"),
        initial,
        print_id,
        "INITIAL",
        bit_equality,
    )
    current_match = values_match(
        current.get("run_value"),
        current.get("gdb_value"),
        current,
        print_id,
        "CURRENT",
        bit_equality,
    )

    if initial_match:
        if not current_match:
            fail(f"id {print_id}: initial matched, current does not")
        return

    if is_concrete(initial.get("gdb_value")):
        if not is_concrete(current.get("gdb_value")):
            fail(f"id {print_id}: initial gdb_value was concrete, current is not")
        if current_match:
            fail(f"id {print_id}: initial mismatched (concrete gdb), current matches")
        return

    if not same_gdb_value(initial.get("gdb_value"), current.get("gdb_value")):
        fail(
            f"id {print_id}: non-concrete gdb_value "
            f"{initial.get('gdb_value')!r} -> {current.get('gdb_value')!r}"
        )


def main() -> int:
    args = parse_args()
    initial_rows = load_rows(args.initial)
    current_rows = load_rows(args.current)
    targeted = parse_ids(args.ids, sorted(initial_rows))
    bit_equality = not args.no_bit_equality

    for print_id in targeted:
        if print_id not in current_rows:
            fail(f"id {print_id} is in INITIAL but missing from CURRENT")
        check_id(
            print_id,
            initial_rows[print_id],
            current_rows[print_id],
            bit_equality,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
