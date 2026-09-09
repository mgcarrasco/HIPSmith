#!/usr/bin/env python3
"""Run a compiled HIPSmith kernel under timeout and write a JSON report."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PRINT_RE = re.compile(
    r"^\[line (?P<line>-?\d+)\]"
    r"(?: tid\((?P<tx>\d+),(?P<ty>\d+),(?P<tz>\d+)\))?"
    r" (?P<expr>.+?) = (?P<value>-?\d+)"
    r"(?: how='(?P<how>.*)')?"
    r"(?: id=(?P<id>-?\d+))?"
    r"(?: sizeof=(?P<sizeof>\d+))?"
    r"\s*$"
)
CRC_RE = re.compile(r"^Thread (\d+) CRC: (\d+)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a HIPSmith kernel with timeout -s9 and write PRINT lines, "
            "CRC, wall time, and exit code to JSON. Requires a single CRC."
        ),
        epilog="Kernel arguments go after -- .",
    )
    parser.add_argument(
        "kernel",
        type=Path,
        help="Compiled HIPSmith kernel binary",
    )
    parser.add_argument(
        "--timeout",
        required=True,
        metavar="DURATION",
        help="Duration passed to timeout(1), e.g. 30 or 30s",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON report path (default: <kernel>.json next to the binary)",
    )
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        extra = argv[idx + 1 :]
        argv = argv[:idx]
    args = parser.parse_args(argv)
    args.extra = extra
    return args


def parse_prints(text: str) -> list[dict[str, Any]]:
    prints: list[dict[str, Any]] = []
    for raw in text.splitlines():
        match = PRINT_RE.match(raw)
        if not match:
            continue
        record: dict[str, Any] = {
            "line": int(match.group("line")),
            "expr": match.group("expr"),
            "value": match.group("value"),
        }
        if match.group("tx") is not None:
            record["tid"] = [
                int(match.group("tx")),
                int(match.group("ty")),
                int(match.group("tz")),
            ]
        if match.group("how") is not None:
            record["how"] = match.group("how")
        if match.group("id") is not None:
            record["id"] = int(match.group("id"))
        if match.group("sizeof") is not None:
            record["sizeof"] = int(match.group("sizeof"))
        prints.append(record)
    return prints


def parse_crcs(text: str) -> list[tuple[int, str]]:
    return [
        (int(match.group(1)), match.group(2))
        for match in CRC_RE.finditer(text)
    ]


def multi_thread_reason(
    prints: list[dict[str, Any]], crcs: list[tuple[int, str]]
) -> str | None:
    tids = {tuple(p["tid"]) for p in prints if "tid" in p}
    if len(tids) > 1:
        shown = ", ".join(str(list(t)) for t in sorted(tids))
        return f"PRINT output used multiple thread ids: {shown}"
    if len(crcs) > 1:
        thread_ids = [tid for tid, _ in crcs]
        return f"kernel reported {len(crcs)} CRC lines: {thread_ids}"
    return None


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    kernel = args.kernel.resolve()
    if not kernel.is_file():
        print(f"error: no such file: {kernel}", file=sys.stderr)
        return 1
    if not os.access(kernel, os.X_OK):
        print(f"error: not executable: {kernel}", file=sys.stderr)
        return 1

    extra = list(args.extra)

    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        print("error: timeout(1) not found on PATH", file=sys.stderr)
        return 1

    output = args.output
    if output is None:
        output = kernel.with_suffix(".json")
    output = output.resolve()

    cmd = [timeout_bin, "-s9", args.timeout, str(kernel), *extra]
    print(" ".join(cmd))

    started = time.perf_counter()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    elapsed = time.perf_counter() - started

    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    combined = result.stdout + result.stderr

    prints = parse_prints(combined)
    crcs = parse_crcs(combined)
    reason = multi_thread_reason(prints, crcs)

    report: dict[str, Any] = {
        "prints": prints,
        "crc": crcs[0][1] if len(crcs) == 1 else None,
        "total_time": elapsed,
        "exit_code": result.returncode,
    }
    write_report(output, report)

    if reason is not None:
        print(f"error: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
