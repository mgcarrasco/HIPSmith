#!/usr/bin/env python3
"""Stop ROCgdb once per PRINT line and record `print` of the macro expression.

Breakpoints are set on PRINT source lines. On each stop the PC's source line is
checked; a mismatch is treated as a slide (status no_line_debug_info), the
breakpoint is dropped, and execution continues so later sites are not polluted.

gdb_print is always the first file:line stop. If that print is not a concrete
integer, one-shot breakpoints are planted on later line-table PCs of the same
source line; the first concrete print among those hits sets located_on_line.
A stop at a different PRINT line is not an inner PC of this line. If the line
table cannot be enumerated, located_on_line stays unset (fail closed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from print_sites import parse_print_sites

GDB_HELPER = r'''
import gdb
import json
import os
import re
from pathlib import Path

SITES_PATH = os.environ["PROBE_SITES_JSON"]
RESULT_PATH = os.environ["PROBE_RESULT_JSON"]
HIP_BASENAME = os.environ["PROBE_HIP_BASENAME"]

CHAR_SUFFIX_RE = re.compile(r"^(.*\S)\s+'.*'$")
GDB_VALUE_RE = re.compile(r"^[^=\n]*=\s*(.*)$", re.DOTALL)

# line -> extra *pc breakpoints planted for a walk of that PRINT line
SCAN_BREAKPOINTS = {}


def run_capture(cmd):
    try:
        return gdb.execute(cmd, to_string=True)
    except gdb.error as exc:
        return str(exc)


def current_sal():
    try:
        frame = gdb.selected_frame()
        sal = frame.find_sal()
    except gdb.error:
        return None, None
    filename = None
    if sal is not None and sal.symtab is not None:
        filename = sal.symtab.filename
    line = sal.line if sal is not None else None
    return filename, line


def current_pc():
    try:
        return int(gdb.selected_frame().pc())
    except (gdb.error, TypeError, ValueError):
        return None


def gdb_value(gdb_print):
    if gdb_print is None:
        return None
    text = str(gdb_print).strip()
    match = GDB_VALUE_RE.match(text)
    return match.group(1).strip() if match else text


def parse_int(value):
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


def is_concrete_print(gdb_print):
    return parse_int(gdb_value(gdb_print)) is not None


def line_pcs(line):
    """Runtime PCs whose line-table row is this source line, or None if unknown."""
    try:
        frame = gdb.selected_frame()
        sal = frame.find_sal()
        if sal is None or sal.symtab is None:
            return None
        lt = sal.symtab.linetable()
        if lt is None:
            return None
        pcs = []
        # Walk the whole table. LineTable.line() is only the first contiguous
        # range, which misses later fragments of the same source line.
        for entry in lt:
            try:
                if int(entry.line) == int(line):
                    pcs.append(int(entry.pc))
            except (TypeError, ValueError, AttributeError):
                continue
        pcs = sorted(set(pcs))
        return pcs if pcs else None
    except (gdb.error, TypeError, ValueError, AttributeError):
        return None


def dump_results():
    Path(RESULT_PATH).write_text(json.dumps(RESULTS, indent=2) + "\n")


def mark_scan_unavailable(sites):
    for site in sites:
        rec = RESULTS[str(site["id"])]
        if rec.get("located_on_line") is False:
            rec["located_on_line"] = None


def drop_scan_breakpoints(line):
    for bp in SCAN_BREAKPOINTS.pop(line, []):
        try:
            bp.enabled = False
        except (gdb.error, RuntimeError):
            pass


def finish_scan_if_done(line, sites):
    if any(RESULTS[str(site["id"])].get("located_on_line") is False for site in sites):
        return
    drop_scan_breakpoints(line)


class LineScanBreakpoint(gdb.Breakpoint):
    """One later line-table PC of a PRINT line. Not a first stop of any line."""

    def __init__(self, pc, expected_line, sites):
        super().__init__(
            "*0x%x" % pc, gdb.BP_BREAKPOINT, internal=True, temporary=True)
        self.planted_pc = int(pc)
        self.expected_line = expected_line
        self.sites = sites

    def stop(self):
        filename, actual_line = current_sal()
        base = Path(filename).name if filename else ""
        pc = current_pc()
        if (
            base != HIP_BASENAME
            or actual_line != self.expected_line
            or pc != self.planted_pc
        ):
            return False
        for site in self.sites:
            rec = RESULTS[str(site["id"])]
            if rec.get("located_on_line") is not False:
                continue
            expr = site["how"] or site["expr"]
            printed = run_capture("print " + expr)
            if is_concrete_print(printed):
                rec["located_on_line"] = True
        finish_scan_if_done(self.expected_line, self.sites)
        return False


def plant_line_scan(line, sites, stop_pc):
    pcs = line_pcs(line)
    if stop_pc is None or pcs is None or stop_pc < pcs[0] or stop_pc > pcs[-1]:
        mark_scan_unavailable(sites)
        gdb.write("line-scan %s:%d unavailable\n" % (HIP_BASENAME, line))
        return
    later = [pc for pc in pcs if pc > stop_pc]
    if not later:
        return
    planted = []
    for pc in later:
        try:
            planted.append(LineScanBreakpoint(pc, line, sites))
        except (gdb.error, RuntimeError, TypeError, ValueError):
            continue
    if not planted:
        mark_scan_unavailable(sites)
        gdb.write("line-scan %s:%d unavailable (no breakpoints)\n" % (
            HIP_BASENAME, line))
        return
    SCAN_BREAKPOINTS[line] = planted
    gdb.write("line-scan %s:%d stop=0x%x later=%d planted=%d\n" % (
        HIP_BASENAME, line, stop_pc, len(later), len(planted)))


class PrintLineBreakpoint(gdb.Breakpoint):
    def __init__(self, line, sites):
        super().__init__("%s:%d" % (HIP_BASENAME, line))
        self.expected_line = line
        self.sites = sites

    def stop(self):
        filename, actual_line = current_sal()
        base = Path(filename).name if filename else ""
        if base != HIP_BASENAME or actual_line != self.expected_line:
            for site in self.sites:
                rec = RESULTS[str(site["id"])]
                rec["status"] = "no_line_debug_info"
                rec["gdb_print"] = None
                rec["stopped_line"] = actual_line
                rec["stopped_file"] = filename
            self.enabled = False
            return False
        needs_scan = False
        for site in self.sites:
            expr = site["how"] or site["expr"]
            printed = run_capture("print " + expr)
            rec = RESULTS[str(site["id"])]
            rec["status"] = "printed"
            rec["gdb_print"] = printed.rstrip("\n")
            rec["stopped_line"] = actual_line
            rec["stopped_file"] = filename
            if is_concrete_print(printed):
                rec["located_on_line"] = None
            else:
                rec["located_on_line"] = False
                needs_scan = True
        self.enabled = False
        if needs_scan:
            plant_line_scan(self.expected_line, self.sites, current_pc())
        return False


sites = json.loads(Path(SITES_PATH).read_text())
RESULTS = {}
by_line = {}
for site in sites:
    rec = {
        "id": site["id"],
        "line": site["line"],
        "expr": site["expr"],
        "how": site["how"],
        "status": "not_reached",
        "gdb_print": None,
        "stopped_line": None,
        "stopped_file": None,
        "located_on_line": None,
    }
    RESULTS[str(site["id"])] = rec
    by_line.setdefault(site["line"], []).append(site)

try:
    gdb.execute("set amdgpu precise-memory on")
except gdb.error:
    pass

for line, group in by_line.items():
    PrintLineBreakpoint(line, group)

gdb.write("probe: %d PRINT sites on %d lines\n" % (len(sites), len(by_line)))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a HIPSmith kernel under ROCgdb, stop once per PRINT line that "
            "actually lands on that source line, and record gdb print output."
        ),
        epilog="Kernel arguments go after -- .",
    )
    parser.add_argument(
        "kernel",
        type=Path,
        help="Compiled HIPSmith kernel (build with -g)",
    )
    parser.add_argument(
        "hip_file",
        type=Path,
        help="HIPProg.hip source with PRINT_* macros",
    )
    parser.add_argument(
        "--rocgdb",
        required=True,
        type=Path,
        help="Path to rocgdb",
    )
    parser.add_argument(
        "--timeout",
        required=True,
        metavar="DURATION",
        help="Duration passed to timeout(1), e.g. 30 or 120s",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="JSON report path (default: <kernel>.gdb.json next to the binary)",
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


def main() -> int:
    args = parse_args()
    kernel = args.kernel.resolve()
    hip_file = args.hip_file.resolve()
    rocgdb = args.rocgdb.resolve()

    if not kernel.is_file():
        print(f"error: no such file: {kernel}", file=sys.stderr)
        return 1
    if not os.access(kernel, os.X_OK):
        print(f"error: not executable: {kernel}", file=sys.stderr)
        return 1
    if not hip_file.is_file():
        print(f"error: no such file: {hip_file}", file=sys.stderr)
        return 1
    if not rocgdb.is_file() or not os.access(rocgdb, os.X_OK):
        print(f"error: rocgdb not executable: {rocgdb}", file=sys.stderr)
        return 1

    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        print("error: timeout(1) not found on PATH", file=sys.stderr)
        return 1

    sites = parse_print_sites(hip_file)
    output = args.output
    if output is None:
        output = kernel.with_name(kernel.name + ".gdb.json")
    output = output.resolve()

    if not sites:
        output.write_text("{}\n", encoding="utf-8")
        print(f"no PRINT macros in {hip_file}")
        return 0

    with tempfile.TemporaryDirectory(prefix="run-gdb_") as tmp:
        tmp_path = Path(tmp)
        sites_json = tmp_path / "sites.json"
        result_json = tmp_path / "result.json"
        helper = tmp_path / "probe.py"
        driver = tmp_path / "probe.gdb"
        sites_json.write_text(json.dumps(sites, indent=2) + "\n", encoding="utf-8")
        helper.write_text(GDB_HELPER.lstrip("\n"), encoding="utf-8")
        driver.write_text(
            "set pagination off\n"
            "set confirm off\n"
            "set breakpoint pending on\n"
            "set startup-with-shell off\n"
            f"source {helper}\n"
            "run\n"
            "python dump_results()\n"
            "quit\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PROBE_SITES_JSON"] = str(sites_json)
        env["PROBE_RESULT_JSON"] = str(result_json)
        env["PROBE_HIP_BASENAME"] = hip_file.name

        cmd = [
            timeout_bin,
            "-s9",
            args.timeout,
            str(rocgdb),
            "-batch",
            "-nx",
            "-x",
            str(driver),
            "--args",
            str(kernel),
            *args.extra,
        ]
        print(" ".join(cmd))
        result = subprocess.run(cmd, env=env)

        if result_json.is_file():
            report = json.loads(result_json.read_text(encoding="utf-8"))
        else:
            report = {
                str(site["id"]): {
                    "id": site["id"],
                    "line": site["line"],
                    "expr": site["expr"],
                    "how": site["how"],
                    "status": "not_reached",
                    "gdb_print": None,
                    "stopped_line": None,
                    "stopped_file": None,
                    "located_on_line": None,
                }
                for site in sites
            }
            print("error: gdb did not write a result file", file=sys.stderr)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")

        counts: dict[str, int] = {}
        for rec in report.values():
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        print("status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    sys.exit(main())
