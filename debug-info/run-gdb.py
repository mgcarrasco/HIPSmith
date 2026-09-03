#!/usr/bin/env python3
"""Stop ROCgdb once per PRINT line and record `print` of the macro expression.

Breakpoints are set on PRINT source lines. On each stop the PC's source line is
checked; a mismatch is treated as a slide (status no_line_debug_info), the
breakpoint is dropped, and execution continues so later sites are not polluted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PRINT_SITE_RE = re.compile(
    r"PRINT_(?:INT8|UINT8|INT16|UINT16|INT64|UINT64|INT|UINT)\("
    r"(?P<expr>.+?), __LINE__, \"(?P<how>[^\"]*)\""
    r"(?:, (?P<id>-?\d+))?"
    r"\)"
)

GDB_HELPER = r'''
import gdb
import json
import os
from pathlib import Path

SITES_PATH = os.environ["PROBE_SITES_JSON"]
RESULT_PATH = os.environ["PROBE_RESULT_JSON"]
HIP_BASENAME = os.environ["PROBE_HIP_BASENAME"]


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


def dump_results():
    Path(RESULT_PATH).write_text(json.dumps(RESULTS, indent=2) + "\n")


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
        for site in self.sites:
            expr = site["how"] or site["expr"]
            printed = run_capture("print " + expr)
            rec = RESULTS[str(site["id"])]
            rec["status"] = "printed"
            rec["gdb_print"] = printed.rstrip("\n")
            rec["stopped_line"] = actual_line
            rec["stopped_file"] = filename
        self.enabled = False
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


def parse_print_sites(hip_file: Path) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    missing_id = 0
    for line_no, raw in enumerate(hip_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for match in PRINT_SITE_RE.finditer(raw):
            id_text = match.group("id")
            if id_text is None:
                missing_id += 1
                continue
            sites.append(
                {
                    "id": int(id_text),
                    "line": line_no,
                    "expr": match.group("expr").strip(),
                    "how": match.group("how"),
                }
            )
    if missing_id:
        raise SystemExit(
            f"error: {missing_id} PRINT_* macros in {hip_file} have no id; "
            "need PRINT_TYPE(expr, __LINE__, \"how\", id)"
        )
    ids = [s["id"] for s in sites]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"error: duplicate PRINT ids in {hip_file}")
    return sites


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
