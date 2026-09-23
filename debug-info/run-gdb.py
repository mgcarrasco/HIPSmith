#!/usr/bin/env python3
"""Stop ROCgdb at each normal breakpoint location of a PRINT line.

Breakpoints are set on PRINT source lines. gdb_print is the first of those
stops. located_on_line is separate: null until a clean probe finishes, true if
any location of that breakpoint prints a concrete integer, false only when the
probe finished with no concrete print. Each location is stopped once. A failed
probe leaves null and the process exits non-zero.
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
STATUS_PATH = os.environ["PROBE_STATUS_PATH"]
HIP_BASENAME = os.environ["PROBE_HIP_BASENAME"]
KERNEL = "hipsmith_kernel"

CHAR_SUFFIX_RE = re.compile(r"^(.*\S)\s+'.*'$")
GDB_VALUE_RE = re.compile(r"^[^=\n]*=\s*(.*)$", re.DOTALL)
STRAY_LIMIT = 64

STATE = {
    "walk_ok": False,
    "armed": False,
    "at_kernel": False,
    "exited": False,
    "stray": 0,
    "reason": None,
}

LINE_BPS = []
WANTED = {}
SEEN = set()


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


def is_amdgpu():
    try:
        return gdb.selected_frame().architecture().name().startswith("amdgcn")
    except gdb.error:
        return False


def mark_located(site):
    rec = RESULTS[str(site["id"])]
    if rec.get("located_on_line") is True:
        return
    expr = site["how"] or site["expr"]
    if is_concrete_print(run_capture("print " + expr)):
        rec["located_on_line"] = True


class CollectError(Exception):
    pass


def loc_addr(loc):
    addr = loc.address
    if isinstance(addr, int):
        return addr
    return int(str(addr).strip().split()[0], 0)


def record_stop(bp):
    """First stop writes status and gdb_print. Later stops only set sticky true."""
    filename, actual_line = current_sal()
    base = Path(filename).name if filename else ""
    on_line = base == HIP_BASENAME and actual_line == bp.expected_line
    if not bp.status_recorded:
        bp.status_recorded = True
        for site in bp.sites:
            rec = RESULTS[str(site["id"])]
            rec["stopped_line"] = actual_line
            rec["stopped_file"] = filename
            if not on_line:
                rec["status"] = "no_line_debug_info"
                rec["gdb_print"] = None
                continue
            expr = site["how"] or site["expr"]
            printed = run_capture("print " + expr)
            rec["status"] = "printed"
            rec["gdb_print"] = printed.rstrip("\n")
            if is_concrete_print(printed):
                rec["located_on_line"] = True
        return
    if on_line:
        for site in bp.sites:
            mark_located(site)


def refresh_locations():
    """Pick up locations that resolved after the code object loaded.

    A breakpoint that is still pending has nowhere a normal breakpoint can
    stop. That is an empty stop set, not a failed probe.
    """
    added = 0
    for bp in LINE_BPS:
        if getattr(bp, "pending", False):
            continue
        for loc in list(bp.locations):
            try:
                if not loc.enabled:
                    continue
            except (gdb.error, AttributeError):
                pass
            try:
                pc = loc_addr(loc)
            except (TypeError, ValueError, gdb.error):
                continue
            if pc in SEEN or pc in WANTED:
                continue
            WANTED.setdefault(pc, []).append((bp, loc))
            added += 1
    return added


def snapshot_locations():
    """The places a normal file:line breakpoint can stop, once device code is in."""
    WANTED.clear()
    refresh_locations()
    for bp in LINE_BPS:
        if getattr(bp, "pending", False):
            gdb.write("locations %s: pending\n" % bp.location)
            continue
        n = sum(1 for hits in WANTED.values() for owner, _loc in hits if owner is bp)
        gdb.write("locations %s: %d\n" % (bp.location, n))
    gdb.write("collect: %d normal breakpoint locations\n" % sum(
        len(hits) for hits in WANTED.values()))


def disable_location(loc):
    try:
        loc.enabled = False
    except (gdb.error, RuntimeError, AttributeError):
        pass


def take_stop(pc):
    hits = WANTED.pop(pc, None)
    if not hits:
        return False
    SEEN.add(pc)
    seen = set()
    for bp, loc in hits:
        disable_location(loc)
        key = id(bp)
        if key in seen:
            continue
        seen.add(key)
        record_stop(bp)
    return True


def disarm_rest():
    for hits in list(WANTED.values()):
        for _bp, loc in hits:
            disable_location(loc)
    WANTED.clear()


def dump_results():
    Path(RESULT_PATH).write_text(json.dumps(RESULTS, indent=2) + "\n")
    Path(STATUS_PATH).write_text("ok\n" if STATE["walk_ok"] else "fail\n")


def finish():
    if STATE["walk_ok"]:
        for rec in RESULTS.values():
            if rec.get("located_on_line") is None:
                rec["located_on_line"] = False
    else:
        gdb.write("probe failed: %s\n" % (STATE["reason"] or "walk incomplete"))
    dump_results()


def fail(reason):
    STATE["walk_ok"] = False
    STATE["reason"] = reason
    gdb.write("collect/walk: %s\n" % reason)


class KernelEntry(gdb.Breakpoint):
    def stop(self):
        if STATE["armed"] or not is_amdgpu():
            return False
        STATE["at_kernel"] = True
        return True


class PrintLineBreakpoint(gdb.Breakpoint):
    def __init__(self, line, sites):
        super(PrintLineBreakpoint, self).__init__("%s:%d" % (HIP_BASENAME, line))
        self.expected_line = line
        self.sites = sites
        self.status_recorded = False
        try:
            self.silent = True
        except AttributeError:
            pass
        LINE_BPS.append(self)

    def stop(self):
        # The driver prints and disables the location. Altering breakpoints
        # from inside stop is forbidden.
        return STATE["armed"]


def on_exit(event):
    STATE["exited"] = True


def resume():
    if STATE["exited"]:
        return False
    try:
        gdb.execute("continue")
    except gdb.error as exc:
        gdb.write("continue: %s\n" % exc)
        return False
    return not STATE["exited"]


def drive():
    gdb.events.exited.connect(on_exit)
    try:
        gdb.execute("set amdgpu precise-memory on")
    except gdb.error:
        pass
    gdb.execute("set breakpoint pending on")
    gdb.execute("set breakpoint always-inserted on")

    for line, group in by_line.items():
        PrintLineBreakpoint(line, group)
    entry = KernelEntry(KERNEL)
    gdb.write("probe: %d PRINT sites on %d lines\n" % (len(sites), len(by_line)))

    try:
        gdb.execute("run")
    except gdb.error as exc:
        fail("run: %s" % exc)
        return

    if not STATE["at_kernel"] or STATE["exited"]:
        fail("kernel never hit on GPU")
        return

    entry.enabled = False
    try:
        snapshot_locations()
    except CollectError as exc:
        fail(str(exc))
        return
    except (gdb.error, RuntimeError, TypeError, ValueError) as exc:
        fail("locations: %s" % exc)
        return

    STATE["armed"] = True
    here = current_pc()
    if here in WANTED:
        take_stop(here)

    while WANTED and STATE["stray"] < STRAY_LIMIT and resume():
        pc = current_pc()
        if take_stop(pc):
            continue
        if refresh_locations() and take_stop(pc):
            continue
        STATE["stray"] += 1
        gdb.write("  stray stop at %s\n" % (
            "0x%x" % pc if pc is not None else "?"))

    if STATE["stray"] >= STRAY_LIMIT:
        disarm_rest()
        fail("stray-stop cutoff (%d)" % STRAY_LIMIT)
        return

    disarm_rest()
    if not STATE["exited"]:
        try:
            gdb.execute("continue")
        except gdb.error:
            pass
    STATE["walk_ok"] = True


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
    drive()
except Exception as exc:
    fail("probe error: %s" % exc)
finally:
    try:
        finish()
    except Exception as exc:
        gdb.write("finish error: %s\n" % exc)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a HIPSmith kernel under ROCgdb and stop once at each normal "
            "breakpoint location of a PRINT line."
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
        status_path = tmp_path / "probe_status"
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
            "quit\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PROBE_SITES_JSON"] = str(sites_json)
        env["PROBE_RESULT_JSON"] = str(result_json)
        env["PROBE_STATUS_PATH"] = str(status_path)
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

        probe_ok = False
        if result_json.is_file():
            report = json.loads(result_json.read_text(encoding="utf-8"))
            if status_path.is_file() and status_path.read_text(encoding="utf-8").strip() == "ok":
                probe_ok = True
            elif result.returncode == 0:
                print("error: line walk did not finish cleanly", file=sys.stderr)
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
        if result.returncode != 0:
            return result.returncode
        return 0 if probe_ok else 1


if __name__ == "__main__":
    sys.exit(main())
