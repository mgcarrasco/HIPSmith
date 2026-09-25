#!/usr/bin/env python3
"""Stop ROCgdb at PRINT line breakpoints and at loads in PRINT segments.

Breakpoints are set on PRINT source lines. gdb_print is the first of those
stops. located_on_line is separate: it is decided by every load inside each
PRINT's DWARF column segments, which overapproximates the volatile read.
true if any stopped load prints a concrete integer. false when the probe
finished cleanly and none of the loads that stopped did, including when no
load was found or every load went unexecuted. A failed probe leaves null
and exits non-zero.
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

from print_load_pcs import LoadPcError, annotate_sites_with_loads
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
KERNEL_ELF = int(os.environ["PROBE_KERNEL_ELF"], 0)

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
    "slide": None,
}

LINE_BPS = []
LOAD_BPS = []
WANTED = {}
LOAD_WANTED = {}
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


def record_line_stop(bp):
    """First stop writes status and gdb_print. Does not set located_on_line."""
    filename, actual_line = current_sal()
    base = Path(filename).name if filename else ""
    on_line = base == HIP_BASENAME and actual_line == bp.expected_line
    if bp.status_recorded:
        return
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


def record_load_stop(sites):
    for site in sites:
        mark_located(site)


def refresh_locations():
    """Pick up line-breakpoint locations that resolved after the code object loaded."""
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
            if pc in SEEN or pc in WANTED or pc in LOAD_WANTED:
                continue
            WANTED.setdefault(pc, []).append(("line", bp, loc))
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
        n = sum(
            1
            for hits in WANTED.values()
            for kind, owner, _loc in hits
            if kind == "line" and owner is bp
        )
        gdb.write("locations %s: %d\n" % (bp.location, n))
    gdb.write("collect: %d normal breakpoint locations\n" % sum(
        len(hits) for hits in WANTED.values()))


def is_kernel_name(name):
    if not name:
        return False
    if name == KERNEL or name.startswith(KERNEL + "("):
        return True
    return name.startswith("_Z15hipsmith_kernel") and ".kd" not in name


def function_address(func):
    try:
        return int(func.value().address)
    except (gdb.error, AttributeError, TypeError, ValueError):
        return int(func.value())


def kernel_runtime_address():
    """Loaded address of hipsmith_kernel, not the PC of the entry stop.

    The entry breakpoint stops at the first source line, which can be past
    the symbol, and the selected frame can be a device function inlined into
    the kernel or called from it. Walk out to the hipsmith_kernel frame and
    use that symbol. Its address is the ELF value plus the load slide.
    """
    try:
        frame = gdb.selected_frame()
    except gdb.error as exc:
        raise CollectError("kernel frame: %s" % exc)
    names = []
    while frame is not None:
        try:
            func = frame.function()
        except gdb.error as exc:
            raise CollectError("kernel frame: %s" % exc)
        name = func.name if func is not None else None
        names.append(name)
        if func is not None and is_kernel_name(name):
            if len(names) > 1:
                gdb.write(
                    "kernel frame under %s\n" % " -> ".join(str(n) for n in names)
                )
            try:
                return function_address(func)
            except (gdb.error, TypeError, ValueError) as exc:
                raise CollectError("kernel symbol address: %s" % exc)
        try:
            frame = frame.older()
        except gdb.error as exc:
            raise CollectError("kernel frame: %s" % exc)
    raise CollectError(
        "no %s frame (%s)" % (KERNEL, ", ".join(str(n) for n in names) or "none")
    )


def compute_slide():
    """ELF-to-runtime slide from the loaded hipsmith_kernel symbol."""
    runtime = kernel_runtime_address()
    slide = runtime - KERNEL_ELF
    gdb.write(
        "kernel elf 0x%x runtime 0x%x\n" % (KERNEL_ELF, runtime)
    )
    return slide


def plant_load_breakpoints(slide):
    by_runtime = {}
    for site in sites:
        for elf in site.get("load_elf_pcs") or []:
            runtime = elf + slide
            by_runtime.setdefault(runtime, []).append(site)
    for runtime, group in sorted(by_runtime.items()):
        bp = LoadBreakpoint(runtime, group)
        LOAD_WANTED[runtime] = group
        gdb.write(
            "load bp *0x%x for ids %s\n"
            % (runtime, ",".join(str(s["id"]) for s in group))
        )
    gdb.write("collect: %d load breakpoints\n" % len(by_runtime))


def disable_location(loc):
    try:
        loc.enabled = False
    except (gdb.error, RuntimeError, AttributeError):
        pass


def take_stop(pc):
    if pc in LOAD_WANTED:
        sites = LOAD_WANTED.pop(pc)
        SEEN.add(pc)
        for bp in list(LOAD_BPS):
            if getattr(bp, "runtime_pc", None) == pc:
                bp.enabled = False
        record_load_stop(sites)
        return True
    hits = WANTED.pop(pc, None)
    if not hits:
        return False
    SEEN.add(pc)
    seen = set()
    for kind, bp, loc in hits:
        disable_location(loc)
        if kind != "line":
            continue
        key = id(bp)
        if key in seen:
            continue
        seen.add(key)
        record_line_stop(bp)
    return True


def disarm_rest():
    for hits in list(WANTED.values()):
        for _kind, _bp, loc in hits:
            disable_location(loc)
    WANTED.clear()
    for bp in LOAD_BPS:
        try:
            bp.enabled = False
        except (gdb.error, RuntimeError, AttributeError):
            pass
    LOAD_WANTED.clear()


def dump_results():
    Path(RESULT_PATH).write_text(json.dumps(RESULTS, indent=2) + "\n")
    Path(STATUS_PATH).write_text("ok\n" if STATE["walk_ok"] else "fail\n")


def finish():
    if STATE["walk_ok"]:
        for rec in RESULTS.values():
            if rec.get("located_on_line") is True:
                continue
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
        return STATE["armed"]


class LoadBreakpoint(gdb.Breakpoint):
    def __init__(self, runtime_pc, sites):
        super(LoadBreakpoint, self).__init__("*%#x" % runtime_pc)
        self.runtime_pc = runtime_pc
        self.sites = sites
        try:
            self.silent = True
        except AttributeError:
            pass
        LOAD_BPS.append(self)

    def stop(self):
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


def still_wanted():
    return bool(WANTED) or bool(LOAD_WANTED)


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
        slide = compute_slide()
        STATE["slide"] = slide
        gdb.write("slide: 0x%x\n" % slide)
        plant_load_breakpoints(slide)
    except CollectError as exc:
        fail(str(exc))
        return
    except (gdb.error, RuntimeError, TypeError, ValueError) as exc:
        fail("locations: %s" % exc)
        return

    STATE["armed"] = True
    here = current_pc()
    if here is not None:
        take_stop(here)

    while still_wanted() and STATE["stray"] < STRAY_LIMIT and resume():
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
    site_id = str(site["id"])
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
    RESULTS[site_id] = rec
    line_no = int(site["line"])
    by_line.setdefault(line_no, []).append(site)

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
            "Run a HIPSmith kernel under ROCgdb. Line breakpoints fill "
            "gdb_print; located_on_line is decided by loads in "
            "each PRINT column segment."
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


def empty_report(sites: list[dict[str, Any]]) -> dict[str, Any]:
    return {
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

    try:
        sites, kernel_elf = annotate_sites_with_loads(sites, hip_file, kernel, rocgdb)
    except LoadPcError as exc:
        print(f"error: {exc}", file=sys.stderr)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(empty_report(sites), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {output}")
        return 1

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
        env["PROBE_KERNEL_ELF"] = str(kernel_elf)

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
                print("error: probe did not finish cleanly", file=sys.stderr)
        else:
            report = empty_report(sites)
            print("error: gdb did not write a result file", file=sys.stderr)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")

        counts: dict[str, int] = {}
        for rec in report.values():
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        print("status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        located = {"true": 0, "false": 0, "null": 0}
        for rec in report.values():
            value = rec.get("located_on_line")
            if value is True:
                located["true"] += 1
            elif value is False:
                located["false"] += 1
            else:
                located["null"] += 1
        print(
            "located_on_line: "
            + ", ".join(f"{k}={v}" for k, v in located.items())
        )
        if result.returncode != 0:
            return result.returncode
        return 0 if probe_ok else 1


if __name__ == "__main__":
    sys.exit(main())
