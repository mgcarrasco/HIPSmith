#!/usr/bin/env python3
"""Classify one fuzz-one.py report as a debug-info failure, or not.

Failures are only reported for a well defined iteration, where every build ran
and agreed, so a kernel-vs-ROCgdb disagreement can only come from debug info.

  incorrectness   a target build gave a concrete value the kernel did not print
  incompleteness  a value that provably existed went missing, either because
                  the reference build recovered it (way 1) or because escape
                  mode's volatile load must have materialised it (way 2)

Both ways are split by how the value went missing -- optimized_out,
no_line_debug_info, not_reached -- as three separate defects.

ROCgdb breakpoints disable after one hit, so comparisons use the first
execution of each print id. Values compare as bit patterns of width 8 * sizeof.

Way 2 assumes ROCgdb's stop PC is at or after the volatile load; run-gdb.py
checks only the stopped line, not the position within it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CHAR_SUFFIX_RE = re.compile(r"""^(.*\S)\s+'.*'$""")
GDB_VALUE_RE = re.compile(r"^[^=\n]*=\s*(.*)$", re.DOTALL)

SIZEOF_BYTES = (1, 2, 4, 8)

PRINT_MODES = ("printf", "noop", "escape")
TARGET_BUILDS = tuple(f"target.{mode}" for mode in PRINT_MODES)
REFERENCE_BUILDS = tuple(f"reference.{mode}" for mode in PRINT_MODES)
DI_BUILDS = TARGET_BUILDS + REFERENCE_BUILDS
UBCHECK_BUILDS = ("ubcheck.O0", "ubcheck.O3", "ubcheck.ubsan", "ubcheck.asan")
ALL_RUNS = DI_BUILDS + UBCHECK_BUILDS

# noop and escape emit no PRINT lines, so only these can be compared.
PRINTING_BUILDS = ("target.printf", "reference.printf") + UBCHECK_BUILDS

INCORRECTNESS = "incorrectness"
WAY1 = "incompleteness.way1"
WAY2 = "incompleteness.way2"

MISSING_OPTIMIZED_OUT = "optimized_out"
MISSING_NO_LINE_INFO = "no_line_debug_info"
MISSING_NOT_REACHED = "not_reached"

FINDING_KINDS = (
    INCORRECTNESS,
    f"{WAY1}.{MISSING_OPTIMIZED_OUT}",
    f"{WAY1}.{MISSING_NO_LINE_INFO}",
    f"{WAY1}.{MISSING_NOT_REACHED}",
    f"{WAY2}.{MISSING_OPTIMIZED_OUT}",
    f"{WAY2}.{MISSING_NO_LINE_INFO}",
    f"{WAY2}.{MISSING_NOT_REACHED}",
)

INFO_REFERENCE_INCORRECT = "info.reference_incorrect"
INFO_TARGET_KEPT = "info.target_kept_reference_lost"
INFO_KINDS = (INFO_REFERENCE_INCORRECT, INFO_TARGET_KEPT)


class OracleError(Exception):
    """The report cannot be classified at all (missing or malformed)."""


def load_report(path: Path) -> dict[str, Any]:
    """Read a fuzz-one.py report, accepting either the JSON or its run dir."""
    if path.is_dir():
        for candidate in (path / "fuzz-one.json", path / "out" / "fuzz-one.json"):
            if candidate.is_file():
                path = candidate
                break
        else:
            raise OracleError(f"no fuzz-one.json under {path}")
    if not path.is_file():
        raise OracleError(f"no such file: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OracleError(f"{path}: {exc}") from None
    if not isinstance(report, dict):
        raise OracleError(f"{path}: report is not a JSON object")
    return report


def parse_int(value: Any) -> int | None:
    """Integer behind a printed value, or None if it is not one.

    Accepts ROCgdb's "61 '='" and 0x forms; `<optimized out>` yields None.
    """
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


def gdb_value(gdb_print: Any) -> str | None:
    """Strip ROCgdb's "$1 = " prefix from a `print` transcript."""
    if gdb_print is None:
        return None
    text = str(gdb_print).strip()
    match = GDB_VALUE_RE.match(text)
    return match.group(1).strip() if match else text


def values_match(left: Any, right: Any, size: int) -> bool:
    """Compare as bit patterns of width 8 * size.

    PRINT_UINT64 renders through %llu while ROCgdb prints the same datum
    signed, so exact equality would call identical values different.
    """
    left_n = parse_int(left)
    right_n = parse_int(right)
    if left_n is None or right_n is None:
        return False
    if size not in SIZEOF_BYTES:
        return left_n == right_n
    mask = (1 << (size * 8)) - 1
    return (left_n & mask) == (right_n & mask)


def print_sequence(run: dict[str, Any]) -> list[tuple[int, str]]:
    """The (id, value) pairs a build printed, in execution order."""
    report = run.get("report") or {}
    return [
        (int(rec["id"]), str(rec["value"]))
        for rec in report.get("prints") or []
        if "id" in rec and "value" in rec
    ]


def first_prints(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """First execution of each print id, which is what ROCgdb observed."""
    first: dict[int, dict[str, Any]] = {}
    for rec in (run.get("report") or {}).get("prints") or []:
        if "id" not in rec or "value" not in rec:
            continue
        first.setdefault(int(rec["id"]), rec)
    return first


def well_definedness_failures(report: dict[str, Any]) -> list[str]:
    """Reasons this iteration cannot support a verdict; empty means it can."""
    reasons: list[str] = []
    runs = report.get("runs") or {}
    gdb = report.get("gdb") or {}

    # Check the runs before the probes: an iteration that died in the run
    # stage has no gdb probes, and the run failure is the useful reason.
    missing = [name for name in ALL_RUNS if name not in runs]
    if missing:
        return [f"runs missing: {', '.join(missing)}"]

    for name in ALL_RUNS:
        run = runs[name]
        if not run.get("ok"):
            reasons.append(f"{name}: run stage failed ({run.get('failure')})")
            continue
        exit_code = (run.get("report") or {}).get("exit_code")
        if exit_code != 0:
            reasons.append(f"{name}: kernel exited {exit_code}")

    crcs = {name: (runs[name].get("report") or {}).get("crc") for name in ALL_RUNS}
    absent = sorted(name for name, crc in crcs.items() if crc is None)
    if absent:
        reasons.append(f"no CRC reported by: {', '.join(absent)}")
    elif len(set(crcs.values())) != 1:
        shown = ", ".join(f"{name}={crc}" for name, crc in sorted(crcs.items()))
        reasons.append(f"CRC disagreement: {shown}")

    # CRC only covers final state; the UB-check rebuilds exist to catch
    # intermediate values shifting under -O0/-O3/UBSan/ASan.
    expected = print_sequence(runs["target.printf"])
    for name in PRINTING_BUILDS[1:]:
        if print_sequence(runs[name]) != expected:
            reasons.append(f"{name}: PRINT values differ from target.printf")

    missing_gdb = [name for name in DI_BUILDS if name not in gdb]
    if missing_gdb:
        reasons.append(f"gdb probes missing: {', '.join(missing_gdb)}")
    for name in DI_BUILDS:
        if name in gdb and not gdb[name].get("ok"):
            reasons.append(f"{name}: gdb probe failed ({gdb[name].get('failure')})")

    return reasons


def probe(gdb: dict[str, Any], build: str, print_id: int) -> dict[str, Any] | None:
    return ((gdb.get(build) or {}).get("report") or {}).get(str(print_id))


def missing_kind(record: dict[str, Any] | None) -> str | None:
    """How this probe failed to produce a concrete value, or None if it did."""
    if record is None:
        return MISSING_NOT_REACHED
    status = record.get("status")
    if status == "printed":
        if is_concrete(gdb_value(record.get("gdb_print"))):
            return None
        return MISSING_OPTIMIZED_OUT
    if status == "no_line_debug_info":
        return MISSING_NO_LINE_INFO
    return MISSING_NOT_REACHED


def _site(record: dict[str, Any] | None, run_rec: dict[str, Any],
          print_id: int) -> dict[str, Any]:
    """Identity of a probe site, preferring the gdb record's own copy."""
    source = record or run_rec
    return {
        "id": print_id,
        "line": source.get("line"),
        "expr": source.get("expr"),
        "how": source.get("how"),
    }


def _observed(record: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "gdb_status": record.get("status") if record else "not_reached",
        "gdb_value": gdb_value(record.get("gdb_print")) if record else None,
        "stopped_line": record.get("stopped_line") if record else None,
    }


def classify(report: dict[str, Any]) -> dict[str, Any]:
    """Full verdict for one fuzz-one.py report."""
    gate = well_definedness_failures(report)
    verdict: dict[str, Any] = {
        "seed": report.get("seed"),
        "opt": (report.get("config") or {}).get("opt"),
        "well_defined": not gate,
        "well_definedness_failures": gate,
        "counts": {kind: 0 for kind in FINDING_KINDS},
        "findings": [],
        "info": [],
        "stats": {"ids_executed": 0, "ids_skipped": []},
    }
    if gate:
        return verdict

    runs = report["runs"]
    gdb = report["gdb"]
    executed = first_prints(runs["target.printf"])
    verdict["stats"]["ids_executed"] = len(executed)

    findings: list[dict[str, Any]] = verdict["findings"]
    info: list[dict[str, Any]] = verdict["info"]

    for print_id in sorted(executed):
        run_rec = executed[print_id]
        expected = str(run_rec["value"])
        size = int(run_rec.get("sizeof") or 0)
        if parse_int(expected) is None:
            verdict["stats"]["ids_skipped"].append(
                {"id": print_id, "reason": f"printf value {expected!r} is not an integer"})
            continue
        if size not in SIZEOF_BYTES:
            verdict["stats"]["ids_skipped"].append(
                {"id": print_id, "reason": f"sizeof {size!r} is not 1, 2, 4 or 8"})
            continue

        common = {
            "printf_first": expected,
            "sizeof": size,
        }

        for mode in PRINT_MODES:
            target_build = f"target.{mode}"
            reference_build = f"reference.{mode}"
            target = probe(gdb, target_build, print_id)
            reference = probe(gdb, reference_build, print_id)
            target_missing = missing_kind(target)
            reference_missing = missing_kind(reference)
            site = _site(target, run_rec, print_id)

            # A concrete target value that is not what ran.
            if target_missing is None:
                if not values_match(expected, gdb_value(target["gdb_print"]), size):
                    findings.append({
                        "kind": INCORRECTNESS, "build": target_build,
                        **site, **common, **_observed(target)})

            # way 1: the reference build recovered what the target lost, and
            # recovered the value the kernel printed.
            elif reference_missing is None and values_match(
                    expected, gdb_value(reference["gdb_print"]), size):
                findings.append({
                    "kind": f"{WAY1}.{target_missing}", "build": target_build,
                    **site, **common, **_observed(target),
                    "reference_build": reference_build,
                    "reference_value": gdb_value(reference["gdb_print"])})

            # Mirror cases, not failures: a wrong reference value says
            # nothing about the target, and target-kept is an improvement.
            if reference_missing is None and not values_match(
                    expected, gdb_value(reference["gdb_print"]), size):
                info.append({
                    "kind": INFO_REFERENCE_INCORRECT, "build": reference_build,
                    **site, **common, **_observed(reference)})
            if target_missing is None and reference_missing is not None:
                info.append({
                    "kind": INFO_TARGET_KEPT, "build": target_build,
                    **site, **common, **_observed(target),
                    "reference_build": reference_build,
                    "reference_missing": reference_missing})

        # way 2: escape mode is a volatile load, so the value provably exists.
        # Independent of way 1; both may fire on one site.
        escape = probe(gdb, "target.escape", print_id)
        escape_missing = missing_kind(escape)
        if escape_missing is not None:
            findings.append({
                "kind": f"{WAY2}.{escape_missing}", "build": "target.escape",
                **_site(escape, run_rec, print_id), **common, **_observed(escape)})

    for finding in findings:
        verdict["counts"][finding["kind"]] += 1
    verdict["counts"]["findings_total"] = len(findings)
    verdict["counts"]["info_total"] = len(info)
    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify one fuzz-one.py report as a debug-info failure. Reports "
            "incorrectness (ROCgdb gave a concrete value the kernel did not "
            "print) and incompleteness (ROCgdb lost a value that provably "
            "existed), for iterations whose builds all agree the program was "
            "well defined."
        )
    )
    parser.add_argument(
        "report",
        type=Path,
        help="fuzz-one.json, or a run directory containing one",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the full verdict JSON here (default: stdout)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a one-line human summary to stderr instead of the verdict",
    )
    return parser.parse_args()


def summarise(verdict: dict[str, Any]) -> str:
    if not verdict["well_defined"]:
        return f"not well defined: {verdict['well_definedness_failures'][0]}"
    counts = verdict["counts"]
    fired = [f"{kind}={counts[kind]}" for kind in FINDING_KINDS if counts[kind]]
    if not fired:
        return f"clean ({verdict['stats']['ids_executed']} print ids)"
    return " ".join(fired)


def main() -> int:
    args = parse_args()
    try:
        report = load_report(args.report)
    except OracleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    verdict = classify(report)
    verdict["report"] = str(args.report)

    text = json.dumps(verdict, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.summary:
        print(summarise(verdict), file=sys.stderr)
    elif args.output is None:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
