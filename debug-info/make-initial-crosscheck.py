#!/usr/bin/env python3
"""Build the initial crosscheck JSON that C-Vise reduction starts from.

Mirrors what run-vs-gdb-interestingness.py does for binaries A and C: a
printf -O0 build supplies the run values, the -g target build (optionally
in noop/escape print mode) supplies the gdb values. The result is written
as <stem>.crosscheck.initial.json so cvise-run-vs-gdb.py finds it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def load_sibling(filename: str) -> Any:
    """Import a dash-named sibling script so its helpers can be reused."""
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").removesuffix(".py"), path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interestingness = load_sibling("run-vs-gdb-interestingness.py")
predicate = load_sibling("test-run-vs-gdb.py")


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the initial crosscheck for a HIPSmith reduction: printf "
            "-O0 run values against the target -g build's gdb values."
        ),
        epilog="Extra compiler flags for the target build go after -- .",
    )
    parser.add_argument("hip_file", type=Path, help="HIPProg.hip to reduce")
    parser.add_argument("--compiler", required=True, type=Path, help="amdclang++")
    parser.add_argument("--rocgdb", required=True, type=Path, help="rocgdb")
    parser.add_argument(
        "--offload-arch",
        required=True,
        help="GPU arch, e.g. gfx90a",
    )
    parser.add_argument(
        "--print-mode",
        choices=("printf", "noop", "escape"),
        default="printf",
        help="Print mode for the -g build (default: %(default)s)",
    )
    parser.add_argument(
        "-I",
        "--include-dir",
        dest="include_dirs",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra include directory (the hip file's own directory is "
        "always included)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Crosscheck JSON path (default: <stem>.crosscheck.initial.json "
        "next to the hip file)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep binaries and intermediate JSON here (default: a "
        "build/ subdirectory next to the hip file)",
    )
    parser.add_argument("--run-timeout", default="120", metavar="DURATION")
    parser.add_argument("--gdb-timeout", default="600", metavar="DURATION")
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        extra = argv[idx + 1 :]
        argv = argv[:idx]
    args = parser.parse_args(argv)
    if not extra:
        parser.error("extra compiler flags after -- are required")
    args.extra = extra
    return args


def run_step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"[{name}] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        die(f"{name} exited {proc.returncode}")


def main() -> int:
    args = parse_args()
    hip_file = args.hip_file.resolve()
    if not hip_file.is_file():
        die(f"no such file: {hip_file}")
    compiler = interestingness.existing_exe(args.compiler)
    rocgdb = interestingness.existing_exe(args.rocgdb)
    for exe in (compiler, rocgdb):
        if not exe.is_file():
            die(f"not found: {exe}")

    work = (args.work_dir or hip_file.parent / "build").resolve()
    work.mkdir(parents=True, exist_ok=True)
    output = args.output or hip_file.with_suffix(".crosscheck.initial.json")
    env = interestingness.toolchain_env(compiler)
    py = sys.executable or "python3"
    include_dirs = [str(hip_file.parent), *args.include_dirs]

    def build(name: str, debug: bool, flags: list[str]) -> Path:
        binary = work / name
        cmd = [py, str(HERE / "build_kernel.py"), str(hip_file),
               "--compiler", str(compiler), "-o", str(binary)]
        for inc in include_dirs:
            cmd.extend(["-I", inc])
        if debug:
            cmd.append("--debug")
            if args.print_mode != "printf":
                cmd.append(f"--print-{args.print_mode}")
        cmd.append("--")
        cmd.extend(interestingness.COMMON_BUILD_FLAGS)
        cmd.extend(flags)
        cmd.append(f"--offload-arch={args.offload_arch}")
        run_step(f"build {name}", cmd, env)
        return binary

    binary_a = build("A", False, ["-O0"])
    binary_c = build("C", True, list(args.extra))

    report_a = work / "A.json"
    run_step("run A", [py, str(HERE / "run_kernel.py"), str(binary_a),
                       "--timeout", args.run_timeout, "-o", str(report_a)], env)
    report = json.loads(report_a.read_text(encoding="utf-8"))
    if report.get("exit_code") != 0:
        die(f"A exit_code {report.get('exit_code')}")
    if not report.get("prints"):
        die("A produced no PRINT lines")

    gdb_json = work / "C.gdb.json"
    run_step("run-gdb C", [py, str(HERE / "run-gdb.py"), str(binary_c), str(hip_file),
                           "--rocgdb", str(rocgdb), "--timeout", args.gdb_timeout,
                           "-o", str(gdb_json)], env)

    run_step("crosscheck", [py, str(HERE / "crosscheck-run-vs-gdb.py"),
                            str(report_a), str(gdb_json), "-o", str(output),
                            "--unexpected", str(work / "gdb-only.json")], env)

    # Same notion of divergence test-run-vs-gdb.py uses: concrete gdb value
    # that does not bit-match the run value.
    rows = json.loads(output.read_text(encoding="utf-8"))
    diverging = [
        row
        for row in rows
        if predicate.is_concrete(row.get("gdb_value"))
        and not predicate.values_match(
            row.get("run_value"), row.get("gdb_value"),
            row, int(row["print_id"]), "initial", True,
        )
    ]
    print(f"\nwrote {output} ({len(rows)} rows, "
          f"{len(diverging)} with a concrete diverging gdb value)")
    for row in diverging[:20]:
        print(f"  id {row['print_id']}: run={row['run_value']} "
              f"gdb={row['gdb_value']} status={row.get('gdb_status')}")
    if not diverging:
        print("  (nothing to reduce for an incorrectness case)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
