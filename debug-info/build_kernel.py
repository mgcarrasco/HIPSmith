#!/usr/bin/env python3
"""Compile a HIPSmith HIPProg.hip + HIP-driver.cpp pair."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_build_argv(argv: list[str]) -> argparse.Namespace:
    """Parse ``build_kernel.py`` arguments, including flags after ``--``."""
    parser = argparse.ArgumentParser(
        description=(
            "Compile HIPProg.hip with the HIP-driver.cpp next to it. "
            "Pass extra compiler flags after -- ."
        ),
        epilog="Example: build_kernel.py HIPProg.hip -I runtime --debug -- -O3 --offload-arch=gfx90a",
    )
    parser.add_argument(
        "hip_file",
        type=Path,
        help="Path to HIPProg.hip (HIP-driver.cpp must sit beside it)",
    )
    parser.add_argument(
        "-I",
        "--include-dir",
        dest="include_dirs",
        action="append",
        required=True,
        metavar="DIR",
        help="Include directory with HIPSmith.h and safe_math_macros.h "
        "(repeatable)",
    )
    parser.add_argument(
        "--compiler",
        default="amdclang++",
        help="HIP compiler (default: amdclang++)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output binary (default: <hip_file stem> in the same directory)",
    )
    parser.add_argument(
        "--print-noop",
        action="store_true",
        help="Define HIPSMITH_PRINT_NOOP (PRINT_* becomes ((void)0))",
    )
    parser.add_argument(
        "--print-escape",
        action="store_true",
        help="Define HIPSMITH_PRINT_ESCAPE (PRINT_* becomes a volatile read "
        "of the variable, forcing it to keep its storage without printing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass -g",
    )
    extra: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        extra = argv[idx + 1 :]
        argv = argv[:idx]
    args = parser.parse_args(argv)
    args.extra = extra
    return args


def parse_args() -> argparse.Namespace:
    return parse_build_argv(sys.argv[1:])


def recorded_argv(cmd: list[str]) -> list[str]:
    """Strip the python interpreter and this script from a recorded command."""
    i = 0
    if cmd and Path(cmd[0]).name.startswith("python"):
        i = 1
    if i < len(cmd) and Path(cmd[i]).name == "build_kernel.py":
        i += 1
    return cmd[i:]


def compile_argv(
    compiler: str | Path,
    hip_file: Path,
    include_dirs: list[str | Path],
    extra: list[str],
    output: Path,
    *,
    debug: bool = False,
    print_noop: bool = False,
    print_escape: bool = False,
) -> list[str]:
    """Return the amdclang++ argv used for a HIPSmith kernel link."""
    hip_file = hip_file.resolve()
    driver = hip_file.with_name("HIP-driver.cpp")
    cmd = [str(compiler), "-x", "hip"]
    if debug:
        cmd.append("-g")
    if print_noop:
        cmd.append("-DHIPSMITH_PRINT_NOOP")
    if print_escape:
        cmd.append("-DHIPSMITH_PRINT_ESCAPE")
    cmd.append("-DHIP_ENABLE_EXTRA_WARP_SYNC_TYPES=1")
    cmd.extend(
        [
            "-fno-strict-aliasing",
            "-Wno-c++11-narrowing",
            "-Wno-unused-value",
            "-fno-finite-loops",
        ]
    )
    for include_dir in include_dirs:
        cmd.extend(["-I", str(Path(include_dir).resolve())])
    cmd.extend(["-I", str(hip_file.parent)])
    cmd.extend([str(driver), str(hip_file), "-o", str(output)])
    cmd.extend(extra)
    return cmd


def main() -> int:
    args = parse_args()
    hip_file = args.hip_file.resolve()
    if not hip_file.is_file():
        print(f"error: no such file: {hip_file}", file=sys.stderr)
        return 1

    driver = hip_file.with_name("HIP-driver.cpp")
    if not driver.is_file():
        print(f"error: expected driver next to kernel: {driver}", file=sys.stderr)
        return 1

    extra = list(args.extra)

    output = args.output
    if output is None:
        output = hip_file.with_suffix("")
    output = output.resolve()

    cmd = compile_argv(
        args.compiler,
        hip_file,
        args.include_dirs,
        extra,
        output,
        debug=args.debug,
        print_noop=args.print_noop,
        print_escape=args.print_escape,
    )

    print(" ".join(cmd))
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
