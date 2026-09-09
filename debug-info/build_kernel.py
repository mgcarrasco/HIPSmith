#!/usr/bin/env python3
"""Compile a HIPSmith HIPProg.hip + HIP-driver.cpp pair."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
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

    cmd = [args.compiler, "-x", "hip"]
    if args.debug:
        cmd.append("-g")
    if args.print_noop:
        cmd.append("-DHIPSMITH_PRINT_NOOP")
    if args.print_escape:
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
    for include_dir in args.include_dirs:
        cmd.extend(["-I", str(Path(include_dir).resolve())])
    cmd.extend(["-I", str(hip_file.parent)])
    cmd.extend([str(driver), str(hip_file), "-o", str(output)])
    cmd.extend(extra)

    print(" ".join(cmd))
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
