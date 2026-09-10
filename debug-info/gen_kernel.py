#!/usr/bin/env python3
"""Generate a random HIP kernel with HIPSmith.

Picks a fresh random seed and a random subset of the HIP generation flags, runs
HIPSmith in the output directory, and copies in the headers the generated code
includes by name so the directory can be compiled as-is.

  ./gen_kernel.py                    # generate into ./hipsmith-<seed>/
  ./gen_kernel.py -o out             # generate into ./out/
  ./gen_kernel.py --seed 42          # reproduce a previous seed
  ./gen_kernel.py --no-same-line     # plain --hip-print instead
  ./gen_kernel.py --no-print         # no print statements at all
  ./gen_kernel.py --no-argc-threads  # let the flags decide the launch geometry
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Flags that are picked at random, in any combination.
# --small is deliberately absent, and the print and --hip-argc-threads flags are
# driven from the CLI rather than chosen randomly.
HIP_FLAGS = [
    "--vectors",
    "--atomics",
    "--hip-consts",
    "--hip-shared",
    "--hip-managed",
    "--hip-device",
    "--hip-builtins",
    "--hip-sync",
    "--hip-warp",
    "--hip-warp-match",
    "--hip-warp-shuffle",
    "--hip-warp-reduce",
]

# Headers the generated kernel and driver include by name.
REQUIRED_HEADERS = ["HIPSmith.h", "HIPSmithPrint.h", "safe_math_macros.h"]

# Files HIPSmith writes into its working directory.
GENERATED = ["HIPProg.hip", "HIP-driver.cpp"]


def fresh_entropy():
    """A value that differs even between processes started in the same instant.

    os.urandom draws from the kernel entropy pool, so it does not collide the
    way time- or pid-derived seeds do; pid and clock are mixed in only as
    belt and braces.
    """
    value = int.from_bytes(os.urandom(8), "big")
    value ^= os.getpid() << 16
    value ^= time.time_ns()
    return value % (2**31)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output-dir",
                        help="directory to generate into "
                             "(default: ./hipsmith-<seed>)")
    parser.add_argument("--seed", type=int,
                        help="use this seed instead of a random one")
    parser.add_argument("--hipsmith", required=True,
                        help="path to the HIPSmith binary")
    parser.add_argument("--no-print", dest="print_", action="store_false",
                        help="generate no print statements at all")
    parser.add_argument("--no-same-line", dest="same_line",
                        action="store_false",
                        help="pass --hip-print instead of --hip-print-same-line")
    parser.add_argument("--no-argc-threads", dest="argc_threads",
                        action="store_false",
                        help="omit --hip-argc-threads, so the randomly picked "
                             "flags decide the launch geometry")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else fresh_entropy()

    # Drive the flag choice from the seed too, so that --seed reproduces the
    # whole run and not just HIPSmith's own randomness. Two processes spawned at
    # the same moment still differ, because the seed itself comes from urandom.
    rng = random.Random(seed)

    flags = rng.sample(HIP_FLAGS, rng.randint(1, len(HIP_FLAGS)))
    if args.print_:
        # --hip-print-same-line already implies --hip-print.
        flags.append("--hip-print-same-line" if args.same_line else "--hip-print")
    if args.argc_threads:
        # Single thread, so the CRC does not depend on how the threads interleave.
        flags.append("--hip-argc-threads")
    flags.sort()

    binary = Path(args.hipsmith)
    if not binary.is_file():
        sys.exit(f"error: {binary} is not a file")

    out_dir = Path(args.output_dir or f"hipsmith-{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"seed:  {seed}")
    print(f"flags: {' '.join(flags)}")
    print(f"out:   {out_dir}")

    # HIPSmith writes HIPProg.hip and HIP-driver.cpp into the working directory.
    cmd = [str(binary.resolve()), "--seed", str(seed)] + flags
    result = subprocess.run(cmd, cwd=out_dir)
    if result.returncode != 0:
        sys.exit(f"error: HIPSmith exited {result.returncode}")

    missing = [f for f in GENERATED if not (out_dir / f).is_file()]
    if missing:
        sys.exit(f"error: HIPSmith did not produce {', '.join(missing)}")

    # A cmake build puts the headers in the build directory next to the binary.
    build_dir = binary.resolve().parent
    for header in REQUIRED_HEADERS:
        source = build_dir / header
        if not source.is_file():
            sys.exit(f"error: {header} not found next to the binary in "
                     f"{build_dir}; re-run cmake there")
        shutil.copy(source, out_dir / header)

    print("ok:    " + " ".join(GENERATED + REQUIRED_HEADERS))
    print(f"repro: {Path(__file__).name} --seed {seed} " + " ".join(flags))


if __name__ == "__main__":
    main()
