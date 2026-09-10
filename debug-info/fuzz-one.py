#!/usr/bin/env python3
"""One fuzzing iteration for HIP debug information.

Generates a random kernel, picks a random "target" compiler configuration, and
derives a "reference" configuration from it by turning off instruction
referencing. Both are built in the three PRINT modes, run, and probed under
ROCgdb; the outputs are collected into a single JSON report.

Alongside those six, the target is rebuilt in printf mode up to four more times
with its optimisation level replaced by -O0, by -O3, by -O0 plus trapping
UBSan, and — unless --no-asan-check — by -O0 plus device AddressSanitizer on an
xnack+ offload arch. These UB-check variants are built and run like any other, but are not
gdb-probed: if they do not all print what target.printf printed, the kernel has
undefined behaviour or has been miscompiled and is not a trustworthy debug-info
sample. This script does not make that comparison — it records the outputs and
leaves the judgement to whatever reads the report — but it does hold the
variants to the same bar as every other build, so a UB-check build or run that
fails ends the iteration.

The generated source and the seed that produced it are kept in the output
directory, so the binaries can be deleted: any finding is reproducible from
`fuzz-one.py --seed <seed>`.

  ./fuzz-one.py --amdclang .../amdclang++ --rocgdb .../rocgdb \
      --build-timeout 600 --run-timeout 120

Exit codes:
  0  everything ran
  2  HIPSmith failed to generate (commonly a code-generation assertion) or
     timed out
  3  a build failed or timed out
  4  a kernel run failed or timed out
  5  a ROCgdb probe failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
GEN_KERNEL = SCRIPTS / "gen_kernel.py"
BUILD_KERNEL = SCRIPTS / "build_kernel.py"
RUN_KERNEL = SCRIPTS / "run_kernel.py"
RUN_GDB = SCRIPTS / "run-gdb.py"

OPT_LEVELS = ["-O0", "-O1", "-O2", "-O3"]

# Turning instruction referencing off is what makes a build the reference.
REFERENCE_ONLY = ["-mllvm", "-experimental-debug-variable-locations=false"]

# Trapping rather than diagnosing: there is no UBSan runtime for device code,
# so a violation aborts the kernel and the run stage records the failure.
UBSAN_ONLY = ["-fsanitize=undefined", "-fsanitize-trap=all"]

# Device ASan needs an xnack+ offload arch or the driver silently drops the
# instrumentation, and it needs xnack switched on again at run time. There is no
# instrumented libamdhip64.so in the toolchains this runs against, so a device
# fault has no hostcall handler to report through and the process aborts without
# a diagnostic; that still fails the run, which is all this variant is for.
# detect_leaks is off because LeakSanitizer reports the HSA runtime's own
# allocations and would fail every run.
ASAN_ONLY = ["-fsanitize=address"]
ASAN_ARCH_SUFFIX = ":xnack+"
ASAN_ENV = {"HSA_XNACK": "1", "ASAN_OPTIONS": "detect_leaks=0"}

# (name, configuration, extra build_kernel.py flags)
DI_BUILDS = [
    ("target.printf", "target", []),
    ("target.noop", "target", ["--print-noop"]),
    ("target.escape", "target", ["--print-escape"]),
    ("reference.printf", "reference", []),
    ("reference.noop", "reference", ["--print-noop"]),
    ("reference.escape", "reference", ["--print-escape"]),
]

# Printf-mode rebuilds of the target at pinned optimisation levels, used to tell
# a real debug-info divergence from one that only exists because the kernel has
# UB. Always printf mode and never gdb-probed: their whole purpose is to be
# value-comparable against target.printf.
UBCHECK_BUILDS = [
    ("ubcheck.O0", "ubcheck_O0", []),
    ("ubcheck.O3", "ubcheck_O3", []),
    ("ubcheck.ubsan", "ubcheck_ubsan", []),
]

# Sibling of ubcheck.ubsan: also -O0, also a sanitizer, also only ever a
# run-stage verdict. Optional because it is the one variant that cannot use the
# iteration's own offload arch verbatim.
ASAN_BUILD = ("ubcheck.asan", "ubcheck_asan", [])


def ubcheck_builds(asan_check: bool) -> list[tuple[str, str, list[str]]]:
    return UBCHECK_BUILDS + ([ASAN_BUILD] if asan_check else [])


def all_builds(asan_check: bool) -> list[tuple[str, str, list[str]]]:
    return DI_BUILDS + ubcheck_builds(asan_check)

EXIT_OK = 0
EXIT_GEN = 2
EXIT_BUILD = 3
EXIT_RUN = 4
EXIT_GDB = 5


def fresh_entropy() -> int:
    """A seed that differs even between processes started in the same instant."""
    value = int.from_bytes(os.urandom(8), "big")
    value ^= os.getpid() << 16
    value ^= time.time_ns()
    return value % (2**31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--amdclang", required=True, type=Path,
                        help="Path to amdclang++")
    parser.add_argument("--rocgdb", required=True, type=Path,
                        help="Path to rocgdb")
    parser.add_argument("--generate-timeout", type=float, default=30.0,
                        metavar="SECONDS",
                        help="Timeout for generation (default: 30)")
    parser.add_argument("--build-timeout", required=True, type=float,
                        metavar="SECONDS",
                        help="Timeout for each build")
    parser.add_argument("--run-timeout", required=True, type=float,
                        metavar="SECONDS",
                        help="Timeout for each kernel run and each gdb probe")
    parser.add_argument("--offload-arch", default="native",
                        help="Value for --offload-arch (default: native)")
    parser.add_argument("--gisel", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Let the configuration pick -mllvm "
                             "-global-isel=true at random (default: --no-gisel, "
                             "never add it to any build)")
    parser.add_argument("--asan-check", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Also build and run the target at -O0 under device "
                             "AddressSanitizer (default: on). Needs an xnack+ "
                             "offload arch, so a 'native' --offload-arch is "
                             "resolved to a concrete one first")
    parser.add_argument("--jobs", type=int, default=10,
                        help="Parallel builds, runs and gdb probes (default: 10, "
                             "one per build variant)")
    parser.add_argument("--seed", type=int,
                        help="Reproduce a previous iteration")
    parser.add_argument("--out-dir", type=Path,
                        help="Where to keep the generated source and the report "
                             "(default: ./fuzz-<seed>)")
    parser.add_argument("--hipsmith", required=True, type=Path,
                        help="Path to the HIPSmith binary, passed to gen_kernel.py")
    parser.add_argument("-o", "--json", type=Path,
                        help="Also write the report here (default: stdout)")
    return parser.parse_args()


def resolve_offload_arch(arch: str, amdclang: Path) -> str:
    """Turn 'native' into a concrete arch.

    `--offload-arch=native:xnack+` is rejected outright by the driver, and plain
    `native` makes it drop the ASan instrumentation with only a warning, so the
    ASan variant cannot work off 'native'. amdgpu-arch ships beside the compiler
    and prints one line per visible GPU; the first is the one to build for.
    """
    if arch != "native":
        return arch
    for candidate in (amdclang.parent / "amdgpu-arch",
                      amdclang.parent.parent / "lib/llvm/bin/amdgpu-arch"):
        if not candidate.is_file():
            continue
        try:
            out = subprocess.run([str(candidate)], stdout=subprocess.PIPE,
                                 text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if lines:
            return lines[0]
    raise RuntimeError(
        "--asan-check needs a concrete --offload-arch and amdgpu-arch could not "
        "supply one; pass --offload-arch explicitly or --no-asan-check")


def pick_config(rng: random.Random, offload_arch: str,
                allow_gisel: bool = False) -> dict[str, Any]:
    """Choose the target configuration, then derive the reference and the
    UB-check configurations from it.

    With allow_gisel false — the default — GlobalISel is off for every build in
    the iteration, not merely for the UB-check ones, so that the target and its
    UB-check rebuilds still differ in optimisation alone.
    """
    opt = rng.choice(OPT_LEVELS)
    # Drawn either way so that a seed maps to the same opt and liveness whether
    # or not GlobalISel is allowed; only the outcome is forced.
    global_isel = rng.choice([False, True]) and allow_gisel
    extend_liveness = rng.choice([False, True])

    # Everything the target is built with except its optimisation level and its
    # offload arch. The UB-check variants pin their own -O onto this same base,
    # so they differ from the target in optimisation alone.
    extras: list[str] = []
    if global_isel:
        # The driver rejects a bare -global-isel=true; it is an LLVM option.
        extras += ["-mllvm", "-global-isel=true"]
    if extend_liveness:
        extras += ["-fextend-variable-liveness=all"]

    base = [f"--offload-arch={offload_arch}"] + extras
    target = [opt] + base

    return {
        "opt": opt,
        "global_isel": global_isel,
        "extend_variable_liveness": extend_liveness,
        "offload_arch": offload_arch,
        "target_flags": target,
        "reference_flags": target + REFERENCE_ONLY,
        "ubcheck_O0_flags": ["-O0"] + base,
        "ubcheck_O3_flags": ["-O3"] + base,
        "ubcheck_ubsan_flags": ["-O0"] + base + UBSAN_ONLY,
        # The only variant that does not share `base`: ASan is ignored unless
        # the offload arch itself carries xnack+.
        "ubcheck_asan_flags": (
            ["-O0", f"--offload-arch={offload_arch}{ASAN_ARCH_SUFFIX}"]
            + extras + ASAN_ONLY),
    }


def ubcheck_aliases(config: dict[str, Any],
                    asan_check: bool) -> dict[str, str]:
    """Map each UB-check variant that is already the target build onto it.

    When the target's own optimisation level is -O0 or -O3 the corresponding
    variant would compile the same source with the same flags in the same print
    mode, so it is built and run once and reported under both names. Callers
    still find all three UB-check entries in the report; the duplicate carries
    an `alias_of` naming what it was copied from.
    """
    return {
        name: "target.printf"
        for name, key, _ in ubcheck_builds(asan_check)
        if config[f"{key}_flags"] == config["target_flags"]
    }


def run_step(cmd: list[str], timeout: float,
             env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a command, capturing what a failure report needs and nothing more."""
    started = time.perf_counter()
    child_env = {**os.environ, **env} if env else None
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout, env=child_env)
        exit_code, stdout, stderr, timed_out = (
            proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as exc:
        exit_code, timed_out = None, True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")

    record: dict[str, Any] = {
        "cmd": cmd,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration": time.perf_counter() - started,
        "ok": exit_code == 0,
    }
    if not record["ok"]:
        # Only keep output for failures; a passing build logs thousands of
        # narrowing warnings that would swamp the report.
        record["stdout_tail"] = stdout[-4000:]
        record["stderr_tail"] = stderr[-4000:]
    return record


def generate(args: argparse.Namespace, gen_seed: int,
             out_dir: Path) -> tuple[dict[str, Any], list[str]]:
    cmd = [sys.executable, str(GEN_KERNEL), "--seed", str(gen_seed),
           "-o", str(out_dir), "--hipsmith", str(args.hipsmith)]
    # Leave --hip-print-same-line on: in noop mode PRINT_* expands to ((void)0),
    # so a breakpoint only has something to land on because the PRINT shares its
    # line with a real statement.
    timed_out = False
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=args.generate_timeout)
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        # Some seeds wedge HIPSmith outright. Nothing downstream bounds this
        # stage, so without a timeout here the whole iteration hangs forever.
        exit_code, timed_out = None, True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")

    flags: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("flags:"):
            flags = line.split(":", 1)[1].split()
    record = {
        "cmd": cmd,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "ok": exit_code == 0,
        "flags": flags,
    }
    if not record["ok"]:
        record["stdout_tail"] = stdout[-4000:]
        record["stderr_tail"] = stderr[-4000:]
    return record, flags


def sha256(path: Path) -> str | None:
    """SHA-256 of the generated kernel, as a sanity check that a given seed
    really does reproduce the same program."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def emit(report: dict[str, Any], out_dir: Path,
         json_path: Path | None) -> None:
    text = json.dumps(report, indent=2) + "\n"
    # Always keep a copy beside the source it describes.
    (out_dir / "fuzz-one.json").write_text(text, encoding="utf-8")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else fresh_entropy()
    rng = random.Random(seed)

    # Derive the generator seed from ours, so --seed reproduces the whole
    # iteration rather than just the compiler configuration.
    gen_seed = rng.randrange(2**31)
    # Resolved before the config so every variant, not just the ASan one, names
    # the same concrete arch in its recorded flags.
    try:
        offload_arch = (resolve_offload_arch(args.offload_arch, args.amdclang)
                        if args.asan_check else args.offload_arch)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    config = pick_config(rng, offload_arch, allow_gisel=args.gisel)
    builds_wanted = all_builds(args.asan_check)

    out_dir = (args.out_dir or Path(f"fuzz-{seed}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aliases = ubcheck_aliases(config, args.asan_check)

    report: dict[str, Any] = {
        "seed": seed,
        "gen_seed": gen_seed,
        "source_dir": str(out_dir),
        "config": config,
        # Stated up front so a reader knows which entries under builds/runs are
        # UB-check variants without having to recognise their names, and which
        # of them are copies of target.printf rather than separate compilations.
        "ubcheck": {
            name: {
                "flags": config[f"{key}_flags"],
                "alias_of": aliases.get(name),
                "gdb_probed": False,
                "env": ASAN_ENV if name == ASAN_BUILD[0] else {},
            }
            for name, key, _ in ubcheck_builds(args.asan_check)
        },
        "stage": "generate",
    }

    print(f"seed:   {seed}", file=sys.stderr)
    print(f"config: {' '.join(config['target_flags'])}", file=sys.stderr)
    print(f"out:    {out_dir}", file=sys.stderr)

    gen_record, gen_flags = generate(args, gen_seed, out_dir)
    report["generate"] = gen_record
    report["gen_flags"] = gen_flags
    if not gen_record["ok"]:
        emit(report, out_dir, args.json)
        return EXIT_GEN

    hip_file = out_dir / "HIPProg.hip"
    report["hip_file"] = hip_file.name
    report["hip_sha256"] = sha256(hip_file)

    with tempfile.TemporaryDirectory(prefix="fuzz-one_") as tmp:
        # Binaries live here and are always removed; the source and the seed in
        # out_dir are what make a finding reproducible.
        tmp_path = Path(tmp)

        # ---- build ----
        report["stage"] = "build"

        def build(entry: tuple[str, str, list[str]]) -> tuple[str, dict[str, Any]]:
            name, which, extra = entry
            cmd = [sys.executable, str(BUILD_KERNEL), str(hip_file),
                   "-I", str(out_dir), "--compiler", str(args.amdclang),
                   "--debug", *extra,
                   "-o", str(tmp_path / f"{name}.out"),
                   "--", *config[f"{which}_flags"]]
            return name, run_step(cmd, args.build_timeout)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            builds = dict(pool.map(
                build, [e for e in builds_wanted if e[0] not in aliases]))
        for name, source in aliases.items():
            builds[name] = dict(builds[source], alias_of=source)
        report["builds"] = builds

        # A UB-check variant that will not build leaves the sample unvetted, so
        # it stops the iteration exactly like a debug-info build failure. -O0
        # strains both the backend and the scratch-frame limit, so this is not a
        # rare path.
        if any(not rec["ok"] for rec in builds.values()):
            emit(report, out_dir, args.json)
            return EXIT_BUILD

        # ---- run ----
        report["stage"] = "run"

        def run_one(name: str) -> tuple[str, dict[str, Any]]:
            out = tmp_path / f"{name}.run.json"
            cmd = [sys.executable, str(RUN_KERNEL), str(tmp_path / f"{name}.out"),
                   "--timeout", str(args.run_timeout), "-o", str(out)]
            env = ASAN_ENV if name == ASAN_BUILD[0] else None
            record = run_step(cmd, args.run_timeout * 2 + 60, env=env)
            kernel_report = read_json(out)
            record["report"] = kernel_report
            # run_kernel.py exits 0 even when the kernel itself died — it only
            # fails on a cross-thread CRC mismatch, and otherwise just records
            # the child's status in the report. Everything this safeguard exists
            # to catch shows up there and nowhere else: a UBSan trap (the driver
            # exits 1 after HIP_CHECK sees the queue exception), an ASan abort,
            # and a kernel killed on its own timeout (-9), whose print list is
            # truncated at an arbitrary point and would otherwise be compared
            # against the other variants as if it were complete.
            kernel_exit = (kernel_report or {}).get("exit_code")
            record["kernel_exit_code"] = kernel_exit
            if record["ok"] and kernel_exit != 0:
                record["ok"] = False
                record["failure"] = (
                    "no report from run_kernel.py" if kernel_report is None
                    else "kernel killed on timeout" if kernel_exit == -9
                    else f"kernel exited {kernel_exit}")
            return name, record

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            runs = dict(pool.map(
                run_one, [n for n, _, _ in builds_wanted if n not in aliases]))
        for name, source in aliases.items():
            runs[name] = dict(runs[source], alias_of=source)
        report["runs"] = runs

        # run-gdb only makes sense once every kernel has run cleanly. That
        # includes the UB-check variants: a trapping UBSan build or an -O0
        # rebuild that faults has shown the sample cannot be trusted, and there
        # is nothing to learn from probing its debug info.
        if any(not rec["ok"] for rec in runs.values()):
            emit(report, out_dir, args.json)
            return EXIT_RUN

        # ---- gdb ----
        report["stage"] = "gdb"

        def gdb_one(name: str) -> tuple[str, dict[str, Any]]:
            out = tmp_path / f"{name}.gdb.json"
            cmd = [sys.executable, str(RUN_GDB), str(tmp_path / f"{name}.out"),
                   str(hip_file), "--rocgdb", str(args.rocgdb),
                   "--timeout", str(args.run_timeout), "-o", str(out)]
            record = run_step(cmd, args.run_timeout * 2 + 60)
            record["report"] = read_json(out)
            return name, record

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            gdb = dict(pool.map(gdb_one, [n for n, _, _ in DI_BUILDS]))
        report["gdb"] = gdb

        gdb_failed = any(not rec["ok"] for rec in gdb.values())

    report["stage"] = "done"
    emit(report, out_dir, args.json)

    if gdb_failed:
        return EXIT_GDB
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
