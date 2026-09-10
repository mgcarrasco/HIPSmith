#!/usr/bin/env python3
"""Run many fuzz-one.py iterations in parallel, continuously, and collect a dataset.

Each iteration is an independent `fuzz-one.py` subprocess: generate a random kernel,
build it 10 ways, run each, and gdb-probe the 6 debug-info builds among them (the other
4 are UB-check rebuilds of target.printf at pinned optimisation levels). This script
only orchestrates that — it does not judge whether an iteration is "interesting" beyond
fuzz-one.py's own mechanical exit code (0 ok, 2 generate failed, 3 build failed, 4 run
failed, 5 gdb probe failed). The UB-check builds count towards codes 3 and 4 like any
other build. Every
iteration's full report lives at `<run-dir>/out/fuzz-one.json`; `campaign.json` is a
live, lightweight index over all of them (seed, exit code, duration, directory), kept
up to date after every completed iteration so it can be read at any time, including
mid-run or after an interrupt.

  ./di-fuzz-campaign.py --amdclang .../amdclang++ --rocgdb .../rocgdb \\
      --hipsmith build-argc/HIPSmith --iterations 100

  ./di-fuzz-campaign.py --amdclang ... --rocgdb ... --hipsmith ...   # runs until Ctrl-C

Ctrl-C once: stop submitting new iterations, let in-flight ones finish and get
recorded. Ctrl-C twice: also kill in-flight fuzz-one.py subprocesses immediately.

Exit codes:
  0  normal stop (Ctrl-C, --iterations/--duration exhausted)
  1  usage/setup error
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
FUZZ_ONE = SCRIPTS / "fuzz-one.py"

# fuzz-one.py builds target/reference x printf/noop/escape plus three UB-check
# rebuilds of target.printf, and a fourth under ASan unless --no-asan-check. It
# runs all of them, but gdb-probes only the six debug-info builds, so its build
# and run stages have more items to get through than its gdb stage.
DI_VARIANTS = 6
UBCHECK_VARIANTS = 3
GDB_VARIANTS = DI_VARIANTS


def build_variants(asan_check: bool) -> int:
    """How many builds fuzz-one.py will make with these settings.

    An upper bound rather than an exact count: when the target is already -O0 or
    -O3 one UB-check variant is aliased onto it instead of being built again.
    Overestimating only makes the watchdog more forgiving.
    """
    return DI_VARIANTS + UBCHECK_VARIANTS + (1 if asan_check else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--amdclang", required=True, type=Path,
                        help="Path to amdclang++, passed to fuzz-one.py")
    parser.add_argument("--rocgdb", required=True, type=Path,
                        help="Path to rocgdb, passed to fuzz-one.py")
    parser.add_argument("--hipsmith", required=True, type=Path,
                        help="Path to the HIPSmith binary, passed to fuzz-one.py "
                             "(fuzz-one.py and gen_kernel.py have no auto-discovery "
                             "fallback, so this is always required)")
    parser.add_argument("--generate-timeout", type=float, default=30.0,
                        metavar="SECONDS",
                        help="Generation timeout inside each iteration "
                             "(default: 30)")
    parser.add_argument("--build-timeout", type=float, default=45.0,
                        metavar="SECONDS",
                        help="Per-build timeout inside each iteration (default: 45)")
    parser.add_argument("--run-timeout", type=float, default=30.0,
                        metavar="SECONDS",
                        help="Per-run and per-gdb-probe timeout inside each "
                             "iteration (default: 30)")
    parser.add_argument("--offload-arch", default="native",
                        help="Value for --offload-arch (default: native)")
    parser.add_argument("--gisel", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Pass --gisel to fuzz-one.py, letting iterations "
                             "pick -mllvm -global-isel=true at random "
                             "(default: --no-gisel)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Concurrent fuzz-one.py subprocesses (default: 12)")
    parser.add_argument("--asan-check", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Pass --asan-check to fuzz-one.py, adding an -O0 "
                             "device-AddressSanitizer rebuild of the target "
                             "(default: on)")
    parser.add_argument("--inner-jobs", type=int, default=10,
                        help="Each fuzz-one.py's own --jobs (default: 10, one per "
                             "build variant). workers * inner-jobs concurrent "
                             "build/run/gdb workers contend for the GPU; lower this "
                             "pair if contention produces spurious timeouts.")
    parser.add_argument("--iterations", type=int,
                        help="Stop after this many total iterations")
    parser.add_argument("--duration", type=float, metavar="SECONDS",
                        help="Stop after this much wall-clock time")
    parser.add_argument("--out-dir", type=Path,
                        help="Root directory for campaign artifacts "
                             "(default: ./fuzz-campaign-<timestamp>)")
    parser.add_argument("--iteration-timeout", type=float, metavar="SECONDS",
                        help="Hard per-iteration watchdog, independent of "
                             "fuzz-one.py's own internal timeouts (default: "
                             "generate-timeout + ceil(builds / inner-jobs) * "
                             "(build-timeout + run-timeout) + ceil(6 / inner-jobs) "
                             "* run-timeout + 60, where builds is 10, or 9 under "
                             "--no-asan-check)")
    parser.add_argument("-o", "--summary", type=Path,
                        help="Path to the live campaign index "
                             "(default: <out-dir>/campaign.json)")
    return parser.parse_args()


def run_iteration(index: int, args: argparse.Namespace, out_dir: Path,
                   iteration_timeout: float,
                   active_procs: dict[int, subprocess.Popen],
                   active_procs_lock: threading.Lock) -> dict[str, Any]:
    run_dir = out_dir / f"run-{index:08d}"
    run_dir.mkdir(parents=True)
    gen_dir = run_dir / "out"

    cmd = [
        sys.executable, str(FUZZ_ONE),
        "--amdclang", str(args.amdclang),
        "--rocgdb", str(args.rocgdb),
        "--hipsmith", str(args.hipsmith),
        "--generate-timeout", str(args.generate_timeout),
        "--build-timeout", str(args.build_timeout),
        "--run-timeout", str(args.run_timeout),
        "--offload-arch", args.offload_arch,
        "--gisel" if args.gisel else "--no-gisel",
        "--asan-check" if args.asan_check else "--no-asan-check",
        "--jobs", str(args.inner_jobs),
        "--out-dir", str(gen_dir),
    ]

    started = time.perf_counter()
    orchestrator_timeout = False
    with open(run_dir / "stdout.log", "wb") as out_f, \
         open(run_dir / "stderr.log", "wb") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
        with active_procs_lock:
            active_procs[index] = proc
        try:
            exit_code: Any = proc.wait(timeout=iteration_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            exit_code = "orchestrator_timeout"
            orchestrator_timeout = True
        finally:
            with active_procs_lock:
                active_procs.pop(index, None)
    duration = time.perf_counter() - started

    (run_dir / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")

    seed = None
    report_path = gen_dir / "fuzz-one.json"
    if report_path.is_file():
        try:
            seed = json.loads(report_path.read_text(encoding="utf-8")).get("seed")
        except (OSError, json.JSONDecodeError):
            pass

    final_dir = run_dir
    if seed is not None:
        final_dir = out_dir / f"run-{index:08d}-{seed}"
        run_dir.rename(final_dir)

    return {
        "dir": final_dir.name,
        "seed": seed,
        "exit_code": exit_code,
        "duration": duration,
        "orchestrator_timeout": orchestrator_timeout,
    }


def main() -> int:
    args = parse_args()

    if not args.amdclang.is_file():
        print(f"error: {args.amdclang} is not a file", file=sys.stderr)
        return 1
    if not args.rocgdb.is_file():
        print(f"error: {args.rocgdb} is not a file", file=sys.stderr)
        return 1
    if not args.hipsmith.is_file():
        print(f"error: {args.hipsmith} is not a file", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("error: --workers must be at least 1", file=sys.stderr)
        return 1
    if args.inner_jobs < 1:
        print("error: --inner-jobs must be at least 1", file=sys.stderr)
        return 1

    out_dir = args.out_dir or Path(f"fuzz-campaign-{datetime.now():%Y%m%dT%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (out_dir / "campaign.json")

    iteration_timeout = args.iteration_timeout
    if iteration_timeout is None:
        # The build and run stages push one item per build variant through a
        # pool of --inner-jobs and the gdb stage pushes GDB_VARIANTS, so each
        # takes that many rounds; generation is a single step. Worst case every
        # step in every round hits its own timeout, then a minute of slack for
        # process spawn, temp-dir cleanup and report writes.
        build_rounds = math.ceil(
            build_variants(args.asan_check) / args.inner_jobs)
        gdb_rounds = math.ceil(GDB_VARIANTS / args.inner_jobs)
        iteration_timeout = (args.generate_timeout
                             + build_rounds * (args.build_timeout
                                               + args.run_timeout)
                             + gdb_rounds * args.run_timeout
                             + 60)

    print(f"out-dir: {out_dir}", file=sys.stderr)
    print(f"summary: {summary_path}", file=sys.stderr)
    print(f"workers: {args.workers}  inner-jobs: {args.inner_jobs}  "
          f"({args.workers * args.inner_jobs} concurrent build/run/gdb workers)",
          file=sys.stderr)
    print(f"watchdog: {iteration_timeout:g}s per iteration", file=sys.stderr)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_perf = time.perf_counter()
    iterations: list[dict[str, Any]] = []
    counts_by_exit_code: dict[str, int] = {}

    def write_summary() -> None:
        payload = {
            "started_at": started_at,
            "elapsed_seconds": time.perf_counter() - start_perf,
            "total_iterations": len(iterations),
            "counts_by_exit_code": counts_by_exit_code,
            "iterations": iterations,
        }
        tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, summary_path)

    stop_requested = False
    active_procs: dict[int, subprocess.Popen] = {}
    active_procs_lock = threading.Lock()

    def handle_sigint(signum, frame) -> None:
        nonlocal stop_requested
        if stop_requested:
            print("\nsecond Ctrl-C: killing in-flight iterations", file=sys.stderr)
            with active_procs_lock:
                for proc in active_procs.values():
                    proc.kill()
        else:
            stop_requested = True
            print("\nCtrl-C: finishing in-flight iterations, not starting new ones "
                  "(press again to stop immediately)", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_sigint)

    deadline = time.perf_counter() + args.duration if args.duration is not None else None
    next_index = 1

    def should_submit_more() -> bool:
        if stop_requested:
            return False
        if args.iterations is not None and next_index > args.iterations:
            return False
        if deadline is not None and time.perf_counter() >= deadline:
            return False
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        in_flight: dict[Any, int] = {}
        try:
            while True:
                while len(in_flight) < args.workers and should_submit_more():
                    future = pool.submit(run_iteration, next_index, args, out_dir,
                                         iteration_timeout, active_procs,
                                         active_procs_lock)
                    in_flight[future] = next_index
                    next_index += 1

                if not in_flight:
                    break

                done, _ = wait(list(in_flight), timeout=1.0,
                               return_when=FIRST_COMPLETED)
                for future in done:
                    idx = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # bug in this script, not the kernel
                        result = {"dir": f"run-{idx:08d}", "seed": None,
                                  "exit_code": "orchestrator_error",
                                  "duration": None, "error": str(exc)}
                    iterations.append(result)
                    key = str(result["exit_code"])
                    counts_by_exit_code[key] = counts_by_exit_code.get(key, 0) + 1
                    write_summary()
                    dur = result["duration"]
                    dur_text = f"{dur:.1f}s" if dur is not None else "?"
                    print(f"[{len(iterations)}] {result['dir']}: "
                          f"exit={result['exit_code']} ({dur_text})", file=sys.stderr)
        finally:
            write_summary()

    elapsed = time.perf_counter() - start_perf
    print(f"done: {len(iterations)} iterations in {elapsed:.1f}s -> {out_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
