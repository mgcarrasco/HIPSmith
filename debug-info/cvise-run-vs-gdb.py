#!/usr/bin/env python3
"""Start C-Vise with run-vs-gdb-interestingness.py as the interestingness test.

Copies the HIP file into a unique work directory and runs C-Vise there, so the
original file is never overwritten. Companion sources (HIP-driver.cpp, headers)
are taken from --resource-dir and are not reduced.

Each invocation gets its own work directory and TMPDIR, so several C-Vise
instances can run at once without sharing files.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
INTERESTINGNESS = HERE / "run-vs-gdb-interestingness.py"
DEFAULT_RUNTIME = REPO / "HIPSmith" / "runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run C-Vise on a HIPSmith HIP file using the PRINT vs ROCgdb "
            "interestingness test. The original HIP file is left unchanged."
        ),
        epilog=(
            "Extra compiler flags for interestingness binary C are required after -- . "
            "Extra cvise flags: --cvise-arg ARG (repeatable)."
        ),
    )
    parser.add_argument(
        "hip_file",
        type=Path,
        help="HIP file to reduce (copied; never modified in place)",
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        required=True,
        help="amdclang++ (required)",
    )
    parser.add_argument(
        "--rocgdb",
        type=Path,
        help="rocgdb (default: $ROCGDB, else rocgdb next to --compiler)",
    )
    parser.add_argument(
        "--compile-timeout",
        default=os.environ.get("COMPILE_TIMEOUT", "30"),
        metavar="DURATION",
        help="timeout(1) duration for each compile (default: 30)",
    )
    parser.add_argument(
        "--run-timeout",
        default=os.environ.get("RUN_TIMEOUT", "5"),
        metavar="DURATION",
        help="timeout(1) duration for each kernel run (default: 5)",
    )
    parser.add_argument(
        "--gdb-timeout",
        default=os.environ.get("GDB_TIMEOUT", "5"),
        metavar="DURATION",
        help="timeout(1) duration for run-gdb.py (default: 5)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("JOBS", "4")),
        metavar="N",
        help="Parallel compiles/runs among A–D inside one interestingness test "
        "(default: 4)",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=8,
        metavar="N",
        dest="cvise_jobs",
        help="C-Vise parallel interestingness workers (default: 8)",
    )
    parser.add_argument(
        "--crosscheck",
        type=Path,
        help="Starting crosscheck JSON (default: the unique "
        "*.crosscheck.initial.json next to hip_file)",
    )
    parser.add_argument(
        "--ids",
        metavar="ID[,ID...]",
        help="Targeted print ids (default: every print_id in --crosscheck)",
    )
    parser.add_argument(
        "--no-bit-equality",
        action="store_true",
        help="Pass through to the interestingness test",
    )
    parser.add_argument(
        "--offload-arch",
        required=True,
        metavar="ARCH",
        help="GPU arch for every compile (passed as --offload-arch=ARCH)",
    )
    parser.add_argument(
        "-I",
        "--include-dir",
        dest="include_dirs",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra -I for build_kernel.py (repeatable)",
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        help="Directory with HIP-driver.cpp / headers (default: hip_file parent)",
    )
    parser.add_argument(
        "--cvise",
        type=Path,
        required=True,
        help="cvise executable (required)",
    )
    parser.add_argument(
        "--clang-delta-std",
        default="c++17",
        choices=["c++98", "c++11", "c++14", "c++17", "c++20", "c++2b"],
        help="Passed to cvise --clang-delta-std (default: c++17)",
    )
    parser.add_argument(
        "--cvise-timeout",
        type=int,
        metavar="SECONDS",
        help="C-Vise interestingness timeout (default: derived from compile/run/gdb "
        "timeouts and --jobs)",
    )
    parser.add_argument(
        "--cvise-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument forwarded to cvise (repeatable; use "
        "--cvise-arg=--not-c so argparse does not eat the flag)",
    )
    parser.add_argument(
        "--save-temps",
        action="store_true",
        help="Pass --save-temps to C-Vise",
    )
    parser.add_argument(
        "--skip-initial-passes",
        action="store_true",
        help="Pass --skip-initial-passes to C-Vise",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Work directory (default: a unique /tmp/hipsmith-cvise-* directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Copy the reduced HIP file here when C-Vise finishes successfully",
    )
    parser.add_argument(
        "--print-mode",
        choices=("printf", "noop", "escape"),
        default="printf",
        help="Pass through to the interestingness test: print mode for the "
        "gdb binary (default: %(default)s)",
    )
    parser.add_argument(
        "--also-reduce",
        dest="also_reduce",
        action="append",
        default=[],
        metavar="FILE",
        help="Companion file to reduce alongside the HIP file, e.g. "
        "safe_math_macros.h (bare name resolved in --resource-dir, or a "
        "path; repeatable)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug to the interestingness test (keeps hipsmith-int-* dirs)",
    )
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


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def existing_exe(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        if len(path.parts) == 1:
            found = shutil.which(str(path))
            if found:
                path = Path(found)
        if not path.is_absolute():
            path = Path.cwd() / path
    return path


def timeout_seconds(text: str) -> int:
    raw = text.strip().lower()
    if not raw:
        die(f"empty timeout duration: {text!r}")
    unit = 1
    if raw.endswith("d"):
        unit = 86400
        raw = raw[:-1]
    elif raw.endswith("h"):
        unit = 3600
        raw = raw[:-1]
    elif raw.endswith("m"):
        unit = 60
        raw = raw[:-1]
    elif raw.endswith("s"):
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError:
        die(f"cannot parse timeout duration: {text!r}")
    seconds = int(math.ceil(value * unit))
    if seconds < 1:
        die(f"timeout duration must be >= 1s: {text!r}")
    return seconds


def default_rocgdb(compiler: Path) -> Path:
    if "ROCGDB" in os.environ:
        return Path(os.environ["ROCGDB"])
    return compiler.parent / "rocgdb"


def default_crosscheck(hip_file: Path) -> Path:
    matches = sorted(hip_file.parent.glob("*.crosscheck.initial.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        die(
            "no *.crosscheck.initial.json next to the HIP file; pass --crosscheck"
        )
    names = ", ".join(p.name for p in matches)
    die(f"multiple crosscheck JSON files next to the HIP file ({names}); pass --crosscheck")


def abs_existing_file(path: Path, what: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        die(f"no such {what}: {path}")
    return path


def abs_existing_dir(path: Path, what: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_dir():
        die(f"no such {what}: {path}")
    return path


def cvise_invocation(path: Path) -> list[str]:
    cvise = abs_existing_file(path, "--cvise")
    if not os.access(cvise, os.X_OK):
        die(f"cvise is not executable: {cvise}")
    return [str(cvise)]


def write_wrapper(path: Path, cmd: list[str]) -> None:
    body = "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(cmd) + "\n"
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        die("--jobs must be >= 1")
    if args.cvise_jobs < 1:
        die("--n must be >= 1")
    if not INTERESTINGNESS.is_file():
        die(f"missing interestingness script: {INTERESTINGNESS}")

    hip_file = abs_existing_file(args.hip_file, "file")
    hip_real = hip_file.resolve()
    resource_dir = (
        abs_existing_dir(args.resource_dir, "--resource-dir")
        if args.resource_dir
        else abs_existing_dir(hip_file.parent, "HIP file parent")
    )
    crosscheck = abs_existing_file(
        args.crosscheck if args.crosscheck else default_crosscheck(hip_file),
        "--crosscheck",
    )
    compiler = existing_exe(args.compiler)
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        die(f"compiler is not executable: {compiler}")
    rocgdb = existing_exe(args.rocgdb if args.rocgdb else default_rocgdb(compiler))
    if not rocgdb.is_file() or not os.access(rocgdb, os.X_OK):
        die(f"rocgdb is not executable: {rocgdb}")
    cvise_cmd_prefix = cvise_invocation(args.cvise)

    include_dirs = list(args.include_dirs)
    if not include_dirs and DEFAULT_RUNTIME.is_dir():
        include_dirs.append(str(DEFAULT_RUNTIME))
    include_dirs = [str(abs_existing_dir(Path(d), "-I")) for d in include_dirs]

    compile_s = timeout_seconds(args.compile_timeout)
    run_s = timeout_seconds(args.run_timeout)
    gdb_s = timeout_seconds(args.gdb_timeout)
    compile_wall = compile_s * math.ceil(4 / args.jobs)
    cvise_timeout = args.cvise_timeout
    if cvise_timeout is None:
        cvise_timeout = compile_wall + run_s + gdb_s + 30
    if cvise_timeout < 1:
        die("--cvise-timeout must be >= 1")

    if args.work_dir is not None:
        work = args.work_dir.expanduser()
        if not work.is_absolute():
            work = Path.cwd() / work
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="hipsmith-cvise-", dir="/tmp"))

    staged = work / hip_file.name
    if staged.resolve() == hip_real:
        die("refusing to use a work directory that would overwrite the original HIP file")
    if args.output is not None:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        if output.resolve() == hip_real:
            die("refusing to write --output over the original HIP file")
    else:
        output = None

    shutil.copy2(hip_file, staged)

    # Companions C-Vise reduces too. The interestingness test's stage_sources()
    # searches the HIP file's own directory (C-Vise's per-test copy, holding
    # these) before --resource-dir (the original, holding the untouched rest),
    # so the reduced version wins without any change there.
    also_reduce: list[str] = []
    for name in args.also_reduce:
        src = Path(name).expanduser()
        if not src.is_file():
            src = resource_dir / name
        src = abs_existing_file(src, "--also-reduce")
        if src.name == hip_file.name:
            die(f"--also-reduce {name} collides with the HIP file")
        if src.name in also_reduce:
            die(f"--also-reduce {src.name} given twice")
        shutil.copy2(src, work / src.name)
        also_reduce.append(src.name)

    # C-Vise's forkserver binds AF_UNIX under TMPDIR/pymp-*/listener-*
    # (Linux cap 108 bytes). Keep TMPDIR in /tmp even if --work-dir is long.
    tmpdir = Path(tempfile.mkdtemp(prefix="hipsmith-cvise-tmp-", dir="/tmp"))
    (work / "TMPDIR").write_text(str(tmpdir) + "\n", encoding="utf-8")
    (work / "ORIGINAL").write_text(str(hip_real) + "\n", encoding="utf-8")

    wrapper_cmd = [
        sys.executable,
        str(INTERESTINGNESS),
        hip_file.name,
        "--compiler",
        str(compiler),
        "--rocgdb",
        str(rocgdb),
        "--compile-timeout",
        args.compile_timeout,
        "--run-timeout",
        args.run_timeout,
        "--gdb-timeout",
        args.gdb_timeout,
        "--jobs",
        str(args.jobs),
        "--crosscheck",
        str(crosscheck),
        "--resource-dir",
        str(resource_dir),
        "--offload-arch",
        args.offload_arch,
    ]
    for include_dir in include_dirs:
        wrapper_cmd.extend(["-I", include_dir])
    if args.ids is not None:
        wrapper_cmd.extend(["--ids", args.ids])
    if args.no_bit_equality:
        wrapper_cmd.append("--no-bit-equality")
    wrapper_cmd.extend(["--print-mode", args.print_mode])
    if args.debug:
        wrapper_cmd.append("--debug")
    wrapper_cmd.append("--")
    wrapper_cmd.extend(args.extra)

    wrapper = work / "interesting.sh"
    write_wrapper(wrapper, wrapper_cmd)

    cvise_cmd = [
        *cvise_cmd_prefix,
        "--timeout",
        str(cvise_timeout),
        "-n",
        str(args.cvise_jobs),
        "--clang-delta-std",
        args.clang_delta_std,
        *args.cvise_arg,
    ]
    if args.save_temps:
        cvise_cmd.append("--save-temps")
    if args.skip_initial_passes:
        cvise_cmd.append("--skip-initial-passes")
    cvise_cmd.extend([str(wrapper), hip_file.name, *also_reduce])

    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)
    env["TEMP"] = str(tmpdir)
    env["TMP"] = str(tmpdir)

    print(f"original: {hip_real}", file=sys.stderr)
    print(f"work:     {work}", file=sys.stderr)
    print(f"reduced:  {staged}", file=sys.stderr)
    for name in also_reduce:
        print(f"          {work / name}", file=sys.stderr)
    print(f"tmpdir:   {tmpdir}", file=sys.stderr)
    print(f"cvise:    {shlex.join(cvise_cmd)}", file=sys.stderr)

    result = subprocess.run(cvise_cmd, cwd=work, env=env)
    if result.returncode == 0 and output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, output)
        print(f"copied:   {output}", file=sys.stderr)
        for name in also_reduce:
            shutil.copy2(work / name, output.parent / name)
            print(f"          {output.parent / name}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
