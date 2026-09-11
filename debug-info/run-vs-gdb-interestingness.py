#!/usr/bin/env python3
"""C-Vise interestingness test for HIPSmith PRINT vs ROCgdb.

Exit 0 if interesting, 1 if not, 2 on usage/setup error.

C-Vise invokes a wrapper with no arguments from a temp copy of the test
files. Point that wrapper at this script and pass HIPProg.hip (usually
the copy in cwd). Build artifacts go to a unique temp directory so
parallel C-Vise workers cannot clash.

Builds A–D in parallel, runs all four kernels in parallel, then gdb on C
and compares C's gdb session against A's printf oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
COMPANIONS = (
    "HIP-driver.cpp",
    "HIPSmith.h",
    "HIPSmithPrint.h",  # included by HIPSmith.h
    "safe_math_macros.h",
)
COMMON_BUILD_FLAGS = (
    "-Werror=uninitialized",
    "-Werror=flexible-array-extensions",
    "-Werror=c99-designator",
    "-ftrivial-auto-var-init=zero",
    "-Werror=return-type",
    "-Wno-c++20-designator",
    "-Wl,--allow-multiple-definition",
    "-Werror=array-bounds",
    "-Werror=zero-length-array",
    "-fno-finite-loops",
)

# From hipfuzz interestingness/template_interesting.py — must survive C-Vise reduction.
REQUIRED_ANYWHERE_LINES = (
    "uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;",
    "int tid = threadIdx.x + blockIdx.x * blockDim.x;",
    "results[tid] = (crc64_context ^ 0xFFFFFFFFFFFFFFFFUL);",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interestingness test: O0/O3/ubsan printf oracles plus gdb on "
            "the target -g binary. Exit 0 if still interesting."
        ),
        epilog="Extra compiler flags go after -- .",
    )
    parser.add_argument(
        "hip_file",
        type=Path,
        help="HIPProg.hip to test (C-Vise: the copy in the job directory)",
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        required=True,
        help="amdclang++ (or other HIP compiler)",
    )
    parser.add_argument(
        "--compile-timeout",
        required=True,
        metavar="DURATION",
        help="timeout(1) duration for each compile",
    )
    parser.add_argument(
        "--run-timeout",
        required=True,
        metavar="DURATION",
        help="timeout(1) duration for each kernel run",
    )
    parser.add_argument(
        "--gdb-timeout",
        required=True,
        metavar="DURATION",
        help="timeout(1) duration for run-gdb.py",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        required=True,
        metavar="N",
        help="Max parallel compiles/runs among A–D",
    )
    parser.add_argument(
        "--crosscheck",
        type=Path,
        required=True,
        help="Starting crosscheck JSON (test-run-vs-gdb.py INITIAL)",
    )
    parser.add_argument(
        "--ids",
        metavar="ID[,ID...]",
        help="Targeted print ids (default: every print_id in --crosscheck)",
    )
    parser.add_argument(
        "--no-bit-equality",
        action="store_true",
        help=(
            "Pass through to test-run-vs-gdb.py: compare concrete run vs gdb "
            "values with exact integer equality instead of 8 * sizeof bits"
        ),
    )
    parser.add_argument(
        "--offload-arch",
        required=True,
        metavar="ARCH",
        help="GPU arch for every compile (passed as --offload-arch=ARCH)",
    )
    parser.add_argument(
        "--rocgdb",
        type=Path,
        help="rocgdb (default: rocgdb next to --compiler)",
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
        help="Directory with HIP-driver.cpp / headers if they are not "
        "next to hip_file (C-Vise: the original repro directory)",
    )
    parser.add_argument(
        "--print-mode",
        choices=("printf", "noop", "escape"),
        default="printf",
        help="Print mode for binary C only; A/B/D stay printf so the run "
        "oracle survives (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep the work directory and write interestingness.log there",
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


def die(message: str, code: int = 2) -> None:
    msg = f"error: {message}"
    _emit(msg)
    raise SystemExit(code)


def fail(message: str) -> None:
    msg = f"not interesting: {message}"
    _emit(msg)
    raise SystemExit(1)


_log: "Log | None" = None


def _emit(msg: str) -> None:
    if _log is not None:
        _log.write(msg)
        if not _log.echo:
            print(msg, file=sys.stderr)
    else:
        print(msg, file=sys.stderr)


class Log:
    def __init__(self, path: Path | None, echo: bool) -> None:
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        self._fh = path.open("w", encoding="utf-8") if path is not None else None

    def write(self, msg: str) -> None:
        if not msg.endswith("\n"):
            msg += "\n"
        with self._lock:
            if self._fh is not None:
                self._fh.write(msg)
                self._fh.flush()
            if self.echo:
                sys.stderr.write(msg)
                sys.stderr.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ids(text: str | None, initial_ids: list[int]) -> list[int]:
    if text is None:
        if not initial_ids:
            die("--crosscheck has no print_id values")
        return initial_ids
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part, 10))
    if not ids:
        die("--ids is empty")
    unknown = [i for i in ids if i not in set(initial_ids)]
    if unknown:
        die(f"--ids not present in --crosscheck: {unknown}")
    return ids


def initial_print_ids(path: Path) -> list[int]:
    payload = load_json(path)
    if not isinstance(payload, list):
        die(f"{path} is not a JSON array")
    ids: list[int] = []
    for rec in payload:
        if "print_id" not in rec:
            die(f"{path} has a row with no print_id")
        ids.append(int(rec["print_id"]))
    return ids


def existing_exe(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        found = shutil.which(str(path))
        if found:
            path = Path(found)
        else:
            path = Path.cwd() / path
    return path


def toolchain_env(compiler: Path) -> dict[str, str]:
    env = os.environ.copy()
    compiler_bin = compiler.parent
    rocm = compiler_bin.parent
    env["ROCM_PATH"] = str(rocm)
    env["HIP_PATH"] = str(rocm)
    env["PATH"] = f"{compiler_bin}{os.pathsep}{env.get('PATH', '')}"
    lib = rocm / "lib"
    llvm_lib = rocm / "lib" / "llvm" / "lib"
    extra_ld = []
    if lib.is_dir():
        extra_ld.append(str(lib))
    if llvm_lib.is_dir():
        extra_ld.append(str(llvm_lib))
    if extra_ld:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(extra_ld + ([prev] if prev else []))
    return env


def check_required_anywhere_lines(hip_file: Path) -> None:
    """Reject reductions that drop essential driver/CRC scaffolding lines."""
    raw_lines = hip_file.read_text(encoding="utf-8").split("\n")
    actual_lines = [line.strip() for line in raw_lines if line.strip()]
    spaceless_actual = [line.replace(" ", "") for line in actual_lines]
    for req_line in REQUIRED_ANYWHERE_LINES:
        if req_line.replace(" ", "") not in spaceless_actual:
            fail(f"missing required standalone line: {req_line}")


def stage_sources(hip_file: Path, work: Path, resource_dir: Path | None) -> Path:
    staged = work / hip_file.name
    shutil.copy2(hip_file, staged)
    search = [hip_file.parent]
    if resource_dir is not None:
        search.append(resource_dir)
    for name in COMPANIONS:
        dest = work / name
        if dest.is_file():
            continue
        for root in search:
            src = root / name
            if src.is_file():
                shutil.copy2(src, dest)
                break
    if not (work / "HIP-driver.cpp").is_file():
        die("HIP-driver.cpp not found next to hip_file or in --resource-dir")
    return staged


def run_timeout(
    timeout_bin: str,
    duration: str,
    argv: list[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [timeout_bin, "-s9", duration, *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def dump_proc(label: str, proc: subprocess.CompletedProcess[str]) -> None:
    chunks = [f"----- {label} exit {proc.returncode} -----"]
    if proc.stdout:
        chunks.append(f"----- {label} stdout -----")
        chunks.append(proc.stdout.rstrip("\n"))
    if proc.stderr:
        chunks.append(f"----- {label} stderr -----")
        chunks.append(proc.stderr.rstrip("\n"))
    text = "\n".join(chunks)
    if _log is not None:
        _log.write(text)
        if not _log.echo:
            sys.stdout.write(text + "\n")
            if proc.stderr:
                sys.stderr.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    sys.stderr.write("\n")
    else:
        sys.stdout.write(text + "\n")


def print_seq(report: dict[str, Any], which: str) -> list[tuple[int, str, int | None]]:
    seq: list[tuple[int, str, int | None]] = []
    for rec in report.get("prints") or []:
        if "id" not in rec or "value" not in rec:
            fail(f"{which} print is missing id or value")
        size = int(rec["sizeof"]) if "sizeof" in rec else None
        seq.append((int(rec["id"]), str(rec["value"]), size))
    return seq


def map_parallel(jobs: int, items: list[Any], fn: Any) -> list[Any]:
    workers = max(1, min(jobs, len(items)))
    if workers == 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def cmdline(argv: list[str]) -> str:
    return " ".join(argv)


def main() -> int:
    global _log
    args = parse_args()
    if args.jobs < 1:
        die("--jobs must be >= 1")

    hip_file = args.hip_file.resolve()
    if not hip_file.is_file():
        die(f"no such file: {hip_file}")
    check_required_anywhere_lines(hip_file)
    compiler = existing_exe(args.compiler)
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        die(f"compiler is not executable: {compiler}")
    crosscheck = args.crosscheck.resolve()
    if not crosscheck.is_file():
        die(f"no such file: {crosscheck}")
    rocgdb = existing_exe(args.rocgdb) if args.rocgdb else existing_exe(compiler.parent / "rocgdb")
    if not rocgdb.is_file() or not os.access(rocgdb, os.X_OK):
        die(f"rocgdb is not executable: {rocgdb}")
    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        die("timeout(1) not found on PATH")

    targeted = parse_ids(args.ids, initial_print_ids(crosscheck))
    resource_dir = args.resource_dir.resolve() if args.resource_dir else None
    env = toolchain_env(compiler)
    extra = list(args.extra)
    py = sys.executable

    keep = args.debug
    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if keep:
        work = Path(tempfile.mkdtemp(prefix="hipsmith-int-", dir="/tmp"))
        _log = Log(work / "interestingness.log", echo=True)
        _log.write(f"keeping artifacts in {work}")
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="hipsmith-int-")
        work = Path(tmp_ctx.name)
        _log = None

    try:
        staged_hip = stage_sources(hip_file, work, resource_dir)
        include_dirs = [str(work), *args.include_dirs]
        if resource_dir is not None:
            include_dirs.append(str(resource_dir))
        if _log is not None:
            _log.write(f"hip {hip_file}")
            _log.write(f"staged {staged_hip}")
            _log.write(f"compiler {compiler}")
            _log.write(f"rocgdb {rocgdb}")
            _log.write(f"crosscheck {crosscheck}")
            _log.write(f"ids {targeted}")
            _log.write(f"extra {extra}")
            _log.write(f"print-mode {args.print_mode}")
            _log.write(f"offload-arch {args.offload_arch}")

        wall_started = time.perf_counter()

        builds = [
            ("A", False, ["-O0"]),
            ("B", False, ["-O3"]),
            ("C", True, extra),
            (
                "D",
                False,
                ["-O0", "-fsanitize=undefined", "-fsanitize-trap=all"],
            ),
        ]

        def compile_one(spec: tuple[str, bool, list[str]]) -> tuple[str, Path, subprocess.CompletedProcess[str]]:
            name, with_g, flags = spec
            binary = work / name
            cmd = [
                py,
                str(HERE / "build_kernel.py"),
                str(staged_hip),
                "--compiler",
                str(compiler),
                "-o",
                str(binary),
            ]
            for inc in include_dirs:
                cmd.extend(["-I", inc])
            if with_g:
                cmd.append("--debug")
                if args.print_mode != "printf":
                    cmd.append(f"--print-{args.print_mode}")
            cmd.append("--")
            cmd.extend(COMMON_BUILD_FLAGS)
            cmd.extend(flags)
            cmd.append(f"--offload-arch={args.offload_arch}")
            wrapped = [timeout_bin, "-s9", args.compile_timeout, *cmd]
            if _log is not None:
                _log.write(f"compile {name}: {cmdline(wrapped)}")
            started = time.perf_counter()
            proc = run_timeout(timeout_bin, args.compile_timeout, cmd, env)
            elapsed = time.perf_counter() - started
            if _log is not None:
                _log.write(
                    f"compile {name} finished in {elapsed:.2f}s exit {proc.returncode}"
                )
            return name, binary, proc

        compiled = map_parallel(args.jobs, builds, compile_one)
        if _log is not None:
            _log.write(
                f"all compiles finished in {time.perf_counter() - wall_started:.2f}s wall"
            )
        for name, binary, proc in compiled:
            if keep or proc.returncode != 0:
                dump_proc(f"compile {name}", proc)
            if proc.returncode != 0:
                fail(f"compile {name} exited {proc.returncode}")
            if not binary.is_file():
                fail(f"compile {name} produced no binary")

        binaries = {name: binary for name, binary, _ in compiled}

        def run_one(name: str) -> tuple[str, Path, subprocess.CompletedProcess[str]]:
            report = work / f"{name}.json"
            cmd = [
                py,
                str(HERE / "run_kernel.py"),
                str(binaries[name]),
                "--timeout",
                args.run_timeout,
                "-o",
                str(report),
            ]
            if _log is not None:
                _log.write(f"run {name}: {cmdline(cmd)}")
            started = time.perf_counter()
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            elapsed = time.perf_counter() - started
            if _log is not None:
                _log.write(
                    f"run {name} finished in {elapsed:.2f}s exit {proc.returncode}"
                )
            return name, report, proc

        run_started = time.perf_counter()
        ran = map_parallel(args.jobs, ["A", "B", "C", "D"], run_one)
        if _log is not None:
            _log.write(
                f"all runs finished in {time.perf_counter() - run_started:.2f}s wall"
            )
        reports: dict[str, dict[str, Any]] = {}
        for name, report_path, proc in ran:
            if keep or proc.returncode != 0:
                dump_proc(f"run {name}", proc)
            if proc.returncode != 0:
                fail(f"run_kernel {name} exited {proc.returncode}")
            if not report_path.is_file():
                fail(f"run_kernel {name} wrote no JSON")
            report = load_json(report_path)
            exit_code = report.get("exit_code")
            if exit_code != 0:
                if not keep:
                    dump_proc(f"run {name}", proc)
                fail(f"kernel {name} exit_code {exit_code}")
            if report.get("crc") is None:
                fail(f"kernel {name} has no CRC")
            reports[name] = report
            if _log is not None:
                _log.write(f"kernel {name} crc={report['crc']} prints={len(report.get('prints') or [])}")

        crcs = {name: reports[name]["crc"] for name in ("A", "B", "C", "D")}
        if len(set(crcs.values())) != 1:
            fail(f"CRC mismatch: {crcs}")
        if _log is not None:
            _log.write(f"crc {crcs['A']}")

        seq_a = print_seq(reports["A"], "A")
        seq_b = print_seq(reports["B"], "B")
        if seq_a != seq_b:
            fail("A and B PRINT sequences differ")

        present = {item[0] for item in seq_a}
        missing = [i for i in targeted if i not in present]
        if missing:
            fail(f"targeted print ids missing from A: {missing}")

        gdb_json = work / "C.gdb.json"
        gdb_cmd = [
            py,
            str(HERE / "run-gdb.py"),
            str(binaries["C"]),
            str(staged_hip),
            "--rocgdb",
            str(rocgdb),
            "--timeout",
            args.gdb_timeout,
            "-o",
            str(gdb_json),
        ]
        if _log is not None:
            _log.write(f"run-gdb C: {cmdline(gdb_cmd)}")
        gdb_started = time.perf_counter()
        gdb_proc = subprocess.run(
            gdb_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if _log is not None:
            _log.write(
                f"run-gdb C finished in {time.perf_counter() - gdb_started:.2f}s "
                f"exit {gdb_proc.returncode}"
            )
        if keep or gdb_proc.returncode != 0:
            dump_proc("run-gdb C", gdb_proc)
        if gdb_proc.returncode != 0:
            fail(f"run-gdb C exited {gdb_proc.returncode}")
        if not gdb_json.is_file():
            fail("run-gdb C wrote no JSON")

        current = work / "current.crosscheck.json"
        unexpected = work / "gdb-only.json"
        cross_cmd = [
            py,
            str(HERE / "crosscheck-run-vs-gdb.py"),
            str(work / "A.json"),
            str(gdb_json),
            "-o",
            str(current),
            "--unexpected",
            str(unexpected),
        ]
        if _log is not None:
            _log.write(f"crosscheck: {cmdline(cross_cmd)}")
        cross_proc = subprocess.run(
            cross_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if keep or cross_proc.returncode != 0:
            dump_proc("crosscheck", cross_proc)
        if cross_proc.returncode != 0:
            fail(f"crosscheck exited {cross_proc.returncode}")

        test_cmd = [
            py,
            str(HERE / "test-run-vs-gdb.py"),
            str(crosscheck),
            str(current),
        ]
        if args.ids is not None:
            test_cmd.extend(["--ids", args.ids])
        if args.no_bit_equality:
            test_cmd.append("--no-bit-equality")
        if _log is not None:
            _log.write(f"test-run-vs-gdb: {cmdline(test_cmd)}")
        test_proc = subprocess.run(
            test_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if keep or test_proc.returncode != 0:
            dump_proc("test-run-vs-gdb", test_proc)
        elif test_proc.stdout or test_proc.stderr:
            if test_proc.stdout:
                sys.stdout.write(test_proc.stdout)
            if test_proc.stderr:
                sys.stderr.write(test_proc.stderr)
        if _log is not None:
            _log.write(f"exit {test_proc.returncode}")
            _log.write(f"total wall {time.perf_counter() - wall_started:.2f}s")
        raise SystemExit(test_proc.returncode)
    finally:
        if keep:
            if _log is not None:
                _log.write(f"artifacts kept in {work}")
                _log.close()
            print(f"debug: artifacts in {work}", file=sys.stderr)
        else:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()
            _log = None


if __name__ == "__main__":
    sys.exit(main())
