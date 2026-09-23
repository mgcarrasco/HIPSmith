#!/usr/bin/env python3
"""Drive cvise-run-vs-gdb.py over representatives.jsonl.

Rebuilds the initial crosscheck for each selected finding, aborts that instance
if the targeted id does not still match oracle_kind, then runs C-Vise in an
instance directory. Each HIP file is copied and has HIPSmith.h and
safe_math_macros.h inlined first; the campaign original is never overwritten.
C-Vise reduces only that prepared HIP file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MAKE_INITIAL = HERE / "make-initial-crosscheck.py"
CVISE_RUN = HERE / "cvise-run-vs-gdb.py"
INTERESTINGNESS = HERE / "run-vs-gdb-interestingness.py"
PREPARE = HERE / "prepare-kernel.py"
DEFAULT_RUNTIME = HERE.parent.parent / "HIPSmith" / "runtime"


def load_sibling(filename: str) -> Any:
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").removesuffix(".py"), path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predicate = load_sibling("test-run-vs-gdb.py")
prepare_kernel = load_sibling("prepare-kernel.py")

PRINT_LOCK = threading.Lock()


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce signature representatives with C-Vise. Extra compiler "
            "flags come from each row's reduce.extra."
        )
    )
    parser.add_argument(
        "--representatives",
        type=Path,
        required=True,
        help="signatures/representatives.jsonl",
    )
    parser.add_argument(
        "--rocm-dir",
        type=Path,
        help="ROCm tree; uses <dir>/bin/amdclang++ and <dir>/bin/rocgdb",
    )
    parser.add_argument("--compiler", type=Path, help="amdclang++")
    parser.add_argument("--rocgdb", type=Path, help="rocgdb")
    parser.add_argument(
        "--cvise",
        type=Path,
        required=True,
        help="cvise checkout directory or executable",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory for per-instance work and results",
    )
    parser.add_argument("--limit", type=int, metavar="N")
    parser.add_argument("--signature", metavar="SIG")
    parser.add_argument(
        "--kind",
        metavar="KIND",
        help="oracle_kind equals KIND, or starts with KIND.",
    )
    parser.add_argument(
        "--parallel-instances",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent C-Vise processes (default: 1)",
    )
    parser.add_argument(
        "--cvise-n",
        type=int,
        metavar="N",
        help="Forwarded as cvise-run-vs-gdb.py -n",
    )
    parser.add_argument(
        "--test-jobs",
        type=int,
        metavar="N",
        help="Forwarded as cvise-run-vs-gdb.py --jobs",
    )
    parser.add_argument("--max-time", type=float, metavar="SECONDS")
    parser.add_argument(
        "--pass-group-file",
        type=Path,
        help="JSON pass group forwarded as cvise --pass-group-file",
    )
    parser.add_argument("--stopping-threshold", type=float, metavar="F")
    parser.add_argument("--compile-timeout", metavar="DURATION")
    parser.add_argument("--gdb-timeout", metavar="DURATION")
    parser.add_argument("--run-timeout", metavar="DURATION")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Prepare, rebuild INITIAL, and run interestingness once; do not start C-Vise",
    )
    args = parser.parse_args()
    if args.rocm_dir is not None:
        args.compiler = args.rocm_dir / "bin" / "amdclang++"
        args.rocgdb = args.rocm_dir / "bin" / "rocgdb"
    if args.compiler is None or args.rocgdb is None:
        parser.error("need --rocm-dir or both --compiler and --rocgdb")
    if args.parallel_instances < 1:
        parser.error("--parallel-instances must be >= 1")
    if args.pass_group_file is not None:
        path = args.pass_group_file.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            parser.error(f"no such --pass-group-file: {path}")
        args.pass_group_file = path.resolve()
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        die(f"no such file: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            die(f"{path}:{lineno}: {exc}")
    return rows


def kind_matches(oracle_kind: str, wanted: str) -> bool:
    return oracle_kind == wanted or oracle_kind.startswith(wanted + ".")


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if args.signature is not None and row.get("signature") != args.signature:
            continue
        kind = str((row.get("reduce") or {}).get("oracle_kind") or "")
        if args.kind is not None and not kind_matches(kind, args.kind):
            continue
        selected.append(row)
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def instance_name(row: dict[str, Any]) -> str:
    sig = str(row.get("signature") or "unsigned").replace("|", "-")
    failure = row.get("failure") or {}
    run = str(failure.get("run") or "run-unknown")
    print_id = failure.get("id")
    return f"{sig}__{run}__id{print_id}"


def parse_ids(text: str) -> list[int]:
    return [int(part, 10) for part in text.split(",") if part.strip()]


def row_by_id(rows: list[dict[str, Any]], print_id: int) -> dict[str, Any] | None:
    for rec in rows:
        if int(rec.get("print_id", -1)) == print_id:
            return rec
    return None


def unsound_reason(kind: str, rec: dict[str, Any], print_id: int) -> str | None:
    """None if this crosscheck row still matches oracle_kind."""
    if kind == "incorrectness":
        if not predicate.is_concrete(rec.get("gdb_value")):
            return f"id {print_id}: gdb_value is not concrete"
        if predicate.values_match(
            rec.get("run_value"), rec.get("gdb_value"),
            rec, print_id, "initial", True,
        ):
            return f"id {print_id}: gdb_value bit-matches run_value"
        return None

    if kind == "incompleteness.way2" or kind.startswith("incompleteness.way2."):
        if rec.get("located_on_line") is not False:
            return (
                f"id {print_id}: located_on_line is "
                f"{rec.get('located_on_line')!r}, want False"
            )
        return None

    if kind == "incompleteness.way1" or kind.startswith("incompleteness.way1."):
        if not predicate.values_match(
            rec.get("run_value"), rec.get("reference_value"),
            rec, print_id, "initial reference", True,
        ):
            return (
                f"id {print_id}: reference_value "
                f"{rec.get('reference_value')!r} does not bit-match run "
                f"{rec.get('run_value')!r}"
            )
        if predicate.is_concrete(rec.get("gdb_value")):
            return f"id {print_id}: target gdb_value is concrete"
        return None

    return f"id {print_id}: unhandled oracle_kind {kind}"


def assert_sound(crosscheck: Path, kind: str, ids: list[int]) -> str | None:
    payload = json.loads(crosscheck.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return f"{crosscheck} is not a JSON array"
    for print_id in ids:
        rec = row_by_id(payload, print_id)
        if rec is None:
            return f"{crosscheck}: targeted id {print_id} is missing"
        reason = unsound_reason(kind, rec, print_id)
        if reason is not None:
            return f"finding did not reproduce: {reason}"
    return None


def byte_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def file_sizes(prepared_hip: Path, instance: Path) -> dict[str, Any]:
    original = byte_size(prepared_hip)
    final = byte_size(instance / "HIPProg.hip")
    return {
        "HIPProg.hip": {
            "original": original,
            "final": final,
            "delta": (
                None if original is None or final is None else final - original
            ),
        }
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_to_log(cmd: list[str], log: Path, cwd: Path | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        fh.write(b"$ " + " ".join(cmd).encode("utf-8", "replace") + b"\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"\nexit {proc.returncode}\n".encode("utf-8"))
    return proc.returncode


def killpg(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def reduce_one(row: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    reduce = row.get("reduce") or {}
    kind = str(reduce.get("oracle_kind") or "")
    name = instance_name(row)
    instance = out_dir / name
    instance.mkdir(parents=True, exist_ok=True)
    campaign_hip = Path(reduce["hip_file"])
    resource_dir = Path(reduce["resource_dir"])
    prepared_hip = instance / "original" / "HIPProg.hip"
    extra = list(reduce.get("extra") or [])
    offload_arch = reduce.get("offload_arch")
    print_mode = reduce.get("print_mode")
    ids_text = str(reduce.get("ids") or "")
    need_reference = kind.startswith("incompleteness.way1")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "instance": name,
        "oracle_kind": kind,
        "signature": row.get("signature"),
        "ids": ids_text,
        "ok": False,
        "timed_out": False,
        "stage": "setup",
        "returncode": None,
        "error": None,
        "campaign_hip": str(campaign_hip),
    }

    def finish(error: str | None = None, **fields: Any) -> dict[str, Any]:
        result.update(fields)
        if error is not None:
            result["error"] = error
        result["elapsed_s"] = round(time.perf_counter() - started, 3)
        result["files"] = file_sizes(prepared_hip, instance)
        write_result(instance / "result.json", result)
        with PRINT_LOCK:
            status = (
                "timeout" if result.get("timed_out")
                else "ok" if result.get("ok")
                else "fail"
            )
            print(
                f"{status:7} {name}  {kind}  "
                f"{result['elapsed_s']:.1f}s"
                + (f"  {error}" if error else ""),
                flush=True,
            )
        return result

    if not campaign_hip.is_file():
        return finish(f"no such hip file: {campaign_hip}")
    if not resource_dir.is_dir():
        return finish(f"no such resource-dir: {resource_dir}")
    if not extra:
        return finish("reduce.extra is empty")
    if not offload_arch:
        return finish("reduce.offload_arch is missing")
    if print_mode not in ("printf", "noop", "escape"):
        return finish(f"bad print-mode: {print_mode!r}")

    try:
        ids = parse_ids(ids_text)
    except ValueError:
        return finish(f"bad reduce.ids: {ids_text!r}")
    if not ids:
        return finish("reduce.ids is empty")
    compiler = args.compiler
    rocgdb = args.rocgdb
    py = sys.executable or "python3"
    crosscheck = instance / "HIPProg.crosscheck.initial.json"

    result["stage"] = "prepare"
    try:
        prepare_kernel.prepare_hip_file(campaign_hip, prepared_hip, resource_dir)
    except prepare_kernel.PrepareError as exc:
        return finish(f"prepare-kernel: {exc}", returncode=2)
    driver_src = resource_dir / "HIP-driver.cpp"
    if not driver_src.is_file():
        return finish(f"no HIP-driver.cpp in {resource_dir}")
    shutil.copy2(driver_src, prepared_hip.with_name("HIP-driver.cpp"))
    for companion in ("HIPSmith.h", "HIPSmithPrint.h"):
        src = resource_dir / companion
        if src.is_file():
            shutil.copy2(src, prepared_hip.with_name(companion))
    result["prepared_hip"] = str(prepared_hip)
    result["prepared_bytes"] = byte_size(prepared_hip)

    initial_cmd = [
        py, str(MAKE_INITIAL), str(prepared_hip),
        "--compiler", str(compiler),
        "--rocgdb", str(rocgdb),
        "--offload-arch", str(offload_arch),
        "--print-mode", str(print_mode),
        "-o", str(crosscheck),
        "--work-dir", str(instance / "build"),
    ]
    if args.run_timeout is not None:
        initial_cmd.extend(["--run-timeout", args.run_timeout])
    if args.gdb_timeout is not None:
        initial_cmd.extend(["--gdb-timeout", args.gdb_timeout])
    if need_reference:
        initial_cmd.append("--reference")
    initial_cmd.append("--")
    initial_cmd.extend(extra)

    result["stage"] = "make-initial-crosscheck"
    code = run_to_log(initial_cmd, instance / "make-initial.log")
    if code != 0:
        return finish(
            f"make-initial-crosscheck exited {code}",
            returncode=code,
        )
    sound_error = assert_sound(crosscheck, kind, ids)
    if sound_error is not None:
        return finish(sound_error, returncode=2)

    if args.check_only:
        compile_timeout = args.compile_timeout or "30"
        run_timeout = args.run_timeout or "5"
        gdb_timeout = args.gdb_timeout or "5"
        jobs = args.test_jobs if args.test_jobs is not None else 5
        int_cmd = [
            py, str(INTERESTINGNESS), str(prepared_hip),
            "--compiler", str(compiler),
            "--rocgdb", str(rocgdb),
            "--compile-timeout", compile_timeout,
            "--run-timeout", run_timeout,
            "--gdb-timeout", gdb_timeout,
            "--jobs", str(jobs),
            "--crosscheck", str(crosscheck),
            "--resource-dir", str(prepared_hip.parent),
            "--offload-arch", str(offload_arch),
            "--ids", ids_text,
            "--print-mode", str(print_mode),
            "--original-hip", str(prepared_hip),
        ]
        if DEFAULT_RUNTIME.is_dir():
            int_cmd.extend(["-I", str(DEFAULT_RUNTIME)])
        if need_reference:
            int_cmd.append("--reference")
        int_cmd.append("--")
        int_cmd.extend(extra)
        result["stage"] = "interestingness"
        code = run_to_log(int_cmd, instance / "interestingness.log")
        shutil.copy2(prepared_hip, instance / "HIPProg.hip")
        return finish(
            None if code == 0 else f"interestingness exited {code}",
            ok=code == 0,
            returncode=code,
            stage="done" if code == 0 else "interestingness",
        )

    cvise_cmd = [
        py, str(CVISE_RUN), str(prepared_hip),
        "--compiler", str(compiler),
        "--rocgdb", str(rocgdb),
        "--offload-arch", str(offload_arch),
        "--print-mode", str(print_mode),
        "--cvise", str(args.cvise),
        "--resource-dir", str(prepared_hip.parent),
        "--crosscheck", str(crosscheck),
        "--work-dir", str(instance),
        "--ids", ids_text,
        f"--cvise-arg=--log-file={instance / 'cvise.log'}",
    ]
    if args.cvise_n is not None:
        cvise_cmd.extend(["-n", str(args.cvise_n)])
    if args.test_jobs is not None:
        cvise_cmd.extend(["--jobs", str(args.test_jobs)])
    if args.compile_timeout is not None:
        cvise_cmd.extend(["--compile-timeout", args.compile_timeout])
    if args.run_timeout is not None:
        cvise_cmd.extend(["--run-timeout", args.run_timeout])
    if args.gdb_timeout is not None:
        cvise_cmd.extend(["--gdb-timeout", args.gdb_timeout])
    if args.stopping_threshold is not None:
        cvise_cmd.append(
            f"--cvise-arg=--stopping-threshold={args.stopping_threshold}"
        )
    if args.pass_group_file is not None:
        cvise_cmd.extend(["--pass-group-file", str(args.pass_group_file)])
    if need_reference:
        cvise_cmd.append("--reference")
    cvise_cmd.append("--")
    cvise_cmd.extend(extra)

    result["stage"] = "cvise"
    timed_out = False
    with (instance / "cvise.stdout.log").open("wb") as log:
        log.write(b"$ " + " ".join(cvise_cmd).encode("utf-8", "replace") + b"\n")
        log.flush()
        proc = subprocess.Popen(
            cvise_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if args.max_time is not None:
                try:
                    code = proc.wait(timeout=args.max_time)
                except subprocess.TimeoutExpired:
                    killpg(proc)
                    timed_out = True
                    code = proc.returncode if proc.returncode is not None else -9
            else:
                code = proc.wait()
        except KeyboardInterrupt:
            killpg(proc)
            raise
        log.write(f"\nexit {code}\n".encode("utf-8"))

    return finish(
        None if code == 0 and not timed_out else (
            f"timed out after {args.max_time}s" if timed_out
            else f"cvise-run-vs-gdb exited {code}"
        ),
        ok=code == 0 and not timed_out,
        timed_out=timed_out,
        returncode=code,
        stage="done" if code == 0 and not timed_out else "cvise",
    )


def format_size(n: int | None) -> str:
    if n is None:
        return "?"
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def print_table(results: list[dict[str, Any]]) -> None:
    print()
    print(f"{'status':7} {'elapsed':>8}  {'HIPProg.hip':>18}  instance")
    for rec in results:
        files = rec.get("files") or {}
        hip = files.get("HIPProg.hip") or {}
        orig = format_size(hip.get("original"))
        final = format_size(hip.get("final"))
        status = (
            "timeout" if rec.get("timed_out")
            else "ok" if rec.get("ok")
            else "fail"
        )
        print(
            f"{status:7} {rec.get('elapsed_s', 0):7.1f}s  "
            f"{orig:>8}->{final:<8}  {rec.get('instance')}"
        )


def main() -> int:
    args = parse_args()
    if not MAKE_INITIAL.is_file() or not CVISE_RUN.is_file():
        die("missing make-initial-crosscheck.py or cvise-run-vs-gdb.py")
    if not PREPARE.is_file() or not INTERESTINGNESS.is_file():
        die("missing prepare-kernel.py or run-vs-gdb-interestingness.py")
    compiler = args.compiler.expanduser()
    rocgdb = args.rocgdb.expanduser()
    if not compiler.is_file():
        die(f"no such compiler: {compiler}")
    if not rocgdb.is_file():
        die(f"no such rocgdb: {rocgdb}")
    args.compiler = compiler
    args.rocgdb = rocgdb

    rows = select_rows(load_jsonl(args.representatives), args)
    if not rows:
        die("no representatives matched")
    out_dir = args.output.expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pass_group_file is not None:
        shutil.copy2(args.pass_group_file, out_dir / args.pass_group_file.name)
        print(f"pass group: {args.pass_group_file}", flush=True)
    print(f"selected {len(rows)} representative(s) -> {out_dir}", flush=True)
    results: list[dict[str, Any]] = []
    workers = min(args.parallel_instances, len(rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(reduce_one, row, args, out_dir) for row in rows]
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda rec: str(rec.get("instance")))
    print_table(results)
    if any(not rec.get("ok") for rec in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
