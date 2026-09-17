#!/usr/bin/env python3
"""Binary-search device OptBisect for one oracle finding and emit a bucket key.

Buckets by LLVM pass name: ``{kind}|{pass}``. Opt-bisect runs only on the
HIPProg device LLVM compile step (amdgcn HIP cc1). The compile plan is
captured once with ``-###`` from a normal ``build_kernel.py``-style link;
Python relocates clang's planned temp paths into the trial directory and
replays those commands, injecting a bisect limit on that one device step.
Do not pass ``-mllvm -opt-bisect-limit`` on a normal HIP link: that fans
out across three independent counters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
RUN_GDB = SCRIPTS / "run-gdb.py"

BISECT_RE = re.compile(
    r"BISECT: running pass \((\d+)\) ([^ ]+) on (.*)$"
)

MANDATORY = "mandatory"
UNREPRODUCIBLE = "unreproducible"
FAILED = "failed"


def therock_root(amdclang: Path) -> Path:
    """Return TheRock install root (directory with ``bin/`` and ``lib/llvm/``)."""
    p = amdclang.absolute()
    if p.parent.name == "bin" and (p.parent.parent / "lib" / "llvm").is_dir():
        return p.parent.parent
    for i, name in enumerate(p.parts):
        if name == "lib" and i + 1 < len(p.parts) and p.parts[i + 1] == "llvm":
            return Path(*p.parts[:i])
    raise SystemExit(f"cannot infer TheRock root from compiler {amdclang}")


def compiler_driver(amdclang: Path) -> Path:
    """Use the HIP driver wrapper; do not resolve through to ``amdllvm``."""
    p = amdclang.absolute()
    if p.parent.name == "bin":
        return p
    root = therock_root(p)
    for name in ("amdclang++", "amdclang"):
        candidate = root / "bin" / name
        if candidate.is_file():
            return candidate
    return p


def load_script(filename: str) -> Any:
    path = SCRIPTS / filename
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oracle = load_script("di_oracle.py")
build_kernel = load_script("build_kernel.py")


@dataclass
class CompilePlan:
    """Captured ``-###`` commands for one HIPSmith kernel link."""

    template_dir: Path
    out_bin: Path
    cmds: list[list[str]]
    hipprog_device_idx: int


def load_report(path: Path) -> tuple[dict[str, Any], Path]:
    report_path = path
    if path.is_dir():
        for candidate in (path / "fuzz-one.json", path / "out" / "fuzz-one.json"):
            if candidate.is_file():
                report_path = candidate
                break
        else:
            raise SystemExit(f"no fuzz-one.json under {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    src_dir = source_dir_for_report(report_path, report)
    report["source_dir"] = str(src_dir)
    return report, src_dir


def source_dir_for_report(report_path: Path, report: dict[str, Any]) -> Path:
    """Return the ``out/`` directory that contains HIPProg.hip for this report."""
    if report_path.is_file() and report_path.parent.name == "out":
        candidate = report_path.parent
        if (candidate / "HIPProg.hip").is_file():
            return candidate.resolve()
    stored = report.get("source_dir")
    if stored:
        stored_path = Path(stored)
        if (stored_path / "HIPProg.hip").is_file():
            return stored_path.resolve()
    if report_path.is_dir():
        for candidate in (report_path / "out", report_path):
            if (candidate / "HIPProg.hip").is_file():
                return candidate.resolve()
    raise SystemExit(
        f"cannot locate HIPProg.hip for report {report_path} "
        f"(source_dir={stored!r})"
    )


def load_finding_line(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8").strip())


def parse_dry_run(amdclang: Path, argv: list[str]) -> list[list[str]]:
    proc = subprocess.run(
        [str(amdclang), "-###", *argv],
        capture_output=True,
        text=True,
    )
    cmds: list[list[str]] = []
    for line in proc.stderr.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        cmds.append(shlex.split(line))
    return cmds


def include_dirs_for_src(recorded_hip: Path, include_dirs: list[str], src_dir: Path) -> list[Path]:
    """Keep extra -I dirs; drop the recorded source dir (replaced by src_dir)."""
    recorded_parent = recorded_hip.resolve().parent
    out: list[Path] = []
    for inc in include_dirs:
        path = Path(inc)
        if path.resolve() == recorded_parent:
            continue
        if path.is_dir():
            out.append(path)
    return out


def options_from_recorded_cmd(build_cmd: list[str], src_dir: Path) -> dict[str, Any]:
    parsed = build_kernel.parse_build_argv(build_kernel.recorded_argv(build_cmd))
    hip_file = src_dir / "HIPProg.hip"
    if not hip_file.is_file():
        raise SystemExit(f"missing HIP source {hip_file}")
    driver = hip_file.with_name("HIP-driver.cpp")
    if not driver.is_file():
        raise SystemExit(f"missing HIP driver {driver}")
    return {
        "hip_file": hip_file,
        "driver": driver,
        "src_dir": src_dir,
        "include_dirs": include_dirs_for_src(
            parsed.hip_file, parsed.include_dirs or [], src_dir
        ),
        "extra": list(parsed.extra),
        "debug": parsed.debug,
        "print_noop": parsed.print_noop,
        "print_escape": parsed.print_escape,
    }


def compile_plan_argv(amdclang: Path, opts: dict[str, Any], out_bin: Path) -> list[str]:
    """Driver argv for a kernel link (no ``-save-temps``; temps come from ``-###``)."""
    return build_kernel.compile_argv(
        amdclang,
        opts["hip_file"],
        opts["include_dirs"],
        list(opts["extra"]),
        out_bin,
        debug=opts["debug"],
        print_noop=opts["print_noop"],
        print_escape=opts["print_escape"],
    )


def find_hipprog_device_opt(
    cmds: list[list[str]],
    main_name: str = "HIPProg.hip",
) -> int:
    """Index of device LLVM cc1 (amdgcn, ``-x hip``, ``-emit-llvm-bc``)."""
    want = Path(main_name).name
    for i, cmd in enumerate(cmds):
        if "-cc1" not in cmd or "-emit-llvm-bc" not in cmd:
            continue
        try:
            triple = cmd[cmd.index("-triple") + 1]
            name = cmd[cmd.index("-main-file-name") + 1]
            lang = cmd[cmd.index("-x") + 1]
        except ValueError:
            continue
        if "amdgcn" in triple and Path(name).name == want and lang == "hip":
            return i
    raise SystemExit(f"cannot find {want} device LLVM cc1 in -### output")


def capture_standalone_plan(
    amdclang: Path,
    source: Path,
    extra: list[str],
    template_dir: Path,
) -> CompilePlan:
    """Capture ``-###`` for a single HIP source (no HIP-driver.cpp)."""
    template_dir.mkdir(parents=True, exist_ok=True)
    out_bin = template_dir / "target.out"
    argv = [*extra, str(source.resolve()), "-o", str(out_bin)]
    cmds = parse_dry_run(amdclang, argv)
    if not cmds:
        raise SystemExit("driver -### produced no commands")
    hipprog_device_idx = find_hipprog_device_opt(cmds, source.name)
    return CompilePlan(template_dir, out_bin, cmds, hipprog_device_idx)


def pipeline_output_paths(cmds: list[list[str]]) -> list[str]:
    """Clang-planned intermediate and final ``-o`` / ``--image=file=`` paths."""
    paths: list[str] = []
    for cmd in cmds:
        i = 0
        while i < len(cmd):
            if cmd[i] == "-o" and i + 1 < len(cmd):
                paths.append(cmd[i + 1])
                i += 2
                continue
            if cmd[i].startswith("--image=file="):
                rest = cmd[i][len("--image=file=") :]
                paths.append(rest.split(",", 1)[0])
            i += 1
    return paths


def relocate_cmds(
    cmds: list[list[str]],
    dest: Path,
    template_dir: Path,
) -> list[list[str]]:
    """Rewrite ``-###`` temps into ``dest`` so jobs do not share ``/tmp`` names.

    Uses every pipeline ``-o`` / ``--image=file=`` path from ``cmds``, so later
    steps still see the relocated inputs. ``-dumpdir`` under the capture
    directory is rewritten the same way.
    """
    dest.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for old in pipeline_output_paths(cmds):
        mapping[old] = str(dest / Path(old).name)
    replacements = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    src_s = str(template_dir)
    dest_s = str(dest)

    def rewrite_arg(arg: str) -> str:
        for old, new in replacements:
            if old in arg:
                arg = arg.replace(old, new)
        # Skip if already under dest (dest may be nested under template_dir).
        if dest_s in arg:
            return arg
        if src_s in arg:
            arg = arg.replace(src_s, dest_s)
        return arg

    return [[rewrite_arg(a) for a in cmd] for cmd in cmds]


def capture_compile_plan(
    amdclang: Path,
    opts: dict[str, Any],
    template_dir: Path,
) -> CompilePlan:
    template_dir.mkdir(parents=True, exist_ok=True)
    out_bin = template_dir / "target.out"
    argv = compile_plan_argv(amdclang, opts, out_bin)
    cmds = parse_dry_run(amdclang, argv[1:])
    if not cmds:
        raise SystemExit("driver -### produced no commands")
    hipprog_device_idx = find_hipprog_device_opt(cmds)
    return CompilePlan(template_dir, out_bin, cmds, hipprog_device_idx)


def _set_opt_bisect(argv: list[str], limit: int) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(argv):
        if argv[i] == "-mllvm" and i + 1 < len(argv) and argv[i + 1].startswith(
            "-opt-bisect-limit="
        ):
            out.extend(["-mllvm", f"-opt-bisect-limit={limit}"])
            i += 2
            replaced = True
            continue
        out.append(argv[i])
        i += 1
    if not replaced:
        out.extend(["-mllvm", f"-opt-bisect-limit={limit}"])
    return out


def run_cmd(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def replay_compile(
    plan: CompilePlan,
    workdir: Path,
    timeout: float,
    bisect_limit: int | None = None,
    stop_before: int | None = None,
) -> tuple[bool, str]:
    """Replay captured commands; optionally bisect HIPProg device LLVM only."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_bin = workdir / "target.out"
    cmds = relocate_cmds(plan.cmds, workdir, plan.template_dir)
    end = stop_before if stop_before is not None else len(cmds)
    for i in range(end):
        cmd = cmds[i]
        if i == plan.hipprog_device_idx and bisect_limit is not None:
            cmd = _set_opt_bisect(cmd, bisect_limit)
        proc = run_cmd(cmd, timeout)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "command failed")[-2000:]
            return False, f"cmd[{i}] {Path(cmd[0]).name}: {tail}"
    if stop_before is not None:
        return True, ""
    if not out_bin.is_file():
        return False, f"missing {out_bin}"
    return True, ""


def list_bisect_passes(
    plan: CompilePlan,
    timeout: float,
) -> tuple[int, dict[int, tuple[str, str, str]]]:
    """Return (hi, index -> (pass, unit, raw_line)) for HIPProg device LLVM."""
    prep = plan.template_dir.parent / "_bisect_list"
    ok, err = replay_compile(
        plan,
        prep,
        timeout,
        stop_before=plan.hipprog_device_idx,
    )
    if not ok:
        raise SystemExit(f"opt-bisect listing prep failed:\n{err}")

    cmd = relocate_cmds(plan.cmds, prep, plan.template_dir)[
        plan.hipprog_device_idx
    ]
    cmd = _set_opt_bisect(list(cmd), -1)
    proc = run_cmd(cmd, timeout)
    if proc.returncode != 0:
        raise SystemExit(
            f"opt-bisect listing failed:\n{(proc.stderr or proc.stdout)[-3000:]}"
        )
    table: dict[int, tuple[str, str, str]] = {}
    hi = 0
    for line in (proc.stderr or "").splitlines():
        match = BISECT_RE.search(line)
        if not match:
            continue
        idx = int(match.group(1))
        hi = max(hi, idx)
        table[idx] = (match.group(2), match.group(3), line.strip())
    if hi == 0:
        raise SystemExit("no BISECT lines in HIPProg device LLVM stderr")
    return hi, table


def build_bisected_kernel(
    plan: CompilePlan,
    out_bin: Path,
    limit: int | None,
    timeout: float,
) -> tuple[bool, str]:
    workdir = out_bin.parent
    ok, err = replay_compile(plan, workdir, timeout, bisect_limit=limit)
    if not ok:
        return False, err
    if out_bin != workdir / "target.out":
        (workdir / "target.out").replace(out_bin)
    return True, ""


def printf_record(report: dict[str, Any], print_id: int) -> dict[str, Any]:
    runs = report.get("runs") or {}
    printf_run = (runs.get("target.printf") or {}).get("report") or {}
    for rec in printf_run.get("prints") or []:
        if int(rec.get("id", -1)) == print_id:
            return rec
    raise SystemExit(f"print id {print_id} not in target.printf")


def finding_suffix(kind: str) -> str:
    if kind == oracle.INCORRECTNESS:
        return ""
    if kind.startswith(oracle.WAY1 + ".") or kind.startswith(oracle.WAY2 + "."):
        return kind.split(".", 2)[2]
    raise SystemExit(f"unsupported finding kind: {kind}")


def finding_reproduces(
    finding: dict[str, Any],
    probe_rec: dict[str, Any] | None,
    printf_value: str,
    sizeof: int,
) -> bool:
    kind = finding["kind"]
    if kind == oracle.INCORRECTNESS:
        if oracle.missing_kind(probe_rec) is not None:
            return False
        return not oracle.values_match(
            printf_value,
            oracle.gdb_value(probe_rec.get("gdb_print")),
            sizeof,
        )

    suffix = finding_suffix(kind)
    if kind.startswith(oracle.WAY2):
        if suffix == oracle.MISSING_OPTIMIZED_OUT:
            return oracle.way2_missing(probe_rec) == oracle.MISSING_OPTIMIZED_OUT
        return oracle.missing_kind(probe_rec) == suffix

    if kind.startswith(oracle.WAY1):
        target_missing = oracle.missing_kind(probe_rec)
        if target_missing != suffix:
            return False
        ref_value = finding.get("reference_value")
        return ref_value is not None and oracle.values_match(
            printf_value, ref_value, sizeof
        )

    return False


def probe_gdb(
    binary: Path,
    hip_file: Path,
    rocgdb: Path,
    timeout: float,
    work: Path,
) -> dict[str, Any] | None:
    out_json = work / "gdb.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_GDB),
            str(binary),
            str(hip_file),
            "--rocgdb",
            str(rocgdb),
            "--timeout",
            str(int(timeout)),
            "-o",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        timeout=timeout * 2 + 60,
    )
    if proc.returncode != 0 or not out_json.is_file():
        return None
    report = json.loads(out_json.read_text(encoding="utf-8"))
    return report


def bisect_finding(
    finding: dict[str, Any],
    report: dict[str, Any],
    amdclang: Path,
    rocgdb: Path,
    build_timeout: float,
    gdb_timeout: float,
    keep: Path | None,
) -> dict[str, Any]:
    build_name = finding["build"]
    builds = report.get("builds") or {}
    if build_name not in builds:
        raise SystemExit(f"build {build_name!r} not in report")
    build_cmd = builds[build_name]["cmd"]
    src_dir = Path(report["source_dir"])
    opts = options_from_recorded_cmd(build_cmd, src_dir)

    print_id = int(finding["id"])
    printf_rec = printf_record(report, print_id)
    printf_value = str(printf_rec["value"])
    sizeof = int(printf_rec.get("sizeof") or 0)
    started = time.perf_counter()

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        result["duration_s"] = round(time.perf_counter() - started, 3)
        return result

    if "-O0" in opts["extra"]:
        return finish(
            _result(
                finding,
                key=f"{finding['kind']}|{MANDATORY}",
                pass_name=MANDATORY,
                limit=None,
                unit=None,
                bisect_line=None,
                n_trials=0,
                note="-O0 in build flags",
            )
        )

    def run_search(tmp_path: Path) -> dict[str, Any]:
        plan = capture_compile_plan(amdclang, opts, tmp_path / "plan")
        hi, table = list_bisect_passes(plan, build_timeout)
        n_trials = 1

        def trial(limit: int | None) -> tuple[bool, dict[str, Any] | None]:
            nonlocal n_trials
            n_trials += 1
            label = limit if limit is not None else "full"
            trial_dir = tmp_path / f"trial_{label}"
            bin_path = trial_dir / "target.out"
            ok, err = build_bisected_kernel(plan, bin_path, limit, build_timeout)
            if not ok:
                return False, {"build_error": err}
            gdb_report = probe_gdb(
                bin_path, opts["hip_file"], rocgdb, gdb_timeout, trial_dir
            )
            if gdb_report is None:
                return False, {"gdb_error": "probe failed"}
            probe_rec = gdb_report.get(str(print_id))
            repro = finding_reproduces(
                finding, probe_rec, printf_value, sizeof
            )
            return repro, probe_rec

        at_full, full_detail = trial(None)
        if not at_full:
            note = "finding absent at full device opt"
            if full_detail and full_detail.get("build_error"):
                return _result(
                    finding,
                    key=f"{finding['kind']}|{FAILED}",
                    pass_name=FAILED,
                    limit=None,
                    unit=None,
                    bisect_line=None,
                    n_trials=n_trials,
                    hi=hi,
                    note=full_detail["build_error"],
                )
            return _result(
                finding,
                key=f"{finding['kind']}|{UNREPRODUCIBLE}",
                pass_name=UNREPRODUCIBLE,
                limit=None,
                unit=None,
                bisect_line=None,
                n_trials=n_trials,
                hi=hi,
                note=note,
            )

        at_zero, _ = trial(0)
        if at_zero:
            return _result(
                finding,
                key=f"{finding['kind']}|{MANDATORY}",
                pass_name=MANDATORY,
                limit=0,
                unit=None,
                bisect_line=None,
                n_trials=n_trials,
                hi=hi,
            )

        lo, hi_search = 1, hi
        answer = hi_search
        while lo <= hi_search:
            mid = (lo + hi_search) // 2
            if trial(mid)[0]:
                answer = mid
                hi_search = mid - 1
            else:
                lo = mid + 1

        pass_name, unit, raw = table[answer]
        return _result(
            finding,
            key=f"{finding['kind']}|{pass_name}",
            pass_name=pass_name,
            limit=answer,
            unit=unit,
            bisect_line=raw,
            n_trials=n_trials,
            hi=hi,
        )

    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        return finish(run_search(keep))
    with tempfile.TemporaryDirectory(prefix="bisect-finding_") as tmp:
        return finish(run_search(Path(tmp)))


def _result(finding: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "finding": {
            "kind": finding.get("kind"),
            "build": finding.get("build"),
            "id": finding.get("id"),
            "expr": finding.get("expr"),
            "line": finding.get("line"),
        },
        **extra,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        help="fuzz-one.json, run directory, or a findings.jsonl line file",
    )
    parser.add_argument("--kind", help="finding kind (required unless --from-jsonl)")
    parser.add_argument("--id", type=int, help="PRINT id")
    parser.add_argument("--build", help="build name, e.g. target.escape")
    parser.add_argument("--amdclang", required=True, type=Path)
    parser.add_argument("--rocgdb", required=True, type=Path)
    parser.add_argument("--build-timeout", type=float, default=300.0)
    parser.add_argument("--gdb-timeout", type=float, default=120.0)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--keep-tmp",
        type=Path,
        help="keep build tree here instead of deleting",
    )
    parser.add_argument(
        "--from-jsonl",
        action="store_true",
        help="report path is one JSONL line with kind/id/build/run",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="run directory containing out/fuzz-one.json (with --from-jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    amdclang = compiler_driver(args.amdclang)
    rocgdb = args.rocgdb.absolute()

    if args.from_jsonl:
        finding = load_finding_line(args.report)
        if args.run_dir is not None:
            report, _src_dir = load_report(args.run_dir)
        elif finding.get("run"):
            run_dir = args.report.parent
            while run_dir != run_dir.parent:
                candidate = run_dir / finding["run"] / "out"
                if (candidate / "fuzz-one.json").is_file():
                    report, _src_dir = load_report(candidate)
                    break
                if run_dir.name == finding["run"] and (run_dir / "out" / "fuzz-one.json").is_file():
                    report, _src_dir = load_report(run_dir / "out")
                    break
                run_dir = run_dir.parent
            else:
                raise SystemExit(
                    f"cannot locate fuzz-one.json for run {finding['run']}; pass --run-dir"
                )
        else:
            raise SystemExit("--from-jsonl needs --run-dir or a finding row with run")
    else:
        if not args.kind or args.id is None or not args.build:
            raise SystemExit("need --kind, --id, and --build")
        report, _src_dir = load_report(args.report)
        finding = {
            "kind": args.kind,
            "id": args.id,
            "build": args.build,
        }
        gdb_report = ((report.get("gdb") or {}).get(args.build) or {}).get(
            "report"
        ) or {}
        probe_rec = gdb_report.get(str(args.id))
        if isinstance(probe_rec, dict):
            finding["expr"] = probe_rec.get("expr")
            finding["line"] = probe_rec.get("line")
        if args.kind.startswith(oracle.WAY1):
            raise SystemExit(
                "pass --from-jsonl with a findings.jsonl row for way1 "
                "(needs reference_value)"
            )

    result = bisect_finding(
        finding,
        report,
        amdclang,
        rocgdb,
        args.build_timeout,
        args.gdb_timeout,
        args.keep_tmp,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(result["key"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
