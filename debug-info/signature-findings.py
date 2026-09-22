#!/usr/bin/env python3
"""Map each oracle finding to a signature, then cluster duplicates.

An input finding is one HIP kernel, one PRINT id, one target build, and the
oracle's full kind (incorrectness, incompleteness.way2.optimized_out, …).
The signature is ``{coarse_kind}|{pass}``: incompleteness subclasses become
``missing``, and ``pass`` is the OptBisect result from bisect-finding.py.

Input is either one fuzz-one report / run directory, or an oracle
``findings.jsonl``. Output is every signed finding plus one representative per
signature cluster, with enough paths and flags to reduce that test later.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent

KIND_INCORRECTNESS = "incorrectness"
KIND_MISSING = "missing"


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


bisect = load_script("bisect-finding.py")
oracle = bisect.oracle


def coarse_kind(kind: str) -> str:
    """Map an oracle kind onto the two signature kinds."""
    if kind in (KIND_INCORRECTNESS, oracle.INCORRECTNESS):
        return KIND_INCORRECTNESS
    if kind == KIND_MISSING or kind.startswith("incompleteness."):
        return KIND_MISSING
    raise SystemExit(f"unsupported finding kind: {kind}")


def print_mode_for_build(build: str) -> str:
    if build.endswith(".escape"):
        return "escape"
    if build.endswith(".noop"):
        return "noop"
    return "printf"


def row_kind_ok(row_kind: str, want: str | None) -> bool:
    if want is None:
        return True
    if want in (KIND_MISSING, KIND_INCORRECTNESS):
        return coarse_kind(row_kind) == want
    return row_kind == want


def finding_key(row: dict[str, Any]) -> str:
    """Identity of one oracle row: kernel, PRINT id, build, full kind."""
    return f"{row['run']}|{int(row['id'])}|{row.get('build')}|{row['kind']}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: {exc}") from None
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    return rows


def looks_like_findings_jsonl(path: Path) -> bool:
    if path.suffix != ".jsonl" or not path.is_file():
        return False
    rows = load_jsonl(path)
    return bool(rows) and "kind" in rows[0] and "id" in rows[0]


def find_campaign_root(start: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        root = explicit.resolve()
        if not root.is_dir():
            raise SystemExit(f"no such campaign directory: {root}")
        return root
    here = start if start.is_dir() else start.parent
    for base in [here, *here.parents]:
        if any(base.glob("run-*/out/fuzz-one.json")):
            return base
    return None


def locate_report(run: str, campaign: Path | None, jsonl: Path | None) -> Path:
    candidates: list[Path] = []
    roots: list[Path] = []
    if campaign is not None:
        roots.append(campaign)
    if jsonl is not None:
        roots.extend([jsonl.parent, jsonl.parent.parent])
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        candidates.append(root / run / "out" / "fuzz-one.json")
        candidates.append(root / run / "fuzz-one.json")
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(
        f"cannot locate fuzz-one.json for {run}; pass --campaign"
    )


def offload_arch(report: dict[str, Any], extra: list[str]) -> str | None:
    for arg in extra:
        if arg.startswith("--offload-arch="):
            return arg.split("=", 1)[1]
        if arg == "--offload-arch":
            continue
    return (report.get("config") or {}).get("offload_arch")


def reduce_payload(
    finding: dict[str, Any],
    report: dict[str, Any],
    src_dir: Path,
) -> dict[str, Any]:
    """Paths and flags later needed to reduce this kernel for this PRINT id."""
    build_name = finding["build"]
    builds = report.get("builds") or {}
    if build_name not in builds:
        raise SystemExit(f"build {build_name!r} not in {src_dir}")
    opts = bisect.options_from_recorded_cmd(builds[build_name]["cmd"], src_dir)
    extra = list(opts["extra"])
    hip_file = src_dir / "HIPProg.hip"
    return {
        "hip_file": str(hip_file),
        "resource_dir": str(src_dir),
        "report": str(src_dir / "fuzz-one.json"),
        "ids": str(int(finding["id"])),
        "print_mode": print_mode_for_build(build_name),
        "build": build_name,
        "oracle_kind": finding["kind"],
        "offload_arch": offload_arch(report, extra),
        "extra": extra,
        "seed": finding.get("seed", report.get("seed")),
        "opt": finding.get("opt", (report.get("config") or {}).get("opt")),
        "expr": finding.get("expr"),
        "line": finding.get("line"),
        "how": finding.get("how"),
        "sizeof": finding.get("sizeof"),
        "printf_first": finding.get("printf_first"),
        "gdb_value": finding.get("gdb_value"),
        "reference_value": finding.get("reference_value"),
        "reference_build": finding.get("reference_build"),
        "hip_bytes": hip_file.stat().st_size if hip_file.is_file() else None,
    }


def work_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One bisect job per oracle row; coarse kind is only for the signature."""
    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "run": str(row["run"]),
                "id": int(row["id"]),
                "build": row.get("build"),
                "oracle_kind": row["kind"],
                "kind": coarse_kind(str(row["kind"])),
                "chosen": row,
            }
        )
    return items


def findings_from_report(
    report: dict[str, Any],
    run: str,
    kind: str | None,
    print_id: int | None,
    build: str | None,
) -> list[dict[str, Any]]:
    verdict = oracle.classify(report)
    if not verdict["well_defined"]:
        reason = (verdict.get("well_definedness_failures") or ["unknown"])[0]
        raise SystemExit(f"{run}: not well defined: {reason}")
    rows: list[dict[str, Any]] = []
    for record in verdict["findings"]:
        row = {
            "run": run,
            "seed": verdict.get("seed", report.get("seed")),
            "opt": verdict.get("opt", (report.get("config") or {}).get("opt")),
            **record,
        }
        if print_id is not None and int(row["id"]) != print_id:
            continue
        if not row_kind_ok(row["kind"], kind):
            continue
        if build is not None and row.get("build") != build:
            continue
        rows.append(row)
    if not rows:
        raise SystemExit(f"{run}: no matching oracle findings")
    return rows


def member_ref(record: dict[str, Any]) -> dict[str, Any]:
    reduce = record.get("reduce") or {}
    failure = record["failure"]
    return {
        "run": failure["run"],
        "id": failure["id"],
        "build": failure.get("build"),
        "oracle_kind": failure.get("oracle_kind"),
        "kind": failure.get("kind"),
        "hip_file": reduce.get("hip_file"),
        "resource_dir": reduce.get("resource_dir"),
        "print_mode": reduce.get("print_mode"),
        "expr": reduce.get("expr"),
        "line": reduce.get("line"),
        "opt": reduce.get("opt"),
        "pass": record.get("pass"),
        "limit": (record.get("bisect") or {}).get("limit"),
    }


def cluster_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    reduce = record.get("reduce") or {}
    size = reduce.get("hip_bytes")
    if size is None:
        size = 2**63
    return (size, record["failure"]["run"], record["failure"]["id"], record["failure"].get("build"))


def build_clusters(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for record in records:
        sig = record["signature"]
        if sig not in buckets:
            order.append(sig)
        buckets[sig].append(record)
    clusters: dict[str, Any] = {}
    for sig in order:
        members = buckets[sig]
        representative = min(members, key=cluster_rank)
        clusters[sig] = {
            "signature": sig,
            "kind": representative["failure"]["kind"],
            "pass": representative.get("pass"),
            "count": len(members),
            "representative": representative,
            "members": [member_ref(item) for item in members],
        }
    return clusters


def write_outputs(out_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = build_clusters(records)
    representatives = [cluster["representative"] for cluster in clusters.values()]
    by_kind: dict[str, int] = defaultdict(int)
    by_sig: dict[str, int] = defaultdict(int)
    for record in records:
        by_kind[record["failure"]["kind"]] += 1
        by_sig[record["signature"]] += 1
    summary = {
        "failures": len(records),
        "clusters": len(clusters),
        "by_kind": dict(by_kind),
        "by_signature": dict(by_sig),
    }
    (out_dir / "signatures.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (out_dir / "representatives.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in representatives),
        encoding="utf-8",
    )
    (out_dir / "clusters.json").write_text(
        json.dumps(clusters, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "clusters": clusters}


def load_completed(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "signatures.jsonl"
    if not path.is_file():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        failure = row.get("failure") or {}
        try:
            key = finding_key(
                {
                    "run": failure["run"],
                    "id": failure["id"],
                    "build": failure.get("build"),
                    "kind": failure.get("oracle_kind") or failure.get("kind"),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
        done[key] = row
    return done


def signature_of(kind: str, pass_name: str) -> str:
    return f"{kind}|{pass_name}"


def failed_record(group: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "failure": {
            "run": group["run"],
            "id": group["id"],
            "build": group["build"],
            "oracle_kind": group["oracle_kind"],
            "kind": group["kind"],
            "kernel": None,
        },
        "signature": signature_of(group["kind"], bisect.FAILED),
        "pass": bisect.FAILED,
        "bisect": {
            "note": str(exc) or f"{type(exc).__name__}: bisect failed"
        },
        "reduce": {},
    }


def sign_one(
    group: dict[str, Any],
    campaign: Path | None,
    jsonl: Path | None,
    amdclang: Path,
    rocgdb: Path,
    build_timeout: float,
    gdb_timeout: float,
    keep: Path | None,
) -> dict[str, Any]:
    try:
        return bisect_one(
            group,
            campaign,
            jsonl,
            amdclang,
            rocgdb,
            build_timeout,
            gdb_timeout,
            keep,
        )
    except (SystemExit, Exception) as exc:
        return failed_record(group, exc)


def bisect_one(
    group: dict[str, Any],
    campaign: Path | None,
    jsonl: Path | None,
    amdclang: Path,
    rocgdb: Path,
    build_timeout: float,
    gdb_timeout: float,
    keep: Path | None,
) -> dict[str, Any]:
    run = group["run"]
    report_path = group.get("report_path")
    if report_path is None:
        report_path = locate_report(run, campaign, jsonl)
    else:
        report_path = Path(report_path)
    report, src_dir = bisect.load_report(report_path)
    finding = dict(group["chosen"])
    finding.setdefault("run", run)
    reduce = reduce_payload(finding, report, src_dir)
    keep_dir = None
    if keep is not None:
        kind_slug = str(group["oracle_kind"]).replace(".", "-")
        keep_dir = keep / f"{run}-id{group['id']}-{group['build']}-{kind_slug}"
    result = bisect.bisect_finding(
        finding,
        report,
        amdclang,
        rocgdb,
        build_timeout,
        gdb_timeout,
        keep_dir,
    )
    pass_name = result.get("pass_name") or bisect.FAILED
    kind = group["kind"]
    return {
        "failure": {
            "run": run,
            "id": group["id"],
            "build": group["build"],
            "oracle_kind": group["oracle_kind"],
            "kind": kind,
            "kernel": reduce["hip_file"],
        },
        "signature": signature_of(kind, pass_name),
        "pass": pass_name,
        "bisect": {
            "key": result.get("key"),
            "limit": result.get("limit"),
            "unit": result.get("unit"),
            "bisect_line": result.get("bisect_line"),
            "n_trials": result.get("n_trials"),
            "hi": result.get("hi"),
            "duration_s": result.get("duration_s"),
            "note": result.get("note"),
        },
        "reduce": reduce,
    }


def print_progress(index: int, total: int, record: dict[str, Any]) -> None:
    bisect_info = record.get("bisect") or {}
    duration = bisect_info.get("duration_s")
    extra = f" {duration}s" if duration is not None else ""
    print(
        f"[{index}/{total}] {record['failure']['run']} "
        f"{record['failure'].get('build')} id={record['failure']['id']} "
        f"{record['failure'].get('oracle_kind')} "
        f"-> {record['signature']}{extra}",
        file=sys.stderr,
    )


def print_summary(summary: dict[str, Any], clusters: dict[str, Any]) -> None:
    print(
        f"failures {summary['failures']}  clusters {summary['clusters']}",
        file=sys.stderr,
    )
    for kind, count in (summary.get("by_kind") or {}).items():
        print(f"  {count:6d}  {kind}", file=sys.stderr)
    print("clusters", file=sys.stderr)
    for sig, cluster in clusters.items():
        reduce = cluster["representative"].get("reduce") or {}
        hip = reduce.get("hip_file") or "-"
        print(
            f"  {cluster['count']:6d}  {sig}  "
            f"rep={cluster['representative']['failure']['run']} "
            f"{cluster['representative']['failure'].get('build')} "
            f"id={cluster['representative']['failure']['id']}  {hip}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="fuzz-one.json, a run directory, or oracle findings.jsonl",
    )
    parser.add_argument("--amdclang", type=Path)
    parser.add_argument("--rocgdb", type=Path)
    parser.add_argument(
        "--campaign",
        type=Path,
        help="campaign root with run-*/out/fuzz-one.json (jsonl input)",
    )
    parser.add_argument("--kind", help="oracle kind or incorrectness/missing")
    parser.add_argument("--id", type=int, help="PRINT id")
    parser.add_argument("--build", help="restrict to this target build")
    parser.add_argument("--build-timeout", type=float, default=45.0)
    parser.add_argument("--gdb-timeout", type=float, default=30.0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=30,
        help="findings to bisect in parallel (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="directory for signatures.jsonl, clusters.json, representatives.jsonl",
    )
    parser.add_argument(
        "--keep-tmp",
        type=Path,
        help="keep per-failure bisect trees here",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="bisect at most this many oracle findings",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip failures already present in -o/signatures.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list each oracle finding and its coarse kind without OptBisect",
    )
    return parser.parse_args()


def collect_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Path | None, Path | None, Path | None]:
    path = args.input.resolve()
    if path.is_file() and looks_like_findings_jsonl(path):
        rows = load_jsonl(path)
        if args.id is not None:
            rows = [row for row in rows if int(row["id"]) == args.id]
        if args.kind is not None:
            rows = [row for row in rows if row_kind_ok(row["kind"], args.kind)]
        if args.build is not None:
            rows = [row for row in rows if row.get("build") == args.build]
        if not rows:
            raise SystemExit(f"no matching findings in {path}")
        campaign = find_campaign_root(path, args.campaign)
        return rows, campaign, path, None

    report, src_dir = bisect.load_report(path)
    if src_dir.name == "out":
        run = src_dir.parent.name
    else:
        run = src_dir.name
    rows = findings_from_report(report, run, args.kind, args.id, args.build)
    campaign = find_campaign_root(src_dir, args.campaign)
    return rows, campaign, None, src_dir / "fuzz-one.json"


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        if args.amdclang is None or args.rocgdb is None:
            raise SystemExit("--amdclang and --rocgdb are required unless --dry-run")
    rows, campaign, jsonl, report_path = collect_rows(args)
    groups = work_items(rows)
    if report_path is not None:
        for group in groups:
            group["report_path"] = str(report_path)
    if args.limit is not None:
        groups = groups[: max(0, args.limit)]
    print(
        f"findings {len(groups)}  jobs {max(1, args.jobs)}",
        file=sys.stderr,
    )
    if args.dry_run:
        for group in groups:
            chosen = group["chosen"]
            print(
                json.dumps(
                    {
                        "run": group["run"],
                        "id": group["id"],
                        "build": group["build"],
                        "oracle_kind": group["oracle_kind"],
                        "kind": group["kind"],
                        "expr": chosen.get("expr"),
                        "line": chosen.get("line"),
                    }
                )
            )
        return 0

    if args.resume and args.output is None:
        raise SystemExit("--resume requires --output")

    amdclang = bisect.compiler_driver(args.amdclang)
    rocgdb = args.rocgdb.absolute()
    completed = load_completed(args.output) if args.resume and args.output else {}
    total = len(groups)
    records_by_key: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for group in groups:
        key = finding_key(group["chosen"])
        if key in completed:
            records_by_key[key] = completed[key]
            print_progress(len(records_by_key), total, completed[key])
            print("  resumed", file=sys.stderr)
        else:
            pending.append(group)

    lock = threading.Lock()

    def ordered_records() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for group in groups:
            record = records_by_key.get(finding_key(group["chosen"]))
            if record is not None:
                out.append(record)
        return out

    def persist() -> list[dict[str, Any]]:
        records = ordered_records()
        if args.output is not None:
            write_outputs(args.output, records)
        return records

    try:
        if pending:
            workers = max(1, args.jobs)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        sign_one,
                        group,
                        campaign,
                        jsonl,
                        amdclang,
                        rocgdb,
                        args.build_timeout,
                        args.gdb_timeout,
                        args.keep_tmp,
                    ): group
                    for group in pending
                }
                for fut in as_completed(futures):
                    group = futures[fut]
                    record = fut.result()
                    with lock:
                        records_by_key[finding_key(group["chosen"])] = record
                        print_progress(len(records_by_key), total, record)
                        persist()
    except KeyboardInterrupt:
        print("interrupted; writing partial results", file=sys.stderr)
        records = persist()
        return 130

    records = persist() if args.output is not None else ordered_records()
    if args.output is not None:
        payload = write_outputs(args.output, records)
    else:
        clusters = build_clusters(records)
        payload = {
            "summary": {
                "failures": len(records),
                "clusters": len(clusters),
                "by_kind": {
                    kind: sum(
                        1
                        for record in records
                        if record["failure"]["kind"] == kind
                    )
                    for kind in (KIND_INCORRECTNESS, KIND_MISSING)
                    if any(record["failure"]["kind"] == kind for record in records)
                },
            },
            "clusters": clusters,
        }
        sys.stdout.write(
            json.dumps(
                {
                    "summary": payload["summary"],
                    "failures": records,
                    "clusters": payload["clusters"],
                    "representatives": [
                        cluster["representative"]
                        for cluster in payload["clusters"].values()
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    print_summary(payload["summary"], payload["clusters"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
