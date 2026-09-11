#!/usr/bin/env python3
"""Run the debug-info oracle over every iteration of a fuzzing campaign.

Classifies each `<run-dir>/out/fuzz-one.json` with di_oracle.py and reports how
many iterations were well defined, what failed the gate, and every finding.
Findings are JSONL so a campaign's worth of them stays greppable.

Findings are sites worth looking at, not verdicts on the compiler.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def load_oracle() -> Any:
    """Import di_oracle.py from beside this script."""
    path = Path(__file__).resolve().parent / "di_oracle.py"
    spec = importlib.util.spec_from_file_location("di_oracle", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load_oracle()


def find_reports(campaign: Path) -> list[Path]:
    """Every per-iteration report under a campaign directory, in run order."""
    if campaign.is_file():
        return [campaign]
    reports = sorted(campaign.glob("run-*/out/fuzz-one.json"))
    if not reports:
        reports = sorted(campaign.glob("**/fuzz-one.json"))
    return reports


def classify_one(path: Path) -> dict[str, Any]:
    try:
        verdict = oracle.classify(oracle.load_report(path))
    except oracle.OracleError as exc:
        return {"report": str(path), "unreadable": str(exc)}
    verdict["report"] = str(path)
    verdict["run"] = path.parent.parent.name
    return verdict


def gate_reason(verdict: dict[str, Any]) -> str:
    """Collapse a gate failure to a class, so the summary stays readable."""
    first = verdict["well_definedness_failures"][0]
    if first.startswith(("runs missing", "gdb probes missing")):
        return first.split(":")[0]
    for tail in ("run stage failed", "kernel exited", "gdb probe failed",
                 "PRINT values differ from target.printf"):
        if tail in first:
            return tail
    return first.split(":")[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the debug-info oracle to a whole fuzzing campaign and "
            "summarise the failures it classifies."
        )
    )
    parser.add_argument(
        "campaign",
        type=Path,
        help="campaign directory written by di-fuzz-campaign.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the campaign summary JSON here",
    )
    parser.add_argument(
        "--findings",
        type=Path,
        help="write every finding here as JSONL, one per line",
    )
    parser.add_argument(
        "--info",
        type=Path,
        help="write every info record here as JSONL (mirror cases, not failures)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="reports to classify concurrently (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.campaign.exists():
        print(f"error: no such path: {args.campaign}", file=sys.stderr)
        return 2
    reports = find_reports(args.campaign)
    if not reports:
        print(f"error: no fuzz-one.json under {args.campaign}", file=sys.stderr)
        return 2

    workers = max(1, min(args.jobs, len(reports)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(classify_one, reports))

    kinds: Counter[str] = Counter()
    gates: Counter[str] = Counter()
    runs_with_findings: set[str] = set()
    findings: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []
    well_defined = unreadable = 0

    for verdict in verdicts:
        if "unreadable" in verdict:
            unreadable += 1
            continue
        if not verdict["well_defined"]:
            gates[gate_reason(verdict)] += 1
            continue
        well_defined += 1
        run = verdict.get("run", verdict["report"])
        for record in verdict["findings"]:
            kinds[record["kind"]] += 1
            runs_with_findings.add(run)
            findings.append({"run": run, "seed": verdict["seed"],
                             "opt": verdict["opt"], **record})
        for record in verdict["info"]:
            info.append({"run": run, "seed": verdict["seed"],
                         "opt": verdict["opt"], **record})

    summary = {
        "campaign": str(args.campaign),
        "iterations": len(reports),
        "unreadable": unreadable,
        "well_defined": well_defined,
        "gate_failures": dict(gates.most_common()),
        "findings_total": len(findings),
        "runs_with_findings": len(runs_with_findings),
        "findings_by_kind": {kind: kinds[kind] for kind in oracle.FINDING_KINDS},
        "info_total": len(info),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for path, records in ((args.findings, findings), (args.info, info)):
        if path is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    print(f"campaign      {args.campaign}")
    print(f"iterations    {len(reports)}")
    print(f"well defined  {well_defined}"
          f"  ({well_defined / len(reports):.1%})")
    if unreadable:
        print(f"unreadable    {unreadable}")
    if gates:
        print("\nnot well defined")
        for reason, count in gates.most_common():
            print(f"  {count:6d}  {reason}")
    print(f"\nfindings      {len(findings)} "
          f"in {len(runs_with_findings)} of {well_defined} well defined runs")
    for kind in oracle.FINDING_KINDS:
        if kinds[kind]:
            print(f"  {kinds[kind]:6d}  {kind}")
    if info:
        print(f"\ninfo          {len(info)} (mirror cases, not failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
