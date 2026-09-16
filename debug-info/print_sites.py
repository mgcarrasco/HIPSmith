"""Parse HIPSmith PRINT_* call sites from HIP source.

A site is PRINT_<KIND>(expr, __LINE__, "how", id). Whitespace inside the
call is ignored when comparing sites; KIND, expr, how, and id must match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# INT64/UINT64/INT16/... before INT/UINT so PRINT_INT64 is not PRINT_INT.
PRINT_SITE_RE = re.compile(
    r"(?<![A-Za-z0-9_])PRINT_"
    r"(?P<kind>INT8|UINT8|INT16|UINT16|INT64|UINT64|INT|UINT)\s*\("
    r"(?P<expr>.*?),\s*__LINE__\s*,\s*"
    r"\"(?P<how>[^\"]*)\""
    r"(?:\s*,\s*(?P<id>-?\d+))?"
    r"\s*\)",
    re.DOTALL,
)


def parse_print_sites_text(text: str) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for match in PRINT_SITE_RE.finditer(text):
        id_text = match.group("id")
        sites.append(
            {
                "kind": match.group("kind"),
                "expr": match.group("expr").strip(),
                "how": match.group("how"),
                "id": int(id_text) if id_text is not None else None,
                "line": text.count("\n", 0, match.start()) + 1,
            }
        )
    return sites


def parse_print_sites(path: Path) -> list[dict[str, Any]]:
    """Parse a file; exit if any PRINT_* is missing an id or ids repeat.

    Same contract run-gdb.py has always used.
    """
    sites = parse_print_sites_text(
        path.read_text(encoding="utf-8", errors="replace")
    )
    missing_id = sum(1 for site in sites if site["id"] is None)
    if missing_id:
        raise SystemExit(
            f"error: {missing_id} PRINT_* macros in {path} have no id; "
            'need PRINT_TYPE(expr, __LINE__, "how", id)'
        )
    ids = [site["id"] for site in sites]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"error: duplicate PRINT ids in {path}")
    return sites


def normalized_print(site: dict[str, Any]) -> tuple[str, str, str, int]:
    if site["id"] is None:
        raise ValueError("PRINT site has no id")
    return (
        site["kind"],
        "".join(site["expr"].split()),
        "".join(site["how"].split()),
        site["id"],
    )


def format_print(site: dict[str, Any]) -> str:
    id_part = "missing" if site["id"] is None else str(site["id"])
    return (
        f'PRINT_{site["kind"]}({site["expr"]}, __LINE__, '
        f'"{site["how"]}", {id_part})'
    )


def sites_by_id(sites: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for site in sites:
        if site["id"] is None:
            continue
        if site["id"] in out:
            raise ValueError(f"duplicate PRINT id {site['id']}")
        out[site["id"]] = site
    return out


def check_prints_preserved(
    original_text: str,
    current_text: str,
    ids: list[int],
) -> str | None:
    """None if each id's PRINT_* matches the original; else a fail reason.

    Raises ValueError if the original text is missing a targeted id or
    contains duplicate ids (caller setup error).
    """
    original = sites_by_id(parse_print_sites_text(original_text))
    try:
        current = sites_by_id(parse_print_sites_text(current_text))
    except ValueError as exc:
        return str(exc)
    for print_id in ids:
        if print_id not in original:
            raise ValueError(f"id {print_id} has no PRINT_* in --original-hip")
        if print_id not in current:
            return f"id {print_id}: PRINT_* call is missing"
        want = original[print_id]
        got = current[print_id]
        if normalized_print(want) != normalized_print(got):
            return (
                f"id {print_id}: PRINT_* call changed: "
                f"{format_print(want)} -> {format_print(got)}"
            )
    return None
