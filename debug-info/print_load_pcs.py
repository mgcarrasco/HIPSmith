"""Find load ELF PCs inside each PRINT column segment on gfx90a.

Used by run-gdb.py to decide located_on_line. Every load in those segments is
kept, which overapproximates the volatile read: flat/global/buffer/scratch
loads, s_load, and ds_read. Only gfx90a is accepted.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_ARCH = "gfx90a"
DEVICE_PREFIX = "hip-amdgcn-amd-amdhsa--"

ROW_RE = re.compile(
    r"^(0x[0-9a-fA-F]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s*(.*)$"
)
FILE_RE = re.compile(r"^file_names\[\s*(\d+)\]:")
NAME_RE = re.compile(r'^\s+name:\s+"(.*)"')
INSN_RE = re.compile(r"^\s*(.+?)\s+//\s+([0-9A-Fa-f]+):")
LOAD_MARKS = (
    "flat_load",
    "global_load",
    "scratch_load",
    "buffer_load",
    "ds_load",
    "ds_read",
    "s_load",
)


class LoadPcError(Exception):
    """Host-side failure while finding PRINT load PCs."""


def llvm_tools_from_rocgdb(rocgdb: Path) -> Path:
    tools = rocgdb.resolve().parent.parent / "lib" / "llvm" / "bin"
    for name in (
        "llvm-objcopy",
        "clang-offload-bundler",
        "llvm-dwarfdump",
        "llvm-objdump",
    ):
        path = tools / name
        if not path.is_file():
            raise LoadPcError(f"missing {name} next to rocgdb at {tools}")
    return tools


def run_tool(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if proc.returncode != 0:
        raise LoadPcError(f"{cmd[0]} failed:\n{proc.stdout}")
    return proc.stdout


def list_device_bundles(listing: str) -> list[str]:
    return [
        line.strip()
        for line in listing.splitlines()
        if line.strip().startswith(DEVICE_PREFIX)
    ]


def require_gfx90a_bundle(device_bundles: list[str]) -> str:
    # gfx90a only. Load mnemonics were checked on this architecture. A
    # different device bundle is a probe failure so a miss is not reported
    # as located_on_line false.
    if len(device_bundles) != 1:
        raise LoadPcError(
            "expected exactly one device bundle "
            f"{DEVICE_PREFIX}{REQUIRED_ARCH}, found {device_bundles or 'none'}"
        )
    bundle = device_bundles[0]
    arch = bundle[len(DEVICE_PREFIX) :]
    if arch != REQUIRED_ARCH:
        raise LoadPcError(
            f"device bundle is {bundle!r}; only {REQUIRED_ARCH} is supported "
            "because the load mnemonics were checked on gfx90a"
        )
    return bundle


def parse_debug_line(dump: str) -> tuple[dict[int, str], list[dict[str, Any]]]:
    files: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    current: int | None = None
    for line in dump.splitlines():
        file_match = FILE_RE.match(line)
        if file_match:
            current = int(file_match.group(1))
            continue
        name_match = NAME_RE.match(line)
        if name_match and current is not None and current not in files:
            files[current] = name_match.group(1)
            continue
        row_match = ROW_RE.match(line)
        if not row_match:
            continue
        flags = row_match.group(5)
        rows.append(
            {
                "addr": int(row_match.group(1), 16),
                "line": int(row_match.group(2)),
                "column": int(row_match.group(3)),
                "file": int(row_match.group(4)),
                "end": "end_sequence" in flags,
                "is_stmt": "is_stmt" in flags,
            }
        )
    return files, rows


def matching_file_indices(files: dict[int, str], hip_basename: str) -> list[int]:
    return [
        index
        for index, name in files.items()
        if Path(name).name == hip_basename
    ]


def print_column(source_line: str) -> int | None:
    at = source_line.find("PRINT_")
    if at < 0:
        return None
    return at + 1


def segments(
    rows: list[dict[str, Any]],
    file_indices: list[int],
    line_no: int,
    column: int,
) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    file_set = set(file_indices)
    index = 0
    while index < len(rows):
        row = rows[index]
        if (
            row["end"]
            or row["file"] not in file_set
            or row["line"] != line_no
            or row["column"] != column
        ):
            index += 1
            continue
        start = row["addr"]
        index += 1
        while index < len(rows) and not rows[index]["end"] and (
            rows[index]["line"] == 0
            or (
                rows[index]["file"] in file_set
                and rows[index]["line"] == line_no
                and rows[index]["column"] == column
            )
        ):
            index += 1
        end = rows[index]["addr"] if index < len(rows) else start
        found.append((start, end))
    return found


def is_load(asm: str) -> bool:
    return any(mark in asm for mark in LOAD_MARKS)


def parse_disassembly(text: str) -> list[tuple[int, str]]:
    insns: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = INSN_RE.search(line)
        if not match:
            continue
        insns.append((int(match.group(2), 16), match.group(1).strip()))
    return insns


def loads_in_segments(
    insns: list[tuple[int, str]],
    spans: list[tuple[int, int]],
) -> list[int]:
    loads: list[int] = []
    seen: set[int] = set()
    for start, end in spans:
        for addr, asm in insns:
            if start <= addr < end and is_load(asm) and addr not in seen:
                seen.add(addr)
                loads.append(addr)
    return loads


def is_kernel_symbol(name: str) -> bool:
    """Device text symbol for hipsmith_kernel, mangled or not."""
    if name == "hipsmith_kernel" or name.startswith("hipsmith_kernel("):
        return True
    # Itanium: _Z15hipsmith_kernel + params. The kernel descriptor is
    # the same prefix with a .kd suffix and a different symbol type.
    return name.startswith("_Z15hipsmith_kernel") and ".kd" not in name


def kernel_elf_address(nm_text: str) -> int:
    """ELF address of the device kernel text symbol. One definition, or fail."""
    hits: list[int] = []
    for line in nm_text.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        addr_text, kind, name = parts
        if kind not in "Tt":
            continue
        if not is_kernel_symbol(name):
            continue
        try:
            hits.append(int(addr_text, 16))
        except ValueError:
            continue
    if len(hits) != 1:
        raise LoadPcError(
            "expected one hipsmith_kernel text symbol in the device object, "
            f"found {len(hits)}"
        )
    return hits[0]


def annotate_sites_with_loads(
    sites: list[dict[str, Any]],
    hip_file: Path,
    binary: Path,
    rocgdb: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Return sites with load_elf_pcs, and the device ELF address of the kernel.

    The kernel address is the base for the ELF-to-runtime slide. Raises
    LoadPcError if the device bundle is missing or not gfx90a, or if the
    DWARF/tools steps fail.
    """
    tools = llvm_tools_from_rocgdb(rocgdb)
    with tempfile.TemporaryDirectory(prefix="print_load_pcs_") as tmp:
        fatbin = Path(tmp) / "fatbin"
        run_tool(
            [
                str(tools / "llvm-objcopy"),
                f"--dump-section=.hip_fatbin={fatbin}",
                str(binary),
            ]
        )
        listing = run_tool(
            [
                str(tools / "clang-offload-bundler"),
                "--type=o",
                f"--input={fatbin}",
                "--list",
            ]
        )
        bundle = require_gfx90a_bundle(list_device_bundles(listing))
        obj = Path(tmp) / "dev.o"
        run_tool(
            [
                str(tools / "clang-offload-bundler"),
                "--unbundle",
                "--type=o",
                f"--input={fatbin}",
                f"--targets={bundle}",
                f"--output={obj}",
            ]
        )
        dump = run_tool([str(tools / "llvm-dwarfdump"), "--debug-line", str(obj)])
        asm = run_tool([str(tools / "llvm-objdump"), "-d", str(obj)])
        nm_text = run_tool([str(tools / "llvm-nm"), str(obj)])
    kernel_elf = kernel_elf_address(nm_text)

    files, rows = parse_debug_line(dump)
    file_indices = matching_file_indices(files, hip_file.name)
    if not file_indices:
        raise LoadPcError(
            f"no DWARF file named {hip_file.name!r} in the device line table"
        )
    insns = parse_disassembly(asm)
    hip_lines = hip_file.read_text(encoding="utf-8", errors="replace").splitlines()

    annotated: list[dict[str, Any]] = []
    for site in sites:
        line_no = int(site["line"])
        column = None
        if 1 <= line_no <= len(hip_lines):
            column = print_column(hip_lines[line_no - 1])
        spans = (
            segments(rows, file_indices, line_no, column)
            if column is not None
            else []
        )
        loads = loads_in_segments(insns, spans)
        copy = dict(site)
        copy["load_elf_pcs"] = loads
        annotated.append(copy)
    return annotated, kernel_elf
