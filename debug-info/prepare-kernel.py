#!/usr/bin/env python3
"""Prepare a HIPProg.hip copy for C-Vise: inline headers, extract host setup.

Writes a new HIP file; the input is never modified. Quoted
#include "HIPSmithPrint.h" (including the one nested in HIPSmith.h) is left
as an include. If present, void setup_hip_globals() is moved to
setup_hip_globals.h next to the output and replaced by a quoted include.
Angled includes are left unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path


INLINE_HEADERS = ("HIPSmith.h", "safe_math_macros.h")
SETUP_HEADER = "setup_hip_globals.h"
NEVER_INLINE = frozenset({"HIPSmithPrint.h", SETUP_HEADER})
SETUP_RE = re.compile(r"^void setup_hip_globals\s*\(\s*\)\s*\{", re.MULTILINE)

INCLUDE_RE = re.compile(
    r"^([ \t]*#[ \t]*include[ \t]+)([\"<])([^\">]+)([\">])([ \t]*)$",
    re.MULTILINE,
)


class PrepareError(ValueError):
    """Invalid include layout or missing header."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inline HIPSmith.h and safe_math_macros.h into a HIPProg.hip copy, "
            "and move setup_hip_globals into a sibling header. Does not rewrite "
            "the input. Leaves #include \"HIPSmithPrint.h\"."
        )
    )
    parser.add_argument("hip_file", type=Path, nargs="?", help="HIPProg.hip to read")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination HIP file (must not be the input)",
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        help="Directory with HIPSmith.h / safe_math_macros.h "
        "(default: hip_file parent). setup_hip_globals.h is written next "
        "to --output, not read from here.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in checks and exit (ignores other arguments)",
    )
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.hip_file is None or args.output is None:
        parser.error("hip_file and --output are required")
    return args


def die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def include_matches(text: str) -> list[re.Match[str]]:
    return list(INCLUDE_RE.finditer(text))


def quoted_includes(text: str, name: str) -> list[re.Match[str]]:
    return [
        match
        for match in include_matches(text)
        if match.group(2) == '"' and match.group(3) == name
    ]


def angled_includes(text: str, name: str) -> list[re.Match[str]]:
    return [
        match
        for match in include_matches(text)
        if match.group(2) == "<" and match.group(3) == name
    ]


def resolve_header(name: str, search: list[Path]) -> Path:
    for root in search:
        path = root / name
        if path.is_file():
            return path
    searched = ", ".join(str(root) for root in search)
    raise PrepareError(f"{name} not found in {searched}")


def wrap_inlined(name: str, body: str) -> str:
    text = body.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return f"// --- inlined {name} ---\n{text}// --- end {name} ---\n"


def _skip_quoted(text: str, i: int, quote: str) -> int:
    i += 1
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    raise PrepareError("unterminated string in setup_hip_globals")


def match_braced_block(text: str, open_brace: int) -> int:
    """Return the index just after the matching `}` for text[open_brace] == '{'."""
    if open_brace >= len(text) or text[open_brace] != "{":
        raise PrepareError("setup_hip_globals opening brace not found")
    depth = 0
    i = open_brace
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                raise PrepareError("unterminated comment in setup_hip_globals")
            i = end + 2
            continue
        if c in "\"'":
            i = _skip_quoted(text, i, c)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise PrepareError("unbalanced braces in setup_hip_globals")


def extract_setup_hip_globals(text: str) -> tuple[str, str | None]:
    """Replace a unique file-scope setup_hip_globals with a quoted include.

    Returns (new_hip_text, header_body_or_None).
    """
    text = text.replace("\r\n", "\n")
    found = list(SETUP_RE.finditer(text))
    if not found:
        return text, None
    if len(found) > 1:
        raise PrepareError(
            f"duplicate void setup_hip_globals() ({len(found)})"
        )
    match = found[0]
    open_brace = match.end() - 1
    end = match_braced_block(text, open_brace)
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == "\n":
        end += 1
    body = text[match.start():end]
    if "void setup_hip_globals" in text[end:]:
        raise PrepareError("setup_hip_globals remains after the extracted definition")
    include = f'#include "{SETUP_HEADER}"\n'
    header = (
        "#ifndef SETUP_HIP_GLOBALS_H\n"
        "#define SETUP_HIP_GLOBALS_H\n\n"
        f"{body.rstrip()}\n\n"
        "#endif\n"
    )
    return text[:match.start()] + include + text[end:], header


def inline_hip_text(text: str, headers: dict[str, str]) -> str:
    """Replace each unique quoted include of headers keys with that body."""
    text = text.replace("\r\n", "\n")
    for name in NEVER_INLINE:
        if angled_includes(text, name):
            raise PrepareError(f'{name} is included with <>')
    replacements: list[tuple[int, int, str]] = []
    for name in INLINE_HEADERS:
        if name not in headers:
            raise PrepareError(f"no body provided for {name}")
        if angled_includes(text, name):
            raise PrepareError(f'{name} is included with <>')
        found = quoted_includes(text, name)
        if not found:
            raise PrepareError(f'missing #include "{name}"')
        if len(found) > 1:
            raise PrepareError(f'duplicate #include "{name}" ({len(found)})')
        match = found[0]
        replacements.append((match.start(), match.end(), wrap_inlined(name, headers[name])))
    # Last-to-first so earlier offsets stay valid. HIPSmith.h is before
    # safe_math_macros.h; this pastes safe_math first, then HIPSmith.
    replacements.sort(key=lambda item: item[0], reverse=True)
    out = text
    for start, end, payload in replacements:
        out = out[:start] + payload + out[end:]
    if quoted_includes(out, "HIPSmith.h") or quoted_includes(out, "safe_math_macros.h"):
        raise PrepareError("inlined HIP still includes HIPSmith.h or safe_math_macros.h")
    if not quoted_includes(out, "HIPSmithPrint.h"):
        raise PrepareError('inlined HIP lost #include "HIPSmithPrint.h"')
    if angled_includes(out, "HIPSmithPrint.h"):
        raise PrepareError("HIPSmithPrint.h is included with <>")
    return out


def prepare_hip_file(hip_file: Path, output: Path, resource_dir: Path | None = None) -> Path:
    hip_file = hip_file.expanduser()
    output = output.expanduser()
    if not hip_file.is_file():
        raise PrepareError(f"no such file: {hip_file}")
    search = [hip_file.parent]
    if resource_dir is not None:
        resource_dir = resource_dir.expanduser()
        if not resource_dir.is_dir():
            raise PrepareError(f"no such resource-dir: {resource_dir}")
        if resource_dir.resolve() != hip_file.parent.resolve():
            search.append(resource_dir)
    if output.resolve() == hip_file.resolve():
        raise PrepareError(f"refusing to overwrite input: {hip_file}")
    headers = {
        name: resolve_header(name, search).read_text(encoding="utf-8", errors="replace")
        for name in INLINE_HEADERS
    }
    inlined = inline_hip_text(
        hip_file.read_text(encoding="utf-8", errors="replace"),
        headers,
    )
    inlined, setup_header = extract_setup_hip_globals(inlined)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(inlined, encoding="utf-8")
    setup_path = output.with_name(SETUP_HEADER)
    if setup_header is None:
        if setup_path.is_file() and setup_path.resolve() != hip_file.resolve():
            setup_path.unlink()
    else:
        setup_path.write_text(setup_header, encoding="utf-8")
    return output


def _self_test() -> None:
    hipsmith = (
        '#include <cstdint>\n'
        '#include "HIPSmithPrint.h"\n'
        '#define HIP_CHECK(x) (x)\n'
    )
    safe_math = "#define SAFE_MATH_H 1\n#define safe_add(a, b) ((a)+(b))\n"
    hip = (
        '#include <hip/hip_runtime.h>\n'
        '#include "HIPSmith.h"\n'
        '#include "safe_math_macros.h"\n'
        "\n"
        "#define uint8_t unsigned char\n"
        "#define int8_t char\n"
        "int kernel;\n"
    )
    out = inline_hip_text(hip, {"HIPSmith.h": hipsmith, "safe_math_macros.h": safe_math})
    assert '#include "HIPSmith.h"' not in out
    assert "#include \"safe_math_macros.h\"" not in out
    assert '#include "HIPSmithPrint.h"' in out
    assert "#include <hip/hip_runtime.h>" in out
    assert "#define uint8_t unsigned char" in out
    hipsmith_at = out.index("// --- inlined HIPSmith.h ---")
    safe_at = out.index("// --- inlined safe_math_macros.h ---")
    redef_at = out.index("#define uint8_t unsigned char")
    print_at = out.index('#include "HIPSmithPrint.h"')
    assert hipsmith_at < print_at < safe_at < redef_at, (
        hipsmith_at, print_at, safe_at, redef_at
    )

    try:
        inline_hip_text(hip.replace('#include "HIPSmith.h"\n', ""), {
            "HIPSmith.h": hipsmith, "safe_math_macros.h": safe_math,
        })
        raise AssertionError("missing HIPSmith.h should fail")
    except PrepareError:
        pass

    try:
        inline_hip_text(
            hip.replace(
                '#include "HIPSmith.h"\n',
                '#include "HIPSmith.h"\n#include "HIPSmith.h"\n',
            ),
            {"HIPSmith.h": hipsmith, "safe_math_macros.h": safe_math},
        )
        raise AssertionError("duplicate HIPSmith.h should fail")
    except PrepareError:
        pass

    try:
        inline_hip_text(
            hip.replace('#include "HIPSmith.h"', "#include <HIPSmith.h>"),
            {"HIPSmith.h": hipsmith, "safe_math_macros.h": safe_math},
        )
        raise AssertionError("angled HIPSmith.h should fail")
    except PrepareError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "HIPProg.hip"
        hdr = root / "HIPSmith.h"
        sm = root / "safe_math_macros.h"
        hdr.write_text(hipsmith, encoding="utf-8")
        sm.write_text(safe_math, encoding="utf-8")
        src.write_text(hip, encoding="utf-8")
        dest = root / "out" / "HIPProg.hip"
        prepare_hip_file(src, dest, root)
        got = dest.read_text(encoding="utf-8")
        assert '#include "HIPSmithPrint.h"' in got
        assert not (root / "out" / SETUP_HEADER).is_file()
        try:
            prepare_hip_file(src, src, root)
            raise AssertionError("overwrite should fail")
        except PrepareError:
            pass
        assert src.read_text(encoding="utf-8") == hip

        src.write_text(
            hip
            + "\n__constant__ int hip_const_1 = {};\n"
            + "void setup_hip_globals() {\n"
            + "        int host_hip_const_1 = 1;\n"
            + "        HIP_CHECK(hipMemcpyToSymbol(hip_const_1, "
            + "&host_hip_const_1, sizeof(host_hip_const_1)));\n"
            + '        char msg[] = "brace { in string";\n'
            + "}\n"
            + "int after_setup;\n",
            encoding="utf-8",
        )
        dest2 = root / "out2" / "HIPProg.hip"
        prepare_hip_file(src, dest2, root)
        got2 = dest2.read_text(encoding="utf-8")
        hdr2 = dest2.with_name(SETUP_HEADER).read_text(encoding="utf-8")
        assert f'#include "{SETUP_HEADER}"' in got2
        assert "void setup_hip_globals" not in got2
        assert "int after_setup;" in got2
        assert "void setup_hip_globals()" in hdr2
        assert "brace { in string" in hdr2
        hip_text, header_none = extract_setup_hip_globals("int x;\n")
        assert header_none is None
        assert hip_text == "int x;\n"
        try:
            extract_setup_hip_globals(
                "void setup_hip_globals() {}\n"
                "void setup_hip_globals() {}\n"
            )
            raise AssertionError("duplicate setup should fail")
        except PrepareError:
            pass
    print("prepare-kernel self-test ok")


def main() -> int:
    args = parse_args()
    if args.self_test:
        _self_test()
        return 0
    resource = args.resource_dir if args.resource_dir is not None else args.hip_file.parent
    try:
        dest = prepare_hip_file(args.hip_file, args.output, resource)
    except PrepareError as exc:
        die(str(exc))
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    setup_path = dest.with_name(SETUP_HEADER)
    if setup_path.is_file():
        print(f"wrote {setup_path} ({setup_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
