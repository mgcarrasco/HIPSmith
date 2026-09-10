#!/usr/bin/env python3
"""Fetch a ROCm nightly tarball matching the GPU installed in this machine.

The nightlies live at https://nightly.repo.amd.com/rocm/core/tarball/ as files named
`therock-dist-<os>-<family>-<version>.tar.gz`. Only Linux builds are considered. Each
tarball unpacks to a self-contained ROCm tree, so `<dest>/bin/amdclang++` and
`<dest>/bin/rocgdb` are all a fuzzing campaign needs.

A `<family>` is not always a bare gfx name: some cover one chip (`gfx90a`, `gfx1151`),
some cover a group with an `X` placeholder plus a market suffix (`gfx94X-dcgpu`,
`gfx103X-all`). Detection reads the installed GPU's arch and picks the family that
covers it, preferring an exact match over a wildcard one.

  ./fetch-rocm-nightly.py --list                 # what is available for this GPU
  ./fetch-rocm-nightly.py --dest /work/rocm-x    # newest nightly for this GPU
  ./fetch-rocm-nightly.py --version 10.1.0a20260901 --dest ...   # pin a nightly
  ./fetch-rocm-nightly.py --arch gfx942 --dest ...               # override detection

On success a JSON object describing the toolchain (arch, family, version, and the
absolute amdclang/rocgdb paths) is written to stdout; progress goes to stderr. A
`DOWNLOADED_NIGHTLY` marker file recording the tarball URL, family, version and arch is
also written into `--dest` — the unpacked tree itself only exposes per-component build
hashes, not this date-tagged nightly version or where it was fetched from.

Exit codes:
  0  toolchain ready at --dest
  1  usage/setup error (no GPU found, no matching family, download failed)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_URL = "https://nightly.repo.amd.com/rocm/core/tarball/"

# The index page carries the listing as a JS literal rather than as <a> links.
FILES_RE = re.compile(r"const files\s*=\s*(\[.*?\]);", re.S)

# therock-dist-linux-gfx94X-dcgpu-10.1.0a20260910.tar.gz
TARBALL_RE = re.compile(
    r"^therock-dist-linux-(?P<family>.+?)-(?P<version>\d[\w.]*a\d{8})\.tar\.gz$"
)

# `gfx90a:sramecc+:xnack-` -> `gfx90a`; the target features do not select a tarball.
ARCH_RE = re.compile(r"^(gfx[0-9a-f]+)")


def detect_arch() -> tuple[str, list[str]]:
    """Return (chosen arch, every arch seen).

    Tries the cheap purpose-built tools first and only falls back to parsing
    rocminfo, which is slow and whose output format is the least stable.
    """
    candidates: list[list[str]] = []
    for name in ("offload-arch", "amdgpu-arch"):
        found = shutil.which(name)
        if found:
            candidates.append([found])
    for path in ("/opt/rocm/llvm/bin/amdgpu-arch", "/opt/rocm/bin/offload-arch"):
        if Path(path).is_file():
            candidates.append([path])

    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        # One line per GPU, so a multi-GPU box repeats an arch once per device;
        # only the distinct archs matter when choosing a tarball.
        archs = []
        for line in proc.stdout.split():
            match = ARCH_RE.match(line.strip())
            if match and match.group(1) not in archs:
                archs.append(match.group(1))
        if archs:
            return archs[0], archs

    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        try:
            proc = subprocess.run([rocminfo], capture_output=True, text=True,
                                  timeout=120)
            archs = []
            for match in re.finditer(r"\bgfx[0-9a-f]+\b", proc.stdout):
                if match.group(0) not in archs:
                    archs.append(match.group(0))
            if archs:
                return archs[0], archs
        except (OSError, subprocess.SubprocessError):
            pass

    raise SystemExit(
        "error: could not detect an AMD GPU (tried offload-arch, amdgpu-arch, "
        "rocminfo). Pass --arch explicitly, e.g. --arch gfx90a"
    )


def fetch_index() -> list[dict[str, Any]]:
    with urllib.request.urlopen(INDEX_URL, timeout=120) as response:
        html = response.read().decode("utf-8", "replace")
    match = FILES_RE.search(html)
    if match is None:
        raise SystemExit(f"error: could not find the file listing in {INDEX_URL}; "
                         "the index format may have changed")
    return json.loads(match.group(1))


def parse_index(files: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """{family: {version: filename}}, Linux only, test bundles dropped."""
    by_family: dict[str, dict[str, str]] = {}
    for entry in files:
        name = entry.get("name", "")
        match = TARBALL_RE.match(name)
        if match is None:
            continue
        family = match.group("family")
        # `<family>-tests` holds the ROCm test suite, not a toolchain.
        if family.endswith("-tests"):
            continue
        by_family.setdefault(family, {})[match.group("version")] = name
    return by_family


def family_covers(family: str, arch: str) -> int:
    """0 = no, 1 = wildcard match, 2 = exact match (higher wins)."""
    # Split off the market suffix: `gfx94X-dcgpu` -> `gfx94X`, `gfx950-dcgpu` -> `gfx950`.
    token = family.split("-", 1)[0]
    if token == arch:
        return 2
    if "X" in token:
        # `gfx101X` covers gfx1010, gfx1012, ... - one placeholder digit each.
        pattern = "^" + re.escape(token).replace("X", ".") + "$"
        if re.match(pattern, arch):
            return 1
    return 0


def pick_family(by_family: dict[str, dict[str, str]], arch: str,
                allow_multiarch: bool) -> str:
    scored = [(family_covers(f, arch), f) for f in by_family]
    best = [f for score, f in scored if score == 2] or \
           [f for score, f in scored if score == 1]
    if best:
        # Deterministic when a wildcard family somehow appears twice.
        return sorted(best)[0]
    if allow_multiarch and "multiarch" in by_family:
        print(f"warning: no family covers {arch}; falling back to multiarch",
              file=sys.stderr)
        return "multiarch"
    raise SystemExit(
        f"error: no nightly family covers {arch}. Available: "
        f"{', '.join(sorted(by_family))}. Re-run with --allow-multiarch to use the "
        "multiarch build instead."
    )


def write_marker(dest: Path, *, url: str, family: str, version: str, arch: str,
                 reused: bool) -> None:
    """Record which nightly is unpacked at dest, so it can be identified later —
    the tree itself only exposes per-component build hashes, not this date-tagged
    nightly version or the URL it came from."""
    marker = dest / "DOWNLOADED_NIGHTLY"
    marker.write_text(
        f"url: {url}\n"
        f"family: {family}\n"
        f"version: {version}\n"
        f"arch: {arch}\n"
        f"recorded: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"{'reused existing toolchain at this path' if reused else 'freshly downloaded'}\n"
    )


def download(url: str, target: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise SystemExit("error: curl is required to download the nightly")
    # -C - resumes a partial file, so an interrupted 2+ GB download is not restarted.
    cmd = [curl, "-fL", "--retry", "3", "--retry-delay", "5", "-C", "-",
           "-o", str(target), url]
    print(f"downloading {url}", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"error: curl exited {result.returncode} fetching {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dest", type=Path,
                        help="Directory to unpack the toolchain into "
                             "(default: ./rocm-nightly/<family>-<version>). "
                             "Needs roughly 10 GB free.")
    parser.add_argument("--arch",
                        help="GPU arch to match, e.g. gfx90a "
                             "(default: detect from the installed GPU)")
    parser.add_argument("--version",
                        help="Nightly version to pin, e.g. 10.1.0a20260910 "
                             "(default: the newest available for this GPU)")
    parser.add_argument("--list", action="store_true",
                        help="List the nightlies available for this GPU and exit")
    parser.add_argument("--allow-multiarch", action="store_true",
                        help="Fall back to the multiarch tarball when no family "
                             "covers this GPU")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-unpack even if --dest already "
                             "holds a usable toolchain")
    parser.add_argument("--keep-tarball", action="store_true",
                        help="Keep the downloaded tarball instead of deleting it "
                             "after unpacking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.arch:
        arch, all_archs = args.arch, [args.arch]
    else:
        arch, all_archs = detect_arch()
        if len(all_archs) > 1:
            print(f"note: several GPU archs present ({', '.join(all_archs)}); "
                  f"building for {arch}", file=sys.stderr)
    print(f"arch:    {arch}", file=sys.stderr)

    by_family = parse_index(fetch_index())
    family = pick_family(by_family, arch, args.allow_multiarch)
    versions = sorted(by_family[family])
    print(f"family:  {family}", file=sys.stderr)

    if args.list:
        print(f"{len(versions)} nightlies for {arch} (family {family}):",
              file=sys.stderr)
        for version in versions:
            print(version)
        return 0

    if args.version:
        if args.version not in versions:
            print(f"error: {args.version} is not available for family {family}. "
                  f"Available: {', '.join(versions)}", file=sys.stderr)
            return 1
        version = args.version
    else:
        version = versions[-1]
    filename = by_family[family][version]
    print(f"version: {version}", file=sys.stderr)

    dest = args.dest or Path("rocm-nightly") / f"{family}-{version}"
    dest = dest.resolve()
    amdclang = dest / "bin" / "amdclang++"
    rocgdb = dest / "bin" / "rocgdb"

    url = INDEX_URL + filename
    reused = amdclang.is_file() and rocgdb.is_file() and not args.force
    if reused:
        print(f"reusing existing toolchain at {dest}", file=sys.stderr)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        tarball = dest.parent / filename
        if not tarball.is_file() or args.force:
            download(url, tarball)
        print(f"unpacking into {dest}", file=sys.stderr)
        # The archive holds ./bin, ./lib, ... so it unpacks straight into dest.
        result = subprocess.run(["tar", "-xzf", str(tarball), "-C", str(dest)])
        if result.returncode != 0:
            print(f"error: tar exited {result.returncode}", file=sys.stderr)
            return 1
        if not args.keep_tarball:
            tarball.unlink(missing_ok=True)

    missing = [str(p) for p in (amdclang, rocgdb) if not p.is_file()]
    if missing:
        print(f"error: toolchain incomplete, missing: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    write_marker(dest, url=url, family=family, version=version, arch=arch,
                 reused=reused)

    json.dump({
        "arch": arch,
        "all_archs": all_archs,
        "family": family,
        "version": version,
        "root": str(dest),
        "amdclang": str(amdclang),
        "rocgdb": str(rocgdb),
    }, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
