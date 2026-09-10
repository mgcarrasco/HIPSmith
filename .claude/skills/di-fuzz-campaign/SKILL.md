---
name: di-fuzz-campaign
description: Run a HIPSmith debug-info fuzzing campaign for a number of iterations or for a fixed duration, on a ROCm nightly. Fetches the newest ROCm nightly matching the machine's GPU (or uses a caller-specified ROCm/nightly version), and builds HIPSmith in debug mode if no build is supplied. Use when asked to fuzz, run a fuzzing campaign, stress the debug-info pipeline, or test a ROCm nightly against HIPSmith.
---

# Fuzzing campaign on a ROCm nightly

Linux only — the nightlies fetched here are the `therock-dist-linux-*` builds. All paths
below are relative to the HIPSmith repository root; run the commands from there.

Three things must be in place before a campaign runs: a **ROCm toolchain** (supplying
`amdclang++` and `rocgdb`), a **HIPSmith binary**, and a **stop condition**. Resolve
each in order, then run `debug-info/di-fuzz-campaign.py`.

Ask the user only for what is genuinely ambiguous. If they said "fuzz for an hour" or
"run 500 iterations", that is the stop condition — do not ask again.

## 1. Resolve the ROCm toolchain

**If the user supplied a ROCm directory** (e.g. `/opt/rocm`, or a nightly they already
unpacked), use `<dir>/bin/amdclang++` and `<dir>/bin/rocgdb`. Verify both exist before
going further.

**Otherwise fetch a nightly matching this machine's GPU:**

```bash
.claude/skills/di-fuzz-campaign/scripts/fetch-rocm-nightly.py --dest <toolchain-dir>
```

It detects the installed GPU arch, picks the nightly family covering it, downloads and
unpacks it, and prints a JSON object with `arch`, `family`, `version`, `amdclang` and
`rocgdb`. Read the paths out of that JSON rather than guessing them.

It also writes a `DOWNLOADED_NIGHTLY` marker file into `--dest`, recording the tarball
URL, family, version and arch used. The unpacked tree itself does not state this
date-tagged nightly version anywhere — component headers and `amdclang++ --version`
only expose per-component git commit hashes — so this marker is the only record of
which nightly (and from which link) is sitting at that path. If `--dest` is later reused
without this marker present (e.g. a toolchain fetched by an older version of the
script, or unpacked by hand), treat the version as unknown rather than guessing.

Useful variants:

- `--list` — show the versions available for this GPU without downloading. Only about
  the last three weeks are retained, so a version the user names may be gone; if so,
  say what *is* available instead of silently substituting.
- `--version 10.1.0a20260910` — pin a specific nightly the user asked for.
- `--arch gfx942` — override detection.
- `--allow-multiarch` — fall back to the multiarch tarball if no family covers the GPU.

Notes:

- The download is roughly 2.3 GB and unpacks to appreciably more, so `--dest` needs
  around 10 GB free and should be on a large filesystem, not `$HOME` by default.
- If `--dest` already holds a usable toolchain it is reused, so re-running is cheap.
  Pass `--force` to re-fetch.
- If several GPUs are present the script picks one arch and reports the rest. That is
  intended; mention which one it chose in the final report.

## 2. Resolve the HIPSmith binary

**If the user supplied a HIPSmith binary or build directory**, use it.

**Otherwise build it in debug mode** — this repository's existing build directory is
configured `Debug`, so that is the convention to follow:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j"$(nproc)"
```

The binary lands at `build-debug/HIPSmith`. The build also copies `HIPSmith.h`,
`HIPSmithPrint.h` and `safe_math_macros.h` next to it; `gen_kernel.py` requires those to
sit beside the binary, so always point `--hipsmith` at the binary *inside* its build
directory rather than at a copy elsewhere.

Build HIPSmith with the **system compiler**, not the nightly's `amdclang++` — HIPSmith
is an ordinary host C++ program. The nightly toolchain is only for the kernels it
generates.

## 3. Run the campaign

```bash
debug-info/di-fuzz-campaign.py \
  --amdclang <toolchain>/bin/amdclang++ \
  --rocgdb   <toolchain>/bin/rocgdb \
  --hipsmith build-debug/HIPSmith \
  --offload-arch <arch> \
  --out-dir <campaign-dir> \
  --duration 3600            # or --iterations 500
```

- `--iterations N` and `--duration SECONDS` may be combined; whichever is reached first
  stops the campaign. With neither, it runs until Ctrl-C.
- Both bounds gate *new* iterations only. In-flight iterations are always allowed to
  finish, so the wall-clock overshoot can be up to one iteration watchdog period.
- Pass `--offload-arch` explicitly using the arch from step 1 rather than relying on its
  `native` default, so the campaign is reproducible and does not silently retarget.
- `--workers` (default 12) and `--inner-jobs` (default 10) multiply into concurrent
  build/run/gdb workers contending for the GPU. Keep the defaults unless the user asks
  otherwise; lower both together if contention shows up as timeouts.
- Use a `--out-dir` on a filesystem with room. Every iteration keeps its generated
  sources and reports; a busy hour produces thousands of run directories.
- `--gisel` lets iterations pick `-mllvm -global-isel=true` at random. Off by default;
  pass it only if the user asks for GlobalISel coverage.

Long campaigns should be started in the background so progress can be checked while
they run.

## 4. Report

`<campaign-dir>/campaign.json` is a live, atomically-rewritten index and is safe to read
at any time, including mid-run. It carries `total_iterations`, `counts_by_exit_code`,
and a per-iteration record of seed, exit code, duration and directory.

Exit codes are mechanical, not verdicts — they say which stage failed, not whether a
debug-info bug was found:

| code | meaning |
|------|---------|
| 0 | generate, build, run and gdb probe all succeeded |
| 2 | generation failed or timed out |
| 3 | a build failed |
| 4 | a run failed |
| 5 | a gdb probe failed |
| `orchestrator_timeout` | the whole iteration exceeded the watchdog |

Report the counts, the campaign directory, and a few example run directories for the
non-zero buckets. Do **not** claim a bug was found: nonzero codes here routinely mean a
backend limitation or a kernel that hangs on device, and judging a real debug-info
divergence needs a separate oracle over the per-iteration
`<run-dir>/out/fuzz-one.json` reports.
