---
name: di-fuzz-campaign-process
description: Process a finished HIPSmith debug-info fuzzing campaign (produced by /di-fuzz-campaign) by running the debug-info oracle over every iteration, then deduplicating and bisecting the resulting findings into signature clusters. Requires the campaign directory and the ROCm toolchain (amdclang/rocgdb) that produced it. Use when asked to process, analyze, triage, or find bugs in a finished fuzzing campaign, or to turn campaign.json/fuzz-one.json reports into concrete debug-info findings.
---

# Process a finished fuzzing campaign

All paths below are relative to the HIPSmith repository root; run the commands from
there. This skill only orchestrates two existing scripts —
`debug-info/di-oracle-campaign.py` and `debug-info/signature-findings.py` — it does not
generate or fetch anything itself.

Two things must be supplied before processing runs: the **campaign directory** to
process, and the **ROCm toolchain** (`amdclang`/`amdclang++` + `rocgdb`) that produced
it. `campaign.json` never records which toolchain built the campaign — that link isn't
stored anywhere on disk — so don't try to infer it; ask the user if it's genuinely
unclear which toolchain to use.

Ask the user only for what is genuinely ambiguous. If a ROCm release/version name was
mentioned, treat it purely as a label for the final report — it is not used to locate
the campaign or the toolchain.

## 1. Resolve inputs

- **Campaign directory**: must contain `campaign.json` and `run-*/out/fuzz-one.json`
  (the layout `/di-fuzz-campaign` produces). If the path doesn't look like that, say so
  before proceeding.
- **Toolchain**: if given a ROCm directory, derive `<dir>/bin/amdclang++` and
  `<dir>/bin/rocgdb`; if given explicit `--amdclang`/`--rocgdb` paths, use them as-is.
  Verify both exist before continuing — same verification habit as
  `/di-fuzz-campaign` step 1.

## 2. Run the oracle over the whole campaign

```bash
debug-info/di-oracle-campaign.py <campaign> \
  -o <campaign>/oracle/summary.json \
  --findings <campaign>/oracle/findings.jsonl \
  --info <campaign>/oracle/info.jsonl \
  --jobs <oracle-jobs>          # default 16, di-oracle-campaign.py's own default
```

This classifies every `run-*/out/fuzz-one.json` report and needs no GPU/toolchain
access — it only reads JSON already on disk. Report the printed summary (well-defined
count/percent, gate-failure breakdown, findings by kind, info-record count) before
moving on.

If `<campaign>/oracle/findings.jsonl` ends up empty, say so and stop — there is nothing
for the next step to bisect.

## 3. Deduplicate and bisect the findings

```bash
debug-info/signature-findings.py <campaign>/oracle/findings.jsonl \
  --campaign <campaign> \
  --amdclang <amdclang> --rocgdb <rocgdb> \
  --jobs <bisect-jobs> \        # default 30, signature-findings.py's own default
  -o <campaign>/signatures/ \
  --resume
```

This step is expensive — a real compiler build plus a gdb probe per finding
(`--build-timeout 45s` / `--gdb-timeout 30s` defaults, matching the fuzzing campaign's
own `--build-timeout`/`--run-timeout` defaults). Unlike the oracle step, it
processes **every** finding by default; there is no cap. For a campaign with many
findings, start this in the background and poll
`<campaign>/signatures/summary.json` for progress, the same way `/di-fuzz-campaign`
recommends backgrounding long runs.

Always pass `--resume`: it's a no-op on a first run (nothing in `-o` yet to skip
against) and lets a re-invocation after an interruption continue instead of re-bisecting
everything already done.

Useful overrides, only if the caller asks:

- `--limit N` — bound how many findings get bisected, if the caller wants a bounded run
  instead of processing everything.
- `--kind`, `--id`, `--build` — restrict to a specific oracle kind
  (`incorrectness`/`missing`), PRINT id, or target build.
- `--build-timeout` / `--gdb-timeout` — raise if builds or gdb probes are timing out.
- `--keep-tmp <dir>` — preserve every per-failure bisect working tree (builds +
  binaries). **Off by default** — `signatures.jsonl`/`representatives.jsonl` already
  capture what's needed to reduce a representative later, and with no `--limit` this can
  be a lot of disk. Only turn it on when the caller wants to inspect a specific
  bisection's tree.

## 4. Report

Summarize, in this order:

1. Oracle pass: well-defined count/percent, gate-failure breakdown, total findings by
   kind, info-record count (from `<campaign>/oracle/summary.json`).
2. Bisection pass: total findings bisected, number of distinct signature clusters
   (dedup ratio = findings ÷ clusters), and per-signature counts (from
   `<campaign>/signatures/summary.json`).
3. The representative finding for each cluster (from
   `<campaign>/signatures/representatives.jsonl`) as the starting point for follow-up
   reduction work.

Label the report with the ROCm release/version string if the caller supplied one.

Do not editorialize about which clusters are "real bugs" beyond what the oracle and
signature data already state — the oracle already distinguishes findings (worth
investigating) from info records (mirror cases, not failures); this skill doesn't add a
further verdict on top of that.
