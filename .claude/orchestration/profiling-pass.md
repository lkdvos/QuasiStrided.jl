# Profiling pass: where does QuasiStrided spend its time?

## Goal / acceptance criteria

Reuse the bucketed profiler on branch `upstream-bench` (PR #5) rather than
build new tooling (user's explicit direction), merge it into `main` first
(user's explicit choice when asked), extend its case coverage to include a
few of this project's own `benchmark/harness.jl` shapes, and produce a
fresh, artefact-traced breakdown of microkernel-vs-packing-vs-driver-vs-glue
time for a bounded set of representative cases. Full narrative, every
measured number, and the fixed verdict rule are in `docs/decisions.md`,
"Profiling pass: where does QuasiStrided spend its time? (2026-09-21)" --
that file, not this one, is authoritative for *why*. This file is the
coordinator-owned status record.

## User decisions (binding)

1. Reuse `upstream-bench`'s profiling tooling (`profile_to_suite.jl` +
   `profile_buckets.jl`) rather than write a new profiler from scratch.
2. Merge PR #5 into `main` first (not: work on an isolated copy, not: work
   directly on the branch).
3. When it turned out PR #5 also carries a real `src/integrations/tensoroperations.jl`
   behavior change (Amendment 7: `tensoradd!`/`tensortrace!` fall back to
   `StridedNative` instead of hard-rejecting) that its own GitHub
   description never mentioned: merge everything as-is (the change was
   already tested and reportedly approved by the user in an earlier
   session, per its own commit message).

## Repo / branch

Worktree `/mnt/home/ldevos/Projects/QuasiStrided.jl/main`. Work done
directly on `main` (no feature branch) -- T0 (the merge) already puts
`main` at a new state; everything after T0 is small, additive
benchmark-tooling edits plus docs.

## Status -- COMPLETE

- [x] Planning (orch-planner execution contract, then revised in-session for
      the tooling-reuse decision; see `docs/decisions.md`'s dated section
      for what superseded what).
- [x] T0: merged `upstream-bench` (PR #5) into `main`. Two doc conflicts
      (`STATUS.md`, `docs/decisions.md`) resolved by keeping both sides in
      chronological order. `Pkg.test()`: 35170/35170 passing post-merge.
      Pushed to `main` (merge commit `19dd25c`); PR #5 auto-closed as merged
      by GitHub.
- [x] T1: environment check -- Julia 1.13.0 (>=1.11 required), `benchmark/`
      environment instantiates cleanly, machine load nominal (load average
      ~2.3 on 32 cores, mostly idle) on `ccqlin038`.
- [x] T2: extended `benchmark/profile_to_suite.jl`'s `CASES` with
      `plain_256`/`plain_512`/`smallN_256x256x12` (label/dims, adapter
      path), added a `DIRECT_CASES`/`profile_one_direct!` mechanism and one
      `scattered_64` case (direct `plan_contract`/`execute!` path, no
      adapter) for `harness.jl`'s permuted/negative-stride/sliced fixture.
- [x] **Correction 1, not in the original plan**: `profile_buckets.jl`'s
      leaf-only self-time classification was found to be wrong for
      compute-bound cases (a 512³ GEMM measured 3.48% microkernel / 94%
      "other" -- impossible for a case that's compute-bound by
      construction). Root cause: the true hot leaf instructions are inlined
      `SIMD.jl` package frames (`fmuladd`/`vload`), which carry neither
      "kernels/" nor "accumulate" in their own path. First fix: walk each
      sample's backtrace leaf-to-root, matching function-name-or-file
      substrings together, first match wins.
- [x] T5 (independent review, `orch-reviewer`, run BEFORE committing
      anything from this pass): reviewed Correction 1 and the draft
      `docs/decisions.md`/`STATUS.md` text. Found **Correction 1 itself was
      still wrong**: a bare file-path catch-all (`"kernels/"` in
      "microkernel", `"driver.jl"` in "planning") could still swallow a
      more specific ancestor frame before the walk ever reached it (e.g. a
      generated store body's `macro expansion` leaf, still in
      `src/microkernels/simd.jl`, matched `"kernels/"` immediately instead of
      continuing up to `_store_tile_vector!`). Quantified: `ccsd_t_1_dim16_f32`'s
      true store share was ~75%, not the ~38% Correction 1 reported; the
      executed macro-blocking loop nest was being counted as "planning"
      (a 4-5x inflation over the real one-time `plan_contract` cost). Also
      flagged: a missing store substring (`_store_tile_vector!` doesn't
      contain literal `"store_tile!"`), an overclaimed noise excuse for an
      18-23% gap from a historical GFLOP/s figure, two GFLOP/s numbers with
      no on-disk source, and fixed output filenames silently overwriting
      between repeats. **Second fix**, applied after the review, before any
      commit: two-pass classification (function-name-specific pass across
      the WHOLE stack first, file-level fallback pass only if nothing
      specific matched anywhere), a new `driver_loop` bucket split out of
      `planning`, the missing store substring added, a `--tag` flag so
      repeats don't overwrite each other, and an honest >=15-rep
      `median_time_s`-based GFLOP/s figure printed into every `.buckets.txt`.
      Full case set re-run (both repeats) after the second fix; all numbers
      in `docs/decisions.md` are post-both-corrections.
- [x] T3: ran the full case set (7 adapter cases x 2 backends where
      applicable + 1 direct case), two independent repeats (`--tag r1`/`r2`,
      non-overwriting), on `ccqlin038`, 2026-09-21. Reproducibility: every
      bucket agreed within ~5 percentage points across repeats. Applied the
      fixed verdict rule (compute-bound / kernel-stalled / overhead-bound,
      see `docs/decisions.md`) -- see the table there for the full result,
      including the correction that neither `ccsd_t_1_dim16` nor its
      Float32 twin is compute-bound (a claim this pass's own first draft
      had gotten wrong, before Correction 2).
- [x] Gate: no fix shipped. Nothing found met the bounded-fix bar; the
      `ccsd_t_*` kernel-stalled/store-dominated findings need their own
      fact-finding before a fix is proposed.
- [x] T4: docs written (`docs/decisions.md` new dated section incl. both
      corrections, `STATUS.md` "Next task" corroborating entry rewritten
      post-review, `STATUS.md` "Integrated revision" stale pointer also
      corrected while there).
- [x] T6 (close): commit pending -- see "Remaining" below for exact scope.

## Key artefacts

- `benchmark/profile_to_suite.jl`, `benchmark/profile_buckets.jl` (both
  edited this pass, the latter in two passes -- see Correction 1/2 above).
- `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/`
  (gitignored; final two repeats tagged `-r1`/`-r2` in their filenames, so
  both survive; `.flat.txt`/`.tree.txt`/`.buckets.txt` per case plus
  `buckets_summary-r1.txt`/`-r2.txt`. Earlier, superseded runs from before
  Correction 2 were deleted before the final run, not kept alongside it.)
- `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/summary_store_path.txt`
  (isolated same-day microkernel reference, from `bench_store_path.jl
  --batch 5000 --reps 9 --skip-native`).
- `docs/decisions.md`, "Profiling pass: where does QuasiStrided spend its
  time? (2026-09-21)" -- full table, both corrections, and reconciliation.

## Remaining / next actions

- Everything above is done; the working tree has uncommitted changes ready
  to commit: `benchmark/profile_to_suite.jl`, `benchmark/profile_buckets.jl`,
  `docs/decisions.md`, `STATUS.md`, this file. The merge itself (T0) is
  already committed and pushed separately (`19dd25c`).
- Follow-up candidates recorded in `docs/decisions.md` but not started:
  the `ccsd_t_1_dim16` kernel-stalled finding and its Float32 twin's
  store-dominance finding, re-measuring the isolated reference on Julia
  1.12.6 to rule out a version-specific codegen confound, the
  Octavian-style `dontpack`/`maybeinline` dispatch tiers (STATUS.md's
  pre-existing "Next task"), a wider case/dtype sweep.
