# QuasiStrided.jl status

Durable status record for the main orchestration process. Update at phase
boundaries or before context compaction. History belongs here only as far as
"what phase are we in"; do not let this rot into a changelog — narrative
detail belongs in `docs/decisions.md`.

## Integrated revision

Local git repo at `/mnt/home/ldevos/Projects/QuasiStrided.jl`, `main` branch,
HEAD `f9beb3d` ("Fix Phase F review findings: stale docstring and
mislabeled bench CSV") — the macro-blocking milestone's closing commit.
Published (see "Published" below).

## Environment

- Julia 1.12.6. `StridedViews` v0.5.1/0.5.2 and `SIMD.jl` v3.7.2 resolvable
  locally (declared deps); `Strided` v2.6.3/2.6.4 resolvable (test-only dep).
- Reference machine: Xeon Gold 6244, Cascade Lake, AVX-512(f/dq/cd/bw/vl) +
  AVX2 + FMA. All measurements in this project were taken on this one
  machine — no cross-microarchitecture claim is made anywhere.

## Phase status — all complete

- [x] **Phase 0** (bootstrap/freeze): package skeleton, `contract!` signature
      frozen, SIMD dependency chosen.
- [x] **Phase 1** (indexing + independent oracle): `AxisGroup`/`offsets`/
      `fill_offsets!`/`BlockDescriptor`/`normalize_group`, independent
      `CartesianIndices` oracle, real `StridedView` integration.
- [x] **Phase 2** (tiles/packing/scalar kernel): `AffineAxis`/`ScatterAxis`/
      `QSTile`, `pack_a!`/`pack_b!`, `ScalarKernel`. Two integration
      reconciliations (destination-tile type drift, kernel/packing coupling).
- [x] **Phase 2b** (Fable review): 2 blocking findings (missing storage-bounds
      checks before `@inbounds` hot paths) fixed and regression-tested; 1
      allocation finding deferred (not correctness-affecting).
      `fable_review_used: true` — spent, do not relaunch.
- [x] **Phase 3** (SIMD + serial driver): `SIMDKernel` (register-resident,
      `@code_llvm`-verified, 3.9-6.3x vs scalar single-tile), `contract!`/
      `plan_contract`/`execute!` driver (label resolution, output tiling,
      multi-K-panel, beta applied once). Both workers converged on the same
      integration pattern independently — `SIMDKernel` is a drop-in `kernel=`
      swap in the driver with zero driver changes needed, verified.
- [x] **Phase 4** (review + measurement + handoff): Sonnet-High review of
      everything since Phase 2b — no blocking findings; 2 coverage gaps
      closed (SIMDKernel-through-driver allocation test, dangling-in-B label
      test). Driver-level benchmark taken (planning vs. steady-state
      execution cost). README.md finished. This file is the closing record.

Full narrative for every decision, integration reconciliation, and review
disposition is in `docs/decisions.md` — that file, not this one, is
authoritative for *why*. This file is only *what phase, what count*. (Test
count as of this milestone's close: see the macro-blocking section below.)

## What works, measured

- Correctness: full test suite green, including an independent oracle
  (Phase 1), real `StridedView` integration (permuted/sliced/negative-stride/
  zero-stride), two independent review passes, and scalar-vs-SIMD numerical
  agreement (`atol=1e-8`–`1e-10`, not bitwise — expected, documented).
- Performance, single machine only: scalar and SIMD kernels are
  zero-steady-state-allocation and (SIMD) register-resident, confirmed by
  `@allocated`/`@code_llvm` inspection, not assumed from source-level tuples.
  SIMD is 3.9-6.3x faster than scalar at the single-tile level (kc 1-256) and
  faster-with-lower-allocation through the full driver on a 64×64×64 case.
  Driver planning cost (~4.4-6.7 µs/call) is under 3% of execution cost at
  that size and fully amortizable across a reused plan.

## What's known and unresolved

- **RESOLVED 2026-09-08**: `pack_a!`/`pack_b!`'s ~80 B/call allocation
  (Phase 2b finding 5) is fixed — root cause was a missing `where`-bound
  type parameter on the forwarding `transform` argument, causing dynamic
  dispatch. Zero allocation now confirmed for both direct-`KernelDescriptor`
  and `ScalarKernel`/`SIMDKernel`-forwarding call paths. Full account in
  `docs/decisions.md`'s Phase 2b finding 5.
- **RESOLVED by the macro-blocking milestone below**: `execute!`'s
  ~10.7 KB scalar / ~5.9 KB SIMD residual allocation was root-caused (Phase
  A) and closed as a side effect of the macro-blocking rewrite (Phase C) —
  see that section below and `docs/decisions.md`.
- No autotuning, threading, or GPU path — still explicitly deferred, not
  gaps in scope. Cache-blocked macro-kernel is now implemented (see below).

## Published

Pushed to `https://github.com/lkdvos/QuasiStrided.jl` (public), `main`
branch, CI green (test matrix: Julia lts/1 x ubuntu/macos; FormatCheck:
runic). Repo also has: MIT `LICENSE`, full `Project.toml` compat bounds,
concise `README.md`.

One thing CI caught that local testing (Julia 1.12.6 only) had missed:
`SIMDKernel`'s `accumulate`/`execute_tile!` are zero-allocation on Julia
1.12 but allocate tens of KB/call on Julia 1.10 (LTS) — a compiler
capability gap (the `NTuple{NV,Vec{W,T}}` accumulator isn't kept
register-resident on the older compiler), not a correctness issue. Handled
by marking the two allocation assertions `skip=(VERSION < v"1.11")` in
`test/test_simd_kernel.jl` rather than weakening or deleting them, so the
gap stays visible rather than hidden. `julia = "1.10"` compat is otherwise
honored (all 13026 correctness assertions pass on 1.10).

## Macro-blocking milestone — complete

Opened and closed 2026-09-08. Goal: replace `execute!`'s tile-by-tile loop
with a BLIS five-loop (`NC`/`KC`/`MC`) macro-blocking nest with
packed-panel reuse. Full design and orchestration plan recorded in the
session's plan file; narrative decisions in `docs/decisions.md`'s
"Macro-blocking milestone" section (frozen interfaces, block-size policy,
allocation root-cause, Phase D/E/F dispositions).

`fable_review_macro_used: true` — spent, do not relaunch for this
milestone. Disposition recorded in `docs/decisions.md`'s "Phase D" section.

- [x] **Phase A** (scout + diagnose): interface/CPU inventory done; the
      residual `execute!` allocation noted below (~10.7 KB scalar / ~5.9 KB
      SIMD) is now root-caused — `axis_from_descriptor`'s
      `Union{AffineAxis,ScatterAxis}` leaves `QSTile`'s type parameters
      unresolved, forcing heap-allocated tiles and a dynamic
      `execute_tile!` call that boxes `alpha`/`beta`. Fix folded into the
      Phase C rewrite (see `docs/decisions.md`).
- [x] **Phase B** (interface additions): `describe_block`/`axis_from_descriptor`
      `first`-offset overloads; `pack_a!`/`pack_b!` widened to
      `AbstractVector{T}`. 12777/12777 passing.
- [x] **Phase C** (macro-kernel rewrite + independent oracle): `execute!`
      rewritten as a BLIS five-loop (`NC`/`KC`/`MC`) nest with packed-panel
      reuse (`src/blocking.jl`, `src/driver.jl`); `execute_tilewise!` kept
      unexported as the pre-existing-behavior oracle; independent
      `test/test_macro_driver.jl` written blind to the implementation and
      integrated. Closed the Phase A allocation bug as a side effect:
      `execute!` measured 0 B (SIMDKernel) / 176 B-per-tile-only
      (ScalarKernel, the already-accepted `zero_accumulator` cost) — no
      residual driver-induced allocation. `@code_warntype` confirmed no
      `Union`/partially-applied `QSTile` downstream of axis construction.
      12901/12901 passing.
- [x] **Phase D** (Fable review): no blocking findings. One should-fix
      (missing permanent test coverage for the irregular-sliver-at-
      nonzero-offset driver path) fixed and verified; three notes accepted
      without action (one pre-existing, unchanged behavior; two design
      observations for a future threading milestone). Full disposition in
      `docs/decisions.md`. 12903/12903 passing.
- [x] **Phase E** (benchmark sweep, replace provisional block-size
      constants): `benchmark/bench_driver.jl` written and run on
      `ccqlin038` (Cascade Lake), single machine, 2026-09-08 — 7 shapes x a
      36-combo `(mc,kc,nc)` grid per dtype x `ScalarKernel`/`SIMDKernel`.
      `default_blocking` now returns measured values (`Float64:
      mc=64,kc=128,nc=768`; `Float32: mc=96,kc=384,nc=1152`), no longer
      `# PROVISIONAL`. `execute!` faster than `execute_tilewise!` at all 28
      measured `(kernel,dtype,shape)` points (1.66x-10.5x). Full provenance
      in `docs/decisions.md` ("Phase E") and
      `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/`.
      `Pkg.test()` after the constant swap: 12903/12903 passing.
- [x] **Phase F** (final review, docs, CI, merge): Sonnet-High review of
      everything since Phase D — no blocking findings; two should-fix
      items (a stale "provisional" docstring in `src/blocking.jl` after
      Phase E's constants landed, and a committed benchmark CSV whose
      `reps` column was mislabeled `5` instead of the actual `9` used)
      fixed and verified; independently re-derived and confirmed the
      Phase D should-fix test's core claim (`describe_block` returns
      `regular=false` at `first=4` for the intended fixture) rather than
      trusting the prior write-up. README/STATUS/`docs/decisions.md`
      updated to describe the shipped macro-blocking driver and measured
      defaults. `Pkg.test()`: 12903/12903 passing.

**Final test count: `Pkg.test("QuasiStrided")` → 12903/12903 passing.**

## Next task

None outstanding for the macro-blocking milestone's scope. Deferred items
for a future session (see README "Not implemented" and
`docs/decisions.md`'s "Explicitly deferred" note): threading (state is
already organized to not preclude it — see `docs/decisions.md`'s Phase D
finding 4), autotuning across shapes, GPU, complex-arithmetic methods, K
padding, orientation swap. Start by reading `docs/decisions.md` in full.
