# QuasiStrided.jl status

Durable status record for the main orchestration process. Update at phase
boundaries or before context compaction. History belongs here only as far as
"what phase are we in"; do not let this rot into a changelog — narrative
detail belongs in `docs/decisions.md`.

## Integrated revision

Local git repo at `/mnt/home/ldevos/Projects/QuasiStrided.jl`, `main` branch,
HEAD `5d50a53` ("Phase 4: review since Phase 2b, close 2 coverage gaps,
driver benchmarks"). No remote; not published, registered, or pushed, per
handoff §2/§8.

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

**Final test count: `Pkg.test("QuasiStrided")` → 12627/12627 passing.**

Full narrative for every decision, integration reconciliation, and review
disposition is in `docs/decisions.md` — that file, not this one, is
authoritative for *why*. This file is only *what phase, what count*.

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
- **New, smaller, open item**: `execute!` through the driver still allocates
  (~10.7 KB scalar / ~5.9 KB SIMD for a small multi-tile/multi-panel case;
  down from ~15/~10.2 KB before the fix above, since packing no longer
  contributes). `ScalarKernel`'s 176 B/call `zero_accumulator` is
  spec-accepted (a `Matrix{T}`, explicitly fine for the scalar reference);
  the rest is somewhere in `src/driver.jl`'s own tiling loop, not yet
  diagnosed. This is the next thing to pick up if continuing performance
  work — scope a fresh bounded diagnosis task to `src/driver.jl` the same
  way the packing one was scoped to `src/packing.jl`.
- No cache-blocked macro-kernel, autotuning, threading, or GPU path — all
  explicitly deferred per the handoff, not gaps in this milestone's scope.

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

## Macro-blocking milestone (in progress)

Opened 2026-09-08. Goal: replace `execute!`'s tile-by-tile loop with a
BLIS five-loop (`NC`/`KC`/`MC`) macro-blocking nest with packed-panel reuse.
Full design and orchestration plan recorded in the session's plan file;
narrative decisions in `docs/decisions.md`'s "Macro-blocking milestone"
section (frozen interfaces, block-size policy, allocation root-cause).

`fable_review_macro_used: true` (launched at Phase D; disposition to be
recorded in `docs/decisions.md` once triaged).

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
- [ ] Phase E (benchmark sweep, replace provisional block-size constants)
- [ ] Phase F (final review, docs, CI, merge)

## Next task

Continue the macro-blocking milestone above (Phase B next). Otherwise, a
future session extending scope should start by reading `docs/decisions.md`
in full, then the two specs' "Deferred work"/"Deferred extensions" sections.
