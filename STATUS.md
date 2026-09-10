# QuasiStrided.jl status

Durable status record for the main orchestration process. Update at phase
boundaries or before context compaction. History belongs here only as far as
"what phase are we in"; do not let this rot into a changelog — narrative
detail belongs in `docs/decisions.md`.

## Integrated revision

Local git repo at `/mnt/home/ldevos/Projects/QuasiStrided.jl`, `main` at
`314c47f` ("Fix Runic formatting in bench_driver.jl") — the macro-blocking
milestone's closing state. Published (see "Published" below).

The TensorOperations integration milestone (below) lives in the
`tensoroperations` worktree/branch, based on that same revision. As of this
milestone's documentation close (T13, 2026-09-09) that branch is still *at*
`314c47f`: the entire milestone exists as uncommitted working-tree changes —
11 modified files plus 6 new untracked ones (`src/tensoroperations.jl`,
`src/workspace.jl`, `test/test_tensoroperations.jl`, `test/test_quality.jl`,
`benchmark/bench_tensoroperations.jl`, and the 2026-09-09 benchmark results
directory). Nothing from it is committed, merged into `main`, or pushed, so
the "Published" section below still describes `main` alone and is
deliberately unchanged. Committing/merging is the step *after* T14.

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

## TensorOperations integration milestone — complete

Opened 2026-09-09 on branch/worktree `tensoroperations`, base `314c47f`.
Goal: make the real, external **TensorOperations.jl** the interface users
write against (`@tensor`, `ncon`) and demote QuasiStrided to an internal
contraction backend it dispatches to. QuasiStrided takes a hard dependency
on TensorOperations and defines `QuasiStridedBackend <:
TO.AbstractBackend`; opt-in only (no `select_backend` hook), hard-reject
(never falls back), contraction only. Full design and orchestration plan in
the session's plan file; every frozen decision in `docs/decisions.md`'s
"TensorOperations integration milestone: Phase A direction freeze" section
— that section, not this one, is authoritative.

Review budget: two gated passes, in Phase E (T10, Fable) and Phase F (T14,
Sonnet-High) — both spent, do not relaunch either for this milestone.
`fable_review_tensorops_used: true`. T10's disposition is recorded in
`docs/decisions.md` (the aliasing-order addendum under "Required argument-
checking order in the adapter"); T14's disposition (no blocking/should-fix
findings against shipped code, one should-fix against this file's own
bookkeeping, since fixed) is in `docs/decisions.md`'s close section.

- [x] **Phase A** (T0 — milestone open + interface freeze): direction
      (hard dep, no `ext/`, no upstream PR on the critical path);
      `QuasiStridedBackend` as a plain singleton; `select_backend` not
      hooked; hard-reject for ineligible `tensorcontract!` inputs and for
      all `tensoradd!`/`tensortrace!`; eligibility predicate and the
      load-bearing "conj/`op` is a no-op only because eltype is real"
      invariant; the `_qs_labels` mapping with a numerically verified
      worked example; the adapter's required argcheck order; the three-tier
      public/internal name table (1 exported / 9 `public` / 34 internal);
      two recorded amendments (`ContractWorkspace` + `allocator` keyword;
      default kernel `ScalarKernel` → `SIMDKernel`); extended file-ownership
      table. All in `docs/decisions.md`; no source touched.
- [x] **Phase B** (four-way parallel, disjoint files, all depending on T0
      only): **T1** un-export internals + version-guarded `public` tier +
      fix all consumers (behavior-neutral, pass count held at
      12903/12903); **T3** `ContractWorkspace` (`src/workspace.jl`, new) +
      allocator-backed buffer reuse + lazy oracle buffers + default-kernel
      flip to `SIMDKernel`; **T4** backend struct, `_qs_labels`,
      eligibility, hard-reject `tensoradd!`/`tensortrace!`
      (`src/tensoroperations.jl`, new); **T6** independent TO-comparison
      test suite, authored blind against `StridedNative`/`StridedBLAS`;
      **T7** `Project.toml` deps/compat + CI matrix. The anticipated T1/T4
      collision on `src/QuasiStrided.jl` was reconciled at integration as
      planned.
- [x] **Phase C** (integration-dependent): **T5** workspace pooling wired
      into the backend path — one task-local `ContractWorkspace` per
      `(task, eltype)` on the `DefaultAllocator` path, call-scoped
      `tensoralloc`/`tensorfree!` buffers bracketed by
      `allocator_checkpoint!`/`allocator_reset!` on the explicit-allocator
      path; **T2** edge-case/invariant tests after integration, including
      both allocator paths and a Bumper arena; **T8** Aqua quality gate
      (`test/test_quality.jl`, new) — no piracy or method ambiguity from
      adding methods to TO's generics.
- [x] **Phase D** (T9): head-to-head `@tensor` benchmark across
      `StridedNative()`/`StridedBLAS()`/`QuasiStridedBackend()` over the
      `bench_driver.jl` shape grid, on `ccqlin038` (Cascade Lake), single
      machine, 2026-09-09 — `benchmark/bench_tensoroperations.jl`, results
      in `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-09/`.
      Outcome as expected and recorded honestly: QuasiStrided loses to
      `StridedBLAS` at every measured point (2.4x-5.4x slower for
      `Float64`, 3.4x-10.9x for `Float32`, worst at the smallest shapes)
      while beating `StridedNative` everywhere (5.3x-11.1x `Float64`,
      6.1x-18.7x `Float32`). That is the evidence base for leaving
      `select_backend` unhooked. Per-point ratios re-derived from the
      committed CSV at T13 and tabulated in `docs/decisions.md`'s close
      section.
- [x] **Phase E** (gated review 1 + fixes): **T10** read-only review of
      `_qs_labels` against TO's `pA`/`pB`/`pAB` semantics, the conj/`op`
      invariant, aliasing/argcheck coverage, workspace-reuse staleness, and
      that no path silently falls back instead of throwing — **no blocking
      findings**; two should-fix items, both fixed by **T11**. (1) The
      frozen argcheck order ran `Base.mightalias` on the *raw* `C`/`A`/`B`,
      and `Base` defines no `Base.dataids` for `PermutedDimsArray`, so
      `C = PermutedDimsArray(A, ...)` was silently accepted and produced a
      wrong result; the check moved after the `StridedView` wrap, where
      `dataids` reaches the shared parent. Behavior-preserving reorder for
      every previously-correct outcome; frozen order amended in
      `docs/decisions.md` and pinned by a new regression testset. (2)
      `README.md`'s allocation claim overstated the pooling result —
      corrected to the measured ~3.8 KB/call of `plan_contract` bookkeeping
      on a 64³ `Float64` contraction, rather than an unqualified
      "allocation-free". T11 also fixed doc drift in `docs/decisions.md`
      (the adapter calls `plan_contract` + `execute!` separately, not the
      frozen `contract!` wrapper, because `contract!` cannot pass
      `workspace`/`oracle`). 13167 → 13171 passing, no regressions.
- [x] **Phase F** (docs, close, gated review 2) — **complete**: **T12**
      README + docstrings done (README rewritten around `@tensor
      backend=QuasiStridedBackend()`, direct `contract!` demoted to an
      "engine interface" subsection, flat export table replaced by the
      three-tier table, "Not implemented" stating contraction-only /
      real-`Float32`/`Float64`-only / not-the-default plainly; adapter
      docstrings added). **T13** `STATUS.md`/`docs/decisions.md` milestone
      close done — this entry and `docs/decisions.md`'s "TensorOperations
      integration milestone: close" section are its output. **T14** final
      read-only review (second and last gated pass; no budget for a
      third) done — no blocking or should-fix findings against shipped
      code/tests/benchmarks; one should-fix against this file's/
      `docs/decisions.md`'s own bookkeeping (a stale "nits handed to T14"
      note describing two README defects that had already been fixed
      directly), corrected in `docs/decisions.md`'s close section. **T15**
      optional, non-blocking upstream docs PR to TensorOperations.jl not
      started — deliberately out of scope for this close, may happen any
      time after.

**Final test count: `test/runtests.jl` → 13171/13171 passing** (Julia
1.12.6, `ccqlin038`, re-run at T13 on 2026-09-09). Provisional only in the
sense that T14 has not run yet; no code change is expected from T13.

## Next task

**Milestone closed.** T14 (final gated review) ran and returned no blocking
and no should-fix findings against the shipped code, tests, or benchmarks —
it independently re-verified the test count, every benchmark ratio, the
three-tier API table's counts, the argcheck order, T11's `PermutedDimsArray`
aliasing fix, and git hygiene, all matching what's documented. Its one
should-fix was against this file's and `docs/decisions.md`'s own bookkeeping
(the two README nits noted below had already been fixed directly in
`README.md` between T13 and T14, but the close-section prose still described
them as outstanding); that record has now been corrected in
`docs/decisions.md`'s close section. Full disposition there.

`fable_review_tensorops_used: true`, and the Sonnet-High T14 pass is also
spent — there is no budget for a third review this milestone.

Remaining step: commit and merge (see "Integrated revision" — the branch is
still entirely uncommitted as of this writing). T15 (an optional, non-blocking
upstream docs PR to TensorOperations.jl) is out of scope for that and can
happen any time after.

Every worker must read `docs/decisions.md`'s "TensorOperations integration
milestone" sections in full first; they are authoritative over the plan file
wherever the two differ (they correct the plan's claim that `tensoralloc`
always returns a concrete `Vector{T}`, supersede its
`ManualAllocator()`-backed default pool, and carry T11's addendum moving
the adapter's aliasing check onto the `StridedView`-wrapped operands).

Still deferred beyond this milestone (see README "Not implemented" and
`docs/decisions.md`): threading (state is already organized to not preclude
it — see `docs/decisions.md`'s Phase D finding 4), autotuning across
shapes, GPU, complex-arithmetic methods, K padding, orientation swap,
`tensoradd!`/`tensortrace!` support in this backend.
