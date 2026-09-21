# QuasiStrided.jl status

Durable status record for the main orchestration process. Update at phase
boundaries or before context compaction. History belongs here only as far as
"what phase are we in"; do not let this rot into a changelog — narrative
detail belongs in `docs/decisions.md`.

## Integrated revision

Local git repo at `/mnt/home/ldevos/Projects/QuasiStrided.jl`. `main` is at
`a14e334`, following (in order) the merge of `upstream-bench` (PR #5,
`19dd25c`), the profiling-pass benchmark tooling (`f318eb9`), the
packing-speed vectorized fast path (`8dd01dd`), the extended profiling
grid (`ebfbeb3`), the dispatch-tiers design proposal (`7f6a4cb`, docs
only), and the `ccsd_t_*` kernel-shape-demotion/inline-accumulate fix
(`a14e334`) -- all 2026-09-21, on top of the label-order milestone (PR #7,
`49213bc`), the vectorized store fast-path fix (PR #6, `4ff065a`), and
complex element-type support (PR #4, `71c1536`), on top of the original
TensorOperations integration (PR #2, `3e712ea`, squash-merged 2026-09-10).
All published, all on `main`; there is no separate long-lived work branch
as of this entry (three local worktrees from the 2026-09-21 work --
`ccsd-t-stall`, `profile-grid`, `packing-speed` -- are fully merged and can
be removed on request). Note
`Manifest.toml` and `benchmark/results/` are both gitignored, so the
committed manifest is stale relative to `Project.toml` and a fresh clone
needs `Pkg.resolve()`; benchmark result directories exist on disk but are not
in the tree.

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
- **CORRECTED 2026-09-11**: the standing claim that "the microkernel gap
  against a tuned vendor GEMM is not closing with size" is **false**. At the
  BLIS `skx` register shape the pure-Julia `SIMDKernel` reaches 101-103
  GFLOP/s, ~88% of this machine's peak. The binding constraint is that the
  driver passes each micro-tile a `view` of its macro panel, which costs ~4x
  above `NV = 16`. See "Next task" and `docs/decisions.md`'s Phase G.

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

## Hardware-derived register shape milestone — complete

Goal: make the engine perform sensibly on any machine without per-machine
retuning. Narrative, measurements and provenance in `docs/decisions.md`,
"Hardware-derived register shape milestone: Phase G" — that file, not this
one, is authoritative for *why*.

Shipped:

- `src/target.jl` (new): runtime hardware detection with **no new
  dependency** — `Sys.CPU_NAME` against a recognition table of LLVM uarch
  names, falling back to a `Base.BinaryPlatforms.CPUID` probe, then to
  `:unknown`. Plus `cache_topology()`, which reads Linux sysfs / macOS
  `sysctl` on demand and reports cache size, ways, line size and **sharing**.
  Detection runs once per process in `__init__`, never at precompile time.
  `Project.toml` `[deps]` is unchanged.
- A single capability-derived register-shape rule in `src/driver.jl`:
  `W = vector_bytes/sizeof(T)`, `MR = 2W`, `NR = 6` (so `NV = 12`). It
  reproduces the swept optimum for both dtypes independently on AVX-512, and
  reduces to the old hardcoded `(8,6,4)` on AVX2. Applied only on those two
  measured ISAs (`_rule_applies`); `:neon` and `:unknown` take the previous
  constants bit-identically, so aarch64 is unchanged rather than guessed at.
- Extent-aware demotion: `_default_kernel(T, Qm, Qn)` falls back to the
  legacy shape when `Qm < MR`, applied only when the caller did not name a
  kernel. `plan_contract`'s `kernel` keyword now defaults to `nothing` and is
  resolved after the M/N/K groups are built, behind a `_plan_contract`
  function barrier that keeps `ContractPlan`'s `Kern` parameter concrete.
- `default_blocking` gained one ISA-keyed measured row (AVX-512), with the
  Phase E constants as the fallback for every other ISA.
- `benchmark/harness.jl` (new, factored out of `bench_driver.jl`),
  `benchmark/bench_kernel_shape.jl` (new, the register-shape sweep with a
  validated spill detector), `benchmark/bench_default_vs_legacy.jl` (new,
  the shipped-vs-previous regression guard). `bench_driver.jl` now sweeps at
  the derived shape.
- `test/test_target.jl` (new, 128 tests).

Measured, single machine (ccqlin038, Cascade Lake), complete new
configuration against the complete previous one: **geomean 1.264x (Float64)
and 1.274x (Float32)**, up to 1.78x/1.91x at 512³, worst point 0.963x
(inside this session's 4-12% canary spread). On Julia 1.10 LTS the speedups
are larger (1.79x/1.87x) and both paths stay allocation-free.

Test suite green on both CI Julia versions: Julia 1.12.6 (13171 pre-existing
tests all still passing, plus 128 new in `test/test_target.jl`), and Julia
1.10.11 LTS (13687 pass, 9 broken — the pre-existing
`skip=(VERSION < v"1.11")` allocation assertions, unchanged by this work).

## Panel addressing milestone — complete

Goal: remove the two ceilings Phase G identified, prompted by reading
Octavian.jl. Narrative and measurements in `docs/decisions.md`, "Panel
addressing milestone: Phase H".

Shipped:

- `src/panel.jl` (new): `PackedPanel`, a borrowed `(Ptr{T}, len)` handed to
  the kernel instead of `view(ws.packed_a, range)`, with
  `panel_vload`/`panel_load`/`panel_store!` accessors that also have
  `AbstractVector` methods so every existing caller keeps working. Needs **no
  new dependency** — `SIMD.vload` already accepts a `Ptr{T}`. `execute!` wraps
  the nest in one `GC.@preserve ws`; the nest moved to `_execute_nest!`.
- `PtrScatterAxis` in `src/tiles.jl` (new): `Ptr{Int} + count`, so it is
  `isbits` and `Union{AffineAxis,PtrScatterAxis}` needs no heap box.
  `ScatterAxis` is unchanged and stays the vector-backed public form; only
  `_axis_of` switched.
- A generated, statically-indexed scattered store path in
  `src/kernels/simd.jl`, replacing dynamic `NTuple` indexing in
  `store_tile!`'s fallback.
- `benchmark/bench_default_vs_legacy.jl` and the register-shape sweep re-run;
  `_shape_override` added as a hook and left deliberately **empty**.
- `test/test_target.jl` grew a zero-allocation assertion through a
  permuted-A / negative-stride-B / sliced-C fixture.

**A Phase G default was allocating and the suite missed it.** The Float32
default `(32,6,16)` allocated 24576 B per `execute!` on scattered
contractions; every existing allocation assertion used plain regular
contractions, so nothing caught it. Fixed by `PtrScatterAxis`, and the new
scattered assertion closes the gap.

Measured: 0 B on every shape up to `MR = 48` (was 24 KB above `MR = 16`);
microkernel 62.6 → 79.5 GFLOP/s at the shipped shape and 30.7 → 96.6 at
`(16,14,8)`. **End-to-end geomean is unchanged** at 1.283 (Float64) against
the pre-Phase-G configuration. These fixes removed ceilings; they did not add
throughput at the shipped shape.

Test suite: 13299/13299 on Julia 1.12.6.

## `ccsd-t-stall` branch: kernel-stalled/store-dominated fix, complete (2026-09-21)

On the `ccsd-t-stall` worktree/branch (based on `main` @ `f318eb9`, before
`8dd01dd`'s packing fast-path landed on `main` -- rebase before merging):
both mechanisms found by the profiling pass below (F2, run-length-aware
kernel-shape demotion; F1, `@inline` on `SIMDKernel`'s `Base.accumulate`)
are now implemented in `src/driver.jl`/`src/kernels/simd.jl`, each with a
measured before/after (~2x on `ccsd_t_1_dim16_f32`; ~9% on the isolated
small-`kc` case), full suite green, and an ABBA guard showing no regression.
Post-review: a blocking ISA-specific test hardcoding was fixed and verified
portable via `test/forced_isa_runner.jl` (avx2/unknown), the F2 test fixture
was shrunk, and **F2 was found to regress large-K compute-bound cases by
~18-25%** (no cost model gates the demotion on `Qk`) -- documented as a known
limitation, not fixed this pass. See `docs/decisions.md`, "Kernel-stalled/
store-dominated fix: run-length-aware demotion (F2) and inlined `accumulate`
(F1)" and its "Post-review fixes" subsection.

## Next task

**Superseded 2026-09-21, by user decision: do not build Octavian-style
`dontpack`/`maybeinline` dispatch tiers.** The paragraphs that used to stand
here framed that as the highest-value next item, based on a plain-matmul
comparison against Octavian.jl/OpenBLAS. A profiling pass, a packing-speed
fix, a `ccsd_t_*` mechanism fix, and a workload-representativeness analysis
(all 2026-09-21, full narrative in `docs/decisions.md`) together showed that
framing doesn't hold on this project's actual workload: the one real,
large, reproducible loss case (`ccsd_t_*_dim16`) had the *highest* packing-
reuse numbers in the whole measured sweep and a packing cost near zero --
its cause was a store-path/kernel-shape defect, now fixed, unrelated to
packing amortization. On the 3 cases in the profiling passes that are
genuine multi-index contractions (not plain matmul with a suggestive name),
packing is now 1-11% of runtime, and QuasiStrided already matches or beats
TBLIS and beats StridedBLAS by 1.0x-2.4x. `docs/proposals/dispatch-tiers.md`
is the sign-off document; the user accepted its recommendation against
building the tiers now. Do not restart this item without new evidence
against that recommendation (see the proposal's evidence gate, below).

**Current next task, two tracks, both authorized 2026-09-21:**

1. **Evidence gate** (`docs/proposals/dispatch-tiers.md` section 5.1):
   confirm the recommendation above holds on the upstream `:mps`/`:ctmrg`/
   `:trg` categories, never before measured against this engine. An sbatch
   script for this is prepared for the user to submit on Rusty/Popeye; not
   run as part of this session (Slurm job submission is the user's own
   action per this project's operating constraints). Wires those categories
   into `benchmark/bench_to_suite.jl` first (previously only `:pairwise`/
   `:tccg` were runnable).
2. **Per-call floor** (`docs/proposals/dispatch-tiers.md` section 5.2): the
   residual loss pattern found by the analyses above is per-call/per-block
   overhead, not packing -- `plan_contract`'s ~3.8-4.4 KB/~4.4-6.7 us
   allocation-heavy label bookkeeping, per-sliver validation
   (`_check_pack_a`/`_check_pack_b` + `checked_tile_storage_bounds`, 7.4% of
   `ao2mo_2_dim16`), and `driver_loop` bookkeeping (`fill_offsets!`/
   `describe_block`/`_classify_slivers!`, 28.7% of `ao2mo_2_dim16`).
   User has explicitly authorized hoisting the per-sliver validation to
   once-per-macro-block even though this changes `pack_a!`/`pack_b!`'s
   documented "all validation before any write" contract, on the condition
   that the resulting now-unchecked internal entry points are named with an
   `unsafe_` prefix so the shifted safety contract is visible at every call
   site. See `docs/decisions.md` for this track's outcome once it lands.

Two smaller follow-ons, both unblocked but not urgent: the register tile
can be enlarged (`NV` up to 28 is allocation-free and spill-free -- it just
measured no faster), and `default_blocking`'s `mc`/`nc` were validated at
the previous register shape.

Full narrative for the 2026-09-21 profiling pass, the packing-speed fix,
the `ccsd_t_*` mechanism fix, the full-grid extension, and the
dispatch-tiers design review is in `docs/decisions.md` -- this file only
records what phase the project is in, per its own stated purpose at the
top.

**Measurement hygiene, learned the hard way this milestone:** ccqlin038 is
not reliably exclusive. Canary spreads of 4-15% were normal (against Phase
E's 0.71%), three successive register-shape sweeps produced three different
"winners", and an 11-rep comparison invented two regressions that 21 reps
erased. Use >= 15 reps, time compared configurations adjacently, check for
other users' processes first, and treat anything under ~10% as noise.

Still deferred (see README "Not implemented" and `docs/decisions.md`):
threading (state is still organized to not preclude it, but note
`PackedPanel`/`PtrScatterAxis` borrow pointers into workspace buffers, so a
threaded driver must keep the `GC.@preserve` and the per-worker split of the
M-side state consistent), autotuning across shapes, GPU, K padding,
orientation swap, `tensoradd!`/`tensortrace!`. T15 (an optional upstream docs
PR to TensorOperations.jl) remains unstarted. **Complex-arithmetic methods are
no longer deferred** -- they are the open milestone below; what stays out of
scope there is 3m, mixed real/complex operands, and writing into a conjugated
output `C`.

**BLIS microkernels stay closed** (Phase G): all of `blis_jll` 0.9/1.0/2.0
ship the asm kernels and `bli_cntx_get_*` as local symbols only, so `dlsym`
cannot reach them. **LoopVectorization.jl was considered and rejected** as a
dependency (grant-funded maintenance, compiler-fragile, and unnecessary — the
addressing idea was reproduced with zero new dependencies).

## Complex element-type milestone — open

Opened 2026-09-14 on branch/worktree `complex`, base `114e594`. Goal: support
`ComplexF32`/`ComplexF64` end to end — engine and `QuasiStridedBackend` — with
two switchable microkernel methods, and discharge the conjugation invariant that
has blocked this since the TensorOperations milestone.

Full design, every frozen decision, and the reasoning behind each rejected
alternative are in `docs/decisions.md`, "Complex element-type milestone: Phase A
direction freeze" and "Amendment 3" — **that file, not this one, is
authoritative for *why***. This section is only *what phase, what count*.

Direction in one paragraph: planar (split-complex, BLIS "1r", a genuinely
complex microkernel issuing four real FMAs with no shuffles) is the
unconditional default; 1m (Van Zee's induced method — the *existing real kernel*
over `2*kc` real steps, fed by "1e"-packed A) is selectable by naming the
kernel; 3m is out. The design is taken from the user's sibling project
`tensorcontract-rs`, which was built to measure exactly these methods and whose
refutations are binding here — in particular that **the method ranking does not
transfer between machines** (four orderings measured on four machines), so no
auto-dispatch rule is derived from any sweep and every ratio names `ccqlin038`.

Review budget: two gated passes, one Fable scoped to the conjugation/`op`
semantics and the frozen-record amendment (a wrong `_op_conjugates` entry is a
silent wrong answer, which is this package's worst failure mode), then one
Sonnet-High over everything since Phase A. Neither is spent yet;
`fable_review_complex_used: false`.

- [x] **Phase A** (direction freeze): `docs/decisions.md` milestone section +
      Amendment 3 + the second addendum to the frozen argcheck order;
      `STATUS.md` entry; `src/complex_format.jl` (new) with declarations and
      total real-side defaults only — `PackFormat`/`ComplexMethod` singletons,
      `reals_per_element`, `a_reals`/`b_reals`/`accumulator_planes`,
      `ComplexKernelDescriptor` and its accessors, and the
      `realtype`/`packed_a_per_k`/`packed_b_per_k`/`complex_method` defaults
      plus `DescriptorKernel` forwarders that keep every generic total. **No
      behaviour**: `git diff` touches no existing function body, and the only
      edit to an existing source file is one `include` line in
      `src/QuasiStrided.jl`.
- [ ] **Phase B** (five-way parallel, disjoint files, all depending on A only):
      **B1** plumbing (`src/workspace.jl`, `src/driver.jl`, `src/blocking.jl`) —
      the `VT` bound relaxation, `realtype`-driven buffer allocation, the four
      `_sliver_panel` call sites, `_pack_sliver!`'s `TF` parameter,
      `ContractPlan`'s `TA`/`TB`, the complex shape rule and menus, and the
      3-argument `default_blocking`. `src/driver.jl` is **exclusive** to B1.
      **B2** packing (additive only; `_pack_panel!` untouched). **B3** the
      planar kernel. **B4** the adapter. **B5** adapter tests, authored blind
      against the Phase A freeze.
- [x] **Phase C** (integration): B1–B5 merged; `_complex_kernel_from_shape`'s
      `PlanarMethod` arm wired (generated over that method's own menu, so every
      branch builds a concrete kernel from literal `Val`s); the method-generic
      arm still throws, so an unimplemented method can never silently fall back
      to planar. Three reconciliations, all recorded in `docs/decisions.md`'s
      "Phase C integration findings": the adapter's frozen argcheck order was
      violated by keyword evaluation (a pooled workspace was acquired before a
      rejected call could throw — found by the blind test author, whose own
      assertion was passing for the wrong reason); complex kernel *construction*
      is now ISA-gated as well as the shape rule, since the legacy complex
      fallback was over AVX2's register budget; and B1's seam scaffolding test
      was replaced by its inverse. **20586/20586 passing** (Julia 1.12.6), Runic
      clean, 0 failed / 0 errored.
- [x] **Phase D** (1m): `src/kernels/onem.jl` (new). The freeze's "nearly free"
      claim holds — the FMA loop, packing, lengths, sliver addressing, blocking
      and `execute_tilewise!` are all reused unchanged, and the only genuinely
      new logic is a ~25-line generated `OneM` tile reader. Two freeze defects
      found: the frozen `inner::SIMDKernel{2MR,NR,real(T),W}` field spelling is
      **not legal Julia** (computations on `TypeVar`s), and **`W` must be even**
      — a requirement the freeze never stated, and one whose violation would
      corrupt the tile reader silently rather than error. Both fixed and tested.
      Every shipped 1m shape is spill-free; 1m is *better* behaved than planar
      on Julia 1.10.
- [x] **Phase E** (end-to-end randomized oracle): `test/test_macro_driver.jl`
      gained the fourth oracle layer — 501 randomized complex cases against a
      dense-matmul oracle with its **own** conjugation table, the conj/`op`
      cross-product drawn rather than enumerated under a fixed seed. Pins the
      xor (not `||`), pins that `adjoint` conjugates with no flag set (which an
      `op === conj` implementation fails *only* here), and pins `execute!`
      against `execute_tilewise!` with conjugation forced on. Cache-crossing
      extents derived from each method's own blocking, crossing asserted.
      Purely additive: 448 insertions, 0 deletions.
- [x] **Phase F** (measurement on `ccqlin038`, 21 reps, canary spread 0.4-3.0%).
      **The derived register-shape rule was wrong for complex by 38-41%.** The
      shape `MR = 2W, NR = 6` derives was the *worst* planar configuration
      measured; the spill-free `24x3`/`48x3` shape that Phase C's spill
      analysis had flagged wins outright, and beats every 1m shape too — so the
      spill analysis predicted the ranking before the ranking was measured.
      Acted on through the `_shape_override` hook that has been deliberately
      empty since Phase G; these are the **only swept rows in the package**, and
      they apply on `:avx512` only. Complex-efficiency geomean went
      **1.256 → 1.829** (`ComplexF64`) and **1.457 → 1.910** (`ComplexF32`),
      and the sub-1.0 dip at the large compute-bound sizes disappeared
      (`ComplexF64` 512³ 40.5 → 73.7 GF/s, +82%). Real-path guard: **no regression
      detectable at the instrument's ~5-6% resolution**, which is all it can
      support — the earlier "pooled geomean 0.988" is retracted, since the
      between-tree effect (2.0%) is smaller than the same-tree run-to-run
      noise (up to 6.3% geomean) and the sign flips between rounds. No
      systematic one-sided shift, and the resolved kernel shape is identical
      at every point in both trees. Planar stays the default; 1m stays selectable only by name.
- [x] **Phase G** (the two gated reviews, both spent —
      `fable_review_complex_used: true`). The Fable pass found **no blocking
      numerical finding**: 1352 adapter + 512 direct-engine + 4 macro cases
      against oracles it wrote itself, zero failures, and it hand-verified the
      1e conjugated layout via the observation that the 2×2 block is the real
      matrix representation of "multiply by z", so substituting `conj(z)` gives
      exactly `M(z)ᵀ`. It also established something stronger than the freeze
      claimed: **xor is TensorOperations' own semantics**, forced by TO
      realising `conjA` as `conj(SV(A))` and StridedViews flipping `op` through
      its `_conj` table — not merely TBLIS's convention. Four record/doc
      defects found and fixed (two false claims in Amendment 3, a
      self-contradicting source comment, a stale docstring default).
      The Sonnet pass found **one blocking finding and it was right**: the
      real-path guard's quoted geomean was not substantiated by the artefacts
      on disk. The number is retracted and the conclusion narrowed — see below.
      It independently re-measured zero allocation in all 24 (method × shape ×
      precision × function) cells on a *scattered* fixture.
- [x] **Phase H** (close): `docs/decisions.md` carries the freeze, Amendments
      3 and 4, the Phase C/D/F findings and corrections, and the Phase G
      disposition; `README.md` describes complex support as a capability
      (including the measured efficiency ratio and the one machine-specific
      constant in the package) and narrows its "Not implemented" list to what
      is genuinely still out of scope; this section is the closing record.

**Milestone closed.** `ComplexF32`/`ComplexF64` work through both `contract!`
and `@tensor backend = QuasiStridedBackend()`, with `conjA`/`conjB` and
`StridedView.op` composed by xor and absorbed at pack time.

### What shipped

- `src/complex_format.jl` (new): the packed formats (`RealFormat`,
  `PlanarFormat` = BLIS "1r", `OneEFormat` = BLIS "1e"), the method singletons,
  `ComplexKernelDescriptor`, and the accessors that keep every generic total on
  the real path (`realtype`, `packed_a_per_k`, `packed_b_per_k`,
  `complex_method`). `src/kernel_descriptor.jl` is **untouched**: the frozen
  packed format is preserved as the `RealFormat` instance of a more general
  offset formula, not redefined.
- `src/kernels/planar.jl` (new): the default. A genuinely complex microkernel,
  four real FMAs per (A-vector, B-scalar) pair with no shuffles, flat
  `NTuple{2NV,Vec{W,real(T)}}` accumulator, every body `@generated` with
  literal tuple indices.
- `src/kernels/onem.jl` (new): Van Zee's induced method, reusing the **real**
  `SIMDKernel` verbatim over `2*kc` real steps. Selectable only by naming it.
- Conjugation through the `transform` seam that already existed, folded with
  `.op` at plan time and stored as `ContractPlan`'s `TA`/`TB`; a conjugated
  *output* is rejected.
- `ContractWorkspace{T,VT}`'s bound relaxed (not extended) so every existing
  `ContractWorkspace{Float64,Vector{Float64}}` spelling still type-checks.
- The one machine-specific constant in the package: `_shape_override` for
  complex on `:avx512` only.

### Measured (ccqlin038, Cascade Lake, Julia 1.12.6, 21 reps)

- **Complex efficiency geomean 1.83 (`ComplexF64`) / 1.91 (`ComplexF32`)** —
  complex GFLOP/s over the same engine's real GFLOP/s at the same shape, with
  complex charged the textbook 8 flops/MAC. Above 1.0 means complex is treated
  *better* than real, as twice the arithmetic intensity predicts.
- **No real-path regression detectable** at the guard's ~5-6% resolution.
- **Zero allocation** in all 24 (method × shipped shape × precision ×
  `accumulate`/`execute_tile!`) cells on a scattered fixture, independently
  re-measured at review.

### The two things most worth remembering

1. **The hardware-derived register-shape rule did not survive the complex
   extension.** It is arithmetically sound and reproduces the reference
   project's measured shape, yet on this machine through Julia's register
   allocator it selects the *worst* planar configuration by 38-41%. Phase C's
   spill analysis predicted that before Phase F measured it. The lesson is the
   project's own "ranking does not transfer between machines", applying one
   level lower than it was written for: a shape validated through one
   compiler's register allocator need not hold under another's.
2. **A claim about evidence failed review where no claim about code did.**
   Both passes probed the conjugation logic hard and found nothing wrong with
   it; the one blocking finding was that the real-path guard's quoted number
   was not substantiated by the files on disk. It was retracted. Benchmark
   artefacts that do not say which tree they measured are not evidence.

### Still out of scope on the complex side

3m; mixed real/complex operands; writing into a conjugated output `C`; complex
register shapes for AVX2 or NEON (the engine refuses to pick a complex kernel
on an unmeasured ISA rather than ship a guaranteed-spilling default); the
unit-stride plane-to-interleave store (measurement-gated, not yet justified);
complex `tensoradd!`/`tensortrace!` (contraction-only is unchanged).

### Open, deliberately

"Which spill detector is right" — Phase C's versus Phase D's — is not settled
and no longer needs to be: Phase F ranked on measured throughput, which
supersedes the spill-count question, and both detectors agreed on the shape
that mattered. It is a disagreement about an instrument nothing shipped now
depends on.

**Phase D also corrected Phase C.** Phase C's planar spill table presented
spilling as monotone in a single pressure number; Phase D's detector — which
counts folded FMA reload operands and `rbp`-relative traffic, and which
reproduces both of Phase C's *real* controls exactly — shows it is not:
planar `(24,3,8)` at pressure 26 is clean while `(8,8,8)` at pressure 20
spills. So the frozen budget inequality is a necessary condition, not a
predictor. The menu order still stays untouched and unranked, which the
correction vindicates rather than undermines: the quantity it would have been
reordered on was being misread. Full account in `docs/decisions.md`.

**Test count: 13299 at Phase A open → 20586 after Phase C → 34480 after
Phase F** (Julia 1.12.6, 0 failed, 0 errored, Runic clean).
B1's acceptance criterion — the real path unchanged with no complex kernel
wired in — was proved on an isolated tree: pristine `7503fdd` 13299/13299,
base + B1 alone 13612/13612, 0 failed, 0 errored. The measured half of that
guard (`benchmark/bench_default_vs_legacy.jl` inside the canary spread) is
Phase F's first sweep and gates the rest.

Verified end to end at Phase C close, independently of the in-tree suite: all
four element types against `mul!`; the full 16-combination `conj`×`op`
cross-product against **both** `StridedNative()` and a hand-written oracle, 0
mismatches; the xor in isolation (`conj(A)` with `conjA = true` must give the
*unconjugated* product); a genuine 3-index contraction on permuted/scattered
operands; conjugated complex output rejected while a real `adjoint` output is
still accepted; β = 0 not propagating NaN from a poisoned `C`. Backend
allocation is **4432 B/call for complex, identical to real's 4432** — the
`plan_contract` bookkeeping the README already documents, so complex adds none
of its own.

Two notes for anyone picking this up on a fresh checkout of this worktree.
`Manifest.toml` is gitignored, so the environment must be resolved before
anything runs. And on the *first* full run after that, Aqua's
`test_persistent_tasks` can fail on a timeout: it loads the package in a
subprocess, which on a cold depot has to precompile first (observed here at
2m24s, against 5.5s once warm). It is not a real failure — re-run the suite,
and check it in isolation before believing it.

## Upstream TensorOperations.jl benchmark-suite comparison -- preparatory, complete

Opened 2026-09-15 on branch `upstream-bench`, base `71c1536`. Goal: a working
three-way (`StridedNative`/`StridedBLAS`/`QuasiStrided`) comparison driven by
TensorOperations.jl PR #303's unmerged `TensorOperationsBenchmarks` suite
(`:pairwise` + `:tccg` categories only), plus a first-look profiling triage.
No engine change; this is a **preparatory milestone**, not an optimization
pass. Full design, every measured number, and the profiling triage's
SHOWS/SUGGESTS reasoning are in `docs/decisions.md`, "Upstream
TensorOperations.jl benchmark suite comparison: preparatory milestone" --
that file, not this one, is authoritative for *why*.

- [x] **T0** (scouting): upstream API + dependency fact-finding, read-only.
- [x] **T1** (`benchmark/Project.toml`): pinned `TensorOperationsBenchmarks`
      to PR #303's unmerged commit via `[sources]`.
- [x] **T2** (`benchmark/composite_backend.jl`): benchmark-only
      `QuasiStridedComposite`, does not touch `QuasiStridedBackend`'s frozen
      hard-reject invariant.
- [x] **T3** (`benchmark/bench_to_suite.jl` + measurement run): 118 cases,
      354 timed rows, 21 reps, `ccqlin038.flatironinstitute.org` 2026-09-15.
      Zero mismatches, zero backend rejections.
- [x] **T4** (`benchmark/profile_to_suite.jl`): bucketed profiling tool.
- [x] **T5** (profiling triage): 4 cases profiled x 2 backends. Two
      separable causes found for the `ccsd_t_*_dim16` regression, both
      unverified/untested.
- [x] **T7** (this entry + the `docs/decisions.md` section).
- [ ] **T6** (independent review) -- not yet run.
- [ ] **T8** (address T6 findings, commit, PR) -- not yet run.

**Milestone (T0-T5, T7) is complete as a preparatory milestone.** T6/T8
(review and any resulting fix-ups/PR) are the coordinator's next step, not
part of this section's scope.

### What shipped

`benchmark/Project.toml` (new; pins `TensorOperationsBenchmarks` to PR #303's
commit `528dd85d8bf886c734a207732a7cb591a3691dd3`), `benchmark/composite_backend.jl`
(new), `benchmark/bench_to_suite.jl` (new), `benchmark/profile_to_suite.jl`
(new).

### Measured

Correctness: clean -- 0/118 mismatches (rtol 1e-10 F64 / 1e-5 F32 against
`StridedBLAS`), 0 backend rejections, across `:pairwise` (11 cases x 3 dims)
and `:tccg` (48 cases x 2 dims) x 2 dtypes. Performance: QuasiStrided wins
outright on 32/118 cases (real chemistry `:tccg` shapes at dim16, e.g.
`ao2mo_2` 0.315x BLAS's time), and beats `StridedNative` on 83/118; one
substantive throughput regression class found -- the four `ccsd_t_*_dim16`
six-index-output cases, 6.6-14.2x slower than `StridedBLAS` and 2.0-4.4x
slower than plain `StridedNative` too (the only case class where that
happens at a non-trivial absolute size) -- triaged by profiling to two
separable, unverified causes: (A) the vectorized store fast-path guard
(`src/kernels/simd.jl:217`) is unsatisfiable for any `Array`-backed
destination, not just on the TensorOperations path (`src/driver.jl:815`
hands it `Memory{T}`, never `Vector{T}`, on every plan-construction call)
and so every case pays for the scattered-store path unconditionally; (B)
`ccsd_t_1`'s 134.2 MB output has a non-monotonic GEMM-M stride pattern,
giving 36-41x
higher per-element store cost than a cache-resident output. Both are read-
only findings (`src/` was read, not edited) and neither is confirmed by a
second reviewer.

### Still out of scope

Repointing `TensorOperationsBenchmarks` to a registered release once PR #303
merges; fixing the `store_tile!` `Vector{T}`/`Memory{T}` guard (Cause A);
any engine change for the `ccsd_t_*` regression class (Cause B); the
remaining upstream categories (`:permute`, `:trace`, `:mixed_precision`,
`:mps`, `:ctmrg`, `:trg`); a wider profiling sweep (not recommended by T5's
own triage).

## Store fast-path investigation milestone — complete

Opened 2026-09-15 on branch `store-fastpath-investigation`, base `main`
(`71c1536`). Follow-up to the (unmerged) upstream TensorOperations.jl
benchmark-suite comparison (`https://github.com/lkdvos/QuasiStrided.jl/pull/5`,
branch `upstream-bench`), which found and triaged one regression class
(`ccsd_t_*_dim16`, six-index output, 6.6-14.2x slower than `StridedBLAS`) to
two separable, unverified causes. Goal here: resolve Cause A (the vectorized
store fast-path guard in `src/kernels/simd.jl:217` appears unsatisfiable for
any `Array`-backed destination on Julia >= 1.11) with evidence, execute
whichever of (a) fix / (b) docs-only correction / (c) escalate the evidence
actually supports, then re-measure the four regression cases with controls
that separate Cause A's contribution from Cause B's (the cache/TLB-driven
cost on very large, non-monotonically-strided outputs, left explicitly
out of scope for a fix here). Full design, every frozen decision boundary,
and the planning-time evidence (E1-E7) are in `docs/decisions.md`, "Store
fast-path investigation: Phase A" — that file, not this one, is authoritative
for *why*.

This milestone is **complete as a real `src/` fix**, not just a benchmarking
exercise: the evidence-gated decision (T1-T3) selected "(a) fix", T4 shipped
it (`src/kernels/simd.jl`, `test/test_simd_kernel.jl`, `test/test_driver.jl`),
and T5 measured its effect both in isolation and through the real
`ccsd_t_*_dim16` regression this milestone exists to investigate. T8 review
and T9/T10 close/PR are the remaining next steps (see checklist below).

Non-goals: `QuasiStridedBackend`'s hard-reject invariant, the macro-blocking
five-loop structure, `src/target.jl`'s register-shape derivation, a general
fix for Cause B, a wider profiling sweep, repointing the `TensorOperationsBenchmarks`
dependency (PR #303 upstream still unmerged, confirmed 2026-09-15).

- [x] **T1** (fact probes: storage-type reachability per Julia version, SIMD.jl
      on `Memory{T}`, the four regression cases' actual C-side stride layout,
      provenance of the "101-103 GFLOP/s" claim).
- [x] **T2** (tile-level store-path microbenchmark).
- [x] **T3** (ccsd_t regression + control script, no timing run yet).
- [x] **Decision gate** (fix / docs-only / escalate, per the evidence — chose
      "(a) fix").
- [x] **T4** (fix — `src/kernels/simd.jl` + `test/test_simd_kernel.jl` + one
      `test/test_driver.jl` testset only).
- [x] **T5** (measurement campaign).
- [x] **T7** (docs — this section and `docs/decisions.md`'s "T4-T5: the fix
      and its measured effect").
- [x] **T8** (one gated independent review — no blocking findings; several
      should-fix findings on evidence precision addressed in T9).
- [x] **T9** (fix review findings: corrected an overstated ratio figure, an
      under-hedged regression-comparison claim, an undisclosed noise floor,
      a stale task-graph reference, and added a test-coverage gap the review
      found — unit-stride rows with scattered columns on the new vectorized
      store path).
- [x] **T10** (close, PR). Full suite 34856/34856 passing.

**What shipped (T4).** `src/kernels/simd.jl`'s store fast-path guard
(`_vector_store_eligible`) widened from an inline `isa Vector{T}` check to
any concrete `DenseVector{T}` (covering `Memory{T}`, the type the real driver
actually hands the kernel on Julia >= 1.11), with a `@generated`,
statically-indexed `_store_tile_vector!` replacing the old runtime-indexed
tail (an allocation-cliff risk once the branch became reachable). Test
additions in `test/test_simd_kernel.jl` and `test/test_driver.jl`.

**Measured.** Full suite 34853/34853 passing (was 34654; +199 assertions, no
regressions) -- re-verified independently by the coordinator. Post-fix, the
D-mem vs D-vec per-element store-cost gap (was 1.69-2.26 ns/element,
3.9x-9.9x) is eliminated to below this run's own measurement resolution
(delta now -0.10 to +0.06 ns/elem; this run's canary spread was 9.19%, worse
than the original 3.7%, but the gap closed by 3.1x-30x depending on shape,
far larger than either run's noise); native-code stack-store counts for
D-mem and D-vec are now identical. Two-tree ABBA re-benchmark
(`bench_real_path_guard.jl`): no one-sided regression on any of 18 shapes,
up to ~2.5-3.3x speedup on unit-stride-destination shapes. The
`ccsd_t_*_dim16` regression itself (Arm 1, adapter path) is not moved by the
fix, as predicted -- ratios vs `StridedBLAS` are 8.1x-14.5x post-fix,
close to but not exactly matching the 6.6-14.2x pre-fix baseline (Float64
agrees to +/-3%; Float32 drifted +5% to +23%, within this machine's
~25% same-contraction run-to-run noise at this size, per `docs/decisions.md`
-- the mechanism-level evidence (zero unit-stride slivers in all 8
case x dtype cells, both before and after) is the stronger support for
"not moved", not the timing comparison.

**Still out of scope**: repointing `TensorOperationsBenchmarks` (PR #303
still unmerged upstream); the label-ordering lever surfaced by Arm 3 (see
callout below — a candidate new milestone, not started); any remaining
upstream benchmark-suite categories.

> **Callout: an unplanned finding that needs a decision, not yet acted on.**
> T5's measurement sweep also ran a label-order diagnostic control (Arm 3,
> built by T3 only to separate the store fast-path's contribution from other
> effects — not part of this milestone's own goals). At `dim=16`, simply
> permuting the operand carrying the destination's stride-1 label to be that
> operand's own first physical axis gives a **3.3x-20x speedup** over the
> adapter's real path (Arm 1) on the exact four `ccsd_t_*_dim16` regression
> cases, for both Float64 and Float32 — far larger than anything this
> milestone's own scope (the store fast-path) could ever have delivered for
> this case class. This points at a different, product-level lever
> (`_classify_labels`'s label ordering in `src/driver.jl`, currently pinned
> by an existing test) that this milestone's task graph did **not**
> authorize touching, per its own frozen "replanning trigger" language
> (`docs/decisions.md`, "Store fast-path investigation: Phase A"). **No code
> was changed in response to this finding.** It is reported here, prominently,
> as a candidate follow-up milestone requiring the user's/coordinator's
> decision — closing this milestone does not mean the `ccsd_t_*` regression
> story is finished, only that the specific hypothesis this milestone was
> built to test (Cause A, the store fast-path) is now fully resolved. See
> `docs/decisions.md`'s "T4-T5: the fix and its measured effect" for the
> exact per-case numbers.

**Correction (2026-09-19, Label-order milestone).** "Currently pinned by an
existing test", above, was inaccurate at the time it was written -- no test
pinned the composite label order before that milestone. It is accurate now
only because the Label-order milestone's own new pinning test does so. That
milestone also fixed the underlying lever itself (`_order_free_labels`, plus
a guarded M/N orientation swap); see `docs/decisions.md`, "Label-order
milestone", for the mechanism and the measured before/after on these same
four cases.
