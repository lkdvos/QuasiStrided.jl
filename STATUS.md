# QuasiStrided.jl status

Durable status record for the main orchestration process. Update at phase
boundaries or before context compaction. History belongs here only as far as
"what phase are we in"; do not let this rot into a changelog — narrative
detail belongs in `docs/decisions.md`.

## Integrated revision

Local git repo at `/mnt/home/ldevos/Projects/QuasiStrided.jl`. `main` is at
`3e712ea` ("Add TensorOperations.jl backend (QuasiStridedBackend) (#2)") --
the TensorOperations integration milestone was squash-merged as PR #2 on
2026-09-10 and is published. The stale text that used to stand here, saying
that milestone was "still entirely uncommitted", is resolved.

Current work is on branch `blis`, based on `3e712ea`: the hardware-derived
register shape milestone (see "Hardware-derived register shape milestone"
below). Note `Manifest.toml` and `benchmark/results/` are both gitignored, so
the committed manifest is stale relative to `Project.toml` and a fresh clone
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

## Next task

**Packing and per-call overhead is now the whole gap.** Benchmarked
single-threaded against Octavian.jl and OpenBLAS (`docs/decisions.md`,
Phase H), QuasiStrided reaches 25-54 GFLOP/s on plain matmul where Octavian
is essentially flat at 79-98 and beats OpenBLAS at most points. Since the
microkernel measures 79-100 GFLOP/s in isolation, the deficit is entirely
packing plus per-call cost. Octavian's answer is a three-tier dispatch this
engine has no equivalent of:

1. `maybeinline` — statically small, fully inlined, no packing;
2. `dontpack`/`nᵣ ≥ N` → `loopmul!` — **no packing at all**, straight over the
   unpacked arrays;
3. otherwise pack A only, or pack A and B.

QuasiStrided always packs both. Adding tiers 1-2 is the highest-value next
item, and it targets exactly the small and skewed shapes a tensor network
produces (`256x256x12` measures 16 GFLOP/s here against Octavian's 87).

Keep the priority honest, though: on **genuine multi-index contractions** —
the actual target — QuasiStrided already matches or beats TBLIS, the C++ BSMTC
reference, on 4 of 5 measured points, and beats `StridedBLAS` by 1.0x-2.4x.
The plain-matmul gap is real but is not this package's workload.

Two smaller follow-ons, both now unblocked rather than urgent: the register
tile can be enlarged (`NV` up to 28 is allocation-free and spill-free — it
just measured no faster), and `default_blocking`'s `mc`/`nc` were validated
at the previous register shape.

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
- [ ] **Phase C** (integration): merge B1–B5, wire the default complex kernel,
      budget one reconciliation.
- [ ] **Phase D** (1m).
- [ ] **Phase E** (remaining test layers, four disjoint files in parallel).
- [ ] **Phase F** (measurement on `ccqlin038`).
- [ ] **Phase G** (the two gated reviews).
- [ ] **Phase H** (close: `docs/decisions.md`, this file, `README.md`).

**Test count at Phase A open: 13299/13299** (Julia 1.12.6). B1's acceptance
criterion is that this number is unchanged *and* that
`benchmark/bench_default_vs_legacy.jl` stays inside the canary spread, with no
complex kernel wired in — B1 is the real-path guard and it lands first.

Two notes for anyone picking this up on a fresh checkout of this worktree.
`Manifest.toml` is gitignored, so the environment must be resolved before
anything runs. And on the *first* full run after that, Aqua's
`test_persistent_tasks` can fail on a timeout: it loads the package in a
subprocess, which on a cold depot has to precompile first (observed here at
2m24s, against 5.5s once warm). It is not a real failure — re-run the suite,
and check it in isolation before believing it.
