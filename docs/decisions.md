# API and ownership decision record

Main process only. Record every public-interface decision and file ownership
change here before workers depend on it.

## Frozen interfaces

### `contract!` entry point (Phase 0)

```julia
contract!(C::StridedView, alpha::Number,
          A::StridedView, indA::NTuple{NA,Int},
          B::StridedView, indB::NTuple{NB,Int},
          beta::Number,
          indC::NTuple{NC,Int}) where {NA,NB,NC}
```

Label semantics: one `Int` label per axis, given in axis order for `indA`,
`indB`, `indC`. A label in both `indA` and `indB` but absent from `indC` is a
contracted (K) axis. A label in `indC` and in exactly one of `indA`/`indB` is
free (M if from A, N if from B). Diagonals (repeated label within one tensor's
own index tuple) and labels appearing in `indC` but not in `indA`/`indB` are
out of scope this milestone: raise `ArgumentError`. Matched labels across
tensors must have equal axis length: raise `DimensionMismatch` otherwise.

Rationale: mirrors the M/N/K `AxisGroup` construction directly
(`Julia-Tensor-Indexing-Agent-Spec.md` §3) and is a well-precedented labeled-
index convention (cf. TensorOperations.jl), while staying an explicit
axis-list front end per the handoff (no einsum string parsing).

### Namespace and package location

- Package: `QuasiStrided.jl`, module `QuasiStrided`, at
  `/mnt/home/ldevos/Projects/QuasiStrided.jl` (sibling to the Rust repo's
  local checkouts, matching this user's existing convention of
  `~/Projects/<Name>.jl`).
- Depends on `StridedViews` (installed locally, `~/Projects/StridedViews.jl`
  v0.5.1; registry has up to 0.5.2). Does not depend on `Strided` itself in
  `src/` for this milestone — nothing here needs `Strided`'s functionality
  (its blocked `mapreducedim!` machinery is explicitly out of scope); `Strided`
  is a `test`-only extra so integration tests can build `StridedView`s the
  way a real user would via `Strided.sview`/`StridedView` construction if
  useful.
- `StridedView` accessors used: `size(a)`, `strides(a)`, `offset(a)` (exported
  from `StridedViews`), `parent(a)` (Base). Verified against
  `~/Projects/StridedViews.jl/src/stridedview.jl` (repo HEAD
  922d6c6bf03d67fb5f093452c3d159d80e343738) and the installed package source
  under `~/.julia/packages/StridedViews/vC5J6/src/stridedview.jl:27-96`.
- SIMD dependency: `SIMD.jl` (v3.7.1 installed locally, providing `Vec{N,T}`).
  Chosen because it's already resolvable in the depot, is the direct
  `Vec{N,T}`-style abstraction the microkernel spec describes, and needs no
  extra setup. `VectorizationBase.jl` is also present but is a heavier,
  lower-level dependency of packages like LoopVectorization; not needed for
  one explicit-SIMD candidate.
- Reference machine for this bootstrap: Cascade Lake-class, AVX-512 (avx512f/
  dq/cd/bw/vl) + AVX2 + FMA, Julia 1.12.6.

## File ownership (updated at phase boundaries)

| File | Owner | Phase |
| --- | --- | --- |
| `Project.toml`, `src/QuasiStrided.jl`, `test/runtests.jl`, this file, `STATUS.md` | main process | all |
| `src/axis_group.jl` | indexing implementer | 1 |
| `test/test_axis_group.jl` | oracle/test implementer | 1 |
| `test/strided_integration.jl` | oracle/test implementer | 1 |
| `src/tiles.jl` | packing implementer | 2 |
| `src/packing.jl` | packing implementer | 2 |
| `test/test_packing.jl` | packing implementer | 2 |
| `src/kernel.jl` | scalar implementer | 2 |
| `test/test_kernel.jl` | scalar implementer | 2 |
| `src/kernels/simd.jl` | SIMD implementer | 3 |
| `src/driver.jl` | driver implementer | 3 |
| `test/test_driver.jl` | driver implementer | 3 |
| `benchmark/*.jl` | respective implementer of the component benchmarked | 1-3 |
| `README.md` | main process (content from all phases) | 4 |

A worker needing to edit a file outside this table requests a transfer through
the main process; this table is updated first.

## Phase 1 integration note

`Project.toml`'s `[targets] test` was missing `Random` (needed by the
independent oracle/property tests, which use `Random.MersenneTwister` with a
fixed seed). Added by the main process post-integration: `Random` in
`[extras]` and `[targets] test`. No other cross-worker conflicts.

## Phase 2 integration notes

Two reconciliations, both anticipated by each worker's own task instructions
(neither worker was told to block on the other):

1. **Destination-tile type drift.** The scalar implementer (owning
   `src/kernel.jl`) was explicitly told not to block on `src/tiles.jl`
   landing, and wrote its own `ScalarDestination` (closures for row/col
   addressing) rather than depending on the packing implementer's real
   `QSTile`/`DestinationTile`. Both express the same addressing contract
   (`base + row_offset(i) + col_offset(j)`, affine-or-scattered per axis).
   Fix: added `scale_tile!`/`store_tile!`/`execute_tile!` methods dispatching
   on `QSTile` directly (mirroring the `ScalarDestination`-typed methods
   line-for-line, substituting `nrows`/`ncols`/`tile_load`/`tile_store!` for
   `destination.m`/`.n`/closures) to `src/kernel.jl`. **`DestinationTile` +
   these new methods is the path Phase 3's driver should use**;
   `ScalarDestination` remains only as the scalar worker's own test
   scaffolding and is not exported.
2. **Kernel/packing coupling.** `pack_a!`/`pack_b!` (packing implementer)
   dispatch on a bare `KernelDescriptor`; `ScalarKernel` (scalar implementer)
   wraps one rather than being one, so passing a `ScalarKernel` straight
   through to `pack_a!`/`pack_b!` was a `MethodError`. Fix: two one-line
   forwarding methods in `src/kernel.jl`,
   `pack_a!(packed, source, k::ScalarKernel, transform) = pack_a!(packed, source, k.descriptor, transform)`
   (and same for `pack_b!`). Any future concrete kernel wrapping a
   `KernelDescriptor` as a `.descriptor` field should add the same two
   one-liners.
3. Added `test/test_phase2_integration.jl` (main-process-owned): drives the
   full `AxisGroup -> block_descriptors! -> axis_from_descriptor -> QSTile ->
   pack_a!/pack_b! -> ScalarKernel execute_tile!` path by hand on the shared
   worked A[a,k,b]/B[k,n]/C[a,n,b] fixture, including 3 K-panels (lengths
   2,2,1) with beta applied once and nontrivial alpha/beta — this is also the
   fixture supplied to the Phase 2b Fable review.
4. New exports added to `src/QuasiStrided.jl`: `axis_from_descriptor`,
   `nrows`, `ncols` (from `tiles.jl`), `ScalarKernel`, `scale_tile!` (from
   `kernel.jl`). `ScalarDestination`, `affine_axis`, `scatter_axis` remain
   unexported (internal to the scalar worker's own tests).

`Pkg.test()` after both reconciliations: 12318/12318 passing, 25.4s.

## Phase 2b Fable review: triage and disposition

Fable High review completed (`fable_review_used: true`), 2026-09-08. Full
findings kept in `STATUS.md`'s history is unnecessary; disposition of each:

1. **[blocking, FIXED]** `pack_a!`/`pack_b!` never validated reachable
   source-storage bounds before `@inbounds` reads. Added
   `axis_offset_range` (per-axis min/max offset, `Int128`-checked) and
   `checked_tile_storage_bounds` (`src/tiles.jl`, exported), called once per
   tile in `pack_a!`/`pack_b!` before their loops (`src/packing.jl`).
2. **[blocking, FIXED]** `execute_tile!` on a real `DestinationTile`
   (`QSTile`) validated shape (`m<=MR`,`n<=NR`) but never storage bounds
   before `scale_tile!`/`store_tile!`'s `@inbounds` writes. Added the same
   `checked_tile_storage_bounds` call in `execute_tile!` (the `QSTile`
   overload in `src/kernel.jl`), before both the short-circuit and the full
   accumulate/store path (the review's reproducer used a short-circuit call,
   so the check had to precede that branch too, not just the main path).
   `scale_tile!`/`store_tile!` themselves remain intentionally unchecked hot
   primitives (spec section 9's "do not run an exhaustive... search for
   every microtile" — the check belongs once, at `execute_tile!`, exactly as
   `tile_offset`/`checked_tile_offset` already split unchecked/checked).
3. **[should-fix, FIXED]** Neither `execute_tile!` overload validated
   `packed_a`/`packed_b` length against `packed_a_length`/`packed_b_length(kernel,kc)`
   before `accumulate`'s `@inbounds` reads. Added `DimensionMismatch` checks
   in both overloads in `src/kernel.jl`, positioned after the short-circuit
   (kc=0/alpha=0 legitimately doesn't need the buffers, so an undersized
   buffer that is never read must not be rejected — a case added as a test).
4. **[should-fix, FIXED]** `QSTile`/`DestinationTile` execution path had no
   dedicated tests (all of `test_kernel.jl`'s coverage was against
   `ScalarDestination`). Extended `test/test_phase2_integration.jl` with a
   `QSTile execute_tile!: coverage parity` testset covering alpha=0/beta=0,
   alpha=0/beta=1, beta=0-with-NaN-old-C, empty destination, extent errors,
   canaries, Float32, and mixed affine/scattered addressing — plus a
   regression testset for findings 1-3's exact reproducers and direct unit
   tests of `axis_offset_range`/`checked_tile_storage_bounds`.
5. **[should-fix, RESOLVED 2026-09-08]** `pack_a!`/`pack_b!` allocated ~80
   B/call in steady state. **Root cause**: `transform` (and, in
   `_pack_panel!`, `transform`/`load`/`packed_offset`) were accepted with no
   `where`-bound type parameter — a forwarding argument that is only ever
   passed along, never called directly in the outer method, is compiled
   against a widened type unless explicitly bound, forcing a dynamic
   dispatch into `_pack_panel!` that heap-allocates the closure passed
   alongside it. `Profile.Allocs` pinpointed the exact allocation (an
   anonymous-closure struct, 80 bytes) at the `_pack_panel!` call site,
   confirming the mechanism. The three sub-pieces measured "0 B in
   isolation" (Phase 2b) were genuinely 0 B — the bug was invisible to
   per-piece measurement because it was specifically the missing binding on
   the *forwarding* parameter, not any one piece's own logic. **Fix**: gave
   `transform` (`pack_a!`/`pack_b!`) and `transform`/`load`/`packed_offset`
   (`_pack_panel!`) their own free type parameters. No signature, contract,
   or packed-offset-formula change. The identical bug recurred one layer up
   in `ScalarKernel`'s and `SIMDKernel`'s `pack_a!`/`pack_b!` forwarding
   methods (`src/kernel.jl`, `src/kernels/simd.jl` — untyped `kernel`/
   `transform` parameters), found and fixed by the main process with the
   same pattern (the diagnosis task was correctly scoped to not touch those
   two files, so it flagged this precisely rather than exceeding scope).
   Verified zero allocation, after warmup, for both `pack_a!`/`pack_b!`
   called directly with `KernelDescriptor` and via both kernel forwarding
   paths, across affine/scattered sources, `kc=0`, and a nontrivial
   transform. Regression tests in `test/test_packing.jl`.
   **Residual, newly observed, NOT part of this fix**: `execute!` through
   the driver still allocates (measured: ScalarKernel ~10.7KB, SIMDKernel
   ~5.9KB for a 9x10x8 case with 2 K-panels) — smaller than before this fix
   (was ~15KB/~10.2KB), but not zero. `ScalarKernel`'s `zero_accumulator`
   (176 B/call, a `Matrix{T}`) is spec-accepted and not a bug (see design
   doc section 8: acceptable for the scalar reference, unlike SIMD, whose
   `zero_accumulator` measures 0 B). The rest of the residual is in
   `src/driver.jl`'s own tiling loop, not diagnosed here — this fix was
   correctly scoped to `pack_a!`/`pack_b!` only, per its task instructions,
   and did not touch the driver. Left as a new, smaller, open item — see
   `STATUS.md`.
6. **[note, ACCEPTED]** `accumulate` is a `Base.accumulate` method, not a
   fresh binding — kept as documented in `src/kernel.jl` (avoids an
   `export`/`Base` name collision under `using QuasiStrided`); no better
   option without renaming the frozen public API.
7. **[note, ACCEPTED]** `AffineAxis.stride*t` is a single multiplication, not
   repeated addition; sound in practice because every `AffineAxis` the
   production chain builds comes from a `BlockDescriptor` whose excursion is
   already bounded by `AxisGroup`'s constructor-time check. Documented as a
   deviation, not fixed (would need `AffineAxis` to carry provenance to
   distinguish "built from a checked descriptor" from "hand-built").
8. **[note, ACCEPTED]** No `code_llvm`/`code_native` inspection proving the
   `beta==0` branch has no destination load; relying on textual inspection
   (documented in `test_kernel.jl`/`test_phase2_integration.jl` comments).
   Deferred to Phase 3, where the SIMD kernel makes generated-code inspection
   unavoidable anyway.

`Pkg.test()` after Phase 2b fixes: 12346/12346 passing, 28.7s.

## Phase 3 integration notes

Two workers (SIMD implementer, driver implementer) ran in parallel with
disjoint file ownership (`src/kernels/simd.jl`+`test/test_simd_kernel.jl` vs
`src/driver.jl`+`test/test_driver.jl`) and, unlike Phase 2, needed almost no
reconciliation: both independently followed the `ScalarKernel` pattern
(wrap `KernelDescriptor`, forward `mr`/`nr`/`scalartype`/`packed_*`, add
`pack_a!`/`pack_b!` forwarding methods per the Phase 2 integration note),
so `SIMDKernel` slotted into the driver as a drop-in `kernel=` swap with zero
source changes to `src/driver.jl`. Main process integration work:

1. Exported `SIMDKernel`, `lanewidth`, `avecs_per_column` (from
   `kernels/simd.jl`) and `plan_contract`, `execute!`, `ContractPlan` (from
   `driver.jl`) in `src/QuasiStrided.jl` — neither worker could edit that
   file themselves.
2. Added `test/test_simd_kernel.jl` and (new, main-process-owned)
   `test/test_phase3_integration.jl` to `test/runtests.jl`'s include list.
3. `test/test_phase3_integration.jl`: verifies `plan_contract`/`execute!`
   actually accept `SIMDKernel` in place of the default `ScalarKernel` and
   agree numerically (both against a direct `Amat*Bmat` reference and
   against each other, `atol=1e-10`/`1e-8`) — through the real driver
   (multi-output-tile, multi-K-panel), not just at the single-tile level
   that `test_simd_kernel.jl` already covers against `ScalarKernel` directly.
   Neither worker's own tests exercised this combination.
4. One incident, no data lost: the driver implementer accidentally deleted
   an untracked, never-committed scratch file (`scratch_simd_smoke.jl`)
   belonging to the SIMD implementer while both worked in the same shared
   checkout concurrently. Not part of any deliverable (git-tracked files were
   unaffected); no recovery needed. Noted here as a caution for any future
   phase running two workers against one shared checkout rather than
   worktrees: scratch files are not protected by "own these files" scoping.

`Pkg.test()` after Phase 3 integration: 12623/12623 passing.

SIMD kernel measured performance (Cascade Lake, MR=8/NR=6/W=4, Float64,
warmed, single `execute_tile!` call): 3.9-6.3x faster than the scalar
reference across kc in {1,4,16,64,256} — see the SIMD implementer's own
report for exact numbers; not independently re-measured by the main process
(no cluster-exclusivity requirement applies to a single-call microbenchmark
of this kind, but see `docs/measurement-rules.md`-equivalent caution: this is
a single machine, single microarchitecture, not a cross-machine claim).

## Phase 4 review: findings and disposition

Sonnet High review of everything since Phase 2b (`src/kernels/simd.jl`,
`src/driver.jl`, `test/test_phase3_integration.jl`), 2026-09-08. No blocking
findings — confirmed the Phase 2b bounds checks are not bypassed anywhere in
the driver's tiling loop or in `SIMDKernel`'s own `execute_tile!`, beta is
applied exactly once per output tile across K panels, and label
classification matches the frozen table exactly. Two coverage gaps, both
closed:

1. No test measured `execute!` allocation with `SIMDKernel` specifically
   (only `ScalarKernel` was covered). Reviewer measured it directly and
   found no driver-induced allocation regression (`SIMDKernel` was in fact
   *lower*, not higher, allocation than `ScalarKernel` through the driver —
   both nonzero only from the already-deferred `pack_a!`/`pack_b!` finding).
   Added a permanent regression test,
   `test/test_driver.jl`'s "execution allocation through SIMDKernel is not
   worse than ScalarKernel" testset, asserting this rather than relying on a
   one-off manual check.
2. `_classify_labels`'s dangling-only-in-B case ((F,T,F) — a distinct code
   path from the dangling-only-in-A case, the B loop rather than the A
   loop) was untested. Added to `test/test_driver.jl`'s label-validation
   testset.

Driver-level benchmark (Cascade Lake, 64x64x64 contraction, MR=8/NR=6,
kc_panel=32, warmed, `Pkg.test()`-independent one-off measurement, not part
of the committed test suite): planning (`plan_contract`) is ~4.4-6.7 µs/call
regardless of kernel — under 3% of total cost for this size and fully
amortizable across repeated `execute!` calls on a reused plan. Steady-state
`execute!`: scalar kernel 301.6 µs/call (156288 B), SIMD kernel 188.2 µs/call
(68992 B) — SIMD is both faster and lower-allocation than scalar at this
size, through the real driver (not just a single-tile microbenchmark). All
allocation is attributable to the deferred `pack_a!`/`pack_b!` ~80B/call
finding, scaling with (M-tiles × N-tiles × K-panels) — 8×11×2×2 (A and B) ≈
352 pack calls for this shape, consistent with the measured totals.

`Pkg.test()` after Phase 4 fixes: 12627/12627 passing.

## Macro-blocking milestone: Phase A scouting and interface freeze

Base revision `d4ab46b` (clean). This milestone replaces `execute!`'s
tile-by-tile loop with a BLIS five-loop (`NC`/`KC`/`MC`) macro-blocking nest
with packed-panel reuse. Orchestration plan: see the session's saved plan;
narrative decisions recorded here as they're made.

### Phase A findings (scouting + allocation root-cause)

**Reference machine** (same one used throughout this project): Xeon Gold
6244, Cascade Lake, 2 sockets x 8 cores, L1d 32 KiB/core, L2 1 MiB/core
(16 MiB / 16 instances), L3 ~24.75 MiB/socket (49.5 MiB / 2 instances),
Julia 1.12.6. All defaults chosen this milestone are single-machine
measurements unless a second machine class is explicitly recorded.

**`SIMD.vload` accepts a contiguous `SubArray` view** of a `Vector`
(confirmed: `FastContiguousArray` in SIMD.jl's `arrayops.jl` includes
`Base.FastContiguousSubArray`; a direct `vload(Vec{4,Float64}, view(...), 1)`
call succeeds). So widening `pack_a!`/`pack_b!` to accept `AbstractVector{T}`
requires no change on the kernel/consume side — `execute_tile!` already
takes `AbstractVector{T}`.

**Root cause of the existing residual `execute!` allocation** (documented as
open in `STATUS.md`, not previously diagnosed): reproduced at exactly
10656 B (ScalarKernel) / 5904 B (SIMDKernel) for the `test_driver.jl`
9x10x8/kc_panel=4 case, matching `STATUS.md`'s figures. `Profile.Allocs` +
`@code_warntype` confirm: `axis_from_descriptor` returns
`Union{AffineAxis, ScatterAxis}`; every `QSTile` built from such an axis
(`destination`, `source_A`, `source_B` in `execute!`) infers only as the
partially-applied `QuasiStrided.QSTile{Memory{Float64}}` — an `UnionAll`
left open over its `R`/`C` type parameters — which is heap-boxed on every
construction (80 B x 63 constructions = 5040 B). The resulting
`execute_tile!(kernel, destination, ...)` call becomes a **dynamic
(generic) call** (confirmed in IR: no `invoke`/`Core.Const` resolution),
which boxes its scalar arguments (`alphaT`/`beta_eff`: 16 B x 54 = 864 B).
The remaining 4752 B is `ScalarKernel`'s already-spec-accepted
`zero_accumulator` `Matrix` allocation (176 B/call x 27), unrelated to this
bug and out of scope.

**Binding requirement for Phase C (the macro-kernel rewrite):** a
`Union{AffineAxis,ScatterAxis}` value must never flow into a data structure
(`QSTile` or its replacement) that is then passed on to further
type-unstable code inside the hot loop. Each block's regularity
(`descriptor.regular`) must be branched on **once**, dispatching to a
concrete-typed inner function/method for that iteration, so no `Union`
crosses a function boundary into packing/kernel calls. Verify with
`@code_warntype` that no local downstream of `axis_from_descriptor` ever
prints as `Union{...}` or as a partially-applied `QSTile{Memory{Float64}}`
missing its `R`/`C` parameters — that specific pattern, not a small
few-way union, is the actual trigger. This closes the existing open
allocation item as a side effect of the rewrite (§2 of the plan sets this
as a hard target: 0 B for `SIMDKernel` on Julia >= 1.11, bounded for
`ScalarKernel`).

### Frozen interface additions (additive; commit before any Phase B/C worker launches)

1. `describe_block(buffer::Vector{Int}, first::Int, count::Int)::BlockDescriptor`
   — classify `buffer[first+1 : first+count]` (zero-based `first`). The
   existing 2-arg `describe_block(buffer, count)` becomes
   `describe_block(buffer, 0, count)` (kept, unchanged behavior/signature,
   forwarding).
2. `axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}, first::Int)`
   — `AffineAxis` when regular, else `ScatterAxis(view(buffer, first+1 :
   first+count), count)`. Existing 2-arg form forwards with `first = 0`.
3. `pack_a!`/`pack_b!` (`src/packing.jl`) widen their `packed` parameter from
   `Vector{T}` to `AbstractVector{T}`, so a macro panel view can be packed
   into directly. `_check_packed_eltype` and `_pack_panel!` widen the same
   way. `src/kernel.jl`'s two forwarding methods (`pack_a!`/`pack_b!` for
   `DescriptorKernel`) widen identically — this was the exact site of the
   Phase 2b finding-5 recurrence (missing `where`-bound forwarding
   parameter), so the Phase B implementer must re-verify zero allocation on
   the widened signature, not assume it's preserved. Packed offset formulas
   (`i + MR*p`, `j + NR*p`) are unchanged. No change to `pack_a!`/`pack_b!`'s
   validation/never-allocates/padding contract.
4. New: `struct Blocking; mc::Int; kc::Int; nc::Int; end` (fields >= 1),
   `default_blocking(kernel) -> Blocking` (dispatches on kernel type/
   scalartype so a future kernel can declare its own budget), and
   `plan_contract(...; kernel, mc=nothing, kc=nothing, nc=nothing)`. Absent
   keywords use `default_blocking(kernel)`. Effective values (stored on
   `ContractPlan`, replacing the `kc_panel::Int` field): round `mc`/`nc` up
   to whole `mr(kernel)`/`nr(kernel)` multiples, then
   `mc_eff = min(mc_rounded, roundup(Qm, MRk))`,
   `nc_eff = min(nc_rounded, roundup(Qn, NRk))`,
   `kc_eff = min(kc, Qk)`. The `kc_panel` keyword and `ContractPlan.kc_panel`
   field are **removed** (unregistered v0.1.0 API; only internal tests
   reference it — the Phase C implementer owns migrating those tests to
   `kc`/`mc`/`nc`).
5. New, unexported: `execute_tilewise!(plan, alpha, beta)` — the current
   `execute!` body verbatim (packs one `MR x kc`/`kc x NR` sliver per output
   tile, no panel reuse), kept as the independent-of-the-macro-nest
   correctness oracle. It continues to size its own buffers off `MR`/`NR`
   only, not the new `mc`/`nc` blocks (or, if it shares `ContractPlan`'s
   larger buffers, it must only ever address sliver 0 of them — the Phase C
   implementer decides and documents which, since both satisfy "packs one
   sliver, doesn't touch macro-block iteration").

### Block-size policy (settled)

Tunable keywords (`mc`, `kc`, `nc` on `plan_contract`) with hardcoded,
measured, single-machine-labeled defaults via `default_blocking(kernel)`.
Explicitly **not** an analytical or cache-probing model: `tensorcontract-rs`
already tried exactly that and it is a recorded refutation there (A33: lost
14-34% even on its own reference machine, failed on two further machine
classes; A57: its L2-privacy assumption is false on Apple Silicon, where
multiple P-cores share one L2 — relevant since this package's CI runs
macOS). `docs/refuted.md` in that project also failed a depth-adaptive-MC
attempt. Given MC is documented there as a wide plateau (16x range moves
geomean <= 3%), a conservative hardcoded constant is forgiving, and a
keyword lets a caller or future autotuner override without an API change.
Phase C ships clearly-marked `# PROVISIONAL` constants; Phase E replaces
them with sweep-measured values and records provenance here.

### File ownership additions (append to the existing table)

| File | Owner | Phase |
| --- | --- | --- |
| `src/axis_group.jl` (additive `describe_block`/related methods only), `src/tiles.jl` (additive `axis_from_descriptor` only), `src/packing.jl` (`AbstractVector` widening only), `test/test_axis_group.jl`/`test_packing.jl` (additive tests) | interface implementer | Macro-B |
| `src/blocking.jl` (new), `src/driver.jl`, `test/test_driver.jl` | macro-kernel implementer | Macro-C |
| `test/test_macro_driver.jl` (new) | oracle/test implementer | Macro-C |
| `benchmark/bench_driver.jl` (new) | benchmark implementer | Macro-E |

`src/kernel.jl` (beyond the two forwarding methods above), `src/kernels/simd.jl`,
`src/kernel_descriptor.jl` remain frozen this milestone.

## Phase D: Fable review of the integrated macro path

Fable High review completed (`fable_review_macro_used: true`), 2026-09-08,
on the integrated `execute!` rewrite (HEAD `b33678e`). No blocking findings.
Confirmed by direct trace + a 160-combination scratch check (regular and
irregular slivers, tail blocks in M/N/K simultaneously): beta is applied
exactly once per output element (the `pc==0`/`firstpanel` decision is made
once per K block, not per tile, and the M/N/K block partitions are disjoint
and exhaustive); sliver-stride addressing agrees between the pack and
consume sides at every block (both always use the current block's actual
`kc_len` via `_sliver_range`, and a stride mismatch would raise
`DimensionMismatch`, not misread); no borrowed-buffer staleness (each of
`m_buf_*`/`n_buf_*`/`k_buf_*` is filled and consumed within its own
`ic`/`jc`/`pc` scope, never read after a later refill); `Blocking`'s
effective-value clamping cannot produce a zero-length buffer or zero-sliver
block on a nonzero input shape.

Disposition of each finding:

1. **[should-fix, FIXED]** The driver's irregular-sliver-at-nonzero-offset
   path (`describe_block`/`_axis_of` called with `regular=false` and
   `first>0`, `driver.jl:426-427,446,460-461,467,475,479`) had zero
   permanent test coverage — every existing multi-block test used a
   single-label (dense-matmul) M/N/K group, which is always classified
   `regular` regardless of `first`, since a single dimension has no
   internal "carry" boundary to break the constant-stride sequence. The
   reviewer verified correctness via an ad hoc 160-combination scratch
   check, not a committed test. Added
   `test/test_macro_driver.jl`'s "irregular sliver at nonzero offset
   (multi-label M group)" testset: a 2-label M group (`a`,`q`) where `A`'s
   map is naturally contiguous (regular even across the `a`/`q` boundary)
   but `C`'s map is deliberately padded so it fails the affine-fold
   condition at that same boundary — forcing `describe_block` to return
   `regular=false` for the C-side sliver at `first=4`. Directly confirmed
   (not just inferred from the test passing) via a standalone
   `describe_block` call reproducing the exact offset sequence:
   `buf_A=[4,5,6,7]` (regular=true), `buf_C=[4,5,6,10]` (regular=false).
   `_axis_of` is the one helper used identically for the M/N/K sides
   (`driver.jl:126-129`), so this exercises the same code path the N and K
   sides share, without needing to also construct a multi-label N or K
   fixture. `Pkg.test()`: 12903/12903 passing.
2. **[note, ACCEPTED, pre-existing]** `_scale_all_of_C!`
   (`driver.jl:157-160`, reached only on the `Qk==0`/`alpha==0`
   short-circuit) calls `scale_tile!` without a
   `checked_tile_storage_bounds` call first — unchanged from the old
   driver's equivalent short-circuit (its old line 247), and the reviewer
   could not construct a failing input (offsets come from the validated
   `AxisGroup`). Not fixed: no demonstrated failure, and it predates this
   milestone.
3. **[note, ACCEPTED]** `test/test_macro_driver.jl`'s buffer-poisoning
   testset can only detect read-before-fill, not intra-call staleness (the
   actual macro-blocking risk) — but the small-random-block cases in
   testset 1 (`mc,kc,nc ∈ 1:13` against shapes up to 37) are what would
   actually catch a staleness bug, and did not. No action needed.
4. **[note, ACCEPTED]** `ContractPlan`'s shared `(jc,pc)` state (`n_buf_*`,
   `n_desc_*`, `k_buf_*`, `packed_b`) and per-`ic` state (`m_buf_*`,
   `m_desc_*`, `packed_a`) are disjoint fields today, so a future
   parallelization over `ic` is not precluded, but would need the M-side
   extracted into a per-worker struct — a design note for a future
   threading milestone, not an action now.
