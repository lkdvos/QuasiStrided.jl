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

## Phase E: benchmark sweep and measured block-size defaults

Machine: Xeon Gold 6244, Cascade Lake, 2x8 cores, L1d 32 KiB/core, L2
1 MiB/core, L3 ~24.75 MiB/socket, hostname `ccqlin038`, single machine only
— no second machine class was measured this phase, and none of this
should be read as portable to a different microarchitecture (this
project's own standing A56-style rule). Julia 1.12.6. Date 2026-09-08,
`git` base revision `550468d`.

Script: `benchmark/bench_driver.jl` (new). `julia --project=.
benchmark/bench_driver.jl`, `Threads.nthreads() == 1`,
`LinearAlgebra.BLAS.set_num_threads(1)`. Full raw output, chosen-default
summary, canary CSV and `PROVENANCE.txt` are committed under
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/`.

**Grid actually run** (a first 9-point corners+center trial finished the
whole script in ~24s, so the grid was widened to the following before the
measurement run that produced the numbers below — nothing further needed
cutting):

* Shapes: `64^3`, `128^3`, `256^3`, `512^3`, a shallow-K case
  (`256x24x256`), `1024x256x1024`, and a 3-index scattered/sliced-C-row
  case (`A[a,k,b] B[k,n] -> C[a,n,b]`, `a=64,k=64,b=16,n=64`, adapted
  directly from `test/test_macro_driver.jl`'s permuted-A /
  negative-stride-B / sliced-with-offset-C fixture). 512^3 was not skipped
  — the trial run showed 256^3 costs only ~5 ms/`execute!` call
  (`ScalarKernel`), so 512^3 (~40 ms) fit the full grid comfortably.
* `(mc,kc,nc)` grid: Float64 — `kc ∈ {128,256,512}`, `mc ∈
  {64,128,256,512}`, `nc ∈ {768,1536,3072}` (36 combos; the task spec's
  text said "27" but listed 4 `mc` values against 3 `kc`/`nc` values, which
  multiplies to 36 — used the literal sets rather than guessing which
  number was the typo). Float32 — an analogous grid scaled ~1.5x (`kc ∈
  {192,384,768}`, `mc ∈ {96,192,384,768}`, `nc ∈ {1152,2304,4608}`, 36
  combos), run for both `ScalarKernel` and `SIMDKernel`, both at register
  shape `MR=8,NR=6` (`SIMDKernel`'s lane width `W=4` for `Float64`, `W=8`
  for `Float32`). Full grid at all 5 main shapes; the two largest/extra
  shapes (`1024x256x1024`, the scattered case) were run only at each
  dtype's own middle-of-grid combo (both kernels), to check the grid
  winner generalizes without paying for the full grid there too — this
  is the one deliberate reduction from "grid x every shape".
* 9 timed reps per point (plus 1 discarded warm-up), median reported. Not
  the task spec's minimum-5 exactly, but higher (9) for better statistics,
  since the machine time allowed it.

**Canary bracket** (`64^3`, `Float64`, `SIMDKernel`, `mc,kc,nc =
128,256,1536`, 15 reps, run at the start, the mid-point, and the end of the
whole sweep): medians `2.5798e-5 s`, `2.5982e-5 s`, `2.5946e-5 s` —
relative spread `(max-min)/min = 0.71%`. No drift detected during the run;
the machine was exclusive throughout. (An earlier 7-rep trial on a smaller
grid had shown a 43% spread on this same canary shape — a timer-resolution
artefact at a ~25 microsecond measurement, not real drift; raising reps to
15 resolved it. Recorded here because a shortfall like that is exactly
what this bracket exists to catch, and it did.)

**Chosen defaults** (geomean of `execute!` time, normalized per
`(kernel,shape)` by that pair's own minimum across the grid so shapes of
very different absolute cost weigh equally, computed jointly over both
`ScalarKernel` and `SIMDKernel` since `default_blocking` dispatches only on
`scalartype`, not kernel type — the two kernels share one constant per
dtype):

* **Float64: `mc=64, kc=128, nc=768`.** Geomean ratio 1.0588 (i.e. 5.88%
  above the grid's best point). Unconstrained best was `mc=64, kc=512,
  nc=1536` at 1.0098; the chosen point is within the project's own 6%
  noise-floor convention of that best (4.85% worse) and is the
  smallest-footprint (`mc*kc*nc`) point satisfying that bound. Runner-up
  by geomean: `mc=64, kc=512, nc=768` at 1.0124 (a 0.26-point spread from
  the unconstrained best).
* **Float32: `mc=96, kc=384, nc=1152`.** Geomean ratio 1.0339 (2.49% above
  the best). Unconstrained best was `mc=192, kc=768, nc=1152` at 1.0088;
  runner-up `mc=96, kc=768, nc=4608` at 1.0110. The chosen point is again
  the smallest footprint within 6% of the best.
* Both rankings show the wide-plateau pattern this project's own MC
  precedent predicts (docs/decisions.md, "Block-size policy" cites a 16x
  range moving geomean <=3% elsewhere): most of each 36-point grid sits
  within ~6% of its own best, and `kc` is the axis with the clearest signal
  (small `kc` values are consistently worse for Float64; small `kc` is
  worse for Float32 too — the worst Float32 points are all `kc=192`).

**`execute!` vs `execute_tilewise!`**: `execute!` was faster than
`execute_tilewise!` at **every one of the 28 measured
`(kernel,dtype,shape)` combinations** — no shortfall to report. Speedup
(best-grid `execute!` time vs. the single `execute_tilewise!` time at the
mid-grid combo) ranged from 1.66x (`ScalarKernel`, `Float64`, `64^3` — the
smallest case, where per-call overhead dominates) up to 10.5x
(`SIMDKernel`, `Float32`, `512^3`). `SIMDKernel` consistently benefits more
from macro-blocking than `ScalarKernel` (3.5x-10.5x vs. 1.66x-2.4x),
consistent with panel-reuse mattering most once the micro-kernel itself is
fast enough that repacking would otherwise dominate.

The `mul!` reference line (BLAS-backed, not a competitor — see
`benchmark/bench_driver.jl`'s header comment) is, as expected, faster than
`execute!` throughout (2x-3x for `SIMDKernel`, 10x-36x for `ScalarKernel`);
this is a reference point, not a regression.

**Known limitations of this measurement**: one machine class only (Cascade
Lake / AVX-512); the two largest/extra shapes were validated only at the
mid-grid combo per dtype rather than the full 36-point grid; and the
6%-noise-floor convention borrowed here is this project's own choice, not
independently re-derived on this machine this phase.

## TensorOperations integration milestone: Phase A direction freeze

Base revision `314c47f` (clean), branch `tensoroperations`; `main` is at the
same commit. This milestone makes the real, external **TensorOperations.jl**
(QuantumKitHub/TensorOperations.jl, v5.8.0 resolvable locally) the interface
users actually write against (`@tensor`, `ncon`) and demotes QuasiStrided to
an internal contraction backend that TensorOperations dispatches to. Nearly
all of QuasiStrided's current flat export surface becomes implementation
detail. Orchestration plan: see the session's saved plan
(`i-want-to-make-polymorphic-kettle.md`); narrative decisions recorded here
as they are made.

Everything in this section is **frozen**: it lands before any Phase B worker
launches, and it is authoritative over the plan file wherever the two
differ. Two subsections below are explicit **amendments** to interfaces
frozen in earlier milestones; each names what it supersedes. Phase letters
for this milestone are A-F as listed in `STATUS.md`; the plan file's own
"Phase B-D" lumping of the four-way parallel group is superseded by that
lettering, with no change to the dependency graph.

### Direction: hard dependency, no extension, no upstream PR

QuasiStrided takes a **hard dependency** on TensorOperations and defines its
own `QuasiStridedBackend <: TensorOperations.AbstractBackend` in its own
`src/`. No `ext/` directory, no weak dependency, and no PR to
QuantumKitHub/TensorOperations.jl on the critical path.

Rationale: subtyping `TO.AbstractBackend` requires TO to be loaded at the
point the struct is *defined*. Under a weakdep/extension the marker struct
would have to live inside the extension module, where it is not reachable as
ordinary user-facing API (`Base.get_extension` reflection would be needed to
name it) — unlike `TBLISBackend`/`cuTENSORBackend`, whose marker structs
live in TO core (`src/backends.jl`) precisely so the extension only has to
supply methods. Putting `QuasiStridedBackend` into TO core is the option
this milestone's premise excludes. TensorOperations is pure Julia with no
binary dependencies, its `julia = "1.10"` compat matches QuasiStrided's own,
and there is no dependency cycle (TO does not know about QuasiStrided). A
small, purely additive, non-blocking docs PR to TensorOperations.jl
(`docs/src/man/backends.md`) is task T15, deliberately off the critical
path.

Consequence for `Project.toml` (T7): `TensorOperations` moves into `[deps]`
and `[compat]`, not `[weakdeps]`. `TupleTools` is also required in `[deps]`
— `_qs_labels` below calls `TupleTools.invperm`, and TO's own dependency on
TupleTools is not a public re-export.

**Module-level import convention (frozen).** `src/QuasiStrided.jl` brings
TensorOperations in as `import TensorOperations as TO`, plus at most an
explicit `using TensorOperations: <names>` list restricted to names that do
not collide with QuasiStrided's own (`Index2Tuple`, `linearize` are safe).
A bare `using TensorOperations` is **not** permitted: TO exports
`scalartype` (re-exported from VectorInterface) and QuasiStrided defines its
own, unrelated `scalartype` in `src/kernel_descriptor.jl`. Which binding
wins then depends on definition ordering inside the module, which is exactly
the kind of implicit coupling this record exists to prevent. Every TO name
used in `src/tensoroperations.jl` is written `TO.<name>` unless it appears
in the explicit `using ... : ...` list.

### `QuasiStridedBackend`

```julia
struct QuasiStridedBackend <: TensorOperations.AbstractBackend end
```

A plain singleton: no fields, no fallback backend field, no type
parameters. The hard-reject decision below is what removes the need for a
fallback field (and with it the delegation-recursion hazard the original
design sketch carried). It is the **only** name this package exports.

### Not hooked into `select_backend`

QuasiStrided defines **no** `TO.select_backend` method. Users must pass
`backend=QuasiStridedBackend()` explicitly (`@tensor backend=... `,
`ncon(...; backend=...)`, or a direct `TO.tensorcontract!` call). Merely
loading QuasiStrided must never change the behavior, performance, or
allocation profile of existing `@tensor` code.

This is a deliberate, evidence-backed choice, not caution for its own sake:
QuasiStrided is expected to lose to `StridedBLAS()` on most shapes (T9
measures this honestly), so silently capturing the default dispatch would be
a regression for every existing user of the default path.

### Hard-reject, never fall back

This backend follows TensorOperations' `TBLISBackend` convention exactly
(`ext/TensorOperationsTBLISExt.jl`, its `check_arguments`/`throw_eltype`/
`throw_strided` helpers): when it cannot handle an operation it throws, and
never silently routes the work elsewhere.

1. `TO.tensoradd!(..., ::QuasiStridedBackend, ...)` and
   `TO.tensortrace!(..., ::QuasiStridedBackend, ...)` **always** throw
   `ArgumentError`, unconditionally, with a message stating that this
   backend implements contraction only and pointing at using a different
   `backend=` for networks that need add/trace. QuasiStrided has no
   `tensoradd!`/`tensortrace!` analog and no diagonal/trace support at all
   (`src/driver.jl`'s `_classify_labels` rejects repeated labels), so there
   is nothing to delegate to.
2. `TO.tensorcontract!(..., ::QuasiStridedBackend, ...)` throws
   `ArgumentError` for every ineligible input class — wrong or mixed
   eltype, non-strided operands, aliased output — rather than falling back
   to `StridedNative`/`StridedBLAS`.

Rationale for hard-reject over fallback: a fallback makes the observed
performance of `backend=QuasiStridedBackend()` silently depend on whether
the request was actually served by this engine, which destroys the value of
T9's measurement and of any user-side benchmark. It also matches the only
in-tree precedent (TBLIS).

Note that a `@tensor` network containing both a contraction and an
add/trace step therefore cannot be run wholesale under this backend this
milestone. That is a known, accepted scope limitation, and `README.md` (T12)
must say so plainly.

### Eligibility predicate, and the conjugation invariant

`TO.tensorcontract!` for `QuasiStridedBackend` accepts exactly:

```
eltype(A) === eltype(B) === eltype(C)  &&
eltype(C) ∈ (Float32, Float64)         &&
all(StridedViews.isstrided, (A, B, C))
```

Anything else throws (previous subsection). The eltype restriction is not a
convenience: `src/kernel_descriptor.jl` throws unless
`T === Float32 || T === Float64`, and `default_blocking` only has measured
constants for those two.

**LOAD-BEARING INVARIANT (frozen).** QuasiStrided ignores
`StridedView.op` and TO's `conjA`/`conjB` flags **entirely**. This is
correct *only* because the eltype is restricted to real `Float32`/`Float64`,
on which every value `op` can take (`identity`, `conj`, `transpose`,
`adjoint` — all applied elementwise) is the identity, and conjugating a real
`α`/`β` is likewise a no-op. TO's own TBLIS extension encodes the same fact
as `isconj(A::StridedView{T}, conjA) = T <: Complex && (conjA ⊻ (A.op === conj))`,
which is unconditionally `false` for real `T`.

The consequences are binding on T4 and on any later relaxation of the
eltype restriction:

- `src/tensoroperations.jl` must carry this reasoning as a source comment at
  the point where `conjA`/`conjB` are dropped — not as a silent omission.
- T6/T2 must include a test that pins it (real operands with `conjA`/
  `conjB` set true still give results identical to `StridedNative()`).
- **Adding complex eltype support in a future milestone is not a matter of
  widening the eltype check.** It requires handling `op`/`conj` explicitly
  first, and the eltype check is the only thing standing between the current
  code and silently wrong results for complex inputs. Do not widen it
  without doing that work.

### Label mapping `pA`/`pB`/`pAB` -> QuasiStrided labels (frozen contract)

TensorOperations' `tensorcontract!` computes
`C = β*C + α*permutedims(contract(opA(A), opB(B)), pAB)`, where the indices
`pA[2]` of `A` are contracted with `pB[1]` of `B` and the remaining indices
`(pA[1]..., pB[2]...)` are permuted by `pAB` (`src/interface.jl:121-135` of
TensorOperations v5.8.0). QuasiStrided's own convention is one `Int` label
per axis in axis order (Phase 0 freeze, top of this file). The bijection
between the two is:

```julia
function _qs_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple)
    NoA, Nk = numout(pA), numin(pA)
    qA = TupleTools.invperm(linearize(pA))
    qB = TupleTools.invperm(linearize(pB))
    indA = map(s -> s <= NoA ? s : -(s - NoA), qA)
    indB = map(s -> s <= Nk  ? -s : NoA + (s - Nk), qB)
    return indA, indB, linearize(pAB)
end
```

Name resolution for that snippet: `Index2Tuple` and `linearize` are exported
by TensorOperations; `numout`/`numin` are **not**, and must be written
`TO.numout`/`TO.numin` (or added to the explicit `using ... : ...` list) per
the import convention above. `TupleTools.invperm` needs `TupleTools` in
`[deps]`.

Label alphabet it produces: `1:NoA` for A's open axes (in `pA[1]` order),
`NoA+1 : NoA+NoB` for B's open axes (in `pB[2]` order), and `-1:-1:-Nk` for
the contracted pairs (in `pA[2]`/`pB[1]` order, which TO guarantees are
positionally matched). `indC` is `linearize(pAB)` verbatim, because the
intermediate tensor's slot `j` carries label `j` by construction and
`permutedims`' convention is "output axis `c` takes input axis `perm[c]`".

This is structurally the same bijection TensorOperations computes for its
own backends in `contract_labels` (`src/implementation/indices.jl:140-157`),
which assigns `pA[1] -> 1:numout(pA) .+ OFFSET_OPEN`,
`pA[2] -> 1:numin(pA) .+ OFFSET_CLOSED`,
`pB[2] -> 1:numin(pB) .+ (OFFSET_OPEN + numout(pA))`,
`pB[1] -> 1:numout(pB) .+ OFFSET_CLOSED`, and
`linearize(pAB) .+ OFFSET_OPEN` — the identical assignment in a `Char`
alphabet with two offsets instead of an `Int` alphabet with a sign bit. That
independent agreement, not just the local derivation, is the reason this is
frozen rather than provisional.

**Worked example, verified numerically 2026-09-09** against
TensorOperations v5.8.0 on Julia 1.12.6 (the plan file asserted the mapping
was checked against TO's TBLIS fixtures but did not record a worked case;
this one was computed and cross-checked during T0, both against a hand
derivation and against a brute-force loop over the QuasiStrided labels):

```
pA  = ((3,1,4),(2,5))        # A has 5 axes: 3 open, 2 contracted
pB  = ((3,1),(2,4))          # B has 4 axes: 2 contracted, 2 open
pAB = ((4,2),(5,1,3))        # linearize(pAB) = (4,2,5,1,3)

indA = ( 2, -1,  1,  3, -2)
indB = (-2,  4, -1,  5)
indC = ( 4,  2,  5,  1,  3)
```

Reading it back: A's open axes in `pA[1]` order are `(3,1,4)`, and they
carry labels `1,2,3` (`indA[3]=1`, `indA[1]=2`, `indA[4]=3`); A's contracted
axes `(2,5)` carry `-1,-2`; B's contracted axes `(3,1)` carry the *same*
`-1,-2` (`indB[3]=-1`, `indB[1]=-2`), so `A`'s axis 2 pairs with `B`'s
axis 3 and `A`'s axis 5 with `B`'s axis 1, exactly as `pA[2]`/`pB[1]`
require; B's open axes `(2,4)` carry `4,5`. With extents
`size(A)=(2,3,4,5,6)` and `size(B)=(6,7,3,8)`, `size(C)=(7,2,8,4,5)` and the
`tensorcontract!` result agreed with the brute-force reference to
`2.7e-15` (Float64).

`_qs_labels` performs **no** validation of its own. The consistency checks
`numout(pA) + numin(pB) == numind(pAB)` and `numin(pA) == numout(pB)` are
left to TO's `argcheck_tensorcontract`, which the adapter calls first
(next-but-one subsection). Do not duplicate them.

**Index-model coverage.** QuasiStrided's index model covers TO's contraction
index space exactly: diagonal labels, batch labels, C-only labels and
dangling labels cannot arise from `pA`/`pB`/`pAB`, which are permutations by
construction. `_classify_labels`' rejection paths in `src/driver.jl` are
therefore dead code *from this entry point* — they stay, because
`contract!` remains reachable directly, but no new index-model work is
needed this milestone. Empty index groups (outer products, full contraction
to a 0-dim output) already work; they need a pinning test (T6/T2), not code.

### Required argument-checking order in the adapter (frozen)

`TO.tensorcontract!` for `QuasiStridedBackend` must perform, in this order,
before touching the engine:

1. the eligibility predicate above (eltype, strided), throwing
   `ArgumentError`;
2. `TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)`;
3. `TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)`;
4. `(Base.mightalias(C, A) || Base.mightalias(C, B)) && throw(ArgumentError(...))`.

Step 4 is not optional and has no equivalent inside QuasiStrided:
`plan_contract`/`execute!` have **no aliasing check whatsoever**, and TO's
`tensorcontract!` docstring states as a warning that `C` must not alias `A`
or `B`. This ordering mirrors `TensorOperationsTBLISExt.jl:172-185`
line for line.

Only after those four does the adapter convert `α`/`β` to `eltype(C)` and
hand the work to the engine. It does **not** call the frozen
`QuasiStrided.contract!` entry point: it calls `plan_contract` and `execute!`
separately, because `contract!` exposes no way to pass the `workspace` and
`oracle` keywords that Amendment 1's pooling requires (`workspace = ws,
allocator, oracle = false` on the `DefaultAllocator` path; `workspace =
nothing, allocator, oracle = false`, bracketed with
`allocator_checkpoint!`/`allocator_reset!` and a `release!`, on the
explicit-allocator path). The label semantics are identical either way —
`contract!` is itself a thin `plan_contract` + `execute!` wrapper.

**Addendum (T11, 2026-09-09): the aliasing check runs on the `StridedView`s.**
This is a small **amendment to the frozen order above**, recorded here rather
than as a numbered Amendment because it changes only *where in the sequence*
the wrap happens, not what is checked or what is accepted. The order is now
eligibility → argcheck → dimcheck → **wrap with `StridedView`** → aliasing,
with step 4 reading `Base.mightalias(Cv, Av) || Base.mightalias(Cv, Bv)` on
the wrapped operands instead of on the raw `C`/`A`/`B`. Reason: `Base` defines
no `Base.dataids` method for `PermutedDimsArray`, so it falls back to an
`objectid`-derived id rather than the parent array's pointer, and
`Base.mightalias(PermutedDimsArray(P, (2,1)), P)` returns `false`. The frozen
step 4 therefore silently accepted `C = PermutedDimsArray(A, ...)` and
produced a wrong result. `StridedViews.jl` defines `Base.dataids(a::StridedView)
= Base.dataids(a.parent)` (`StridedViews` v0.5.2,
`src/stridedview.jl:288`) and
`StridedView` unwraps `PermutedDimsArray`/`Adjoint`/`Transpose`/`SubArray`
down to the shared parent, so the same test on the wrapped views collapses
correctly. This is an upstream `Base` gap that `StridedNative`/`StridedBLAS`
share; the adapter closes it for its own path only. The reorder is
behavior-preserving for every previously-correct outcome — `StridedView` is
pure, side-effect-free, and the identity on an operand that is already a
`StridedView`, and eligibility (step 1) has already established that all three
operands are strided, so the wrap cannot itself throw where the old order
would not have. Verified by re-running the full suite (13167 → 13171, no
regressions) and by the `"C is a PermutedDimsArray of A"` regression testset
in `test/test_tensoroperations.jl`.

### Amendment 1: `ContractWorkspace` and the `allocator` keyword

This **amends** two earlier freezes: the Phase 0 `contract!` entry point
("Frozen interfaces" at the top of this file) and the macro-blocking
milestone's `plan_contract(...; kernel, mc, kc, nc)` signature ("Frozen
interface additions", item 4). Both gain an `allocator` keyword; the
buffer fields currently inlined in `ContractPlan` move into a separate
`ContractWorkspace{T,...}`. Nothing else about either signature changes, and
the label semantics of the Phase 0 freeze are untouched.

```julia
plan_contract(C, A, indA, B, indB, indC;
              kernel = ..., mc = nothing, kc = nothing, nc = nothing,
              workspace = nothing,
              allocator = TensorOperations.DefaultAllocator(),
              oracle = true) -> ContractPlan
```

`workspace = nothing` builds a fresh one; passing an existing
`ContractWorkspace` reuses it via `reserve!`. `oracle` controls whether the
`execute_tilewise!` buffers (`tw_*`) are allocated at all; the backend path
passes `oracle = false`, the in-repo tests keep the default so
`execute_tilewise!` stays available as the independent oracle the
macro-blocking milestone froze it as.

Motivation: default `Float64` blocking allocates roughly 860 KiB of
`zeros(...)` per `plan_contract` call (`packed_b` ~768 KiB, `packed_a`
~64 KiB, plus the offset/descriptor buffers). Under TO's per-call
`tensorcontract!` entry point there is no user-visible plan to hold onto, so
without reuse every single `@tensor` contraction pays that. Every buffer
consumer (`pack_a!`/`pack_b!`, `fill_offsets!`, `_classify_slivers!`,
`_sliver_range`) already tolerates oversized buffers, and `_pack_panel!`
writes every slot including padding, so reuse needs **no change to any
hot-loop signature** — only `resize!`-upward on the buffers themselves. That
last property is also why `zeros(...)` can become `undef` buffers.

#### Verified allocator behavior (measured during T0, 2026-09-09)

TensorOperations v5.8.0, Julia 1.12.6, `tensoralloc(Vector{Float64}, (16,),
Val(istemp), allocator)`:

| allocator | `Val(true)` returns | `Val(false)` returns | `resize!`able | `SIMD.vload` |
| --- | --- | --- | --- | --- |
| `DefaultAllocator()` | `Vector{Float64}` | `Vector{Float64}` | yes | yes |
| `ManualAllocator()` | `PtrArrays.PtrArray{Float64,1}` | `Vector{Float64}` | **no** (`MethodError`) | yes |
| `BufferAllocator()` | `Vector{Float64}` (unsafe-wrapped into the buffer) | `Vector{Float64}` | *apparently* — do not | yes |
| Bumper `SlabBuffer` | `UnsafeArrays.UnsafeArray{Float64,1}` | `Vector{Float64}` | **no** (`MethodError`) | yes |

Three facts from that table are load-bearing and **correct the plan file**,
which asserted that `tensoralloc` "returns a concrete `Vector{T}` regardless
of which allocator branch is taken":

1. It does not. `ManualAllocator` yields a `PtrArrays.PtrArray` and Bumper
   yields an `UnsafeArrays.UnsafeArray`. Only `Val(false)` (non-temporary)
   requests return a plain `Vector` from every allocator, because every
   allocator falls through to `A(undef, structure)` in that case.
2. All four returned types are `<: DenseArray`, hence within SIMD.jl's
   `FastContiguousArray` union, so `SIMDKernel`'s `vload`/`vstore` on the
   packed buffers works for all of them — verified by direct call, not
   inferred. This is what makes the explicit-allocator path viable at all.
3. Allocator-provided temporaries must **never** be `resize!`d. For
   `PtrArray`/`UnsafeArray` it is a `MethodError`; for `BufferAllocator`'s
   unsafe-wrapped `Vector` the call appears to *succeed*, which would
   silently detach the buffer from the arena. Sizing happens once, at
   acquisition.

#### Binding constraints on T3 and T5

- **`ContractWorkspace{T, VT<:AbstractVector{T}}`** is parameterized on the
  packed-buffer vector type; `VT` is `Vector{T}` on the default path and
  `PtrArray{T,1}`/`UnsafeArray{T,1}`/`Vector{T}` on explicit-allocator
  paths. Every *instance* is concretely typed. This is **not** the widening
  the macro-blocking Phase A finding warned about: that bug was a
  *partially-applied* `QSTile{Memory{Float64}}` left open over its `R`/`C`
  parameters, boxed on every construction. A `where`-bound struct parameter
  resolved at construction is the pattern `ScatterAxis{V}` and
  `ContractPlan{T,Kern,...}` already use. The prohibition that stands
  unchanged: **no `Union`-typed field, no `AbstractVector` field, no
  `view`s into a differently-typed buffer, and no value of union type
  crossing a function boundary into packing or kernel code.** Verify with
  `@code_warntype` exactly as Phase C did.
- **Offset buffers stay `Vector{Int}`, unconditionally.** `fill_offsets!`
  (`src/axis_group.jl:136`), `block_descriptors!` (`:261`),
  `describe_block` (`:227`) and `axis_from_descriptor` (`src/tiles.jl:82`)
  are all frozen with concrete `Vector{Int}` parameters. Widening them is
  explicitly **out of scope** this milestone. The offset buffers are
  therefore acquired with `Val(false)`, which yields a real `Vector{Int}`
  from every allocator; only `packed_a`/`packed_b` are `Val(true)`
  temporaries that genuinely route through the allocator. Only the packed
  buffers were widened to `AbstractVector{T}` (macro-blocking Phase B,
  item 3), and that is exactly what makes this split possible.
- **Default path** (`allocator isa TO.DefaultAllocator`, i.e. the user asked
  for nothing special): one persistent `ContractWorkspace{T,Vector{T}}` per
  `(task, eltype)` in **task-local storage** (correct under task migration,
  no locks), grown by `reserve!` with `resize!`-upward-only on the plain,
  GC-owned `Vector`s it holds. This is the zero-steady-state-allocation fast
  path and needs no user opt-in. It is built from ordinary `Vector`
  allocation (equivalently `tensoralloc(..., Val(false), DefaultAllocator())`),
  **not** against a `TO.ManualAllocator()` as the plan file said: a
  `ManualAllocator` temporary is a `PtrArray`, which cannot be `resize!`d,
  so `reserve!`'s upward-growth discipline and `ManualAllocator` are
  mutually exclusive. The plan file's `ManualAllocator()` wording is
  superseded by this paragraph.
- **Explicit-allocator path** (the user wrote `@tensor allocator=...`, e.g.
  `BufferAllocator` or a Bumper-backed buffer): the adapter builds a
  `ContractWorkspace` scoped to that call by calling `TO.tensoralloc` with
  `Val(true)` against the caller's allocator, sized exactly once from the
  plan's effective blocking, and releases it with `TO.tensorfree!` (plus
  `allocator_checkpoint!`/`allocator_reset!` around the call) before
  returning. No `reserve!`, no resizing, no task-local state touched. This
  is what actually gives arena/LIFO semantics, and it composes with whatever
  other temporaries the same `@tensor` network draws from that allocator.
- `reserve!`'s contract: grow-only on GC-owned `Vector`s; never shrink;
  never reallocate a buffer that is already large enough; never hand back a
  `view`. An oversized buffer must be safe to consume, which is a property
  the existing consumers already have and T3 must regression-test rather
  than assume.
- QuasiStrided gains **no new runtime dependency** from this: `Bumper` stays
  a weak dependency on TensorOperations' side, and is added to QuasiStrided
  only as a *test-only* extra (T7) so T2 can prove the arena path really is
  scoped rather than silently falling back to the default pool.

### Amendment 2: `SIMDKernel` becomes the default kernel

`_default_kernel(::Type{T}) = ScalarKernel(Val(8), Val(6), T)`
(`src/driver.jl:109`) becomes `SIMDKernel(Val(8), Val(6), T)`, engine-wide —
for the frozen `contract!` entry point, for `plan_contract`'s own `kernel`
keyword default, and for the new backend path alike. This is a deliberate
amendment to the default as it stands in the macro-blocking milestone's
frozen `plan_contract(...; kernel, mc, kc, nc)` signature ("Frozen interface
additions", item 4), in `plan_contract`'s docstring, and in `README.md` —
not a backend-path-only override, so that `contract!` and `@tensor
backend=QuasiStridedBackend()` never disagree about what "default" means.

Evidence: Phase E of the macro-blocking milestone measured `SIMDKernel` at
3.5x-10.5x faster than `ScalarKernel` through the full macro-blocking driver
across the shape grid (and 3.9x-6.3x at the single-tile level in Phase 3).
`ScalarKernel` remains public and selectable; it stays the readable
reference implementation and the oracle the SIMD kernel is checked against.

**Known caveat, carried forward, not fixed here.** Per `STATUS.md`'s
"Published" note: `SIMDKernel`'s `accumulate`/`execute_tile!` are
zero-allocation on Julia >= 1.11 but allocate tens of KB per call on Julia
1.10 (LTS) — a compiler capability gap (the `NTuple{NV,Vec{W,T}}`
accumulator is not kept register-resident by the older compiler), not a
correctness issue. All correctness assertions pass on 1.10. Flipping the
default therefore changes the *allocation* profile of the default path on
1.10 while improving it everywhere else. Consequences:

- The existing Julia `lts`/`1` x ubuntu/macos CI matrix is **kept** (T7) so
  the gap stays visible rather than hidden.
- Any allocation assertion added this milestone against the default path
  must be marked `skip=(VERSION < v"1.11")`, matching the existing treatment
  in `test/test_simd_kernel.jl` — weakening or deleting such an assertion is
  not an acceptable alternative.
- `README.md` (T12) states the default kernel and this caveat.

### Public / internal API split: three tiers

Required, not cosmetic: `scalartype` is exported by both packages today, and
`mr`, `nr`, `accumulate`, `offsets` and `execute!` are all plausible
collisions for a user doing `using TensorOperations, QuasiStrided`. After
this split there is exactly one exported name, so the two packages coexist
under simultaneous `using` by construction.

| Tier | Mechanism | Names |
| --- | --- | --- |
| Exported | `export` | `QuasiStridedBackend` |
| Public, unexported | `public` (Julia >= 1.11 only) | `contract!`, `plan_contract`, `execute!`, `ContractPlan`, `ContractWorkspace`, `Blocking`, `default_blocking`, `ScalarKernel`, `SIMDKernel` |
| Internal | none | `AxisGroup`, `axis_length`, `offsets`, `fill_offsets!`, `BlockDescriptor`, `describe_block`, `block_descriptors!`, `normalize_group`, `KernelDescriptor`, `mr`, `nr`, `scalartype`, `packed_a_offset`, `packed_b_offset`, `packed_a_length`, `packed_b_length`, `AffineAxis`, `ScatterAxis`, `SourceTile`, `DestinationTile`, `axis_from_descriptor`, `nrows`, `ncols`, `axis_offset_range`, `checked_tile_storage_bounds`, `pack_a!`, `pack_b!`, `zero_accumulator`, `accumulate`, `scale_tile!`, `store_tile!`, `execute_tile!`, `lanewidth`, `avecs_per_column` |

That is 1 exported + 9 public + 34 internal. The 34 are exactly the 42 names
`src/QuasiStrided.jl` exports at revision `314c47f` minus the 8 that move to
the public tier (`ContractWorkspace` is new this milestone). Names that are
already unexported stay unexported and are not listed — notably
`execute_tilewise!`, `ScalarDestination`, `affine_axis`, `scatter_axis`,
`QSTile`.

`public` is a Julia 1.11 keyword and this package's compat is `julia =
"1.10"`, so it must be version-guarded. Frozen form:

```julia
@static if VERSION >= v"1.11"
    eval(Expr(:public, :contract!, :plan_contract, :execute!, :ContractPlan,
              :ContractWorkspace, :Blocking, :default_blocking,
              :ScalarKernel, :SIMDKernel))
end
```

(`Expr(:public, ...)` rather than `Meta.parse("public ...")`: `public x, y`
is a syntax error on 1.10, so the surface syntax cannot appear in the file
at all, even unreached.)

Internal names remain docstringed and remain freely usable inside the
package and its own tests; they are simply not API and carry no semver
promise. `test/runtests.jl` `include`s every test file into `Main`, so
restoring them for the test suite is a single `using QuasiStrided: <names>`
block in that one file (T1) rather than a per-file edit; each benchmark
script needs the same one-line treatment. **T1 is behavior-neutral: the
`Pkg.test()` pass count must be unchanged at 12903/12903.**

### File ownership additions (append to the existing table)

| File | Owner | Phase |
| --- | --- | --- |
| `src/tensoroperations.jl` (new) | TO-adapter implementer | TO-B (T4), TO-C (T5) |
| `src/workspace.jl` (new), `src/driver.jl`, `test/test_driver.jl` | workspace implementer | TO-B (T3) |
| `src/QuasiStrided.jl` (export/`public`/`include` blocks), `test/runtests.jl`, `benchmark/bench_axis_group.jl`, `benchmark/bench_driver.jl` (name-restoring `using` blocks only) | API-split implementer | TO-B (T1) |
| `test/test_tensoroperations.jl` (new) | TO-comparison test author (blind) | TO-B (T6), then TO-C (T2) |
| `Project.toml`, `.github/workflows/CI.yml` | packaging implementer | TO-B (T7) |
| `test/test_quality.jl` (new) | quality-gate implementer | TO-C (T8) |
| `benchmark/bench_tensoroperations.jl` (new) | TO-benchmark implementer | TO-D (T9) |
| `README.md`, docstrings in `src/tensoroperations.jl` | main process | TO-F (T12) |
| `docs/decisions.md`, `STATUS.md` | main process | all |

Known, anticipated collision: T1 and T4 both edit `src/QuasiStrided.jl` (one
rewrites the `export` blocks and adds the `public` block, the other adds one
`include` and one `export`). Reconcile at integration, as Phase 2 and
Phase 3 of the bootstrap milestone already did; neither worker blocks on the
other. `src/axis_group.jl`, `src/tiles.jl`, `src/packing.jl`,
`src/kernel.jl`, `src/kernels/simd.jl` and `src/kernel_descriptor.jl` are
**frozen** this milestone — the offset-buffer typing constraint above exists
precisely so none of them needs to change.

### Correctness oracle: cross-package, one deliberate deviation from precedent

Every previous milestone used an in-repo independent oracle (the
`CartesianIndices` oracle in Phase 1, `execute_tilewise!` in the
macro-blocking milestone). This milestone's oracle is **cross-package**:
TensorOperations' own `StridedNative()` and `StridedBLAS()` backends, run on
identical inputs. That is a stronger check for an adapter whose entire job
is to agree with TO's semantics, and it is the reason T6's author is pointed
at TensorOperations' `test/tblis.jl` as the structural template.

T6 is **authored blind**, per this repo's established practice: its author
works from this frozen section and from `test/tblis.jl`, and must not read
`src/tensoroperations.jl`. T2 comes strictly after both T6 and the adapter
integration, and adds only what someone who has read the adapter can write.

## TensorOperations integration milestone: close

Written at T13, 2026-09-09, on branch `tensoroperations` (base `314c47f`,
still entirely uncommitted — see `STATUS.md`'s "Integrated revision"). The
Phase A section above is the frozen *design*; this section is the record of
what actually shipped against it, what was measured, how the gated review was
dispositioned, and what stays out of scope. **T14, the second and last gated
review pass, runs after this section and is not reflected in it** — Phase F's
checkbox in `STATUS.md` stays unchecked until it is.

Why this milestone gets a distinct "close" subsection when the macro-blocking
one did not: there, Phase D (review) and Phase E (benchmark) each already had
their own narrative section here, and Phase F's close needed only a
`STATUS.md` entry. Here the only section written above is Phase A's freeze
plus T11's addendum inside it — the T9 benchmark and the T10/T11 review
disposition would otherwise have no home in the authoritative *why* record.

### What shipped

1. **The adapter** — `src/tensoroperations.jl` (new, 378 lines). One
   `TO.tensorcontract!` method on `::QuasiStridedBackend`, plus
   `TO.tensoradd!`/`TO.tensortrace!` methods that throw `ArgumentError`
   unconditionally. `QuasiStridedBackend` is the package's single exported
   name. The frozen argcheck order is implemented as written, as amended by
   T11: eligibility (`eltype(C) === eltype(A) === eltype(B) ∈
   (Float32, Float64)` and all three `StridedViews.isstrided`) →
   `TO.argcheck_tensorcontract` → `TO.dimcheck_tensorcontract` →
   `StridedView` wrap → `Base.mightalias` on the wrapped operands. The
   conj/`op` invariant is carried as a source comment at the point where
   `conjA`/`conjB` are dropped, as the freeze required. `_qs_labels` shipped
   verbatim from the frozen snippet, with only the `TO.numout`/`TO.numin`
   qualification the import convention requires. No `TO.select_backend`
   method exists, and no path falls back: every ineligible input throws.
2. **Workspace pooling** — `src/workspace.jl` (new) factors the packed and
   offset buffers out of `ContractPlan` into
   `ContractWorkspace{T,VT<:AbstractVector{T}}`, with `reserve!`
   (grow-only, never shrink, never a `view`) and `release!`.
   `plan_contract`/`execute!`/`contract!` gained `workspace`, `allocator`
   and `oracle` keywords (Amendment 1). The backend runs two paths: on
   `TO.DefaultAllocator` it reuses one persistent workspace per
   `(task, eltype)` from `task_local_storage`; on an explicit allocator it
   builds a call-scoped workspace from `TO.tensoralloc`, bracketed by
   `allocator_checkpoint!`/`allocator_reset!` with a `release!` in a
   `finally`, touching no task-local state. `oracle = false` on both backend
   paths, so `execute_tilewise!`'s buffers are never allocated there.
3. **Three-tier API split** — 1 exported (`QuasiStridedBackend`), 9 `public`
   but unexported behind the `@static if VERSION >= v"1.11"` /
   `Expr(:public, ...)` guard, 34 internal, exactly as tabulated above. The
   split is what makes `using TensorOperations, QuasiStrided` safe (both
   packages define an unrelated `scalartype`). T1 held the pass count at
   12903/12903, confirming it was behavior-neutral.
4. **Default-kernel flip** — `_default_kernel` returns
   `SIMDKernel(Val(8), Val(6), T)` engine-wide (Amendment 2), for
   `contract!`, `plan_contract` and the backend path alike, with the Julia
   1.10 allocation caveat carried into `README.md` and the CI matrix left
   intact so it stays visible.
5. **Tests and packaging** — `test/test_tensoroperations.jl` (new, 521
   lines; T6 authored blind against `StridedNative`/`StridedBLAS`, extended
   by T2 after integration), `test/test_quality.jl` (new; `Aqua.test_all`
   with `ambiguities=false` and `unbound_args=false`, each disabled with a
   recorded, reproduced justification — the piracy check that matters for
   adding methods to TO's generics stays on). `Project.toml` gained
   `TensorOperations` 5.8 and `TupleTools` 1.6 as hard deps and `Aqua`
   0.8 / `Bumper` 0.7 as test-only extras. `.github/workflows/CI.yml` was
   **not** changed: Amendment 2 requires keeping the existing
   `lts`/`1` x ubuntu/macos matrix.

### T9 benchmark outcome: honest, and the reason `select_backend` stays unhooked

`benchmark/bench_tensoroperations.jl`, run on `ccqlin038` (Xeon Gold 6244,
Cascade Lake), Julia 1.12.6, `Threads.nthreads() == 1`,
`BLAS.set_num_threads(1)`, 9 timed reps per point (plus a discarded warm-up),
median reported. Full provenance and raw CSV in
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-09/`. Canary
bracket (three repeats of the same point across the sweep): relative spread
`(max-min)/min = 3.22%`, no drift. Same single-machine caveat as every other
measurement in this project: one microarchitecture, one session, no
portability claim.

All three backends driven through the same `@tensor` expression. Times in
microseconds (median):

| Float64 shape | StridedBLAS | QuasiStrided | StridedNative | QS/BLAS | Native/QS |
| --- | --- | --- | --- | --- | --- |
| `64^3` | 5.3 | 28.7 | 181.3 | 5.44x | 6.3x |
| `128^3` | 56.0 | 142.0 | 1,379.3 | 2.54x | 9.7x |
| `256^3` | 431.0 | 1,044.0 | 11,235.5 | 2.42x | 10.8x |
| `512^3` | 2,981.0 | 7,966.3 | 87,681.9 | 2.67x | 11.0x |
| `shallowK_256x24x256` | 46.3 | 202.4 | 1,074.5 | 4.37x | 5.3x |
| `1024x256x1024` | 5,562.9 | 15,529.5 | 172,916.6 | 2.79x | 11.1x |

| Float32 shape | StridedBLAS | QuasiStrided | StridedNative | QS/BLAS | Native/QS |
| --- | --- | --- | --- | --- | --- |
| `64^3` | 2.5 | 27.1 | 188.2 | 10.91x | 6.9x |
| `128^3` | 29.4 | 123.3 | 1,449.0 | 4.20x | 11.8x |
| `256^3` | 201.1 | 681.9 | 11,930.5 | 3.39x | 17.5x |
| `512^3` | 1,454.8 | 5,139.6 | 92,054.6 | 3.53x | 17.9x |
| `shallowK_256x24x256` | 25.6 | 183.9 | 1,126.9 | 7.18x | 6.1x |
| `1024x256x1024` | 2,834.2 | 9,754.3 | 182,471.1 | 3.44x | 18.7x |

Reading it plainly:

- **`StridedBLAS()` wins at every one of the 12 measured points**, by
  **2.42x-5.44x** for `Float64` and **3.39x-10.91x** for `Float32`. The gap
  is worst at the smallest and shallowest shapes (`64^3`,
  `shallowK_256x24x256`), where per-call planning overhead is a large
  fraction of the work, and settles to a ~2.4-2.8x (`Float64`) / ~3.4-3.5x
  (`Float32`) plateau at `256^3` and above. That plateau is the honest
  measure of the microkernel gap against a tuned vendor GEMM; it is not
  closing with size.
- **QuasiStrided beats `StridedNative()` at every point**, by **5.3x-11.1x**
  (`Float64`) and **6.1x-18.7x** (`Float32`). Same ordering as the
  `StridedBLAS` result, for the same reason: the advantage is smallest where
  per-call overhead dominates.
- Ratios above are computed from the committed medians in
  `bench_tensoroperations.csv`, re-derived at T13. Note that
  `summary_tensoroperations.txt`'s "x of fastest" column is each backend
  relative to `StridedBLAS`, *not* QuasiStrided relative to
  `StridedNative`; dividing one by the other is what gives the `Native/QS`
  column. An earlier `STATUS.md` phrasing ("up to 34x/76x") had read those
  two off the wrong column and has been corrected.

This is exactly the evidence the "Not hooked into `select_backend`" decision
above was written in anticipation of, and it holds: silently capturing TO's
default dispatch would be a 2.4x-10.9x regression for every existing
`@tensor` user on this machine class. `backend=QuasiStridedBackend()` stays
opt-in, and hard-reject stays the policy so that any timing taken under it
provably measures this engine.

### T10 / T11 gated review disposition

Fable-High read-only review completed 2026-09-09
(`fable_review_tensorops_used: true` — spent, do not relaunch this
milestone). Scope: `_qs_labels` against TO's `pA`/`pB`/`pAB` semantics, the
conj/`op` invariant, aliasing and argcheck coverage, workspace-reuse
staleness across pooled calls, and whether any path silently falls back
instead of throwing. **No blocking findings.** Two should-fix items, both
fixed by T11:

1. **[should-fix, FIXED]** The frozen argcheck order's step 4 ran
   `Base.mightalias` on the raw `C`/`A`/`B`. `Base` defines no
   `Base.dataids` method for `PermutedDimsArray`, so it falls back to an
   `objectid`-derived id rather than the parent's pointer, and
   `Base.mightalias(PermutedDimsArray(P, (2,1)), P)` is `false`. The adapter
   therefore accepted `C = PermutedDimsArray(A, ...)` and produced a wrong
   result. Fixed by moving the check *after* the `StridedView` wrap:
   `StridedViews.jl` defines `Base.dataids(a::StridedView) =
   Base.dataids(a.parent)` and unwraps
   `PermutedDimsArray`/`Adjoint`/`Transpose`/`SubArray` to the shared
   parent, so the same test collapses correctly. Behavior-preserving for
   every previously-correct outcome (`StridedView` is pure and the identity
   on an already-wrapped operand, and step 1 has already established all
   three operands are strided, so the wrap cannot itself throw where the old
   order would not have). Full reasoning, and the amendment to the frozen
   order, is the "Addendum (T11, 2026-09-09)" under "Required
   argument-checking order in the adapter" above. Pinned by the
   `"C is a PermutedDimsArray of A"` regression testset in
   `test/test_tensoroperations.jl`. This is an upstream `Base` gap that
   `StridedNative`/`StridedBLAS` share; the adapter closes it for its own
   path only.
2. **[should-fix, FIXED]** `README.md`'s allocation claim overstated what
   workspace pooling delivers. Corrected to state the measured truth:
   `execute!` itself is allocation-free (Julia >= 1.11), and pooling keeps
   per-call buffer allocation flat instead of growing with problem size, but
   the call *as a whole* is not allocation-free — `plan_contract`'s
   per-call bookkeeping (label classification, plan construction) still
   costs roughly **3.8 KB** on a 64³ `Float64` contraction, measured both
   through `@tensor` and through a direct `TO.tensorcontract!`, against tens
   of bytes for `StridedBLAS` on the same call.

T11 additionally fixed one **doc drift** item in this file, not a review
finding: the frozen argcheck subsection had described the adapter as going
through the frozen `QuasiStrided.contract!` entry point, which it cannot —
`contract!` exposes no way to pass the `workspace`/`allocator`/`oracle`
keywords Amendment 1's pooling requires, so the adapter calls
`plan_contract` and `execute!` separately. The label semantics are identical
either way (`contract!` is itself a thin wrapper over the two), so this was a
correction to the record, not to behavior.

Test count across Phase E: 13167 → 13171 passing (the four new assertions are
the `PermutedDimsArray` regression testset), no regressions. Re-confirmed at
T13: **13171/13171**.

### Explicitly NOT done (unchanged from the sections above)

This closing summary adds nothing to the deferred list and contradicts
nothing in it; it exists so the list can be read in one place at close.

- **`tensoradd!` / `tensortrace!`.** Contraction only. Both throw
  `ArgumentError` unconditionally under this backend, per "Hard-reject,
  never fall back". A `@tensor` network mixing a contraction with an
  add/permute or trace step therefore cannot run wholesale under
  `backend=QuasiStridedBackend()`. Known and accepted; `README.md` says so
  plainly.
- **Complex element types.** Blocked by the load-bearing conj/`op`
  invariant under "Eligibility predicate, and the conjugation invariant":
  QuasiStrided ignores `StridedView.op` and `conjA`/`conjB` entirely, which
  is correct *only* for real `Float32`/`Float64`. Widening
  `_qs_eltype_ok` without first handling `op`/`conj` explicitly would
  produce silently wrong results, not an error. Do not treat this as a
  one-line change.
- **Threading.** Deferred. The state layout does not preclude it — see the
  macro-blocking "Phase D" finding 4 above (`ContractPlan`'s `(jc,pc)` and
  per-`ic` state are already disjoint fields; parallelizing over `ic` would
  need the M-side extracted into a per-worker struct). The task-local
  workspace pool added this milestone is per-`(task, eltype)` and correct
  under task migration, so it does not add a new obstacle either.
- **GPU.** Deferred, untouched.
- **Autotuning across shapes**, CPU-dispatch tables, cache-probing or
  analytical block-size derivation. The `mc`/`kc`/`nc` defaults remain
  hardcoded per-dtype constants measured on one machine, by design — see
  "Block-size policy (settled)".
- **Mixed element types, non-strided operands, aliased output.** All
  rejected with `ArgumentError` rather than converted or worked around.
- **K padding, orientation swap, einsum string parsing, batch axes,
  diagonals, isolated reductions, a separate beta-addend tensor.** All
  unchanged and still out of scope.
- **Registration, merge, publication.** Nothing in the plan file or in this
  record specifies a merge or release step for this branch; `STATUS.md`'s
  "Published" section still describes `main` only and was deliberately left
  untouched at T13.

### Nits handed to T14 (found at T13, not fixed here)

T13's file ownership is `STATUS.md` and this file only, so two `README.md`
inconsistencies noticed while cross-checking the close against T12's rewrite
are recorded rather than fixed:

1. `README.md`'s "Status" section says `Pkg.test()`: **13167/13167**. The
   current count is **13171/13171** — the 13167 figure predates T11's
   `PermutedDimsArray` regression testset.
2. `README.md`'s closing paragraph says `QuasiStridedBackend()` "beats
   `StridedNative()` by 6x-22x on every shape". The committed CSV gives
   **5.3x-18.7x** (5.3x-11.1x `Float64`, 6.1x-18.7x `Float32`); neither
   endpoint of "6x-22x" is supported by the data. The `StridedBLAS`
   range quoted in the same sentence (2.4x-10.9x) *is* correct.

Neither affects behavior; both are exactly the kind of stale-number drift the
macro-blocking milestone's own Phase F review caught, and are left for T14 to
confirm and dispose of.

**Disposed of before T14 ran.** Both were corrected directly in `README.md`
(main process, between T13 and T14): the test count now reads
`13171/13171` and the `StridedNative()` comparison now reads `5.3x-18.7x`.
T14 confirmed both strings are absent from the current `README.md` and
verified the corrected numbers are accurate — no further action needed.
