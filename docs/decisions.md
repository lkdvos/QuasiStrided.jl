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

> **Amended.** The register shape `(MR, NR, W)` is now derived from detected
> hardware, and `default_blocking` gained one ISA-keyed measured row. See
> "Hardware-derived register shape milestone: Phase G" at the end of this
> file for what survives of the decision below and what changed. No
> analytical or probing model chooses a number at runtime, which is the part
> of this section that still stands.

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

### Amendment 7 (2026-09-16): `tensoradd!`/`tensortrace!` fall back to `StridedNative`

Reverses clause 1 of "Hard-reject, never fall back" above, at the user's
explicit direction, prompted by the upstream benchmark-suite comparison
milestone's benchmark-only `QuasiStridedComposite` wrapper (`benchmark/composite_backend.jl`
on branch `upstream-bench`) — a type built only so that suite's
one-backend-per-provider interface could exercise `:permute`/`:trace`
categories against `QuasiStridedBackend`. Rather than keep that forwarding
logic quarantined in `benchmark/`, it becomes `QuasiStridedBackend`'s own
default behavior for these two operations, and the composite wrapper is
retired (no longer needed — `QuasiStridedBackend` itself now handles every
category the suite's `ArrayProvider` interface can throw at it).

**What changes:** `TO.tensoradd!`/`TO.tensortrace!` now forward
unconditionally to `TO.StridedNative()` instead of throwing. **What does
not change:** clause 2 — `TO.tensorcontract!` still hard-rejects (throws,
never falls back) every ineligible input (wrong/mixed eltype, non-strided
operand, aliased or conjugated output). A timing taken on a *contraction*
under `QuasiStridedBackend()` still always measures this engine, unaffected
by this amendment; only a timing taken on a bare `tensoradd!`/`tensortrace!`
call (or the add/trace portion of a mixed `@tensor` network) now measures
`StridedNative`, not this engine — callers comparing against a `StridedNative`
baseline should be aware the two are no longer independently distinguishable
on that portion of the work.

The original rationale for hard-rejecting add/trace ("a fallback makes the
observed performance of `backend=QuasiStridedBackend()` silently depend on
whether the request was actually served by this engine") is **not
refuted** — it is accepted, with the same tradeoff the benchmark-only
composite already made and documented, in exchange for `@tensor` networks
mixing contraction with an add/trace step now being runnable wholesale
under one backend (removing the exact scope limitation the prior paragraph
just described as "known, accepted" — it's the reason a fallback was worth
building in the first place, first as a benchmark-only wrapper and now as
the real default). The frozen `QuasiStridedBackend` struct (no fields, no
type parameters — see above) is unchanged; the fallback target is hardcoded
to `TO.StridedNative()`, not configurable, matching the composite wrapper's
own default and keeping the struct a plain singleton.

`benchmark/composite_backend.jl` and `benchmark/check_composite_backend.jl`
(from the upstream-bench milestone) are removed as part of this amendment —
their forwarding logic is now `QuasiStridedBackend`'s own behavior, so the
benchmark-only wrapper has no remaining purpose; `benchmark/bench_to_suite.jl`
is updated to call `QuasiStridedBackend()` directly wherever it previously
built a `QuasiStridedComposite()`.

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
  measure of the gap against a tuned vendor GEMM; it is not closing with
  size.

  > **Attributed 2026-09-11 (Phase G, refined in Phase H): that plateau is
  > not a *microkernel* gap.** At the BLIS `skx` register shape the pure-Julia `SIMDKernel`
  > reaches 101-103 GFLOP/s, ~88% of this machine's peak. What the plateau
  > measures is the driver handing each micro-tile a `view` of its macro
  > panel, which costs ~4x once the accumulator exceeds 16 vectors, plus
  > per-call planning overhead. See Phase G, "NV is held at 12
  > deliberately".
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
- **Complex element types.** ~~Blocked by the load-bearing conj/`op`
  invariant~~ — **no longer true as of the complex element-type milestone;
  see "Amendment 3" and the milestone sections at the end of this file.**
  `ComplexF32`/`ComplexF64` are supported, `conjA`/`conjB` are folded with
  each operand's `StridedView.op` by xor, and a conjugated *output* is
  rejected. What remains out of scope on the complex side is narrower: the
  3m method, mixed real/complex operands, and writing into a conjugated
  output view.

  The original text is struck rather than deleted because the reasoning it
  recorded — that widening `_qs_eltype_ok` without first handling `op`/`conj`
  would produce silently wrong results rather than an error — was correct
  about the code as it then stood, and is exactly why the milestone had to do
  that work first. Amendment 3 claimed this bullet had already been struck
  while it was in fact still standing and asserting the opposite of the
  shipped behaviour; that was caught by the milestone's gated review and is
  fixed here.
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

  **Correction (2026-09-19, Label-order milestone).** "Orientation swap" in
  this bullet is discharged: `plan_contract` now performs a guarded M/N
  operand-role swap (`_prefer_swap`, `T <: Real`) as part of ordinary
  planning. See "Label-order milestone" below. K padding, einsum string
  parsing, batch axes, diagonals, isolated reductions and a separate
  beta-addend tensor remain out of scope, untouched by this milestone.
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

## Hardware-derived register shape milestone: Phase G

Machine: Xeon Gold 6244, Cascade Lake, 2x8 cores, `Sys.CPU_NAME == "cascadelake"`,
L1d 32 KiB/core 8-way (shared with the SMT sibling only), L2 1 MiB/core 16-way
(likewise), L3 24.75 MiB 11-way shared across 8 cores, hostname `ccqlin038`.
Julia 1.12.6. Date 2026-09-11. Base revision `3e712ea`. One machine class only
— every caveat in Phase E's "known limitations" still applies.

### Amendment to "Block-size policy (settled)"

The settled policy (this file, "Block-size policy (settled)") forbade an
analytical or cache-probing model choosing block sizes, citing
`tensorcontract-rs`'s A33 (a runtime analytical cache model lost 14-34% on its
own reference machine and failed on two further machine classes) and A57 (that
model's L2-privacy assumption is false on Apple Silicon). `README.md` also
listed "CPU-dispatch tables" as not implemented.

**What survives, unchanged:**

* No analytical or probing model ever chooses a number at runtime. Nothing in
  this milestone computes a block size from a cache size. `src/target.jl` now
  *detects* cache sizes, ways and sharing, and `default_blocking` deliberately
  does not consult any of it — see "Why cache geometry is still not used".
* No probing at package load or at first call. `__init__` does one dictionary
  lookup and reads sysfs; it runs no benchmark.
* Hardcoded measured constants remain the mechanism, and remain the fallback.
* A33 and A57 are not contradicted. Where cache *sharing* would have mattered,
  it is read (`shared_cpu_list` on Linux, `hw.perflevel0.cpusperl2` on macOS —
  literally the A57 datum) rather than assumed. It is currently read for
  reporting only.

**What changes:**

* The register shape `(MR, NR, W)` becomes hardware-derived. This was outside
  the settled decision's scope: that text is entirely about `mc`/`kc`/`nc`, and
  says nothing about register blocking. `MR=8, NR=6, W=4` had never been
  derived and had never been swept by any benchmark in this project.
* `default_blocking` gains one measured row keyed on the detected vector ISA,
  because `kc` had to be re-measured at the new register shape (below).
* `README.md`'s "CPU-dispatch tables" exclusion is dropped. What ships is a
  single capability-derived rule plus one measured row, not a per-uarch table.

### The old default was BLIS's AVX2 shape, on an AVX-512 machine

`nm` on the `blis_jll` 2.0.0+2 artifact shows both `bli_dgemm_haswell_asm_6x8`
and `bli_dgemm_haswell_asm_8x6`. So `MR=8, NR=6, W=4` was never arbitrary — it
is exactly BLIS's `haswell` (AVX2) dgemm register shape. It was simply the
wrong ISA's shape for this machine, which is AVX-512 with 32 vector registers.

BLIS's own register blocking, harvested from the artifact's shipped config
registry (`share/blis/config/<arch>/bli_kernel_defs_<arch>.h`, 28
architectures), for reference: `skx` MR_d=16 NR_d=14, MR_s=32 NR_s=12;
`haswell`/`zen`..`zen3` MR_d=6 NR_d=8, MR_s=6 NR_s=16; `firestorm` (Apple M1)
MR_d=6 NR_d=8; `a64fx` MR_d=16 NR_d=10.

### The rule that ships

    W  = vector_bytes / sizeof(T)     # one full vector register
    MR = 2 * W                        # two A vectors per output column
    NR = 6                            # six output columns

so `NV = (MR/W)*NR = 12` accumulators wherever the rule applies.

`benchmark/bench_kernel_shape.jl` swept 11 Float64 and 8 Float32 candidate
shapes, each crossed with three `kc` values, over `MAIN_SHAPES` plus three new
small-extent shapes plus the scattered 3-index fixture, 9 reps, median, with
the start/middle/end canary bracket. **This rule reproduces the swept optimum
for both dtypes independently** — Float64 `(16, 6, 8)`, Float32 `(32, 6, 16)` —
and reduces to the previous hardcoded `(8,6,4)`/`(8,6,8)` on AVX2, so an AVX2
machine is unchanged. `:unknown` resolves to the legacy shape, so an
unrecognized CPU is bit-identical to the old behavior.

**The rule applies only to the ISAs it was validated on** (`:avx512`,
`:avx2`); `_rule_applies` gates it. `:neon` is detected -- it feeds
`cache_topology` and is there for a future measurement -- but deliberately
takes the legacy shape, because there is no aarch64 measurement and the rule
would pick `MR = 2W = 4` on 128-bit lanes: 12 of 32 NEON registers, narrower
*and* smaller than the legacy `(8,6,4)`, with nothing to justify it. Derive
where measured, fall back everywhere else.

Caught by CI, not by local testing: an earlier revision did derive for
`:neon`, and `test_driver.jl`'s "SIMDKernel is the engine-wide default"
assertion failed on both macOS runners. That assertion pinned the literal
`SIMDKernel{8,6,T}` and had been passing on x86 only *by accident* -- its
fixture has `Qm = 9`, which is below the derived `MR = 16` and so triggers the
demotion. It now pins the resolution (`_default_kernel(T, Qm, Qn)`) rather
than a literal shape, which is machine-independent by construction. Worth
recording as a pattern: making a constant hardware-derived silently converts
every test that asserted its old value into a platform-dependent test.

`kc` was swept jointly with the register shape and not held fixed, because
`MR*kc*sizeof(T)` is the A-micropanel L1 footprint: the old Float64 point
`MR=8, kc=128` is exactly 8 KiB, a quarter of this machine's L1d, so doubling
`MR` at fixed `kc` doubles that footprint and would measure the wrong thing.
Both dtypes preferred twice the old `kc`.

Two different comparisons, both reported because they answer different
questions and it is easy to conflate them.

**(a) Register shape alone**, `mc`/`nc` pinned identically on both sides,
`kc` at each shape's own best (`benchmark/bench_kernel_shape.jl`, 9 reps):

| shape | Float64 | Float32 |
| --- | --- | --- |
| 64^3 | 1.345 | 1.384 |
| 128^3 | 1.494 | 1.536 |
| 256^3 | 1.491 | 1.655 |
| 512^3 | 1.475 | 1.803 |
| shallowK 256x24x256 | 1.342 | 1.374 |
| smallN 256x256x12 | 1.165 | 1.437 |
| smallM 12x256x256 | 1.200 | **0.885** |
| smallMN 16x256x16 | 1.119 | **0.868** |
| scattered a64k64b16n64 | 1.393 | 1.153 |

The two Float32 regressions are the `MR=32` padding cases and are what
motivated the extent-aware demotion below; this table is measured with the
shape forced, so the demotion is deliberately not in effect here.

**(b) The complete shipped configuration** -- derived shape plus its
ISA-keyed blocking, against the complete previously shipped configuration
(the `(8,6)` shape plus the Phase E constants). Measured through the real
default path, so the demotion *is* in effect
(`benchmark/bench_default_vs_legacy.jl`, 21 reps):

| shape | Float64 | shape used | Float32 | shape used |
| --- | --- | --- | --- | --- |
| 64^3 | 1.148 | 16x6/W8 | 1.264 | 32x6/W16 |
| 128^3 | 1.320 | 16x6/W8 | 1.445 | 32x6/W16 |
| 256^3 | 1.636 | 16x6/W8 | 1.492 | 32x6/W16 |
| 512^3 | 1.781 | 16x6/W8 | 1.909 | 32x6/W16 |
| shallowK 256x24x256 | 1.219 | 16x6/W8 | 1.423 | 32x6/W16 |
| smallN 256x256x12 | 1.204 | 16x6/W8 | 1.052 | 32x6/W16 |
| smallM 12x256x256 | 1.088 | 8x6/W4 (demoted) | 1.027 | 8x6/W8 (demoted) |
| smallMN 16x256x16 | **0.963** | 16x6/W8 | 0.996 | 8x6/W8 (demoted) |
| scattered a64k64b16n64 | 1.209 | 16x6/W8 | 1.109 | 32x6/W16 |
| **geomean** | **1.264** | | **1.274** | |

The single sub-unity point, Float64 `smallMN` at 0.963, is a 3.7% shortfall
against a canary spread of 4-12% in this session (the machine was not
exclusive), so it is at or below the noise floor rather than a measured
regression. `Qm = 16` equals `MR = 16` there, so the demotion correctly does
not fire.

An earlier pass of (b) at 11 reps reported 0.840 and 0.890 at the two
small-`N`/`MN` Float64 points. Those did not reproduce at 21 reps with the
configurations measured adjacently, and a separate four-way comparison
(legacy / shape-only / two blockings, 15 reps) put every Float64 small-extent
point at 1.10-1.35. Recorded because it is exactly the kind of shortfall this
project's canary discipline exists to catch, and it was caught.

### Julia 1.10 (LTS): no regression, no version-conditional shape needed

The standing concern was that `NV` scaling would worsen the recorded Julia
1.10 gap ("the `NTuple{NV,Vec{W,T}}` accumulator is not kept register-resident
by the older compiler"). Measured on Julia 1.10.11, 256^3, single-threaded:

| dtype | legacy NV | legacy alloc / time | new NV | new alloc / time | speedup |
| --- | --- | --- | --- | --- | --- |
| Float64 | 12 | 0 B / 1.086 ms | 12 | 0 B / 0.605 ms | 1.794 |
| Float32 | 6 | 0 B / 0.581 ms | 12 | 0 B / 0.311 ms | 1.865 |

Float64's `NV` is unchanged at 12 -- `(8,6,4)` and `(16,6,8)` have the same
accumulator count, only the lane width differs -- and Float32's doubles from 6
to 12, still well inside the limit. Both stay allocation-free on 1.10 through
the driver, and the speedup there is *larger* than on 1.12. So no
`VERSION`-conditional shape is needed and the `julia = "1.10"` compat floor is
unaffected. This is also why holding `NV` at 12 is cheap insurance rather than
a sacrifice.

### Extent-aware demotion, and why it is not the refuted depth-adaptive MC

The two Float32 regressions are `MR=32` padding against `Qm = 12` and `16`:
every micro-tile's row block is then mostly padding. `_default_kernel(T, Qm,
Qn)` falls back to the legacy shape when `Qm < MR`, which removes them.

This keys on `cld(Qm,MR)*MR/Qm` — padding waste, a deterministic countable
quantity known at plan time — and not on a cache-residency estimate. That is
what distinguishes it from the depth-adaptive MC that `docs/refuted.md` records
as failed. It is applied only when the caller did not name a kernel.

To make it possible, `plan_contract`'s `kernel` keyword now defaults to
`nothing` and is resolved after the M/N/K groups are built (it needs `Qm`).
Nothing between the eltype checks and that point reads `kernel`. The body then
tail-calls a `_plan_contract` function barrier that specializes on the concrete
kernel type, so the small `Union` over the closed shape set dies there and
`ContractPlan`'s `Kern` parameter stays concrete: one dynamic dispatch per
`plan_contract` call, zero per micro-tile or K step. One user-visible
consequence: a kernel-scalartype mismatch is now reported after a label error
rather than before.

### Why cache geometry is still not used, despite now being detected

Re-running Phase E's 36-point `(mc,kc,nc)` grid on 2026-09-11 found the whole
grid spans only 9% (Float64, 1.0091 to 1.0990) and 11% (Float32, 1.0124 to
1.1201) between its best and worst points. A model's available upside here is
therefore a few percent, against the tens of percent A33 records such models
losing. `cache_topology()` is exposed for reporting and used for nothing that
picks a number at runtime.

Two incidental findings from that re-run, both worth recording:

* Float32's chosen default `(96, 384, 1152)` reproduced exactly. Float64's
  reproduced `mc` and `kc` and moved `nc` one grid step (`768` to `1536`); the
  two points are 3.3% apart in geomean, inside this project's 6% convention.
* **Phase E's shipped Float64 default `(64, 128, 768)` is the worst of all 36
  grid points** (geomean 1.0990 against the best 1.0091). It was selected by
  the "smallest footprint within 6% of best" tiebreak, not on speed, and it
  sits at the `kc` value Phase E's own text identified as consistently worst.
  This is why `kc` was given the most freedom this milestone.

### NV is held at 12 deliberately — the real ceiling is panel addressing

A larger register tile makes the *isolated* microkernel substantially faster.
Measured with the real `SIMDKernel` and `Base.accumulate`, Float64, `kc=256`,
against this machine's ~115 GFLOP/s peak:

| (MR,NR,W) | NV | packed panel as `view` | as plain `Vector` |
| --- | --- | --- | --- |
| (8,6,4) | 12 | 54.0 | 55.9 |
| (16,8,8) | 16 | 77.6 | 78.3 |
| (16,12,8) | 24 | 23.7 | 93.4 |
| (16,14,8) | 28 | 25.8 | **101.8** |
| (32,6,8) | 24 | 26.9 | **102.6** |

`src/driver.jl` hands each micro-tile `view(ws.packed_a, _sliver_range(...))`.
That costs nothing up to `NV = 16` and about **4x above it**: the view's
address arithmetic stops the `NTuple{NV,Vec{W,T}}` accumulator from staying
register-resident. Confirmed independently by profiling `(16,14,8)` through the
driver — 388 of 560 self samples land on `src/kernels/simd.jl:147`, the K-loop
line itself, i.e. the accumulator round-trip, and none do for `(16,8,8)` — and
by a temporary driver patch that copies each sliver into a plain `Vector`
before the kernel call, which took `(16,14,8)` at 256^3 from 22.2 to 37.4
GFLOP/s *while paying the copy*.

**This contradicts `STATUS.md`'s standing hypothesis that "the microkernel gap
is not closing with size".** At `(16,14,8)` the pure-Julia SIMD microkernel
reaches ~88% of machine peak. The microkernel is not the bottleneck; how the
driver addresses packed panels is. Fixing it is worth an estimated further ~1.9x
on top of what shipped here, and is the next milestone's headline item.

It was not attempted now because `packed_a`/`packed_b` are the
allocator-routed temporaries, under a documented "never `resize!` an allocator
temporary" and reverse-order-`release!` contract (Amendment 1), so giving each
sliver its own plain `Vector` means reworking `ContractWorkspace` and the
allocator path.

Approaches tried for that fix and rejected, with numbers (Float64, `kc=256`,
`(16,14,8)` unless noted):

* **StaticArrays `MVector{NV,Vec{W,T}}` accumulator, mutated in place: 2.2
  GFLOP/s**, against 61 for the `NTuple`. A 25x collapse, reproduced at NV=12
  and NV=16 too. LLVM does not promote an `MVector` whose element type is not
  a primitive to registers, so the indexed stores stay in memory. A mutable
  static accumulator is a dead end here, which is worth recording because it
  is the obvious thing to try.
* A `PanelRef <: AbstractVector` wrapper carrying parent + `Int` offset: 25.4
  GFLOP/s — no better than the view.
* Plumbing a runtime `Int` offset through the generated K step: 27.0 GFLOP/s.
  It did help `(32,6,8)` (98.9) but not the large-`NR` shapes.

Only handing the kernel a genuine plain `Vector` recovers full throughput.

### BLIS microkernels are not reachable from `blis_jll`: dropped

The milestone also evaluated calling BLIS's assembly microkernels directly on
QuasiStrided's packed panels. The packed formats are bit-for-bit BLIS
micropanels already (`i + MR*p`, `j + NR*p`), and `_pack_panel!` already
zero-pads edge panels, which is BLIS's requirement — so the integration would
have been small. It is nonetheless **not possible** against `blis_jll`:

* The 22 assembly microkernels (including `bli_dgemm_skx_asm_16x14`) are
  present only as **local** symbols (`t` in `nm -a`, absent from `.dynsym`);
  `Libdl.dlsym` returns `NULL`. Verified on `blis_jll` 0.9.0, 1.0.0 and
  2.0.0+2 — all three: 22 local asm kernels, 0 dynamically linkable.
* `bli_cntx_get_ukr_dt` and `bli_cntx_get_blksz_def_dt` are likewise
  local-only, so neither the function-pointer route nor the named-symbol route
  works, and BLIS's tuned `MC`/`KC`/`NC` cannot be queried at runtime either.
* `bli_cntx_print` *is* exported but aborts (`SIGABRT`, "Requested index is out
  of bounds") when passed `bli_gks_query_cntx()`'s pointer.
* Only the BLAS-level `bli_?gemm`/`bli_?gemm_ex` are reachable, which is the
  wrong granularity.

Consequently no `[weakdeps] blis_jll` was added and the package remains
genuinely zero-external-dependency. Useful things BLIS still provided: its
shipped config registry as an offline oracle for register shapes (above), and
the confirmation that the old default was its `haswell` shape. `bli_dgemm_ex`
remains available as a benchmark reference line if wanted.

Recorded for anyone revisiting this: it would become possible if `blis_jll`
were built with default symbol visibility, or if the microkernels were reached
through a BLIS "sandbox" build. Neither is in this package's control.

## Panel addressing milestone: Phase H

Machine, Julia version and single-machine caveats as in Phase G. Date
2026-09-11. **Measurement caveat specific to this phase:** the machine was
*not* exclusive for part of it — two unrelated Julia processes from another
project held two cores — and the canary spread reached 15.5% on the final
sweep against Phase E's 0.71%. Several conclusions below are explicitly
"inside noise" for that reason, and are recorded as such rather than resolved.

Motivated by reading Octavian.jl, which solves the problem Phase G left open.

### `view` was the register-tile ceiling; a borrowed pointer removes it

Phase G recorded that the driver's `view(ws.packed_a, _sliver_range(...))`
costs ~4x above `NV = 16`, and left it as the next milestone's headline item.
Octavian.jl never forms a `SubArray` at all: its data path is
`AbstractStridedPointer` throughout and it sub-addresses panels by *pointer
bumping* (`A = gesp(A, (msize, Zero()))`), reconstructing a pointer wrapper
only inside the `let` that `@turbo` consumes. It also uses `Int32` loop
counters to cut register pressure and `offsetprecalc(B, Val{(9,9)}())` to
precompute access-pattern offsets.

Measured here, Float64, `kc = 256`, real `SIMDKernel`, GFLOP/s (machine peak
~115):

| (MR,NR,W) | NV | `view` | `Vector` | raw `Ptr` |
| --- | --- | --- | --- | --- |
| (8,6,4) | 12 | 47.0 | 47.2 | **56.5** |
| (16,6,8) | 12 | 62.6 | 63.5 | **79.5** |
| (16,8,8) | 16 | 71.0 | 62.9 | **78.2** |
| (16,14,8) | 28 | 30.7 | 73.0 | **96.6** |
| (32,6,8) | 24 | 30.1 | 77.1 | **100.1** |

A raw pointer removes the cliff *and* beats a plain `Vector` at every shape,
because the base address is loop-invariant by construction. Critically this
needs **no new dependency**: `SIMD.vload` already accepts a `Ptr{T}`.

Shipped as `PackedPanel` (`src/panel.jl`): a borrowed `(ptr, len)` pair, with
`panel_vload`/`panel_load`/`panel_store!` accessors that also have
`AbstractVector` methods, so every existing caller and the whole test suite
keeps working with `Vector`s and `view`s. `len` is carried only so
`execute_tile!`'s capacity checks keep working. `execute!` wraps the loop nest
in one `GC.@preserve ws`, and the nest moved into `_execute_nest!` so that
preserve has a single obvious scope.

Approaches tried and rejected, with numbers (Float64, `kc=256`, `(16,14,8)`):
StaticArrays `MVector{NV,Vec{W,T}}` mutated in place, **2.2 GFLOP/s** against
61 for the `NTuple` — a 25x collapse, reproduced at NV=12/16/28, because LLVM
does not promote an `MVector` of a non-primitive element type to registers; a
`PanelRef <: AbstractVector` wrapper carrying parent + offset, 25.4; a runtime
`Int` offset plumbed through the generated step, 27.0. Only a genuine raw
pointer works.

### The scattered axis was boxing, and that was the *real* MR ceiling

With panels in place the first re-sweep made `NV = 24-28` the clear winner
(Float64 `(32,6,8)` at geomean 1.0058 against the shipped `(16,6,8)` at
1.1829). But the allocation column showed every shape with `MR > 16`
allocating — 24576 B per `execute!` on the 3-index scattered fixture, 0 B at
`MR = 16`.

Allocation profiling named it exactly: `ScatterAxis{SubArray{Int64,1,...}}`
boxes, 64 B each, attributed to the packing call site.
`Union{AffineAxis,ScatterAxis}` is **not an isbits union** — `ScatterAxis`
holds an `AbstractVector` — so whenever Julia cannot union-split that union
(which depends on callee size, hence on `MR`), the scattered arm is
heap-boxed. The Phase A barrier methods are still correct and still there;
they cannot help with this, because the cost is in the union's
representation, not in a partially-applied `QSTile`.

Two things worth recording about how this was found. First, **a Float32
default shipped in Phase G allocated 24576 B per call on scattered
contractions** and the suite did not catch it: the existing allocation
assertions all use plain, regular contractions, and this only manifests on an
irregular destination. `test/test_target.jl` now asserts zero allocation
through a permuted-A / negative-stride-B / sliced-C fixture at the shipped
defaults — the case this engine exists for. Second, the initial diagnosis was
wrong: `_acc_lane`'s dynamic tuple indexing in `store_tile!` looked like the
obvious culprit and was fixed first, with no effect on the allocation. The
generated, statically-indexed scattered store was kept anyway (it is a
genuine improvement to that path), but the finding is that guessing cost a
round trip and the allocation profiler settled it in one command.

Fixed by `PtrScatterAxis` (`src/tiles.jl`): the same borrowed-pointer idea as
`PackedPanel`, holding `Ptr{Int} + count` instead of a vector, so it is
`isbits` and `Union{AffineAxis,PtrScatterAxis}` lives in a tagged stack slot.
`Base.isbitsunion` confirms it. `ScatterAxis` is unchanged and remains the
vector-backed, bounds-checkable form for the public surface and the tests;
only `_axis_of` switched. Result: **0 B on every shape measured, up to
`MR = 48`**, verified against the independent `ScalarKernel`/
`execute_tilewise!` oracle at exact equality.

### Outcome: the register shape is now a plateau, and the rule already wins

Re-sweeping after both fixes, with all shapes allocation-free, put everything
from `NV = 12` to `NV = 28` within ~4% geomean — against a 15.5% canary
spread. Three successive sweeps produced three different "winners", which is
the signature of fitting noise. So **no override row was added**, and
`_shape_override` ships deliberately empty with that reasoning attached.

The derivation rule from Phase G (`MR = 2W`, `NR = 6`, `NV = 12`) is what
ships, and the final sweep says it is already at the optimum: its answer
ranked **1st of 24 for Float32** (1.0521) and **2nd of 33 for Float64**
(1.0483 against 1.0482 for the best). `NV = 12` also remains the only setting
that fits a 16-register AVX2 machine and Julia 1.10's weaker register
allocation, so it is kept on both counts.

End-to-end geomean against the pre-Phase-G configuration is 1.283 (Float64) /
~1.3 (Float32) — statistically unchanged from Phase G's 1.264/1.274. **These
two fixes bought no end-to-end throughput at the shipped shape**, and that is
the honest result. What they bought is: a real allocation bug fixed on this
engine's core workload, and the removal of two ceilings — the register tile
can now be enlarged, and irregular destinations no longer box — so neither is
a constraint on future work.

### Where the remaining gap actually is

Benchmarked single-threaded on ccqlin038 against Octavian.jl 0.3.29 (pure
Julia, LoopVectorization-based) and OpenBLAS, plain matmul, GFLOP/s:

| shape | QuasiStrided | Octavian | OpenBLAS |
| --- | --- | --- | --- |
| 64^3 | 24.9 | **95.5** | 92.2 |
| 256^3 | 49.7 | **96.6** | 71.3 |
| 512^3 | 53.9 | **89.1** | 80.0 |
| 1024^3 | 50.5 | 79.3 | **88.0** |
| 256x24x256 | 18.9 | **89.7** | 62.7 |
| 256x256x12 | 16.0 | **87.2** | 77.5 |

Octavian is essentially *flat* in size — 95.5 GFLOP/s already at 64^3 — and
beats OpenBLAS at most points. QuasiStrided ramps from 25 to 54. Since the
microkernel reaches 79-100 GFLOP/s in isolation, the deficit is **packing and
per-call overhead**, not the kernel. Octavian's answer is a three-tier
dispatch QuasiStrided has no equivalent of: `maybeinline` (statically small →
fully inlined, no packing), `dontpack`/`nᵣ ≥ N` → `loopmul!` (no packing at
all, `@turbo` straight over the unpacked arrays), and only otherwise pack A,
or pack A and B.

But on this engine's *actual* target — genuine multi-index contractions — the
comparison inverts. Against TBLIS (Matthews' own C++ BSMTC implementation,
verified single-threaded), `C[a,n,b] = A[a,k,b] * B[k,n]`, GFLOP/s:

| dims (a,k,b,n) | QuasiStrided | TBLIS | StridedBLAS |
| --- | --- | --- | --- |
| (64,64,16,64) | 29.2 | **32.2** | 12.0 |
| (128,128,32,128) | **43.2** | 35.1 | 24.7 |
| (256,128,8,256) | **49.2** | 46.8 | 30.0 |
| (64,256,64,64) | **40.8** | 35.5 | 23.4 |
| (256,256,4,256) | **52.4** | 48.0 | 53.8 |

QuasiStrided already matches or beats TBLIS on 4 of 5 points and beats
`StridedBLAS` by 1.0x-2.4x. The plain-matmul gap against Octavian is real but
it is not this package's workload, and it should not drive the roadmap ahead
of the packing/overhead work that the table above actually indicts.

### Dependency note

`LoopVectorization.jl` was considered and rejected as a dependency. Its README
states maintenance runs "through the SciML Small Grants program", with a grant
specifically for Julia v1.12 support — a large, compiler-fragile package on
grant funding. It is also unnecessary: the valuable idea was the addressing
layer, and that was reproduced with `SIMD.vload(Vec{W,T}, ::Ptr{T})` and zero
new dependencies. Noted for a future maintainer: `CPUSummary.jl` is a
better-tested replacement for `src/target.jl`'s cache detection (it already
reports cache inclusivity, and divides L3 by the number of sharing cores),
and `LayoutPointers`/`StrideArraysCore` provide `gesp`/`PtrArray`, if a
dependency ever becomes acceptable.

## Complex element-type milestone: Phase A direction freeze

Opened 2026-09-14 on branch/worktree `complex`, base `114e594`. Goal: support
`ComplexF32`/`ComplexF64` end to end -- engine and `QuasiStridedBackend` -- with
two switchable microkernel methods, and discharge the conjugation invariant
frozen in "Eligibility predicate, and the conjugation invariant" above.

Everything in this section is frozen before any implementation worker launches.
Amendments follow the house style: a numbered `### Amendment N` naming what it
amends, with the original text left standing.

### Direction, and where it comes from

The design is not invented here. `tensorcontract-rs`
(`/mnt/home/ldevos/Projects/tensorcontract-rs/main`) was built to measure exactly
these methods on this class of workload and keeps a candid refutation record.
What it establishes, and what is therefore binding on this milestone:

- **Planar (split-complex, BLIS "1r") is the default.** Both panels hold
  `[re_0..re_{n-1} | im_0..im_{n-1}]` per K step, driven by a genuinely complex
  microkernel: four real FMAs per (A-vector, B-scalar) pair on data already in
  the right lanes. No shuffles, no `fmaddsub`, no duplicated lanes.
- **1m is nearly free in Julia.** Van Zee's induced method is literally the
  *existing real kernel* over `2*kc` real steps, fed by "1e"-packed A and
  "1r"-packed B (`crates/tensorcontract/src/kernel/simd.rs:170-177` is one line:
  `real::<MV,NR>(2*kc, a, b, ab)`). B's "1r" is bit-identical to planar's and
  shares the code path (their D14), so 1m's entire marginal cost is on the A
  side.
- **3m is out of scope.** `docs/refuted.md`'s "The complex-method ranking, and
  the memory-bound inversion, as facts about the engine (A44)": last in every
  column on Ice Lake, 0 of 49 cases won, per-case ratios 0.63-0.84 -- uniform,
  not a few bad shapes. The mechanism that was supposed to justify it (the 25%
  flop saving paying with L1-resident panels) did not survive either.
- **The ranking does not transfer between machines.** Four orderings measured:
  Cascade Lake, Ice Lake, portable scalar, and Apple M3 Max -- where 3m wins
  outright at 1.135x planar. Their `docs/results.md:226`: "Ranking anything below
  planar without naming the machine is a mistake this project has made twice."
- **The methods differ in bytes moved per useful flop, not in flop count.**
  Their A7 is refuted: "The three complex methods differ mainly in flop count.
  **Refuted in Phase 3.** They differ mainly in **bytes moved per useful flop**."
  3m does 25% *fewer* FMAs and still loses, because it loads three planes of both
  operands to do it.
- **Complex is not intrinsically disadvantaged.** 4x the flops on 2x the bytes is
  twice the arithmetic intensity, so packing and indexing overheads amortise
  *better*. They measure complex throughput *higher* than real on the same
  shapes, 1.42-1.47x, replicated on two microarchitectures.

Also adopted from that project, as method rather than as result: one kernel body
written once and parameterised (their D17/D24 -- "a comparison between three
methods must not also be a comparison between three hand-tunings"). 1m therefore
reuses the real `_accumulate_step` **verbatim** rather than getting a copy.

### Method ranking does not transfer between machines

**No auto-dispatch rule is derived from any sweep this milestone runs.**
`PlanarMethod` is the unconditional default; `OneMMethod` is selected only by
naming the kernel, exactly as `benchmark/bench_default_vs_legacy.jl` already
names the legacy kernel. Naming the kernel already picks up the matching
blocking through `default_blocking(kernel)`, so no second mechanism is needed.

This is the same decision, for the same reason, as `_shape_override(::Val,
::Type) = nothing` in `src/driver.jl` -- "deliberately empty: ... a row here
would only pin this package to one machine's noise" (Phase H). Every
planar-vs-1m ratio this milestone reports names `ccqlin038`.

### The frozen packed format is preserved, not extended

`src/kernel_descriptor.jl` stays **completely unmodified**: its `T` remains the
real type, its `T === Float32 || T === Float64` guard remains, and
`packed_a_offset`/`packed_b_offset`/`packed_a_length`/`packed_b_length` on
`KernelDescriptor` remain as written. It is the *real-panel* descriptor.

`src/complex_format.jl` (new) introduces a strictly more general offset formula
under new names on a new type, `ComplexKernelDescriptor{MR,NR,T,FA,FB}`:

    p * reg_tile * reals_per_element  +  plane * reg_tile  +  index

**At `reals_per_element == 1, plane == 0` this reduces exactly to the frozen
`i + MR*p`.** The frozen layout is therefore the `RealFormat` instance of a more
general formula, not something that was redefined; every existing caller, test
and docstring is untouched. That sentence is the whole argument for why this is
additive rather than a breaking change to a frozen interface.

Formats and their `reals_per_element`: `RealFormat` 1, `PlanarFormat` 2
(BLIS "1r"), `OneEFormat` 4 (BLIS "1e", the real `2x2` block `[[re,-im],[im,re]]`
stored as two real K steps of `2n`). 3m's `ThreeM` format is deliberately absent.

### Three meanings of `T`, pinned

On the real path the storage element type, the packed-buffer element type and
the `SIMD.Vec` lane type are all the same type, which is why the ambiguity is
currently invisible. Complex splits them three ways, and conflating them is the
main hazard in this milestone:

| notion | accessor | `Float64` | `ComplexF64` planar | `ComplexF64` 1m |
| --- | --- | --- | --- | --- |
| storage element of A/B/C | `scalartype(k)` | `Float64` | `ComplexF64` | `ComplexF64` |
| packed buffer / `Vec` lane | `realtype(k)` | `Float64` | `Float64` | `Float64` |
| logical register tile | `mr(k)`, `nr(k)` | `MR`,`NR` | complex rows/cols | complex rows/cols |
| reals per A sliver per **logical** K step | `packed_a_per_k(k)` | `MR` | `2MR` | `4MR` |
| reals per B sliver per logical K step | `packed_b_per_k(k)` | `NR` | `2NR` | `2NR` |

`packed_a_per_k`/`packed_b_per_k` are the Julia counterparts of that project's
`Ukr::a_per_k`/`b_per_k`, and they are **the only new quantity the driver
reads**. For every real kernel they are identically `mr`/`nr`, so substituting
them at the four `_sliver_panel` call sites is provably the identity -- pinned by
a test rather than argued.

`packed_a_length(kernel, kc)` takes the **logical** (complex) `kc` and returns a
count of **reals**. 1m's internal doubling to `2*kc` real steps is confined to
its own `accumulate` and never appears in a length, an offset, or a driver loop
bound (their D16: "Exposing that to the driver would leak the method into the
loop nest").

### Kernel types

    PlanarKernel{MR,NR,T,W} <: DescriptorKernel{MR,NR,T}
        descriptor::ComplexKernelDescriptor{MR,NR,T,PlanarFormat,PlanarFormat}

    OneMKernel{MR,NR,T,W} <: DescriptorKernel{MR,NR,T}
        descriptor::ComplexKernelDescriptor{MR,NR,T,OneEFormat,PlanarFormat}
        inner::SIMDKernel{2MR,NR,real(T),W}

Both carry a field named `descriptor`, so **`src/kernel.jl` needs no changes at
all**: `mr`/`nr`/`scalartype`/`packed_a_length`/`packed_b_length` forward through
the existing `DescriptorKernel` methods, and `pack_a!`/`pack_b!` forward through
the existing generic whose per-argument `where` bounds are the Phase 2b finding-5
allocation fix. That forwarding generic is the seam, and it already exists.

`packed_a_offset(k::DescriptorKernel, i, p)` on a complex kernel resolves to a
`ComplexKernelDescriptor` method that does not exist, and the resulting
`MethodError` is correct and deliberate: no complex kernel should ever be asked
for a single-plane offset.

### Buffer element type: the `VT` bound relaxes, the arity does not

`ContractWorkspace{T,VT}`'s bound goes from `VT <: AbstractVector{T}` to
`VT <: AbstractVector`, with `eltype(VT) === real(T)` enforced in the inner
constructor. `ContractPlan`'s `VT` follows.

**Amends** the typing discipline recorded under Amendment 1 ("`VT` is a
`where`-bound parameter resolved at construction, never a `Union`- or
`AbstractVector`-typed field"). That requirement still holds: `VT` is still
resolved to a concrete `Vector{Float64}` at construction and the *field* is still
`::VT`. Only the upper bound in the parameter list loosens, and the new inner
invariant is what keeps the loosening from admitting a wrong workspace.

Why not add an `R` parameter: `ContractWorkspace{Float64,Vector{Float64}}` is
spelled literally in six places across `src/` and `test/`, plus a user-facing
error message and three docstrings. With the relaxed bound **every one of those
spellings stays valid and every one of those tests keeps passing unedited** --
`ContractWorkspace{ComplexF64,Vector{Float64}}` is simply the new instance.

Why `T` stays the *storage* type: the backend's workspace pool is a
`Dict{DataType,ContractWorkspace}` keyed by `eltype(C)`. If `T` meant the packed
eltype, a `Float64` and a `ComplexF64` contraction would collide on one pooled
workspace, silently coupling two dtypes' `reserve!` footprints. Planar and 1m
share `VT` and `reserve!` is grow-only, so the key needs no method component --
but `_reuse_workspace`'s fast path gains a guard on
`eltype(ws.packed_a) === realtype(kernel)`, so a pooled `ComplexF32` workspace
cannot serve a `ComplexF64` plan.

Consequences traced and frozen: `_workspace_sizes` needs **no change** (`pa =
packed_a_length(kernel, kc)` already returns reals, and `m_slivers =
cld(blocking.mc, mr)` correctly uses the logical extent); `PackedPanel{T}` needs
**no change** (it is parameterised on the buffer element, so a complex kernel
receives a `PackedPanel{Float64}` and `panel_vload(Vec{W,Float64}, ...)` works
verbatim); and `execute_tilewise!`, the in-tree oracle, works for complex with
**no changes**, which is a large correctness win.

### The planar microkernel: accumulator, body, and two independent cliffs

Accumulator: one flat `NTuple{2NV, Vec{W,R}}` with `NV = (MR÷W)*NR`, real plane
at tuple indices `1:NV` and imaginary at `NV+1:2NV`, preserving the existing
`(v,j) -> v + (MR÷W)*j + 1` convention within each plane. Flat, not nested: it
keeps every signature the same *shape* as the real path, which is the pattern
Phase H proved keeps the accumulator register-resident.

Body, `@generated` and closure-free, with `nai_v = -ai_v` hoisted out of the `j`
loop:

    cr = muladd(nai_v, bi_j, muladd(ar_v, br_j, acc[idx]))
    ci = muladd(ai_v,  br_j, muladd(ar_v, bi_j, acc[NV+idx]))

Julia exposes no `vfnmadd` intrinsic. `cr - ai*bi` is **rejected**: Julia does
not set LLVM's `contract` fast-math flag by default, so it would not fuse -- two
instructions, and different rounding from the other three terms.
`muladd(-ai, bi, cr)` lowers to `llvm.fmuladd` with an `fneg` operand, which
LLVM's x86 backend folds into `vfnmadd213pd`. Hoisting the negation to `nai_v`
(once per `(v,p)`, not per `(v,j,p)`) bounds the worst case if it declines to
fold at `MV` extra `vxorpd` against `4*MV*NR` FMAs -- 1.4% at `(MV,NR) = (2,6)`.
**This is to be verified by `@code_native` and the result recorded**, following
this project's `@code_llvm`-verification convention; it does not ship as an
assumption.

**Cliff A -- architectural register spill** (their `kernel/x86.rs:35-42`; 30-50%
loss, and "the sweep is full of these cliffs"). Live state per K step:

    2*MV*NR accumulators + 2*MV A vectors + 2 B broadcasts <= nregisters

At the reference planar shape `(MV,NR) = (2,6)` on AVX-512 that is
`24 + 4 + 2 = 30 <= 32` -- tight, consistent with their menu putting it first.
On AVX2's 16 ymm even `(1,6)` leaves exactly zero spare, which is why the complex
derived rule applies on `:avx512` only (below).

**Cliff B -- the Phase H heap-allocation cliff**, from dynamic `NTuple` indexing
above NV = 16 (24576 B per `execute!`, measured). Planar at 16x6 is
**NV_total = 24**, so this cliff is live from the first line of code, not
eventually. Therefore every complex accumulate *and* store is `@generated` with
literal tuple indices from the first commit -- **including the lane tail**, which
the real path handles with a runtime-indexed helper (`_acc_lane`) that the
complex path must not use. Precedent that static indexing scales: after Phase H
the real path measured 0 B up to `MR = 48`, i.e. NV up to 36.

These two cliffs are independent and must be kept apart in any analysis: Cliff A
is about the architectural register file, Cliff B about Julia's tuple lowering.

### The fused `store_tile!` is kept; the reference's writeback split is not adopted

That project's kernels overwrite a stack tile and do nothing else --
`alpha`/`beta`/conjugation/scattered write-back/plane recombination all happen
afterwards in `writeback.rs` (their D7). **Not adopted, deliberately.**

Their split exists because `Ukr::func` is a bare `unsafe fn` pointer and
therefore *cannot* be generic over the destination; a stack tile is the only way
one kernel body serves the regular path, the gather path and every edge block.
Julia has no such constraint: `store_tile!` is already specialised per
`(QSTile{S,R,C}, kernel)` by ordinary dispatch and the `@generated` fallback
already handles every edge block, so the motivating benefit is already had.

The cost here is not free. Phase H measured that routing the accumulator through
memory cost ~4x and heap-allocated 24 KB per `execute!`; inserting a tile
store/load round-trip re-introduces exactly the memory hop Phase H removed, on a
`2*MR*NR`-real tile. And `store_tile!`'s existing contract -- `alpha == 0` never
reads `acc`, `beta == 0` never reads old `C`, padding lanes are never read
(pinned by the nonfinite-poisoning tests) -- would have to be re-derived across
the split for no gain.

The unit-stride plane-to-interleave store (their `writeback.rs:174-183` exists
solely so LLVM vectorises that conversion) is a **measurement-gated follow-on**,
not part of the first implementation: they price recombination at
`~c/(4*min(K,KC))` of a tile's compute, negligible for `K >= KC` and material
only at single-digit `K`. The scattered `@generated` path ships first, reusing
the existing generic `_axpby_tile!` unchanged.

### Conjugation: semantics, and where each piece is absorbed

`TO.tensorcontract!`'s contract is
`C = beta*C + alpha*permutedims(contract(opA(A), opB(B)), pAB)`. There are two
*independent* sources of conjugation per input:

| source | applies to | effect on a `Number` element |
| --- | --- | --- |
| `conjA`/`conjB` (TO's flags) | A, B | `conj` if true |
| `StridedView.op` | A, B, **and C** | `conj`/`adjoint` conjugate; `identity`/`transpose` do not |
| `alpha`, `beta` | -- | **not** conjugated |

`alpha`/`beta` need no conjugation: TO applies `conjA` to A's *data* only.

The mechanism behind the frozen warning, stated precisely: **the engine never
goes through `StridedView` indexing at all.** `_plan_contract` takes
`parent(A)`/`offset(A)` and the packing and store paths address the parent
directly through `tile_load`/`tile_store!`, while `StridedViews`' own
`getindex(a, ::ParentIndex)` and `setindex!` are what apply `a.op`. So `op` is
silently dropped on **all three** operands, C included.

The real path is *structurally* immune, which is stronger than the frozen text
claims: `StridedViews` defines `Base.conj(a::StridedView{<:Real}) = a` and
`adjoint(a::StridedView{<:Number,2}) = permutedims(conj(a), (2,1))`, so for a
real element type `op` is always `identity` no matter what wrapper the user
passes. Nothing about the real path can change.

**The combining rule.** TensorOperations' own TBLIS extension encodes
`isconj(A::StridedView{T}, conjA) = T <: Complex && (conjA ⊻ (A.op === conj))`.
Correct, but **not total**: `StridedView(p, sz, st, off, adjoint)` is directly
constructible, and `=== conj` would silently treat it as unconjugated -- exactly
the silent-wrong-answer class this milestone exists to close. Frozen form:

    _op_conjugates(::typeof(identity))  = false
    _op_conjugates(::typeof(conj))      = true
    _op_conjugates(::typeof(transpose)) = false   # elementwise identity on Number
    _op_conjugates(::typeof(adjoint))   = true
    @noinline _op_conjugates(f) = _qs_throw("unsupported StridedView.op $f ...")

    _qs_isconj(v::StridedView{T}, flag::Bool) where {T} =
        (T <: Complex) && (flag ⊻ _op_conjugates(v.op))

The throwing fallback is `@noinline` and unreachable for every `op` that
`StridedViews` itself constructs, so it costs nothing; hard-rejecting an unknown
`op` matches "Hard-reject, never fall back".

**A conjugated output `C` is rejected this milestone.** In-tree precedent, same
rejection for a related reason: TensorOperations' TBLIS extension does
`isconj(SV(C), false) && throw_conj_output(f)`. Supporting it would thread a
conjugation flag as a type parameter through `store_tile!` -> `execute_tile!` ->
`_execute_micro_tile!` -> the nest, doubling specialisations of the *innermost*
code for a case TO's public API cannot even express (there is no `conjC`
parameter), and would require re-deriving the beta-applied-once argument under
`beta_eff = firstpanel ? betaT : one(T)`. The door, explicitly: supporting it
later means conjugating on the `C` read *and* on the final store (their
`writeback.rs:244-257`), plus that re-derivation -- their multi-block argument
re-reads `D` as `C` with `conj_c = conj_d` and is exact "because conjugation is
additive and involutive", which is a different scheme from `beta_eff` and does
not transfer without work.

**Where the flags live.** `plan_contract` gains `conjA::Bool = false`,
`conjB::Bool = false` keywords. They are folded with each view's `.op` *inside*
`plan_contract`, converted to singleton function values, and passed through the
existing `_plan_contract` function barrier, where the small
`Union{typeof(identity),typeof(conj)}` dies exactly as `_default_kernel`'s Union
already does. `ContractPlan` gains `atransform::TA`, `btransform::TB` fields with
their own type parameters.

    ta = _qs_isconj(A, conjA) ? conj : identity
    tb = _qs_isconj(B, conjB) ? conj : identity
    _qs_isconj(C, false) && throw(ArgumentError("... conjugated output view ..."))

Rejected alternatives, recorded so they are not re-proposed: a runtime `Bool`
field (the transform then crosses `_pack_sliver!` as a `Union`, which is
*precisely* Phase 2b finding 5 and its ~80 B/call of dynamic dispatch);
`Val{Bool}` (identical specialisation cost, worse ergonomics, still needs mapping
to a function at the pack site); conjugation as part of the kernel type
(conjugation is per-operand and per-call, not a kernel property -- it would break
`_default_kernel`, `KernelDescriptor`'s role, and the `DataType`-keyed pool).

**Real-path guarantee, and it is directly testable:** `_qs_isconj` returns
`false` unconditionally for real `T`, so `TA === TB === typeof(identity)` always
and **no new `execute!` specialisation exists on the real path**, even when a
caller passes `conjA = true`.

**The plan/view mismatch hazard is made unrepresentable, not detected.** That
project had to add a runtime guard (`Error::ElementOpMismatch`, their D57) after
`plan.run(.., view.conj(), ..)` silently computed the *unconjugated* contraction
and returned `Ok` -- found by writing documentation, not by a test, because their
property suite drove a lower-level entry point that has no views. QuasiStrided
has a strictly stronger position available: `execute!(plan, alpha, beta)` takes
no operands at all (the plan stores `parent`/`offset`), so there is nothing to
re-supply and mismatch. `plan_contract` folding `.op` itself, plus
`_op_conjugates`' throwing fallback, closes the hole structurally. Recorded as a
deliberate divergence, with their failure mode cited, so the reasoning is
auditable rather than accidental.

**Where conjugation is applied: the existing `transform` seam.** `pack_a!` and
`pack_b!` already take a `transform` argument, applied in `_pack_panel!`, already
forwarded with per-argument bound type parameters, and already tested
("transform is never called on padding lanes", via call counting). The driver's
`_pack_sliver!` stops hardcoding `identity` and takes `transform::TF` with **its
own bound type parameter** -- leaving it unbound reintroduces finding 5, as the
comment at `src/kernel.jl:30-33` says in as many words.

Three call sites, and the third is the trap: `_execute_nest!`'s `pack_b!` and
`pack_a!` calls, and **`execute_tilewise!`'s**. If the third is missed, the
*oracle* is silently wrong for conjugated inputs and the disagreement will
present as an engine bug. A test pinning `execute!` against `execute_tilewise!`
**with conjugation set** is therefore mandatory, not optional.

**The `transform` contract is restated format-independently**, because the
literal `convert(T, transform(v))` in `_pack_panel!` cannot survive planar
packing into a `real(T)` buffer:

> `transform` is an elementwise scalar function applied to each loaded **source
> element** before it is committed to the packed buffer, in whatever physical
> format that buffer uses. A packer that splits an element into planes must
> produce a result identical to applying `transform` to the complex value and
> then splitting. Padding lanes are never read and never call `transform`.

That is exactly what "negate the imaginary plane as it is written" implements,
and it keeps one definition of correctness across all formats. The hazard it
forbids, stated so a reviewer can look for it: a packer that reinterprets the
source into real halves and applies `transform` per *half*, where `conj` is a
no-op. Pinned bitwise by `pack_*!(..., conj)` equalling `pack_*!(..., identity)`
on a pre-conjugated source, in every format.

Note that conjugation is genuinely free here: it is a sign flip on a value the
scatter/gather pass has already loaded, in a pass the engine has to make anyway.

### Eligibility

    const _QS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)
    _qs_eltype_ok(C, A, B) =
        eltype(A) === eltype(B) === eltype(C) && eltype(C) ∈ _QS_ELTYPES

`_qs_strided_ok` and `_qs_eligible` are unchanged in structure.

Still rejected, unchanged: mixed element types, `Float16`, non-strided operands,
an aliased output. Also rejected, and newly so: `Complex{Float16}`,
`Complex{Int}`, `Complex{BigFloat}` (not in the tuple); a conjugated output view;
an unrecognised `StridedView.op`.

**Mixed real-A / complex-B is out of scope.** `KernelDescriptor` and
`ComplexKernelDescriptor` each carry one element type; the workspace pool is
keyed by a single `DataType`; `plan_contract` already throws on
`eltype(A) !== eltype(C)`; and TO's `promote_contract` machinery is the right
layer for promotion. TBLIS requires a shared eltype too. Supporting it would mean
either a second scalar type through the kernel or a materialising promotion --
both contrary to "a timing taken with `backend=QuasiStridedBackend()` always
measures this engine".

### Register shape and blocking

`_derived_shape` gets a **new `T <: Complex` method rather than an edit**, so
that "the real path is bit-identical" is a `git diff` fact rather than an
argument about whether `real(Float64) === Float64`. The complex method takes
`W = vector_bytes ÷ sizeof(real(T))` -- 8 for `ComplexF64` on AVX-512, 16 for
`ComplexF32` -- and then **the same rule**, `MR = 2W`, `NR = NR_DEFAULT`.

That is worth stating rather than hiding: their measured planar menu head is
`(MV,NR) = (2,6)`, i.e. a 16x6 complex tile, which is exactly `MR = 2W, NR = 6`
with `W` taken from the real type. The one-line Phase G rule survives the complex
extension; only the `sizeof` argument changes.

`_rule_applies` gains a complex counterpart true on `:avx512` **only**. Their
AVX2 complex shapes are marked explicitly provisional and unmeasured, and
Cliff A leaves AVX2 planar with zero spare registers. Same treatment and same
wording as the `:neon` precedent: derive where measured, fall back everywhere
else.

Shape menus are **seeded from their measured AVX-512 shapes**
(`crates/tensorcontract/src/kernel/x86.rs:182-190`), converted to this project's
`(MR, NR, W)` with `MR` in logical complex rows, at most three per menu so the
compiled specialisation set stays bounded:

    KERNEL_SHAPES_C64_PLANAR = ((16,6,8), (24,3,8), (8,8,8))
    KERNEL_SHAPES_C64_ONEM   = ((12,8,8), (16,6,8), (8,8,8))   # their MV counts REAL rows
    KERNEL_SHAPES_C32_PLANAR = ((32,6,16), (48,3,16), (16,8,16))
    KERNEL_SHAPES_C32_ONEM   = ((24,8,16), (32,6,16), (16,8,16))

Cliff-A check on the planar menu (`2*MV*NR + 2*MV + 2`): 30, 26, 20 -- all within
32, and in their measured order.

**Blocking derives from packed reals, not from `sizeof(T)`.** Their D13: doing it
from element size "would hand 1m double the L2 footprint and quietly rig the
comparison", because 1m's packed A carries four reals per complex element instead
of two. But this project's `default_blocking` is deliberately *measured
constants, not a cache model* -- "Block-size policy (settled)" records that such
models lost 14-34% even on their own reference machine. The two are reconciled by
anchoring to the measured real row and holding the packed **byte** budget fixed:

    a_reals(::PlanarMethod) = 2;  b_reals(::PlanarMethod) = 2
    a_reals(::OneMMethod)   = 4;  b_reals(::OneMMethod)   = 2

    function default_blocking(v::Val, ::Type{T}, m::ComplexMethod) where {T<:Complex}
        base = default_blocking(v, real(T))          # the MEASURED real row
        Blocking(max(1, base.mc ÷ a_reals(m)), base.kc, max(1, base.nc ÷ b_reals(m)))
    end

So AVX-512 `Float64`'s measured `(128, 256, 768)` yields `ComplexF64` planar
`(64, 256, 384)` and 1m `(32, 256, 384)`: **1m's `mc` is exactly half planar's,
derived, never tabulated**, and both methods get the same L2/L3 byte budget, so
the head-to-head is fair by construction rather than by a reviewer noticing. It
also makes the `_legacy_blocking(::Type{ComplexF64})` `MethodError` disappear
without a new hand table. Any swept override is a row in this same 3-argument
table, not a new mechanism.

### Verification contract

Four oracle layers, following this project's existing practice:

1. **Packing.** Per-format conjugation; packed-length ratios (1m's A is exactly
   twice planar's, 1m's B exactly equal -- their D14); and the bitwise pin that
   `pack_*!(..., conj)` equals `pack_*!(..., identity)` on a pre-conjugated
   source. Bitwise is correct here because these are exact; it remains wrong for
   SIMD-vs-scalar.
2. **Layout pin.** The Julia analogue of their `offset_of!` test:
   `reinterpret(Float64, [ComplexF64(1,2)]) == [1.0, 2.0]` -- re then im,
   adjacent, unit stride. Any planar or 1m packer depends on it.
3. **Kernel contract.** A **test-local, independent re-implementation** of each
   packed format plus a test-local tile reader, compared against a scalar complex
   dot product, run for every method x every shape in the menu and at every lane
   width the machine supports (not only the default).
4. **End-to-end randomised**, against the independent macro-nest oracle.
   **Randomise the conjugation/`op` cross-product; do not enumerate it** -- per
   iteration draw `conjA`/`conjB` and draw `opA`/`opB` from
   `(identity, conj, adjoint, transpose)`, under `Random.seed!` so failures
   reproduce. Exhaustive enumeration is reserved for the microsecond-scale matmul
   fixture in `test/test_tensoroperations.jl`, whose `conjA, conjB ∈ (false,true)`
   loop already exists and simply becomes non-trivial.

Plus: a cache-crossing case **per method**, since each method has its own `mc`;
and `execute!` against `execute_tilewise!` **with conjugation set**.

**Tolerance is an acceptance criterion, not a guideline.** Complex gets the
**same tolerance as real at the same precision**. Planar and 1m require no
widening -- only 3m does, and it is out of scope (their D15 gives 3m 100x, and
explains why: its error bound is relative to `|Ar||Br| + |Ai||Bi|` rather than to
the complex magnitudes). **If a complex test needs a looser tolerance than its
real counterpart, that is a bug signal, not a property of complex arithmetic.**

**The real path is proven unchanged five independent ways**: (1) `git diff` shows
additions, not edits, to `_pack_panel!`, `_accumulate_step`,
`store_tile!(::SIMDKernel)`, `_derived_shape` and `_legacy_blocking` -- a
checklist item, not a judgement call; (2) `typeof(plan.atransform) ===
typeof(identity)` for real `T` even with `conjA = true`; (3) the existing
zero-allocation assertions hold unchanged and are duplicated for complex; (4) the
"no union-typed or partially-applied types reach the nest" assertion extends to
the complex path and the new `TF` parameter; (5) a measured regression guard.

`test/test_target.jl`'s register-budget assertion `(MR÷W)*NR + MR÷W <= 32` is
**generalised, not widened**: keep it verbatim and add a method-aware complex
assertion driven by the detected `nregisters` rather than a literal 32, since
planar carries separate real and imaginary accumulator planes.

### Measurement contract

`benchmark/harness.jl`'s `build_plain`/`build_scattered` are already generic in
`T` and need no change. `DTYPES` stays as it is so the real baseline is
byte-identical; complex dtypes are added alongside.

`flops_per_mac(T) = T <: Complex ? 8 : 2`. **8 is the textbook count and is
deliberately not reduced for induced methods** (their `element.rs:100-106`:
charging 3m fewer flops "would flatter it"; the same argument applies to 1m,
which issues fewer real multiplies than the naive four). A **bytes-moved column**
is reported alongside, because their A7 refutation says that is the quantity the
methods actually differ in.

Two harness defects to fix while there, both found while planning:
`benchmark/bench_tensoroperations.jl` duplicates `median_time_s(...; reps = 9)`
instead of including the harness, and 9 is below the standing >= 15 rule; and
`benchmark/bench_kernel_shape.jl`'s `FMA_RE = r"vfmadd"` does **not** match
`vfnmadd`, so the validated spill detector would undercount planar's FMAs.

Sweeps, in dependency order, each at >= 21 reps with compared configurations
timed **adjacently** and the start/middle/end canary bracket -- `ccqlin038` is not
reliably exclusive, canary spreads of 4-15% are normal, and an 11-rep comparison
once invented two regressions that 21 reps erased:

1. **Real-path regression guard**, gating everything else: the shipped tree
   against `114e594`, same machine, same day. Every point inside the 10% band;
   a *systematic one-sided* shift across all points counts as a regression even
   under 10%.
2. Complex register shapes per method, with the (fixed) spill detector -- Cliff A
   is the binding constraint, so the detector matters more than the timing.
3. `kc`, swept **jointly with the shape**, since `packed_a_per_k * kc *
   sizeof(real(T))` is the L1 A-micropanel footprint and complex doubles or
   quadruples it (the same reasoning as Phase G).
4. `mc`/`nc` re-validation against the derived scaling.
5. Planar vs 1m at matched packed footprint, over the main shapes including the
   memory-bound `shallowK_256x24x256`, reporting bytes per flop.
6. `@tensor` head-to-head against `StridedNative()`/`StridedBLAS()`.

**Headline metric: the complex-efficiency ratio**, within the same engine, with
complex charged 8 flops/MAC: `GFLOPs(ComplexF64) / GFLOPs(Float64)` per shape and
as a geomean. `1.0` means complex is treated exactly as well as real. Complex has
twice the arithmetic intensity, so it should amortise overheads *better* -- they
measure 1.42-1.47x. **Acceptance: geomean >= 1.0 for planar.** Below ~0.9 means a
structural overhead -- most likely a packing cost or an accumulator spill -- and
is a **blocking finding for the review phase**, not a result to publish.

### Forward-binding notes (cheap now, expensive to rediscover)

- **Orientation swap**, still deferred: if it lands it must swap
  `atransform`/`btransform` **and** the packed formats along with the operands.
  That is correct precisely because the kernel contract is about row and column
  panels, not about which user tensor they came from (their `driver.rs:332-347`).

  **Correction (2026-09-19, Label-order milestone): landed, see "Label-order
  milestone" below.** The swap that landed does exactly what this note
  required -- `atransform`/`btransform` (and the storage/base fields and K
  maps) move with the operands, not with the M/N role labels -- and is
  additionally guarded to real dtypes only (`T <: Real`), since complex
  kernels have no packed-format seam for this swap to help.
- **Octavian-style no-pack tiers**, this project's standing "Next task": a
  no-pack tier has **no pack-time seam**, so it would have to absorb conjugation
  some other way -- at load, inside the kernel, or by excluding conjugated
  operands from the tier. Decide that before building the tier, not after.

### Explicitly not done this milestone

3m; mixed real/complex operands; writing into a conjugated output `C`; complex
`tensoradd!`/`tensortrace!` (still throw, contraction-only is unchanged); the
unit-stride plane-to-interleave store (measurement-gated); AVX2 and NEON complex
register shapes (fall back, never guessed at); `select_backend` (still unhooked);
threading; GPU; autotuning; K padding.

### File ownership additions (append to the existing table)

| File | Owner | Phase |
| --- | --- | --- |
| `src/complex_format.jl` (new) | main process (freeze), then packing implementer | A, B2 |
| `src/kernels/planar.jl` (new), `test/test_planar_kernel.jl` (new) | planar-kernel implementer | B3 |
| `src/kernels/onem.jl` (new) | 1m implementer | D |
| `src/workspace.jl`, `src/driver.jl`, `src/blocking.jl` | plumbing implementer (**exclusive** -- no other worker edits `src/driver.jl`) | B1 |
| `src/packing.jl` (additive only; `_pack_panel!` untouched) | packing implementer | B2 |
| `src/tensoroperations.jl` | adapter implementer | B4 |
| `test/test_tensoroperations.jl` | adapter-test implementer (authored blind against this section) | B5 |
| `benchmark/bench_complex_shape.jl`, `benchmark/bench_complex_method.jl` (new) | measurement implementer | F |

### Amendment 3: the conjugation invariant is discharged, and the eltype gate widens

Amends "Eligibility predicate, and the conjugation invariant" above. That section
was written as a *precondition*, not a prohibition -- "Adding complex eltype
support in a future milestone is not a matter of widening the eltype check. It
requires handling `op`/`conj` explicitly first." This amendment records that the
precondition has now been met, and by what.

What is **left standing, unedited**:

- The diagnosis paragraph ("correct *only* because the eltype is restricted to
  real `Float32`/`Float64` ...", and the `isconj` quotation). It is a true
  statement about the code as it stood, and it is still exactly why this work was
  necessary.
- The instruction that adding complex support "requires handling `op`/`conj`
  explicitly first" and is "not a one-line change". Now **discharged** rather than
  repealed: it reads as a satisfied obligation. The work it demanded is the
  "Conjugation: semantics, and where each piece is absorbed" subsection above.

What is **amended**:

- **The predicate block.** Superseded by

      const _QS_ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)
      _qs_eltype_ok(C, A, B) =
          eltype(A) === eltype(B) === eltype(C) && eltype(C) ∈ _QS_ELTYPES

  with `_qs_strided_ok` and `_qs_eligible` unchanged in structure.
- **"`src/tensoroperations.jl` must carry this reasoning as a source comment at
  the point where `conjA`/`conjB` are dropped."** They are no longer dropped, so
  there is no such point. Superseded by: the adapter must carry, as source
  comments, (a) the `_qs_isconj` combining rule and why `⊻` is the right
  composition of the flag with `.op`, (b) why `_op_conjugates` is a *total* table
  with a throwing fallback rather than TBLIS's `=== conj` test, and (c) why a
  conjugated output `C` is rejected rather than supported.
- **"The eltype check is the only thing standing between the current code and
  silently wrong results for complex inputs."** No longer true, and that is the
  point of the milestone. The guard is now three things, none of which is the
  eltype check: `_qs_isconj` folding `conjA`/`conjB` with `.op`;
  `_op_conjugates`' throwing fallback, so no `op` can be silently mishandled;
  and the conjugated-`C` rejection.
- **The supporting facts about the engine-level gate** ("`src/kernel_descriptor.jl`
  throws unless `T === Float32 || T === Float64`, and `default_blocking` only has
  measured constants for those two"). `src/kernel_descriptor.jl` is unchanged --
  complex goes through the separate `ComplexKernelDescriptor`, per "The frozen
  packed format is preserved, not extended" above -- and `default_blocking` gains
  a derived complex row rather than a hand-tabulated one.
- **The bullet requiring "a test that pins it (real operands with
  `conjA`/`conjB` set true still give results identical to `StridedNative()`)".**
  **Kept and strengthened** — but *not* textually unchanged, which an earlier
  revision of this amendment wrongly claimed. The testset was renamed ("conj is
  a no-op for real eltype" -> "conj is real conjugation for complex eltype, and
  still a no-op for real"), its loop widened from `eltypes` to `all_eltypes`,
  and a `T <: Complex` branch added. The real-path *assertion* (`Rq ≈ Rn`)
  survives and gained `Rq ≈ A * B`, so the guard is intact and stronger; only
  the claim about its text was wrong. It is now joined by its complex
  counterpart and by the type-level assertion that
  `typeof(plan.atransform) === typeof(identity)` for real `T` even when
  `conjA = true`.

In the closing summary's "Explicitly NOT done" list, the **Complex element types**
bullet is struck and replaced by a pointer to this milestone plus the residual
list in "Explicitly not done this milestone" above: 3m, mixed real/complex,
writing into a conjugated output `C`, and complex `tensoradd!`/`tensortrace!`.

### Second addendum to "Required argument-checking order in the adapter (frozen)"

Recorded here rather than as a numbered Amendment because, like the T11 addendum,
it changes only *where in the sequence* a check runs and adds one step; no
previously-correct outcome changes.

The order gains a **conjugated-output rejection**, and it necessarily runs
**after** the `StridedView` wrap, because it needs `Cv` -- the same reason T11's
aliasing check had to move there:

    eligibility -> argcheck -> dimcheck -> wrap -> aliasing -> conjugated-C rejection

> **Superseded in part — see "Phase C integration findings" below.** The
> placement argued for here (the check in `plan_contract` only, the adapter not
> duplicating it) turned out to be violated by Julia's keyword-argument
> evaluation: the adapter passes `workspace = _qs_task_workspace(...)` as an
> *argument* to `plan_contract`, so a pooled workspace was acquired before
> `plan_contract`'s own rejection could fire. The adapter now performs the
> rejection too. The reasoning below still stands for *why* the engine keeps a
> check; it is no longer the only one.

The check itself lives in **`plan_contract`, not in `_qs_prepare`**.
`plan_contract` and `contract!` are public entry points reachable without the
adapter, and the same silent-wrongness applies to a caller who wraps a complex
array in a conjugated `StridedView` and calls the engine directly. Putting it at
the engine boundary protects both paths and means the adapter does not duplicate
it. `plan_contract` is also where `conjA`/`conjB` are folded with `.op`, so the
rejection sits next to the code whose invariant it protects.

## Complex element-type milestone: Phase C integration findings

Five Phase B workers ran in parallel on disjoint files against the Phase A
freeze. Four of them reported something the freeze got wrong or left
under-determined. Recorded here at integration rather than at close, because
three of the four changed shipped code.

### Cliff A bites at the shipped shape: the freeze's register arithmetic was optimistic

**The freeze says** `2*MV*NR + 2*MV + 2 <= nregisters`, and that planar's
`(MV,NR) = (2,6)` at `24 + 4 + 2 = 30 <= 32` is "tight, consistent with their
menu putting it first". **Measured on ccqlin038 (Julia 1.12.6, `:avx512`, 32
zmm), it does not fit.** Counting `%rsp` stores and reloads in `accumulate`'s
inner loop:

| shape | pressure | stores/K step | reloads |
| --- | --- | --- | --- |
| planar `(24,3,8)` / `(48,3,16)` | 26 | 0 | 0 |
| planar `(8,8,8)` / `(16,8,16)` | 20 | 0 | 0 |
| real `SIMDKernel` `(32,6,8)`, same NV = 24 (control) | 29 | 0 | 0 |
| **planar `(16,6,8)` / `(32,6,16)`** | **30** | **26** | **3** |
| real `SIMDKernel` `(48,6,8)` (control) | 43 | 12 | 12 |
| planar `(16,6,4)`, deliberately over budget | 58 | 96 | 74 |

The transition sits between **29 and 30**, not at 32, and the shape seeded from
the reference project's measured menu is on the wrong side of it. The real
kernel at an *identical* NV = 24 accumulator count is clean, so this is register
pressure and not a coding defect in the planar body.

Mitigating detail, and the reason this is not being treated as a blocking
finding: 23 of the 26 are **store-only** -- accumulator vectors written with no
matching reload, LLVM rematerialising the loop-carried tuple -- so the cost is
store-port bandwidth rather than a load-use dependency chain. Only 3 are genuine
B-broadcast spill/reloads.

**What this is really an instance of.** The freeze's own rule -- "the method
ranking does not transfer between machines" -- turns out to apply to the *shape
menu* as well, and for a reason the freeze did not anticipate: the reference
measured `(2,6)` best on this same microarchitecture, but in Rust, with a
different register allocator. A shape that fits LLVM-via-Rust's allocation need
not fit LLVM-via-Julia's. The menus stay seeded from that project's
measurements, because a measured starting point beats a guessed one, but **their
*order* is now explicitly unvalidated here** and is Phase F's to settle.

**No code changed for this.** The menu order is untouched, the frozen `<= 32`
assertion is untouched, and no throughput claim is made -- spill counts are not
timings, and this project does not rank shapes it has not timed. The measured
table is recorded in `planar_register_pressure`'s docstring in
`test/test_planar_kernel.jl`. Phase F must decide whether `(16,6)`/`(32,6)`
stays at the head of the menu or whether `(24,3)`/`(48,3)` (pressure 26, zero
spills) should lead, **on measured throughput, not on this table.**

### The `vfnmadd` prediction held exactly

The freeze predicted `muladd(-ai, bi, cr)` would fold its `fneg` operand into
`vfnmadd213pd/ps`, and required verification rather than assumption. Verified on
Julia 1.12.6, counting `accumulate`'s inner loop at every menu shape: exactly
`MV*NR` `vfnmadd231` + `3*MV*NR` `vfmadd231` = `4*MV*NR`, with **zero** separate
negations (the single `vxorp` per function is prologue register-zeroing). The
hoist of `nai_v` out of the `j` loop is therefore free, not merely cheap, and
the 1.4% worst case the freeze budgeted for does not arise.

### The adapter's frozen argument-checking order was violated by keyword evaluation

**Found by the blind test author**, which is precisely what that role is for.
The freeze places the conjugated-`C` rejection in `plan_contract`, on the
argument that this protects direct `contract!` callers too. It does -- but both
`TO.tensorcontract!` methods pass `workspace = _qs_task_workspace(eltype(C))`
(or open an allocator checkpoint) as an **argument** to `plan_contract`, and
Julia evaluates arguments before the call. So a rejected call would first
acquire and possibly `reserve!`-grow a pooled workspace: process-visible
mutation on behalf of a call that is about to throw, at a point the frozen order
does not mention at all.

Worse, the blind author's own `@test_throws ArgumentError` for the rejection was
**passing for the wrong reason** -- the kernel-lookup error fired first. A test
that passes for the wrong reason is worse than a missing one.

**Fixed** by performing the rejection in `_qs_prepare` as well, immediately
after the aliasing check, which is where the frozen order puts it. The
duplication with `plan_contract`'s check is deliberate and commented: the engine
keeps its own so neither entry point depends on the other. The frozen order now
reads, for adapter callers:

    eligibility -> argcheck -> dimcheck -> wrap -> aliasing -> conjugated-C rejection

### The complex legacy shape is over the AVX2 register budget

`_rule_applies_complex` correctly refuses to *derive* a complex shape off
`:avx512`, but the legacy fallback would still hand one back -- and `(8,6,4)`
under planar needs 30 registers against AVX2's 16 ymm, i.e. a guaranteed
spill. **Fixed** by gating complex kernel *construction* by ISA as well
(`_complex_default_supported`), so on an unmeasured ISA the engine refuses to
pick a complex kernel and says why, rather than silently shipping a
guaranteed-spilling default. An explicitly named `kernel =` is unaffected;
this governs only what the engine chooses on its own.

This is the same policy as the shape rule and the `:neon` precedent -- derive
where measured, refuse elsewhere -- extended one step further down.

### `_op_conjugates`' throwing fallback is unreachable through `StridedView`

The freeze says the guard against a silently mishandled `op` is "three things",
one of them `_op_conjugates`' throwing fallback. That **overstates what the
fallback contributes**: `StridedViews` bounds its own `F` parameter to
`Union{typeof(identity), typeof(conj), typeof(adjoint), typeof(transpose)}`, so
a fifth `op` cannot be constructed at all and the fallback can only be reached
by calling `_op_conjugates` directly. The guarantee comes from `StridedViews`'
type bound; the fallback is a total table that documents the reasoning and
would catch a future widening of that bound.

Kept as written -- a total table is still the right shape, and it costs nothing
(`@noinline`, unreachable) -- but the claim is corrected here. Two consequences
for test authors, both now pinned: constructing a `StridedView` with an
unsupported `op` raises `TypeError`, not `ArgumentError`; and `_qs_isconj`
short-circuits on `T <: Complex` before consulting the table, so the totality
guarantee is complex-only. The latter is consistent with "the real path cannot
change" but the freeze did not say it.

### Also corrected at integration

- `_qs_task_workspace`'s return assertion was `ContractWorkspace{T, Vector{T}}`,
  which would have **thrown for every complex call** (the packed panels are
  `Vector{real(T)}`). Now `ContractWorkspace{T, Vector{real(T)}}` -- still fully
  concrete, correct on both paths. A latent bug the freeze's `VT`-relaxation
  reasoning implied but did not spell out.
- The frozen signature `default_blocking(v, ::Type{T}, m::ComplexMethod) where
  {T<:Complex}` is **ambiguous** with the `RealMethod` method at
  `(Val, Type{<:Complex}, RealMethod)` -- neither is more specific. Dropping the
  `T <: Complex` bound resolves it (the `RealMethod` method then strictly wins,
  and `real(T) === T` makes the arm total). Behaviour is exactly as frozen; only
  the signature differs.
- `_reuse_workspace`'s packed-eltype guard is currently **unreachable**: the
  `ContractWorkspace` inner constructor forces `eltype(VT) === real(T)` and
  `_plan_contract` checks `scalartype(kernel) === T` first. Kept as
  defence-in-depth; it becomes load-bearing only if the pool key or the `VT`
  invariant changes.
- `execute_tilewise!` needed the third `_pack_sliver!` transform (as the freeze
  warned) but **not** the `packed_a_per_k` treatment: it hands whole `tw_packed_*`
  buffers, already sized by `packed_*_length`, which already count reals.

## Complex element-type milestone: Phase D/E findings, and a correction to Phase C

### Correcting the Phase C planar spill table

The Phase C table above is **partly wrong, and its qualitative conclusion is
wrong in a way that matters.** Phase D built an independent spill detector and
ran it over both methods; the two instruments agree on some rows and not others,
and the disagreement is diagnosable rather than mysterious.

Phase D's detector counts a stack *reload* even when it appears as a folded FMA
memory operand (`vfmadd213pd zmm, zmm, [rbp-N]`), and matches `rbp`-relative
traffic as well as `rsp`-relative. Phase C's counted neither. Evidence that the
newer instrument is the trustworthy one: it **reproduces both of Phase C's real
controls exactly** — real `SIMDKernel (32,6,8)` clean at pressure 29, and real
`(48,6,8)` at 12 stores / 12 reloads at pressure 43.

| shape | Phase C | Phase D | agree? |
| --- | --- | --- | --- |
| real `(32,6,8)` control | 0 / 0 @ 29 | 0 / 0 @ 29 | yes |
| real `(48,6,8)` control | 12 / 12 @ 43 | 12 / 12 @ 43 | yes |
| planar `(24,3,8)` / `(48,3,16)` | 0 / 0 @ 26 | 0 / 0 | yes |
| planar `(16,6,8)` / `(32,6,16)` | 26 / 3 @ 30 | 24 / 6 | ~ (same conclusion) |
| **planar `(8,8,8)` / `(16,8,16)`** | **0 / 0 @ 20** | **15 / 8 @ 20** | **no** |

**What this overturns.** Phase C reported "the transition sits between 29 and
30", which presented spilling as monotone in the pressure number and therefore
as something the frozen budget inequality could predict. It is not monotone:
`(24,3,8)` at pressure **26 is clean** while `(8,8,8)` at pressure **20 spills**.
A single scalar budget cannot order these, so the inequality
`2*MV*NR + 2*MV + 2 <= nregisters` is a **necessary condition at best, not a
predictor** — aspect ratio matters independently of total pressure, presumably
through how LLVM schedules the loop-carried tuple.

Consequences, and what is deliberately *not* being done:

- The frozen `<= 32` assertion stays in `test/test_target.jl`. It is still a
  sound lower bar, and weakening or complicating it on the strength of a
  spill-count reading would be the wrong trade.
- The menu order still stays untouched. Phase C declined to reorder on spill
  counts; Phase D's correction makes that restraint look better, not worse --
  the quantity the menus would have been reordered on turns out to have been
  misread. **Spill counts are not timings.** Phase F ranks on measured
  throughput or not at all.
- Phase G should treat "which spill detector is right" as an open item with a
  concrete, cheap resolution (read the two regexes against one shared `.asm`
  dump), not as a matter of opinion.

The narrower Phase C claims that survive unchanged: the reference-seeded menu
head does spill, the real kernel at an identical NV = 24 does not, and the cost
is mostly store-port traffic rather than a load-use chain.

### Every shipped 1m shape is spill-free

Measured with the Phase D detector: 1m at `(12,8,8)`, `(16,6,8)`, `(8,8,8)` and
`(8,4,4)` is **0 stores / 0 reloads**, at pressures 28, 29, 19, 21, issuing
exactly `MV*NR` FMAs per *real* K step. That is the expected shape of the
result -- 1m holds `MV*NR` accumulators against planar's `2*MV*NR` -- and it is
measured on `kernel.inner`, because the code running 1m's K loop **is** the real
path's `accumulate`, byte for byte. That is the reuse confirmed as an executed
fact rather than an architectural intention.

### Two things the freeze got wrong about `OneMKernel`

1. **The frozen field spelling is not legal Julia.**

       inner::SIMDKernel{2MR, NR, real(T), W}

   cannot be a struct field type: `2MR` and `real(T)` are computations on
   `TypeVar`s (`MethodError: *(::Int, ::TypeVar)`). Fixed with a fifth type
   parameter carrying the computed type, pinned in the inner constructor to
   exactly `SIMDKernel{2MR, NR, real(T), W}`. `OneMKernel{MR,NR,T,W}` still
   works for `isa` and dispatch, and the field stays concrete.

2. **`W` must be even, and the freeze never says so.** The freeze's stated
   requirement is `mod(2MR, W) == 0`, which odd `W` can satisfy -- `MR = 3,
   W = 3` does -- while a complex row straddles a vector boundary, silently
   breaking the `OneM` tile reader's adjacency assumption. Now enforced at
   construction and tested. This is the kind of gap that produces a wrong
   answer rather than an error, so it is recorded rather than quietly fixed.

The adjacency assumption itself was verified rather than assumed, structurally
(`iseven(W)` and `mod(2MR,W) == 0` give `MR == MV*(W÷2)`, so rows partition
cleanly and `2i`, `2i+1` always land in the same vector at adjacent lanes) and
numerically (a ramp accumulator holding `1000j + r` at real row `r`, so any
lane or row mis-assignment is visible in the output).

### 1m on Julia 1.10 is better behaved than planar

Both are allocation-free on 1.12.6 at every shape. On 1.10.11, where the
compiler cannot keep a large `NTuple{NV,Vec}` accumulator register-resident,
`accumulate` allocates for both -- but `execute_tile!` is **0 B for every 1m
shape** against planar's 128 B (`ComplexF64`) / 96 B (`ComplexF32`). Consistent
with 1m holding half planar's accumulator state. Recorded because the project's
convention is to keep the 1.10 gap visible rather than hidden; the
`skip=(VERSION < v"1.11")` markers are on the assertions, not on the knowledge.

### The end-to-end randomized oracle, and what it pins that nothing else did

`test/test_macro_driver.jl` gained the fourth oracle layer: 501 randomized
complex cases (300 `ComplexF64` + 200 `ComplexF32`, spread over every available
(method, shape) combination) against a dense-matmul oracle that uses **its own**
conjugation table and rule, never the engine's `_qs_isconj`/`_op_conjugates`.
The conj/`op` cross-product is **drawn, not enumerated**, under a fixed seed,
with the seed and full case description printed on failure.

Three pins there are worth naming because no other layer provides them:

- **XOR, not `||`.** `conjA = true` on a `conj`-op view must *cancel*, and the
  conjugated answer must be a demonstrably different matrix (so the assertion
  cannot pass vacuously).
- **`adjoint` must conjugate with no flag set.** An implementation written as
  `v.op === conj` -- which is what TensorOperations' own TBLIS extension does --
  fails *only* this case. This is the totality argument of "Phase C integration
  findings" turned into an executable test.
- **`execute!` against `execute_tilewise!` with conjugation forced on**, plus an
  in-loop assertion that at least one operand really is conjugated. The freeze
  warned that a missed third `_pack_sliver!` call site would make the in-tree
  oracle silently wrong; this is what would catch it.

Cache-crossing extents are **derived** from each method's own
`default_blocking` and the crossing asserted against the plan's *effective*
blocking, rather than hardcoded. That matters: 1m's `mc` really is half
planar's, so one shape provably would not have covered both methods.

**Tolerance, measured.** No complex case was granted a looser tolerance than
its real counterpart, and none needed one. Worst-case consumption of the
allowed error budget: real 0.0017 (`Float64`) / 0.0023 (`Float32`); complex
0.012-0.021 across planar and 1m at both precisions. So complex uses 6-9x more
of the budget than real -- expected, since a complex MAC is four real products
plus two adds -- while staying roughly 50x inside it. Had a complex case needed
widening, the freeze's rule is that this is a bug signal; it did not arise.

## Complex element-type milestone: Phase F measurement

All on `ccqlin038` (Xeon Gold 6244, Cascade Lake, `:avx512`), Julia 1.12.6,
single-core, 21 reps, median. **No number here transfers to another
microarchitecture**, and none of it is wired into dispatch beyond the two
`_shape_override` rows named below.

### The register shape: the derived rule was wrong for complex by 38-41%

Phase C measured the reference-seeded planar menu head spilling and declined to
reorder the menu on spill counts, deferring it here. That was the right call and
the deferral resolved cleanly: **the spill analysis predicted the throughput
ranking before the ranking was measured.**

`benchmark/bench_complex_efficiency.jl` arm 2, ranked by geomean of per-shape
time normalised to the best at that shape (1.000 = best), canary spread 0.4%:

| ComplexF64 | | ComplexF32 | |
| --- | --- | --- | --- |
| planar **24x3** | **1.055** | planar **48x3** | **1.104** |
| 1m 16x6 | 1.116 | 1m 16x8 | 1.170 |
| 1m 12x8 | 1.179 | 1m 24x8 | 1.172 |
| 1m 8x8 | 1.224 | 1m 32x6 | 1.202 |
| planar 16x6 *(what the rule derived)* | **1.452** | planar 16x8 | 1.311 |
| planar 8x8 | 1.502 | planar 32x6 *(what the rule derived)* | **1.562** |

The shape the `MR = 2W, NR = 6` rule derives was the **worst planar
configuration measured** -- 38% off the best for `ComplexF64`, 41% for
`ComplexF32`. The spill-free `24x3`/`48x3` shape wins outright, and beats every
1m shape as well.

**Acted on**, via the `_shape_override` hook that has existed since Phase G for
exactly this purpose and has until now been deliberately empty:

    _shape_override(::Val{:avx512}, ::Type{ComplexF64}) = (24, 3, 8)
    _shape_override(::Val{:avx512}, ::Type{ComplexF32}) = (48, 3, 16)

These are the **only swept rows in the package**, and the asymmetry with the
real path is deliberate and worth stating: the real rule landed within noise of
its sweep's best, so a row there would have encoded noise (Phase G's argument,
still standing). Here a row corrects a 38-41% error. The rule itself is
untouched and still shared with the real path; `_complex_rule_shape` was
factored out so the rule can be tested independently of the override. Off
`:avx512`, `_rule_applies_complex` is false and the engine never reaches
either, so no other machine is handed a ccqlin038 constant.

The planar menus were reordered to lead with the swept winner, so that the menu
head and the shape the engine resolves to agree. **The set is unchanged** --
only the order -- so the compiled specialization count does not move, and a
test pins that.

### The headline metric: complex is treated better than real, by a lot

Complex efficiency = one engine's complex GFLOP/s over its own real GFLOP/s at
the same shape, complex charged 8 flops/MAC (the textbook count, not reduced
for 1m). `1.0` means complex is treated exactly as well as real.

| | geomean, derived shape | geomean, swept shape |
| --- | --- | --- |
| `ComplexF64` | 1.256 | **1.829** |
| `ComplexF32` | 1.457 | **1.910** |

Acceptance was "geomean >= 1.0 for planar". Met before the override and
comfortably exceeded after. For scale, the reference project measures 1.42-1.47
on the same class of machine, so 1.83/1.91 is on the high side of the expected
range rather than anomalous -- complex really does amortise this engine's
packing and per-call overheads better than real, because it is 4x the flops on
2x the bytes.

The shape change is visible as much more than a geomean shift. At the derived
shape, efficiency **fell below 1.0 at the large compute-bound sizes** -- 0.870
at 256^3 and 0.827 at 512^3 for `ComplexF64`, 0.838 at 512^3 for `ComplexF32`
-- exactly where the microkernel rather than the overhead is the constraint,
and therefore exactly where a spilling kernel should hurt. After the override
that dip is gone: 1.628 and 1.515, and 1.544. In absolute terms `ComplexF64`
512^3 went 40.5 -> 73.7 GF/s (+82%) and `ComplexF32` 512^3 went 82.1 -> 151.8
GF/s (+85%). Canary spread 3.0% on that run.

The shape of the remaining curve is the physically expected one: efficiency is
highest where overhead dominates (2.59 on `shallowK_256x24x256`, 2.19-2.33 on
`smallN`) and lowest where the kernel dominates (1.32 on `smallM_12x256x256`,
which pads every micro-tile away). Nothing here is a claim about absolute
competitiveness against a tuned vendor library; that comparison is
`bench_tensoroperations.jl`'s and was not re-run this milestone.

### Planar remains the default, and 1m is not promoted

Planar `24x3` beats every 1m shape measured, so the default is unchanged and
`OneMMethod` stays selectable only by naming the kernel. That agrees with the
reference project's own default, but the agreement is a coincidence of this
machine and must not be read as a general result: **1m's best shape (1.116) is
closer to planar's best (1.055) than planar's own worst shape is (1.452)**, so
"planar beats 1m" is a smaller effect here than "pick the right shape". The
reference records four different method orderings on four machines; nothing in
this run is evidence against that.

### The real-path regression guard: no regression, and the resolution is ~5%

`benchmark/bench_real_path_guard.jl` (new) runs the real default path across two
trees -- the working tree and the milestone base `114e594` -- because that
comparison cannot be made in one process, both trees defining a module named
`QuasiStrided`.

**Corrected after the gated review.** An earlier revision of this section quoted
"overall geomean new/base = 0.9883" from four runs per tree. That number is
**retracted**: it does not reproduce. Re-run in ABBA order with both trees
explicitly labelled (`QS_GUARD_LABEL`, added for this reason), the same
comparison gives

    same-tree run-to-run noise:  base geomean 0.985 (range 0.922-1.059)
                                 head geomean 0.937 (range 0.698-1.079)
    between-tree effect, pooled: overall 1.020   (Float64 1.027, Float32 1.014)
                                 slower on 13 of 18, range 0.957-1.089
    per round:                   1.045 and 0.994  -- the sign flips

The between-tree difference (2.0%) is **smaller than the same-tree
run-to-run noise** (up to 6.3% on geomean, 30% on a single point), and the
sign flips both between rounds within a session and between sessions (0.988
then 1.020). So the only claim this instrument supports is:

> **No real-path regression detectable at this measurement's resolution, which
> is roughly 5-6% on geomean.** There is no systematic one-sided shift -- which
> is what a lost specialization would look like -- and the resolved kernel shape
> is identical at every point in both trees.

That is sufficient for the acceptance criterion, which asks for absence of
regression rather than a precise figure. It is not sufficient to claim a
speedup, and the earlier revision should not have quoted one.

Two things the review was right to object to, both now fixed:

- **The artefacts did not identify which tree they measured.** `results_dir()`
  is keyed by host and date, and `git_commit()` returns a human sentence in a
  `git archive`-extracted tree, so all four files carried the same tag and were
  indistinguishable from a same-tree noise run. A reviewer reading
  `benchmark/results/` could not verify the claim -- and was correct not to take
  it on trust. Filenames now carry an explicit label, and both trees' artefacts
  are preserved side by side under the same results directory.
- **The base-tree run lived only in ephemeral scratch.** Note that
  `benchmark/results/` is gitignored by long-standing project convention, so
  *no* benchmark evidence in this package is committed; the fix is that the two
  sides are now co-located and self-identifying on disk, not that they are in
  git. Anyone re-deriving this needs to extract `114e594`, copy in the current
  `benchmark/` directory (the instrument must be shared, only the engine
  differs), and set `QS_GUARD_LABEL`.

Three methodological notes, recorded because each cost time to find:

- **The guard's first ordering was confounded.** Running base-then-new twice
  means any downward drift over wall-clock makes "new" look faster; the drift
  was real (base run 2 came in 2.7% under run 1). Re-run in ABBA order and
  pooled over four runs per tree, which is what the numbers above are.
- **The guard overwrote its own data.** `results_dir()` is keyed by host and
  date, so the second run of the day silently replaced the first -- and did,
  destroying a round before it was noticed. Filenames now carry the commit and
  a run counter, which matters specifically because comparing the *same* tree
  twice is how the noise floor gets established.
- **The canary's own first sample is a warm-up artefact.** `canary[start]` reads
  ~10% faster than `canary[middle]`/`canary[end]` in every run of that script,
  on both trees, while the middle-to-end spread is 0.2-3.4%. So the script
  reported an 11% "canary spread" and tripped its own quietness warning on a
  machine that the six-canary efficiency sweep measured at 0.4%. Judge
  quietness from the middle/end pair; the start canary is a cross-run reference
  only. Documented in the script header.

### Arm 2's ranking is block-sequential, which bounds how finely it can be read

Raised by the gated review and accepted. `bench_complex_efficiency.jl`'s arm 2
loops `for method, for shape: time every case`, so all six configurations for a
given element type run back to back over minutes rather than interleaved. That
is the same class of confound the real-path guard was re-run in ABBA order to
remove, and arm 2 did not get the same treatment.

What bounds it: the canary bracket immediately around each element type's arm-2
window reads 0.4-3.0%, so drift cannot manufacture the 38-41% headline effect,
nor most of the finer ordering. What it does *not* bound: a ~6% gap between
adjacently ranked configurations -- planar `24x3` at 1.055 against 1m `16x6` at
1.116 -- is only about twice the canary-bounded drift. **So the headline result
(the derived shape is the worst planar configuration, and `24x3`/`48x3` is the
best) is load-bearing; the finer ordering between planar's winner and 1m's
winner is not.** Nothing in the shipped code depends on that finer ordering:
planar is the default for reasons the freeze fixed in advance, and no
auto-dispatch rule is derived from any of it.

### Harness defects fixed en route

- `benchmark/bench_kernel_shape.jl`'s `FMA_RE = r"vfmadd"` does **not** match
  `vfnmadd` -- verified against the literal strings. A planar kernel issues
  `MV*NR` negated FMAs per K step out of `4*MV*NR`, so the validated spill
  detector would have undercounted planar's FMAs by a quarter and read the
  "fmas should equal NV" check as a spurious shortfall. Now `r"vfn?madd"`.
- `benchmark/bench_tensoroperations.jl` timed at `reps = 9`, below the standing
  `>= 15` rule, in the script whose output the README quotes. Raised to 15. Its
  duplication of the harness is left in place with the reason narrowed and
  written down: the original justification ("bench_driver.jl is not a library")
  expired when `harness.jl` was factored out, and what keeps it now is that
  every committed number for that script was taken against its own literals.
- Two distinct bytes-per-flop metrics now exist and are documented as distinct,
  because conflating them credits a method for its blocking rather than its
  format: `panel_reals_per_element` (format only -- real 2, planar 4, 1m 6, so
  1m/planar = **1.5x**, reproducing the reference's figure from the kernels' own
  geometry) and `packed_bytes_per_flop` (one macro block at that method's own
  shipped blocking, which comes out near 2x because `default_blocking` halves
  1m's `mc` and the B term dominates at the shipped `nc`).

### Amendment 4: the public tier gains `PlanarKernel` and `OneMKernel`

Amends "Public / internal API split: three tiers", which froze the tier
membership at 1 exported + 9 `public` + 34 internal.

`PlanarKernel` and `OneMKernel` move into the `public` tier, alongside
`ScalarKernel` and `SIMDKernel`. The tier count becomes 1 exported + 11
`public`.

The reason is narrow and is about `OneMKernel` specifically. The freeze's
"Method ranking does not transfer between machines" decision means the engine
**never** selects 1m on its own -- `_default_complex_method` returns
`PlanarMethod()` unconditionally, and no sweep result is allowed to change
that. So the only way any caller can ever use 1m is
`plan_contract(...; kernel = OneMKernel(...))`. A selection mechanism whose
sole handle is an internal name is not a selection mechanism: it would make 1m
either unreachable in practice or reachable only by writing
`QuasiStrided.OneMKernel`, which is precisely the internal-name dependency the
three-tier split exists to prevent.

`PlanarKernel` follows for symmetry and for a second reason: it is what
`_default_kernel` returns for a complex element type, so it appears in the type
of any `ContractPlan` a user inspects, and in the error message when a shape or
ISA is rejected. A name a user is shown should be a name a user may write.

Nothing is exported. The single export remains `QuasiStridedBackend`.

## Complex element-type milestone: Phase G gated review disposition

Two gated passes, both spent; `fable_review_complex_used: true`. Neither may be
relaunched for this milestone.

### Pass 1 (Fable) — conjugation semantics and record integrity

Scoped deliberately narrow, to the one area whose failure mode is a *silently
wrong number* rather than an error. **No blocking numerical finding.** The
reviewer could not make the engine produce a wrong answer through any
combination of `conjA`/`conjB`, the four `op` values, either operand, either
element type, either complex kernel, either driver, the adapter, or `@tensor` --
1352 adapter cases, 512 direct-engine cases (each run through both `execute!`
and `execute_tilewise!`), and 4 macro cases, all against oracles the reviewer
wrote rather than against this suite. Zero failures.

Worth recording because it strengthens the freeze's own argument: **xor is
TensorOperations' semantics, not merely TBLIS's convention.** TO 5.8.0's
`StridedNative` realises `conjA` as `conj(SV(A))`, and StridedViews 0.5.2
realises `conj` on a view by flipping `op` through its `_conj` table
(`identity<->conj`, `adjoint<->transpose`). That is conjugation *parity* by
construction, which is xor. The freeze inferred the rule from TBLIS; it turns
out to be forced by the upstream implementation.

Also independently hand-verified, by an argument worth preserving: the 1e 2x2
block `[[re,-im],[im,re]]` is the real matrix representation `M(z)` of
"multiply by z". Substituting `conj(z) = (re,-im)` yields
`[[re,im],[-im,re]] = M(conj z) = M(z)ᵀ` -- so "negate the imaginary part
before applying the layout" and "apply the layout to `conj(z)`" *coincide*,
which is exactly what the restated `transform` contract requires. Confirmed
numerically: the conj-packed panel is bit-equal to a hand-written layout, and
read back as the real 4x4 matrix the 1m inner kernel sees, times a
planar-packed B, it reproduces `conj(A)*B` exactly and differs from `A*B`.
Padding writes literal `+0.0`, not `-0.0`, in all four 1e reals.

Findings, all fixed:

- **B1 (record).** Amendment 3 claimed the "Complex element types" bullet in
  the closing summary's "Explicitly NOT done" list had been struck. It had not.
  A reader landing there was told complex was blocked and that `op` is ignored
  entirely -- both false of the shipped code. The bullet is now struck in
  place, with the original reasoning retained (struck, not deleted) because it
  was true of the code as it then stood and is why the work was necessary.
- **B2 (record).** Amendment 3 claimed the real-path conjugation pin test
  "stays, textually unchanged". It was renamed, its loop widened, and a
  complex branch added. The real-path *assertion* survives and gained
  `Rq ≈ A * B`, so the guard is intact and stronger -- but the claim about its
  text was false, and the freeze's "git diff shows additions, not edits" proof
  does not hold for that file. Reworded to "kept and strengthened".
- **S1 (source).** `src/tensoroperations.jl` contradicted itself twenty lines
  apart: the pre-Phase-C comment still said "the adapter does not duplicate
  it", while the Phase C comment below said the duplication is deliberate and
  why. Phase C corrected the record but not the comment. Rewritten.
- **S2 (record).** The "Second addendum" stood uncorrected in place while Phase
  C reversed it 70 lines later. Now carries an in-place forward pointer, which
  is how every other superseded section in this file is handled.
- **N3 (docs).** `plan_contract`'s docstring still advertised
  `kernel = SIMDKernel(Val(8), Val(6), eltype(C))` as the default; the real
  default has been `nothing` (resolved after the M/N/K groups exist, so the
  extent-aware demotion can see `Qm`) since Phase G. Corrected, and extended to
  say what it resolves to per element type and that 1m is never automatic.

Accepted without action: on a non-`:avx512` machine an *eligible* complex
adapter call throws from the workspace argument's `_default_kernel` before
`plan_contract` runs -- an `ArgumentError`, never silent, but a step the frozen
order does not mention. It is the same keyword-evaluation ordering quirk Phase C
found, in a case where the outcome is a loud error either way.

### Pass 2 (Sonnet-High) — everything else

**One blocking finding, and it was right**: the real-path regression guard's
quoted geomean was not substantiated by the artefacts on disk. See "The
real-path regression guard" above for the correction -- the number is retracted,
the conclusion narrowed to what the instrument can resolve, and the artefacts
now identify which tree they measured. This is the most valuable finding of
either pass, because it was a claim about *evidence* rather than about code, and
the evidence did not support it.

One should-fix on arm 2's block-sequential ordering, accepted and recorded above.

Independently re-measured and confirmed, with numbers, rather than taken on
trust: **zero allocation in all 24 cells** of (2 methods x 3 shipped shapes x 2
precisions x `accumulate`/`execute_tile!`) on a *scattered* destination -- the
fixture class that hid a real 24 KB regression in this project before. Also
verified by direct reading: the planar arithmetic and its plane indices across
`zero_accumulator`/accumulate/store; 1m's adjacency argument and `(2u+1, 2u+2)`
lane extraction, with no shipped menu entry violating it; that no complex
accumulate or store uses a runtime tuple index; that the relaxed
`ContractWorkspace` bound leaves no field abstract and cannot be bypassed
(immutable struct, single constructor path, pool keyed on `T` so `real(T)` is
fixed); that `_workspace_sizes` genuinely needs no change; that logical `kc` and
real counts are nowhere mixed, including at `OneEFormat` edge slivers; that the
menu reorder left the specialization *set* unchanged (pinned by a test, not
prose); and that all 16 `skip=(VERSION < v"1.11")` markers sit on allocation
assertions only, never shielding a correctness assertion.

The reviewer also confirmed the packing tests' "guard on the guard" is real: a
positive assertion that the conj and identity packings genuinely differ, so the
main bitwise pin is capable of failing.

### The one open item, deliberately left open

"Which spill detector is right" (Phase C's versus Phase D's) is **not settled**,
and no longer needs to be. Phase F ranked the shapes on measured throughput,
which supersedes the spill-count question as its own tie-breaker, and both
detectors agreed on the shape that matters. Recorded so a future reader does not
mistake the disagreement for a live risk: it is a disagreement about an
instrument nothing shipped now depends on.

## Comment/structure cleanup pass (post-milestone)

A readability and de-duplication pass over the whole tree, with no behaviour
change: no new features, no changed defaults, no changed error messages, and
the suite unchanged at 34480/34480. Recorded because it moved material *into*
this file and left pointers behind, which is a change a future reader can
otherwise mistake for lost knowledge.

**The rule applied.** This project keeps `docs/decisions.md` as the
authoritative record of *why* and source comments as pointers to it. The
milestone that just closed wrote a great deal of narrative into the source that
this file already held -- the planar and 1m spill tables, the `vfnmadd`
verification, the `_shape_override` ranking, the conjugation essay, the
real-path guard's how-to-read notes, the two bytes-per-flop metrics. Those
source blocks were condensed to a pointer plus the load-bearing number. Two
pieces of material lived **only** in the source and are transcribed below
before being condensed there.

**Guardrail comments were deliberately kept in place, in the source**, tightened
but never removed: the per-argument bound-type-parameter rule (`src/kernel.jl`,
`_pack_sliver!`), the "do not collapse the barrier methods" warning, the
borrowed-pointer-not-`view` result (`src/panel.jl`), the literal-tuple-index
rule (Cliff B, 24576 B), `muladd(-ai, bi, c)` over `c - ai*bi`, 1m's even-`W`
requirement, `complex_format.jl`'s reduction-to-the-frozen-format argument,
`ContractWorkspace`'s `eltype(VT) === real(T)` invariant, and Amendment 3's
three conjugation comments. A guardrail is one sharp sentence in the source, not
a pointer to this file, because the reader who needs it is editing the line
above it.

### Transcribed from `src/kernels/onem.jl`: why the induced method works

`OneEFormat` A at logical K step `p` occupies `4*MR` reals laid out as two
consecutive *real* K steps of `2*MR`:

    reals   0 .. 2MR-1 :  re_0, im_0, re_1, im_1, ...
    reals 2MR .. 4MR-1 : -im_0, re_0, -im_1, re_1, ...

and `PlanarFormat` B at that step occupies `2*NR` reals as two real K steps of
`NR` (`re_0..re_{NR-1}`, then `im_0..im_{NR-1}`). A real `SIMDKernel{2MR,NR}`
addresses A at `i' + 2MR*p'` and B at `j + NR*p'`, which walks both buffers
linearly -- so it reads exactly those blocks, with real step `p' = 2p` the first
and `p' = 2p+1` the second. The real product it computes is therefore

    Ar[2t,   2p] =  re(A[t,p])   Ar[2t,   2p+1] = -im(A[t,p])
    Ar[2t+1, 2p] =  im(A[t,p])   Ar[2t+1, 2p+1] =  re(A[t,p])
    Br[j,    2p] =  re(B[p,j])   Br[j,    2p+1] =  im(B[p,j])

whose row `2t` sums `re*re - im*im` (the real part) and whose row `2t+1` sums
`im*re + re*im` (the imaginary part). Hence the accumulator's real row `2i` is
the real part and real row `2i+1` the imaginary part of complex row `i`, which
is what the `(2u+1, 2u+2)` lane pair in `_store_tile_onem!` reads back.

The `2*kc` doubling is confined to 1m's own `accumulate` and never appears in a
length, an offset or a driver loop bound.

### Transcribed from `src/kernels/planar.jl`: the per-shape `vfnmadd` count

Phase C recorded that `muladd(-ai, bi, cr)` folds its `fneg` into
`vfnmadd231pd`/`ps` with zero separate negations. The per-shape table behind
that claim lived only in the source. `@code_native` on the inner loop of
`accumulate` (Julia 1.12.6, ccqlin038, cascadelake `:avx512`), per logical K
step -- which is the shipped form, since `execute_tile!` calls out to
`accumulate` rather than inlining it, on the real path too:

| (MR,NR,W) | eltype | vfnmadd231 | vfmadd231 | vxorp | vsubp | vmulp |
| --- | --- | --- | --- | --- | --- | --- |
| (16,6,8) | `ComplexF64` | 12 | 36 | 0 | 0 | 0 |
| (24,3,8) | `ComplexF64` | 9 | 27 | 0 | 0 | 0 |
| ( 8,8,8) | `ComplexF64` | 8 | 24 | 0 | 0 | 0 |
| (32,6,16) | `ComplexF32` | 12 | 36 | 0 | 0 | 0 |
| (48,3,16) | `ComplexF32` | 9 | 27 | 0 | 0 | 0 |
| (16,8,16) | `ComplexF32` | 8 | 24 | 0 | 0 | 0 |

i.e. exactly `MV*NR` vfnmadd + `3*MV*NR` vfmadd = `4*MV*NR` FMAs and **zero**
separate negations at every menu shape, so hoisting `nai_v = -ai_v` out of the
`j` loop is free rather than merely cheap.

### What was unified, and what was deliberately left alone

Unified, each as one `@inline` helper with per-argument bound type parameters so
no specialization is lost:

- `execute_tile!`'s validation sequence, previously four near-identical copies
  (`ScalarKernel`, `SIMDKernel`, `PlanarKernel`, `OneMKernel`) -- extent checks,
  `kc >= 0`, the `convert`s, the empty short-circuit,
  `checked_tile_storage_bounds`, both buffer-length checks, and the
  `kc == 0 || iszero(alpha)` branch -- as `_execute_tile_prologue!`.
- `store_tile!`'s alpha/beta preamble across the same four, as
  `_store_prologue!`.
- The `pack_a!`/`pack_b!` validation preambles across all four packer methods
  (real and complex descriptors), as `_check_pack_a`/`_check_pack_b`. The
  *loops* are untouched, which is what keeps the real path's generated code
  provably unchanged.
- Constructor validation: `_check_reg_tile` (shared by `KernelDescriptor` and
  `ComplexKernelDescriptor`), `_check_lanewidth` (all three vector kernels) and
  `_check_mr_multiple` (`SIMDKernel` and `PlanarKernel`). Every message is
  byte-identical to what it replaced.
Attempted and REVERTED, with a number, because this is the interesting one:

- The two `TO.tensorcontract!` methods differ only in allocator handling, and
  merging them into a single method over a dispatched `_qs_run!` helper reads
  better and removes a duplicated 10-line signature. It also **measures
  worse**: `+32 B/call` (`Float64`) and `+64 B/call` (`ComplexF64`) against the
  two-method form, on *both* allocator regimes, reproducibly, and reverting the
  merge restores the two-method numbers exactly (3952 / 8144 B and 3952 /
  8624 B on a 40x40x40 `@tensor` call). The extra frame changes what escapes,
  so the `ContractPlan` stops being elided. Dispatching on the allocator was
  preserved in the merged form, so this is not the hazard the original split
  was guarding -- it is a new one, found only because it was measured.

  Left as two methods, with a comment in `src/tensoroperations.jl` carrying the
  numbers so the merge is not re-proposed. **Worth generalising**: "reads
  better" and "allocates the same" are independent properties in this
  codebase, and a readability refactor of a plan-constructing entry point needs
  an allocation measurement even when nothing about its typing changed.

Everything else below was verified allocation-neutral: zero allocations in all
21 cells of (17 shipped kernel configurations + 4 default paths) x `execute!`
on a **scattered** fixture -- permuted A, negative-stride B, sliced C, the
fixture class that hid a real 24 KB regression in this project before -- and
the shared `_execute_tile_prologue!` infers to a concrete `Tuple{Bool,T,T}`
with a concrete `execute_tile!` return at every shipped kernel type crossed
with every `(rows, cols)` axis-kind pair.

**Left alone deliberately.** `_pack_panel!` and `_pack_panel_complex!`
(`src/packing.jl`) are parallel loops and stay parallel. The milestone kept them
separate so that "the real path is byte-identical" is a `git diff` fact rather
than an argument, and unifying them would require *proving* the real path's
generated code unchanged -- which a hoisted `emit` callback cannot be shown to
do without a per-shape `@code_native` comparison this pass did not run. The
shared validation was extracted instead, which captures most of the duplicated
text at none of that risk.

The independent oracles in `test/` were not shrunk: several deliberately
re-implement a packed format so they can catch the implementation being wrong,
and they are the evidence this pass is safe.

### Amendment 5: the engine fits a complex shape to the register file; it does not refuse

Amends the Phase C finding "The complex legacy shape is over the AVX2 register
budget" and its `_complex_default_supported` gate.

**Phase C was wrong, and CI proved it within minutes of the first push.** That
gate refused to pick a complex kernel on any ISA but `:avx512`, on the argument
that "an error beats a guaranteed-spilling default". Three of five CI jobs
failed on it: every Linux runner is AVX2 and every macOS runner is `:neon`, so
complex support was **unavailable through `@tensor` on every machine this
project tests on** -- and on most machines anyone would run it on. The two
jobs that passed were the one configuration the work was developed against.

The priority was backwards. "An error beats a guaranteed spill" is defensible
only when the user has an alternative; here they did not, and a
slow-but-correct kernel beats no complex support at all. `_shape_override`'s
38-41% result had also made spilling feel more expensive than it is -- a spill
costs tens of percent, while a refusal costs everything.

**What ships instead.** Off `:avx512`, `_complex_fitted_shape` selects the
largest shape **from the menu** that this host can actually run, subject to two
necessary conditions: `W <= hardware lanes` (a `Vec{8,Float64}` on 128-bit NEON
is emulated across four registers, so a shape that "fits" on paper would not)
and `_planar_pressure <= nregisters`, with an unrecognised CPU assuming 16, the
conservative x86 baseline. On AVX2 `ComplexF64` that lands on `(4, 6, 4)` at
pressure 16, exactly the ymm budget.

Two details that are load-bearing rather than incidental:

- **Selection is from the menu, not free computation.**
  `_complex_kernel_from_shape` is `@generated` over the menu and falls through
  to its *last* entry on no match, so a freely computed shape absent from the
  menu would silently build a different kernel than was asked for -- worse than
  either a spill or an error. An earlier revision of this fix did compute
  freely, and the test that sweeps synthetic `(vector_bytes, nregisters)` pairs
  caught it. Selecting from the menu makes the membership invariant hold by
  construction rather than by having enumerated the right hardware.
- **Each planar menu gained one `MV = 1` entry per lane width** (so six
  entries, still bounded), which is what guarantees something always fits: at
  `MV = 1, NR = 6` the pressure is `2*6 + 4 = 16`. The 1m menus are unchanged
  at three, since 1m is never selected automatically.

The extent-aware demotion also stops falling back to `_legacy_shape`, which for
`ComplexF64` is `(8, 6, 4)` at pressure 30 -- over AVX2's budget. It now
demotes to the fitted shape.

**These off-`:avx512` shapes are unmeasured** and are not claimed to be good,
only to run without spilling by the budget's own reckoning. The measured
`:avx512` rows are untouched.

**The pattern, recurring for the third time.** `docs/decisions.md`'s Phase G
already recorded: "making a constant hardware-derived silently converts every
test that asserted its old value into a platform-dependent test." Phase C made
the complex *default* hardware-gated and thereby converted every test that asks
for a default complex kernel into a platform-dependent test -- roughly 50 of
them, across four files. The lesson generalises one step further than Phase G
put it: **gating a capability on detected hardware makes the capability itself
platform-dependent, not merely the tests.** A local suite on the one machine the
work was developed on cannot see either.

Also fixed here: the complex packing allocation assertions were missing the
`skip=(VERSION < v"1.11")` marker that every other allocation assertion in this
suite carries, so Julia 1.10 LTS failed on the documented compiler gap rather
than skipping it. Marked, not weakened.

### Amendment 6: per-ISA complex shape rows, and why BLIS is not the source for them

Amends Amendment 5, which left the off-`:avx512` complex shapes as a
register-budget fit with no provenance beyond "it fits".

**BLIS was the first place looked, and it is the wrong source for *planar*
shapes.** Recorded because it is a reasonable thing to try and the reasoning
is not obvious:

- This project already took what BLIS has for the real path. `_legacy_shape`
  `(8, 6, 4)` *is* BLIS's AVX2 dgemm shape -- see "The old default was BLIS's
  AVX2 shape, on an AVX-512 machine" in the Phase G section.
- BLIS computes complex two ways, and neither matches planar's register
  profile. Its native complex asm kernels work on interleaved data and spend
  registers on shuffles, which planar needs none of; its 1m path runs *real*
  kernels. So a BLIS complex `MR`/`NR` encodes the register needs of a
  different method. Transplanting it would be cargo-culting a number whose
  justification does not apply.
- Where BLIS *does* transfer is 1m, which runs a real kernel at `2MR x NR`, so
  BLIS's real shapes are directly meaningful there. Low value at present: 1m is
  never selected automatically, so the shape only matters to a caller who names
  the kernel and could name the shape too.
- Independently, "BLIS microkernels are not reachable from `blis_jll`" (Phase G)
  means only the published shapes were ever available, not the kernels.

**The right source was the sibling `tensorcontract-rs` project**, which sweeps
planar specifically:

| ISA | row | provenance | pressure / budget |
| --- | --- | --- | --- |
| `:avx512` | `ComplexF64 (24,3,8)`, `ComplexF32 (48,3,16)` | swept on ccqlin038, Phase F | 26 / 32 |
| `:neon` | `ComplexF64 (4,6,2)`, `ComplexF32 (8,6,4)` | **measured on an Apple M3 Max** (`aarch64.rs`, `cfg_neon_*`, three arms at `kc = 384`) | 30 / 32 |
| `:avx2` | `ComplexF64 (4,5,4)`, `ComplexF32 (8,5,8)` | **modelled, unmeasured** (`cfg_avx2_f64`, labelled so there) | 14 / 16 |
| `:unknown` | -- | register-budget fit | 14 / 16 |

Two things worth separating.

**NEON is a pin, not a change.** The budget fit already selected exactly the
M3 Max winners -- `(MV, NR) = (2, 6)` for both precisions, converted through
`MR = MV * lanes`. Pinned anyway: arriving at a measured optimum by coincidence
is fragile, because a later menu edit would move it silently and nothing would
notice.

**AVX2 is a change, and it is an argument about headroom rather than about the
optimum.** The fit picked `NR = 6`, whose pressure is `2*6 + 2 + 2 = 16` out of
AVX2's 16 registers -- *zero spare*, leaving LLVM nothing for address
arithmetic or loop counters, so it would spill something regardless of how well
the shape otherwise suits. `NR = 5` costs 14 and leaves two; the sibling
project's own table records the same figure as `live 14`. **The headroom
argument is sound independently of whether 5 is the exact optimum**, which is
the part nobody has measured. Every shipped row now has at least two spare
registers, and a test asserts strict inequality rather than `<=`.

**This cannot be measured here.** `ccqlin038` has 32 registers, so forcing an
AVX2 *shape* on it would not exercise the 16-register constraint that motivates
the row. It needs AVX2-only hardware, and revisiting is cheap:
`benchmark/bench_complex_efficiency.jl` arm 2 is the sweep.

One structural fix came with this. `_shape_override` was consulted *after*
`_rule_applies_complex`, which is `:avx512`-only -- so an AVX2 or NEON row
could never have been reached and would have been dead code. Precedence is now
uniform: override, then the derived rule where validated, then the fit.

Menu membership is unchanged in size (six planar entries per dtype): the two
zero-spare fit shapes were *replaced* by the AVX2 rows rather than added
alongside. A test verifies every resolved shape constructs at the shape asked
for, which is the invariant that matters -- `_complex_kernel_from_shape` falls
through to the menu tail on no match, so a row absent from the menu would
silently build a different kernel.

## Upstream TensorOperations.jl benchmark suite comparison: preparatory milestone

Written on branch `upstream-bench` (base `71c1536`), covering T0-T8. T3
(three-way measurement) and T5 (profiling triage) are the substantive tasks;
this section is their record.

### Motivation and scope

TensorOperations.jl PR #303 (`QuantumKitHub/TensorOperations.jl#303`, branch
`benchmark`, commit `528dd85d8bf886c734a207732a7cb591a3691dd3`, **unmerged**)
adds a standardized `TensorOperationsBenchmarks` suite with case categories
(`:pairwise`, `:tccg`, `:permute`, `:trace`, `:mixed_precision`, `:mps`,
`:ctmrg`, `:trg`). This milestone is **preparatory**: get a working three-way
comparison (`StridedNative`/`StridedBLAS`/`QuasiStrided`) running against that
suite's `:pairwise` and `:tccg` categories only, plus an initial profiling
triage on four cases drawn from that run. No engine change was made anywhere
in this milestone.

Stated plainly because it is easy to miss later: the dependency is pinned to
the **unmerged PR's exact commit SHA** via `benchmark/Project.toml`'s
`[sources]` table, not to a registered release (PR #303 has no release).
**Repointing to the registered release once the PR merges is an explicit,
unstarted follow-up** — nothing here depends on the PR's diff surviving
review unchanged, but the case generators (`_pairwise_cases`,
`_tccg_cases`) and the `within_memory_budget`/`MAX_CASE_BYTES` filter used
below are read directly from that pinned commit and could change before
merge.

### The composite backend (`benchmark/composite_backend.jl`)

The upstream suite's `AbstractProvider` interface takes exactly one
`backend` per provider, but its category list spans both operations
`QuasiStridedBackend` implements (`tensorcontract!`) and operations it does
not (`tensoradd!`/`tensortrace!`, needed for `:permute`/`:trace`/etc.).
`QuasiStridedComposite{F<:AbstractBackend} <: AbstractBackend` exists purely
to give the suite's one-backend-per-provider interface something to point
at:

- `tensorcontract!` dispatches to `QuasiStridedBackend()` **unconditionally**
  — an ineligible contraction still throws exactly as it would through the
  real backend, verified (this milestone's 118 cases pass through this path
  and none of them throw; see the T3 correctness finding below).
- `tensoradd!`/`tensortrace!` dispatch to `backend.addtrace`, which defaults
  to `StridedNative()`, because `QuasiStridedBackend` does not implement
  these operations at all — not as a performance choice.

**This does not change `QuasiStridedBackend`'s frozen hard-reject/no-fallback
invariant** (see "Hard-reject, never fall back" above, in the TensorOperations
integration milestone's Phase A freeze). `QuasiStridedComposite` is a
benchmark-only wrapper one level up from `QuasiStridedBackend`; it must not
migrate into `src/` — that would reverse a frozen product decision, and the
source comment in `composite_backend.jl` says so explicitly.

**Hazard flagged for future readers, not yet materialized.** Any timing taken
under `QuasiStridedComposite` for an add/trace/permute operation measures
`StridedNative` (the `addtrace` field), *not* QuasiStrided — a future
`:permute`/`:trace`/`:mps`/etc. run using this same composite must label
results accordingly. This pass never times those categories (only
`:pairwise`/`:tccg`, both pure contractions), so the hazard did not
materialize here, but it is recorded because the next milestone to touch this
file may not re-read this reasoning.

**Superseded 2026-09-16, at the user's explicit direction** (see
"Amendment 7" above, in the TensorOperations integration milestone's
frozen section): the "must not migrate into `src/`" statement two
paragraphs up no longer holds. `QuasiStridedComposite`'s
`tensoradd!`/`tensortrace!` forwarding to `StridedNative()` is now
`QuasiStridedBackend`'s own default behavior, and `benchmark/composite_backend.jl`/
`benchmark/check_composite_backend.jl` are removed — this section is kept as
the historical record of why the wrapper was built benchmark-only in the
first place (the tradeoff it accepted is the same one Amendment 7 accepts),
not as a statement of the current design. `benchmark/bench_to_suite.jl` now
calls `QuasiStridedBackend()` directly wherever it previously built a
`QuasiStridedComposite()`; every hazard/label-clearly caveat above applies
identically to the real backend now.

### T3 measurement results

**Run metadata.** `ccqlin038.flatironinstitute.org`, 2026-09-15, Julia
1.12.6, this repo's commit `3b864afd57c151ecf7b07a810cadd8d59c711275` (per
`PROVENANCE_to_suite.txt`, matching this branch's "Add benchmark env +
composite backend" commit), `TensorOperationsBenchmarks` pinned rev
`528dd85d8bf886c734a207732a7cb591a3691dd3`, `TensorOperations` resolved
version 5.8.1, `nthreads=1`, `blas_threads=1`, 21 reps (median, one discarded
warm-up). Single machine, **not exclusive**: `machine_load_at_run` recorded
`load average: 1.61, 1.93, 2.86` with other `julia`/`herdr`/`claude`
processes visible in `top_processes_at_run` at the time of the run. Full
provenance and raw CSV in
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/` (gitignored;
see the numbers inlined here and below for anything load-bearing).

**Canary.** StridedBLAS 64³ Float64 `@tensor` matmul, 15 reps, taken at
start/middle/end of the sweep: medians `[6.021e-6, 5.891e-6, 5.651e-6]` s,
relative spread `(max-min)/min = 6.55%`. Noise floor used:
`max(10%, canary spread) = 10.0%` — read any ratio smaller than that as
noise, not as a result.

**Case counts.** 11 of a nominal 15 `:pairwise` cases (dims {15, 63, 128}) —
upstream's own `within_memory_budget` filter (`registry.jl`'s
`MAX_CASE_BYTES = 256 MiB`, Float64-assumed) drops `dim63_2_2_2`,
`dim128_2_1_2`, `dim128_2_2_2`, `dim128_1_3_1`; this is **upstream's filter**,
not a trim this project applied. 48 `:tccg` cases (24 real quantum-chemistry
contraction specs — `ccsd_*`, `ccsd_t_*`, `ao2mo_*`, `intensli_*` — at dims
{8, 16}). Both dtypes: 118 total cases, 354 timed backend-rows.

**Correctness: zero mismatches, zero rejections.** All 118 cases pass
`isapprox` against `StridedBLAS` at rtol `1e-10` (Float64) / `1e-5`
(Float32); `mismatches_to_suite.txt` records 0. Zero backend
rejections/throws on any case, **including** the pure-outer-product
`(1,0,1)` pairwise shapes (`dim15_1_0_1`, `dim63_1_0_1`, `dim128_1_0_1`,
`ncontract=0`) and all 4-6-index chemistry shapes (`ccsd_t_*`'s six-index
outputs among them). This is a positive correctness finding for
`src/tensoroperations.jl` on shapes it had not previously been measured
against — the TensorOperations integration milestone's T9 benchmark used
plain-matmul and shallow-K shapes only.

**Geomean ratios** (from `bench_to_suite.csv`/`summary_to_suite.txt`,
QS/BLAS and Native/QS, geometric mean):

| category | dtype | n | QS/BLAS | Native/QS |
| --- | --- | --- | --- | --- |
| pairwise | Float64 | 11 | 4.293 | 2.586 |
| tccg | Float64 | 48 | 1.774 | 2.068 |
| pairwise | Float32 | 11 | 6.844 | 3.241 |
| tccg | Float32 | 48 | 1.780 | 2.431 |

QuasiStrided beats `StridedBLAS` outright in 32/118 cases; is within-noise-
or-better (ratio <= 1.1x) in 39/118; beats `StridedNative` in 83/118.

**Best wins** (Float64, `:tccg` dim16, QS/BLAS ratio, lower is better for
QuasiStrided): `ao2mo_2` 0.315x, `ao2mo_3` 0.323x, `ccsd_3` 0.372x,
`intensli_1` 0.450x, `ccsd_8` 0.508x, `ccsd_6` 0.509x.

**The one substantive throughput finding: the `ccsd_t_*_dim16` regression
class.** The four `ccsd_t_*_dim16` cases (six-index output, CCSD(T)-shaped
contractions) are **6.6-14.2x** slower than `StridedBLAS` across both dtypes,
and **2.0-4.4x** slower than plain `StridedNative` — the only case class
where QuasiStrided loses to `StridedNative` at a non-trivial absolute size
(re-derived from all 8 rows, not just `ccsd_t_1`; a review pass caught an
earlier draft of this section that quoted only `ccsd_t_1`'s own ratios,
10.1-11.5x / 2.5-3.6x, as if they bounded all four equations). Concretely,
`ccsd_t_1_dim16`:

| dtype | StridedNative | StridedBLAS | QuasiStrided | QS/BLAS |
| --- | --- | --- | --- | --- |
| Float64 | 0.527 s | 0.186 s | 1.883 s | 10.110 |
| Float32 | 0.324 s | 0.124 s | 1.425 s | 11.509 |

(`summary_to_suite.txt`: Float64 section lines 235-239, Float32/tccg section
lines 536-540; full per-equation QS/BLAS and QS/Native ratios for all four
`ccsd_t_*_dim16` cases x both dtypes independently recomputed from
`bench_to_suite.csv` at review time: QS/BLAS in {10.110, 13.983, 9.677,
14.239} (Float64), {11.509, 10.747, 6.604, 9.888} (Float32); QS/Native in
{3.575, 2.544, 2.639, 2.621} (Float64), {4.394, 2.180, 2.259, 2.007}
(Float32).)

Many small `:pairwise`/`:tccg` cases sit at a flat QuasiStrided per-call floor
of roughly 5-25 µs regardless of how little arithmetic the case does — worst
example, Float32 `dim63_1_0_1` (a ~450-8k-flop outer product): QuasiStrided
14.3 µs vs StridedBLAS's 0.50 µs, 28.5x slower. This is consistent with, and
further evidence for, the pre-existing "packing plus per-call overhead
dominates small shapes" finding already on record (STATUS.md's "Next task"
section, and this file's Phase H/"Where the remaining gap actually is").

Best absolute QuasiStrided throughput seen in this run: 75.5 GFLOP/s
(`ccsd_9_dim16`, Float32) against `StridedBLAS`'s 58.5 GFLOP/s at the same
case.

### T5 profiling triage results

Four cases profiled, each on both `QuasiStridedComposite` and `StridedBLAS`:
`ccsd_t_1_dim16`, `ao2mo_2_dim16`, `dim15_2_2_2` (the known small-shape
pattern), and `ccsd_t_1_dim16_f32` (the Float32 repeat of the regression
case). Artifacts: `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/profiles/{buckets_summary.txt, *.flat.txt, *.tree.txt}`.

**Instrument caveat, read before the numbers below.** `compute_buckets`
classifies only leaf frames into named buckets (`microkernel`, `packing`,
`store`, `blas`, etc.). Two separate issues make the raw bucket tables
under-report store cost: (1) a genuine tool bug, caught at review (T6) and
since fixed in `benchmark/profile_buckets.jl` — `"microkernel"`'s bucket
matched on the bare file-path substring `"kernels/"`, and the store path's
own named frames (`_store_tile_scattered!`, `tile_store!`, `tile_offset`,
`_axpby_tile!`) live in that same file (`src/kernels/simd.jl`) as the FMA
microkernel, so first-match-wins ordering swallowed them into `microkernel`
instead of `store` (the original run's `ccsd_t_1_dim16` `microkernel: 7.97%`
figure was mostly store cost, not arithmetic — do not use that number); (2)
even after that fix, the scattered-store path's *further* leaves — in
`SIMD/src/LLVM_intrinsics.jl` and `Base/genericmemory.jl` — still don't match
any bucket substring and still land in `other` alongside real idle-thread
sampling noise (re-run after the fix: `ccsd_t_1_dim16` QuasiStridedComposite
`store` rises from 0.00% to a still-small 1.46%/4.27%-scale figure, `other`
still ~91-95%). Read the bucket tables' `other` row at face value in either
version and you would conclude nothing costly is happening in the store
path; it is. The quantitative claims below come entirely from the
`.tree.txt`/`.flat.txt` inclusive-count profiles (which classify by the full
call stack, not a single leaf), normalized to the compute-thread root, not
from either version of the bucket percentages.

**Verdict: two different mechanisms, not one.** `dim15_2_2_2`'s loss and
`ccsd_t_1_dim16`'s loss are **not** the same story.

Every derived-seconds figure below (as opposed to a plain percentage) uses
one fixed convention: (tree-profile bucket's share of the compute-thread
root sample count) x (T3's measured median time for that case/backend), so
it is reconstructible from `bench_to_suite.csv` plus the cited
`.tree.txt`/`.flat.txt` frame counts alone.

- `dim15_2_2_2` (the known small-shape pattern): microkernel `accumulate`
  (FMA) is 46.63% of QuasiStrided's own time, packing 16.12%, store 21.22% —
  the already-documented "real arithmetic dominates, packing plus per-call
  overhead adds a multiplier" story. QuasiStrided's microkernel time alone
  (2.587e-4 s) is ~0.85x of StridedBLAS's **entire** GEMM time (3.056e-4 s,
  i.e. 94.19% of StridedBLAS's own 3.24365e-4 s median) at this shape.
- `ccsd_t_1_dim16`: `store_tile!`/`_store_tile_scattered!` is 75.30%
  (Float64) / 88.36% (Float32) of QuasiStrided's time; the FMA microkernel is
  0.40%/0.24%; packing is 0.02%/0.03%. Essentially no arithmetic or packing
  cost at all — this is an output-store problem, full stop.

**Two separable causes found for the store cost, both cited to a specific
location.** Neither was fixed or modified — both `src/kernels/simd.jl:217`
and `src/driver.jl:815` were read only, as a read-only diagnostic pass.

- **Cause A: the vectorized store fast-path guard is unsatisfiable for any
  `Array`-backed destination, not just on the TensorOperations path.**
  `src/kernels/simd.jl:217`'s guard —
  `_unit_stride_rows(destination.rows) && destination.storage isa Vector{T}`
  — never passes: `src/driver.jl:815` sets `Cstorage = parent(C)` inside
  `_plan_contract(C::StridedView, ...)` (`src/driver.jl:776`), which every
  plan-construction call goes through regardless of entry point (native
  `contract!`/`plan_contract` or the TensorOperations adapter). `parent` of a
  `StridedView` wrapping a plain `Array` resolves to `Memory{T}`, never
  `Vector{T}`, on Julia >= 1.11 — this is provable **statically** from
  `StridedViews.jl`'s own source (v0.5.2, the version resolved here):
  `_normalizeparent(A::Array) = A.ref.mem` under
  `@static if isdefined(Core, :Memory)` (`StridedViews/src/auxiliary.jl:50-55`),
  applied in the `StridedView` constructor (`StridedViews/src/stridedview.jl:54`),
  with `Base.parent(a::StridedView) = a.parent` (`:121`) — not merely
  consistent with the on-disk runtime probe
  (`profiles/T5_probe_storage_type.jl`/`.txt`, ranks 1, 2, 4, 6, all reporting
  `Memory{T}`), independently confirmed this way at review time. Zero
  `vstore` samples appeared in any of the 8 profiles taken (grepped for
  `vstore` across `profiles/`: zero matches; `vload` frames from
  `panel_vload` do appear, so the absence is informative, not a symbolization
  gap). Consequence: **every** case in this run — including cache-resident
  ones like `dim15_2_2_2`, whose M direction is in fact contiguous and would
  satisfy `_unit_stride_rows` — pays for the scattered-store path
  unconditionally. This is a shape/rank-independent tax on any Array-backed
  destination, not something that only bites `ccsd_t_1` or only the
  TensorOperations entry point.

  **This SUGGESTS, but does not yet verify**, that STATUS.md's existing
  "packing plus per-call cost is the whole gap" attribution (measured with
  the microkernel benchmarked in isolation — see STATUS.md's "Next task" and
  this file's Phase H sections) may have been handed a real `Vector` in that
  isolated benchmarking context, and so never exercised this scattered-store
  path at all. If so, the isolated-microkernel numbers and the TO-adapter
  numbers measured in this milestone may not be measuring the same store code
  path. **Stated as an open, unverified hypothesis, not a conclusion** — no
  attempt was made in this milestone to re-run the isolated microkernel
  benchmark and check its own `destination.storage` type.

- **Cause B: `ccsd_t_1`'s destination has a cache/TLB-unfriendly stride
  pattern, independent of Cause A.** `C[a,b,c,i,j,k]` is 134.2 MB (Float64) —
  this machine's L3 is 25,952,256 bytes (~24.75 MiB/socket, already on record
  above in this file; L3 far smaller than the output) — and its GEMM-M composite axis
  `(i,j,a)` has a non-monotonic C-stride pattern `(4096, 65536, 1)`. Per-
  element store cost measured at 84.5 ns (Float64) / 75.1 ns (Float32),
  versus 2.05-2.33 ns on cache-resident outputs (`ao2mo_2`, `dim15_2_2_2`) —
  a 36-41x per-element blow-up. Attributed **by interpretation**, not
  hardware performance counters, to cache/TLB-unfriendly access. Confirmed
  dtype-independent via the Float32 repeat (`ccsd_t_1_dim16_f32`): same
  dominant bucket, same top leaf frame (`SIMD extractelement`), byte-
  identical 5792-byte-per-call allocation as the Float64 case.

**Why `ao2mo_2` wins despite the same store-dominated profile (48% store
share) as `ccsd_t_1`.** Identical flops-per-output-element (32, since both
have a single contracted index of extent 16) but a 256x smaller output by
element count (16^4 = 65,536 vs 16^6 = 16,777,216; ~129x smaller by total
operand bytes, 1.05 MB vs 135.3 MB) that is cache-resident rather than L3-
exceeding; and `StridedBLAS` must additionally pay for two `Strided` permutes
(66.4% of its own time) plus a 1.05 MB-per-call allocation with a visible GC
tail at this shape. So QuasiStrided's win at `ao2mo_2` is "avoided the
temp/permute", not "faster GEMM" — worth distinguishing from the pairwise/
tccg wins tabulated above, none of which come from a faster microkernel.

**Recommendation, as given: a further broad blind profiling sweep is NOT
warranted.** The two named, separable, directly-testable causes above are
worth a targeted next milestone instead of a wider sweep. Two proposed cheap
checks:

1. Confirm/refute whether `store_tile!`'s `Vector{T}` guard is ever
   satisfiable *anywhere* in the package as currently used — a static/dynamic
   check, no benchmarking required. If never, every existing packing-vs-
   kernel decomposition claim in this file needs re-reading against which
   store path it actually measured.
2. Re-time the four `ccsd_t_*_dim16` cases against a control case differing
   only in having a cache-resident output, to separate Cause A's contribution
   (shape/rank-independent tax) from Cause B's (this specific stride
   pattern's cache/TLB cost).

**These are unverified findings from a diagnostic pass, not fixes, and not
yet confirmed by a second reviewer.** `src/kernels/simd.jl:217` and
`src/driver.jl:815` were read, not modified; no attempt was made in this
milestone to fix or test the fast-path guard.

### Follow-ups, explicitly out of scope for this milestone

- Repointing the `TensorOperationsBenchmarks` dependency once PR #303 merges
  or is released, replacing the pinned-commit `[sources]` entry.
- Investigating/fixing the `store_tile!` `Vector{T}` vs `Memory{T}` guard
  (Cause A above).
- Any engine change targeting the `ccsd_t_*` six-index-output regression
  class (Cause B above).
- Running the remaining upstream categories: `:permute`, `:trace`,
  `:mixed_precision`, `:mps`, `:ctmrg`, `:trg`.
- A wider profiling sweep — explicitly not recommended by this triage (see
  "Recommendation, as given" above).

## Store fast-path investigation: Phase A

Opened 2026-09-15 on branch `store-fastpath-investigation`, base `main`
(`71c1536`). Follow-up to the (unmerged) upstream TensorOperations.jl
benchmark-suite comparison milestone
(`https://github.com/lkdvos/QuasiStrided.jl/pull/5`, branch `upstream-bench`),
whose profiling triage found the `ccsd_t_*_dim16` regression class (six-index
output, 6.6-14.2x slower than `StridedBLAS`) traced to two separable,
**unverified** causes: (A) the vectorized store fast-path guard in
`src/kernels/simd.jl:217` (`_unit_stride_rows(destination.rows) &&
destination.storage isa Vector{T}`) appearing unsatisfiable for any
`Array`-backed destination; (B) that specific case class's output exceeding
L3 with a cache/TLB-unfriendly stride pattern. This milestone resolves Cause
A with evidence before touching anything, per this project's standing
"scout/measure before committing" convention (macro-blocking Phase A,
register-shape milestone).

### Non-goals (frozen for this milestone)

`QuasiStridedBackend`'s hard-reject/no-fallback invariant; the macro-blocking
five-loop structure in `src/driver.jl`; the register-shape/blocking constant
derivation in `src/target.jl`; a general fix for Cause B (output-side
blocking or an accepted temp, like `StridedBLAS`'s own strategy); a wider
profiling sweep across more upstream-suite cases (the prior triage explicitly
recommended against this); repointing the `TensorOperationsBenchmarks`
dependency (PR #303 upstream confirmed still unmerged, 2026-09-15, via `gh pr
view 303 --repo QuantumKitHub/TensorOperations.jl`).

### Planning-time evidence (E1-E7), to be confirmed or refuted by T1-T3

- **E1.** The guard is analytically dead on every real driver path on Julia
  >= 1.11, and live on Julia 1.10. The only destination-tile constructor on
  the real path is `src/driver.jl:815` (`Cstorage = parent(C)`, inside
  `_plan_contract(C::StridedView, ...)` at `:776`). StridedViews v0.5.2
  (`~/.julia/packages/StridedViews/MHBDj/src/auxiliary.jl:50-55`) resolves
  `parent` of an `Array`-backed `StridedView` to `Memory{T}` under `@static
  if isdefined(Core, :Memory)` (true on 1.11+), and to a `Vector{T}` sharing
  memory otherwise (1.10). The reference machine (`ccqlin038`) has run Julia
  1.12.6 for this whole project, so **no driver-level real-path number ever
  recorded here used the vectorized store**.
- **E2.** `SIMD.jl` v3.7.2's `vload`/`vstore` array methods are defined on
  `FastContiguousArray{T,1}` (`~/.julia/packages/SIMD/UiGbs/src/arrayops.jl`),
  and `Memory{T} <: DenseVector{T}` -- no API gap is expected, but must be
  confirmed by direct call (T1), not assumed.
- **E3 (analytical prediction, T1/T3 confirm empirically).** In all four
  `ccsd_t_*` equations the destination's M-composite's first label has a
  large C-stride (e.g. `i`'s stride is 16^3 = 4096 for `dim=16`) and the
  N-composite's first label likewise (`k` or `j`, stride 16^5 or 65536) --
  every micro-tile's rows are a *regular but non-unit-stride* `AffineAxis`.
  **`_unit_stride_rows` is predicted false for every tile in these four
  cases regardless of storage type** -- so even a fully-restored fast path
  would change nothing for the actual regression. If T1/T3 instead find a
  unit-stride sliver in any of the four cases, that refutes E3 and the
  attribution below must be redone before any fix decision.
- **E4.** The `ccsd_t_1` flat profile (prior milestone,
  `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/profiles/ccsd_t_1_dim16-QuasiStridedComposite.flat.txt`)
  shows `_store_tile_scattered!` dominating (19014 of 25182 `execute_tile!`
  samples), confirming the store, not the FMA, is where time goes on this
  case -- but the prior milestone's own `*.buckets.txt` files classify
  88-96% of *all* samples as `other` because they also count the idle
  profile-listener thread; do not cite those bucket percentages as if they
  were the store's true share.
- **E5.** The "SIMDKernel reaches 101-103 GFLOP/s, ~88% of peak" claim
  (this file, "Attributed 2026-09-11 (Phase G...)" blockquote above) was an
  **accumulate-only** isolated microkernel measurement (Float64, `kc=256`),
  never through `store_tile!` at all, and is not reproduced by any script
  currently committed to the tree (`benchmark/bench_kernel_shape.jl` times
  `execute!` through the full driver, not a standalone tile). So that number
  is unrelated to Cause A in either direction -- it neither used nor was
  degraded by the scattered-store path. `benchmark/bench_kernel_shape.jl`'s
  own driver-level numbers, and every other driver-level number in this
  project's history, **were** taken on the scattered-store path, since the
  driver has always gone through `Cstorage = parent(C)` on this project's
  one measurement machine.
- **E6.** Restoring the fast path is not a one-line change: its tail loop
  (`simd.jl:234-238`) indexes the accumulator tuple dynamically, which is
  exactly the allocation-cliff pattern the GUARDRAIL comment at
  `simd.jl:158-166` forbids above `NV = 16` -- currently harmless only
  because the branch is dead on 1.11+. A restored fast path needs a
  statically-indexed tail (mirroring `_store_tile_scattered!`'s own
  generated-code style), and the guard must stay rank-1
  (`DenseVector{T}`, not `DenseArray{T}`) since SIMD's array methods only
  exist for rank-1 arrays.
- **E7.** `benchmark/bench_to_suite.jl`/`profile_to_suite.jl`/
  `composite_backend.jl`/`benchmark/Project.toml` exist only on the unmerged
  `upstream-bench` branch (PR #5). This milestone does not merge or rebase
  onto that branch (two open PRs would become entangled); it recreates a
  minimal, self-contained control script instead (`benchmark/bench_ccsd_t_store.jl`),
  and its docs section cross-references PR #5's section by title rather than
  duplicating it. Expect a trivial append-conflict between the two PRs at
  merge time.

### Decision boundaries (fixed now, before any `src/` edit)

Choose **(a) fix** the guard only if: T1 confirms `vload`/`vstore` on
`Memory{T}` is correct and allocation-free; T1/T3 confirm E3 (so the fix is
not motivated by a false belief that it closes the regression); and T2's
tile-level measurement shows a real, above-noise gain at the shipped
register shapes from a genuine `Vector`/`Memory` destination taking the fast
path. Choose **(b) docs-only correction** if T2 shows no tile-level gain at
any shipped shape, or if a fix cannot reach zero steady-state allocation at
the shipped register shapes (`NV` up to 28) with tail rows without touching
`src/driver.jl`/`src/target.jl`/`src/blocking.jl` (frozen, non-goals above) --
in that case the code is left as-is, the guard gets a comment stating its
per-Julia-version reachability, and the deferral is recorded here. Choose
**(c) escalate to the user** if T1 finds `SIMD.jl` misbehaves on
`Memory{T}` (wrong results or allocation) -- a pointer-based store would then
be a new design question, not a bug fix.

**Replanning triggers**: T1/T3 finds a unit-stride C row in any of the four
regression cases (E3 refuted -- redo the attribution before deciding
anything); a fix would require touching `src/driver.jl`/`src/target.jl`/
`src/blocking.jl`; the re-measurement shows a one-sided regression across
shapes after a fix; the C-local label-order control (Arm 3) in
`bench_ccsd_t_store.jl` runs materially faster than the adapter's own label
order (Arm 1) on the `dim=16` cases -- that would point at a different,
product-level lever (`_classify_labels`'s label ordering, currently pinned
by an existing test) requiring the user's decision as a separate follow-up,
not something this milestone acts on unilaterally.

Review budget: one gated pass (independent review, after docs are written),
`fable_review_storefastpath_used: true` -- spent, do not relaunch for this milestone. Disposition: no blocking findings; several should-fix findings addressed (see "T4-T5" subsection and this milestone's T9 commit).

### T1-T3 results and the decision gate

**T1** (`benchmark/probes/`): E1 confirmed empirically -- `parent(StridedView(::Array{T}))`
is `Memory{T}` (never `Vector{T}`) on this Julia 1.12.6 install, for every
tested rank/dtype. E2 confirmed -- `SIMD.vload`/`vstore` on `Memory{T}` are
correct and allocation-free (0 B, measured inside a compiled wrapper function
to avoid top-level-scope measurement artifacts). **E3 confirmed empirically,
not just analytically**: zero unit-stride C-side M/N slivers across all four
`ccsd_t_*` equations x both dtypes at `dim=16` -- the row stride is always
4096 (`i`'s stride), never 1. Julia 1.10 LTS is installed locally via
`juliaup` but requires its own `Pkg.instantiate()` to test directly (deferred
to CI's `lts` matrix entry, per this milestone's own decision boundaries);
Julia 1.10's behavior is otherwise established by direct reading of
`StridedViews.jl`'s source (E1), not merely inferred.

**T2** (`benchmark/bench_store_path.jl`): the core deliverable for the gate.
`Memory{T}` (D-mem, today's real path) costs **more per element to store
than `Vector{T}`** (D-vec, today's fast path) when rows ARE unit-stride --
a consistent 1.69-2.26 ns/element gap across all 4 kernel shapes x 2 `kc`
values x both beta regimes (a **3.9x-9.9x ratio**, not a flat "4-5x" -- the
ratio varies by shape since the D-vec baseline itself varies; re-derived
from `summary_store_path.txt` at review, T8), far above the 3.7% canary
noise floor. Zero unexpected allocation in any of the 144 measured cells.
Separately, D-strided-hot/cold (non-unit-stride rows, approximating the
actual `ccsd_t_*` addressing pattern) cost 8-25 ns/element **regardless of
storage type** -- confirming the storage-type gap and the regression are
orthogonal, exactly as E3 predicts.

**T3** (`benchmark/bench_ccsd_t_store.jl`, smoke-tested at `dim=8` only):
zero correctness mismatches across all three arms. Arm 3 (a label-order
control, out of this milestone's scope to act on) showed a notable speedup
at `dim=8` on 3 of 4 cases, but the script's own on-the-record analysis
shows the effect's sign is `dim`-vs-`MR`/`NR`-dependent, not a clean win --
flagged for the coordinator to check again at `dim=16` if a future milestone
picks up the label-order lever; **not** investigated further here per the
frozen non-goal.

**Gate decision: (a) fix.** All three conditions from the decision
boundaries above are met: (1) `SIMD` is correct and allocation-free on
`Memory{T}` (T1); (2) E3 is confirmed, so the fix is not motivated by a
false belief that it closes the `ccsd_t_*` regression -- it does not, and
the docs must say so plainly (T1/T3); (3) T2 shows a real, consistent,
above-noise gain at every shipped/swept register shape from a genuine
`Vector`/`Memory` destination taking the fast path. Proceeding to **T4**:
widen `src/kernels/simd.jl:217`'s guard from `destination.storage isa
Vector{T}` to a `DenseVector{T}` check (covering `Memory{T}` too, per D2),
with a statically-indexed tail body (per E6) verified allocation-free at
`NV` up to 28 with tail rows, on Julia >= 1.11. This will speed up ordinary
(unit-stride-destination) contractions on Julia 1.12; it will not move the
`ccsd_t_*_dim16` regression at all, and the docs close-out (T7) must state
that explicitly so nobody reads a future `Pkg.test()`-adjacent benchmark
re-run as evidence either way for that specific case class.

### T4-T5: the fix and its measured effect

**T4 (the fix, already committed on this branch, `f467b45`).**
`src/kernels/simd.jl`'s store fast-path guard (`_vector_store_eligible`,
formerly an inline `isa Vector{T}` check) was widened to accept any concrete
`DenseVector{T}` (covering `Memory{T}`, which is what the real driver always
hands the kernel on Julia >= 1.11), while continuing to exclude
`Matrix`/`SubArray`/non-unit-stride destinations. The old fast path's tail
handling used a runtime-indexed accumulator access (an allocation-cliff risk
once the branch became reachable, per the GUARDRAIL convention already
established elsewhere in this file for the scattered-store path) — replaced
with a `@generated`, statically-indexed `_store_tile_vector!`, mirroring
`_store_tile_scattered!`'s existing style. Verified: full test suite
34853/34853 passing (was 34654 before this milestone; +199 new assertions, no
regressions); zero steady-state allocation with tail rows at register widths
up to `NV=28` on Julia >= 1.11 (skip-marked on 1.10, this project's existing
convention); `@code_native` confirmed the vectorized path (direct `vmovupd`
to the destination pointer) is genuinely reached for `Memory{Float64}`-backed
tiles, with the scattered path (element-by-element, stack round-tripping)
still reached for `SubArray`/`Matrix`/`ScatterAxis`-backed tiles.

**T5 (the measurement campaign).** All results below are from
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/T5_*` files, run
on this machine, 2026-09-15.

- Two-tree ABBA guard comparison (`benchmark/bench_real_path_guard.jl`, base
  commit `d1127dd` vs fixed commit `f467b45`, 21 reps): no one-sided
  regression on any of 18 measured shapes; ratios (fixed/base) ranged from
  ~0.98 (near-parity, e.g. scattered-input shapes untouched by this fix) to
  ~0.31-0.41 (shapes with unit-stride destinations, e.g.
  `shallowK_256x24x256` at 0.407x/0.305x for F64/F32 — a ~2.5-3.3x speedup).
  Canary bracket judged quiet by this project's own established convention
  (middle-vs-end spread, not the full start/mid/end spread, since the
  script's own documented caveat is that the start point reads faster from a
  warm-up effect) — all middle/end spreads were small (0.21-5.58%).
- Re-run of `benchmark/bench_store_path.jl` (unmodified script) on the fixed
  tree: the D-mem vs D-vec per-element store-cost gap that T2 measured
  before the fix (1.69-2.26 ns/element, 3.9x-9.9x, "far above noise") is now
  eliminated — post-fix delta(mem-vec) ranges from -0.10 to +0.06 ns/element
  across all 8 (shape,kc) x 2 beta-regime cells. **This run's own canary
  spread was 9.19%** (worse than T2's original 3.7%, `T5_summary_store_path_after_fix.txt`)
  -- the residual +/-0.10 ns/elem is below this run's own resolution, so read
  "eliminated" as "below this run's resolution", not as a bitwise-proven zero;
  the conclusion still holds because the gap closed by 3.1x-30x depending on
  shape (e.g. shipped F64 kc=16: 2.56 -> 0.03 ns/elem), vastly larger than
  either run's noise. Native-code stack-store instruction counts for D-vec and
  D-mem are now identical (8 total vector stores, 4 stack, 4 other, at both
  shipped shapes) — direct confirmation the two code paths are now genuinely
  the same path. Zero allocation confirmed again (144 cells).
- Full sweep of `benchmark/bench_ccsd_t_store.jl` (both dims, both dtypes,
  all 3 arms, post-fix): Arm 1 (`QuasiStridedBackend` via the TensorOperations
  adapter) at `dim=16` (ratios: `ccsd_t_1` 10.1x/12.1x, `ccsd_t_2` 14.5x/12.9x,
  `ccsd_t_3` 9.5x/8.1x, `ccsd_t_4` 14.4x/11.6x for F64/F32 respectively) is
  close to, but not perfectly matching, this milestone's opening
  characterization of the regression (6.6-14.2x, from
  `benchmark/results/.../summary_to_suite.txt` on the pre-fix `upstream-bench`
  harness, cited here at review time since the original comparison wasn't
  otherwise traceable to an artifact): Float64 agrees to within +/-3%, but
  **Float32 drifted +5% to +23%** (the pre-fix F32 range's own low end, 6.6x,
  came from this same `ccsd_t_3` case). Some of that drift is run-to-run
  noise, not a real change: at `dim=16` Arms 1 and 2 measure the identical
  contraction with identical labels, and for `ccsd_t_3` F32 they differ by
  27% from each other in this same run (1.270 s vs 0.999 s, reps=15) — so the
  dim=16 F32 numbers carry roughly +/-25% run-to-run uncertainty on this
  machine, and the observed F32 drift is inside that band. **Read this
  regression class as "not moved by the fix, within this machine's
  measurement precision" rather than as a precise "unchanged" claim** — the
  mechanism-level evidence (below) is the stronger support, not the timing
  comparison. **Confirming the fix does not move this regression class**
  (E3: zero unit-stride M/N-slivers under the adapter's own label order,
  reconfirmed at `dim=16` specifically, not just analytically inferred, in
  every one of the 8 case x dtype cells).

**Confirmed closing statement**: Cause A fixed for ordinary
(unit-stride-destination) contractions on Julia >= 1.11; Cause B (the
`ccsd_t_*_dim16` regression) confirmed untouched by this fix, exactly as this
milestone predicted from the start.

**A new, unplanned finding, flagged prominently — not folded quietly into the
close-out.** T5's full sweep also ran Arm 3 (a label-order diagnostic control
T3 built, NOT part of this milestone's own goals — it exists only to check
whether the store fast-path or something else was the real lever). At
`dim=16`, permuting the operand carrying label `a` (the destination's
stride-1 axis) to be that operand's own first physical axis gives a **3.3x
to 20x speedup** over Arm 1 on the SAME four regression cases, consistently
across all 8 case x dtype combinations (`ccsd_t_1`: 20.0x F64 / 13.6x F32;
`ccsd_t_2`: 4.1x / 3.7x; `ccsd_t_3`: 4.1x / 4.2x; `ccsd_t_4`: 4.0x / 3.3x —
derived from the medians in `T5_summary_ccsd_t_full.txt`, already committed). This is
**far larger** than anything this milestone's own scope (the store fast-path)
could ever deliver for this case class. Per this milestone's own frozen
"replanning trigger" language (above, "Decision boundaries"/"Replanning
triggers"):

> the C-local label-order control (Arm 3) in `bench_ccsd_t_store.jl` runs
> materially faster than the adapter's own label order (Arm 1) on the
> `dim=16` cases — that would point at a different, product-level lever
> (`_classify_labels`'s label ordering, currently pinned by an existing test)
> requiring the user's decision as a separate follow-up, not something this
> milestone acts on unilaterally.

This is being **reported, NOT acted on**: it points at a different,
product-level lever (`_classify_labels`'s label ordering, currently pinned by
an existing test per that same frozen text) that needs the user's/
coordinator's decision as a SEPARATE follow-up milestone, not something this
milestone's own task graph authorized touching. **No code was changed in
response to this finding.** Closing THIS milestone does not mean the
`ccsd_t_*` regression's story is finished — only that the specific hypothesis
(Cause A, the store fast-path) this milestone was built to test has been
fully resolved.

**Correction (2026-09-19, label-order milestone, T3).** The phrase
"currently pinned by an existing test", used twice above for
`_classify_labels`'s label ordering, was inaccurate: no test pinned the
order of labels inside the M/N composites before this date. The only tests
touching `_classify_labels` were its error paths (`test/test_driver.jl`,
"driver: label validation errors") and an unrelated `ContractPlan`
field-passthrough test; neither asserts anything about composite order. The
first test that does is the label-order milestone's pinning testset,
`test/test_driver.jl`, "label order: pinning test on the ccsd_t shapes
(composite order and swap)", which asserts the post-sort `mgroup`/`ngroup` C
maps and the orientation-swap decision on all four `ccsd_t_*` shapes. That
milestone changed the ordering rule itself (see `plan_contract`'s docstring:
free labels sorted by `|stride|` in `C`, guarded M/N orientation swap);
`_classify_labels`'s own body and return order are unchanged.

**Follow-ups, explicitly out of scope for this milestone:**

- Repointing the `TensorOperationsBenchmarks` dependency (still blocked, PR
  #303 upstream unmerged).
- The label-ordering lever surfaced by Arm 3 — a candidate NEW milestone,
  not started, needs a decision from the user.
- Any remaining upstream benchmark-suite categories (this milestone's own
  frozen non-goals already excluded a wider profiling sweep).
- **Stale `_acc_lane` cross-references** (found at T8 review; not fixed here
  since the affected files are this milestone's own frozen non-goals):
  `_acc_lane` is now unused by any store path (both `_store_tile_scattered!`
  and the new `_store_tile_vector!` are `@generated` with literal indices),
  but `src/kernels/planar.jl` and `src/kernels/onem.jl` each have a comment
  stating the real path *uses* `_acc_lane`, and `test/test_quality.jl`'s Aqua
  `unbound_args = false` justification cites it as the reason -- all three
  are now stale (the justification is not wrong, `_acc_lane` still exists
  and is still unbound-arg-shaped, but its "still in use" premise no longer
  holds). A future task touching those files should update the three
  comments (or delete `_acc_lane` and re-enable `unbound_args = true`, which
  T8 notes would be a net Aqua-coverage gain) -- out of scope here since none
  of the three files were in this milestone's edit scope.
- A tile-level numerical-agreement test for the new vectorized store path
  with unit-stride rows but scattered/irregular *columns* (`ScatterAxis`
  cols) -- found at T8 review as a coverage gap: this combination is newly
  routed to `_store_tile_vector!` and is exercised for allocation by
  `test/test_target.jl`'s sliced-C fixture, but that test only asserts zero
  allocation, not numerical correctness, for this specific combination.

## Label-order milestone

Closes the follow-up flagged (and deliberately not acted on) above: "the
label-ordering lever surfaced by Arm 3". Branch `label-order`, base `main`
(`e2b5e5c`), three commits: `e7c4787` (the fix), `f06fde0` (a benchmark
prototype, additive, no `src/` change), `a650af7` (a review-driven narrowing
of the fix's dtype guard).

### The mechanism

`_classify_labels` (`src/driver.jl:19`) splits free labels into the M list
(A's free labels) and N list (B's free labels) in A's/B's own incidental
physical axis order -- an accident of how the caller happened to lay out its
operands, not a property of the contraction. `fill_offsets!` then walks
whichever composite is built from that list with its *first* label fastest,
so that incidental order was silently deciding the memory-access pattern of
the store into `C`, which is the tensor whose layout actually matters for the
store.

`plan_contract` (`src/driver.jl:795`) now inserts two planning-time,
allocation-free steps between `_classify_labels` and building the
`AxisGroup`s:

1. `_order_free_labels` (`src/driver.jl:127-133`) stable-sorts each of the M
   and N label lists by `abs(stride(C))` of that label's own axis, ascending,
   ties keeping the operand's incidental order. This runs unconditionally,
   for every dtype and every kernel -- there is no guard, because there is no
   plausible downside to walking `C` with its own fastest axis fastest.
2. `_prefer_swap` (`src/driver.jl:174-180`), built on `_leading_unit_run`
   (`src/driver.jl:142-156`), decides whether to additionally swap which
   operand plays the M role and which plays the N role (B feeds M, A feeds N,
   with the K maps, storage/base fields and `atransform`/`btransform` moving
   together -- `src/driver.jl:858-870`). `_leading_unit_run` measures, for an
   already-sorted label list, how many leading elements form a unit-stride
   run in `C`; the swap fires only when the as-is orientation's M list falls
   short of a full `mr(kernel)`-wide register sliver while the swapped
   orientation would clear it. This is guarded to real dtypes only (`T <:
   Real`, `src/driver.jl:858`): `PlanarKernel`/`OneMKernel` (complex) always
   ship the scattered/scalar store regardless of layout (`_vector_store_eligible`
   only exists on the real/`SIMDKernel` path), so the swap has nothing to win
   for them and was measured to cost the as-is orientation's N-side locality
   for no gain (~2-4% regression, `ccsd_t_3`, both complex dtypes, dim=16 --
   this is why `a650af7` narrowed the original unconditional-swap guard down
   to `T <: Real` after `e7c4787` first landed it unconditionally).

### The four regression cases, measured before/after

The regression this discharges was first found in "Store fast-path
investigation: Phase A" above: four TCCG quantum-chemistry contractions
(`ccsd_t_1..4`, six-index output, one contracted index, dim=16),
6.6-14.2x slower than `StridedBLAS` through the adapter path, confirmed
*not* moved by that milestone's own fix (the vectorized-store guard) because
their M/N composites had zero unit-stride register slivers under the
adapter's own (unsorted) label order -- exactly the mechanism this milestone
now sorts away.

Freshly regenerated for this record (not reused from the implementation
commits), `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-19/`
(`bench_ccsd_t_store.csv`/`summary_ccsd_t_store.txt`/
`PROVENANCE_ccsd_t_store.txt`), provenance confirms `git_commit =
a650af768b7b1f6aa4b76ef77bcd67bb30a1a527` (this milestone's tip), Cascade
Lake, Julia 1.13.0, single core, `--dims 8,16 --dtypes Float64,Float32`. Arm
1 is `QuasiStridedBackend` through the real `TensorOperations.tensorcontract!`
adapter path -- the actual production code, not a hand-permuted diagnostic
arm. Reading the ratio table's "Arm-1-StridedBLAS" column (StridedBLAS time /
QuasiStrided time; >1 means QuasiStrided is faster) at `dim=16`, the case
QuasiStrided now used to be slower on:

| case | Float64 | Float32 |
|---|---|---|
| ccsd_t_1 | 6.91x | 2.61x |
| ccsd_t_2 | 4.00x | 1.99x |
| ccsd_t_3 | 4.56x | 4.71x |
| ccsd_t_4 | 3.99x | 1.96x |

So at `dim=16` specifically, QuasiStrided now beats `StridedBLAS` by
**1.96x-6.91x** across these eight case/dtype cells, not the "~1.2x-6.9x"
figure this milestone's own opening brief estimated -- the actual floor is
higher (1.96x, `ccsd_t_4`/Float32) than that estimate. (At `dim=8`, two of
the eight Float32 cells -- `ccsd_t_2` and `ccsd_t_4` -- are instead 0.90x and
0.81x, i.e. slightly *slower* than `StridedBLAS`; `dim=8` is small enough
that both the fixed and swap-guard code paths are already close to
`StridedBLAS`'s own floor and the comparison is noisier. The swap itself
fires for `ccsd_t_2/3/4` at Float64/`dim=16` and `ccsd_t_3` at Float32 (the
16-wide N run there is at least `mr=8`... actually the specific per-case
fire/no-fire pattern is unpacked in `e7c4787`'s own commit message and in the
CSV's `swapped` column) and correctly does not fire where the swapped run
would still be short of `mr`. `bench_ccsd_t_store.jl`'s own Arms 2-5 (the
hand-permuted diagnostic controls that motivated the fix) still show near-
zero mismatches and near-identical timings to Arm 1 post-fix, because the
engine now does internally, for every contraction, what those arms used to
do by hand for these four cases only -- see the clarifying note added to that
script's header this milestone.

### Broader regression sweep

A separate, already-completed sweep (`benchmark/bench_to_suite.jl`, ~150
synthetic + TCCG shapes, two dtypes, base commit `e2b5e5c` vs fixed commit
`a650af7`, two independent rounds per tree) was re-derived from its own CSVs
for this record rather than only quoted:

- Overall QuasiStrided geomean(fix/base) across all 156 measured cases =
  **0.7418** (net faster). By source/dtype: `tccg` Float64 0.527, `tccg`
  ComplexF64 0.653 (the four `ccsd_t_*` cases above are inside this group and
  dominate it), `synthetic` Float64 1.093, `synthetic` ComplexF64 1.067 (the
  synthetic shapes are mostly unaffected either way, as expected -- most were
  already close to their best achievable order).
- Checking each case's ratio independently in *both* rounds against a 12.7%
  noise floor (the worst single-run canary spread across the four runs):
  **zero cases with base median > 100us regress reproducibly in both
  rounds** -- two `dim32_2_1_2_*` (synthetic, Float64, base ~1.8ms) and one
  `ccsd_6_dim16` (tccg, ComplexF64, base ~203us) cross the noise floor in the
  *combined*-median check but do not reproduce independently round-by-round,
  so they read as noise, not as regressions.
- **18 cases do reproduce independently in both rounds**, all at
  microsecond scale: measured absolute times (base or fixed) span
  4.1-38.5us, and ratios span **1.15x-2.48x** (this milestone's opening
  estimate said "1.13-2.48x, 3.8-35us"; the re-derived floor is slightly
  different -- 1.15x not 1.13x, and the absolute range extends to 38.5us not
  35us -- close enough to not change the conclusion, but the exact figures
  above are what is actually on disk, not the opening estimate). All 18 are
  `dim<=16`-scale TCCG cases or small synthetic shapes; none is a shape this
  project would recommend anyone run at that size in isolation for
  performance reasons.
- `benchmark/bench_real_path_guard.jl` (18 shape/dtype combos, one run per
  tree side): geomean(fix/base) = **0.9335**. Per this project's own standing
  convention for this script (see "The real-path regression guard: no
  regression, and the resolution is ~5%" above), **read this as "no
  regression detectable at the guard's own ~5-6% resolution," not as a
  proven per-shape win** -- one cell (`smallMN_16x256x16`, Float64) measured
  +11.7% in this single comparison, which is inside plausible single-run
  noise at this instrument's stated resolution, not evidence of a real
  regression on that shape.

### Test count

Chain: `34856/34856` (this branch's base, `e2b5e5c`, per `STATUS.md`'s own
"T10" line) -> `35165/35165` after `e7c4787` (+309, per that commit's own
message) -> `35168/35168` after `a650af7` (net +3: one existing swap/complex
test was rewritten in place to pin "does not fire" instead of "fires", and a
new "real-kernel proxy" testset was added to keep direct coverage of the
swap's storage/transform bookkeeping now that no production complex path
reaches that branch). `f06fde0` (the benchmark-only commit) does not touch
`test/`. `Pkg.test()` was re-run for this record on the final tree (doc-only
changes on top of `a650af7` do not touch any test file, so the count is
expected to be unchanged at `35168/35168`, 0 failed/errored).

### Known open items

- **The 18 microsecond-scale regressions' mechanism is unconfirmed.** A
  plausible, but *unverified*, candidate: `plan_contract` now resolves
  `_default_kernel` twice unconditionally (`kernel_asis` and
  `kernel_swapped`, `src/driver.jl:849-850`) before the `T <: Real` guard
  even runs, so when the M/N composite ranks differ between the two
  candidate orientations, `execute!`/`_plan_contract` can end up with two
  distinct `ContractPlan` specializations reachable across a program's
  lifetime instead of one -- extra compile/dispatch surface that would show
  up disproportionately at microsecond scale and wash out at millisecond
  scale, consistent with what was measured. Nobody has confirmed this by
  disassembly or by patching out the double resolution and re-measuring; it
  is recorded here as a hypothesis for whoever picks this up next, not as a
  finding.
- **`ContractPlan`'s own docstring did not warn that `Astorage` may be
  `parent(B)` after a swap** (a future maintainer relying on
  `plan.Astorage === parent(A)` would be misled by the field name alone).
  Fixed this milestone with one added sentence on the struct's docstring
  (`src/driver.jl`) -- a doc-only change, not a logic change.
- **`benchmark/bench_ccsd_t_store.jl`'s Arms 3/4-*/5 comments described
  pre-fix semantics** -- they read as if permuting an operand's axes by hand
  still changes what the engine does, but `plan_contract` now performs that
  same sort (and, for Arm 5's swap, the same orientation decision)
  internally and unconditionally, so those arms no longer change engine
  behaviour on current `src/driver.jl`; they remain useful only as the
  working record of what motivated the fix. One clarifying paragraph was
  added near the top of that script's Arms list this milestone, without
  rewriting the arms themselves.

## Profiling pass: where does QuasiStrided spend its time? (2026-09-21)

Opened 2026-09-21 directly on `main`. Goal: reuse the bucketed profiler built
on the (now-merged) `upstream-bench` branch to get a general-purpose,
artefact-traced answer to "how much of QuasiStrided's own time is spent in
the microkernel vs. everything else" across a handful of representative
benchmark cases, rather than relying only on the isolated micro-benchmarks
behind STATUS.md's "Next task" claim ("packing and per-call overhead is now
the whole gap"). No `src/` change; benchmark-tooling only.

### T0: merged `upstream-bench` (PR #5) into `main`

`git merge-tree` showed only two conflicts, both append-only docs
(`STATUS.md`, this file); every `benchmark/*.jl` file merged cleanly,
including keeping `main`'s own later additions (`bench_ccsd_t_store.jl`,
`bench_store_path.jl`, `probes/`) untouched, since the branch never touched
those paths. Resolved both doc conflicts by keeping both sides' sections in
chronological order (no content dropped). `Pkg.test()` on the merged tree:
35170/35170 passing. Pushed directly to `main` (merge commit `19dd25c`);
GitHub auto-detected and closed PR #5 as merged.

**Scope note, found while resolving the merge and confirmed with the user
before proceeding**: the branch is not purely benchmark tooling. A later
commit on it (`3428c61`, "QuasiStridedBackend: fall back to StridedNative
for tensoradd!/tensortrace!") reverses clause 1 of "Hard-reject, never fall
back" (see "Amendment 7" above) -- a real `src/tensoroperations.jl` behavior
change, done "at the user's explicit direction" per its own commit message
in an earlier session, tested (34656/34656 passing at the time) and
documented (Amendment 7, above), but never reflected in PR #5's own GitHub
description. Merged in as-is per the user's explicit choice when asked.

### T2: extended `benchmark/profile_to_suite.jl`'s case coverage

Added three label/dims cases to `CASES` (`plain_256`, `plain_512`: plain
square GEMM at the two largest `MAIN_SHAPES` sizes; `smallN_256x256x12`:
the exact shape STATUS.md's "Next task" cites as 16 vs. Octavian's 87
GFLOP/s), and one new `DIRECT_CASES` list + `profile_one_direct!` for
fixtures that go through `plan_contract`/`execute!` directly rather than
the TensorOperations adapter -- specifically `harness.jl`'s
`build_scattered` fixture (permuted A, negative-stride B, sliced-with-offset
C), which isn't expressible as a plain label/dims dict. `profile_buckets.jl`
was **not** left unchanged as originally planned -- see the two corrections
below (the second found by an independent review of the first).

### Correction 1 (found during T2 smoke-testing): leaf-only classification was wrong for compute-bound cases

The merged `profile_buckets.jl` classified each sample by its **leaf frame
only** (documented as deliberate self-time attribution, to avoid
inclusive/"Count"-column double-counting). Smoke-testing the new `plain_512`
case (a 512³ square GEMM, compute-bound by construction) exposed why that's
wrong: it measured **3.48% microkernel, 94.15% "other"** -- worse than
every small/skewed case, which should be impossible for a plain dense
matmul. The `.flat.txt` dump's self-time column showed the actual hot leaf
frames were `@SIMD/…/LLVM_intrinsics.jl` (`fmuladd`, `vload`, vector
construction) directly beneath `_accumulate_step`/`accumulate`
(`src/kernels/simd.jl:80,138`) in the tree profile -- i.e. genuine FMA-loop
work, but attributed to zero named bucket because the leaf instruction lives
in the `SIMD.jl` *package's own* source file, which carries neither
"kernels/" nor "accumulate" in its path. Leaf-only self-time systematically
undercounts "microkernel" whenever the actual hot instruction is an inlined
third-party intrinsic. First fix: walk each sample's backtrace leaf-to-root
and take the first frame matching a named bucket by EITHER function name or
file substring, instead of checking only the leaf.

### Correction 2 (found by an `orch-reviewer` pass on Correction 1, before committing): file-level catch-alls swallow more-specific ancestors

An independent review of Correction 1 (before any of this was committed)
found that a single leaf-to-root pass checking function-name-or-file
substrings together, first-match-wins, has the same problem one level up:
`profile_buckets.jl`'s `"microkernel"` bucket has a bare `"kernels/"`
file-path catch-all, and `"planning"` has a bare `"driver.jl"` catch-all.
A generic, unnamed frame (e.g. a `macro expansion` thunk inside
`_store_tile_vector!`'s generated body, still in `src/kernels/simd.jl`)
would match `"kernels/"` immediately and stop the walk right there --
never reaching `_store_tile_vector!` itself one frame further up, which
should have classified it as `"store"`. Quantified by the reviewer against
Correction 1's own artefacts: `ccsd_t_1_dim16_f32`'s true store-path share
was ~74-76% self-time (`_store_tile_scattered!`/`_axpby_tile!`), not the
37.9% Correction 1 reported -- the difference had been silently absorbed
into "microkernel". Similarly, `"driver.jl"` absorbed the entire executed
macro-blocking loop nest (`_execute_nest!`, `execute!`, per-block
bookkeeping) into "planning", inflating that bucket 4-5x over the actual
one-time `plan_contract` cost (`ao2mo_2_dim16`: reviewer measured
`plan_contract` inclusive at ~5% vs. a reported "planning" bucket of ~21%).
The reviewer also found: a missing store-path substring (`_store_tile_vector!`
does not contain the literal substring `"store_tile!"` -- the `!` lands
after "vector", not "tile" -- so it matched nothing until the `"kernels/"`
catch-all grabbed it under Correction 1); an overclaimed noise excuse (18-20%
below the historical isolated-GFLOP/s claim is well outside this project's
own "~10% is noise" convention, and this run was on Julia 1.13.0 while every
prior measurement was 1.12.6 -- a live, unmentioned confound); two headline
GFLOP/s figures with no artefact on disk to trace them to; and that
`profile_to_suite.jl` used fixed output filenames, so a same-day repeat
silently overwrote the first repeat's artefacts (`.claude/orchestration/profiling-pass.md`'s
claim of "two repeats' worth of artefacts" was therefore false for the run
it described).

**Second fix** (`benchmark/profile_buckets.jl`): split each backend's bucket
list into `SPECIFIC_*_BUCKETS` (function-name substrings only -- unambiguous
regardless of which file a frame happens to live in) and `FALLBACK_*_BUCKETS`
(the old file-level catch-alls). `_classify_backtrace` now runs the SPECIFIC
pass across a sample's *entire* stack first; only if nothing anywhere in the
stack matches specifically does it re-walk the same stack allowing FALLBACK
matches. This is what makes the generated store body classify correctly:
its `macro expansion` leaf matches nothing specific, but continuing the walk
(still within the specific-only pass) reaches `_store_tile_vector!` itself.
Also: added the missing `"store_tile"` (no bang) substring; split the old
`"planning"` bucket into `"planning"` (the one-time `plan_contract`
construction: `_classify_labels`, `_order_free_labels`, `_default_kernel`,
`default_blocking`, ...) and a new `"driver_loop"` bucket (the *executed*
macro-blocking nest: `_execute_nest!`, `execute!`, `_axis_of`,
`_classify_slivers!`, `fill_offsets!`, ...); tightened several
over-broad substrings the reviewer flagged as latent risks (bare `"gc"`,
`"promote"`, `"StridedView"`, `"tile_offset"`) even though none of them were
empirically wrong in this run. Also (in `profile_to_suite.jl`): added a
`--tag` flag so repeated runs no longer overwrite each other's artefacts,
and a proper >=15-rep `median_time_s` measurement (reused from `harness.jl`,
independent of the profiling loop's own untimed rep-count heuristic) whose
GFLOP/s figure is now printed directly into every `.buckets.txt` file, so
every throughput number below has an on-disk source.

**Net effect of both fixes, `plain_512`**: microkernel 3.48% (leaf-only) ->
84.90-86.11% (Correction 1, file-catch-all-first) -> ~80.3-80.4% + a
correctly separated ~4.6-5.1% `store` (Correction 2). `ccsd_t_1_dim16_f32`:
59.0% microkernel / 37.9% store (Correction 1) -> 16.3-16.8% microkernel /
75.1-75.6% store (Correction 2) -- confirming the reviewer's prediction that
this case is store-dominated, not compute-dominated, by roughly 4:1. All
numbers below are post-both-fixes, from a fresh two-repeat run.

### T3: run and triage (two repeats `r1`/`r2`, `ccqlin038`, 2026-09-21, Julia 1.13.0)

Every case reproduced within ~5 percentage points across two independent
runs (`--tag r1`/`--tag r2`, non-overwriting) on any bucket >=10% share
(the largest drift was `ao2mo_2_dim16`'s `driver_loop`, 21.5% vs. 25.0%);
every bucket table's own sanity line summed to 100.00%; "other" <=0.21%
everywhere. Artefacts:
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/*-r{1,2}.{flat,tree,buckets}.txt`,
GFLOP/s figures quoted below are each traceable to that case's own
`.buckets.txt` (repeat 1) on disk.

Verdict rule (fixed before looking at the corrected numbers): **compute-bound**
requires microkernel+store share >=80% *and* in-situ kernel-path throughput
(achieved GFLOP/s / (microkernel+store) share) within 80% of a same-day
isolated reference; **kernel-stalled** if the share threshold passes but the
throughput leg does not (the "compute path" bucket is large, but per-sample
throughput inside it is far below what an isolated kernel call reaches --
i.e. it isn't actually issuing FMAs efficiently, more likely stalled on
scattered stores or cache misses); otherwise **overhead-bound**, naming the
dominant non-kernel bucket. Isolated same-day reference
(`benchmark/bench_store_path.jl --batch 5000 --reps 9 --skip-native`,
shipped default shapes, kc=256): Float64 `(16,6,8)`: 63.77 GFLOP/s as a
plain `Vector`, 82.66 GFLOP/s as a `PackedPanel` (the driver's actual
representation); Float32 `(32,6,16)`: 108.31 / 159.50 GFLOP/s. Neither
Float64 figure matched the historical "101-103 GFLOP/s" claim on this run
(reported honestly by that script's own repro check) -- this is an 18-23%
gap, outside this project's own "~10% is noise" convention, so it is NOT
waved off as noise here; the most likely explanation is that this run used
Julia 1.13.0, while every prior measurement in this project used 1.12.6 --
a codegen/inlining difference between compiler versions is a live,
unconfirmed alternative to a real regression, and re-measuring on 1.12.6 is
a natural follow-up, not done this pass.

| case (backend=QuasiStrided unless noted) | achieved GFLOP/s | microkernel | store | driver_loop | packing | planning | verdict |
|---|---|---|---|---|---|---|---|
| `plain_512` (512³ GEMM) | 61.6 | 80.4% | 4.6% | 3.6% | 11.3% | 0.2% | **compute-bound**: combined share 85.0%; in-situ = 61.6/0.850 = 72.5 GFLOP/s = 87.7% of the 82.66 GFLOP/s isolated (`PackedPanel`) reference |
| `plain_256` (256³ GEMM) | 54.5 | 73.0% | 3.4% | 3.7% | 18.5% | 0.9% | overhead-bound: combined share 76.4% (just under the 80% threshold); named bucket: packing |
| `scattered_64` (permuted/negative-stride/sliced, direct path) | 41.7 | 57.5% | 9.6% | 8.3% | 24.5% | 0.0% | overhead-bound: packing (share 67.1%) |
| `ccsd_t_1_dim16` (six-index output, post label-order fix) | 17.4 | 63.8% | 16.1% | 17.0% | 3.0% | 0.1% | **kernel-stalled**, not compute-bound: combined share 79.9% (borderline) but in-situ = 17.4/0.799 = 21.8 GFLOP/s = only 26% of the 82.66 GFLOP/s isolated reference -- the throughput leg fails decisively even though the share leg is close to passing |
| `ccsd_t_1_dim16_f32` (same shape, Float32) | 15.9 | 16.8% | 75.1% | 7.0% | 1.0% | 0.1% | overhead-bound: **store**, decisively -- combined share is nominally 91.9%, but it is store, not FMA work, that dominates it (in-situ throughput would be 15.9/0.919 = 17.3 GFLOP/s, 10.8% of the Float32 isolated reference, confirming this is not "fast work counted as compute") |
| `dim15_2_2_2` (rank-4, GEMM-like, dim=15) | 46.9 | 71.5% | 4.4% | 4.4% | 17.7% | 1.5% | overhead-bound: combined share 75.9% (just under threshold); named bucket: packing |
| `ao2mo_2_dim16` (small multi-index) | 11.9 | 28.4% | 9.8% | 21.5% | 33.5% | 4.5% | overhead-bound: packing, with driver-loop bookkeeping a close second -- the smallest-dims case here, consistent with per-call/per-block overhead dominating tiny problems |
| `smallN_256x256x12` (STATUS.md's own cited shape) | 19.0 | 24.0% | 1.4% | 4.2% | 62.7% | 4.9% | **overhead-bound: packing**, decisively -- the strongest, most direct confirmation of STATUS.md's "packing is the whole gap" claim in this whole pass |

`StridedBLAS` reference buckets (for cases where it's profiled): 98.5-99.94%
"blas" on the three plain-GEMM-shaped cases (`plain_256`, `plain_512`,
`dim15_2_2_2`); 23-32% "blas" and 63-76% "permute/copy" on the three
multi-index `ccsd_t_*`/`ao2mo_2` cases (StridedBLAS pays a real permute/copy
cost to reshape into a 2D GEMM view for these shapes, which is a fair
comparison point, not a QuasiStrided-specific defect).

### Reconciliation with STATUS.md's "Next task"

**Corroborated, with numbers**: `smallN_256x256x12` is the same shape
STATUS.md's "Next task" already names, and this pass's fresh, general-purpose
profiler puts 62.7% of its own time in `packing` alone (24.0% microkernel) --
a stronger, more specific statement than the prior isolated-microbenchmark
comparison ("25-54 GFLOP/s vs. Octavian's 79-98"), because it now names
*which* engine-internal code the missing time goes to, not just that overall
throughput is lower. **Qualified**: large square GEMM (`plain_512`) *is*
compute-bound by this pass's fixed threshold; `plain_256` and `dim15_2_2_2`
are close (75-76% combined share, just under the 80% cut) but not over it --
"packing is the whole gap" should not be read as "the microkernel share is
ever small on large cases", but it is also not uniformly >=80% on every
case above the smallest sizes either; the gap narrows with size rather than
vanishing at a clean cutoff. **New evidence, not previously isolated, and
corrected from this pass's own first attempt**: the `ccsd_t_1_dim16_f32`
case is genuinely store-dominated (~75% store share, confirmed by two
independent classification methods after Correction 2), not a borderline
compute-bound case as this pass's own first (pre-review) draft claimed --
worth a follow-up look at whether the vectorized store fast-path's
eligibility guard is Float32-specific in some way for six-index outputs,
not investigated further this pass (single case, read-only finding, `src/`
untouched). **Also new**: `ccsd_t_1_dim16` (Float64, same shape) is
kernel-stalled, not compute-bound, despite a combined microkernel+store
share near the 80% cutoff -- its in-situ throughput is only ~26% of the
isolated reference, meaning whatever is inside that "compute path" bucket
is not running anywhere near peak FMA rate. Neither `ccsd_t_1_dim16` nor its
Float32 twin should be cited as evidence this project is compute-bound on
six-index outputs; both are evidence of the opposite.

### Gate: no fix shipped this pass

None of the findings above met the bounded-fix bar (<=~50 lines, one file,
outside the frozen kernel/driver/blocking core, with a predicted measurable
win, verifiable by the full suite + an ABBA guard + re-profiling). The
`ccsd_t_1_dim16`/`ccsd_t_1_dim16_f32` findings are genuine candidates for a
future milestone but need their own fact-finding (why is the in-situ
microkernel-path throughput so low even where its sample share is large?)
before any fix is proposed. The larger, already-known lever -- Octavian-style
`dontpack`/`maybeinline` dispatch tiers for small/skewed shapes -- remains
explicitly out of scope for this pass, as before.

### Follow-ups, explicitly out of scope for this pass

- Investigate why `ccsd_t_1_dim16`'s microkernel+store-attributed samples
  correspond to only ~26% of isolated-reference throughput (kernel-stalled,
  not compute-bound) and whether `ccsd_t_1_dim16_f32`'s store-path share
  (~75%) is dtype-specific or shape-specific, and whether either is worth a
  fix.
- Re-measure the isolated microkernel reference on Julia 1.12.6 (this
  project's other reference measurements) to check whether the 18-23% gap
  from the historical "101-103 GFLOP/s" figure is a Julia-1.13-specific
  codegen change or a real regression -- not distinguished this pass.
- The Octavian-style `dontpack`/`maybeinline` dispatch tiers STATUS.md's
  "Next task" already flags -- a substantial engine-design decision, not
  started.
- A wider case sweep (this pass profiled 9 cases total; the full
  `MAIN_SHAPES`/`SMALL_SHAPES`/`EXTRA_SHAPES`/dtype grid was not run).
- Complex dtypes beyond the one Float32 case above (planar/1m kernels
  untouched by this pass).

## Packing speed: vectorizing the real packing loop as it exists today (2026-09-21)

Opened 2026-09-21 on branch `packing-speed` (worktree off `main` @ `f318eb9`),
as a bounded follow-up to the profiling pass above, which put 62.7% of
`smallN_256x256x12`'s time in `packing` alone. Question: can the packing
code itself be made faster -- loop structure, vectorization, bounds-check
placement -- *without* changing when or whether packing happens? (The
"should we pack at all" question -- Octavian-style `dontpack`/`maybeinline`
dispatch tiers -- is a separate track and was not touched.) Scope pinned in
advance: `src/kernels/*.jl`, `src/target.jl`, `QuasiStridedBackend`'s
hard-reject invariant and the complex packing loop all off-limits; a fix
ships only if it is roughly <=50 lines in one `src/` file, has a *measured*
win, and is fully verified (full suite, new tests against the scalar path
as a reference across tail/padding combinations, re-profile).

### Evidence: the scalar loop never vectorized, because of the conditional load

`@code_llvm` on `pack_a!` at the driver's exact argument types
(`PackedPanel{Float64}`, `QSTile{Memory{Float64},AffineAxis,AffineAxis}`,
`KernelDescriptor{16,6,Float64}`, `identity`) showed no `<N x double>`
anywhere: the old `_pack_panel!` body `v = i < valid ? load(i,p) : zero(T)`
compiled to a per-element *branch* around the load (blocks `L64 -> L68
(load) / L99 (phi with 0.0) -> store double ... align 1`, 16 trips per K
step, loop bound compared against a runtime `valid`). A load under a
condition cannot be if-converted without a masked load, so LLVM refused to
vectorize the loop at all; every element paid a compare, a branch, a scalar
load, a scalar store, and a `stride*i` multiply (the row stride is a runtime
field). Micro-timing on the driver's types: 0.74-0.94 ns/element for A at
Float64 vs 0.26-0.28 ns/element for `copyto!` of the same bytes. The
baseline tree profile (`smallN_256x256x12-...-D1-probe.tree.txt`) had 30433
of 42774 samples under `pack_a!`, split roughly 13.4k on the load+select
line, 8.8k on the store line, 3.1k loop control.

Tile geometry of the four profiled cases, read off their `ContractPlan`s:
every one has an A whose M axis is unit-stride (column-major or
unit-stride-fastest multi-index) and a `Memory{Float64}` storage, kernel
`SIMDKernel{16,6,Float64,8}`, transforms `identity`/`identity`. So "unit-stride
rows filling a full MR sliver, straight copy, `PackedPanel` destination" is
the common A case, not a special one.

### Change (`src/packing.jl` only; 40 non-comment lines)

1. **`_pack_panel!` full/tail split.** The physical dim (MR or NR) is now a
   `Val{PD}` compile-time constant, and the body has two branches: a full
   sliver (`valid == PD`) runs a constant-trip inner loop with an
   *unconditional* load, which LLVM fully unrolls (and, on the B side,
   vectorizes as gathers); a tail sliver writes its `0:valid-1` values and
   then its `valid:PD-1` literal zeros as two separate loops. No loop body
   holds a conditional load any more. Per-K-step store order and the padding
   contract (padding lanes never read `source`, never call `transform`) are
   unchanged. This alone is 1.5-2x on A's fallback and 1.3-1.7x on B.
2. **`_pack_a_contiguous!` fast path**, gated by `_pack_a_contiguous_eligible`:
   `packed isa PackedPanel{T}` && `source.storage isa DenseVector{T}` &&
   `_copies_unchanged(transform, T)` (`identity`, or `conj` on a real `T`)
   && `m == MR` && `_unit_stride_rows(source.rows)`. Each K step is then one
   `vload(Vec{MR,T})` from `base + rows.base + axis_offset(cols, p)` and one
   `vstore` to `packed.ptr + MR*p` -- the column axis may be any `Axis`
   (affine with any stride including negative/zero, or `PtrScatterAxis`);
   only the row axis needs contiguity. Type-only tests fold at compile time.
   The gate is its own function so a test can assert it *fires* for the
   driver's argument types and stays off for every ineligible shape (a
   review finding: without that, a silently broken gate would pass every
   value test and only show up as a benchmark regression).

**Why it is safe.** Padding never arises in the fast path (`m == MR`
exactly). `_check_pack_a` runs first and `checked_tile_storage_bounds`
validates every address `base + rows.base + i + col_offset(p)`, `0 <= i <
MR`, against `length(storage)` -- exactly the span each `Vec{MR}` load
covers -- so nothing is read that the scalar path would not have read.
Negative or non-unit row strides fall through to the scalar path (a
negative-stride "contiguous" run is not a forward vector load). `conj` on a
real eltype is the identity, so the copy is exact; on a complex eltype
`_copies_unchanged` is `false` (pinned by test). The destination is the
borrowed `PackedPanel` pointer `execute!` already `GC.@preserve`s; the source
is `GC.@preserve`d locally. `_unit_stride_rows` (src/kernels/simd.jl) has
methods for exactly the three `Axis` kinds and deliberately no fallback, so
an unknown axis type is a `MethodError`, never a silent default.
`_pack_panel_complex!` is untouched (its header comment now says so).

### Measured

Micro (driver argument types, min of 1000 reps, ccqlin038, Julia 1.13.0,
shared machine at load ~4-5/32):

| pack | shape | before | after | ratio |
|---|---|---|---|---|
| A, Float64 MR=16 | 256x256 | 48.7 us | 18.5 us (`copyto!`: 17.7 us) | 2.6x |
| A, Float64 | 225x225 (tail slivers) | 33.8 us | 13.8 us | 2.5x |
| A, Float64 | 4096x16 | 55.0 us | 28.1 us | 2.0x |
| A, Float32 MR=32 | 256x256 | 36.6 us | 5.3 us | 7.0x |
| B, Float64 NR=6 | 256x12 | 1.55 us | 1.03 us | 1.5x |
| B, Float64 | 225x225 | 26.1 us | 18.8 us | 1.4x |

Re-profile with `benchmark/profile_to_suite.jl`, tags `D1-probe` (before) and
`D1-post` (after), same session; artefacts under
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/*-D1-{probe,post}.*`:

| case | GFLOP/s before -> after | packing share | microkernel share |
|---|---|---|---|
| `smallN_256x256x12` | 16.0 -> **34.6** (2.2x) | 62.1% -> **32.8%** | 21.9% -> 40.3% |
| `scattered_64` | 39.3 -> **51.6** (1.3x) | 25.7% -> **7.0%** | 57.5% -> 71.5% |
| `ao2mo_2_dim16` | 11.7 -> **14.5** (1.2x) | 34.2% -> **10.1%** | 27.5% -> 42.5% |
| `dim15_2_2_2` | 46.0 -> **49.4** (1.07x) | 19.5% -> **11.2%** | 70.1% -> 76.5% |

Post-change, ~90% of `smallN`'s remaining packing samples sit inside
`SIMD.vload`/`vstore` -- i.e. plain data movement at ~35 GB/s against
`copyto!`'s ~60 GB/s -- so the loop is no longer the problem; what is left
is memory traffic that only "don't pack A for small N" removes. Note the
`D1-probe` baseline for `smallN` (16.0 GFLOP/s) came in below the earlier
`r1` figure (19.0) on a busier machine; the relative packing-share drop and
the 2x micro-timings are far outside that noise and were reproduced in three
separate micro-runs.

### Tests added (`test/test_packing.jl`)

Gate-predicate unit tests (incl. `conj` on `ComplexF64` is NOT a straight
copy); a PackedPanel-vs-direct-indexing oracle over `{Float64,Float32} x MR
in {4,16} x {Vector,Memory} storage x 8 row cases (eligible unit-stride at
two bases / stride 2 / negative / tail / empty / contiguous-but-`ScatterAxis`
/ contiguous-but-`PtrScatterAxis`) x 6 column cases (affine +/unit/negative/
zero stride, `ScatterAxis`, `PtrScatterAxis`) x {identity, conj, x->-x} x 2
tile bases`, asserting at each point that the gate evaluates to exactly
`row_eligible && transform in (identity, conj)` for the panel and `false` for
a `Vector` destination, that the panel and the `Vector` path agree with the
oracle, and that a suffix canary is untouched; `_pack_a_contiguous!` called
directly against the oracle into a middle sliver of a canaried buffer; every
tail width `m in 0:MR`, `n in 0:NR` for both destination kinds with
literal-zero padding and a call-counting transform; zero steady-state
allocation on the driver's exact argument types (PackedPanel + Memory +
Affine/PtrScatter, identity and conj, full/tail/strided) for (16,6,F64),
(32,6,F32), (4,3,F64). `benchmark/profile_buckets.jl`'s "packing" bucket
also names `_pack_a_contiguous!` explicitly so the attribution does not
depend on the `pack_a!` ancestor frame surviving inlining.

### Not done, flagged for later

- **Per-sliver validation cost on tiny tiles.** `checked_tile_storage_bounds`
  + `axis_offset_range` + `_check_pack_a/b` self-time is 7.4% of
  `ao2mo_2_dim16` (K extent 16 = one MR sliver of 256 elements per call),
  3.2% of `scattered_64`, ~1% elsewhere. Hoisting the storage-bounds check
  to once per macro block in `_execute_nest!` would remove it, but needs an
  unchecked internal `pack_a!` entry point and so changes the public
  "all validation before any write" contract -- a separate decision.
- **B-side loop order** (`j` outer / `p` inner, contiguous source reads for
  unit-stride K) measured 1.1-1.4x over the old loop but no better than the
  full-sliver branch above on the same shapes; dropped.
- **Complex packing** (`_pack_panel_complex!`) has the same per-element
  `if t < valid` and would take the same full/tail split, plus a
  deinterleaving vector path for planar A with unit-stride rows; untouched,
  needs its own tests against `test/test_packing_complex.jl`'s oracle.
- `ao2mo_2_dim16`'s largest non-kernel bucket is now `driver_loop` (28.7%:
  `fill_offsets!`/`describe_block`/`_classify_slivers!`), outside this task.

### T4: extended the grid to every `MAIN_SHAPES`/`SMALL_SHAPES`/`EXTRA_SHAPES` shape, all four dtypes, and the `1m` complex kernel (2026-09-21, follow-up pass)

Addresses the last two T3 follow-ups above (wider case sweep; complex
dtypes) additively: `benchmark/profile_to_suite.jl`'s `CASES` gained every
`MAIN_SHAPES ∪ SMALL_SHAPES ∪ EXTRA_SHAPES` shape (`harness.jl`'s
`ShapeSpec`s, looked up by name into a new `SHAPES_BY_NAME` dict rather than
re-hardcoding dims) as a plain square-GEMM-shaped label/dims case, at both
real dtypes (`Float64`/`Float32`, all shapes) and both complex dtypes
(`ComplexF64`/`ComplexF32`, `MAIN_SHAPES ∪ SMALL_SHAPES` only --
`1024x256x1024` was left out of the complex grid to keep this pass's total
sweep time bounded, deliberately narrower than the real-dtype grid, which
also covers `EXTRA_SHAPES`. **Correction**: an earlier draft of this section
justified the exclusion by claiming the fixture is ">1 GB" at `ComplexF64`;
that is wrong by about 40x -- the shape's `A`+`B`+`C` arrays together are
only ~25 MB at `ComplexF64` (`C` alone is 1024×1024×16 B ≈ 16 MB). The
exclusion itself may still be reasonable on sweep-runtime grounds; the
memory-footprint reason was simply false and is retracted here). The three
shapes T3 already covered at `Float64` (`plain_256`, `plain_512`,
`smallN_256x256x12`) are guarded by `_EXISTING_CASE_IDS` so this pass never
emits a duplicate case id for them; every other id is new. `DIRECT_CASES`
gained two `OneMKernel` entries (`onem_256`/`onem_512`, both complex dtypes)
that call `plan_contract(...; kernel = ...)` directly with
`kernel_shapes(T, OneMMethod())[end]` -- the only way to reach `1m` at all,
since `src/driver.jl`'s `_default_complex_method` always returns
`PlanarMethod()` for a plain label/dims contraction, so none of the new
complex `CASES` above ever exercise it. **Correction**: an earlier draft
called `[end]` "the shipped default register shape"; there is no shipped
`1m` default -- `1m` is never auto-selected, and `[end]` is simply the
menu's fallthrough/last entry. Concretely, `kernel_shapes(ComplexF64,
OneMMethod())[end] = (MR=8, NR=8, W=8)`, which the Complex element-type
milestone's own Phase F sweep (see that section above) measured as the
*worst*-ranked of the three `1m` shapes on this machine; `kernel_shapes(
ComplexF32, OneMMethod())[end] = (MR=16, NR=8, W=16)`, which that same sweep
ranked *best* of its three. So the `onem_256_c64`/`onem_512_c64` rows below
and the `onem_256_c32`/`onem_512_c32` rows are each measuring a different,
dtype-specific point in the `1m` shape menu -- opposite ends of the ranking
-- not a matched pair; do not read the `c64`-vs-`c32` `onem_*` comparison
below as apples-to-apples. `case_flops` was switched from a hardcoded
`2.0 * ...` to `flops_per_mac(case.dtype) * ...` (`harness.jl`'s existing
helper, `8` for complex dtypes vs `2` for real) so GFLOP/s figures stay
correct for the new complex cases. No other files touched; changes are
additive only, `benchmark/profile_buckets.jl` untouched.

**Correction, 2026-09-21 (this entry rewritten after an independent
review): the first full sweep below was contaminated by concurrent
execution, and its write-up's contamination story was backwards.** The
smoke test (`--tag smoke`, three representative new ids) and what was then
called "the quiet, uncontended full run" (`--tag r1`) were **not**
sequential -- `buckets_summary-r1.txt`'s header timestamp is
`2026-09-21T12:23:55`, `buckets_summary-smoke.txt`'s is
`2026-09-21T12:30:35`, and the full 81-unit `r1` sweep's own artefact
mtimes span `12:23:55`-`12:42:32` (about 18.5 minutes, matching the clean
re-run's duration below) -- so the smoke test ran entirely *inside* the
full sweep's own execution window, not before or after it. The full `r1`
sweep was therefore almost certainly the smoke test's own contamination
source (two Julia processes on the same machine at once), not an
independent "unrelated CPU-bound job" as originally written, and `r1`'s own
numbers for whatever cases happened to profile during that ~64 s overlap
are suspect on exactly the same grounds. This was caught by comparing `r1`
against this same shape/dtype's original, independent profiling pass (`main`
worktree, `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/*-r{1,2}.buckets.txt`):
`ccsd_t_1_dim16`'s microkernel share and `smallN_256x256x12`'s throughput/
gc-alloc share both drifted outside this project's standing 5-point
reproducibility bar. `r1`'s own `.buckets.txt` sanity lines and low "other"
shares (<=0.52%) do NOT catch this kind of contamination -- `other` only
inflates when the *profiled* process itself gets descheduled, not when a
second, separately-profiled process on the same core steals cycles between
samples -- so "clean-looking buckets" was never good evidence that `r1` was
uncontended. **`r1` is superseded below by a clean re-run; do not treat any
`r1` number in this section as trustworthy without cross-checking r2/r3.**

**Clean re-run (`--tag r2`, `ccqlin038`, 2026-09-21, Julia 1.13.0)**: run as
the *only* process on the machine -- verified before starting (no other
`julia`/benchmark processes running, `uptime` load average 2.4-2.8 on this
32-core box, the only sustained CPU consumer being an unrelated, pre-existing
stale VSCode Julia language-analysis helper pinned to one core since
2026-09-16, which cannot contend with a `-t 1` benchmark process on an
otherwise-idle 32-core machine) and monitored throughout (no other
`profile_to_suite.jl`/`bench_*` process observed while it ran; header
timestamp `2026-09-21T15:01:14`, last artefact written `15:20:27` EDT, ~19
minutes end to end, in line with `r1`'s own ~18.5-minute duration). All 81
(case, backend) units completed without error; "other" was
<=0.39% on every row (worst: `smallMN_16x256x16_c64` at 0.39%), at least as
clean as the (falsely) "quiet" `r1` claim and with none of `r1`'s timing
overlap. Artefacts:
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/*-r2.{flat,tree,buckets}.txt`,
`buckets_summary-r2.txt`.

**Reproducibility check, r1 vs r2, whole grid**: bucket-share agreement (any
bucket >=10% share, 5-point bar) held for most of the 81 units, but not all
-- seven cases exceeded the bar (`ccsd_t_1_dim16` microkernel +4.4 to +7.1
points depending on which `r1`/original-pass number it's compared against;
`smallMN_16x256x16` planning; `smallM_12x256x256` and
`smallM_12x256x256_f32` packing/microkernel; `smallN_256x256x12`
gc/alloc+throughput; `smallN_256x256x12_c32` and `smallN_256x256x12_f32`
packing). All seven are small/packing-dominated cases -- exactly the
regime most sensitive to a few hundred microseconds of contention per call.
Given `r1` is established contaminated, `r2` (not `r1`) is treated as
authoritative for the table below, and a third repeat (`--tag r3`, same
"only process on the machine" protocol, targeted at just the flagged case
ids to keep the check cheap) was run for all seven to confirm `r2` itself is
not also an outlier:

- `ccsd_t_1_dim16` (`QuasiStridedBackend`): microkernel/store/GFLOP-s across
  `r1`=70.9%/12.6%/17.4, `r2`=68.1%/13.6%/18.1, `r3`=63.6%/16.4%/17.4, vs. the
  **original, independent pass**'s own `r1`=63.8%/16.1%/17.4 and
  `r2`=62.9%/18.3%/17.3. `r3` reproduces the original pass closely; `r2` was
  within-bar against the original's `r1` (+4.4 points) but just outside it
  against the original's `r2` (+5.3 points) -- resolved by `r3` as ordinary
  noise, not a real regression. Combined (microkernel+store) share is
  stable at 79.9-84.6% across all four clean readings, so the case's
  *verdict* (compute-path, ~80% share) never moved; only the
  microkernel/store split wobbles by a few points run to run, which this
  case has done since T3 first profiled it.
- `smallN_256x256x12` (`QuasiStridedBackend`): the blocking finding was a
  43% throughput drop (18.97 -> 10.85 GFLOP/s) and gc/alloc jumping
  1.57%->6.50% in the contaminated `r1`. Clean `r2` measures 17.085 GFLOP/s
  (within ~10% of the original pass's 17.5-19.0 GFLOP/s, ordinary run-to-run
  noise) and gc/alloc at 4.61% -- better than the contaminated reading but
  still above the original pass's 1.57-1.60%. gc/alloc is a <10%-share
  bucket so it falls outside this project's 5-point reproducibility bar by
  convention, but the residual elevation (4.6% vs 1.6%) is real enough to
  flag rather than wave away; every other bucket (packing 62.6% vs
  62.7-64.6%, microkernel 21.6% vs 22.9-24.0%) reproduces the original pass
  within ~2.4 points. Not investigated further this pass -- see follow-ups.
- `smallMN_16x256x16`, `smallM_12x256x256_f32`, `smallN_256x256x12_c32`: all
  three resolved cleanly -- `r2` and `r3` agree within the 5-point bar on
  every bucket (e.g. `smallMN_16x256x16` planning 22.5%/21.7%,
  `smallM_12x256x256_f32` microkernel 35.0%/34.2%), confirming `r1` (not
  `r2`) was the outlier for these.
- `smallM_12x256x256` and `smallN_256x256x12_f32`: **genuinely noisy after
  three repeats, reported honestly rather than forced to a number.**
  `smallM_12x256x256` packing share reads 37.4% (`r1`), 45.4% (`r2`), 38.3%
  (`r3`) -- `r2` is the outlier here (`r1`/`r3` agree within 1 point), but
  three data points aren't enough to call it settled either way.
  `smallN_256x256x12_f32` packing reads 65.5% (`r1`), 79.1% (`r2`), 71.0%
  (`r3`) -- a ~14-point spread across three clean-looking runs of the same
  tiny (256×256×12, `Float32`), sub-30-µs-per-call case. In both cases the
  *verdict* (packing is the dominant, >=37% bucket by a wide margin over
  every other bucket) never changes, only the exact percentage; the table
  below reports each as a range rather than picking one run's number. This
  is consistent with -- and a sharper instance of -- this project's own
  standing "measurement hygiene" note (STATUS.md, "Next task": "ccqlin038 is
  not reliably exclusive... treat anything under ~10% as noise") applied to
  the shortest-duration cases in this grid.

**Provenance note**: this worktree's `-r1`/`-r2`/`-r3` tags are **not** the
same artefacts as the original profiling pass's own `-r1`/`-r2` in the
`main` worktree (`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/`
there) -- both happen to share a results-directory date because both ran on
2026-09-21, but they are separate measurement passes (the original pass's
8-case T3 sweep vs. this pass's 81-unit grid extension) with independently
numbered repeat tags. Where this section compares against "the original
pass", that always means the `main`-worktree artefacts explicitly, never
this worktree's own superseded `r1`.

**Verdict rule for this section, deliberately simplified from T3's**: T3
used a two-legged rule (combined microkernel+store share >=80% for
"compute-bound", but *only if* in-situ throughput --
`achieved GFLOP/s / share` -- was also within 80% of a same-day isolated
microkernel reference; otherwise "kernel-stalled" even at a high share).
That second leg needs a fresh isolated reference per dtype/kernel
combination (`bench_store_path.jl` only covers the two real dtypes'
`PlanarMethod`-adjacent shapes; the `1m`/complex-planar kernels have no
equivalent isolated-reference script run this pass). Measuring that for
four dtypes x nine new shapes x two kernel families was judged out of
scope for a grid-extension pass -- **the table below uses the share leg
only** (combined microkernel+store share >=80% -> "compute-path", else
name the largest of packing/driver_loop/planning as "overhead: <bucket>").
This means a few rows below that would likely be reclassified
"kernel-stalled" under T3's full rule are instead reported as "compute-path"
here on share alone; where a case's `store` share, not `microkernel`, is
the larger of the two (a T3-flagged smell -- see `ccsd_t_1_dim16_f32`
above), that is called out in the table notes rather than silently folded
into "compute-path". Treat every "compute-path" verdict below as
share-only, not confirmed-efficient; T3's own worked examples show both can
diverge.

New cases only (the eight already-tabulated T3 cases -- `ao2mo_2_dim16`,
`ccsd_t_1_dim16[_f32]`, `dim15_2_2_2`, `plain_256`, `plain_512`,
`scattered_64`, `smallN_256x256x12` -- reproduce within a few points of T3's
own numbers on the clean `r2` re-run and are not repeated here; see T3's
table above for those, and the reproducibility-check discussion above for
the two that did *not* reproduce on the first, contaminated `r1` attempt).
**All numbers below are from the clean `r2` re-run** (not the superseded
`r1`), except the two cells flagged noisy in the reproducibility check
above, which report the `r1`/`r2`/`r3` range instead of a single number.
`QuasiStridedBackend` (or `QuasiStridedDirect` for the `onem_*` rows)
buckets only; `bl gflop/s` is that same case's `StridedBLAS` figure for
reference (`-` where there is none, i.e. the `onem_*`/`scattered_64`
direct-path cases). **Caveat on the `qs`/`bl` GFLOP/s columns**: a handful
of the shortest-duration cases (tens of µs/call) showed 2-3x run-to-run
swings in this *timing* figure between `r1` and `r2` (e.g. `plain_64`
24.4->8.5 GFLOP/s, `plain_128_c32` 88.2->31.3 GFLOP/s) with no
corresponding drift in that same case's bucket shares (all within the
5-point bar) -- i.e. the wall-clock median is noisier than the profiler's
own cost attribution at this size, consistent with STATUS.md's standing
"treat anything under ~10% as noise" caveat, here exceeded by a wide margin
for these two specific readings. Treat single-run GFLOP/s figures below as
indicative, not precise, for any case under ~50 µs/call; the bucket-share
columns are the load-bearing data in this table:

| case | qs GFLOP/s | bl GFLOP/s | microkernel | store | packing | driver_loop | planning | combined | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `plain_64` | 8.5 | 84.0 | 33.2% | 3.7% | 30.2% | 9.6% | 17.5% | 36.9% | overhead: packing |
| `plain_64_f32` | 30.3 | 198.0 | 23.5% | 2.8% | 37.1% | 7.5% | 21.6% | 26.3% | overhead: packing |
| `plain_64_c64` | 36.7 | 60.2 | 41.0% | 18.1% | 13.7% | 5.5% | 16.9% | 59.1% | overhead: planning |
| `plain_64_c32` | 45.2 | 97.6 | 30.4% | 19.4% | 19.6% | 5.4% | 17.6% | 49.8% | overhead: packing |
| `plain_128` | 43.6 | 64.8 | 57.2% | 3.7% | 27.3% | 5.1% | 5.1% | 60.9% | overhead: packing |
| `plain_128_f32` | 63.3 | 56.3 | 40.1% | 2.8% | 40.5% | 6.2% | 7.8% | 42.9% | overhead: packing |
| `plain_128_c64` | 54.1 | 74.9 | 64.4% | 14.2% | 13.5% | 3.6% | 3.4% | 78.6% | overhead: packing (borderline) |
| `plain_128_c32` | 31.3 | 139.9 | 51.9% | 20.0% | 19.2% | 3.6% | 4.0% | 71.9% | overhead: packing |
| `plain_256_f32` | 91.0 | 156.7 | 60.7% | 1.9% | 31.1% | 3.9% | 1.7% | 62.6% | overhead: packing |
| `plain_256_c64` | 68.3 | 77.4 | 77.5% | 9.7% | 9.8% | 1.9% | 0.7% | 87.2% | compute-path |
| `plain_256_c32` | 122.6 | 155.0 | 70.1% | 13.9% | 12.5% | 1.9% | 1.1% | 84.1% | compute-path; store is about a fifth of microkernel's own share (not "half", correcting the earlier draft) |
| `plain_512_f32` | 124.4 | 172.3 | 75.7% | 1.6% | 19.8% | 2.4% | 0.4% | 77.3% | overhead: packing (borderline) |
| `plain_512_c64` | 73.9 | 85.1 | 76.8% | 11.2% | 10.0% | 1.8% | 0.2% | 88.0% | compute-path |
| `plain_512_c32` | 158.2 | 175.4 | 77.5% | 11.3% | 9.8% | 0.9% | 0.3% | 88.8% | compute-path |
| `big_1024x256x1024` | 56.9 | 84.5 | 80.5% | 3.2% | 10.9% | 5.1% | 0.2% | 83.8% | compute-path |
| `big_1024x256x1024_f32` | 118.4 | 176.6 | 81.8% | 3.0% | 10.9% | 4.0% | 0.2% | 84.9% | compute-path |
| `shallowK_256x24x256` | 29.8 | 63.4 | 56.7% | 13.4% | 10.1% | 13.3% | 4.8% | 70.1% | overhead: driver_loop |
| `shallowK_256x24x256_f32` | 51.5 | 104.8 | 45.1% | 11.2% | 18.4% | 14.4% | 8.1% | 56.3% | overhead: packing |
| `shallowK_256x24x256_c64` | 34.1 | 43.0 | 42.6% | 43.5% | 4.0% | 6.6% | 2.4% | 86.1% | compute-path; **store > microkernel**, likely store-dominated not compute-dominated (T3's `ccsd_t_1_dim16_f32` pattern) |
| `shallowK_256x24x256_c32` | 46.9 | 85.8 | 31.3% | 52.4% | 5.4% | 5.4% | 4.0% | 83.7% | compute-path; **store > microkernel** (same caveat) |
| `smallM_12x256x256` | 14.1 | 25.3 | 41.9% | 1.9% | 45.4% (noisy: 37.4-45.4% across `r1`/`r2`/`r3`, see reproducibility check above) | 4.8% | 4.3% | 43.9% | overhead: packing |
| `smallM_12x256x256_f32` | 22.2 | 102.3 | 35.0% | 3.3% | 47.8% | 6.2% | 5.2% | 38.3% | overhead: packing |
| `smallM_12x256x256_c64` | 19.2 | 16.8 | 61.8% | 3.3% | 28.6% | 2.1% | 3.3% | 65.1% | overhead: packing |
| `smallM_12x256x256_c32` | 20.7 | 59.1 | 63.1% | 3.7% | 27.0% | 1.4% | 3.0% | 66.9% | overhead: packing |
| `smallMN_16x256x16` | 3.2 | 63.9 | 17.0% | 0.6% | 42.0% | 9.7% | 22.5% | 17.6% | overhead: packing |
| `smallMN_16x256x16_f32` | 11.5 | 79.6 | 11.1% | 0.6% | 39.8% | 11.3% | 26.8% | 11.7% | overhead: packing |
| `smallMN_16x256x16_c64` | 16.4 | 38.2 | 43.6% | 3.3% | 15.6% | 2.5% | 23.4% | 46.9% | overhead: planning |
| `smallMN_16x256x16_c32` | 14.8 | 57.8 | 50.8% | 3.7% | 25.1% | 2.4% | 9.6% | 54.4% | overhead: packing |
| `smallN_256x256x12_f32` | 16.6 | 45.8 | 11.0% | 0.5% | 79.1% (noisy: 65.5-79.1% across `r1`/`r2`/`r3`, see reproducibility check above) | 3.0% | 4.7% | 11.4% | overhead: packing |
| `smallN_256x256x12_c64` | 32.2 | 43.7 | 38.7% | 4.7% | 46.2% | 2.4% | 5.5% | 43.5% | overhead: packing |
| `smallN_256x256x12_c32` | 51.5 | 72.4 | 28.5% | 6.6% | 51.4% | 3.1% | 7.7% | 35.1% | overhead: packing |
| `onem_256_c64` (`1m` kernel, direct) | 57.6 | - | 77.0% | 10.1% | 11.3% | 1.5% | 0.0% | 87.1% | compute-path |
| `onem_256_c32` | 110.0 | - | 64.9% | 16.0% | 17.5% | 1.6% | 0.0% | 80.9% | compute-path (borderline) |
| `onem_512_c64` | 57.0 | - | 71.8% | 11.4% | 15.5% | 1.2% | 0.0% | 83.2% | compute-path |
| `onem_512_c32` | 147.8 | - | 74.7% | 12.3% | 12.0% | 1.0% | 0.0% | 87.0% | compute-path |

**Caveat on all four `onem_*` rows**: these are `DIRECT_CASES` -- `plan_contract`
runs once, outside the profiled loop, so `planning`/`adapter/prepare` read
essentially 0% for every `onem_*` row by construction. Every `CASES` (adapter-
path) row above re-plans on every call (the `@tensor`/`tensorcontract!`
adapter calls `plan_contract` fresh each time), so its `planning` bucket is a
real per-call cost. Do not compare an `onem_*` row's combined
(microkernel+store) share, or its `planning`/`adapter/prepare` figures,
directly against an adapter-path row's as if they measured the same thing --
the `onem_*` numbers are structurally advantaged on exactly the buckets this
section calls "overhead". See also the dtype-ranking caveat on the `1m`
shape menu earlier in this section: `onem_256_c64`/`onem_512_c64` use the
*worst*-ranked `1m` shape for `ComplexF64` and `onem_256_c32`/`onem_512_c32`
use the *best*-ranked shape for `ComplexF32`, so the `c64`-vs-`c32` `onem_*`
comparison is doubly non-apples-to-apples (different plan-cost structure,
different relative position in each dtype's own shape ranking).

### Findings from the extended grid

- **Packing dominance is not a Float64-only or `MAIN_SHAPES`-only story.**
  Every `smallM`/`smallMN`/`smallN` case is `overhead: packing` at all four
  dtypes, **except one** -- `smallMN_16x256x16_c64` is `overhead: planning`
  (23.4% planning vs. 15.6% packing; `smallMN_16x256x16_c32` is
  `overhead: packing`, 25.1% packing vs. 9.6% planning, so this is *not* "the
  two `smallMN` complex cases" as an earlier draft of this section claimed --
  only the `ComplexF64` one is planning-dominated, correcting that count).
  `planning`'s one-time `plan_contract` cost is large relative to a tiny
  per-call workload in that one case. STATUS.md's headline
  `smallN_256x256x12` finding (T3: 62.7% packing) generalizes cleanly across
  dtype and across the other two small-shape families, not a fluke of one
  shape/dtype pair.
- **The share gap narrows with size, at every dtype, same as T3 found for
  Float64 alone.** `plain_64` -> `plain_512` combined share climbs
  monotonically within each dtype column on the clean `r2` data (Float64:
  36.9% -> 84.7%[^plain512-f64]; `f32`: 26.3% -> 77.3%; `c64`: 59.1% ->
  88.0%; `c32`: 49.8% -> 88.8%), and `big_1024x256x1024` (both real dtypes)
  clears 80% -- consistent with T3's "narrows with size rather than a clean
  cutoff" framing, now confirmed across the whole dtype range rather than
  just Float64/Float32. **Correction**: an earlier draft of this bullet, built
  from the contaminated `r1` run, reported a non-monotonic dip in the
  `ComplexF64` column (86.6% at `plain_256_c64` down to 84.6% at
  `plain_512_c64`). The clean `r2` re-run resolves this to a monotonic
  increase (87.2% -> 88.0%) -- the dip was a contamination artifact, not a
  real effect, and every dtype column is monotonic in the clean data.
  `Float32` is the one dtype where even `plain_512` stays just under the 80%
  share line (77.3% on the clean re-run, up from the contaminated run's
  75.9% but still under the line) -- worth a look, not chased further this
  pass; see follow-ups.
- **New store-dominated cases, same pattern as T3's `ccsd_t_1_dim16_f32`.**
  Both `ComplexF64`/`ComplexF32` variants of `shallowK_256x24x256` have
  `store` share exceeding `microkernel` share outright (43.5%/52.4% vs.
  42.6%/31.3%) -- clearing the 80% combined-share bar on `store` weight, not
  microkernel weight, exactly the pattern T3 flagged as a false
  "compute-bound" read without the throughput leg. `plain_256_c32` shows a
  milder version of the same thing (13.9% store vs. 70.1% microkernel --
  microkernel still leads by a wide margin, but store is a much larger
  fraction of the combined share than any real-dtype `plain_*` case at the
  same size). Neither is chased to a root cause this pass (no isolated
  complex-kernel store-path reference exists yet to confirm "kernel-stalled"
  the way T3 did for `ccsd_t_1_dim16`); flagged as a follow-up below.
- **`1m` (`OneMKernel`) looks broadly similar to the planar complex kernel's
  own share profile** (`onem_256_c64`/`onem_512_c64`/`onem_512_c32` clear
  80% combined share on microkernel weight, not store weight, unlike the
  `shallowK` complex cases above; `onem_256_c32` is the one borderline case
  at 80.9%, packing-heavy at 17.5%) -- no evidence from this pass that `1m`
  has a qualitatively different overhead profile than `PlanarMethod` at the
  one square-GEMM shape tested per size, but only two sizes were profiled,
  the direct-vs-adapter plan-cost asymmetry noted above applies, and no
  isolated `1m` reference exists to check throughput, so this is a weak,
  share-only observation.

[^plain512-f64]: `plain_512` (`Float64`) is a T3-carryover case, not repeated
in the table above; its clean `r2` combined share is 84.7% (microkernel
80.9% + store 3.7%), used here for the column endpoint.

### Follow-ups from this pass, not investigated further

- An isolated-reference throughput measurement for the complex
  (`PlanarMethod`/`OneMMethod`) kernel families, analogous to
  `bench_store_path.jl`'s real-dtype `PackedPanel` figures, so the new
  complex/`1m` rows above can be reclassified under T3's full two-legged
  rule instead of the share-only simplification used here.
- Root-cause the two `shallowK_256x24x256` complex store-dominated cases
  and `plain_256_c32`'s elevated store share (see "New store-dominated
  cases" above) -- is the vectorized store fast-path's eligibility guard
  narrower for complex dtypes on this shape family, similar to the
  Float32-specific question T3 already raised for `ccsd_t_1_dim16_f32`?
- `plain_512_f32` staying just under the 80% combined-share line while
  every other dtype's `plain_512` clears it -- confirmed on both `r1`
  (75.9%) and `r2` (77.3%), i.e. this is not run-to-run noise from the
  contamination episode, but it's still only two data points; worth a
  dedicated look at whether it's a real `Float32`-specific packing-cost
  effect at this size.
- `smallM_12x256x256` (real `Float64`) and `smallN_256x256x12_f32`: packing
  share did not settle after three repeats (see the reproducibility-check
  discussion above) -- worth either more reps or a look at whether these two
  specific (shape, dtype) pairs are unusually close to a cache/allocation
  boundary that makes packing cost bimodal, rather than assuming more
  repeats alone will converge it.

## Kernel-stalled/store-dominated fix: run-length-aware demotion (F2) and inlined `accumulate` (F1) (2026-09-21)

The prior pass (immediately above) found, but did not fix, two mechanisms
behind `ccsd_t_1_dim16`/`ccsd_t_1_dim16_f32` being store-dominated/kernel-
stalled. This pass implements both, in `src/`, each verified with real
before/after numbers on `ccqlin038`.

### F2: run-length-aware kernel-shape demotion (`src/driver.jl`)

**Mechanism.** `QuasiStridedBackend`'s vectorized store
(`_store_tile_vector!`) requires EVERY register sliver of a macro block to be
unit-stride in `C`. Given a composite M-axis with a leading unit-stride run
of length `run` and a kernel's register-tile height `mr`, this holds iff
`Qm == run || run % mr == 0` -- confirmed by direct counterexample sweep
(`benchmark/probes/probe_ccsd_t_stall_f2rule.jl`): the weaker-looking
`mr <= run` is WRONG (e.g. `run=20, mr=16` satisfies it but only 40% of
slivers are actually contiguous). `ccsd_t_1_dim16_f32`'s shipped default
kernel is `(32,6,16)` (`mr=32`), while C's leading run there is only 16:
`16 % 32 != 0`, so every M-sliver falls to the slow scattered store path,
which measured at ~75% of total time in the prior pass's profile.

**Fix.** In `plan_contract` (`src/driver.jl`), a new `_demote_for_run(T,
kernel, run, Qm)` helper runs DOWNSTREAM of the existing `_default_kernel`
call and the `_prefer_swap` M/N-orientation decision, keyed on whichever
orientation was actually chosen to feed M (its own run length against the
kernel it would actually run). If the predicate fails, it searches that
dtype's `kernel_shapes(T)` menu for shapes whose `mr` divides `run`, and
picks the LARGEST matching `mr` -- not the smallest: measured directly, at
`run=16` for Float32 the `(16,6,8)` shape beats `(8,6,8)`. Both demotion
targets are already-compiled menu entries (`_kernel_from_shape`), so this
adds no new `SIMDKernel` specialization. Real dtypes only (`T <: Real`); the
`T` fallback method is a no-op, matching the complex path's unconditional
scattered store. ~30 lines in `src/driver.jl` (`_demote_for_run` plus the two
call sites after the swap decision).

**Pinning-test interaction, checked not assumed.** `test/test_driver.jl`'s
swap-decision pinning test (`"label order: pinning test on the ccsd_t
shapes..."`) reads `MRk = mr(plan.kernel)` AFTER `plan_contract` returns --
i.e. from the (possibly F2-demoted) final kernel -- and compares it against
the swap boolean, which was decided using the PRE-demotion `mr`. This is
exactly the fragility flagged as a risk before implementation. Checked by
running the suite: at that test's fixture (`d = 5`), the leading run is
never more than `d = 5` or `d^2 = 25`, and the smallest real menu `mr` is 8
(Float64) -- `5 % 8 != 0` and `25 % 8 != 0` for every menu entry -- so
`_demote_for_run` always finds no matching shape and returns the kernel
unchanged; the assertion never actually observes a post-demotion `mr`. Full
suite green with no edit needed to that test. Left as-is rather than
"fixed proactively", since forcing a change into a passing, correctly-reasoned
test would have been fixing a problem that measurement showed does not exist.

**`_default_kernel`/`test_target.jl` pinning.** F2 is a new function called
after `_default_kernel`'s result is already resolved, never folded into it;
`_default_kernel(T, 9, 8)`'s `===`-pinned identity in `test/test_driver.jl`
and `test/test_target.jl`'s Qm-based demotion tests are unaffected by
construction (neither exercises `plan_contract`'s post-swap step).

**New correctness test** (`test/test_driver.jl`, `"F2: run-length-aware
kernel demotion"`): for the `ccsd_t_1` fixture at `dim=16` (both dtypes),
asserts `mr(plan.kernel)` resolves to the expected value (16 for Float32,
demoted from 32; 16 for Float64, unchanged since the default already
satisfies the predicate), and that both `execute!` and `execute_tilewise!`
agree with an engine-free reference loop. A second loop over a plain-GEMM
fixture asserts F2 never fires there (`plan.kernel === _default_kernel(T, Ma,
Na)`, by identity) since `Qm == run` always holds for a bare matmul's M
composite.

**Measured (ccqlin038, `benchmark/bench_ccsd_t_store.jl --dims 8,16
--dtypes Float64,Float32`, reps=15, ABBA-adjacent -- same machine, back to
back, ~1-3 load-average noise from concurrent orchestration work noted
below):**

| case | dim | dtype | before (Arm 1-QuasiStrided) | after | speedup |
|---|---|---|---|---|---|
| ccsd_t_1 | 16 | Float32 | 3.522e-2 s | 1.626e-2 s | **2.17x** |
| ccsd_t_1 | 8  | Float32 | 5.086e-4 s | 2.649e-4 s | **1.92x** |
| ccsd_t_1 | 16 | Float64 | 2.788e-2 s | 2.387e-2 s (F2+F1 combined) | 1.17x (F2 does not fire here; see F1 below for the isolated attribution) |
| ccsd_t_2/3/4 | 8,16 | both | -- | -- | within noise / small F1-driven gains, no regression |

Both within the predicted 1.4x-2.2x range for the cases F2 targets, and no
regression on cases where the default kernel already satisfies the
predicate (`ccsd_t_1` at Float64, and plain GEMM in general, per the ABBA
guard below).

**Re-profile** (`julia -t 1 --project=benchmark benchmark/profile_to_suite.jl
ccsd_t_1_dim16 ccsd_t_1_dim16_f32 --tag A2-post-f2`, F2 only, F1 not yet
applied): `ccsd_t_1_dim16_f32`'s `store` bucket share dropped from ~75% (prior
pass's finding) to **21.73%** of `QuasiStridedBackend`'s own samples, with
`microkernel` rising to 57.47%. Artefact:
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/buckets_summary-A2-post-f2.txt`.

### F1: inline `Base.accumulate` (`src/kernels/simd.jl`)

**Mechanism.** `Base.accumulate(kernel::SIMDKernel{MR,NR,T,W}, ...)` built and
returned its 768-byte (at the shipped `(16,6,8)`/`(32,6,16)` shapes)
accumulator tuple without an `@inline` annotation, so LLVM round-tripped it
through memory (memset + stack allocation + store-and-reload) at every
micro-tile call instead of keeping it register-resident, per the prior
pass's LLVM/native codegen dump.

**Fix.** One line: `function Base.accumulate(...)` -> `@inline function
Base.accumulate(...)` in `src/kernels/simd.jl`. No semantic change --
`accumulate` already delegated every K-step to the `@generated`,
already-`@inline`d `_accumulate_step`; only the outer wrapper's own inlining
status changed.

**Allocation-cliff check.** `test/test_simd_kernel.jl`'s "allocation:
`accumulate` and `execute_tile!` are steady-state allocation-free" and "...
WITH TAIL ROWS ..." testsets already sweep register shapes up to `NV = 24`
(`(16,6,4)`) and `NV = 28` (`(16,7,4)`) -- the documented dynamic-tuple-
indexing allocation cliff's range -- and both stayed allocation-free with
`@inline` added; confirmed by the full suite passing (see below), not by a
separate ad hoc run, since these tests are exactly the standing regression
guard for this cliff.

**Measured, isolated from F2** (driver.jl at its pre-F2 baseline, only
`simd.jl`'s `@inline` applied, `ccsd_t_1` dim=16 Float64 -- F2 never fires
on this case, so its effect is F1 alone):

| | median (15 reps) | vs. no-F1 baseline |
|---|---|---|
| Arm 1 (QuasiStrided), no F1 | 2.788e-2 s | -- |
| Arm 1 (QuasiStrided), F1 only | 2.561e-2 s | **1.089x (+8.9%)** |

Matches the predicted +5-10% for a small-`kc` case. On a large-`kc` case
(plain 512x512x512 GEMM, `kc=256` forced, Float64, `median_time_s`, 15 reps,
via a scratch harness script calling `plan_contract`/`execute!` directly):
F1-off 4.4897e-3 s, F1-on 4.3389e-3 s -- **+3.4%, no regression**, consistent
with "negligible change" (F1's win is proportional to per-call overhead,
which large-`kc` amortizes away).

**Re-profile** (same command, `--tag A2-post-f1`, both fixes applied):
`ccsd_t_1_dim16` (Float64) end-to-end time dropped from 0.029243 s/call (F2
only) to 0.026680 s/call (F2+F1), a further 8.8% -- consistent with the
isolated measurement above. `profile_buckets.jl`'s bucket table folds
`zero_accumulator` into the same `microkernel` bucket as `_accumulate_step`
(see its `BUCKETS` table), so the ~29-30% "zero_accumulator share" figure
from the prior pass's finer-grained analysis is not separately visible in
this profiler's coarser bucket; the wall-clock improvement above is the
figure of record for this pass. Artefact:
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/buckets_summary-A2-post-f1.txt`.

### Full-suite and ABBA guard results

- `Pkg.test()`: 35180/35180 pass (was 35170 before this pass's +10 new F2
  tests), including `test/test_driver.jl`'s and `test/test_target.jl`'s
  pinning tests, unedited.
- `benchmark/bench_real_path_guard.jl`, ABBA order (`new(A1)`, `base(B1)`,
  `base(B2)`, `new(A2)`, `MAIN_SHAPES`+`SMALL_SHAPES`+scattered, both
  dtypes, 21 reps): geomean `base/new` = 0.962x, i.e. the fixed tree is
  **~3.8% faster on plain-GEMM shapes on average**, no shape showing a
  systematic one-sided regression (per-shape ratios ranged 0.96x-1.13x,
  consistent with the ~10-14% canary spread the guard itself flagged --
  the machine was not fully quiet, other orchestration work was running
  concurrently on `ccqlin038` during this pass; see caveat below).
  Artefacts: `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/
  real_path_guard_{newf2f1A_run{1,2},baseB_run{1,2}}.csv`.

**Machine-load caveat.** `uptime` load average rose from ~1.4-1.9 to ~2.2-3.2
over the course of this pass's measurements (other worktrees' concurrent
benchmarking, per the shared-machine note in this task's brief). The canary
spread on two of the four guard runs exceeded the guard's own 10% quiet-
machine threshold. The ccsd_t_1_dim16 F2 speedup (~2x) and the F1 isolated
+8.9% are both far larger than that noise band and are trusted; the ABBA
guard's ~3.8% aggregate improvement is closer to the noise floor and should
be read as "no regression, and probably a small real win" rather than a
tight number -- re-run on a quiet machine for a tighter figure if that
matters later.

### Artefacts

- `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/
  bench_ccsd_t_store_*.csv`/`summary_*` (baseline-pre-f2, after-f2-f1 tags).
- `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/
  real_path_guard_{newf2f1A,baseB}_run{1,2}.csv`.
- `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-21/profiles/
  buckets_summary-A2-post-f2.txt`, `buckets_summary-A2-post-f1.txt`.
- `benchmark/probes/probe_ccsd_t_stall_f2rule.jl` (prior pass, reused as the
  predicate's correctness evidence for F2, unchanged this pass).

### Gate

Both fixes are small (F2 ~30 lines in one file, F1 one line in one file),
each has a measured before/after matching or exceeding its prediction, the
full suite is green including both named pinning tests, and the ABBA guard
shows no regression. Committed to the `ccsd-t-stall` branch.

### Post-review fixes (2026-09-21, same day)

An independent review of the first commit (`a0b337a`) found one blocking
issue and three should-fix items, addressed as follows.

**Blocking, fixed: the F2 correctness test was ISA-specific.** The original
`test/test_driver.jl` F2 testset hardcoded `expect_mr = 16` for BOTH
Float64 and Float32 on the `ccsd_t_1` dim=16 fixture -- true only on this
machine's `:avx512` profile. On `:avx2` (`_derived_shape` gives Float64
`(8,6,4)`, and `16 % 8 == 0` so F2 correctly no-ops there, leaving `mr = 8`,
not 16) and on an unrecognized/NEON-like ISA (`_legacy_shape` gives `(8,6,4)`
for both dtypes, `mr = 8` for both), the hardcoded `16` is wrong -- exactly
the pattern this file's own header comment on `"plan_contract: SIMDKernel is
the engine-wide default kernel"` already warns against ("held here only
because it happens to trigger on x86, and broke on aarch64"), and exactly
what `test/forced_isa_runner.jl` exists to catch. Confirmed by running it
before the fix:

```
QS_FAKE_ISA=avx2 QS_FAKE_VB=32 QS_FAKE_NREG=16 julia --project=. test/forced_isa_runner.jl
```

failed 2 in the F2 testset (beyond the harness's own documented 1-failure
residue in `test_target.jl`). **Fix**: the testset no longer asserts a
literal `mr`. It computes the expected outcome from `_default_kernel`/
`kernel_shapes(T)` themselves: if the shipped default's own `mr` already
satisfies `Qm == run || run % mr == 0`, `plan.kernel` must equal the
default, unchanged, on every ISA; otherwise it asserts `run % mr(plan.kernel)
== 0` and that `mr(plan.kernel)` is the LARGEST entry in `kernel_shapes(T)`
satisfying that (or, if no entry does, that the kernel is left unchanged,
mirroring the `d=5` pinning fixture's own finding). Re-run after the fix:
`QS_FAKE_ISA=avx2 QS_FAKE_VB=32 QS_FAKE_NREG=16` and
`QS_FAKE_ISA=unknown QS_FAKE_VB=0 QS_FAKE_NREG=0`, both give exactly the
harness's documented 1-failure/1-error residue (`test_target.jl`'s "runs on
this host without throwing", which compares a fresh `_detect_target()`
against the forced profile and fails by construction under
`forced_isa_runner.jl`) and the F2 testset itself passes 12/12 on both. Full
suite on the real (`:avx512`) host: 35183/35183.

**S3, fixed: the test fixture was needlessly large.** `_leading_unit_run`
only reads the run-forming label's (`a`'s) own extent and that the NEXT M
label's C-stride differs from the running total -- it does not depend on any
other axis's size. The fixture now keeps `a`'s extent at 16 (the register-
tile-sized run under test) and shrinks the other six axes (`i,j,m,k,b,c`) to
4, reproducing the identical `run`/`Qm`/predicate outcome (verified: `run =
16` and the swap-avoidance argument both hold independent of the other axes'
sizes, since `nrun = 1` there regardless) at roughly `(4/16)^6 ~ 1/4000` the
array/reference-loop cost. Testset wall time dropped from ~52s to ~4s in the
full-suite run.

**S1, investigated, found to be a REAL regression, documented as a known
limitation, not fixed.** F2 has no cost model: every demotion in the real
menus (`KERNEL_SHAPES_F64`/`KERNEL_SHAPES_F32`) also narrows the SIMD lane
width `W` (e.g. Float64 `(16,6,8) -> (8,6,4)` halves `W` from 8 to 4), and
that cost is paid on every K-step regardless of how large `kc`/`Qk` is, while
the store-path saving F2 is chasing is a fixed per-macro-block cost that
`kc` does NOT amortize away. None of this pass's own verification exercised
a case where F2 fires AND K is large -- the ABBA guard is plain-GEMM shapes
only (`Qm == run` always there, so F2 never fires), and the "+3.4% on
512^3" datapoint is F1-only (F2 does not fire on that shape either, for the
same reason).

Measured directly, per the review's request: `C[a,b,c,i,j,k] = A[i,j,m,a] *
B[m,k,b,c]`, Float64, `a = 8` (a divisor of the default `mr = 16` but not
equal to it -- `run = 8`, `Qm = 8*6*6 != 8`, so F2 fires and demotes to
`(8,6,4)`), `i=j=k=b=c=6`, `m = 512` (the contracted extent, made large so
this is compute-bound-ish). Comparing the auto-resolved (F2-demoted) plan
against the SAME contraction with an explicitly named, non-demoted
`SIMDKernel(Val(16),Val(6),Float64,Val(8))` (naming a kernel bypasses F2
entirely, per its own "only for an auto-selected kernel" guard):

| | kernel (mr,nr,W) | median (15 reps) |
|---|---|---|
| auto (F2 fires) | (8,6,4) | 1.735e-3 s, 1.750e-3 s (2 runs) |
| forced (no demotion) | (16,6,8) | 1.415e-3 s, 1.402e-3 s (2 runs) |

F2's demotion is **~18-25% SLOWER** here than not demoting would have been
-- the narrower-lane compute cost over `m = 512` K-steps outweighs the
scattered-store saving on this shape's much smaller M/N extents. This is a
genuine, reproducible (two back-to-back runs, same ratio within 2%)
regression risk for F2 as shipped: it is a pure store-path-share heuristic
with no awareness of `Qk`/`kc`, and can make the wrong call whenever a
shape's contracted extent is large relative to its M/N extents. **Not fixed
in this pass** -- a correct fix needs a K-aware (or straight cost-model)
check before demoting, which is a real design change to `_demote_for_run`,
out of scope for a same-day post-review patch. Documented here as the
known limitation; any future work on F2 should gate the demotion on `Qk`
being small relative to `Qm*Qn`, or on a direct cost estimate, before
trusting it unconditionally on a new case class.

**S2, investigated, confirmed not currently wrong, documented as a known
limitation, not fixed.** The M/N orientation swap (`_prefer_swap`) and F2
are sequenced, not jointly optimized: the swap decision is made first,
using each orientation's PRE-demotion `mr`, and only the chosen orientation
is then offered to F2. This means a theoretical case exists -- e.g. as-is
run = 4, swapped run = 16, Float32 default `mr = 32` -- where swapping
first (to the run-16 orientation) would let F2 achieve full vectorization
at a large `mr`, but the current order never tries that combination if the
as-is orientation's own run already loses the swap comparison for an
unrelated reason. Verified this is a missed-optimization, not a
correctness bug: every `execute!`/`execute_tilewise!` agreement check in
this pass's own tests and the pre-existing suite passes, so whichever
orientation+kernel combination is chosen still produces the right answer,
just not necessarily the fastest available one. Not fixed -- jointly
optimizing the swap and the demotion is a larger design change (the swap
decision would need to be re-run per candidate kernel shape, not just per
orientation) than this pass's scope.

**Cheap improvement applied: `benchmark/harness.jl`'s `git_commit()` now
flags a dirty working tree** (appends `-dirty` to the SHA when `git status
--porcelain` is non-empty), so a profile/benchmark artefact's provenance
header no longer silently shows a clean commit hash while measuring
uncommitted changes -- exactly what happened to this pass's own `A2-post-f2`/
`A2-post-f1` profile artefacts (both were taken before commit `a0b337a`
existed).

Second commit: see this file's own git history / `STATUS.md` for the SHA
this section's own changes landed under.

## `bench_to_suite.jl`: wiring up `:mps`/`:ctmrg`/`:trg`, smoke-tested (2026-09-21)

Per the dispatch-tiers review (section 5.1), extended
`benchmark/bench_to_suite.jl` to also run the upstream suite's `:mps`,
`:ctmrg`, `:trg` categories, additively -- `:pairwise`/`:tccg` behavior and
CLI flags are unchanged. New flags: `--mps-bonddims` (MPS/MPO
effective-Hamiltonian bond `D`, default `32,64,128`), `--ctmrg-chis` (CTMRG
environment bond `chi`, default `16,32,64`), `--trg-chis` (TRG plaquette
bond `chi`, default `16,32,48`). No `src/` changes; scope confirmed via
`git diff --stat` (`benchmark/bench_to_suite.jl` only).

**The three new categories are `NetworkSpec` cases (multi-tensor `ncon`
networks), not `ContractSpec` (two-tensor `tensorcontract!`) like
`:pairwise`/`:tccg`.** The script's `build_case`/`run_case!`/`alloc_output`/
`case_bytes` were given `NetworkSpec` methods that call
`TensorOperations.ncon(tensors, indexlists, conjlist; order, output,
backend)` directly (mirroring what `TensorOperationsBenchmarks`'s own
`execute(::NetworkSpec, ...)` does), rather than routing through
`build_suite`/`BenchmarkTools`. `ncon` has no in-place variant, so timing for
these three categories includes output allocation -- this is upstream's own
accepted discipline for `NetworkSpec` (see `TensorOperationsBenchmarks/src/
lowering.jl`'s header comment), not a gap introduced here. A small generic
helper, `case_sweepparam`, was added so the CSV/summary's single "sweep
parameter" column reads `case.params.dim` for `:pairwise`/`:tccg`, `.D` for
`:mps`, `.chi` for `:ctmrg`/`:trg`, without renaming pairwise/tccg's own
`dim` field. `plot_bench_to_suite.jl` needed **no changes** -- it already
groups purely by the CSV's `category` string column and treats `dim` as a
generic sweep value, so it plots the new categories unmodified (verified: ran
it against an `:mps` result CSV, got correct per-dtype PNGs).

### Smoke test: QuasiStrided runs cleanly on all three, no rejections or mismatches

Ran each new category standalone at minimal size/reps (`ccqlin038`, load
average ~3.3-3.9 at the time, single-core measurement discipline
unaffected):

```
julia --project=benchmark benchmark/bench_to_suite.jl --categories mps   --mps-bonddims 8  --reps 3
julia --project=benchmark benchmark/bench_to_suite.jl --categories ctmrg --ctmrg-chis 8     --reps 3
julia --project=benchmark benchmark/bench_to_suite.jl --categories trg   --trg-chis 8       --reps 3
```

All three: **zero correctness mismatches, zero backend rejections/errors**,
CSV/summary/mismatches/PROVENANCE files written correctly, `QuasiStridedBackend`
matched `StridedBLAS` to `rtol` on every case (`1e-10` Float64, `1e-5`
Float32) -- confirmed both via the script's own mismatch gate and by an
independent standalone `ncon(...; backend=QuasiStridedBackend())` vs.
`backend=StridedBLAS()` comparison run directly in the REPL before touching
the script, for `_trg_case(4)`, `_mps_1site_case(8)`, `_mps_2site_case(8)`,
`_ctmrg_case(8)`. `QuasiStridedBackend` does **not** reject any of these
networks -- every edge of every network in all three categories is a plain
two-tensor pairwise contraction once `ncon` decomposes it (no `tensoradd!`/
`tensortrace!` ever required), which is exactly what `QuasiStridedBackend`
already supports; nothing here exercises `tensoradd!`/`tensortrace!` the way
`:permute`/`:trace` would.

Quick-signal throughput from the smoke runs (`dim`/reps too small to be a
result, just orientation): at `D=8`/`chi=8`, QuasiStrided ran ~1.05x-1.25x
slower than `StridedBLAS` across all three categories (`mps` 1.17-1.25x,
`ctmrg` 1.17-1.18x, `trg` 1.03-1.05x) -- consistent with the
`:pairwise`/`:tccg` milestone's small-shape overhead findings, nothing new.

### Finding: `StridedNative` (not QuasiStrided) has a severe, size-growing pathology on `:ctmrg`/`:trg` `ncon` networks

Not a QuasiStrided result, but load-bearing for planning the real
evidence-gate run's walltime budget, so recorded here. Isolated
single-call timings (`ncon` directly, warm-up discarded, `@elapsed`, no
harness overhead), Float64:

| category/case | BLAS | QuasiStrided | StridedNative |
|---|---|---|---|
| ctmrg chi=8  | 8.20 GFLOP/s | 7.01 GFLOP/s | 0.47 GFLOP/s (17.6x slower) |
| ctmrg chi=64 | 15.15 GFLOP/s | -- | 0.54 GFLOP/s (28x slower) |
| trg chi=8    | 4.08 GFLOP/s | 3.87 GFLOP/s | 1.37 GFLOP/s (3.0x slower) |
| trg chi=32   | 20.03 GFLOP/s | 14.41 GFLOP/s | 0.48 GFLOP/s (42x slower) |
| trg chi=48   | 36.54 GFLOP/s | 23.32 GFLOP/s | **did not finish in 100s** (single call) |

`ctmrg`'s `StridedNative` penalty is a roughly *constant* ~0.5 GFLOP/s
regardless of `chi` (a fixed per-call inefficiency, not a scaling blow-up),
so `:ctmrg`'s total real-run cost stays trivial (sub-minute) even including
`StridedNative`. **`:trg` is different: `StridedNative`'s penalty grows with
`chi`** (3x at chi=8, 42x at chi=32, apparently super-linear), and at
chi=48 a single `ncon` call under `StridedNative` did not complete within
100 seconds (backtrace shows time spent inside `Strided.jl`'s
`_mapreduce_kernel!`/`stridedtensorcontract!`, i.e. genuinely inside
`Strided.jl`'s own contraction path for this network's index/stride
pattern, not inside `QuasiStrided` or this script). Upstream's own default
`:trg` sweep goes to `chi=96` (`flops(chi=96) / flops(chi=48) = 64x`), so a
naive full sweep with `StridedNative` included at upstream's default sizes
risks **hours-to-indefinite** wall-clock for the `:trg` category alone.
**Recommendation for the real run**: either cap `--trg-chis` well below
upstream's default ceiling (`16,24,32` is safely fast; `48` is already
borderline; do not include `64`/`96` unless `StridedNative` is dropped from
that category or run under its own generous timeout), or run `:trg` at
larger `chi` with only `StridedBLAS`/`QuasiStrided` (this script always runs
all three backends together per case, so excluding `StridedNative` for
`:trg` specifically would need a small script change, not attempted here --
out of scope for this pass, which is tooling-plus-smoke-test only).

### Recommended command for the real (Slurm) evidence-gate run

Full default-size run, all five categories, `--reps 21` (this pass did
**not** run this -- it is the campaign handed to the user to run on Slurm):

```
julia --project=benchmark benchmark/bench_to_suite.jl \
    --categories pairwise,tccg,mps,ctmrg,trg \
    --pairwise-sizes 15,63,128 --tccg-sizes 8,16 \
    --mps-bonddims 32,48,64,100,128,256,300,512 \
    --ctmrg-chis 16,24,32,48,64,100 \
    --trg-chis 16,24,32,48 \
    --reps 21
```

Note `--trg-chis` above is **capped at 48**, deliberately narrower than
upstream's own default `16,24,32,48,64,96` sweep, per the `StridedNative`
finding above -- go beyond 48 only with a plan for `StridedNative`'s
non-termination risk at `chi>=64`.

Rough walltime budget (single core, per the project's pinned measurement
discipline; `:pairwise`/`:tccg` numbers are this project's pre-existing
experience, not re-measured here): `:pairwise`/`:tccg` together, a few
minutes (unchanged from the existing milestone); `:mps` (16 cases x 2
dtypes x 3 backends x ~22 evaluations, `BLAS`/`QuasiStrided`-like
throughput throughout, no `StridedNative` pathology observed here), roughly
**5 minutes**; `:ctmrg` (6 cases x 2 dtypes, `StridedNative`'s fixed ~0.5
GFLOP/s penalty included), well under **2 minutes**; `:trg` capped at
`chi<=48` (4 cases x 2 dtypes), dominated by the chi=48 `StridedNative`
calls -- budget **15-30 minutes** conservatively for `:trg` alone, since
`StridedNative`'s chi=48 single-call cost was not fully characterized (only
bounded below at >100s for one un-warmed call; the harness's own warm-up
plus 21 timed reps at chi=48 could be substantially longer). **Total
suggested Slurm walltime request: 1 hour**, to leave headroom for the
`:trg`/`StridedNative` uncertainty and machine-load variance. Memory: every
tensor across all five categories at these sizes is well under 256 MiB
(upstream's own `within_memory_budget`/`MAX_CASE_BYTES` ceiling, `2^28`
bytes, already filters anything bigger) -- **a few GB is generous**; nothing
here approaches this project's usual single-workstation memory budget.

Commit: see this file's own git history / `STATUS.md` for the SHA this
section's own changes landed under.

## Per-call floor: cheaper planning, once-per-block bounds validation, closed-form affine blocks (2026-09-21)

Follow-up track authorized off `docs/proposals/dispatch-tiers.md` section 5.2,
which recommended *against* Octavian-style dispatch tiers and pointed at the
per-call/per-block floor instead. Three items were named there; all three are
implemented here. Measured on `ccqlin038` (Cascade Lake, AVX-512), Julia
1.13.0, single-threaded, load average 2.5-3.1 throughout (32 cores; the
machine was shared but quiet -- canary spreads 0.5-8.4%, see below).

### Where the time actually was, before

Profiler baseline, taken on a `git archive` extract of the base commit
`20e1002` so it is a genuine same-day A/B rather than a recollection
(`benchmark/profile_to_suite.jl ... --tag basetree-pre`):

| case | s/call | GFLOP/s | planning | driver_loop | gc/alloc | alloc/call |
|---|---|---|---|---|---|---|
| `ao2mo_2_dim16` | 115.4 us | 18.18 | 7.08% | 29.83% | 4.62% | 6080 B |
| `plain_64` | 16.34 us | 32.08 | 23.51% | 10.95% | 6.96% | 4368 B |
| `smallMN_16x256x16` | 10.09 us | 12.99 | 33.88% | 11.67% | 11.86% | 4368 B |

**Data-quality note, recorded because it nearly produced a false headline.**
The first "pre" run in this pass (`--tag percallfloor-pre`, on the working
tree before any edit) reported `smallMN_16x256x16` at **48.6 us / 2.70
GFLOP/s**. That figure is an outlier and is not used anywhere below: the base
tree re-measured 10.09 us minutes later, the ABBA guard independently puts
base `execute!` alone at 5.9 us for that shape, and the same run's
`ao2mo_2_dim16` (17.73) and `plain_64` (31.10) agree with the base tree's
(18.18, 32.08) to within 3%. Taking the outlier at face value would have
claimed an 8x win where the real one is 1.67x.

### 1. `plan_contract`: 4.1-6.4 us and 5.0-7.0 KB per call, now 1.0-2.3 us and 1.8-2.6 KB

Three changes in `src/driver.jl`, all behaviour-preserving:

* **`_classify_labels` no longer builds three `Set`s.** The label tuples'
  LENGTH is a compile-time constant (the `NA`/`NB`/`NC` type parameters -- one
  specialization per arity), even though their contents are not, so membership
  is a `_label_in(lbl, tuple)` scan that unrolls into integer compares. The
  three output `Vector`s are sized once to their `NA`/`NB` worst case and
  `resize!`d down instead of `push!`ed into.
* **`_build_pair_group` no longer calls `ntuple(f, ::Int)`.** This was the
  single largest item: 656 B and 0.75 us per group *at `D == 1`*, three groups
  per plan. `D = length(labels)` is genuinely a runtime value (which labels are
  shared is a property of the label *values*), so the old `ntuple(f, D)` calls
  inferred as `Tuple{Vararg{Int}}`, heap-boxed, and every element was then read
  back through a dynamic `getindex`. But `D <= N1` and `N1` *is* static, so the
  new `_pair_group_rank(Val(N1), ...)` ladder resolves the rank in at most
  `N1 + 1` compile-time-dispatched compares and `_pair_group_static(Val{D})`
  then runs with statically sized tuples throughout: **656 B / 0.75 us -> 32 B
  / 0.06 us**. `Base.strides(v1)` is also hoisted out of the per-dimension
  closures, where it was rebuilt once per `d`.
* **`_order_free_labels` replaces `sortperm` + permuted copy with an in-place
  stable insertion sort of a single copy.** These lists have at most `ndims(C)`
  entries, so `O(n^2)` with `n <= 6` is free and three `Vector` allocations
  become one. A fresh vector is still returned: sorting `labels` in place would
  mutate `_classify_labels`'s output, which `test/test_driver.jl`'s label-order
  pinning reads afterwards.

Measured end to end (`plan_contract` with a warm reused workspace):

| case | before | after |
|---|---|---|
| `plain_64` | 4976 B, 5.46 us | 1840 B, 2.31 us |
| `smallMN_16x256x16` | 4976 B, 4.13 us | 1840 B, 1.39 us |
| `ao2mo_2_dim16` | 6784 B, 6.36 us | 2176 B, 0.96 us |
| `ccsd_t_1` (dim 4) | 7040 B, 6.37 us | 2560 B, 1.56 us |

**Not made allocation-free, and it cannot be** by this route. Each composite's
RANK is a value property of the label sets, so the three `AxisGroup`s and the
`ContractPlan` built from them are type-unstable until the `_plan_contract`
function barrier; the residual ~1.8 KB is the three label `Vector`s, the boxed
groups, and that barrier's return. Removing it needs the label tuples to carry
their classification in their types, which is a `@tensor`-macro-level change,
not a driver one. `test/test_per_call_floor.jl` pins a 3 KB ceiling (against
the old 5-7 KB) rather than zero, so a regression to the `Set` / runtime-
`ntuple` construction is caught without pretending the floor is zero.

### 2. Storage-bounds validation hoisted to once per macro block

Authorized explicitly, including the change to `pack_a!`'s documented "all
validation happens before any write" contract, on condition that any entry
point that skips a check a caller would expect is named `unsafe_*`. That
condition is met: the new entry points are `unsafe_pack_a!`, `unsafe_pack_b!`
(`src/packing.jl`), `unsafe_execute_tile!` (`src/kernel.jl`) and
`unsafe_execute_micro_tile!` (`src/driver.jl`), and they are spelled out at
every call site -- `_pack_sliver!(unsafe_pack_b!, ...)`, not a boolean flag.
`pack_a!`/`pack_b!`/`execute_tile!` and all four kernels' `execute_tile!`
bodies are **unchanged**; the only edit to the checked path is that the check
now sits behind a `Val`-typed compile-time constant (`BOUNDS &&
checked_tile_storage_bounds(...)`), which folds to the same code at
`Val(true)`.

**What moved, and only what moved.** The `unsafe_*` forms still validate the
row/column extent against the kernel's `(MR, NR)`, `kc >= 0`, both packed
buffer capacities, the packed eltype, and the empty / `kc == 0` / `alpha == 0`
short-circuits. Exactly one thing is skipped: the address-range check against
`length(storage)`.

**The safety argument, which is an equivalence and not an approximation.**
`_execute_nest!` now makes three `checked_span_bounds` calls, and each is
provably the conjunction of the per-sliver / per-tile checks it replaces:

1. *The check is exact for a rectangular region, not conservative.*
   `checked_tile_storage_bounds` never enumerates addresses; it computes
   `[base + rlo + clo, base + rhi + chi]` and requires that interval inside
   `[0, len-1]`. For a region that is a full row-set x column-set product --
   which every tile and every macro block here is -- `rlo + clo` and
   `rhi + chi` are both *realized* addresses (take the argmin row with the
   argmin column, and the argmax with the argmax), so the interval's endpoints
   are attained and the test is accept-iff-all-addresses-in-bounds.
   `checked_span_bounds` is that arithmetic, factored out and fed ranges
   instead of axes.
2. *The block's offset set is exactly the union of its slivers'.*
   `_classify_slivers!` partitions `buf[0:blocklen)` into slivers at
   `sfirst = s*reg_tile`, `scount = min(reg_tile, blocklen - sfirst)`; every
   buffer entry belongs to exactly one sliver, and no sliver is empty
   (`s < cld(blocklen, reg_tile)` forces `scount >= 1`). So
   `min_s(min(rows_s)) = min(rows_block)` and likewise for max.
3. *The other axis is shared, or likewise a union.* For A, every M-sliver is
   paired with the same K axis `colsA_k`; for B, every N-sliver with the same
   `rowsB_k`. For C the micro-tile loop is the **full cross product** of the
   M-sliver rows and N-sliver columns, so the union over tiles is exactly
   (block rows) x (block columns).

   Combining 1-3: block-check-passes <=> every-sliver-check-passes. Not a tight
   superset -- an equality. `test/test_per_call_floor.jl` asserts it as a
   randomized property over 400 cases with positive/negative/zero strides and
   scattered axes, comparing the two decisions directly.
4. *Ordering is preserved.* Each hoisted check precedes every read of, or
   write to, the operand it guards, in the same relative order as before
   (B before A before C). A hoisted check that throws may leave the workspace
   packed, which is unobservable; nothing has been written to `C`.
5. *`execute_tilewise!` is untouched* and still checks per tile, so the
   package keeps an independent oracle for both the values and the rejections.

Ranges are produced without an extra pass: `_classify_slivers!` accumulates the
block range from each sliver descriptor as it classifies it (`O(1)` per regular
sliver via the new `descriptor_offset_range`; for an irregular one, exactly the
scan the per-sliver check used to do anyway).

Tested, not assumed: a `CountingStorage <: AbstractVector` whose `length`
increments a counter shows `execute!` asking each of A/B/C for its length
**exactly once** per call on a 40x7x30 plan with 20+ micro-tiles, while
`execute_tilewise!` on the same plan asks `>= ntiles` times. Rejection is
tested through six deliberately-broken plans (destination one element short,
each source one element short, and negative bases on all three), each of which
must still raise `BoundsError` with nothing written -- and the destination case
is cross-checked against `execute_tilewise!` raising too.

### 3. Closed-form block description for affine-ramp composites

New `affine_ramp(g::AxisGroup{D,P})` in `src/axis_group.jl` answers whether
every map of `g` satisfies `offsets(g, q)[p] == q * step[p]` over the whole
domain -- exactly the condition under which `normalize_group` would fold `g` to
rank `<= 1`, tested with the same `Int128` comparison so a non-representable
product is "not a ramp" rather than a wrapped accidental match. Singleton
dimensions are skipped (their coordinate never advances), rank zero is a ramp
with step 0, and an empty domain is vacuously one.

`_execute_nest!` evaluates it once per `execute!` (each composite's type is
concrete there, so it unrolls to a handful of integer compares) and, where it
holds, replaces `fill_offsets!` + `describe_block` with `_ramp_slivers!` /
`_ramp_descriptor` / `_ramp_offset_range`: no offset buffer is materialized and
no buffer is scanned. `_ramp_descriptor` reproduces `describe_block`'s output
*field for field*, including its `stride == 0` convention for a count-1 block,
so the two paths are comparable with `==` and not merely behaviourally
equivalent -- `test/test_per_call_floor.jl` asserts exactly that over 300
randomized (group, block, sliver-size) triples.

**Deliberately not generalized.** Anything not provably a ramp falls back to
the existing buffer path, unchanged, per the brief. The `if` is a runtime
branch on a per-call `Bool`, not a plan type parameter, so no `ContractPlan`
field or type parameter changed and no existing pinning test moved.

**`normalize_group` is NOT applied to the plan's composites, deliberately.**
Doing so would widen the ramp path to every *foldable* composite (e.g.
`ccsd_t_1`'s N composite folds from rank 3 to rank 2; `dim15_2_2_2`'s fold to
contiguous), but it rewrites `plan.mgroup`/`ngroup`'s rank, and
`test/test_driver.jl:1194-1195` pins `plan.mgroup.lengths == ntuple(_->d, 3)`
as part of the *label-order* milestone's contract. Changing a pin that belongs
to another milestone to buy a performance path is a separate decision, and the
`affine_ramp` predicate above is itself the fold test, so the extension is
cheap to make later: apply `normalize_group` to the three groups and update
those two pins to compare against the normalized expectation. Left undone.

### Measured result

Re-profile on the working tree (`--tag percallfloor-post`), against the base
tree run above:

| case | GFLOP/s base -> new | planning | driver_loop | gc/alloc | alloc/call |
|---|---|---|---|---|---|
| `ao2mo_2_dim16` | 18.18 -> **21.67** (1.19x) | 7.08% -> 1.61% | 29.83% -> 27.27% | 4.62% -> 0.48% | 6080 -> 1472 B |
| `plain_64` | 32.08 -> **42.62** (1.33x) | 23.51% -> 5.80% | 10.95% -> 4.88% | 6.96% -> 1.63% | 4368 -> 1232 B |
| `smallMN_16x256x16` | 12.99 -> **21.64** (1.67x) | 33.88% -> 9.50% | 11.67% -> 3.05% | 11.86% -> 4.23% | 4368 -> 1232 B |

Both targeted buckets shrank in absolute time, not only in share: `planning`
8.2 -> 1.6 us / 3.8 -> 0.7 us / 3.4 -> 0.6 us, `driver_loop` 34.4 -> 26.4 us /
1.8 -> 0.6 us / 1.2 -> 0.2 us on the three cases respectively.

`ao2mo_2_dim16`'s `driver_loop` share barely moves because its N composite is
`C[a,b,r,s] = A[q,b] * B[a,q,r,s]`, whose N labels `(a, r, s)` have C-strides
`(1, 256, 4096)` with the `b` label's stride-16 axis sitting *between* them:
`16*1 != 256`, so it does not fold and is not a ramp. Its M and K composites
are ramps and do take the fast path, which is the 8 us that did come off. Two
further, bounded ideas for that case are recorded under "Deferred" below.

**ABBA guard** (`benchmark/bench_real_path_guard.jl`, 21 reps, base tree
extracted from `20e1002` with `git archive`, A-B-A-B-B-A over three runs per
tree, per-shape median of the three): geomean new/base **0.9207**, median
0.9343, **0 of 18 shapes slower than base by more than 5%**. 16 of 18 are
faster; the two that are not (`Float64/256^3` 1.040, `Float64/512^3` 1.026)
are large compute-bound shapes this change cannot touch, are inside the
instrument's own ~5-6% resolution, and are not part of a one-sided shift --
which is the pattern the guard's header says to look for. Canary spreads
0.5%/1.1%/0.5%/5.9%/8.4% across the six runs. Note the guard times `execute!`
only (the plan is built once outside the loop), so it sees item 2 + item 3 and
*none* of item 1.

Full suite green: **52679 passed, 0 failed** (47224 before this pass; 5455 new
in `test/test_per_call_floor.jl`). Forced-ISA runs (`QS_FAKE_ISA=avx2/32/16`
and `unknown/0/0`) pass with only the two residues
`test/forced_isa_runner.jl`'s own header documents (`test_target.jl:38`'s
re-detection assertion, and the Bumper-dependent allocator testsets being
absent outside the test environment).

### Deferred, with the reason

* **`normalize_group` on the plan's composites** -- see above; blocked only by
  two label-order pinning tests, and worth revisiting as its own small change.
* **`ao2mo_2_dim16`'s remaining 27% `driver_loop`.** Two bounded ideas, both
  unimplemented here:
  1. *Identical maps.* That composite's two maps have **equal** stride tuples
     (`B` and `C` agree on `a, r, s`), so `fill_offsets!` fills two buffers
     with identical contents and `_classify_slivers!` classifies each twice.
     Filling one and reusing its descriptors would roughly halve the cost, and
     the condition (`g.strides[1] == g.strides[2]`) is a cheap exact test. It
     needs the `_axis_of` call sites to select the shared buffer, which is a
     third structural path through the nest -- judged not worth stacking on
     this pass's two.
  2. *Piecewise-affine slivers.* A non-folding composite's offset sequence is
     still affine with step `strides[p][1]` on any interval that does not cross
     a multiple of `lengths[1]`. A sliver `[q0, q0+c)` is therefore a closed-
     form `AffineAxis` iff `q0 ÷ L1 == (q0+c-1) ÷ L1`, with only the crossing
     slivers needing the buffer. On `ao2mo_2` (`L1 = 16`, `NR = 6`) that is 2
     of every 3 slivers. This is a genuine generalization of item 3 and the
     brief explicitly scoped item 3 to the rank-1 case, so it is written up
     rather than built.
* **Complex packing's conditional load** (`_pack_panel_complex!`) -- still
  untouched, as it was after the packing-speed milestone; unrelated to the
  floor, and the complex path is not among the profiled regressions.

**Bucket attribution note.** The profiler's `SPECIFIC_QS_BUCKETS` walks a
sample's whole stack, so every new helper was already attributed correctly
before this pass touched `benchmark/profile_buckets.jl` -- `affine_ramp` /
`_ramp_slivers!` / `checked_span_bounds` reach `_execute_nest!` (driver_loop)
and `_pair_group_static` reaches `plan_contract` (planning) -- and the
`unsafe_pack_a!` / `unsafe_execute_tile!` names contain the `pack_a!` /
`execute_tile!` substrings their buckets already match on. The names were
added to the lists anyway, for explicitness; because each one's only caller is
already in the same bucket, the addition cannot change any classification, and
the post numbers above (taken before the addition) stand. `other` was
0.18-0.60% in all three post profiles, which is the check that nothing
escaped.

### Adversarial review of the bounds-check hoist: two test gaps closed

An independent adversarial review of item 2 found **no correctness defect** --
the reviewer tried and failed to construct a counterexample to the equivalence
argument above, for negative strides, zero strides, irregular/scattered
slivers and all four kernel types -- but flagged one real hole in how it was
*tested*, specifically in the silently-under-validated direction. Both gaps
are now closed in `test/test_per_call_floor.jl`.

**Gap 1: the equivalence property test supplied its own aggregate range.** It
computed `blockrange = (minimum(rowoffs), maximum(rowoffs))` and fed that to
`checked_span_bounds`, so it proved the check is equivalent *given a correct
aggregate range* -- and would have passed unchanged if `_classify_slivers!`'s
own accumulation tracked only the first or only the last sliver. That is the
dangerous direction: a too-SMALL aggregate range throws nothing and still
computes correct values on every in-bounds input, so no other test in the file
would have noticed either. Added: a 500-case test that calls
`_classify_slivers!` on hand-written (hence genuinely irregular; the
ramp-vs-classify test can only reach regular blocks by construction) buffers
and compares its returned range against `minimum`/`maximum` over the true
buffer contents, plus a hand-built case with both extremes planted in an
*interior* sliver -- one reached through `descriptor_offset_range`'s scan
branch, one through its affine branch at a negative stride -- where the first
and last slivers are deliberately unremarkable.

**Gap 2: every rejection test used a dense column-major GEMM**, whose offsets
increase monotonically, so its out-of-bounds address is necessarily in the
LAST sliver -- which even a last-only accumulation would catch. Added a
fixture whose binding address is strictly interior: `C[m,n1,n2] = A[m,k] *
B[k,n1,n2]` with `C` a REVERSED view along `n2`, giving the N composite the C
map `(+M, -M*N1)` over lengths `(13, 3)`. Offsets climb by `M` inside each
`n1` run and fall by `13M` at every run boundary, so at `NR = 6` the slivers
straddle those boundaries (irregular) and the block maximum lands at logical
coordinate `q = 12`, sliver 2 of 7. One element off the destination then makes
exactly that interior address overflow. The test derives where the extreme
sits rather than asserting it by hand, checks the correctly sized plan still
computes the right answer, and confirms `BoundsError` with **nothing written**.

That fixture's discriminating power is asserted directly, by handing
`checked_span_bounds` the true range (rejects) and the first-only and
last-only ranges (both silently accept), rather than by mutating the driver.
A mutation run was done once, out of tree, and is why: a driver with the
first-sliver-only bug does not merely accept the plan, it performs the
out-of-bounds write, corrupts the heap and hangs. Recording that here because
it is the concrete demonstration that this hoist is the part of the change
with real teeth, and that the accumulation -- not just `checked_span_bounds`
-- has to be tested directly.

**Incidental behaviour change, pinned rather than left implicit.** Hoisting
made `execute!`'s rejection *stricter*: it now fails before any write for the
whole block, where the old per-tile check wrote every tile preceding the
offending one. `execute_tilewise!` still has the old behaviour, so the two
differ, and the test asserts both sides of that (`all(iszero, C)` after
`execute!`, `any(!iszero, C)` after `execute_tilewise!`). "Atomic" is a claim
about one macro block only: with several blocks, earlier ones can still have
been written before a later block's check throws.

**One docstring narrowed.** `execute!`'s said "no address this driver can
reach is unvalidated"; that is very slightly too strong, because the
pre-existing `Qk == 0 || alpha == 0` beta-only short-circuit reaches
`_scale_all_of_C!`, which has never had a storage-bounds check (it writes
through `scale_tile!`'s `@inbounds` path on `AxisGroup`s validated at
construction). Unchanged behaviour, not introduced here; the sentence is now
scoped to the macro-blocking pack/execute path and the exception named
explicitly.

## tensorcontract-rs comparison (2026-09-22)

Milestone opened from a comparison against the user's sibling Rust project,
`tensorcontract-rs` (the same design source cited by the complex-element-type
milestone). Full milestone brief, evidence-verification corrections, and the
orch-planner execution contract are in
`.claude/orchestration/tensorcontract-rs-lessons.md` (tracked in git, per
this project's convention for orchestration files, e.g. `label-order.md`).
Two of the original
five comparison items were found moot before this milestone opened: the
allocation-light label classification (already shipped as the per-call-floor
milestone above) and an opt-in cache-blocking model (already measured and
rejected, `src/blocking.jl`'s own docstring). Three items remained in scope,
each gated on its own measurement before implementation, per this project's
standing evidence-first convention.

### Item 1: unified real/complex packing (shipped)

**Gate (before implementing).** `src/packing.jl` kept two structurally
separate packing loops: `_pack_panel!` (real, had the full/tail vectorized
split from the "Packing speed" milestone above) and `_pack_panel_complex!`
(complex, still had that milestone's original per-element `t < valid ? load
: zero` conditional). Evidence gathered
(`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/pack_loops_PROVENANCE.txt`,
`profiles/buckets_summary-tcrs-pre.txt`): packing share up to **62.8%**
(`smallN_256x256x12`, `ComplexF32`, named-buckets-only) against a 10%
go/no-go floor; complex-loop cost **1.58x** per real-emitted vs. the real
fallback loop on the B operand (0.4583 vs. 0.2904 ns/real) against a 1.3x
floor. Gate passed decisively on both independent criteria.

**Design.** One `_pack_panel!(packed, ::Type{T}, format::FMT, ::Val{PD},
kc, valid, transform, load, plane_offset)`, keeping the real loop's exact
full/tail structure, dispatching the per-lane store through
`_pack_emit!`/`_pack_emit_zero!` on `format`. `RealFormat` gained emit
methods (it had none before -- `PlanarFormat`/`OneEFormat` already had
theirs from the complex-element-type milestone); real callers pass
`RealFormat()` and a `plane_offset` closure over `packed_a_offset`/
`packed_b_offset`; complex callers pass `Val(MR)`/`Val(NR)` in place of the
old `Int physical_dim`. `_pack_panel_complex!` deleted. Untouched (test-bound
or out of scope): `_pack_a_contiguous_eligible`, `_pack_a_contiguous!`,
`_copies_unchanged`, `src/complex_format.jl`, `src/kernel_descriptor.jl`, any
kernel file, `src/driver.jl`.

**Shipped as commit `69f8e4f`**, "Unify real/complex packing into one loop,
format-dispatched" (`src/packing.jl` +106/-44 net across the diff;
`test/test_packing.jl` +38, `test/test_packing_complex.jl` +39 -- Float32
tail-width coverage and `kc==1` full/tail coverage for Planar A, Planar B,
and 1e A under `conj`, closing gaps the evidence-gate work had left
unexercised). Full suite **54431/54431 passing** (baseline 54203 + 228 new),
0 failed. Runic-clean on every added/touched line (two pre-existing,
unrelated diffs in `test/test_packing.jl`/`test/test_per_call_floor.jl`,
present since before this milestone, deliberately left alone). Both
forced-ISA runs (`avx2`, `unknown`) show only the residues
`test/forced_isa_runner.jl`'s own header documents.

**AC4a (real-dtype LLVM IR structurally identical): PASSED.** `code_llvm`
(debuginfo=:none) dumped for `_pack_a!`/`_pack_b!` at `KernelDescriptor{16,6}`
and `{32,6}`, both dtypes, both bounds-check `Val`s, across
contiguous-fast-path/full-sliver/tail-sliver source shapes: 25496 lines each
pre/post; raw diff nonempty (9080 lines) but entirely SSA-numbering and
per-session pointer-literal noise; diff after normalizing digit sequences is
**empty**. No fallback needed.

**AC4b (two-tree ABBA guard): PASSED, narrowly.** Base tree archived from
`f7fa490` (`git archive` + the working tree's gitignored `Manifest.toml`
copied in, both environments `Pkg.instantiate()`d), A-B-B-A over
`benchmark/bench_real_path_guard.jl`, 21 reps/run, 2 runs/tree, all 4 canary
spreads 2.3-4.5% (quiet). Per-shape median over 18 shapes (real dtypes only,
9 shapes x {Float32,Float64}): **geomean new/base = 0.9718** -- inside the
required `[0.97, 1.03]` band by 0.0018. 12/18 shapes moved >2% faster, 3/18
>2% slower, 3 within noise -- 12 < the 14-shape one-sided-shift trigger, so
no flag. **Honestly noted, not chased further**: 12 of 18 *real-dtype* shapes
(this commit touches only complex packing; AC4a already proved the real-path
IR is bit-identical) show a mild, fairly consistent 3-5% speedup on the new
tree. The harness/archive setup was checked and found sound (identical
`Manifest.toml`, matching thread settings, correct ABBA ordering, all four
canaries quiet). Most likely ordinary cross-tree noise -- this project's own
"Measurement hygiene" note (above) already documents 4-15% canary spreads and
rank reversals between successive sweeps as normal on this shared machine --
but the one-sidedness is recorded rather than smoothed over, since it sits
inside accepted thresholds without a mechanism to explain it.
Artifacts: `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/
real_path_guard_new69f8e_run{1,2}.csv` (new tree, on disk); the base-tree
runs were written under a session scratchpad and are not guaranteed to
persist -- the geomean/split figures above are the artifact of record for
this entry.

**AC4c (complex packing microbenchmark): PASSED via the B operand.**
Re-measured with `benchmark/bench_pack_loops.jl`. **Corrected against the
T5 review (S2)**: an earlier draft of this paragraph quoted numbers from an
intermediate re-run that no longer matches the artefact actually on disk,
because `pack_loops.csv`/`pack_loops_PROVENANCE.txt` were overwritten by a
later run during this same milestone (item 2's evidence-gathering reused the
same benchmark directory). The numbers below are re-derived directly from
the CURRENT on-disk artefact
(`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/
{pack_loops.csv,pack_loops_PROVENANCE.txt}`, provenance `69f8e4f-dirty`,
canary spread **6.23%**, 21 reps), against the preserved pre-change baseline
at `pack_loops.tcrs-pre.csv`:

| case | pre | post | ratio |
|---|---|---|---|
| Planar-A-full ComplexF64 | 0.3826 | **0.3977** | 1.040x slower |
| Planar-B-full ComplexF64 | 0.4583 | **0.2949** | **1.554x faster** |
| 1e-A-full ComplexF64 | 0.3912 | **0.4910** | **1.255x slower** |
| 1e-A-tail ComplexF64 | 0.3845 | **0.4380** | **1.139x slower** |

The acceptance criterion required >= 1.2x on *at least one* of the two
originally-named numbers (Planar-A-full or Planar-B-full); Planar-B-full
clears it at 1.554x, so **AC4c still passes**, but two regressions the
original paragraph did not mention are now on record: 1e (`OneEFormat`, the
Van Zee 1m method's packing) is 14-26% *slower* post-unification on both
sliver kinds, for `ComplexF64`. This is plausible on structural grounds
(1e's emit path may not vectorize as cleanly inside the shared full/tail
loop as it did in the old, format-specific loop) but has not been profiled
further -- flagged here rather than silently omitted, since this project's
own canary convention (Real-B-tail `Float64` alone moves +42% between two
supposedly-identical-IR runs in this same session, per T5's cross-check)
means neither the Planar-B win nor the 1e loss can be fully separated from
noise on a single un-repeated run. A repeat run with alternation would
settle it; not done here, out of this review-response's immediate scope.

Post-unification `code_llvm` of the unified `_pack_panel!` at the Planar-A
argument types confirms the per-element branch this item exists to remove
is gone (`icmp eq i64 %valid, 24` / `br ... label %L3, label %L101` -- a
single coarse full-vs-tail branch per call, not a per-element one) --
checked directly, not assumed. This structural finding (not a timing
number) is the strongest evidence for this item, and is unaffected by the
artefact/noise concerns above.

**AC4d (existing tests pass unmodified): PASSED**, folded into the
54431/54431 full-suite result above -- no existing testset body was edited,
only new testsets added.

**Profiling re-run** (`benchmark/profile_to_suite.jl`, same 10 cases, `--tag
tcrs-post` vs. the pre-change `tcrs-pre` run, named-buckets-only packing
share):

| case | pre | post |
|---|---|---|
| `plain_64_c64` | 18.18% | 18.29% |
| `plain_64_c32` | 26.94% | 21.46% |
| `shallowK_256x24x256_c64` | 4.45% | 4.81% |
| `shallowK_256x24x256_c32` | 5.37% | 4.51% |
| `smallN_256x256x12_c64` | 48.66% | 48.96% |
| `smallN_256x256x12_c32` | 62.81% | 53.97% |
| `smallM_12x256x256_c64` | 28.72% | 28.61% |
| `smallM_12x256x256_c32` | 28.11% | 23.56% |
| `smallMN_16x256x16_c64` | 30.49% | 30.32% |
| `smallMN_16x256x16_c32` | 30.97% | 27.59% |

Pattern matches the microbenchmark: every `ComplexF32` (Planar `48x3`) case's
packing share drops (14-20% relative), every `ComplexF64` (Planar `24x3`)
case stays flat -- consistent with the B-operand-only improvement measured
above, since the packing-share reduction tracks which format's B-side
packing dominates the mix, not a uniform effect. Artifacts:
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/profiles/
buckets_summary-tcrs-{pre,post}.txt`.

**Unplanned finding, corrected against the T5 review (S3), narrowed to what
it actually is: an unrelated, pre-existing effect, not something this item
caused or that AC4c leaves ungated.** `benchmark/bench_complex_efficiency.jl`
on the post-change tree read ComplexF64/ComplexF32 geomean **0.909/0.734**
(run 1, canary 10.4%) and **0.939/0.786** (run 2, canary 16.6%). An earlier
draft of this paragraph said this metric "is not gated by any AC4 item" --
that is wrong: **AC4c's second clause is exactly "`bench_complex_efficiency.jl`
geomeans not below base by more than the run's canary spread"**, and the
correct baseline for that clause is the CLEAN, pre-milestone `f7fa490`
reading the planning contract's own Section 0 already recorded: **0.96
(ComplexF64) / 0.75 (ComplexF32)** -- not the much older
**1.829/1.910** complex-element-type-milestone figure, which is a different
comparison. Read against the right baseline, the post-change numbers
(0.909-0.939 / 0.734-0.786) are flat within each run's own canary spread:
**AC4c's second clause passes.** The real, and separate, observation is that
`f7fa490`'s own 0.96/0.75 was ALREADY roughly half of the 1.829/1.910
complex-element-type-milestone figure -- i.e. whatever caused that drop
happened at some point BEFORE this milestone opened, not as a consequence of
item 1's packing change. The base-tree artefact this comparison would need
has since been overwritten by a later run in this same benchmark directory
(the same overwrite problem noted in AC4c above); the planner's Section 0
reading of it is the only record left. Flagged for the user as a genuine,
separate, pre-existing question -- a bisection between the complex-element-
type milestone's close and `f7fa490` (most plausibly implicating the
per-call-floor work, given the timing) is the natural next step if pursued,
but it is not this milestone's regression to fix.

### Item 2: F2 K-depth guard (2-A)

**Gate (before implementing, G2).** A 28-point sweep
(`benchmark/probes/probe_f2_kdepth.jl`, `C[a,b,c,i,j,k] = A[i,j,m,a] *
B[m,k,b,c]`, `i=j=k=b=c=6`, `m` swept 8..512) found F2 (`_demote_for_run`)
fires unconditionally at every point regardless of the contracted extent
`Qk = m`, and crosses over from a win to a loss between `Qk=32` and `Qk=64`
for Float64 and between `Qk=64` and `Qk=128` for Float32. The "broken-enough"
fraction guard (tensorcontract-rs's second guard on the same mechanism) gave
conflicting evidence between dtypes at the tested default fraction (Float64
supported adopting `0.75`; Float32 contradicted it), so per the planning
contract's instruction to keep ONE constant (not split by dtype), it was
adopted inert (`1.0`) rather than guessed. Four points on the "less-broken"
`a=40`(F64)/`a=80`(F32) rows exceed the sweep's own 15% record-and-ask
threshold (up to 37.4%) even with the Qk-cutoff guard in place -- recorded as
a known, accepted residual (U2), not something this item fixes.

**Design.** `_demote_for_run(::Type{T}, kernel, run::Int, Qm::Int, Qk::Int)
where {T<:Real}` (`src/driver.jl:299`) gained a `Qk` parameter. Two new
guards run BEFORE the existing `Qm == run || run % mr(kernel) == 0`
short-circuit and search: return `kernel` unchanged if `Qk >
F2_DEMOTE_KMAX_F64`/`F2_DEMOTE_KMAX_F32` (by `T`), or if
`_unbroken_fraction(Qm, run, mr(kernel)) > F2_BROKEN_ENOUGH`. New
`_unbroken_fraction(Qm, run, mr)` (`:262`) computes the fraction of the
`cld(Qm, mr)` register slivers that lie entirely inside one run of length
`run` -- by construction exactly `1.0` iff the existing predicate holds,
checked over a 5000-point randomized grid
(`test/test_driver.jl`, `"F2 K-depth guard: _unbroken_fraction
equivalence"`) rather than a couple of hand-picked cases. Three new
constants next to `KERNEL_SHAPES_F64`/`KERNEL_SHAPES_F32`
(`src/driver.jl:442-454`): `F2_DEMOTE_KMAX_F64 = 32`, `F2_DEMOTE_KMAX_F32 =
64`, `F2_BROKEN_ENOUGH = 1.0` (inert -- a fraction in `[0,1]` can never
exceed `1.0`, so this guard is dead code today, deliberately left wired in
rather than simplified away so a future pass can activate it by changing
one constant). Both `plan_contract` call sites
(`src/driver.jl:1117`/`1125`) updated to pass `Qk` (already in scope). The
generic complex no-op fallback (`:314`) got the same signature extension,
still a no-op. Nothing else touched (`_prefer_swap`, `_leading_unit_run`,
`_default_kernel`, the kernel-shape menus, `_plan_contract`, the five-loop
nest, and every kernel file are unmodified).

**New tests** (`test/test_driver.jl`, both additive, no existing testset
body edited): (1) the `_unbroken_fraction` equivalence property test above;
(2) `"F2 K-depth guard: deep-K no longer demotes, shallow-K still does"`,
reusing the sweep's own fixture family for `(Float64, a=8)` and `(Float32,
a=16)` -- at `Qk = 2*kmax(T)` asserts `plan.kernel === _default_kernel(T,
Qm, Qn)` (no demotion); at `Qk = 8` the expectation is DERIVED from the
run-length predicate (`Qm == run || run % mr(default_kernel) == 0`), exactly
like the pre-existing F2 testset above, rather than hardcoded to "always
demotes" -- an earlier draft of this test hardcoded that assumption and
failed under a forced AVX2 profile (`QS_FAKE_ISA=avx2`), where the
default kernel's narrower `mr` already satisfies the predicate at this
fixture's run length; fixed before landing, see below. No literal `mr` in
either new testset.

**Verification.**
- `Pkg.test()`: baseline (unmodified tree, HEAD `d27286e`) **54431/54431**;
  post-change **64443/64443** (delta = 5000x2 property-test assertions + 6x2
  new assertions in the deep/shallow-K testset per dtype, matching exactly).
- Runic: `runic --check src/driver.jl` and `runic --check test/test_driver.jl`
  both exit 0 (clean) on the changed lines.
- Forced-ISA (`test/forced_isa_runner.jl`): `avx2/32/16` -- 62435 passed, 1
  failed (`test_target.jl` "runs on this host without throwing", the
  documented-expected residue of running under a forced profile) + 1 errored
  (`Package Bumper not found`, expected per the runner's own docstring when
  invoked outside the `Pkg.test()` environment). `unknown/0/0` -- 62431
  passed, same 1 failed + 1 errored residue, plus 2 broken. **Identified, not
  left as a flag** (T5 review, N7, which correctly caught that an earlier
  draft of this paragraph guessed "a standing Aqua ambiguity-check artifact"
  without checking): both are the pre-existing `skip =
  (target_profile().nregisters == 0)` condition in
  `test/test_planar_kernel.jl`/`test/test_onem_kernel.jl`, which fires only
  under `QS_FAKE_NREG=0` (the `unknown` profile) and not under `avx2`'s
  `QS_FAKE_NREG=16` -- unrelated to any F2/driver code path touched here.
  The AVX2 run is also what caught the hardcoded-assumption test bug noted
  above.

**AC5a (K-sweep re-run vs. pre-change).** Re-ran
`benchmark/probes/probe_f2_kdepth.jl` on the changed tree; pre-change CSV
preserved at
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/f2_kdepth_pre-guard.csv`,
post-change at `f2_kdepth_post-guard.csv` (canary spread 9.1%, `uptime` load
average 2.8 -- noted, not clean, per this project's measurement-hygiene
convention).

Kernel-selection outcome, checked by TYPE at every point (not just timing):
for `(Float64, a=8)`, `Qk in {8,16,32}` still resolve to the demoted
`(8,6,4)` kernel and `Qk in {64,128,256,512}` now resolve to the shipped
default `(16,6,8)` -- the guard fires exactly at the measured crossover. For
`(Float32, a=16)`, `Qk in {8,16,32,64}` still demote to `(16,6,8)` and `Qk in
{128,256,512}` now resolve to `(32,6,16)`. Both match the intended cutoff
(`kmax(F64)=32`, `kmax(F32)=64`) exactly, with `Qk=64` itself (== kmax, not
`>` kmax) correctly still on the demoting side for Float32.

`auto` vs. `min(forced-default, forced-demoted)` timing ratio on the two
FULLY-BROKEN rows (`a=8` F64, `a=16` F32) at every measured `Qk`: **0.983 -
1.037** across all 16 points (median ~1.00), i.e. tracking the pointwise
optimum to within the run's own 9.1% canary spread -- consistent with exact
(`1.000`) tracking modulo ordinary measurement noise, not a systematic gap.
(Per-point ratios: F64 a=8 -- 0.989, 0.991, 0.996, 1.002, 1.003, 1.032, 1.037,
1.014, 1.015 for `Qk = 8,16,32,64,128,256,512,512,512`; F32 a=16 -- 1.004,
0.983, 1.001, 0.998, 0.998, 1.026, 1.000 for `Qk = 8,16,32,64,128,256,512`.)

**Residual (U2), confirmed present as expected, not fixed.** The four points
the planning contract flagged in advance (`a=40`/F64, `a=80`/F32, the
"less-broken" rows, where demotion still fires for `Qk <= kmax(T)` but is
not the pointwise-optimal choice) show gaps consistent with the pre-change
measurement, within noise:

| point | pre-change r | post-change r |
|---|---|---|
| F64 a=40 Qk=16 | 1.148 | 1.190 |
| F64 a=40 Qk=32 | 1.374 | 1.387 |
| F32 a=80 Qk=32 | 1.107 | 1.139 |
| F32 a=80 Qk=64 | 1.292 | 1.256 |

Max observed gap post-change: **38.7%** (F64 a=40, Qk=32) -- same order as
the pre-change 37.4% figure and the same documented mechanism (F2 still
fires unconditionally for `Qk <= kmax(T)`, with no regularity-based
tie-break; `F2_BROKEN_ENOUGH` is inert). This is the expected, previously
recorded residual, not a new or smaller/larger-than-expected finding.

**AC5c (shallow-K/`ccsd_t_1` still demotes, still wins).** Re-ran
`julia -t 1 --project=benchmark benchmark/bench_ccsd_t_store.jl --dims 8,16
--dtypes Float32,Float64`. For `ccsd_t_1` at both `dim=8` and `dim=16`, both
dtypes (`Qk = dim`, well inside both `kmax` values), the M-composite's
register slivers are reported **fully regular/unit-stride** (e.g. `dim=16`,
Float32: `M-slivers: regular&&unit-stride = 256  other = 0`; `dim=16`,
Float64: same, `256/0`) -- i.e. the vectorized store path is fully engaged,
confirming F2 still demotes exactly as before at these `Qk`. This is also
guaranteed by construction, not just observed: `Qk in {8,16}` is `<=
kmax(T)` for both dtypes and `F2_BROKEN_ENOUGH` is inert, so
`_demote_for_run` takes the identical code path (falls through both new
guards unconditionally false) it did before this change -- the kernel
choice on this fixture family is provably unaffected. Absolute timings on
this run (`arm=1`/QuasiStrided, `ccsd_t_1`): `dim=8` Float32 `1.613e-4 s`,
`dim=16` Float32 `1.349e-2 s`, `dim=8` Float64 `2.489e-4 s`, `dim=16`
Float64 `2.106e-2 s` -- not directly comparable to the original F2 pass's
table (2026-09-21, `1.626e-2`/`2.649e-4 s` for Float32) since three
unrelated performance passes (F1 inlining, per-call floor, packing
unification) landed on this tree since then; the "still wins" claim here
rests on the code-path-identity argument above, not a re-measured speedup
ratio against a pre-F2 baseline.

**Probe-script artifact found and resolved, not a driver defect -- and
corrected here against the T5 review (N8), which caught that this paragraph
misdescribed which column was wrong.** One sweep point (`Float32, a=80,
Qk=32`) initially looked contradictory when eyeballed mid-investigation:
timing said the point was on the "less-broken residual" side (`r > 1`,
demotion not pointwise-optimal), yet a quick read seemed to disagree with
which kernel actually ran. Checked directly (not just from the CSV) by
reproducing the exact plan: `auto`'s kernel at that point is
`SIMDKernel{16,6,Float32,8}`, bit-identical to the recorded `demoted_shape`
-- the correct, expected pick for a "less-broken" point inside `kmax`. The
CSV itself is consistent with this: `auto_matches_default_type=false`,
`auto_matches_demoted_type=true` at that row (the demoted type, correctly).
**The actually-inconsistent field is the separate, redundant `auto_matches`
text column**, which reads `default` at that row despite
`auto_matches_demoted_type=true` -- a labeling bug in that one derived,
human-readable column of `benchmark/probes/probe_f2_kdepth.jl` (predating
this item, unrelated to `_demote_for_run`'s own logic); the type-identity
columns and the timing/ratio data in the same row are unaffected and were
used as-is above. Not fixed here (probe tooling, out of this item's edit
scope) -- noted so the `auto_matches` text column specifically isn't
trusted blindly in the future; the type-identity columns are fine.

**Commit**: `5efa37b` "F2 K-depth guard: gate run-length demotion on the
contracted extent (item 2)".

### Item 3: one run-length derivation, explicit consumers (conditional on item 2)

**Correction to the original comparison and to the planning contract.** Both
described this as "three independent re-derivations of run length in C" and
proposed consolidating them behind a new shared primitive. Neither is
accurate: there was already exactly ONE derivation, `_leading_unit_run`
(`src/driver.jl`) -- `_prefer_swap` called it twice (once per composite) and
the two `_demote_for_run` call sites in `plan_contract` called it a third
time, for whichever orientation was actually chosen. The only real
redundancy was that `plan_contract` recomputed the same two values
(`_leading_unit_run(morder,...)`/`_leading_unit_run(norder,...)`) a second
time at the demotion call sites, having already computed them once inside
`_prefer_swap`'s old body. The planner's proposed relation to `affine_ramp`
(the per-call-floor milestone's closed-form block primitive) was ALSO
checked and found not to hold as literally stated -- see below.

**Design shipped.** `plan_contract` now computes `run_m = _leading_unit_run(
morder, indC, C)` and `run_n = _leading_unit_run(norder, indC, C)` once,
immediately after `morder`/`norder` are built, and both the swap decision and
both `_demote_for_run` call sites consume these same two values -- one
derivation, three consumers, all reading the same computed value rather than
recomputing it. `_prefer_swap` gained a genuine two-method split: a core
`_prefer_swap(run_m::Int, run_n::Int, mr_asis::Int, mr_swapped::Int=mr_asis)`
holding the actual logic, and the pre-existing label-list signature
(`morder::Vector{Int}, norder::Vector{Int}, indC, C, mr_asis, mr_swapped`)
kept as a one-line wrapper that derives the same two run lengths and calls
the core -- so the existing external test coverage of the label-list form
(`test/test_driver.jl`, `"label order: _leading_unit_run / _prefer_swap"`)
needed no changes, and a new testset
(`"label order: the two _prefer_swap methods agree"`) cross-checks the two
methods directly on four label/mr combinations rather than just trusting the
wrapper reads correctly. No decision semantics changed anywhere -- this is
pure dead-computation removal, not a behavior change.

**The `affine_ramp` relation, checked and corrected.** The contract proposed
testing `(_leading_unit_run(morder, indC, C) == Qm) == (affine_ramp(mgroup)[1]
&& affine_ramp(mgroup)[2][2] == 1)` against `mgroup` (the TWO-map `AxisGroup`
`_build_pair_group` returns, pairing the composite with operand A). This is
WRONG in general: `_leading_unit_run`'s quantity depends only on C's own
per-label strides/extents, but `mgroup`'s `affine_ramp` is a condition on
BOTH its maps (A's structure too) -- strictly stronger, and false on
fixtures where C's map ramps cleanly but A's (arbitrary, e.g. permuted or
scattered) does not. Testing the literal proposal would have produced a
test that fails whenever A doesn't happen to ramp, for reasons having
nothing to do with `_leading_unit_run`. The corrected, provably-equivalent
primitive is a SINGLE-map (`P=1`) `AxisGroup` built from C's own map alone
(`src/driver.jl`'s `_build_pair_group`-style construction, but with one map
instead of two): `(run == Qm) == (isramp && steps[1] == 1)` holds exactly,
verified over 2000 randomized `(D, lengths, strides, order)` draws with zero
mismatches (`test/test_driver.jl`, `"label order: _leading_unit_run's
full-coverage condition is a single-map affine_ramp on C's own strides"`).
Two edge cases needed excluding from the random draw, both reproduced as
real (not hypothetical) mismatches before being excluded: an empty `order`
(rank-0) and an `order` consisting entirely of singleton axes (`Qm == 1`) --
`affine_ramp`'s own documented convention reports `steps = (0,)`, not
`(1,)`, for a vacuous ramp, while `_leading_unit_run` returns `1` (matching
`Qm == 1`) unconditionally in both cases; this project's own existing
convention (see `_lo_run`'s separate `Int[]`/zero-length-axis assertions,
above) already treats these as separate cases rather than folding them into
a general property, and this test follows that precedent. This relation is
not wired into any shipped code path -- `affine_ramp`/`normalize_group`
remain deliberately NOT applied to `plan_contract`'s composites (per-call
floor's own "Deferred" note) -- it is recorded here as a correction to the
planning record, not a new mechanism.

**Verification.** Full suite **66447/66447** (baseline before this item
64443 + 2004 new assertions: the 2000-point relation property test + 4
cross-check assertions in the two-methods-agree testset). Runic clean
(`src/driver.jl`, `test/test_driver.jl`). Both forced-ISA profiles show
only their by-now-documented residues (`avx2`: 1 failed + 1 errored;
`unknown`: same two + 2 broken from the pre-existing `nregisters == 0`
`skip=` condition, unrelated to this item).

**AC6 (non-regression on the per-call-floor fixtures).** `plan_contract`
timing/allocation, reused workspace, on `plain_64`, `smallMN_16x256x16`,
`ao2mo_2_dim16`-shaped (a rank-4 analog at extent 16), and `ccsd_t_1`-shaped
(dim 4) fixtures, before (`96c976e`, archived) vs. after (this item):

| fixture | alloc before | alloc after | time before (ns) | time after (ns) |
|---|---|---|---|---|
| `plain_64` | 1664 B | 1664 B | 910.8 | 903.5 |
| `smallMN_16x256x16` | 1664 B | 1664 B | 868.4 | 1393.0 (single-shot) |
| `ao2mo_2_dim16`-ish | 2000 B | 2000 B | 3974.9 | 4039.9 |
| `ccsd_t_1_dim4` | 2384 B | 2384 B | 1168.3 | 1150.0 |

Allocation is byte-identical on all four (expected -- this item removes a
redundant computation, it does not change what's allocated). The single-shot
`smallMN` timing looked like a ~60% regression; re-measured as a median of 9
rounds of 3000 calls each, alternating trees, and it converged to the same
~815-950 ns range on BOTH trees across all 4 rounds -- a warm-up/GC-pause
artifact of that one single-shot sample, not a real effect. No regression
found; consistent with the change being pure dead-computation removal.

**Commit**: `65ae65a` "Item 3: one run-length derivation, explicit consumers,
no decision change".

## ComplexF64 `:tccg` slowdown: root-caused to the planar store path, no fix shipped (2026-09-22)

**Trigger.** Job 7087420's evidence-gate run (`benchmark/results/worker6160-2026-09-22/bench_to_suite.csv`) showed QuasiStrided's median ComplexF64/Float64 GFLOP/s ratio on `:tccg` sitting at **0.29** across all 48 (dtype, case) pairs, against StridedBLAS's **0.68** on the identical cases. A truly efficient complex kernel doing ~4x the real arithmetic of its real counterpart (complex MAC = 4 real multiplies + rounding, against `flops(spec)`'s dtype-blind nominal count) would land near 0.25 by construction; QuasiStrided sits right on that naive floor while StridedBLAS's ZGEMM extracts real efficiency beyond it. That gap, not "complex is inherently slower," is what needed explaining.

**Method.** Added four cases to `benchmark/profile_to_suite.jl`'s `CASES` (additive, existing entries untouched): a ComplexF64 twin of the already-present `ccsd_t_1_dim16` (`ccsd_t_1_dim16_c64`), and a new Float64/ComplexF64 pair for TCCG's `ccsd_6` (`C[i,j,k] = A[i,l,m,k] * B[m,j,l]`, dim=16), chosen because job 7087420 measured its QS/BLAS ratio (36.02 vs 10.46 GFLOP/s, ratio 0.29) squarely at the observed median. Ran `julia --project=benchmark benchmark/profile_to_suite.jl ccsd_t_1_dim16 ccsd_t_1_dim16_c64 ccsd_6_dim16 ccsd_6_dim16_c64` on the login node (login-node CPU, not the Slurm allocation `bench_to_suite.jl`'s numbers came from -- so GFLOP/s figures under the profiler are not directly comparable to job 7087420's; only the **bucket shares** below are used for diagnosis, matching this session's established profiling convention). Every profile's "other" bucket sits at exactly 50.00% in each case -- the idle profiler-listener thread noted in `profile_to_suite.jl`'s own caveat comment -- so all shares below are renormalized against the classified (non-"other") sample count, not the raw total.

**Finding: `ccsd_t_1_dim16` flips from microkernel-dominated to store-dominated under ComplexF64.**

| bucket | Float64 (of classified, non-"other") | ComplexF64 |
|---|---|---|
| microkernel | 13053/16467 = **79.3%** | 5058/14631 = **34.6%** |
| store | 2652/16467 = 16.1% | 8890/14631 = **60.8%** |
| packing | 273/16467 = 1.7% | 284/14631 = 1.9% |
| driver_loop | 482/16467 = 2.9% | 393/14631 = 2.7% |

**Finding: `ccsd_6_dim16` (more contracted indices, already packing-heavy at Float64) shifts further toward packing.**

| bucket | Float64 | ComplexF64 |
|---|---|---|
| microkernel | 7033/12948 = 54.3% | 4632/10374 = 44.6% |
| packing | 4009/12948 = 31.0% | 4191/10374 = **40.4%** |
| store | 267/12948 = 2.1% | 641/10374 = 6.2% |
| driver_loop | 861/12948 = 6.6% | 235/10374 = 2.3% |

**Root cause, confirmed by reading the source, not just the profile.** `src/kernels/planar.jl`'s `store_tile!` docstring says outright: "Ships the scattered/scalar path only... The unit-stride plane-to-interleave fast path is a deliberately deferred, measurement-gated follow-on and is NOT built here." `_store_tile_planar!` writes every output element through a per-lane call to `_axpby_tile!` (src/kernel.jl) inside a `for lane in 1:W` loop, recombining `Complex(revec[lane], imvec[lane])` one scalar at a time -- there is no equivalent of the real path's `_store_tile_vector!` (src/kernels/simd.jl) for the planar format, at any eligibility condition. `_axpby_tile!` is already `@inline` and already has its `alpha`/`beta` branch resolved once by `_store_prologue!` outside the loop, so there is no cheap hoist left to find here -- the cost is the fundamental one of `MR*NR` scalar scatter-stores instead of `MV*NR` vectorized `W`-wide ones. Complex packing has the analogous gap one layer down: `_pack_panel_complex!` (src/packing.jl:401) is, per its own comment, "a parallel loop rather than a generalisation of `_pack_panel!`", and has no counterpart to `_pack_a_contiguous!` (the vectorized fast path the real path got in this session's "Packing speed" milestone above) -- consistent with `ccsd_6`'s packing share growing 31.0% -> 40.4% under ComplexF64 even though its store share barely moves.

**Disposition: not fixed, written up instead.** Both gaps are exactly the "deliberately deferred, measurement-gated follow-on" work the planar kernel's own docstring flagged when it was built -- a real vectorized planar store path needs a correct interleave-into-`Complex{T}` write (plane-pair -> struct-of-complex, not a same-format copy) under the same alpha/beta/edge-tile contract `_store_tile_vector!` already meets for real, plus its own unit-stride eligibility predicate; a vectorized complex pack path needs the equivalent generalization one layer down. Both are comparable in scope to the real path's Phase H store work and this session's real-packing milestone, not a "tens of lines" patch, and per this investigation's own scoping instruction, a change of that size is a decision for sign-off, not something to ship under time pressure. No `src/` change was made this pass; the four new `profile_to_suite.jl` cases are the only diff. **Recommended next step**, if the ComplexF64 gap is worth closing: a `docs/proposals/`-style design note (mirroring `dispatch-tiers.md`'s format) scoping a vectorized planar store fast path first (bigger single-case win per the `ccsd_t_1_dim16` numbers above), with the complex packing fast path as a candidate follow-on once that lands and is re-measured.

## Complex packing and planar store fast paths: shipped for avx512, `:tccg` gap traced to `_demote_for_run`/`_prefer_swap` (2026-09-22)

**What was built** (the recommended next step from the entry immediately above, followed through). `docs/proposals/complex-fast-paths.md` is the design note; two phases shipped on the `complex-packing-fastpath` branch (commits `4a0f942`, `f62ff28`):

  * A vectorized complex A-panel packing fast path (`_pack_complex_contiguous!`/`_pack_complex_contiguous_eligible`, `src/packing.jl`) for `PlanarFormat`, gated on unit-stride `AffineAxis` rows into dense `Complex` storage on a shipped ISA (avx512 today -- `_isa_vector_bytes(Val(:avx512)) == 64`), with an `identity`/`conj` transform and the `RealFormat`/`OneEFormat` exclusions the gate already documents.
  * A vectorized planar store fast path (`_store_tile_planar_vector!`/`_complex_vector_eligible`, `src/kernels/planar.jl`), same ISA/format/unit-stride gating, on the `PlanarKernel` store side (`store_tile!`).

`OneMKernel` (the `1m`/`1e` format) is unaffected by the store-side phase -- it never had a vectorized store on this branch -- and the packing phase covers only the `PlanarFormat` A-panel, not `OneEFormat` or the B-panel, matching the design doc's stated scope.

**Measured**: `bench_complex_efficiency.jl`'s geomean complex-efficiency ratio (ComplexF64/ComplexF32, `julia --project=benchmark -O3 benchmark/bench_complex_efficiency.jl`, ccqlin038, 21 reps/shape over `MAIN_SHAPES ∪ SMALL_SHAPES`):

| | ComplexF64 | ComplexF32 |
|---|---|---|
| before (`f7fa490`, 1 run) | 0.961 | 0.750 |
| after (this branch, run 1) | 1.134 | 0.967 |
| after (this branch, run 2) | 1.122 | 0.935 |

Two "after" runs (not two "before" runs -- one `f7fa490` run was taken given the time budget for this review-response pass; the two "after" runs, taken back to back, show ~0.01-0.03 run-to-run noise in the geomean, which is small next to the ~0.16-0.2 shift from "before") on the same machine, same session. Both dtypes cross the design doc's ">1 means complex amortises overhead better than real" bar for the first time on this branch; ComplexF32 in particular moves from clearly under 1 (paying a structural complex tax) to roughly parity. This is a real, measured improvement on the shapes `bench_complex_efficiency.jl` sweeps (GEMM-shaped and small-N/M/MN cases at avx512), not a projection.

**Honest null result: `ccsd_t_1_dim16` (ComplexF64) store share did not move.** `benchmark/profile_to_suite.jl`'s `ccsd_t_1_dim16` case (dims 16, 6-index output, 1 contracted index -- the same case the entry above profiled) was re-run at ComplexF64 before/after on this branch (temporarily re-adding the `ccsd_t_1_dim16_c64` case used in that entry; not committed here, since it belongs to the `main`-branch profiling session, not this one). Renormalized against the classified (non-"other") sample count, same convention as the entry above:

| bucket | before (`f7fa490`) | after (this branch) |
|---|---|---|
| store | 9418/15546 = 60.6% | 7756/13187 = 58.8% |
| microkernel | 5380/15546 = 34.6% | 4837/13187 = 36.7% |

A ~2 point shift, within noise for a single profiling run each side -- effectively unmoved, despite this branch shipping exactly the planar store fast path the previous entry's root cause pointed at. **Traced to `_demote_for_run`'s `T <: Real` guard** (`src/driver.jl`): `ccsd_t_1_dim16`'s A/B index structure does not give the default complex kernel's `mr` a leading unit-stride C run that's already a multiple of it, which is precisely the condition `_demote_for_run` exists to fix for real dtypes by picking a smaller-`mr` menu shape (`docs/decisions.md`, "F2" -- see the entry above the store-fastpath section). Because `_demote_for_run`'s specific method is `where {T <: Real}` with a real-only fallback, `ComplexF64` never gets demoted, the store-eligible unit-stride run condition is never satisfied for this shape's default kernel, and `_complex_vector_eligible` (this branch's new gate) correctly falls back to the scalar path -- not because the new fast path is broken, but because the kernel it would apply to is never selected here. The store fast path is real and measured on shapes where the default kernel's `mr` already divides the run (per the `bench_complex_efficiency.jl` numbers above); `ccsd_t_1_dim16` simply isn't one of them.

**Two now-measurably-justified follow-ups**, per this design's own Decision 3 (ship the store fast path first, measure, extend the demotion/swap guards as separate later changes):

  1. **Extend `_demote_for_run` to complex** (higher expected value for `:tccg`-like cases such as `ccsd_t_1_dim16`): per the landmine now recorded at its definition site (`src/driver.jl`), this is not a simple `T <: Real` bound relaxation -- the function's demotion search reads the single-argument `kernel_shapes(T)`, whose complex fallback is `_legacy_shape(T)` at register pressure 30 (over AVX2's 16 ymm register budget). A correct extension must route through the two-argument `kernel_shapes(T, method::ComplexMethod)`/`_complex_kernel_from_shape` construction path instead. Needs its own before/after measurement on `ccsd_t_1_dim16`-shaped cases once built.
  2. **Extend `_prefer_swap` to complex**: now that `PlanarKernel` has a vector store to potentially win with a swap, the old "nothing to win" rationale (recorded stale at `_prefer_swap`'s definition and call site, `src/driver.jl`) no longer holds by construction, but whether the swap is actually a net win for complex on real workloads is unmeasured. Needs its own before/after measurement, separate from (1) -- the two guards interact (a swap changes which orientation's `mr`/run-length `_demote_for_run` would need to fix) and should not be bundled into one unvalidated change.

**Merge note (2026-09-23, tensorcontract-rs-comparison / complex-fast-paths reconciliation).** These two follow-ups and the "tensorcontract-rs comparison" milestone above landed independently on separate branches, both touching `_demote_for_run`/`_prefer_swap`. No conflict in substance: this branch's `_demote_for_run` extension (item 2, the `Qk` cutoff) stayed real-dtype-only, exactly per the landmine warning in follow-up (1) above, and item 3's run-length dedup is orthogonal to both follow-ups (it changes how `run_m`/`run_n` are computed and threaded, not the real-vs-complex scope of either guard). Both follow-ups above remain open and unmeasured after this merge.
