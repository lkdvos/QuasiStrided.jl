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
5. **[should-fix, DEFERRED]** `pack_a!`/`pack_b!` allocate ~80 B/call in
   steady state even with concrete, function-local types (confirmed; not a
   measurement artifact — checked with `KernelDescriptor` directly, bypassing
   the `ScalarKernel` forwarding method, same result). Root cause not
   isolated in the time available: `checked_tile_storage_bounds`, closure
   construction, and `_check_packed_eltype` each measured 0 B in isolation,
   but the full `pack_a!` body still allocates. Not a correctness issue and
   explicitly lower priority than findings 1-3 per the review itself ("measure
   before Phase 3 SIMD reuses these wrappers"). Deferred to whoever next
   touches `pack_a!`/`pack_b!` (packing follow-up or the Phase 3 SIMD
   implementer, who needs zero-allocation packing regardless).
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
