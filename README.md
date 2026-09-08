# QuasiStrided.jl

A native-Julia dense tensor contraction engine for `StridedView`s
(`StridedViews.jl`): grouped-axis / block-scatter indexing, packing, and
fixed-shape microkernels, in the style of block-scatter-matrix tensor
contraction (BSMTC, Matthews arXiv:1607.00291). This is a **first-milestone
CPU implementation**: one scalar reference kernel, one explicit-SIMD
candidate, and a serial (non-cache-blocked) driver — see "Status and
limitations" below before relying on this for anything performance-critical.

## Installation

Not registered. Develop locally:

```julia
using Pkg
Pkg.develop(path="/mnt/home/ldevos/Projects/QuasiStrided.jl")
```

## Usage

```julia
using QuasiStrided
using StridedViews

# C[a,n,b] = alpha * sum_k A[a,k,b] * B[k,n] + beta * C[a,n,b]
A = randn(3, 5, 2)
B = randn(5, 4)
C = zeros(3, 4, 2)

# One-shot: label each tensor's axes (shared label = contracted if absent
# from indC, free if present in indC); labels 1=a, 2=k, 3=b, 4=n here.
contract!(StridedView(C), 1.0,
          StridedView(A), (1, 2, 3),
          StridedView(B), (2, 4),
          0.0,
          (1, 4, 3))

# Reused plan/workspace, for repeated contractions of the same shape:
plan = plan_contract(StridedView(C), StridedView(A), (1, 2, 3),
                      StridedView(B), (2, 4), (1, 4, 3))
execute!(plan, 1.0, 0.0)   # planning cost (AxisGroup construction, buffer
execute!(plan, 1.0, 0.0)   # sizing) is paid once, not on every execute! call

# Pick a kernel and register-tile shape explicitly (default is
# ScalarKernel(Val(8), Val(6), eltype(C))):
plan = plan_contract(StridedView(C), StridedView(A), (1, 2, 3),
                      StridedView(B), (2, 4), (1, 4, 3);
                      kernel=SIMDKernel(Val(8), Val(6), Float64), kc_panel=64)
```

Run the test suite:

```julia
using Pkg
Pkg.test("QuasiStrided")
```

## API

| Layer | Exports |
| --- | --- |
| Indexing (grouped-axis / block-scatter) | `AxisGroup`, `axis_length`, `offsets`, `fill_offsets!`, `BlockDescriptor`, `describe_block`, `block_descriptors!`, `normalize_group` |
| Kernel shape/format contract | `KernelDescriptor`, `mr`, `nr`, `scalartype`, `packed_a_offset`, `packed_b_offset`, `packed_a_length`, `packed_b_length` |
| Tiles / axis addressing | `AffineAxis`, `ScatterAxis`, `SourceTile`, `DestinationTile`, `axis_from_descriptor`, `nrows`, `ncols`, `axis_offset_range`, `checked_tile_storage_bounds` |
| Packing | `pack_a!`, `pack_b!` |
| Scalar kernel | `ScalarKernel`, `zero_accumulator`, `accumulate`, `scale_tile!`, `store_tile!`, `execute_tile!` |
| SIMD kernel | `SIMDKernel`, `lanewidth`, `avecs_per_column` (same `zero_accumulator`/`accumulate`/`store_tile!`/`execute_tile!` interface, dispatched by kernel type) |
| Driver | `contract!`, `plan_contract`, `execute!`, `ContractPlan` |

Every layer has its own docstrings (`?AxisGroup`, `?pack_a!`, etc.) and its
own focused test file under `test/`. `docs/decisions.md` is the authoritative
record of every frozen interface decision and why it was made that way —
read it before changing any exported signature.

## Status and limitations

Implemented and tested (`Pkg.test("QuasiStrided")`: **12627/12627 passing**
on Julia 1.12.6, Xeon Gold 6244 / Cascade Lake, AVX-512):

- Grouped-axis (block-scatter) indexing: affine/irregular classification,
  overflow-checked construction, normalization, an independent
  `CartesianIndices`-based oracle, and real `StridedView` integration
  (permuted/sliced/negative-stride/zero-stride).
- Checked source/destination tiles and A/B packing against a frozen packed
  format, with padding, transform, and alpha/beta shortcut semantics matched
  exactly between the scalar and SIMD kernels.
- A scalar reference kernel and one explicit-SIMD candidate
  (`SIMD.jl`-based, `NTuple`-of-`Vec` accumulator, confirmed register-resident
  via `@code_llvm` inspection — 12 loop-carried vector `phi`s, zero spills,
  zero steady-state heap allocation). Measured **3.9-6.3x faster than the
  scalar reference** across K-depths 1-256 (single-tile), and **faster with
  lower allocation than scalar through the real driver** on a 64×64×64
  contraction — all on one machine; not a cross-microarchitecture claim.
- A serial `contract!` driver: label-based axis resolution (arbitrary free/
  reduction axis ordering and explicit output ordering), output tiling into
  register-sized chunks, multiple K panels with beta applied exactly once,
  and a `ScalarKernel`/`SIMDKernel` swap that requires no driver changes
  (verified end to end, not just at the single-tile level).
- One completed, bounded Fable-model review (Phase 2b) plus a Sonnet-High
  review (Phase 4); both found real issues, both are fully triaged in
  `docs/decisions.md` with disposition and, where fixed, regression tests.

Explicitly **not** implemented in this milestone (see the two companion
specs and `docs/decisions.md` for the full list): an einsum string parser;
batch indices, repeated labels (diagonals), or isolated reductions; a
separate beta-addend tensor distinct from the destination; cache-blocked
macro-kernel tuning, autotuning, or CPU-feature dispatch tables; threading;
GPU execution / GemmKernels integration; complex-arithmetic methods
(planar/1m/3m); K padding.

Known, deferred, non-correctness issue: `pack_a!`/`pack_b!` allocate ~80
bytes/call in steady state even with concrete, function-local, reused
buffers — root cause not isolated (closures, bounds-check call, and eltype
check each measured zero in isolation, but the full function body still
allocates). Documented in `docs/decisions.md`'s Phase 2b disposition (finding
5) and re-confirmed, not re-investigated, in Phase 3/4. Does not affect
correctness; affects the "zero steady-state allocation" performance target
for packing specifically (the kernels themselves — scalar and SIMD — are
zero-allocation, confirmed by `@allocated` after warmup).

## Development process

Built by a main Claude Code orchestration process delegating bounded,
single-level subagent tasks (file-ownership-scoped implementers, an
independent oracle/test worker, one Fable-model review, one Sonnet-High
review), per `/mnt/home/ldevos/Projects/tensorcontract-rs/julia/QuasiStrided-Claude-Handoff.md`
and its two companion specifications
(`Julia-Tensor-Indexing-Agent-Spec.md`, `Julia-Microkernel-Tile-Interface-Design.md`).
Every phase gate, integration reconciliation, and review disposition is
recorded in `docs/decisions.md`; `STATUS.md` is the phase-by-phase summary.
