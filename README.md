# QuasiStrided.jl

[![CI](https://github.com/lkdvos/QuasiStrided.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lkdvos/QuasiStrided.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A native-Julia dense tensor contraction engine for `StridedView`s
(`StridedViews.jl`): grouped-axis / block-scatter indexing, packing, and
fixed-shape microkernels, in the style of block-scatter-matrix tensor
contraction (BSMTC, Matthews arXiv:1607.00291).

**First-milestone status**: not registered, not yet performance-tuned beyond
a scalar reference and one SIMD kernel candidate. See "Status" below.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/lkdvos/QuasiStrided.jl")
```

## Usage

```julia
using QuasiStrided, StridedViews

# C[a,n,b] = alpha * sum_k A[a,k,b] * B[k,n] + beta * C[a,n,b]
A, B, C = randn(3, 5, 2), randn(5, 4), zeros(3, 4, 2)

# indA/indB/indC label each axis; a label shared by A and B but absent from
# indC is contracted, one present in indC is kept (M if from A, N if from B).
contract!(StridedView(C), 1.0,
          StridedView(A), (1, 2, 3),   # a, k, b
          StridedView(B), (2, 4),      # k, n
          0.0,
          (1, 4, 3))                   # a, n, b

# Reuse a plan/workspace across repeated calls of the same shape, and pick a
# kernel explicitly (default: ScalarKernel):
plan = plan_contract(StridedView(C), StridedView(A), (1, 2, 3),
                      StridedView(B), (2, 4), (1, 4, 3);
                      kernel=SIMDKernel(Val(8), Val(6), Float64))
execute!(plan, 1.0, 0.0)
```

## API

| Layer | Exports |
| --- | --- |
| Indexing | `AxisGroup`, `axis_length`, `offsets`, `fill_offsets!`, `BlockDescriptor`, `describe_block`, `block_descriptors!`, `normalize_group` |
| Kernel format | `KernelDescriptor`, `mr`, `nr`, `scalartype`, `packed_a_offset`, `packed_b_offset`, `packed_a_length`, `packed_b_length` |
| Tiles | `AffineAxis`, `ScatterAxis`, `SourceTile`, `DestinationTile`, `axis_from_descriptor`, `nrows`, `ncols` |
| Packing | `pack_a!`, `pack_b!` |
| Kernels | `ScalarKernel`, `SIMDKernel`, `zero_accumulator`, `accumulate`, `scale_tile!`, `store_tile!`, `execute_tile!` |
| Driver | `contract!`, `plan_contract`, `execute!`, `ContractPlan` |

Every exported name has a docstring. `docs/decisions.md` is the authoritative
record of every frozen interface and why; read it before changing any public
signature.

## Status

Implemented and tested (`Pkg.test()`: **12630/12630**, Julia 1.12.6, Xeon
Gold 6244 / Cascade Lake):

- Grouped-axis (block-scatter) indexing with an independent oracle, overflow
  checking, and real `StridedView` integration.
- Checked tiling/packing against one frozen packed format shared by both
  kernels.
- A scalar reference kernel and a `SIMD.jl`-based kernel, confirmed
  register-resident and zero-allocation (`@code_llvm`/`@allocated`);
  **3.9-6.3x faster than scalar** across K-depths 1-256, swappable in the
  driver with no driver changes.
- A serial `contract!` driver: arbitrary free/reduction axis labeling, output
  tiling, multiple K panels with beta applied exactly once.
- Two independent review passes (Fable-model + Sonnet-High), findings
  triaged and fixed, disposition in `docs/decisions.md`.

**Not implemented** (deliberately, this milestone): einsum string parsing;
batch axes, diagonals, or isolated reductions; a separate beta-addend tensor;
cache-blocked macro-kernel tuning, autotuning, or CPU-dispatch tables;
threading; GPU execution; complex-arithmetic methods (planar/1m/3m); K
padding. See the companion design specs for the full deferred list.

**Known open item**: `execute!`'s driver loop has a small, not yet
diagnosed, non-zero steady-state allocation (both kernels' own packing and
arithmetic are independently zero-allocation) — see `STATUS.md`.
