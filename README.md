# QuasiStrided.jl

[![CI](https://github.com/lkdvos/QuasiStrided.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lkdvos/QuasiStrided.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A native-Julia dense tensor contraction engine for `StridedView`s
(`StridedViews.jl`): grouped-axis / block-scatter indexing, packing, and
fixed-shape microkernels, in the style of block-scatter-matrix tensor
contraction (BSMTC, Matthews arXiv:1607.00291). It ships as a
[TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl)
backend, `QuasiStridedBackend`.

**Status**: not registered. Includes a BLIS five-loop (`NC`/`KC`/`MC`)
macro-blocking driver with packed-panel reuse, on top of a scalar reference
and a SIMD kernel (the engine-wide default), plus planar and 1m complex
microkernels for `ComplexF32`/`ComplexF64`. See "Status" below.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/lkdvos/QuasiStrided.jl")
```

## Usage

The primary, intended way to use this package is through TensorOperations'
own `@tensor`/`ncon` macros, opting into QuasiStrided's engine with
`backend=QuasiStridedBackend()`:

```julia
using TensorOperations, QuasiStrided

A, B = randn(3, 5, 2), randn(5, 4)

# C[a,n,b] = sum_k A[a,k,b] * B[k,n]
@tensor backend = QuasiStridedBackend() C[a, n, b] := A[a, k, b] * B[k, n]

# equivalently, via ncon
C2 = ncon((A, B), ((-1, 1, -3), (1, -2)); backend = QuasiStridedBackend())
```

Loading QuasiStrided never changes the behavior of plain `@tensor` code —
`QuasiStridedBackend` is *not* hooked into `TensorOperations.select_backend`,
so it only ever runs when named explicitly. See "Not implemented" below for
what it accepts and rejects.

### Engine interface (direct/low-level usage)

`QuasiStridedBackend` is a thin adapter over QuasiStrided's own contraction
engine, which remains available directly (and is what the TensorOperations
backend calls internally):

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
# kernel explicitly (default: SIMDKernel):
plan = plan_contract(StridedView(C), StridedView(A), (1, 2, 3),
                      StridedView(B), (2, 4), (1, 4, 3);
                      kernel=SIMDKernel(Val(8), Val(6), Float64))
execute!(plan, 1.0, 0.0)
```

`plan_contract` also accepts `workspace=`/`allocator=` keywords to control
buffer reuse and, e.g., arena/Bumper-backed allocation instead of the
default GC-owned, task-local pool; see the docstring for details.

## API

QuasiStrided exports exactly one name:

| Name | Kind |
| --- | --- |
| `QuasiStridedBackend` | Exported |

Everything else the engine needs to be used directly is `public` but
unexported (Julia >= 1.11; visible via `QuasiStrided.<name>` on 1.10):

| Public, unexported names |
| --- |
| `contract!`, `plan_contract`, `execute!`, `ContractPlan`, `ContractWorkspace`, `Blocking`, `default_blocking`, `ScalarKernel`, `SIMDKernel`, `PlanarKernel`, `OneMKernel`, `target_profile`, `cache_topology`, `TargetProfile`, `CacheLevel` |

Everything not listed above (indexing, packing, tile/kernel internals such
as `AxisGroup`, `KernelDescriptor`, `pack_a!`/`pack_b!`, `accumulate`, etc.)
is internal implementation detail: it carries a docstring and is freely used
within the package and its tests, but no semver promise and no expectation
of stability across releases. The split has three tiers (1 exported + 15
public + internal names); keeping most names unexported avoids export
collisions with TensorOperations (e.g. both packages have a `scalartype`).

## Code map

`src/` follows the algorithm. `plan_contract` runs the planning stages once;
`execute!` runs a BLIS five-loop nest that packs slivers of `A` and `B` into
contiguous panels and drives a microkernel over register tiles of `C`.

| Folder | Stage | Contents |
| --- | --- | --- |
| `src/hardware/` | detection | ISA, vector width, register count and cache topology, detected once per process (`target_profile()`) |
| `src/layout/` | addressing | `AxisGroup` (a grouped tensor axis as zero-based offsets), block descriptors, and the affine/scattered tile axes and `QSTile` the packers and kernels read and write through |
| `src/packing/` | packing | the packed-panel formats and the `Descriptor` that fixes them per kernel (`format.jl`), borrowed-pointer panels, and the `pack_a!`/`pack_b!` packers with their contiguous fast paths |
| `src/microkernels/` | microkernels | the shared kernel interface (`execute_tile!` = `zero_accumulator` + `accumulate` + `store_tile!`) and the scalar, SIMD, planar-complex and 1m-complex kernels |
| `src/planning/` | planning | label classification and ordering, conjugation, kernel selection from the detected hardware, cache blocking, and `plan_contract`/`ContractPlan` |
| `src/execution/` | execution | the reusable `ContractWorkspace`, the five-loop nest behind `execute!`/`contract!`, its macro-kernel helpers, and `execute_tilewise!`, the tile-by-tile correctness oracle |
| `src/integrations/` | adapters | `QuasiStridedBackend` for TensorOperations.jl |

`test/` mirrors this layout; `test/runtests.jl` includes every file into one
scope, in stage order.

## Status

The package implements:

- Grouped-axis (block-scatter) indexing: each tensor axis group is a list of
  zero-based offsets, checked for overflow, over a `StridedView`.
- Packing of `A` and `B` slivers into kernel-specific panel formats, fixed per
  kernel by one descriptor, with contiguous fast paths.
- Four microkernels: `ScalarKernel` (the reference), `SIMDKernel` (the default
  for real element types), `PlanarKernel` (split-complex, the default for
  complex) and `OneMKernel` (the 1m induced method, reusing the real kernel
  over `2*kc` steps).
- A BLIS five-loop (`NC`/`KC`/`MC`) macro-blocking `execute!` that reuses a
  packed B panel per `(jc,pc)` block and a packed A panel per `(jc,pc,ic)`
  block. `Blocking`/`default_blocking` expose measured `mc`/`kc`/`nc`
  defaults.
- A register shape derived from the detected hardware (`target_profile()`:
  ISA, vector width, register count), with per-ISA overrides and a
  register-budget fit for ISAs without one. When `M` is too small to fill one
  register tile, the kernel is demoted to the conservative fitted shape. For
  real element types and a shallow contracted extent, a kernel whose `MR` does
  not divide `C`'s leading unit-stride run is demoted to a menu shape that
  does, so the vectorized store applies.
- `ComplexF32`/`ComplexF64` support; `conjA`/`conjB` and each operand's
  `StridedView.op` are folded into packing. A conjugated output is rejected.
- Zero steady-state allocation for `execute!` on Julia >= 1.11. On Julia 1.10
  the older compiler does not keep `SIMDKernel`'s accumulator in registers, so
  it allocates per call; results are still correct.
- A TensorOperations.jl backend, `QuasiStridedBackend`, with task-local
  workspace pooling for the default allocator and call-scoped buffers for an
  explicit `allocator=` (e.g. `Bumper`). `plan_contract`'s per-call
  bookkeeping still allocates a few KB.
- `execute_tilewise!`, a tile-by-tile driver used in the tests as the
  correctness oracle for `execute!`.

**Not implemented** (deliberately):

- **Contraction only, engine-wise.** `QuasiStridedBackend` implements
  `TensorOperations.tensorcontract!` itself; there is no QuasiStrided analog
  of a standalone add/permute step or of diagonal/trace support at all in
  the engine. `tensoradd!`/`tensortrace!` fall back to `StridedNative()` so
  a `@tensor` network mixing a contraction with an add/permute or trace step
  can still run wholesale under `backend=QuasiStridedBackend()` -- a timing
  taken on those two specific operations under this backend measures
  `StridedNative`, not this engine.
- **One shared element type, out of four.** `tensorcontract!` accepts inputs
  that share a single element type out of
  `Float32`/`Float64`/`ComplexF32`/`ComplexF64` and are all strided; anything
  else (mixed eltypes, mixed real-and-complex operands, other eltypes, a
  non-strided array, an aliased output, a conjugated *output* view) raises
  `ArgumentError` rather than silently falling back to another backend, so a
  timing taken with `backend=QuasiStridedBackend()` always measures this
  engine.
- **Not the default backend.** QuasiStrided defines no
  `TensorOperations.select_backend` method, so merely loading it never
  changes the behavior, performance, or allocation profile of existing
  `@tensor`/`ncon` code that doesn't pass `backend=QuasiStridedBackend()`
  explicitly. This is deliberate: QuasiStrided is expected to lose to
  `StridedBLAS()` on most shapes, so silently capturing the default dispatch
  would be a regression for existing users.
- einsum string parsing; batch axes, diagonals, or isolated reductions; a
  separate beta-addend tensor; autotuning across shapes; runtime cache
  probing (no benchmark ever runs at load or first call; `mc`/`kc`/`nc` are
  picked by an analytical model of the sysfs/sysctl-detected cache geometry,
  with fixed constants only where that is undetected -- see `default_blocking`); threading; GPU execution; K padding.
- On the complex side specifically: the **3m** (Karatsuba) method; **mixed
  real/complex operands** (promotion belongs in TensorOperations'
  `promote_contract` layer, not here); writing into a **conjugated output**
  view; **measured** complex register shapes for AVX2 or NEON (complex
  *works* there — the engine selects the largest shape that fits the detected
  register file — but those shapes are fitted, not swept, and no throughput
  claim is made for them); and the unit-stride plane-to-interleave store, which
  is measurement-gated and not yet justified.

`benchmark/bench_tensoroperations.jl` is a head-to-head comparison against
`StridedNative`/`StridedBLAS` on identical shapes. `StridedBLAS()` is faster
than `QuasiStridedBackend()` on the measured shapes, which is why the backend
is not hooked into `select_backend`; `QuasiStridedBackend()` in turn beats
`StridedNative()`. `benchmark/bench_to_suite.jl` runs the same comparison
over TensorOperations.jl's upstream benchmark suite.
