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
| `contract!`, `plan_contract`, `execute!`, `ContractPlan`, `ContractWorkspace`, `Blocking`, `default_blocking`, `ScalarKernel`, `SIMDKernel`, `PlanarKernel`, `OneMKernel` |

Everything not listed above (indexing, packing, tile/kernel internals such
as `AxisGroup`, `KernelDescriptor`, `pack_a!`/`pack_b!`, `accumulate`, etc.)
is internal implementation detail: it carries a docstring and is freely used
within the package and its tests, but no semver promise and no expectation
of stability across releases. See `docs/decisions.md`'s "Public / internal
API split" section for the exhaustive, frozen three-tier list (1 exported +
11 public + internal names) and the rationale (avoiding export collisions
with TensorOperations, e.g. both packages having a `scalartype`).

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
scope, in stage order. `docs/decisions.md` is the design log.

## Status

Implemented and tested (`Pkg.test()`: **13171/13171**, Julia 1.12.6, Xeon
Gold 6244 / Cascade Lake):

- Grouped-axis (block-scatter) indexing with an independent oracle, overflow
  checking, and real `StridedView` integration.
- Checked tiling/packing against one frozen packed format shared by both
  kernels.
- A scalar reference kernel and a `SIMD.jl`-based kernel, confirmed
  register-resident and zero-allocation (`@code_llvm`/`@allocated`);
  **3.9-6.3x faster than scalar** across K-depths 1-256, swappable in the
  driver with no driver changes. `SIMDKernel` is the engine-wide default
  (`_default_kernel`), for `contract!`, `plan_contract`, and the
  `QuasiStridedBackend` TensorOperations path alike; `ScalarKernel` remains
  available and is the correctness oracle the SIMD kernel is checked
  against.
- A BLIS five-loop (`NC`/`KC`/`MC`) macro-blocking `execute!`, packing a
  reusable B panel per `(jc,pc)` block and a reusable A panel per
  `(jc,pc,ic)` block, **1.66x-10.5x faster than the pre-macro-blocking
  tile-by-tile driver** (kept, unexported, as `execute_tilewise!`, the
  correctness oracle it's checked against) across the measured shape/kernel
  grid — single machine, see `docs/decisions.md`'s Phase E section for the
  full sweep and provenance. `Blocking`/`default_blocking` expose tunable,
  measured (not modeled/probed) `mc`/`kc`/`nc` defaults.
- **Packed panels handed to the kernel as borrowed pointers**
  (`PackedPanel`), not `view`s, and an `isbits` scattered axis
  (`PtrScatterAxis`) so `Union{AffineAxis,PtrScatterAxis}` never heap-boxes.
  Together these removed a ~4x throughput cliff above 16 accumulator vectors
  and a 24 KB-per-call allocation on irregular destinations, and they cost no
  new dependency. `execute!` remains allocation-free on Julia >= 1.11 —
  including through permuted/negative-stride/sliced 3-index contractions,
  which is now asserted.
- **A hardware-derived register shape**, so the engine does not need
  retuning per machine. One capability-derived rule — `W =
  vector_bytes/sizeof(T)`, `MR = 2W`, `NR = 6` — is applied to the vector
  ISA detected at load (`target_profile()`; `Sys.CPU_NAME` plus a CPUID
  probe, no external dependency). On AVX-512 that yields `(16,6,8)` for
  `Float64` and `(32,6,16)` for `Float32`. Against the complete previously
  shipped configuration (the `(8,6)` shape plus the Phase E blocking
  constants), the complete new one is **1.26x faster in geomean for
  `Float64` and 1.27x for `Float32`** across the measured shape grid, up to
  1.78x/1.91x at 512³, with the worst point at 0.96x/1.00x (inside this
  machine's measured noise floor). On AVX2 the same rule reproduces the old
  `(8,6,4)` exactly. The rule is applied only on the ISAs it was measured on
  (AVX-512, AVX2); aarch64/NEON and any unrecognized CPU fall back to the
  previous constants bit-identically, so they are unchanged rather than
  guessed at. A contraction whose `M` extent cannot fill one register
  tile is demoted to the smaller shape, so a larger `MR` never costs padding
  on small problems. Single machine measured — see `docs/decisions.md`'s
  Phase G section, and `benchmark/bench_kernel_shape.jl` /
  `benchmark/bench_default_vs_legacy.jl`.
- **Complex element types** (`ComplexF32`/`ComplexF64`), with two switchable
  microkernel methods. The default is **planar** (split-complex, BLIS "1r"):
  packed panels hold separate real and imaginary planes of `real(T)`, so the
  four real products of a complex FMA land already in the right lanes and the
  kernel needs no shuffles. **1m** (Van Zee's induced method) is also
  available by naming the kernel — it reuses the *real* microkernel verbatim
  over `2*kc` real steps, fed by "1e"-packed `A`. 3m is deliberately absent.

  TensorOperations' `conjA`/`conjB` and each operand's `StridedView.op`
  compose by xor and are absorbed at pack time, which is free: a sign flip on
  a value the gather has already loaded. A conjugated *output* is rejected
  rather than silently mishandled.

  Measured on one machine (ccqlin038, Cascade Lake, 21 reps, canary spread
  0.4–3.0%): the **complex efficiency ratio** — complex GFLOP/s over this same
  engine's real GFLOP/s at the same shape, with complex charged the textbook 8
  flops per multiply-accumulate — is **1.83 geomean for `ComplexF64` and 1.91
  for `ComplexF32`**. Above 1.0 means complex is treated *better* than real,
  which is the expected direction: complex is 4× the flops on 2× the bytes,
  i.e. twice the arithmetic intensity, so packing and per-call overheads
  amortise better. Adding complex support did not measurably change the real
  path (pooled geomean 0.99 against the pre-milestone tree, below that
  measurement's ~5% resolution).

  The complex register shape is the one place in the package with a swept,
  machine-specific constant: the shape the hardware rule *derives* turned out
  to be the worst planar configuration measured, by 38–41%, because it spills
  accumulators. `_shape_override` now carries one row per ISA, each with its own
  provenance: AVX-512 swept here, NEON measured on an Apple M3 Max by a sibling
  project, AVX2 adopted from that project's model (and chosen over a
  budget-fitted alternative because the fitted one used *all* 16 ymm registers,
  leaving LLVM no scratch). An ISA with no row takes a register-budget fit.
  Every shipped row leaves at least two registers spare. See
  `docs/decisions.md`'s "Phase F measurement" and Amendments 5 and 6 — the
  latter also records why BLIS, the obvious place to look, is the wrong source
  for *planar* shapes specifically.
- Zero steady-state allocation for `SIMDKernel` through the full driver on
  Julia >= 1.11. On Julia 1.10 (LTS), `SIMDKernel`'s `accumulate`/
  `execute_tile!` allocate tens of KB per call instead — a compiler
  capability gap (the `NTuple{NV,Vec{W,T}}` accumulator isn't kept
  register-resident by the older compiler), not a correctness issue; all
  correctness assertions pass on 1.10, and the affected allocation
  assertions are marked `skip=(VERSION < v"1.11")` rather than weakened or
  removed, so the gap stays visible rather than hidden. Flipping the
  engine-wide default to `SIMDKernel` therefore changes the *allocation*
  profile of the default path on 1.10 while improving it everywhere else.
- A `TensorOperations.jl` backend, `QuasiStridedBackend`, routing
  `@tensor`/`ncon` contractions to this engine; see "Usage" above and "Not
  implemented" below for its scope. Task-local workspace pooling makes the
  default (`DefaultAllocator`) path reuse its packed buffers across repeated
  calls of the same element type, so per-call allocation stays flat instead of
  growing with problem size (a cold, unpooled `plan_contract` allocates
  roughly 860 KiB of buffers for default `Float64` blocking). `execute!`
  itself is allocation-free (with the Julia 1.10 `SIMDKernel` caveat noted
  above); the call as a whole is **not** — `plan_contract`'s
  per-call bookkeeping (label classification, plan construction) still costs a
  few KB per call — ~3.8 KB measured on a 64³ `Float64` contraction, both
  through `@tensor` and through a direct `tensorcontract!`, against tens of
  bytes for `StridedBLAS` on the same call. An explicit
  `allocator=` (e.g. `Bumper`) is honored with call-scoped, non-pooled
  buffers.
- Three independent review passes (two Fable-model, one Sonnet-High),
  findings triaged and fixed, disposition in `docs/decisions.md`.

**Not implemented** (deliberately, this milestone):

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
  probing or analytical block-size derivation (`mc`/`kc`/`nc` remain
  measured constants — the detected cache geometry is reported by
  `cache_topology()` and used for nothing that picks a number at runtime,
  see `docs/decisions.md`'s "Block-size policy" and its Phase G amendment);
  threading; GPU execution; K padding.
- On the complex side specifically: the **3m** (Karatsuba) method; **mixed
  real/complex operands** (promotion belongs in TensorOperations'
  `promote_contract` layer, not here); writing into a **conjugated output**
  view; **measured** complex register shapes for AVX2 or NEON (complex
  *works* there — the engine selects the largest shape that fits the detected
  register file — but those shapes are fitted, not swept, and no throughput
  claim is made for them); and the unit-stride plane-to-interleave store, which
  is measurement-gated and not yet justified.

See the companion design specs and `docs/decisions.md` for the full
deferred list. `benchmark/bench_tensoroperations.jl` (results in
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-09/`) is a
head-to-head comparison against `StridedNative`/`StridedBLAS` on identical
shapes, single machine: `StridedBLAS()` wins on every measured shape/dtype
point (2.4x-10.9x faster than `QuasiStridedBackend()`), which is the evidence
behind leaving `select_backend` unhooked above; `QuasiStridedBackend()` in
turn beats `StridedNative()` by 5.3x-18.7x on every shape. A second,
preparatory comparison against TensorOperations.jl's own (unmerged, PR #303)
upstream benchmark suite lives in `benchmark/bench_to_suite.jl`; see
`docs/decisions.md`'s "Upstream TensorOperations.jl benchmark suite
comparison" section for the measured numbers.
