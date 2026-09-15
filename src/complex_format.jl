# Packed-panel formats for complex element types, and the descriptor that names
# them. Complex support is deliberately expressed as "N planes of the real
# type": everything below this boundary -- packed buffers, `SIMD.Vec` lanes,
# accumulators -- works exclusively in `real(T)`, which is what makes the planar
# (split-complex) strategy fall out of the design rather than being bolted on.
#
# The frozen packed format in kernel_descriptor.jl is NOT redefined or extended.
# This file introduces a strictly more general offset formula,
#
#     p * reg_tile * rpe  +  plane * reg_tile  +  i
#
# under new names on a new type. At `rpe == 1, plane == 0` it reduces exactly to
# the frozen `i + MR*p`, so the frozen layout is the `RealFormat` instance of
# the general formula and every existing caller is untouched.

# ----------------------------------------------------------------------------
# Packed formats
# ----------------------------------------------------------------------------

"""
    PackFormat

What packing emits for one operand, as a singleton type so it is a compile-time
constant. [`reals_per_element`](@ref) is the single number that propagates to
panel sizing, sliver addressing and cache blocking.
"""
abstract type PackFormat end

"""
    RealFormat()

One real per element: the frozen format of [`KernelDescriptor`](@ref)
(`packed_a_offset(k, i, p) == i + mr(k)*p`).
"""
struct RealFormat <: PackFormat end

"""
    PlanarFormat()

Two planes per logical K step -- all real parts, then all imaginary parts:
`[re_0 .. re_{n-1} | im_0 .. im_{n-1}]`. BLIS's "1r". Used for both operands
under [`PlanarMethod`](@ref), and for operand B under [`OneMMethod`](@ref);
those two are bit-identical, not merely similar.
"""
struct PlanarFormat <: PackFormat end

"""
    OneEFormat()

Four reals per complex element: the real `2x2` block `[[re, -im], [im, re]]`,
stored as two real K steps of `2n`. BLIS's "1e", operand A under
[`OneMMethod`](@ref). Twice the packed footprint of [`PlanarFormat`](@ref) --
the whole structural cost of 1m over planar, and the reason 1m must be given a
proportionally smaller `mc`.
"""
struct OneEFormat <: PackFormat end

"""
    reals_per_element(::PackFormat) -> Int

Reals emitted per source element per *logical* K step: 1, 2 and 4 for
[`RealFormat`](@ref), [`PlanarFormat`](@ref) and [`OneEFormat`](@ref).
"""
reals_per_element(::RealFormat) = 1
reals_per_element(::PlanarFormat) = 2
reals_per_element(::OneEFormat) = 4

# ----------------------------------------------------------------------------
# Complex methods
# ----------------------------------------------------------------------------

"""
    ComplexMethod

Which complex-arithmetic method a kernel implements. Singleton types, so
`default_blocking` and the shape menus dispatch on them without a runtime
branch.

**No auto-dispatch rule is derived from any measurement.** The sibling project
`tensorcontract-rs` measured four different method orderings on four machines
(Cascade Lake, Ice Lake, portable scalar, Apple M3 Max -- where 3m wins
outright), and records "ranking anything below planar without naming the machine
is a mistake this project has made twice". [`PlanarMethod`](@ref) is the
unconditional default; [`OneMMethod`](@ref) is selected only by naming the
kernel. Same reasoning, same precedent, as `_shape_override` in src/driver.jl.
"""
abstract type ComplexMethod end

"""
    RealMethod()

Not a complex method: what `complex_method` reports for a real kernel, so that
`default_blocking` and the shape menus are total without a `T <: Complex` guard
at every call site.
"""
struct RealMethod <: ComplexMethod end

"""
    PlanarMethod()

Split-complex: both operands [`PlanarFormat`](@ref), driven by a genuinely
complex microkernel that issues four real FMAs per (A-vector, B-scalar) pair on
data already in the right lanes -- no shuffles, no `fmaddsub`. The default.
"""
struct PlanarMethod <: ComplexMethod end

"""
    OneMMethod()

Van Zee's induced 1m: one *real* microkernel of shape `2mr x nr` run over
`2*kc` real steps, fed by [`OneEFormat`](@ref) A and [`PlanarFormat`](@ref) B.
The kernel body is the real `SIMDKernel`'s, reused verbatim -- so a
planar-vs-1m measurement compares two methods, not two hand-written kernels.
"""
struct OneMMethod <: ComplexMethod end

"""
    a_reals(::ComplexMethod) -> Int
    b_reals(::ComplexMethod) -> Int

Reals per element in the packed A / B panel under this method. These are what
`default_blocking` divides the measured real `mc`/`nc` by, so that every method
gets the *same packed byte budget* rather than the same element count -- the
1m `mc` halving is derived from this, never tabulated.
"""
a_reals(::RealMethod) = 1
b_reals(::RealMethod) = 1
a_reals(::PlanarMethod) = 2
b_reals(::PlanarMethod) = 2
a_reals(::OneMMethod) = 4
b_reals(::OneMMethod) = 2

"""
    accumulator_planes(::ComplexMethod) -> Int

Accumulator planes the microkernel holds live. Planar keeps separate real and
imaginary planes (so its register budget is twice a real kernel's at the same
`(mr, nr)`); 1m keeps one plane over a doubled real row count. Used by the
register-budget assertion, which must not assume the real kernel's shape.
"""
accumulator_planes(::RealMethod) = 1
accumulator_planes(::PlanarMethod) = 2
accumulator_planes(::OneMMethod) = 1

# ----------------------------------------------------------------------------
# The complex descriptor
# ----------------------------------------------------------------------------

"""
    ComplexKernelDescriptor{MR,NR,T,FA,FB}

Complex counterpart of [`KernelDescriptor`](@ref). `MR`/`NR` are the **logical**
(complex) register-tile extents; `T` is the **storage** element type
(`ComplexF32`/`ComplexF64`); `FA`/`FB` are the packed formats of the A and B
panels.

Three types are distinct here that coincide on the real path, and conflating
them is the main hazard in this file:

| notion | accessor | `ComplexF64` planar | `ComplexF64` 1m |
| --- | --- | --- | --- |
| storage element | `scalartype` | `ComplexF64` | `ComplexF64` |
| packed buffer / `Vec` lane | [`realtype`](@ref) | `Float64` | `Float64` |
| reals per A sliver per logical K step | [`packed_a_per_k`](@ref) | `2MR` | `4MR` |
| reals per B sliver per logical K step | [`packed_b_per_k`](@ref) | `2NR` | `2NR` |

Every length and offset below takes the **logical** `kc` (in complex elements)
and returns a count of **reals**. 1m's internal doubling to `2*kc` real steps is
confined to its own `accumulate` and never appears in a length, an offset, or a
driver loop bound.
"""
struct ComplexKernelDescriptor{MR, NR, T, FA <: PackFormat, FB <: PackFormat}
    function ComplexKernelDescriptor{MR, NR, T, FA, FB}() where {MR, NR, T, FA, FB}
        MR isa Int && NR isa Int ||
            throw(ArgumentError("ComplexKernelDescriptor requires Int type parameters MR, NR"))
        MR > 0 || throw(ArgumentError("ComplexKernelDescriptor requires MR > 0, got MR = $MR"))
        NR > 0 || throw(ArgumentError("ComplexKernelDescriptor requires NR > 0, got NR = $NR"))
        T === ComplexF32 || T === ComplexF64 ||
            throw(
            ArgumentError(
                "ComplexKernelDescriptor requires T in (ComplexF32, ComplexF64), got $T"
            )
        )
        return new{MR, NR, T, FA, FB}()
    end
end

function ComplexKernelDescriptor(
        ::Val{MR}, ::Val{NR}, ::Type{T}, ::FA, ::FB
    ) where {MR, NR, T, FA <: PackFormat, FB <: PackFormat}
    return ComplexKernelDescriptor{MR, NR, T, FA, FB}()
end

mr(::ComplexKernelDescriptor{MR}) where {MR} = MR
nr(::ComplexKernelDescriptor{MR, NR}) where {MR, NR} = NR
scalartype(::ComplexKernelDescriptor{MR, NR, T}) where {MR, NR, T} = T

"""
    a_format(descriptor) -> PackFormat
    b_format(descriptor) -> PackFormat

The packed format of each operand's panel.
"""
a_format(::ComplexKernelDescriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FA()
b_format(::ComplexKernelDescriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FB()

a_format(::KernelDescriptor) = RealFormat()
b_format(::KernelDescriptor) = RealFormat()

"""
    realtype(kernel) -> Type

The real type the packed buffers and `SIMD.Vec` lanes are made of: `T` itself
for a real kernel, `real(T)` for a complex one. This is the eltype of
`ContractWorkspace`'s packed panels, which is *not* `scalartype(kernel)` once
the element type is complex.
"""
realtype(::KernelDescriptor{MR, NR, T}) where {MR, NR, T} = T
realtype(::ComplexKernelDescriptor{MR, NR, T}) where {MR, NR, T} = real(T)

"""
    packed_a_per_k(kernel) -> Int
    packed_b_per_k(kernel) -> Int

Reals in one A / B sliver per **logical** K step: `mr(kernel)` and `nr(kernel)`
for a real kernel, scaled by [`reals_per_element`](@ref) of the operand's format
otherwise. The Julia counterpart of `tensorcontract-rs`'s `Ukr::a_per_k`, and
the only new quantity the driver's sliver addressing reads -- for every real
kernel it is identically `mr`/`nr`, so the substitution at those call sites is
provably the identity.
"""
packed_a_per_k(::KernelDescriptor{MR}) where {MR} = MR
packed_b_per_k(::KernelDescriptor{MR, NR}) where {MR, NR} = NR
packed_a_per_k(d::ComplexKernelDescriptor{MR}) where {MR} = MR * reals_per_element(a_format(d))
packed_b_per_k(d::ComplexKernelDescriptor{MR, NR}) where {MR, NR} =
    NR * reals_per_element(b_format(d))

"""
    packed_a_length(descriptor::ComplexKernelDescriptor, kc::Int) -> Int
    packed_b_length(descriptor::ComplexKernelDescriptor, kc::Int) -> Int

Minimum packed-buffer capacity in **reals** for a full panel at *logical* K
depth `kc`.
"""
packed_a_length(d::ComplexKernelDescriptor, kc::Int) = packed_a_per_k(d) * kc
packed_b_length(d::ComplexKernelDescriptor, kc::Int) = packed_b_per_k(d) * kc

"""
    packed_a_plane_offset(descriptor, plane::Int, i::Int, p::Int) -> Int
    packed_b_plane_offset(descriptor, plane::Int, j::Int, p::Int) -> Int

Zero-based offset, in reals, of plane `plane` of A's row `i` (B's column `j`) at
logical K step `p`:

    p * packed_*_per_k + plane * reg_tile + index

At `RealFormat` (one plane, `reals_per_element == 1`) this is exactly the frozen
`i + MR*p`. The packer and the microkernel both address through these, so the
layout is written down once.
"""
@inline packed_a_plane_offset(d::ComplexKernelDescriptor{MR}, plane::Int, i::Int, p::Int) where {MR} =
    p * packed_a_per_k(d) + plane * MR + i
@inline packed_b_plane_offset(
    d::ComplexKernelDescriptor{MR, NR}, plane::Int, j::Int, p::Int
) where {MR, NR} = p * packed_b_per_k(d) + plane * NR + j

"""
    complex_method(kernel) -> ComplexMethod

Which complex method a kernel implements; [`RealMethod`](@ref) for every kernel
that has not said otherwise, which keeps `default_blocking` and the shape menus
total.
"""
complex_method(::Any) = RealMethod()

# ----------------------------------------------------------------------------
# DescriptorKernel forwarding, mirroring the block in src/kernel.jl
# ----------------------------------------------------------------------------

realtype(k::DescriptorKernel) = realtype(k.descriptor)
packed_a_per_k(k::DescriptorKernel) = packed_a_per_k(k.descriptor)
packed_b_per_k(k::DescriptorKernel) = packed_b_per_k(k.descriptor)
a_format(k::DescriptorKernel) = a_format(k.descriptor)
b_format(k::DescriptorKernel) = b_format(k.descriptor)
