# The contract between packing and the microkernels: the packed-panel formats,
# and the descriptor that fixes a kernel's register tile, element type and the
# format of each operand's panel. Packers and kernels both address panels
# through the offset functions below, so the layout is written down once.
#
# Complex support is expressed as "N planes of the real type": packed buffers,
# `SIMD.Vec` lanes and accumulators all work in `real(T)`. One general offset
# formula covers every format,
#
#     p * reg_tile * rpe  +  plane * reg_tile  +  i
#
# and at `rpe == 1, plane == 0` (`RealFormat`) it reduces exactly to the real
# layout `i + MR*p`. Check that reduction before changing either formula.

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

One real per element: `packed_a_offset(k, i, p) == i + mr(k)*p`.
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
    InterleavedFormat()

Two reals per complex element in `Complex{T}`'s own native order, one
region per logical K step: `[re_0, im_0, re_1, im_1, ..., re_{n-1}, im_{n-1}]`.
Operand A under [`FMAddSubMethod`](@ref). Byte for byte the FIRST of
[`OneEFormat`](@ref)'s two regions, without the second (`[-im, re]`) one: the
fmaddsub kernel derives that pair-swap in a register instead of reading it
from memory, so its packed A footprint equals [`PlanarFormat`](@ref)'s, half
of 1e's.

Like 1e (and unlike planar), it does not fit the "`plane * reg_tile`" shape
the plane-offset helpers describe: `packed_a_plane_offset(k, 0, i, p)` is
addressed with `i` running over REALS (`0:2MR-1`), and plane 1 is never used.
"""
struct InterleavedFormat <: PackFormat end

"""
    reals_per_element(::PackFormat) -> Int

Reals emitted per source element per *logical* K step: 1, 2, 2 and 4 for
[`RealFormat`](@ref), [`PlanarFormat`](@ref), [`InterleavedFormat`](@ref) and
[`OneEFormat`](@ref).
"""
reals_per_element(::RealFormat) = 1
reals_per_element(::PlanarFormat) = 2
reals_per_element(::InterleavedFormat) = 2
reals_per_element(::OneEFormat) = 4

# ----------------------------------------------------------------------------
# The descriptor
# ----------------------------------------------------------------------------

"""
    Descriptor{MR,NR,T,FA,FB}

A microkernel's register-tile shape (`MR`x`NR`, `Int` type parameters), its
storage element type `T`, and the packed formats `FA`/`FB` of its A and B
panels. Stateless: it exists to dispatch the packed-offset formulas. Concrete
kernels wrap one as their `.descriptor`.

Two aliases name the shipped combinations: [`KernelDescriptor`](@ref) (real
`T`, `RealFormat` on both sides) and [`ComplexKernelDescriptor`](@ref)
(complex `T`, a complex format on each side). Any other combination is
rejected at construction.

`MR`/`NR` are **logical** extents (complex elements for a complex `T`). Three
notions coincide on the real path and must not be conflated on the complex
one:

| notion | accessor | `Float64` | `ComplexF64` planar | `ComplexF64` 1m |
| --- | --- | --- | --- | --- |
| storage element | `scalartype` | `Float64` | `ComplexF64` | `ComplexF64` |
| packed buffer / `Vec` lane | [`realtype`](@ref) | `Float64` | `Float64` | `Float64` |
| reals per A sliver per logical K step | [`packed_a_per_k`](@ref) | `MR` | `2MR` | `4MR` |
| reals per B sliver per logical K step | [`packed_b_per_k`](@ref) | `NR` | `2NR` | `2NR` |

Every length and offset below takes the **logical** `kc` and returns a count
of **reals**. 1m's internal doubling to `2*kc` real steps is confined to its
own `accumulate` and never appears in a length, an offset, or a driver loop
bound.
"""
struct Descriptor{MR, NR, T, FA <: PackFormat, FB <: PackFormat}
    function Descriptor{MR, NR, T, FA, FB}() where {MR, NR, T, FA, FB}
        _check_descriptor(MR, NR, T, FA, FB)
        return new{MR, NR, T, FA, FB}()
    end
end

"""
    KernelDescriptor{MR,NR,T}

The [`Descriptor`](@ref) of a real kernel: `T ∈ (Float32, Float64)` and
[`RealFormat`](@ref) panels on both sides.
"""
const KernelDescriptor{MR, NR, T} = Descriptor{MR, NR, T, RealFormat, RealFormat}

"""
    ComplexKernelDescriptor{MR,NR,T,FA,FB}

The [`Descriptor`](@ref) of a complex kernel: `T ∈ (ComplexF32, ComplexF64)`,
with [`PlanarFormat`](@ref), [`InterleavedFormat`](@ref) or
[`OneEFormat`](@ref) panels.
"""
const ComplexKernelDescriptor{MR, NR, T <: Complex, FA, FB} = Descriptor{MR, NR, T, FA, FB}

KernelDescriptor(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} = KernelDescriptor{MR, NR, T}()

function ComplexKernelDescriptor(
        ::Val{MR}, ::Val{NR}, ::Type{T}, ::FA, ::FB
    ) where {MR, NR, T, FA <: PackFormat, FB <: PackFormat}
    return Descriptor{MR, NR, T, FA, FB}()
end

# Construction-time only, never hot.
function _check_descriptor(MR, NR, T, FA, FB)
    real_formats = FA === RealFormat && FB === RealFormat
    name = real_formats ? "KernelDescriptor" : "ComplexKernelDescriptor"
    MR isa Int && NR isa Int ||
        throw(ArgumentError("$name requires Int type parameters MR, NR"))
    MR > 0 || throw(ArgumentError("$name requires MR > 0, got MR = $MR"))
    NR > 0 || throw(ArgumentError("$name requires NR > 0, got NR = $NR"))
    if real_formats
        T === Float32 || T === Float64 ||
            throw(ArgumentError("KernelDescriptor requires T ∈ (Float32, Float64), got $T"))
    else
        (FA === RealFormat || FB === RealFormat) && throw(
            ArgumentError(
                "ComplexKernelDescriptor requires complex formats on both operands, got ($FA, $FB)"
            )
        )
        T === ComplexF32 || T === ComplexF64 || throw(
            ArgumentError("ComplexKernelDescriptor requires T in (ComplexF32, ComplexF64), got $T")
        )
    end
    return nothing
end

mr(::Descriptor{MR}) where {MR} = MR
nr(::Descriptor{MR, NR}) where {MR, NR} = NR
scalartype(::Descriptor{MR, NR, T}) where {MR, NR, T} = T

"""
    a_format(descriptor) -> PackFormat
    b_format(descriptor) -> PackFormat

The packed format of each operand's panel.
"""
a_format(::Descriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FA()
b_format(::Descriptor{MR, NR, T, FA, FB}) where {MR, NR, T, FA, FB} = FB()

"""
    realtype(kernel) -> Type

The real type the packed buffers and `SIMD.Vec` lanes are made of: `T` itself
for a real kernel, `real(T)` for a complex one. This is the eltype of
`ContractWorkspace`'s packed panels, which is *not* `scalartype(kernel)` once
the element type is complex.
"""
realtype(::Descriptor{MR, NR, T}) where {MR, NR, T} = real(T)

"""
    packed_a_per_k(kernel) -> Int
    packed_b_per_k(kernel) -> Int

Reals in one A / B sliver per **logical** K step: `mr(kernel)`/`nr(kernel)`
scaled by [`reals_per_element`](@ref) of the operand's format, so exactly
`mr`/`nr` for a real kernel.
"""
packed_a_per_k(d::Descriptor{MR}) where {MR} = MR * reals_per_element(a_format(d))
packed_b_per_k(d::Descriptor{MR, NR}) where {MR, NR} = NR * reals_per_element(b_format(d))

"""
    packed_a_length(kernel, kc::Int) -> Int
    packed_b_length(kernel, kc::Int) -> Int

Minimum packed-buffer capacity, in reals, for a full A/B panel at *logical*
K depth `kc` (`kc == 0` yields 0).
"""
packed_a_length(d::Descriptor, kc::Int) = packed_a_per_k(d) * kc
packed_b_length(d::Descriptor, kc::Int) = packed_b_per_k(d) * kc

"""
    packed_a_plane_offset(descriptor, plane::Int, i::Int, p::Int) -> Int
    packed_b_plane_offset(descriptor, plane::Int, j::Int, p::Int) -> Int

Zero-based offset, in reals, of plane `plane` of A's row `i` (B's column `j`) at
logical K step `p`:

    p * packed_*_per_k + plane * reg_tile + index

At `RealFormat` (one plane, `reals_per_element == 1`) this is exactly
`i + MR*p`, i.e. [`packed_a_offset`](@ref).
"""
@inline packed_a_plane_offset(d::Descriptor{MR}, plane::Int, i::Int, p::Int) where {MR} =
    p * packed_a_per_k(d) + plane * MR + i
@inline packed_b_plane_offset(d::Descriptor{MR, NR}, plane::Int, j::Int, p::Int) where {MR, NR} =
    p * packed_b_per_k(d) + plane * NR + j

"""
    packed_a_offset(kernel::KernelDescriptor, i::Int, p::Int) -> Int

Zero-based packed offset of A's entry `(row i, K-step p)` in an `(MR, kc)`
panel: `i + MR*p`. Real descriptors only: a complex element has no single
offset, so a complex descriptor has no method (use
[`packed_a_plane_offset`](@ref)).
"""
packed_a_offset(kernel::KernelDescriptor{MR}, i::Int, p::Int) where {MR} = i + MR * p

"""
    packed_b_offset(kernel::KernelDescriptor, j::Int, p::Int) -> Int

Zero-based packed offset of B's entry `(col j, K-step p)` in an `(NR, kc)`
panel: `j + NR*p` (not column-major). Real descriptors only, like
[`packed_a_offset`](@ref).
"""
packed_b_offset(kernel::KernelDescriptor{MR, NR}, j::Int, p::Int) where {MR, NR} = j + NR * p
