# Packs against the frozen physical formats in kernel_descriptor.jl
# (A: i + MR*p, B: j + NR*p; do not redefine).

# `Vec`/`vload`/`vstore` already arrive via src/panel.jl's `using`; the complex
# fast path at the bottom of this file additionally needs the compile-time-index
# shuffle (SIMD.jl v3, `simdvec.jl`), which is one LLVM `shufflevector`
# instruction and never a runtime gather.
using SIMD: shufflevector

# Explicit runtime check (not dispatch) so a mismatch raises ArgumentError.
@inline function _check_packed_eltype(packed, kernel::KernelDescriptor{MR, NR, T2}) where {MR, NR, T2}
    eltype(packed) === T2 ||
        throw(
        ArgumentError(
            "packed buffer eltype $(eltype(packed)) does not match kernel scalar type $T2"
        )
    )
    return nothing
end

# Explicit runtime check (not dispatch), mirroring the real method above. The
# packed buffer holds `realtype(kernel)`, which is *not* `scalartype(kernel)`
# once the element type is complex -- that conflation is the main hazard in the
# complex half of this file.
@inline function _check_packed_eltype(
        packed, kernel::ComplexKernelDescriptor{MR, NR, T2}
    ) where {MR, NR, T2}
    R = realtype(kernel)
    eltype(packed) === R ||
        throw(
        ArgumentError(
            "packed buffer eltype $(eltype(packed)) does not match kernel real type $R " *
                "(scalar type $T2)"
        )
    )
    return nothing
end

# ----------------------------------------------------------------------------
# Shared argument validation for all four pack_a!/pack_b! methods
# ----------------------------------------------------------------------------
# Extent bound, nonnegative `kc`, packed capacity, then (Phase 2b) the
# one-time storage-bounds check before any `@inbounds` loop. Returns
# `(valid, kc)`; `kc == 0` is the no-op the caller returns from. One bound type
# parameter per argument, as at `pack_a!` in src/kernel.jl.
#
# `packed_a_length`/`packed_b_length` count ELEMENTS for a real descriptor and
# REALS for a complex one, at the logical `kc` in both cases, so the same check
# serves both without knowing which it has.

# `BOUNDS` is a compile-time flag, not a runtime one: at `Val(true)` the body
# below is the code this function has always generated, and at `Val(false)`
# the `checked_tile_storage_bounds` call is folded away entirely. Only
# `unsafe_pack_a!`/`unsafe_pack_b!` ever pass `Val(false)`, and only from a
# caller that has already validated the WHOLE macro block this sliver belongs
# to (src/driver.jl, `_execute_nest!`). Every other check -- extents, the
# packed-buffer capacity, the `kc == 0` no-op -- is kept in both modes.
@inline _check_pack_a(packed::V, source::QSTile, kernel::K) where {V, K} =
    _check_pack_a(packed, source, kernel, Val(true))

@inline function _check_pack_a(
        packed::V, source::QSTile, kernel::K, ::Val{BOUNDS}
    ) where {V, K, BOUNDS}
    MR = mr(kernel)
    m = nrows(source)
    kc = ncols(source)
    (0 <= m <= MR) ||
        throw(ArgumentError("pack_a!: source row count m=$m must satisfy 0 <= m <= mr(kernel)=$MR"))
    kc >= 0 || throw(ArgumentError("pack_a!: source column count (kc) must be nonnegative, got $kc"))
    needed = packed_a_length(kernel, kc)
    length(packed) >= needed ||
        throw(
        DimensionMismatch(
            "pack_a!: packed buffer has length $(length(packed)), " *
                "need at least packed_a_length(kernel, kc=$kc) = $needed"
        )
    )
    kc == 0 && return (m, 0)
    BOUNDS && checked_tile_storage_bounds(source)
    return (m, kc)
end

@inline _check_pack_b(packed::V, source::QSTile, kernel::K) where {V, K} =
    _check_pack_b(packed, source, kernel, Val(true))

@inline function _check_pack_b(
        packed::V, source::QSTile, kernel::K, ::Val{BOUNDS}
    ) where {V, K, BOUNDS}
    NR = nr(kernel)
    kc = nrows(source)
    n = ncols(source)
    kc >= 0 || throw(ArgumentError("pack_b!: source row count (kc) must be nonnegative, got $kc"))
    (0 <= n <= NR) ||
        throw(ArgumentError("pack_b!: source column count n=$n must satisfy 0 <= n <= nr(kernel)=$NR"))
    needed = packed_b_length(kernel, kc)
    length(packed) >= needed ||
        throw(
        DimensionMismatch(
            "pack_b!: packed buffer has length $(length(packed)), " *
                "need at least packed_b_length(kernel, kc=$kc) = $needed"
        )
    )
    kc == 0 && return (n, 0)
    BOUNDS && checked_tile_storage_bounds(source)
    return (n, kc)
end

# ----------------------------------------------------------------------------
# Real packing
# ----------------------------------------------------------------------------

# Shared inner loop for pack_a!/pack_b!; `load`/`packed_offset` close over the
# operand-specific index mapping. `kc == 0` is handled by the caller.
#
# `PD` (the physical dim: MR or NR) is a compile-time constant. A full sliver
# (`valid == PD`) gets a constant-trip-count inner loop that LLVM fully
# unrolls; a tail sliver writes its valid lanes and then its zero padding as
# two separate loops. Either way no loop body holds a conditional load: the
# previous `i < valid ? load : zero` select compiled to a per-element branch
# around the load, which blocked if-conversion and kept the whole loop scalar
# (1.3-1.7x on B, 1.5-2x on A's fallback, measured on ccqlin038 / Julia
# 1.13 against the driver's argument types). The per-K-step store order
# (0, 1, ..., PD-1) and the padding contract (padding lanes never read
# `source` and never call `transform`) are unchanged.
@inline function _pack_panel!(
        packed::V, ::Type{T}, ::Val{PD}, kc::Int, valid::Int,
        transform::F, load::L, packed_offset::P
    ) where {V, T, PD, F, L, P}
    if valid == PD
        @inbounds for p in 0:(kc - 1)
            for i in 0:(PD - 1)
                panel_store!(packed, packed_offset(i, p), convert(T, transform(load(i, p)))::T)
            end
        end
    else
        @inbounds for p in 0:(kc - 1)
            for i in 0:(valid - 1)
                panel_store!(packed, packed_offset(i, p), convert(T, transform(load(i, p)))::T)
            end
            for i in valid:(PD - 1)
                panel_store!(packed, packed_offset(i, p), zero(T))
            end
        end
    end
    return packed
end

# `transform` is `identity` or `conj` (src/driver.jl, `plan_contract`). On a
# real element type `conj` is the identity, so both admit a straight copy.
# This is only the *value* half of the eligibility test: a straight copy is
# vectorizable only in conjunction with the eltype/storage/destination
# conditions in `_pack_a_contiguous_eligible` below.
@inline _copies_unchanged(::typeof(identity), ::Type) = true
@inline _copies_unchanged(::typeof(conj), ::Type{T}) where {T <: Real} = true
@inline _copies_unchanged(::Any, ::Type) = false

# Gate for `_pack_a_contiguous!`, kept as its own function so a test can
# assert it fires for the driver's argument types and stays off for every
# ineligible shape (test/test_packing.jl). All but `m == MR` and the stride
# test fold at compile time (they inspect types only). `_unit_stride_rows`
# (src/kernels/simd.jl) has methods for exactly the three `Axis` kinds and
# deliberately NO fallback: an unknown axis type must be a MethodError here,
# never a silent `true`/`false`.
@inline function _pack_a_contiguous_eligible(
        packed::V, source::QSTile, transform::F, m::Int, ::Val{MR}, ::Type{T}
    ) where {V, F, MR, T}
    return packed isa PackedPanel{T} && source.storage isa DenseVector{T} &&
        _copies_unchanged(transform, T) && m == MR && _unit_stride_rows(source.rows)
end

# Fast path for the common A sliver: unit-stride rows filling the whole
# register tile, straight copy, `PackedPanel` destination, dense storage --
# every M-sliver of every column-major (or unit-stride-fastest multi-index) A
# in the profiling pass. Each K step is then `MR` contiguous source elements
# landing at `MR` contiguous packed offsets (`i + MR*p`), i.e. one
# `Vec{MR,T}` load/store per K step. Padding never arises here (`m == MR`),
# and `_check_pack_a` has already validated every address `base + rows.base +
# i + col_offset(p)`, `0 <= i < MR`, against `length(storage)` -- exactly the
# span each `vload` reads -- so nothing is read that the scalar path would not
# have read. Measured 2-2.6x (Float64) / 4-7x (Float32) over the scalar loop
# on the driver's argument types (ccqlin038 / Julia 1.13; `smallN_256x256x12`
# went from 62% to 33% packing share and 16 to 35 GFLOP/s; docs/decisions.md,
# "Packing speed"). Same eligibility shape as `_vector_store_eligible`
# (src/kernels/simd.jl).
@inline function _pack_a_contiguous!(
        packed::PackedPanel{T}, storage::DenseVector{T}, rowbase::Int, cols::C,
        ::Val{MR}, kc::Int
    ) where {T, C, MR}
    GC.@preserve storage begin
        sp = pointer(storage)
        dp = packed.ptr
        for p in 0:(kc - 1)
            v = vload(Vec{MR, T}, sp + sizeof(T) * (rowbase + axis_offset(cols, p)))
            vstore(v, dp + sizeof(T) * (MR * p))
        end
    end
    return packed
end

"""
    pack_a!(packed::AbstractVector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T}, transform) -> packed

Pack an A source tile into `packed` (a reused `Vector{T}`, or a `SubArray`
sliver of a macro panel) at `packed_a_offset(kernel, i, p) == i +
mr(kernel)*p`. `source` has `0 <= m <= mr(kernel)` rows and `kc =
ncols(source)` columns; `packed` needs `length >= packed_a_length(kernel,
kc)`. Row `i < m` writes `convert(T, transform(A[i,p]))`; padding rows (`i >=
m`) write `zero(T)` without reading `source` or calling `transform`. `kc ==
0` is a no-op. All validation happens before any write. Never allocates.

See [`unsafe_pack_a!`](@ref) for the sibling entry point that skips the
storage-bounds half of that validation.
"""
function pack_a!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {V, MR, NR, T2, F}
    return _pack_a!(packed, source, kernel, transform, Val(true))
end

"""
    unsafe_pack_a!(packed, source::QSTile, kernel, transform) -> packed

[`pack_a!`](@ref) **without** the `checked_tile_storage_bounds(source)` call.

PRECONDITION, which the caller must have established: every address `source`
can read -- `source.base + row_offset(i) + col_offset(p)` for `0 <= i <
nrows(source)`, `0 <= p < ncols(source)` -- lies in
`0:length(source.storage)-1`. Violating it is an out-of-bounds read through an
`@inbounds`/pointer path, not an exception.

Every other check `pack_a!` makes is still made here: the row-extent bound,
`kc >= 0`, the packed-buffer capacity, and the packed eltype. Only the
address-range check moves, and it moves to the caller.

The one caller in this package is `_execute_nest!` (src/driver.jl), which
validates the union of an entire macro block's slivers in a single
[`checked_span_bounds`](@ref) call before packing any of them -- an exactly
equivalent test, because the block's slivers partition its offset buffer and
all of them share the same K axis, so the block's offset range is the union of
the slivers' and the check only ever looks at range extremes.
"""
function unsafe_pack_a!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {V, MR, NR, T2, F}
    return _pack_a!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_a!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    m, kc = _check_pack_a(packed, source, kernel, bounds)
    kc == 0 && return packed

    if _pack_a_contiguous_eligible(packed, source, transform, m, Val(MR), T2)
        rowbase = source.base + source.rows.base
        return _pack_a_contiguous!(packed, source.storage, rowbase, source.cols, Val(MR), kc)
    end

    load = (i, p) -> tile_load(source, i, p)
    packed_offset = (i, p) -> packed_a_offset(kernel, i, p)
    _pack_panel!(packed, T2, Val(MR), kc, m, transform, load, packed_offset)
    return packed
end

"""
    pack_b!(packed::AbstractVector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T}, transform) -> packed

Pack a B source tile into `packed` (a reused `Vector{T}`, or a `SubArray`
sliver of a macro panel) at `packed_b_offset(kernel, j, p) == j +
nr(kernel)*p` (not column-major). `source` has `kc = nrows(source)` rows and
`0 <= n <= nr(kernel)` columns; `packed` needs `length >=
packed_b_length(kernel, kc)`. Column `j < n` writes `convert(T,
transform(B[p,j]))`; padding columns write `zero(T)` without reading
`source`. Same validation/allocation contract as [`pack_a!`](@ref).
"""
function pack_b!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {V, MR, NR, T2, F}
    return _pack_b!(packed, source, kernel, transform, Val(true))
end

"""
    unsafe_pack_b!(packed, source::QSTile, kernel, transform) -> packed

[`pack_b!`](@ref) without the `checked_tile_storage_bounds(source)` call; the
B-side counterpart of [`unsafe_pack_a!`](@ref), with the same precondition
(the caller has validated every address `source` can read) and the same single
caller in this package.
"""
function unsafe_pack_b!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {V, MR, NR, T2, F}
    return _pack_b!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_b!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    n, kc = _check_pack_b(packed, source, kernel, bounds)
    kc == 0 && return packed

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    packed_offset = (j, p) -> packed_b_offset(kernel, j, p)
    _pack_panel!(packed, T2, Val(NR), kc, n, transform, load, packed_offset)
    return packed
end

# ===========================================================================
# Complex packing
#
# `_pack_panel_complex!` below is a parallel loop rather than a generalisation
# of `_pack_panel!`, so that the complex milestone left the real path
# byte-identical as a `git diff` fact rather than an argument
# (docs/decisions.md, "Complex element-type milestone"); the later real-path
# restructuring (full/tail split, contiguous A fast path) likewise left THIS
# loop untouched. Only the validation preamble is shared, which cannot change
# either loop's generated code.
#
# Everything below writes `real(T)` into the packed buffer. The `transform`
# contract, frozen format-independently in that section:
#
#   `transform` is applied to each loaded *source element* before it is
#   committed, in whatever physical format the buffer uses. A packer that
#   splits an element into planes must produce a result identical to applying
#   `transform` to the complex value and THEN splitting -- never per real half,
#   where `conj` would be a silent no-op. Padding lanes are never read and
#   never call `transform`.
#
# `real(z)`/`imag(z)` on the loaded element are the only accessors used; the
# source is never `reinterpret`ed, because a `QSTile` addresses arbitrary
# strided (possibly scattered) storage for which that would be unsound.
# ===========================================================================

# --- emit: one logical K step, one lane, one format ------------------------
#
# `plane_offset(plane, index, p)` is `packed_a_plane_offset`/
# `packed_b_plane_offset` for the operand being packed, i.e. exactly
# `p * per_k + plane * vr + index`.

# PlanarFormat ("1r"): `o[t] = re; o[vr + t] = im`.
@inline function _pack_emit!(
        packed::V, ::PlanarFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, t, p), real(z))
    panel_store!(packed, plane_offset(1, t, p), imag(z))
    return nothing
end

# OneEFormat ("1e"): the real 2x2 block [[re, -im], [im, re]], stored as two
# real K steps of `2vr`:
#
#     o[2t] = re;        o[2t + 1] = im
#     o[2vr + 2t] = -im; o[2vr + 2t + 1] = re
#
# This does NOT fit the two-plane shape the plane-offset helpers describe: its
# four reals are two *real* K steps of doubled width, not four planes of `vr`.
# The linear formula still covers it exactly, so addressing through it keeps
# the layout written down once: `plane_offset(0, ...)` is the first real K step
# and `plane_offset(2, ...)` the second, since `plane * vr` at `plane = 2` is
# precisely the `2vr` stride between them. The `index` argument therefore runs
# over reals (`0:2vr-1`), not over logical rows -- the one place in this file
# where it does.
@inline function _pack_emit!(
        packed::V, ::OneEFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    re = real(z)
    im = imag(z)
    panel_store!(packed, plane_offset(0, 2 * t, p), re)
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), im)
    panel_store!(packed, plane_offset(2, 2 * t, p), -im)
    panel_store!(packed, plane_offset(2, 2 * t + 1, p), re)
    return nothing
end

# Padding lanes: literal zero into every real of the lane, without reading
# `source` and without calling `transform`. Written separately rather than as
# `_pack_emit!(..., zero(T))` because 1e's `-im` of a zero is `-0.0`, and the
# contract says padding writes a literal zero.
@inline function _pack_emit_zero!(
        packed::V, ::PlanarFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, t, p), zero(R))
    panel_store!(packed, plane_offset(1, t, p), zero(R))
    return nothing
end

@inline function _pack_emit_zero!(
        packed::V, ::OneEFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, 2 * t, p), zero(R))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), zero(R))
    panel_store!(packed, plane_offset(2, 2 * t, p), zero(R))
    panel_store!(packed, plane_offset(2, 2 * t + 1, p), zero(R))
    return nothing
end

# Complex counterpart of `_pack_panel!`. `T` is the *storage* (complex) type;
# the buffer holds `real(T)`. `kc == 0` is handled by the caller.
@inline function _pack_panel_complex!(
        packed::V, ::Type{T}, format::FMT, physical_dim::Int, kc::Int, valid::Int,
        transform::F, load::L, plane_offset::P
    ) where {V, T, FMT, F, L, P}
    @inbounds for p in 0:(kc - 1)
        for t in 0:(physical_dim - 1)
            if t < valid
                # transform applies to the complex element, THEN it is split.
                z = convert(T, transform(load(t, p)))::T
                _pack_emit!(packed, format, plane_offset, t, p, z)
            else
                _pack_emit_zero!(packed, format, plane_offset, t, p, real(T))
            end
        end
    end
    return packed
end

# ---------------------------------------------------------------------------
# Complex packing fast path (deinterleave-and-copy)
#
# Phase 1 of docs/proposals/complex-fast-paths.md, Sections 4.3 (PlanarFormat)
# and 4.4 (OneEFormat's A panel). It is the complex counterpart of
# `_pack_a_contiguous!` above and nothing more: a leaf-level alternative inside
# `_pack_a!`/`_pack_b!` for the ONE sliver shape it can serve, with the scalar
# `_pack_panel_complex!` loop above left byte-identical as the fallback for
# everything else. No format, offset formula or transform contract changes; the
# fast path is required to produce the same bytes the scalar loop would have.
#
# Why this is the easy half of the complex round trip (proposal Section 4.3):
# packing applies no `alpha`/`beta`, never reads its destination, and its only
# `transform`s are `identity` and `conj` -- and `conj` on a complex element
# touches the imaginary half alone. So there is no cross-plane arithmetic here
# at all, only a deinterleave of the source's native `[re,im,re,im,...]` layout
# and, for `conj`, a sign flip on the lanes that carry `im`.
#
# NOT an `unsafe_` path: it skips no validation a caller would otherwise get.
# `_check_pack_a`/`_check_pack_b` run first and unchanged (including the
# storage-bounds check, in the `Val(true)` mode), and the addresses the `vload`
# below reads are EXACTLY the `PD` elements the scalar loop would have read at
# the same K step -- `m == PD` (no padding lanes) and unit-stride lanes make
# the two address sets identical, element for element.
# ---------------------------------------------------------------------------

# Decision 5 of the proposal: `:avx512` only for this pass. Deliberately NOT
# `target_profile().isa === :avx512` -- nothing else in this package dispatches
# on the ISA *name*, and a name test would both miss a future ISA of the same
# width and hide what the gate is actually about. The question is whether the
# host's native vector register is as wide as AVX-512's, which is the property
# that makes a 2*PD-real load/deinterleave/store cheaper than 2*PD scalar
# stores; `_isa_vector_bytes(Val(:avx512))` folds to that width at compile time
# rather than spelling `64` here. AVX2/NEON/unknown fall through to the scalar
# loop until they have been measured (proposal Section 6.2, Section 7 item 5).
#
# SHARED, deliberately one function and not a per-fast-path copy (proposal
# Section 5: the eligibility question is the same question asked of different
# tiles, and "should be visibly the same function, not two copies that could
# drift"). Phase 2's planar store fast path
# (src/kernels/planar.jl, `_complex_vector_eligible`) calls exactly this; it
# lives here rather than there only because packing was built first. Renamed
# from Phase 1's `_complex_pack_isa_eligible` for that reason -- the predicate
# is unchanged.
@inline _complex_fastpath_isa_eligible(profile::TargetProfile) =
    profile.vector_bytes == _isa_vector_bytes(Val(:avx512))
@inline _complex_fastpath_isa_eligible() = _complex_fastpath_isa_eligible(target_profile())

# The value half of the gate, mirroring `_copies_unchanged` above: the two
# transforms the driver can produce (src/driver.jl, `plan_contract`) are the
# two the shuffle patterns below cover. Anything else -- including the
# arbitrary closures test/test_packing_complex.jl packs with -- is a MUST-fall-
# back, not a MAY: `_pack_alt` has no method for it.
@inline _complex_pack_transform_eligible(::typeof(identity)) = true
@inline _complex_pack_transform_eligible(::typeof(conj)) = true
@inline _complex_pack_transform_eligible(::Any) = false

# `RealFormat` never reaches a `ComplexKernelDescriptor` in this package, but
# the descriptor's format parameters are unconstrained beyond `<:PackFormat`,
# so the gate answers for it rather than leaving a MethodError as the contract.
@inline _complex_pack_format_eligible(::PlanarFormat) = true
@inline _complex_pack_format_eligible(::OneEFormat) = true
@inline _complex_pack_format_eligible(::PackFormat) = false

"""
    _pack_complex_contiguous_eligible(packed, storage, lane_axis, transform, format, valid, ::Val{PD}, ::Type{T}) -> Bool

Whether a complex sliver can take the vectorized pack path. The complex
counterpart of [`_pack_a_contiguous_eligible`](@ref), and the same shape of
predicate: every clause except `valid == PD`, the stride test and the ISA test
inspects types only, so those three are all that survive to run time at each
specialization.

`lane_axis` is the axis the PACKED index runs along: `source.rows` for A (the
`MR` logical rows of one K step) and `source.cols` for B (the `NR` logical
columns). It must be unit-stride `AffineAxis`, because that is what makes `PD`
consecutive source elements `PD` consecutive `Complex{T}` values in storage and
hence `2PD` consecutive `real(T)`s -- the bitcast the fast path performs is
sound for exactly that case and for no other (proposal Section 3.3; the scalar
loop's own header comment at the top of this section says why a `QSTile` is
never `reinterpret`ed in general).

`valid == PD` excludes every partial sliver, so the fast path never has to
write a padding lane and the "padding is a literal zero, never `-0.0`" contract
stays entirely with `_pack_emit_zero!`.
"""
@inline function _pack_complex_contiguous_eligible(
        packed::V, storage::S, lane_axis::AX, transform::F, format::FMT,
        valid::Int, ::Val{PD}, ::Type{T}
    ) where {V, S, AX, F, FMT, PD, T}
    return packed isa PackedPanel{real(T)} && storage isa DenseVector{T} &&
        _complex_pack_format_eligible(format) &&
        _complex_pack_transform_eligible(transform) &&
        valid == PD && _unit_stride_rows(lane_axis) &&
        _complex_fastpath_isa_eligible()
end

# The second shuffle operand. Every pattern below reads the lanes that carry
# `im` out of this vector and the lanes that carry `re` out of `src`, so the
# whole of `transform` is the choice made here -- `identity` takes `im`
# unchanged, `conj` takes it from `-src`. `-src` is an `fneg` on every lane,
# i.e. a sign-bit flip, which is bit-for-bit what the scalar path's
# `imag(conj(z))` computes (including on `-0.0` and on NaN payloads; a multiply
# by `-1.0` would NOT be, which is why no sign-pattern constant appears here).
@inline _pack_alt(src::Vec, ::typeof(identity)) = src
@inline _pack_alt(src::Vec, ::typeof(conj)) = -src

# OneEFormat's second `2PD`-real region wants the OPPOSITE choice from its
# first (proposal Section 4.4: `conj` "just swaps which plane is free"),
# because that region stores `-im` where the first stores `+im`.
@inline _pack_alt_flipped(src::Vec, ::typeof(identity)) = -src
@inline _pack_alt_flipped(src::Vec, ::typeof(conj)) = src

# --- shuffle primitives ----------------------------------------------------
#
# GUARDRAIL: every index tuple below is built HERE, from `PD`, at specialization
# time -- never hardcoded to the AVX-512 shape this was written against, and
# never a runtime gather. `PD` is `mr(kernel)`/`nr(kernel)`, which the driver
# derives from `kernel_shapes`; the patterns therefore follow the shipped menus
# automatically (proposal Section 6.2). These are `@generated` for the same
# reason the store paths in src/kernels/simd.jl are: `shufflevector` needs a
# literal `Val` index tuple, and `Val(ntuple(...))` is not reliably one.
#
# `src` is `[re_0, im_0, ..., re_{PD-1}, im_{PD-1}]`, the `PD` source elements
# of one K step read through their native `Complex{T}` layout.

# PlanarFormat ("1r"), one whole K step: `[re_0 .. re_{PD-1} | im_0 .. im_{PD-1}]`.
# Both halves are one contiguous `2PD`-real run at `p * packed_*_per_k`
# (src/complex_format.jl, `packed_a_plane_offset`: plane 0 at `+0`, plane 1 at
# `+PD`, adjacent), so the whole K step is a single store -- the destination
# needs no shuffle of its own, only the source needs deinterleaving.
@generated function _planar_pack_shuffle(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_planar_pack_shuffle: expected N == 2PD")))
    # lane j < PD  -> re_j      = src[2j]
    # lane PD + t  -> (+/-)im_t = alt[2t + 1]   (offset N: second shuffle source)
    idx = ntuple(k -> (k - 1) < PD ? 2 * (k - 1) : N + 2 * ((k - 1) - PD) + 1, 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# OneEFormat ("1e"), first of the two `2PD`-real regions of a K step
# (`plane_offset(0, ..)`): `[re_0, s*im_0, re_1, s*im_1, ...]`, with `s = +1`
# under `identity` and `-1` under `conj`. At `identity` this is a pure copy of
# `src` -- the pattern is the identity permutation over the first source -- and
# LLVM folds the shuffle away entirely, which is the "literally a memory copy"
# case the proposal's Section 4.4 predicts.
@generated function _onee_pack_shuffle_a(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_onee_pack_shuffle_a: expected N == 2PD")))
    # even lane 2t -> re_t      = src[2t]
    # odd  lane    -> (+/-)im_t = alt[2t + 1]
    idx = ntuple(k -> iseven(k) ? N + (k - 1) : (k - 1), 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# OneEFormat, second region (`plane_offset(2, ..)`, i.e. `+2PD`):
# `[-s*im_0, re_0, -s*im_1, re_1, ...]` -- the adjacent-pair swap, with the
# sign carried by whichever of `src`/`-src` the caller passes as `alt`.
@generated function _onee_pack_shuffle_b(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_onee_pack_shuffle_b: expected N == 2PD")))
    # even lane 2t   -> (-/+)im_t = alt[2t + 1]
    # odd  lane 2t+1 -> re_t      = src[2t]
    idx = ntuple(k -> isodd(k) ? N + k : k - 2, 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# --- the two packers -------------------------------------------------------
#
# `elembase` is the zero-based storage address of lane 0 at K step 0 minus the
# step axis's contribution, i.e. `source.base + lane_axis.base`; `steps` is the
# OTHER axis (`source.cols` for A, `source.rows` for B) and may be scattered --
# only the lane axis has to be unit-stride. Lane `t` of K step `p` therefore
# lives at element `elembase + axis_offset(steps, p) + t`, exactly the address
# `tile_load` would compute, and its two reals at twice that.
#
# `storage::DenseVector{Complex{R}}` is pinned in the signature (rather than a
# free `T`) so that a `packed`/`storage` element-type mismatch is a MethodError
# here instead of a bitcast to the wrong width.

@inline function _pack_complex_contiguous!(
        ::PlanarFormat, packed::PackedPanel{R}, storage::DenseVector{Complex{R}},
        elembase::Int, steps::C, ::Val{PD}, kc::Int, transform::F
    ) where {R, C, PD, F}
    GC.@preserve storage begin
        sp = reinterpret(Ptr{R}, pointer(storage))
        dp = packed.ptr
        for p in 0:(kc - 1)
            src = vload(
                Vec{2 * PD, R},
                sp + sizeof(R) * (2 * (elembase + axis_offset(steps, p)))
            )
            vstore(
                _planar_pack_shuffle(src, _pack_alt(src, transform), Val(PD)),
                dp + sizeof(R) * (2 * PD * p)
            )
        end
    end
    return packed
end

@inline function _pack_complex_contiguous!(
        ::OneEFormat, packed::PackedPanel{R}, storage::DenseVector{Complex{R}},
        elembase::Int, steps::C, ::Val{PD}, kc::Int, transform::F
    ) where {R, C, PD, F}
    GC.@preserve storage begin
        sp = reinterpret(Ptr{R}, pointer(storage))
        dp = packed.ptr
        for p in 0:(kc - 1)
            src = vload(
                Vec{2 * PD, R},
                sp + sizeof(R) * (2 * (elembase + axis_offset(steps, p)))
            )
            # 1e writes twice planar's destination bandwidth by construction
            # (reals_per_element(OneEFormat) == 4); that is the format's cost,
            # not the fast path's.
            at = 4 * PD * p
            vstore(
                _onee_pack_shuffle_a(src, _pack_alt(src, transform), Val(PD)),
                dp + sizeof(R) * at
            )
            vstore(
                _onee_pack_shuffle_b(src, _pack_alt_flipped(src, transform), Val(PD)),
                dp + sizeof(R) * (at + 2 * PD)
            )
        end
    end
    return packed
end

"""
    pack_a!(packed::AbstractVector{real(T)}, source::QSTile, kernel::ComplexKernelDescriptor{MR,NR,T,FA,FB}, transform) -> packed

Pack a complex A source tile into `packed`, a buffer of `realtype(kernel) ==
real(T)`, in the physical format `FA` (see [`PlanarFormat`](@ref),
[`OneEFormat`](@ref)). `source` has `0 <= m <= mr(kernel)` **logical**
(complex) rows and `kc = ncols(source)` columns; `packed` needs `length >=
packed_a_length(kernel, kc)`, which counts **reals** at *logical* `kc`.

Row `i < m` commits `convert(T, transform(A[i,p]))` -- `transform` is applied
to the complex element and the result is then split into the packed format,
never applied per real half. Padding rows (`i >= m`) write literal zeros into
**every** real of the lane (all four, under [`OneEFormat`](@ref)) without
reading `source` or calling `transform`. `kc == 0` is a no-op. All validation
happens before any write. Never allocates.
"""
function pack_a!(
        packed::V, source::QSTile, kernel::ComplexKernelDescriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(true))
end

"""
    unsafe_pack_a!(packed, source::QSTile, kernel::ComplexKernelDescriptor, transform) -> packed

Complex-descriptor counterpart of [`unsafe_pack_a!`](@ref); same precondition.
"""
function unsafe_pack_a!(
        packed::V, source::QSTile, kernel::ComplexKernelDescriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_a!(
        packed::V, source::QSTile, kernel::ComplexKernelDescriptor{MR, NR, T2, FA, FB},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    m, kc = _check_pack_a(packed, source, kernel, bounds)
    kc == 0 && return packed

    # A's packed index runs along `source.rows` (the MR logical rows).
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.rows, transform, FA(), m, Val(MR), T2
        )
        elembase = source.base + source.rows.base
        return _pack_complex_contiguous!(
            FA(), packed, source.storage, elembase, source.cols, Val(MR), kc, transform
        )
    end

    load = (i, p) -> tile_load(source, i, p)
    plane_offset = (plane, i, p) -> packed_a_plane_offset(kernel, plane, i, p)
    _pack_panel_complex!(packed, T2, FA(), MR, kc, m, transform, load, plane_offset)
    return packed
end

"""
    pack_b!(packed::AbstractVector{real(T)}, source::QSTile, kernel::ComplexKernelDescriptor{MR,NR,T,FA,FB}, transform) -> packed

Pack a complex B source tile into `packed`, a buffer of `realtype(kernel) ==
real(T)`, in the physical format `FB`. `source` has `kc = nrows(source)` rows
and `0 <= n <= nr(kernel)` **logical** columns; `packed` needs `length >=
packed_b_length(kernel, kc)` reals. Same `transform`, padding, validation and
allocation contract as [`pack_a!`](@ref).
"""
function pack_b!(
        packed::V, source::QSTile, kernel::ComplexKernelDescriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(true))
end

"""
    unsafe_pack_b!(packed, source::QSTile, kernel::ComplexKernelDescriptor, transform) -> packed

Complex-descriptor counterpart of [`unsafe_pack_b!`](@ref); same precondition.
"""
function unsafe_pack_b!(
        packed::V, source::QSTile, kernel::ComplexKernelDescriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_b!(
        packed::V, source::QSTile, kernel::ComplexKernelDescriptor{MR, NR, T2, FA, FB},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    n, kc = _check_pack_b(packed, source, kernel, bounds)
    kc == 0 && return packed

    # B's packed index runs along `source.cols` (the NR logical columns), so
    # the unit-stride requirement is on the N axis, not the K axis -- the
    # mirror image of A's, and the reason this gate is a separate call rather
    # than a shared `source.rows` test. `FB` is `PlanarFormat` under both
    # shipped complex methods, so proposal Section 4.5's "1m's B panel needs no
    # new code" holds here literally: this is the same predicate and the same
    # packer `PlanarMethod` uses.
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.cols, transform, FB(), n, Val(NR), T2
        )
        elembase = source.base + source.cols.base
        return _pack_complex_contiguous!(
            FB(), packed, source.storage, elembase, source.rows, Val(NR), kc, transform
        )
    end

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_plane_offset(kernel, plane, j, p)
    _pack_panel_complex!(packed, T2, FB(), NR, kc, n, transform, load, plane_offset)
    return packed
end
