# Packs against the frozen physical formats in kernel_descriptor.jl
# (A: i + MR*p, B: j + NR*p; do not redefine).

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

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_plane_offset(kernel, plane, j, p)
    _pack_panel_complex!(packed, T2, FB(), NR, kc, n, transform, load, plane_offset)
    return packed
end
