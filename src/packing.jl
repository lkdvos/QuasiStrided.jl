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

@inline function _check_pack_a(packed::V, source::QSTile, kernel::K) where {V, K}
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
    checked_tile_storage_bounds(source)
    return (m, kc)
end

@inline function _check_pack_b(packed::V, source::QSTile, kernel::K) where {V, K}
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
    checked_tile_storage_bounds(source)
    return (n, kc)
end

# ----------------------------------------------------------------------------
# Real packing
# ----------------------------------------------------------------------------

# Shared inner loop for pack_a!/pack_b!; `load`/`packed_offset` close over the
# operand-specific index mapping. `kc == 0` is handled by the caller.
@inline function _pack_panel!(
        packed::V, ::Type{T}, physical_dim::Int, kc::Int, valid::Int,
        transform::F, load::L, packed_offset::P
    ) where {V, T, F, L, P}
    @inbounds for p in 0:(kc - 1)
        for i in 0:(physical_dim - 1)
            v = i < valid ? convert(T, transform(load(i, p)))::T : zero(T)
            panel_store!(packed, packed_offset(i, p), v)
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
"""
function pack_a!(
        packed::V, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {V, MR, NR, T2, F}
    _check_packed_eltype(packed, kernel)
    m, kc = _check_pack_a(packed, source, kernel)
    kc == 0 && return packed

    load = (i, p) -> tile_load(source, i, p)
    packed_offset = (i, p) -> packed_a_offset(kernel, i, p)
    _pack_panel!(packed, T2, MR, kc, m, transform, load, packed_offset)
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
    _check_packed_eltype(packed, kernel)
    n, kc = _check_pack_b(packed, source, kernel)
    kc == 0 && return packed

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    packed_offset = (j, p) -> packed_b_offset(kernel, j, p)
    _pack_panel!(packed, T2, NR, kc, n, transform, load, packed_offset)
    return packed
end

# ===========================================================================
# Complex packing
#
# `_pack_panel!` above is deliberately UNTOUCHED and `_pack_panel_complex!`
# below is a parallel loop rather than a generalisation of it, so that "the
# real path is byte-identical" stays a `git diff` fact rather than an argument
# (docs/decisions.md, "Complex element-type milestone"). Only the validation
# preamble is shared, which cannot change either loop's generated code.
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
    _check_packed_eltype(packed, kernel)
    m, kc = _check_pack_a(packed, source, kernel)
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
    _check_packed_eltype(packed, kernel)
    n, kc = _check_pack_b(packed, source, kernel)
    kc == 0 && return packed

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_plane_offset(kernel, plane, j, p)
    _pack_panel_complex!(packed, T2, FB(), NR, kc, n, transform, load, plane_offset)
    return packed
end
