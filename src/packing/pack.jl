# Packers: copy one sliver of A (up to MR logical rows) or B (up to NR logical
# columns) over a K range into a contiguous panel in the descriptor's packed
# format. `pack_a!`/`pack_b!` validate and dispatch on the operand's
# `PackFormat`; each format has its own leaf loop, and a contiguous fast path
# (src/packing/pack_contiguous.jl) for the one sliver shape it can serve.

# `Vec`/`vload`/`vstore` already arrive via src/packing/panel.jl's `using`; the
# complex fast path additionally needs the compile-time-index shuffle (SIMD.jl
# v3), which is one LLVM `shufflevector` instruction and never a runtime gather.
using SIMD: shufflevector

# Explicit runtime check (not dispatch) so a mismatch raises ArgumentError. The
# packed buffer holds `realtype(kernel)`, which is *not* `scalartype(kernel)`
# once the element type is complex.
@inline function _check_packed_eltype(packed, kernel::Descriptor{MR, NR, T2}) where {MR, NR, T2}
    R = realtype(kernel)
    eltype(packed) === R && return nothing
    msg = R === T2 ?
        "packed buffer eltype $(eltype(packed)) does not match kernel scalar type $T2" :
        "packed buffer eltype $(eltype(packed)) does not match kernel real type $R " *
        "(scalar type $T2)"
    throw(ArgumentError(msg))
end

# ----------------------------------------------------------------------------
# Shared argument validation for all four pack_a!/pack_b! methods
# ----------------------------------------------------------------------------
# Extent bound, nonnegative `kc`, packed capacity, then the
# one-time storage-bounds check before any `@inbounds` loop. Returns
# `(valid, kc)`; `kc == 0` is the no-op the caller returns from. One bound type
# parameter per argument, as at `pack_a!` in src/microkernels/interface.jl.
#
# `packed_a_length`/`packed_b_length` count ELEMENTS for a real descriptor and
# REALS for a complex one, at the logical `kc` in both cases, so the same check
# serves both without knowing which it has.

# `BOUNDS` is a compile-time flag, not a runtime one: at `Val(true)` the body
# below runs every check, and at `Val(false)` the `checked_tile_storage_bounds`
# call is folded away entirely. Only
# `unsafe_pack_a!`/`unsafe_pack_b!` ever pass `Val(false)`, and only from a
# caller that has already validated the WHOLE macro block this sliver belongs
# to (src/execution/execute.jl, `_execute_nest!`). Every other check -- extents, the
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
# Entry points
# ----------------------------------------------------------------------------

"""
    pack_a!(packed, source::QSTile, kernel::Descriptor{MR,NR,T,FA,FB}, transform) -> packed

Pack an A source tile into `packed` (a buffer of `realtype(kernel)`: a
`Vector`, a `SubArray` sliver of a macro panel, or a [`PackedPanel`](@ref)) in
the physical format `FA`. `source` has `0 <= m <= mr(kernel)` **logical** rows
and `kc = ncols(source)` columns; `packed` needs `length >=
packed_a_length(kernel, kc)`, which counts reals at logical `kc`.

Row `i < m` commits `convert(T, transform(A[i,p]))` -- for a complex `T`,
`transform` is applied to the complex element and the result is then split
into the packed format, never applied per real half. Padding rows (`i >= m`)
write literal zeros into every real of the lane without reading `source` or
calling `transform`. `kc == 0` is a no-op. All validation happens before any
write. Never allocates.

For a real kernel the layout is `packed_a_offset(kernel, i, p) == i +
mr(kernel)*p`; see [`PlanarFormat`](@ref) and [`OneEFormat`](@ref) for the
complex ones. See [`unsafe_pack_a!`](@ref) for the sibling entry point that
skips the storage-bounds half of the validation.
"""
function pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
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

The one caller in this package is `_execute_nest!`
(src/execution/execute.jl), which validates the union of an entire macro
block's slivers in a single [`checked_span_bounds`](@ref) call before packing
any of them -- an exactly equivalent test, because the block's slivers
partition its offset buffer and all of them share the same K axis, so the
block's offset range is the union of the slivers' and the check only ever
looks at range extremes.
"""
function unsafe_pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_a!(packed, source, kernel, transform, Val(false))
end

"""
    pack_b!(packed, source::QSTile, kernel::Descriptor{MR,NR,T,FA,FB}, transform) -> packed

Pack a B source tile into `packed` in the physical format `FB`. `source` has
`kc = nrows(source)` rows and `0 <= n <= nr(kernel)` **logical** columns;
`packed` needs `length >= packed_b_length(kernel, kc)` reals. For a real
kernel the layout is `packed_b_offset(kernel, j, p) == j + nr(kernel)*p` (not
column-major). Same `transform`, padding, validation and allocation contract
as [`pack_a!`](@ref).
"""
function pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
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
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F
    ) where {V, MR, NR, T2, FA, FB, F}
    return _pack_b!(packed, source, kernel, transform, Val(false))
end

@inline function _pack_a!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    m, kc = _check_pack_a(packed, source, kernel, bounds)
    kc == 0 && return packed
    return _pack_a_sliver!(FA(), packed, source, kernel, transform, m, kc)
end

@inline function _pack_b!(
        packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2, FA, FB},
        transform::F, bounds::Val{BOUNDS}
    ) where {V, MR, NR, T2, FA, FB, F, BOUNDS}
    _check_packed_eltype(packed, kernel)
    n, kc = _check_pack_b(packed, source, kernel, bounds)
    kc == 0 && return packed
    return _pack_b_sliver!(FB(), packed, source, kernel, transform, n, kc)
end

# ----------------------------------------------------------------------------
# The leaf loop, shared by every format
# ----------------------------------------------------------------------------

# One sliver over `kc` K steps. `load`/`plane_offset` close over the
# operand-specific index mapping, and `format` dispatches the per-lane store
# through `_pack_emit!`/`_pack_emit_zero!`. `kc == 0` is handled by the caller.
#
# `PD` (the physical dim: MR or NR) is a compile-time constant. A full sliver
# (`valid == PD`) gets a constant-trip-count inner loop that LLVM fully
# unrolls; a tail sliver writes its valid lanes and then its zero padding as
# two separate loops. Either way no loop body holds a conditional load: a
# per-element `t < valid ? load : zero` compiles to a branch around the load,
# which blocks if-conversion and keeps the whole loop scalar. The per-K-step
# store order (0, 1, ..., PD-1) and the padding contract (padding lanes never
# read `source` and never call `transform`) hold for every format.
#
# `transform` applies to the loaded (complex, for a complex format) element
# BEFORE `_pack_emit!` splits it into planes; padding lanes go through
# `_pack_emit_zero!` instead, which writes literal zeros without calling
# `transform` (a `-0.0` hazard for `OneEFormat`'s `-im` plane otherwise).
#
# `pf` is the software-prefetch hook (`nothing` unless a prefetch site is
# switched on; see `_gather_prefetcher` below): it is called once at the top of
# each K step, before that step's loads, with the step and the lane count.
@inline function _pack_panel!(
        packed::V, ::Type{T}, format::FMT, ::Val{PD}, kc::Int, valid::Int,
        transform::F, load::L, plane_offset::P, pf::PF = nothing
    ) where {V, T, FMT <: PackFormat, PD, F, L, P, PF}
    if valid == PD
        @inbounds for p in 0:(kc - 1)
            _prefetch_step!(pf, p, PD)
            for t in 0:(PD - 1)
                z = convert(T, transform(load(t, p)))::T
                _pack_emit!(packed, format, plane_offset, t, p, z)
            end
        end
    else
        @inbounds for p in 0:(kc - 1)
            _prefetch_step!(pf, p, valid)
            for t in 0:(valid - 1)
                z = convert(T, transform(load(t, p)))::T
                _pack_emit!(packed, format, plane_offset, t, p, z)
            end
            for t in valid:(PD - 1)
                _pack_emit_zero!(packed, format, plane_offset, t, p, real(T))
            end
        end
    end
    return packed
end

# ----------------------------------------------------------------------------
# Software prefetch in the gather loop (EXPERIMENTAL, off by default)
# ----------------------------------------------------------------------------
#
# Sites `:pack_a` and `:pack_b` (src/hardware/prefetch.jl). When on, K step `p`
# of the gather first prefetches every valid lane of K step `p + distance`
# (clamped to the sliver's last step), one `prefetcht0` per lane -- the same
# addresses `load` will read `distance` steps later. The clamp, not a branch,
# keeps the loop body straight-line; its cost is a few redundant prefetches of
# the last step's lines at the end of the sliver. It never reads outside the
# sliver: a scattered axis's offset is only ever looked up at a valid step.
#
# (`:pack_a_line`/`:pack_b_line` are the cache-line-granular alternative; see
# `_line_gather_prefetcher` below.) For the per-lane sites: one prefetch per
# LANE, not per cache line, deliberately: the lanes of one K
# step may share a line (unit-stride lanes) or each sit on their own (a
# scattered or long-stride lane axis), and the loop cannot tell which without
# a runtime test. On a unit-stride K axis this also re-prefetches a line the
# previous step already asked for; whether any of that pays is what
# benchmark/bench_prefetch.jl measures.
#
# `nothing` is the disabled hook. `_prefetch_distance` is a literal, so with
# the site off `_gather_prefetcher` returns `nothing` at compile time and
# `_prefetch_step!(nothing, ...)` is an empty inlined method: the disabled
# gather loop is the same code as before this hook existed.
@inline _prefetch_step!(::Nothing, p::Int, nlanes::Int) = nothing
# Call-site `@inline`: the line-granular hook is past the inliner's cost
# threshold, and an out-of-line call per K step would cost more than it saves.
@inline _prefetch_step!(pf::PF, p::Int, nlanes::Int) where {PF} = @inline pf(p, nlanes)

# `LANES_ON_ROWS` is `true` for A (lane `t` is row `t`, K step `p` is column
# `p`) and `false` for B (K step is the row, lane the column), matching the
# `load` closures of the sliver packers below. Only a `DenseArray` storage has
# a `pointer` to prefetch through; anything else gets no prefetch.
#
# `site`/`line_site` are the operand's per-lane and line-granular switches;
# the line site takes precedence when both are on.
@inline function _gather_prefetcher(
        site::Val, line_site::Val, source::QSTile, kc::Int, lr::Val{LANES_ON_ROWS}
    ) where {LANES_ON_ROWS}
    storage = source.storage
    storage isa DenseArray || return nothing
    DL = _prefetch_distance(line_site)
    DL > 0 && return _line_gather_prefetcher(DL, source, kc, lr)
    D = _prefetch_distance(site)
    D > 0 || return nothing
    base = pointer(storage)
    E = sizeof(eltype(storage))
    qmax = kc - 1
    return function (p::Int, nlanes::Int)
        q = min(p + D, qmax)
        for t in 0:(nlanes - 1)
            o = LANES_ON_ROWS ? tile_offset(source, t, q) : tile_offset(source, q, t)
            prefetch(base + E * o)
        end
        return nothing
    end
end

# Whether lanes `0:n-1` of `ax` cover a gap-free byte run, i.e. every line
# between the first and last lane's is touched: an affine lane axis whose
# stride is at most one line. Only this case prefetches by line range; any
# other axis (long stride, scattered) is handled lane by lane.
@inline _dense_lanes(ax::AffineAxis, E::Int) = abs(ax.stride) * E <= PREFETCH_LINE_BYTES
@inline _dense_lanes(::Axis, ::Int) = false

# Inclusive absolute byte range `[a, b]` of lanes `0:n-1` of `ax` at element
# offset `o` (the other axis's contribution plus the tile base). Exact for an
# affine axis, which is the only kind it is called on for a range prefetch.
# `% UInt` (not `UInt(...)`): the offset is a valid, nonnegative element
# offset, and the checked conversion would put an InexactError branch in the
# hot loop.
@inline function _lane_bytes(base::Ptr, E::Int, o::Int, ax, n::Int)
    lo, hi = minmax(o + axis_offset(ax, 0), o + axis_offset(ax, n - 1))
    return (UInt(base) + (E * lo) % UInt, UInt(base) + (E * hi + E - 1) % UInt)
end

# The line-granular gather hook (sites `:pack_a_line`/`:pack_b_line`). K step
# `p` prefetches for step `q = p + D`, and nothing once `q` is past the
# sliver's last step (unlike the per-lane hook, which clamps: that step was
# already prefetched `D` steps earlier, so a clamp would only repeat it). Which lines step `q` needs that earlier
# steps did not is decided by a MODE chosen once per sliver, outside the K
# loop, from the two axes' kinds and strides (`_line_mode`), so the per-step
# cost is a predictable branch or two plus the prefetches themselves:
#
#   period `P`: when the K axis is affine with `d = |stride|*E` bytes and `d`
#     divides 64, a lane's addresses sampled every `P = 64/d` steps are spaced
#     exactly one line apart, so issuing only on steps `q` that are multiples
#     of `P` prefetches each of its lines exactly once, whatever its alignment
#     -- with no per-lane test. `d >= 64` (every step a new line) gives `P = 1`.
#
#   1 dense lanes (gap-free run, e.g. unit-stride N for B): on each issuing
#     step, one prefetch at the run's first byte, one per further 64 bytes,
#     and one at its last byte unless that line is already covered. A scattered K axis issues every step.
#   2 long-stride lanes over a K axis with a period: every lane, on each
#     issuing step (e.g. column-major B: one prefetch per lane per 8 steps).
#   3 anything else (scattered K axis, or `d` not dividing 64): lane by lane,
#     a prefetch only for a lane whose line at `q` differs from its line at
#     `q - 1`.
@inline function _line_mode(lanes, steps, E::Int)
    P = _step_period(steps, E)
    _dense_lanes(lanes, E) && return (1, max(P, 1))
    return P > 0 ? (2, P) : (3, 0)
end

# 0 = no usable period.
@inline function _step_period(steps::AffineAxis, E::Int)
    d = abs(steps.stride) * E
    d >= PREFETCH_LINE_BYTES && return 1
    (d > 0 && PREFETCH_LINE_BYTES % d == 0) && return PREFETCH_LINE_BYTES ÷ d
    return 0
end
@inline _step_period(::Axis, ::Int) = 0

@inline _affine_parts(ax::AffineAxis) = (ax.base, ax.stride)
@inline _affine_parts(::Axis) = (0, 0)  # never reached: only affine lanes are dense

# Each mode is its OWN closure type, so `_gather_prefetcher` returns a small
# `Union` and the call to `_pack_panel!` is union-split: every mode gets its
# own specialized K loop, with no per-step mode branch and no mode/period
# state kept live across the loop.
@inline function _line_gather_prefetcher(
        D::Int, source::QSTile, kc::Int, ::Val{LANES_ON_ROWS}
    ) where {LANES_ON_ROWS}
    storage = source.storage
    base = pointer(storage)
    E = sizeof(eltype(storage))
    lanes = LANES_ON_ROWS ? source.rows : source.cols
    steps = LANES_ON_ROWS ? source.cols : source.rows
    mode, P = _line_mode(lanes, steps, E)
    pmask = P - 1  # P is a power of two whenever it is used as a mask
    qmax = kc - 1
    sbase = source.base
    if mode == 1
        # Dense lanes are affine (`_dense_lanes`), so lane `t` is at
        # `lb + t*ls`; the run's low end and byte span follow from that
        # without a per-step `minmax`.
        lb, ls = _affine_parts(lanes)
        als = abs(ls)
        return function (p::Int, nlanes::Int)
            q = p + D
            (q > qmax || (q & pmask) != 0) && return nothing
            lo = base + E * (sbase + axis_offset(steps, q) + lb + (ls < 0 ? (nlanes - 1) * ls : 0))
            hi = lo + (E * ((nlanes - 1) * als + 1) - 1)
            prefetch(lo)
            a = lo + PREFETCH_LINE_BYTES
            while a < hi
                prefetch(a)
                a += PREFETCH_LINE_BYTES
            end
            # The last line only if it is not the one `a` last covered.
            (UInt(hi) >> _LINE_SHIFT) != (UInt(a - PREFETCH_LINE_BYTES) >> _LINE_SHIFT) && prefetch(hi)
            return nothing
        end
    elseif mode == 2
        return function (p::Int, nlanes::Int)
            q = p + D
            (q > qmax || (q & pmask) != 0) && return nothing
            oq = sbase + axis_offset(steps, q)
            for t in 0:(nlanes - 1)
                prefetch(base + E * (oq + axis_offset(lanes, t)))
            end
            return nothing
        end
    else
        return function (p::Int, nlanes::Int)
            q = p + D
            q > qmax && return nothing
            oq = sbase + axis_offset(steps, q)
            op = sbase + axis_offset(steps, q - 1)
            for t in 0:(nlanes - 1)
                lo = axis_offset(lanes, t)
                a = UInt(base) + (E * (oq + lo)) % UInt
                b = UInt(base) + (E * (op + lo)) % UInt
                (a >> _LINE_SHIFT) != (b >> _LINE_SHIFT) && prefetch(Ptr{Cvoid}(a))
            end
            return nothing
        end
    end
end

# RealFormat: one real per element, no plane split. `plane_offset` is still
# called with a leading `plane` argument (always `0` here) so that every
# format's call sites share the same closure shape.
@inline function _pack_emit!(
        packed::V, ::RealFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, t, p), z)
    return nothing
end

@inline function _pack_emit_zero!(
        packed::V, ::RealFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, t, p), zero(R))
    return nothing
end

# The real sliver packers. A has a contiguous fast path; B does not. Both
# gather fallbacks carry the (default-off) prefetch hook above.
@inline function _pack_a_sliver!(
        ::RealFormat, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, m::Int, kc::Int
    ) where {V, MR, NR, T2, F}
    if _pack_a_contiguous_eligible(packed, source, transform, m, Val(MR), T2)
        rowbase = source.base + source.rows.base
        return _pack_a_contiguous!(packed, source.storage, rowbase, source.cols, Val(MR), kc)
    end

    load = (i, p) -> tile_load(source, i, p)
    plane_offset = (plane, i, p) -> packed_a_offset(kernel, i, p)
    pf = _gather_prefetcher(Val(:pack_a), Val(:pack_a_line), source, kc, Val(true))
    storage = source.storage
    GC.@preserve storage begin
        _pack_panel!(packed, T2, RealFormat(), Val(MR), kc, m, transform, load, plane_offset, pf)
    end
    return packed
end

@inline function _pack_b_sliver!(
        ::RealFormat, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, n::Int, kc::Int
    ) where {V, MR, NR, T2, F}
    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_offset(kernel, j, p)
    pf = _gather_prefetcher(Val(:pack_b), Val(:pack_b_line), source, kc, Val(false))
    storage = source.storage
    GC.@preserve storage begin
        _pack_panel!(packed, T2, RealFormat(), Val(NR), kc, n, transform, load, plane_offset, pf)
    end
    return packed
end


# ===========================================================================
# Complex packing
#
# Everything below writes `real(T)` into the packed buffer. The `transform`
# contract, which holds for every format:
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

# InterleavedFormat: `o[2t] = re; o[2t + 1] = im` -- exactly 1e's first real K
# step and nothing else, addressed the same way (`index` over reals).
@inline function _pack_emit!(
        packed::V, ::InterleavedFormat, plane_offset::P, t::Int, p::Int, z::T
    ) where {V, P, T}
    panel_store!(packed, plane_offset(0, 2 * t, p), real(z))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), imag(z))
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
        packed::V, ::InterleavedFormat, plane_offset::P, t::Int, p::Int, ::Type{R}
    ) where {V, P, R}
    panel_store!(packed, plane_offset(0, 2 * t, p), zero(R))
    panel_store!(packed, plane_offset(0, 2 * t + 1, p), zero(R))
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


# The complex sliver packers, for any complex format.
@inline function _pack_a_sliver!(
        format::FMT, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, m::Int, kc::Int
    ) where {FMT <: Union{PlanarFormat, InterleavedFormat, OneEFormat}, V, MR, NR, T2, F}
    # A's packed index runs along `source.rows` (the MR logical rows).
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.rows, transform, format, m, Val(MR), T2
        )
        elembase = source.base + source.rows.base
        return _pack_complex_contiguous!(
            format, packed, source.storage, elembase, source.cols, Val(MR), kc, transform
        )
    end

    load = (i, p) -> tile_load(source, i, p)
    plane_offset = (plane, i, p) -> packed_a_plane_offset(kernel, plane, i, p)
    pf = _gather_prefetcher(Val(:pack_a), Val(:pack_a_line), source, kc, Val(true))
    storage = source.storage
    GC.@preserve storage begin
        _pack_panel!(packed, T2, format, Val(MR), kc, m, transform, load, plane_offset, pf)
    end
    return packed
end

@inline function _pack_b_sliver!(
        format::FMT, packed::V, source::QSTile, kernel::Descriptor{MR, NR, T2},
        transform::F, n::Int, kc::Int
    ) where {FMT <: Union{PlanarFormat, InterleavedFormat, OneEFormat}, V, MR, NR, T2, F}
    # B's packed index runs along `source.cols` (the NR logical columns), so
    # the unit-stride requirement is on the N axis, not the K axis -- the
    # mirror image of A's. `FB` is `PlanarFormat` under both shipped complex
    # methods, so 1m's B panel uses the same predicate and the same packer as
    # `PlanarMethod`.
    if _pack_complex_contiguous_eligible(
            packed, source.storage, source.cols, transform, format, n, Val(NR), T2
        )
        elembase = source.base + source.cols.base
        return _pack_complex_contiguous!(
            format, packed, source.storage, elembase, source.rows, Val(NR), kc, transform
        )
    end

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    plane_offset = (plane, j, p) -> packed_b_plane_offset(kernel, plane, j, p)
    pf = _gather_prefetcher(Val(:pack_b), Val(:pack_b_line), source, kc, Val(false))
    storage = source.storage
    GC.@preserve storage begin
        _pack_panel!(packed, T2, format, Val(NR), kc, n, transform, load, plane_offset, pf)
    end
    return packed
end
