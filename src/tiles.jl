# OWNER: packing implementer (Phase 2). See docs/decisions.md and
# Julia-Microkernel-Tile-Interface-Design.md section 4.
#
# Implements: AffineAxis, ScatterAxis, SourceTile, DestinationTile, and the
# small addressing helpers packing.jl builds on.
#
# Conventions (spec section 3, retained from the indexing layer):
#   - Tile-local logical coordinates are zero-based.
#   - Relative offsets and operand bases are signed Int values in elements.
#   - Ordinary Julia array/Vector indices are one-based: with parent Vector
#     storage, the Julia index is the computed zero-based address plus one.
#   - A tile's `base` is the zero-based address of its logical origin in
#     parent storage; axis offsets are added directly on top of it.

# ----------------------------------------------------------------------------
# Axis execution representations
# ----------------------------------------------------------------------------

"""
    AffineAxis(base::Int, stride::Int, count::Int)

A regular one-dimensional addressing rule: local coordinate `t` (`0 <= t <
count`) maps to the zero-based offset `base + t*stride`. Built from a regular
[`BlockDescriptor`](@ref) via [`axis_from_descriptor`](@ref).

`count` may be zero (an empty axis); `stride` may be any sign, including
zero (a broadcast/singleton axis).
"""
struct AffineAxis
    base::Int
    stride::Int
    count::Int

    function AffineAxis(base::Int, stride::Int, count::Int)
        count >= 0 || throw(ArgumentError("AffineAxis count must be nonnegative, got $count"))
        return new(base, stride, count)
    end
end

"""
    ScatterAxis(offsets::AbstractVector{Int}, count::Int)

An irregular one-dimensional addressing rule: local coordinate `t` (`0 <= t <
count`) maps to the zero-based offset `offsets[t+1]`.

`offsets` is **borrowed**: `ScatterAxis` stores the vector (or view) it is
given, never copies it, and the caller must keep its populated prefix
`offsets[1:count]` unchanged and valid for the lifetime of any tile or
packing call that references this axis (spec section 3, "scatter buffers are
borrowed"). `count` is a valid logical extent, not `offsets`' capacity, and
must not exceed `length(offsets)`.
"""
struct ScatterAxis{V<:AbstractVector{Int}}
    offsets::V
    count::Int

    function ScatterAxis(offsets::V, count::Int) where {V<:AbstractVector{Int}}
        count >= 0 || throw(ArgumentError("ScatterAxis count must be nonnegative, got $count"))
        count <= length(offsets) ||
            throw(DimensionMismatch("ScatterAxis: offsets has length $(length(offsets)), " *
                                     "need at least count = $count"))
        return new{V}(offsets, count)
    end
end

const Axis = Union{AffineAxis,ScatterAxis}

"""
    axis_length(ax::Union{AffineAxis,ScatterAxis})::Int

The number of valid logical coordinates the axis addresses (`0:axis_length(ax)-1`).
Shares its name with the `AxisGroup` accessor of the same meaning
(Phase 1); this is simply the analogous accessor for an execution-time axis.
"""
axis_length(ax::AffineAxis) = ax.count
axis_length(ax::ScatterAxis) = ax.count

"""
    axis_offset(ax::Union{AffineAxis,ScatterAxis}, t::Int)::Int

Unchecked zero-based offset for local coordinate `t`. Callers must ensure
`0 <= t < axis_length(ax)` themselves; this is the hot-path primitive, not a
validated entry point (see [`checked_axis_offset`](@ref)).
"""
@inline axis_offset(ax::AffineAxis, t::Int) = ax.base + t * ax.stride
@inline axis_offset(ax::ScatterAxis, t::Int) = @inbounds ax.offsets[t+1]

"""
    checked_axis_offset(ax::Union{AffineAxis,ScatterAxis}, t::Int)::Int

Bounds-checked variant of [`axis_offset`](@ref): throws `BoundsError` unless
`0 <= t < axis_length(ax)`.
"""
function checked_axis_offset(ax::Axis, t::Int)
    (0 <= t < axis_length(ax)) || throw(BoundsError(ax, t))
    return axis_offset(ax, t)
end

"""
    axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}) -> Union{AffineAxis,ScatterAxis}

Build the execution-time axis matching a Phase-1 [`BlockDescriptor`](@ref):
an `AffineAxis(descriptor.base, descriptor.stride, descriptor.count)` when
`descriptor.regular`, otherwise a `ScatterAxis` referencing the populated
prefix `view(buffer, 1:descriptor.count)` of `buffer` (a view, never a copy —
`buffer` remains borrowed per the scatter-buffer ownership rule).

Call this once per panel/block classification, never once per scalar
element (spec section 4): the branch here selects a concrete axis
representation, and everything downstream (tile addressing, packing) then
runs against that concrete, statically dispatched type.
"""
function axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int})
    if descriptor.regular
        return AffineAxis(descriptor.base, descriptor.stride, descriptor.count)
    else
        return ScatterAxis(view(buffer, 1:descriptor.count), descriptor.count)
    end
end

# ----------------------------------------------------------------------------
# Source / destination tiles
# ----------------------------------------------------------------------------

"""
    QSTile{S,R,C}

The shared internal representation behind both [`SourceTile`](@ref) and
[`DestinationTile`](@ref) — the design explicitly permits sharing one
internal type (spec section 4), and this package does so: `SourceTile` and
`DestinationTile` are both simply `const` aliases for `QSTile`. The two
public names exist only to document, at each call site, whether a tile is
being read from or written to; the type itself enforces neither direction
and provides both load and store operations.

Fields:
- `storage::S`: the tile's parent storage (typically a `Vector{T}`); see
  "Storage backends" below.
- `base::Int`: the zero-based address of the tile's logical origin `(0,0)`
  in `storage`.
- `rows::R`, `cols::C`: an `AffineAxis` or `ScatterAxis` giving row and
  column addressing respectively.

The zero-based address of logical `(i,j)` is `base + row_offset(i) +
col_offset(j)` (spec section 3); see [`tile_offset`](@ref) /
[`checked_tile_offset`](@ref).

## Storage backends

For the first milestone, `storage` is expected to be a `Vector{T}`: the
computed zero-based address is converted to Julia's one-based `Vector`
index by adding one (spec section 3). A verified `StridedView` adapter is
deferred to whichever layer constructs `QSTile`s from `StridedView`s
(outside Phase 2's scope) — this type itself is storage-agnostic, requiring
only `getindex`/`setindex!` with that same "address + 1" convention.
"""
struct QSTile{S,R<:Axis,C<:Axis}
    storage::S
    base::Int
    rows::R
    cols::C
end

"""
    SourceTile(storage, base::Int, rows, cols)

A read-oriented [`QSTile`](@ref). See [`QSTile`](@ref) for field meaning and
the shared-type note; `SourceTile === DestinationTile` (both are `QSTile`).
"""
const SourceTile = QSTile

"""
    DestinationTile(storage, base::Int, rows, cols)

A write-oriented [`QSTile`](@ref). See [`QSTile`](@ref) for field meaning
and the shared-type note; `DestinationTile === SourceTile` (both are
`QSTile`).
"""
const DestinationTile = QSTile

"""
    nrows(tile::QSTile)::Int
    ncols(tile::QSTile)::Int

The tile's valid row/column extent (`axis_length` of `rows`/`cols`).
"""
nrows(tile::QSTile) = axis_length(tile.rows)
ncols(tile::QSTile) = axis_length(tile.cols)

"""
    tile_offset(tile::QSTile, i::Int, j::Int)::Int

Unchecked zero-based address `base + row_offset(i) + col_offset(j)` for
logical coordinate `(i,j)`. Callers must ensure `0 <= i < nrows(tile)` and
`0 <= j < ncols(tile)` themselves (this is the hot-path primitive used
after shape validation, e.g. inside `pack_a!`/`pack_b!`); see
[`checked_tile_offset`](@ref) for the validated form.
"""
@inline function tile_offset(tile::QSTile, i::Int, j::Int)
    return tile.base + axis_offset(tile.rows, i) + axis_offset(tile.cols, j)
end

"""
    checked_tile_offset(tile::QSTile, i::Int, j::Int)::Int

Bounds-checked variant of [`tile_offset`](@ref): throws `BoundsError` unless
`0 <= i < nrows(tile)` and `0 <= j < ncols(tile)`.
"""
function checked_tile_offset(tile::QSTile, i::Int, j::Int)
    (0 <= i < nrows(tile)) || throw(BoundsError(tile, (i, j)))
    (0 <= j < ncols(tile)) || throw(BoundsError(tile, (i, j)))
    return tile_offset(tile, i, j)
end

"""
    tile_load(tile::QSTile, i::Int, j::Int)

Unchecked read of `tile.storage` at logical `(i, j)`, using the "zero-based
address plus one" convention for `Vector`-backed storage. Callers must have
already validated `(i, j)` (e.g. via a prior shape check); see
[`checked_tile_load`](@ref) for the validated form.
"""
@inline function tile_load(tile::QSTile, i::Int, j::Int)
    return @inbounds tile.storage[tile_offset(tile, i, j)+1]
end

"""
    checked_tile_load(tile::QSTile, i::Int, j::Int)

Bounds-checked read: validates `(i, j)` against the tile's declared extent
before computing the address, then performs an ordinary (non-`@inbounds`)
`storage` read so an out-of-bounds `storage` access (e.g. from a stale or
undersized borrowed buffer) still raises rather than reading garbage.
"""
function checked_tile_load(tile::QSTile, i::Int, j::Int)
    addr = checked_tile_offset(tile, i, j)
    return tile.storage[addr+1]
end

"""
    tile_store!(tile::QSTile, i::Int, j::Int, v)

Unchecked write of `v` into `tile.storage` at logical `(i, j)`. Callers must
have already validated `(i, j)`; see [`checked_tile_store!`](@ref).
"""
@inline function tile_store!(tile::QSTile, i::Int, j::Int, v)
    @inbounds tile.storage[tile_offset(tile, i, j)+1] = v
    return tile
end

"""
    checked_tile_store!(tile::QSTile, i::Int, j::Int, v)

Bounds-checked write: validates `(i, j)` against the tile's declared extent
before computing the address, then performs an ordinary `storage` write.
"""
function checked_tile_store!(tile::QSTile, i::Int, j::Int, v)
    addr = checked_tile_offset(tile, i, j)
    tile.storage[addr+1] = v
    return tile
end

# ----------------------------------------------------------------------------
# Integration (main process, Phase 2b Fable review follow-up): validate that
# every address a tile can name is in bounds for its storage, BEFORE any
# unchecked hot-path loop runs.
#
# `axis_offset(ax, t)` and `tile_offset(tile, i, j)` are deliberately
# unchecked, matching `tile_store!`/`scale_tile!`/`accumulate`'s own
# `@inbounds` hot paths (spec: "do not perform a regularity/bounds test per
# scalar element"). The Fable review (Phase 2b) found that nothing upstream
# of those hot paths actually validated the tile's *storage bounds* before
# entering them — `pack_a!`/`pack_b!` and `execute_tile!` checked shape
# (m/n/kc vs MR/NR) but not whether `base + row_offset(i) + col_offset(j)`
# stays within `0:length(storage)-1` for every valid (i,j). This is the
# missing "reachable storage bounds" check spec section 6 requires before
# packing, and the missing "in bounds before entering unchecked hot paths"
# check spec section 3 requires before execution. `axis_offset_range` and
# `checked_tile_storage_bounds` below are the one-time-per-tile checks that
# close that gap; call sites are in packing.jl and kernel.jl.
# ----------------------------------------------------------------------------

"""
    axis_offset_range(ax::Union{AffineAxis,ScatterAxis}) -> (lo::Int, hi::Int)

The minimum and maximum zero-based offset `ax` can produce over its valid
domain `0:axis_length(ax)-1`. For an empty axis (`axis_length(ax) == 0`) the
result is `(0, -1)` (an empty range, `lo > hi`) since no offset is ever
produced. Uses `Int128` internally so the range computation itself cannot
overflow even when the endpoints are representable as `Int`.
"""
function axis_offset_range(ax::AffineAxis)
    ax.count == 0 && return (0, -1)
    lo128 = Int128(ax.base)
    hi128 = Int128(ax.base) + Int128(ax.count - 1) * Int128(ax.stride)
    lo128, hi128 = minmax(lo128, hi128)
    (typemin(Int) <= lo128 && hi128 <= typemax(Int)) ||
        throw(OverflowError("axis_offset_range: affine axis range not representable as Int"))
    return (Int(lo128), Int(hi128))
end

function axis_offset_range(ax::ScatterAxis)
    ax.count == 0 && return (0, -1)
    prefix = view(ax.offsets, 1:ax.count)
    return (Int(minimum(prefix)), Int(maximum(prefix)))
end

"""
    checked_tile_storage_bounds(base::Int, rows::Axis, cols::Axis, storage_length::Int)

Validate that every address `base + row_offset(i) + col_offset(j)` for
`0 <= i < axis_length(rows)`, `0 <= j < axis_length(cols)` lies in
`0:storage_length-1`. An empty tile (either axis has length 0) always
passes: no address is ever produced. Throws `BoundsError` otherwise. Uses
`Int128` for the summation so the check itself never overflows.
"""
function checked_tile_storage_bounds(base::Int, rows::Axis, cols::Axis, storage_length::Int)
    (axis_length(rows) == 0 || axis_length(cols) == 0) && return nothing
    (rlo, rhi) = axis_offset_range(rows)
    (clo, chi) = axis_offset_range(cols)
    lo128 = Int128(base) + Int128(rlo) + Int128(clo)
    hi128 = Int128(base) + Int128(rhi) + Int128(chi)
    (lo128 >= 0 && hi128 <= Int128(storage_length - 1)) ||
        throw(BoundsError("tile addresses [$lo128, $hi128] exceed storage bounds [0, $(storage_length - 1)]", base))
    return nothing
end

"""
    checked_tile_storage_bounds(tile::QSTile)

Convenience form: validate `tile` against `length(tile.storage)`.
"""
checked_tile_storage_bounds(tile::QSTile) =
    checked_tile_storage_bounds(tile.base, tile.rows, tile.cols, length(tile.storage))
