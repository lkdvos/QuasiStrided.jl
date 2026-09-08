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
