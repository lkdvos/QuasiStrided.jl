# Tile-local coordinates are zero-based; Vector storage index = address + 1.
# A tile's `base` is its logical origin's zero-based address in storage.

"""
    AffineAxis(base::Int, stride::Int, count::Int)

Regular addressing: local `t` (`0 <= t < count`) maps to `base + t*stride`.
Built from a regular [`BlockDescriptor`](@ref) via [`axis_from_descriptor`](@ref).
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

Irregular addressing: local `t` (`0 <= t < count`) maps to `offsets[t+1]`.
`offsets` is **borrowed** (not copied); caller keeps `offsets[1:count]` valid
for the axis's lifetime. `count <= length(offsets)`.
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

Number of valid logical coordinates (`0:axis_length(ax)-1`).
"""
axis_length(ax::AffineAxis) = ax.count
axis_length(ax::ScatterAxis) = ax.count

"""
    axis_offset(ax::Union{AffineAxis,ScatterAxis}, t::Int)::Int

Unchecked offset for `t`; caller ensures `0 <= t < axis_length(ax)` (see
[`checked_axis_offset`](@ref)).
"""
@inline axis_offset(ax::AffineAxis, t::Int) = ax.base + t * ax.stride
@inline axis_offset(ax::ScatterAxis, t::Int) = @inbounds ax.offsets[t+1]

"""
    checked_axis_offset(ax::Union{AffineAxis,ScatterAxis}, t::Int)::Int

Bounds-checked [`axis_offset`](@ref); throws `BoundsError` if out of range.
"""
function checked_axis_offset(ax::Axis, t::Int)
    (0 <= t < axis_length(ax)) || throw(BoundsError(ax, t))
    return axis_offset(ax, t)
end

"""
    axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}) -> Union{AffineAxis,ScatterAxis}

Build the axis matching a [`BlockDescriptor`](@ref): `AffineAxis` when
regular, else a `ScatterAxis` viewing `buffer[1:descriptor.count]` (borrowed,
not copied). Call once per block classification, not per element.
"""
function axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int})
    if descriptor.regular
        return AffineAxis(descriptor.base, descriptor.stride, descriptor.count)
    else
        return ScatterAxis(view(buffer, 1:descriptor.count), descriptor.count)
    end
end

"""
    QSTile{S,R,C}

Shared internal type behind both [`SourceTile`](@ref) and
[`DestinationTile`](@ref) (`const` aliases; the names just document intent
at the call site). Fields: `storage` (typically `Vector{T}`), `base`
(zero-based address of logical `(0,0)`), `rows`/`cols` (`AffineAxis` or
`ScatterAxis`). Logical `(i,j)` addresses `base + row_offset(i) +
col_offset(j)`, converted to storage via "address + 1" for `Vector`s.
"""
struct QSTile{S,R<:Axis,C<:Axis}
    storage::S
    base::Int
    rows::R
    cols::C
end

"""
    SourceTile(storage, base::Int, rows, cols)

Read-oriented [`QSTile`](@ref) (`SourceTile === DestinationTile`).
"""
const SourceTile = QSTile

"""
    DestinationTile(storage, base::Int, rows, cols)

Write-oriented [`QSTile`](@ref) (`DestinationTile === SourceTile`).
"""
const DestinationTile = QSTile

"""
    nrows(tile::QSTile)::Int
    ncols(tile::QSTile)::Int

Tile's valid row/column extent.
"""
nrows(tile::QSTile) = axis_length(tile.rows)
ncols(tile::QSTile) = axis_length(tile.cols)

"""
    tile_offset(tile::QSTile, i::Int, j::Int)::Int

Unchecked address `base + row_offset(i) + col_offset(j)`; caller ensures
`(i,j)` in range (see [`checked_tile_offset`](@ref)).
"""
@inline function tile_offset(tile::QSTile, i::Int, j::Int)
    return tile.base + axis_offset(tile.rows, i) + axis_offset(tile.cols, j)
end

"""
    checked_tile_offset(tile::QSTile, i::Int, j::Int)::Int

Bounds-checked [`tile_offset`](@ref); throws `BoundsError` if out of range.
"""
function checked_tile_offset(tile::QSTile, i::Int, j::Int)
    (0 <= i < nrows(tile)) || throw(BoundsError(tile, (i, j)))
    (0 <= j < ncols(tile)) || throw(BoundsError(tile, (i, j)))
    return tile_offset(tile, i, j)
end

"""
    tile_load(tile::QSTile, i::Int, j::Int)

Unchecked read at logical `(i,j)`; caller ensures it's in range (see
[`checked_tile_load`](@ref)).
"""
@inline function tile_load(tile::QSTile, i::Int, j::Int)
    return @inbounds tile.storage[tile_offset(tile, i, j)+1]
end

"""
    checked_tile_load(tile::QSTile, i::Int, j::Int)

Bounds-checked read: validates `(i,j)`, then a non-`@inbounds` storage read.
"""
function checked_tile_load(tile::QSTile, i::Int, j::Int)
    addr = checked_tile_offset(tile, i, j)
    return tile.storage[addr+1]
end

"""
    tile_store!(tile::QSTile, i::Int, j::Int, v)

Unchecked write of `v` at logical `(i,j)`; caller ensures it's in range (see
[`checked_tile_store!`](@ref)).
"""
@inline function tile_store!(tile::QSTile, i::Int, j::Int, v)
    @inbounds tile.storage[tile_offset(tile, i, j)+1] = v
    return tile
end

"""
    checked_tile_store!(tile::QSTile, i::Int, j::Int, v)

Bounds-checked write: validates `(i,j)`, then a non-`@inbounds` storage write.
"""
function checked_tile_store!(tile::QSTile, i::Int, j::Int, v)
    addr = checked_tile_offset(tile, i, j)
    tile.storage[addr+1] = v
    return tile
end

# One-time-per-tile storage-bounds check (Phase 2b review finding), called
# from packing.jl/kernel.jl before the unchecked @inbounds hot paths run.

"""
    axis_offset_range(ax::Union{AffineAxis,ScatterAxis}) -> (lo::Int, hi::Int)

Min/max offset `ax` can produce over its domain (`(0, -1)` if empty).
Uses `Int128` internally to avoid overflow.
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

Validate every address `base + row_offset(i) + col_offset(j)` lies in
`0:storage_length-1` (empty tile always passes). Throws `BoundsError`
otherwise.
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
