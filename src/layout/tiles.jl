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
struct ScatterAxis{V <: AbstractVector{Int}}
    offsets::V
    count::Int

    function ScatterAxis(offsets::V, count::Int) where {V <: AbstractVector{Int}}
        count >= 0 || throw(ArgumentError("ScatterAxis count must be nonnegative, got $count"))
        count <= length(offsets) ||
            throw(
            DimensionMismatch(
                "ScatterAxis: offsets has length $(length(offsets)), " *
                    "need at least count = $count"
            )
        )
        return new{V}(offsets, count)
    end
end

"""
    PtrScatterAxis(offsets::Ptr{Int}, count::Int)

Like [`ScatterAxis`](@ref) but over borrowed offsets held as a raw pointer, so
that it is `isbits` and `Union{AffineAxis,PtrScatterAxis}` needs no heap box.
`ScatterAxis` holds an `AbstractVector`, which would make that union
non-isbits and heap-allocate per `execute!` on irregular destinations whenever
Julia cannot union-split it. Used by the driver (`_axis_of`,
src/execution/macrokernel.jl); `ScatterAxis` remains the vector-backed,
bounds-checkable form everywhere else. The pointer is borrowed -- `execute!`
holds the `GC.@preserve`.
"""
struct PtrScatterAxis
    offsets::Ptr{Int}
    count::Int

    function PtrScatterAxis(offsets::Ptr{Int}, count::Int)
        count >= 0 ||
            throw(ArgumentError("PtrScatterAxis count must be nonnegative, got $count"))
        return new(offsets, count)
    end
end

const Axis = Union{AffineAxis, ScatterAxis, PtrScatterAxis}

"""
    axis_length(ax::Union{AffineAxis,ScatterAxis})::Int

Number of valid logical coordinates (`0:axis_length(ax)-1`).
"""
axis_length(ax::AffineAxis) = ax.count
axis_length(ax::ScatterAxis) = ax.count
axis_length(ax::PtrScatterAxis) = ax.count

"""
    axis_offset(ax::Union{AffineAxis,ScatterAxis}, t::Int)::Int

Unchecked offset for `t`; caller ensures `0 <= t < axis_length(ax)` (see
[`checked_axis_offset`](@ref)).
"""
@inline axis_offset(ax::AffineAxis, t::Int) = ax.base + t * ax.stride
@inline axis_offset(ax::ScatterAxis, t::Int) = @inbounds ax.offsets[t + 1]
@inline axis_offset(ax::PtrScatterAxis, t::Int) =
    unsafe_load(ax.offsets + sizeof(Int) * t)

"""
    checked_axis_offset(ax::Union{AffineAxis,ScatterAxis}, t::Int)::Int

Bounds-checked [`axis_offset`](@ref); throws `BoundsError` if out of range.
"""
function checked_axis_offset(ax::Axis, t::Int)
    (0 <= t < axis_length(ax)) || throw(BoundsError(ax, t))
    return axis_offset(ax, t)
end

"""
    axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}, first::Int = 0) -> Union{AffineAxis,ScatterAxis}

Build the axis matching a [`BlockDescriptor`](@ref) classified from
`buffer[first+1 : first+descriptor.count]` (zero-based `first`): `AffineAxis`
when regular, else a `ScatterAxis` viewing that same range (borrowed, not
copied). Call once per block classification, not per element.
"""
function axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}, first::Int)
    if descriptor.regular
        return AffineAxis(descriptor.base, descriptor.stride, descriptor.count)
    else
        return ScatterAxis(view(buffer, (first + 1):(first + descriptor.count)), descriptor.count)
    end
end

axis_from_descriptor(descriptor::BlockDescriptor, buffer::Vector{Int}) =
    axis_from_descriptor(descriptor, buffer, 0)

"""
    QSTile{S,R,C}

Shared internal type behind both [`SourceTile`](@ref) and
[`DestinationTile`](@ref) (`const` aliases; the names just document intent
at the call site). Fields: `storage` (typically `Vector{T}`), `base`
(zero-based address of logical `(0,0)`), `rows`/`cols` (`AffineAxis` or
`ScatterAxis`). Logical `(i,j)` addresses `base + row_offset(i) +
col_offset(j)`, converted to storage via "address + 1" for `Vector`s.
"""
struct QSTile{S, R <: Axis, C <: Axis}
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
    return @inbounds tile.storage[tile_offset(tile, i, j) + 1]
end

"""
    checked_tile_load(tile::QSTile, i::Int, j::Int)

Bounds-checked read: validates `(i,j)`, then a non-`@inbounds` storage read.
"""
function checked_tile_load(tile::QSTile, i::Int, j::Int)
    addr = checked_tile_offset(tile, i, j)
    return tile.storage[addr + 1]
end

"""
    tile_store!(tile::QSTile, i::Int, j::Int, v)

Unchecked write of `v` at logical `(i,j)`; caller ensures it's in range (see
[`checked_tile_store!`](@ref)).
"""
@inline function tile_store!(tile::QSTile, i::Int, j::Int, v)
    @inbounds tile.storage[tile_offset(tile, i, j) + 1] = v
    return tile
end

"""
    checked_tile_store!(tile::QSTile, i::Int, j::Int, v)

Bounds-checked write: validates `(i,j)`, then a non-`@inbounds` storage write.
"""
function checked_tile_store!(tile::QSTile, i::Int, j::Int, v)
    addr = checked_tile_offset(tile, i, j)
    tile.storage[addr + 1] = v
    return tile
end

# One-time-per-tile storage-bounds check, called from the packers and
# microkernels before the unchecked @inbounds hot paths run.

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

function axis_offset_range(ax::PtrScatterAxis)
    ax.count == 0 && return (0, -1)
    lo = hi = unsafe_load(ax.offsets)
    for t in 1:(ax.count - 1)
        v = axis_offset(ax, t)
        lo, hi = min(lo, v), max(hi, v)
    end
    return (lo, hi)
end

"""
    descriptor_offset_range(d::BlockDescriptor, buffer::Vector{Int}, first::Int) -> (lo::Int, hi::Int)

Min/max offset the interval `d` classifies can produce (`(0, -1)` if empty) --
the [`BlockDescriptor`](@ref) counterpart of [`axis_offset_range`](@ref), and
numerically identical to `axis_offset_range(_axis_of(d, buffer, first))`
without materializing the axis. `O(1)` when `d.regular`; otherwise a scan of
`buffer[first+1 : first+d.count]`, which is the same scan the scatter axis
would do.
"""
function descriptor_offset_range(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    d.count == 0 && return (0, -1)
    d.regular && return axis_offset_range(AffineAxis(d.base, d.stride, d.count))
    lo = hi = buffer[first + 1]
    for t in 1:(d.count - 1)
        v = buffer[first + t + 1]
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return (lo, hi)
end

"""
    checked_span_bounds(base::Int, rows::Tuple{Int,Int}, cols::Tuple{Int,Int}, storage_length::Int)

Validate that every address `base + r + c` with `r` in the closed row-offset
range `rows` and `c` in the closed column-offset range `cols` lies in
`0:storage_length-1`; an empty range (`hi < lo`, the `(0, -1)` convention of
[`axis_offset_range`](@ref)) always passes. Throws `BoundsError` otherwise.
`Int128` internally so nothing can wrap.

This is the whole of [`checked_tile_storage_bounds`](@ref)'s arithmetic, split
out so a caller that knows the offset RANGES of a region -- rather than the
axes of one tile -- can validate that region in one call. Because the check
only ever looks at the four extremes, and because `(rlo + clo)` and
`(rhi + chi)` are both realized addresses of any rectangular (row-set x
column-set) region, it is exact for such a region, not conservative: it
accepts a region iff it accepts every rectangular sub-region of it, and
rejects iff at least one address is out of bounds. `src/execution/execute.jl` relies on
that equivalence to check a whole macro block once instead of each of its
slivers.
"""
function checked_span_bounds(
        base::Int, rows::Tuple{Int, Int}, cols::Tuple{Int, Int}, storage_length::Int
    )
    (rlo, rhi) = rows
    (clo, chi) = cols
    (rhi < rlo || chi < clo) && return nothing
    lo128 = Int128(base) + Int128(rlo) + Int128(clo)
    hi128 = Int128(base) + Int128(rhi) + Int128(chi)
    (lo128 >= 0 && hi128 <= Int128(storage_length - 1)) ||
        throw(BoundsError("tile addresses [$lo128, $hi128] exceed storage bounds [0, $(storage_length - 1)]", base))
    return nothing
end

"""
    checked_tile_storage_bounds(base::Int, rows::Axis, cols::Axis, storage_length::Int)

Validate every address `base + row_offset(i) + col_offset(j)` lies in
`0:storage_length-1` (empty tile always passes). Throws `BoundsError`
otherwise.
"""
function checked_tile_storage_bounds(base::Int, rows::Axis, cols::Axis, storage_length::Int)
    (axis_length(rows) == 0 || axis_length(cols) == 0) && return nothing
    return checked_span_bounds(
        base, axis_offset_range(rows), axis_offset_range(cols), storage_length
    )
end

"""
    checked_tile_storage_bounds(tile::QSTile)

Convenience form: validate `tile` against `length(tile.storage)`.
"""
checked_tile_storage_bounds(tile::QSTile) =
    checked_tile_storage_bounds(tile.base, tile.rows, tile.cols, length(tile.storage))

# Whether an axis steps through storage one element at a time. Deliberately no
# fallback method: an unknown axis type must be a MethodError, not `false`.
_unit_stride_rows(ax::AffineAxis) = ax.stride == 1
_unit_stride_rows(::ScatterAxis) = false
_unit_stride_rows(::PtrScatterAxis) = false
