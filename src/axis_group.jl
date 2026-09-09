# Zero-based coordinates, fastest dimension first (column-major); an AxisGroup
# is a pure layout descriptor with no base offset, storage, dtype, or conjugation.

"""
    AxisGroup{D,P}

Shared logical coordinate enumeration over `D` dims (`lengths`) with `P`
physical offset maps (`strides`, one `NTuple{D,Int}` per operand).
"""
struct AxisGroup{D, P}
    lengths::NTuple{D, Int}
    strides::NTuple{P, NTuple{D, Int}}

    function AxisGroup{D, P}(
            lengths::NTuple{D, Int},
            strides::NTuple{P, NTuple{D, Int}}
        ) where {D, P}
        P >= 1 || throw(ArgumentError("AxisGroup requires at least one map (P >= 1), got P = $P"))
        for (d, L) in enumerate(lengths)
            L >= 0 || throw(ArgumentError("AxisGroup lengths must be nonnegative, got lengths[$d] = $L"))
        end
        _validate_axis_group_bounds(lengths, strides)
        return new{D, P}(lengths, strides)
    end
end

"""
    AxisGroup(lengths::NTuple{D,Int}, strides::NTuple{P,NTuple{D,Int}}) where {D,P}

Construct a validated `AxisGroup` (axes kept in the order supplied; see
[`normalize_group`](@ref) to fold/drop them). Throws `ArgumentError` for
invalid lengths or `P < 1`, `OverflowError` if the cardinality or any map's
offset excursion doesn't fit `Int`.
"""
AxisGroup(lengths::NTuple{D, Int}, strides::NTuple{P, NTuple{D, Int}}) where {D, P} =
    AxisGroup{D, P}(lengths, strides)

# Q = prod(lengths); empty domain (any zero length) always gives Q = 0.
# Int128 accumulation detects overflow without wrapping (validation-time only).
function _checked_axis_length(lengths::NTuple{D, Int}) where {D}
    any(==(0), lengths) && return 0
    q = one(Int128)
    for L in lengths
        q *= Int128(L)
        q > Int128(typemax(Int)) &&
            throw(OverflowError("AxisGroup cardinality (product of lengths) exceeds typemax(Int)"))
    end
    return Int(q)
end

# Validates cardinality and, per map, sum((L[d]-1)*abs(S[d])) <= typemax(Int),
# all in Int128 so nothing (incl. abs(typemin(Int))) can silently wrap.
function _validate_axis_group_bounds(
        lengths::NTuple{D, Int},
        strides::NTuple{P, NTuple{D, Int}}
    ) where {D, P}
    Q = _checked_axis_length(lengths)
    Q == 0 && return nothing
    for (p, S) in enumerate(strides)
        acc = zero(Int128)
        for d in 1:D
            L = lengths[d]  # >= 1 here, since Q > 0 implies no zero length.
            acc += Int128(L - 1) * abs(Int128(S[d]))
            acc > Int128(typemax(Int)) &&
                throw(
                OverflowError(
                    "AxisGroup map $p: sum((L[d]-1)*abs(S[d])) exceeds typemax(Int); " *
                        "this layout is not representable under the conservative offset-range bound"
                )
            )
        end
    end
    return nothing
end

"""
    axis_length(g::AxisGroup)::Int

Cardinality `Q = prod(lengths)` (already validated to fit `Int`).
"""
axis_length(g::AxisGroup) = _unchecked_axis_length(g.lengths)

@inline function _unchecked_axis_length(lengths::NTuple{D, Int}) where {D}
    q = 1
    for L in lengths
        q *= L
    end
    return q
end

# Type-stable, stack-allocated "replace element i of an NTuple{N,Int}".
@inline _tupleset(t::NTuple{N, Int}, i::Int, v::Int) where {N} =
    ntuple(j -> ifelse(j == i, v, t[j]), Val(N))

# Lifted out of offsets/fill_offsets! to avoid closure-boxing an inline lambda
# over loop-reassigned variables, which would allocate every call.
@inline function _add_offsets(offs::NTuple{P, Int}, g::AxisGroup{D, P}, d::Int, x::Int) where {D, P}
    return ntuple(p -> offs[p] + x * g.strides[p][d], Val(P))
end

@inline function _sub_reset_offsets(offs::NTuple{P, Int}, g::AxisGroup{D, P}, d::Int, Ld::Int) where {D, P}
    return ntuple(p -> offs[p] - (Ld - 1) * g.strides[p][d], Val(P))
end

"""
    offsets(g::AxisGroup{D,P}, q::Int)::NTuple{P,Int}

Decode logical coordinate `0 <= q < axis_length(g)` into the `P` relative
element offsets via mixed-radix decode. Throws `BoundsError` if out of range.
"""
function offsets(g::AxisGroup{D, P}, q::Int) where {D, P}
    Q = axis_length(g)
    (0 <= q < Q) || throw(BoundsError(g, q))
    r = q
    offs = ntuple(_ -> 0, Val(P))
    for d in 1:D
        L = g.lengths[d]
        x = r % L
        r = r ÷ L
        if x != 0
            offs = _add_offsets(offs, g, d, x)
        end
    end
    return offs
end

"""
    fill_offsets!(buffers::NTuple{P,Vector{Int}}, g::AxisGroup{D,P}, first::Int, count::Int)

Fill `buffers[p][t+1] = offsets(g, first + t)[p]` for `0 <= t < count`, via
one mixed-radix decode of `first` plus reset-before-carry increments.
Requires `count >= 0`, `0 <= first <= axis_length(g)`,
`count <= axis_length(g) - first`, buffers long enough and pairwise distinct.
Throws `ArgumentError`/`BoundsError`/`DimensionMismatch` accordingly; returns `buffers`.
"""
function fill_offsets!(
        buffers::NTuple{P, Vector{Int}}, g::AxisGroup{D, P},
        first::Int, count::Int
    ) where {D, P}
    Q = axis_length(g)

    count >= 0 || throw(ArgumentError("count must be nonnegative, got $count"))
    (0 <= first <= Q) ||
        throw(BoundsError("AxisGroup interval start $first out of range [0, $Q]", first))
    count <= Q - first ||
        throw(BoundsError("AxisGroup interval (first=$first, count=$count) exceeds domain size $Q", first))

    for p in 1:P
        length(buffers[p]) >= count ||
            throw(DimensionMismatch("buffer $p has length $(length(buffers[p])), need at least $count"))
    end
    for i in 1:P, j in (i + 1):P
        buffers[i] === buffers[j] &&
            throw(ArgumentError("buffers must be distinct Vector{Int} objects (buffers $i and $j alias)"))
    end

    count == 0 && return buffers

    # Decode `first` once into coordinates and initial per-map offsets.
    r = first
    x = ntuple(_ -> 0, Val(D))
    offs = ntuple(_ -> 0, Val(P))
    for d in 1:D
        L = g.lengths[d]
        xd = r % L
        r = r ÷ L
        x = _tupleset(x, d, xd)
        if xd != 0
            offs = _add_offsets(offs, g, d, xd)
        end
    end

    t = 0
    @inbounds while true
        for p in 1:P
            buffers[p][t + 1] = offs[p]
        end
        t += 1
        t == count && break

        # Advance coordinates fastest-first, reset-before-carry.
        d = 1
        while d <= D
            L = g.lengths[d]
            xd = x[d]
            if xd < L - 1
                x = _tupleset(x, d, xd + 1)
                offs = _add_offsets(offs, g, d, 1)
                break
            else
                x = _tupleset(x, d, 0)
                offs = _sub_reset_offsets(offs, g, d, L)
                d += 1
            end
        end
    end

    return buffers
end

"""
    BlockDescriptor

Classification of a materialized offset interval for one map: `base` (first
element's offset, or 0 if empty), `stride` (constant per-step stride when
`regular`, else meaningless), `count`, and `regular` (whether
`buffer[t+1] == base + t*stride` for all `0 <= t < count`). If `!regular`,
consumers must read the underlying buffer directly; stale once that buffer
is refilled.
"""
struct BlockDescriptor
    base::Int
    stride::Int
    count::Int
    regular::Bool
end

"""
    describe_block(buffer::Vector{Int}, first::Int, count::Int)::BlockDescriptor
    describe_block(buffer::Vector{Int}, count::Int)::BlockDescriptor

Classify `buffer[first+1 : first+count]` (zero-based `first`, defaulting to
0; read-only) as empty/singleton/affine/irregular (overflow-checked adjacent
differences; a non-representable diff is irregular). Throws
`ArgumentError`/`DimensionMismatch` for bad `first`/`count`.
"""
function describe_block(buffer::Vector{Int}, first::Int, count::Int)
    first >= 0 || throw(ArgumentError("first must be nonnegative, got $first"))
    count >= 0 || throw(ArgumentError("count must be nonnegative, got $count"))
    first + count <= length(buffer) ||
        throw(
        DimensionMismatch(
            "buffer length $(length(buffer)) is less than first+count = $(first + count)"
        )
    )

    count == 0 && return BlockDescriptor(0, 0, 0, true)

    @inbounds base = buffer[first + 1]
    count == 1 && return BlockDescriptor(base, 0, 1, true)

    @inbounds stride, overflowed = Base.Checked.sub_with_overflow(buffer[first + 2], buffer[first + 1])
    overflowed && return BlockDescriptor(base, 0, count, false)

    @inbounds for t in 2:(count - 1)
        diff, ovf = Base.Checked.sub_with_overflow(buffer[first + t + 1], buffer[first + t])
        (ovf || diff != stride) && return BlockDescriptor(base, 0, count, false)
    end

    return BlockDescriptor(base, stride, count, true)
end

describe_block(buffer::Vector{Int}, count::Int) = describe_block(buffer, 0, count)

"""
    block_descriptors!(buffers::NTuple{P,Vector{Int}}, g::AxisGroup{D,P}, first::Int, count::Int)

Fill `buffers` via [`fill_offsets!`](@ref) and classify each via
[`describe_block`](@ref); returns `NTuple{P,BlockDescriptor}`.
"""
function block_descriptors!(
        buffers::NTuple{P, Vector{Int}}, g::AxisGroup{D, P},
        first::Int, count::Int
    ) where {D, P}
    fill_offsets!(buffers, g, first, count)
    return ntuple(p -> describe_block(buffers[p], count), Val(P))
end

"""
    normalize_group(g::AxisGroup)::AxisGroup

Return a (possibly lower-rank) `AxisGroup` with the same `axis_length` and
per-map offset sequence: drop singleton dims, then fold adjacent dims
fastest-to-slowest wherever `next_stride[p] == current_length*current_stride[p]`
holds for *every* map. Planning-time only (may allocate); never reorders axes.
"""
function normalize_group(g::AxisGroup{D, P}) where {D, P}
    Q = axis_length(g)
    Q == 0 && return g  # empty domain: leave untouched.

    # Step 2: drop singleton dimensions.
    dims = Tuple{Int, NTuple{P, Int}}[]
    for d in 1:D
        L = g.lengths[d]
        if L != 1
            push!(dims, (L, ntuple(p -> g.strides[p][d], P)))
        end
    end

    # Step 5: all dimensions were singleton -> rank-zero group.
    if isempty(dims)
        emptylengths = NTuple{0, Int}()
        emptystrides = ntuple(_ -> NTuple{0, Int}(), P)
        return AxisGroup(emptylengths, emptystrides)
    end

    # Steps 3-4: fold adjacent dimensions fastest to slowest.
    folded = Tuple{Int, NTuple{P, Int}}[]
    curL, curS = dims[1]
    for i in 2:length(dims)
        nextL, nextS = dims[i]
        foldable = true
        for p in 1:P
            # Compare in Int128 so a non-representable product is simply
            # "not equal", never a wrapped accidental match.
            if Int128(curL) * Int128(curS[p]) != Int128(nextS[p])
                foldable = false
                break
            end
        end
        if foldable
            # curL * nextL is a sub-product of the original (validated) Q,
            # since every dropped/retained length is >= 1; it cannot overflow.
            curL = curL * nextL
        else
            push!(folded, (curL, curS))
            curL, curS = nextL, nextS
        end
    end
    push!(folded, (curL, curS))

    newD = length(folded)
    newlengths = ntuple(i -> folded[i][1], newD)
    newstrides = ntuple(p -> ntuple(i -> folded[i][2][p], newD), P)

    return AxisGroup(newlengths, newstrides)
end
