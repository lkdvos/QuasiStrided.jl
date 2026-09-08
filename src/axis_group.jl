# OWNER: indexing implementer (Phase 1). See docs/decisions.md and
# Julia-Tensor-Indexing-Agent-Spec.md for the full contract.
#
# Implements: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
# describe_block, block_descriptors!, normalize_group.
#
# Conventions (see the spec for the full rationale):
#   - Logical coordinates and element offsets are zero-based.
#   - The first (fastest) dimension of an AxisGroup varies fastest, matching
#     column-major / Fortran-order tensor storage.
#   - Offsets are relative: an AxisGroup knows nothing about a tensor's base
#     pointer, storage bounds, element type, or conjugation.

# ----------------------------------------------------------------------------
# AxisGroup
# ----------------------------------------------------------------------------

"""
    AxisGroup{D,P}

A shared logical coordinate enumeration over `D` dimensions with lengths
`lengths::NTuple{D,Int}`, together with `P` physical offset maps
`strides::NTuple{P,NTuple{D,Int}}` (one map per participating tensor operand).

Coordinates are zero-based and the first dimension varies fastest: for
`0 <= q < axis_length(g)`,

    x[d] = (q ÷ prod(lengths[1:d-1])) % lengths[d]
    offset[p](q) = sum(x[d] * strides[p][d] for d in 1:D)

`AxisGroup` is a pure layout descriptor: it carries no base offset, no
storage, no element type, and no conjugation information. Construction
validates that every quantity used by [`offsets`](@ref) and
[`fill_offsets!`](@ref) is representable as `Int` without overflow; see the
package documentation / spec for the exact conservative bound used.

Use [`axis_length`](@ref) to query the cardinality; `AxisGroup` intentionally
does not implement the `AbstractArray` interface.
"""
struct AxisGroup{D,P}
    lengths::NTuple{D,Int}
    strides::NTuple{P,NTuple{D,Int}}

    function AxisGroup{D,P}(lengths::NTuple{D,Int},
                            strides::NTuple{P,NTuple{D,Int}}) where {D,P}
        P >= 1 || throw(ArgumentError("AxisGroup requires at least one map (P >= 1), got P = $P"))
        for (d, L) in enumerate(lengths)
            L >= 0 || throw(ArgumentError("AxisGroup lengths must be nonnegative, got lengths[$d] = $L"))
        end
        _validate_axis_group_bounds(lengths, strides)
        return new{D,P}(lengths, strides)
    end
end

"""
    AxisGroup(lengths::NTuple{D,Int}, strides::NTuple{P,NTuple{D,Int}}) where {D,P}

Construct a validated `AxisGroup`. `D` may be zero (a rank-zero group, whose
only valid coordinate is `q = 0`); `P` must be at least one. Lengths must be
nonnegative; strides may be any sign, including zero. Dimensions are kept in
the order supplied: construction never reorders, removes, or folds axes (see
[`normalize_group`](@ref) for that, as an explicit, separate step).

Throws `ArgumentError` for invalid lengths or `P < 1`, and `OverflowError` if
the cardinality or any map's maximum offset excursion is not representable as
`Int` (see the overflow policy in the spec, section 4).
"""
AxisGroup(lengths::NTuple{D,Int}, strides::NTuple{P,NTuple{D,Int}}) where {D,P} =
    AxisGroup{D,P}(lengths, strides)

# Checked cardinality: Q = prod(lengths), with an empty domain (any length
# zero) always yielding Q = 0, even if the product of the nonzero lengths
# would itself overflow. Uses Int128 accumulation to detect overflow without
# ever wrapping; this is validation-time-only arithmetic (construction may
# allocate/widen; hot execution below never does).
function _checked_axis_length(lengths::NTuple{D,Int}) where {D}
    any(==(0), lengths) && return 0
    q = one(Int128)
    for L in lengths
        q *= Int128(L)
        q > Int128(typemax(Int)) &&
            throw(OverflowError("AxisGroup cardinality (product of lengths) exceeds typemax(Int)"))
    end
    return Int(q)
end

# Full construction-time validation: cardinality, and (for a nonempty domain)
# the conservative per-map offset-excursion bound from the spec:
#     sum((L[d]-1) * abs(S[p][d]) for d) <= typemax(Int)
# evaluated in Int128 so that neither the multiplication, the sum, nor
# abs(typemin(Int)) can silently wrap.
function _validate_axis_group_bounds(lengths::NTuple{D,Int},
                                      strides::NTuple{P,NTuple{D,Int}}) where {D,P}
    Q = _checked_axis_length(lengths)
    Q == 0 && return nothing  # empty domain: no offsets exist, nothing to validate.
    for (p, S) in enumerate(strides)
        acc = zero(Int128)
        for d in 1:D
            L = lengths[d]  # >= 1 here, since Q > 0 implies no zero length.
            acc += Int128(L - 1) * abs(Int128(S[d]))
            acc > Int128(typemax(Int)) &&
                throw(OverflowError(
                    "AxisGroup map $p: sum((L[d]-1)*abs(S[d])) exceeds typemax(Int); " *
                    "this layout is not representable under the conservative offset-range bound"))
        end
    end
    return nothing
end

"""
    axis_length(g::AxisGroup)::Int

The cardinality `Q = prod(lengths)` of the group's coordinate enumeration
(with the empty product, `D == 0`, equal to one). Already validated at
construction to be representable as `Int`, so this uses plain `Int`
arithmetic.
"""
axis_length(g::AxisGroup) = _unchecked_axis_length(g.lengths)

@inline function _unchecked_axis_length(lengths::NTuple{D,Int}) where {D}
    q = 1
    for L in lengths
        q *= L
    end
    return q
end

# Stack-allocated, type-stable "replace element i of an NTuple{N,Int}".
# Avoids depending on the (undocumented) Base.setindex for tuples.
@inline _tupleset(t::NTuple{N,Int}, i::Int, v::Int) where {N} =
    ntuple(j -> ifelse(j == i, v, t[j]), Val(N))

# The following three helpers exist purely to dodge a Julia closure-boxing
# pitfall: a `ntuple(p -> offs[p] + ..., Val(P))` lambda written *inline*
# inside `offsets`/`fill_offsets!` captures `offs` (and `d`), which are
# reassigned in the enclosing loop — that reassignment forces the compiler to
# box the captured variable, which allocates on every call. Lifting the
# update into a dedicated function makes `offs`/`g`/`d`/`x` plain, never-
# reassigned arguments in a fresh scope, so no boxing occurs and these stay
# fully stack-allocated. Verified with `@allocated` in the benchmark script.
@inline function _add_offsets(offs::NTuple{P,Int}, g::AxisGroup{D,P}, d::Int, x::Int) where {D,P}
    return ntuple(p -> offs[p] + x * g.strides[p][d], Val(P))
end

@inline function _sub_reset_offsets(offs::NTuple{P,Int}, g::AxisGroup{D,P}, d::Int, Ld::Int) where {D,P}
    return ntuple(p -> offs[p] - (Ld - 1) * g.strides[p][d], Val(P))
end

# ----------------------------------------------------------------------------
# 5.1 Reference random access
# ----------------------------------------------------------------------------

"""
    offsets(g::AxisGroup{D,P}, q::Int)::NTuple{P,Int}

Decode a single logical coordinate `0 <= q < axis_length(g)` into the `P`
relative element offsets, one per map. Implemented as a direct mixed-radix
decode (`O(D)` divisions); this is the reference definition that
[`fill_offsets!`](@ref) must reproduce for every requested `q`, not a
fast path. Throws `BoundsError` if `q` is out of `[0, axis_length(g))`.
"""
function offsets(g::AxisGroup{D,P}, q::Int) where {D,P}
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

# ----------------------------------------------------------------------------
# 5.2 Interval generation
# ----------------------------------------------------------------------------

"""
    fill_offsets!(buffers::NTuple{P,Vector{Int}}, g::AxisGroup{D,P}, first::Int, count::Int)

Fill `buffers[p][t+1] = offsets(g, first + t)[p]` for every map `p` and
`0 <= t < count`, using a single mixed-radix decode of `first` followed by
reset-before-carry increments (spec section 6) — never a per-element
`divrem` and never a naive `L[d] * stride` reset step. Returns `buffers`.

Requires `count >= 0`, `0 <= first <= axis_length(g)`, and
`count <= axis_length(g) - first` (checked without ever forming a possibly
overflowing `first + count`). Every buffer must have `length(buffer) >=
count`, and the `P` buffers must be pairwise distinct `Vector{Int}` objects.
All arguments and buffers are validated before any mutation; entries at
positions `count+1:end` are left untouched. An empty interval (`count == 0`)
performs no writes and no coordinate decoding, but is still fully validated.

Throws `ArgumentError` for a negative `count` or repeated buffer objects,
`BoundsError` for an out-of-domain interval, and `DimensionMismatch` for an
insufficient buffer length.
"""
function fill_offsets!(buffers::NTuple{P,Vector{Int}}, g::AxisGroup{D,P},
                        first::Int, count::Int) where {D,P}
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
    for i in 1:P, j in (i+1):P
        buffers[i] === buffers[j] &&
            throw(ArgumentError("buffers must be distinct Vector{Int} objects (buffers $i and $j alias)"))
    end

    count == 0 && return buffers

    # Decode `first` once into coordinates and the corresponding initial
    # offsets for every map (step 3).
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
            buffers[p][t+1] = offs[p]
        end
        t += 1
        t == count && break

        # Advance coordinates fastest-first, reset-before-carry (step 6).
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

# ----------------------------------------------------------------------------
# 5.3 Block descriptor
# ----------------------------------------------------------------------------

"""
    BlockDescriptor

Classification of a materialized offset interval for one map:

- `base`: the offset of the first element (or `0` for an empty interval).
- `stride`: the constant per-step stride when `regular == true` and
  `count >= 2`; otherwise not meaningful (`0`).
- `count`: the number of valid elements the descriptor describes.
- `regular`: whether `buffer[t+1] == base + t*stride` holds exactly for all
  `0 <= t < count`.

When `regular == false`, `base` and `stride` do not define addressing;
consumers must read the underlying buffer directly. A `BlockDescriptor`
becomes stale as soon as its source buffer is refilled with a different
interval — it owns no offset storage of its own.
"""
struct BlockDescriptor
    base::Int
    stride::Int
    count::Int
    regular::Bool
end

"""
    describe_block(buffer::Vector{Int}, count::Int)::BlockDescriptor

Classify `buffer[1:count]` (read-only; `buffer` is never mutated) as empty,
singleton, affine, or irregular:

| Interval             | base       | stride                | regular |
|----------------------|------------|------------------------|---------|
| Empty (`count == 0`) | `0`        | `0`                    | `true`  |
| Singleton            | `buffer[1]`| `0`                    | `true`  |
| Affine, `count >= 2` | `buffer[1]`| `buffer[2]-buffer[1]`  | `true`  |
| Irregular            | `buffer[1]`| `0`                    | `false` |

All adjacent-difference comparisons use overflow-checked subtraction; a
difference that is not representable as `Int` is classified irregular rather
than compared against a wrapped value. Throws `ArgumentError` for a negative
`count` and `DimensionMismatch` if `count` exceeds `length(buffer)`.
"""
function describe_block(buffer::Vector{Int}, count::Int)
    count >= 0 || throw(ArgumentError("count must be nonnegative, got $count"))
    count <= length(buffer) ||
        throw(DimensionMismatch("buffer length $(length(buffer)) is less than count $count"))

    count == 0 && return BlockDescriptor(0, 0, 0, true)

    @inbounds base = buffer[1]
    count == 1 && return BlockDescriptor(base, 0, 1, true)

    @inbounds stride, overflowed = Base.Checked.sub_with_overflow(buffer[2], buffer[1])
    overflowed && return BlockDescriptor(base, 0, count, false)

    @inbounds for t in 2:(count-1)
        diff, ovf = Base.Checked.sub_with_overflow(buffer[t+1], buffer[t])
        (ovf || diff != stride) && return BlockDescriptor(base, 0, count, false)
    end

    return BlockDescriptor(base, stride, count, true)
end

# ----------------------------------------------------------------------------
# 5.4 Combined convenience operation
# ----------------------------------------------------------------------------

"""
    block_descriptors!(buffers::NTuple{P,Vector{Int}}, g::AxisGroup{D,P}, first::Int, count::Int)

Fill `buffers` via [`fill_offsets!`](@ref) and classify each map's resulting
buffer contents via [`describe_block`](@ref), returning
`NTuple{P,BlockDescriptor}`. Shares `fill_offsets!`'s validation and mutation
contract; the returned descriptors all describe the same interval, but
regularity may differ from map to map.
"""
function block_descriptors!(buffers::NTuple{P,Vector{Int}}, g::AxisGroup{D,P},
                             first::Int, count::Int) where {D,P}
    fill_offsets!(buffers, g, first, count)
    return ntuple(p -> describe_block(buffers[p], count), Val(P))
end

# ----------------------------------------------------------------------------
# 7. Explicit normalization
# ----------------------------------------------------------------------------

"""
    normalize_group(g::AxisGroup)::AxisGroup

Return a (possibly lower-rank) `AxisGroup` with the same `axis_length` and
the same complete flattened offset sequence for every map, obtained by:

1. Returning `g` unchanged if its domain is empty (some length is zero).
2. Dropping singleton dimensions (`length == 1`), whose strides are
   irrelevant.
3. Scanning the remaining dimensions fastest to slowest and folding adjacent
   dimensions whenever, for *every* map `p` simultaneously,
   `next_stride[p] == current_length * current_stride[p]` holds exactly
   (checked in a wider integer type so a non-representable product is simply
   treated as "does not fold", never as a wrapped false match).
4. If every original dimension was singleton, returning a rank-zero group
   with the same map count.

Folding is refused unless it holds for every map — a group may be affine for
one map and not another without being jointly foldable; that per-map
classification is what [`describe_block`](@ref) is for, not this function.
Axes are never reordered. The result is revalidated through the `AxisGroup`
constructor.

Intended for planning time (it may allocate); execute interval generation
against whichever group — normalized or not — you have, since correctness
must not depend on prior normalization.
"""
function normalize_group(g::AxisGroup{D,P}) where {D,P}
    Q = axis_length(g)
    Q == 0 && return g  # empty domain: leave untouched.

    # Step 2: drop singleton dimensions.
    dims = Tuple{Int,NTuple{P,Int}}[]
    for d in 1:D
        L = g.lengths[d]
        if L != 1
            push!(dims, (L, ntuple(p -> g.strides[p][d], P)))
        end
    end

    # Step 5: all dimensions were singleton -> rank-zero group.
    if isempty(dims)
        emptylengths = NTuple{0,Int}()
        emptystrides = ntuple(_ -> NTuple{0,Int}(), P)
        return AxisGroup(emptylengths, emptystrides)
    end

    # Steps 3-4: fold adjacent dimensions fastest to slowest.
    folded = Tuple{Int,NTuple{P,Int}}[]
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
