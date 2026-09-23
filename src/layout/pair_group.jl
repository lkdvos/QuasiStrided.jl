# Two-map AxisGroups built from label positions in a pair of operands.

# Build the two-map AxisGroup for one of M/N/K: (v1,v2) is (A,C)/(B,C)/(A,B).
# Raises DimensionMismatch on a matched-label length mismatch.
#
# `D = length(labels)` is a RUNTIME value (which labels are shared is a
# property of the label values, not of their tuple types), so `ntuple`s built
# directly from it are runtime-length -- inferred as `Tuple{Vararg{Int}}`,
# heap-boxed, and read back through a dynamic `getindex`, allocating per group
# even at `D == 1`. `D` is bounded above by `N1` (every label here occurs in
# `ind1`), which IS compile-time known, so the rank is resolved once through
# the unrolled `_pair_group_rank` ladder below and the body then runs at a
# literal `Val{D}` with statically sized tuples throughout.
function _build_pair_group(
        labels::Vector{Int},
        ind1::NTuple{N1, Int}, v1::StridedView,
        ind2::NTuple{N2, Int}, v2::StridedView
    ) where {N1, N2}
    return _pair_group_rank(Val(N1), labels, ind1, v1, ind2, v2)
end

# Unrolled rank ladder: `length(labels) <= N1` always, so descending from
# `Val(N1)` reaches the matching literal in at most `N1 + 1` compares, each arm
# calling a concretely-typed `_pair_group_static`. A plain `Val(D)` on a
# runtime `D` would be a dynamic dispatch instead.
@inline function _pair_group_rank(
        ::Val{K}, labels::Vector{Int}, ind1, v1, ind2, v2
    ) where {K}
    length(labels) == K && return _pair_group_static(Val(K), labels, ind1, v1, ind2, v2)
    return _pair_group_rank(Val(K - 1), labels, ind1, v1, ind2, v2)
end

@inline _pair_group_rank(::Val{0}, labels::Vector{Int}, ind1, v1, ind2, v2) =
    _pair_group_static(Val(0), labels, ind1, v1, ind2, v2)

@inline function _pair_group_static(
        ::Val{D}, labels::Vector{Int},
        ind1::NTuple{N1, Int}, v1::StridedView,
        ind2::NTuple{N2, Int}, v2::StridedView
    ) where {D, N1, N2}
    # Hoisted out of the per-dimension closures: `Base.strides` on a
    # `StridedView` rebuilds a tuple, so call it once, not once per `d`.
    st1 = Base.strides(v1)
    st2 = Base.strides(v2)
    pos1 = ntuple(d -> findfirst(==(@inbounds labels[d]), ind1)::Int, Val(D))
    pos2 = ntuple(d -> findfirst(==(@inbounds labels[d]), ind2)::Int, Val(D))
    lens = ntuple(Val(D)) do d
        l1 = size(v1, pos1[d])
        l2 = size(v2, pos2[d])
        l1 == l2 || _throw_label_length((@inbounds labels[d]), l1, l2)
        l1
    end
    s1 = ntuple(d -> st1[pos1[d]], Val(D))
    s2 = ntuple(d -> st2[pos2[d]], Val(D))
    return AxisGroup(lens, (s1, s2))
end
