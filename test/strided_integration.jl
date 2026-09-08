# StridedViews integration, plus a test-only packing/contraction consumer,
# written against the frozen spec without reading src/axis_group.jl's internals.

using Test
using StridedViews: StridedView, offset
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, normalize_group

# Test-only adapter: StridedView metadata -> AxisGroup (local to this file,
# not a production API).
#
# Uses only documented StridedViews accessors: `size`, `strides` (both
# generic AbstractArray/Base methods that StridedView supports), and the
# exported `offset(::StridedView)` giving the zero-based base offset into
# the parent storage (verified against the installed StridedViews v0.5.2
# source, stridedview.jl:115,120).

"""
    group_from_views(views, axeslist) -> AxisGroup

`views` is a tuple of P StridedViews (one per participating map). `axeslist`
is a matching tuple of D-tuples of 1-based axis positions *within each
view* -- one axis list per view, since the same logical label need not sit
at the same axis position in every operand (e.g. the worked example's K
group: k is axis 2 of A but axis 1 of B). All views must agree on the
length at their respective mapped axis.
"""
function group_from_views(
        views::NTuple{P, StridedView}, axeslist::NTuple{P, NTuple{D, Int}}
    ) where {P, D}
    lens = D == 0 ? () : ntuple(d -> size(views[1], axeslist[1][d]), D)
    for p in 1:P, d in 1:D
        size(views[p], axeslist[p][d]) == lens[d] ||
            throw(DimensionMismatch("view $p disagrees on mapped axis length"))
    end
    smaps = ntuple(P) do p
        D == 0 ? () : ntuple(d -> Base.strides(views[p])[axeslist[p][d]], D)
    end
    return AxisGroup(lens, smaps)
end

# =====================================================================
# Worked example (spec section 8), built from real StridedViews
# =====================================================================

@testset "Strided integration: worked example, full arrays" begin
    A = randn(3, 5, 2)  # A[a,k,b]
    B = randn(5, 4)     # B[k,n]
    C = zeros(3, 4, 2)  # C[a,n,b]

    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
    @test offset(Av) == 0 && offset(Bv) == 0 && offset(Cv) == 0

    M = group_from_views((Av, Cv), ((1, 3), (1, 3))) # A,C: a,b
    N = group_from_views((Bv, Cv), ((2,), (2,)))     # B,C: n
    K = group_from_views((Av, Bv), ((2,), (1,)))     # A,B: k

    @test axis_length(M) == 6
    @test axis_length(N) == 4
    @test axis_length(K) == 5

    bufA, bufC = zeros(Int, 6), zeros(Int, 6)
    fill_offsets!((bufA, bufC), M, 0, 6)
    @test bufA == [0, 1, 2, 15, 16, 17]
    @test bufC == [0, 1, 2, 12, 13, 14]

    bufB, bufC2 = zeros(Int, 4), zeros(Int, 4)
    fill_offsets!((bufB, bufC2), N, 0, 4)
    @test bufB == [0, 5, 10, 15]
    @test bufC2 == [0, 3, 6, 9]

    bufA2, bufB2 = zeros(Int, 5), zeros(Int, 5)
    fill_offsets!((bufA2, bufB2), K, 0, 5)
    @test bufA2 == [0, 3, 6, 9, 12]
    @test bufB2 == [0, 1, 2, 3, 4]
end

# =====================================================================
# Generic full-rank cross-check: AxisGroup offsets vs direct StridedView
# indexing, for permuted / sliced / negative-stride / zero-stride views.
# =====================================================================

"""
    check_view_axisgroup(v, axes)

Build a single-map AxisGroup covering *all* of `v`'s dimensions (in the
order given by `axes`, a permutation of 1:ndims(v)) and check, for every
coordinate, that `parent(v)[offset(v) + relative_offset + 1]` equals `v`'s
own element at the corresponding index -- i.e. cross-check AxisGroup against
Julia's own StridedView indexing, independent of the oracle in
test_axis_group.jl.
"""
function check_view_axisgroup(v::StridedView, axes::NTuple{D, Int}) where {D}
    @assert length(axes) == ndims(v) && sort(collect(axes)) == collect(1:ndims(v))
    lens = ntuple(d -> size(v, axes[d]), D)
    smap = ntuple(d -> Base.strides(v)[axes[d]], D)
    g = AxisGroup(lens, (smap,))
    Q = axis_length(g)
    @test Q == length(v)
    par = parent(v)
    b = offset(v)
    for q in 0:(Q - 1)
        (relo,) = offsets(g, q)
        ci = CartesianIndices(lens)[q + 1]
        vidx = Vector{Int}(undef, ndims(v))
        for (k, ax) in enumerate(axes)
            vidx[ax] = ci[k]
        end
        @test par[b + relo + 1] == v[vidx...]
    end
end

@testset "Strided integration: permuted view" begin
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    Av = StridedView(A)
    Avp = permutedims(Av, (3, 1, 2)) # size (2,3,5), same parent, permuted strides
    @test size(Avp) == (2, 3, 5)
    check_view_axisgroup(Avp, (1, 2, 3))
end

@testset "Strided integration: sliced view" begin
    A = reshape(collect(1.0:60.0), 3, 5, 4)
    Av = StridedView(A)
    sv = view(Av, 2:3, 1:3, 4:2:4) # nontrivial ranges on all three axes
    @test sv isa StridedView
    @test size(sv) == (2, 3, 1)
    @test offset(sv) != 0 # slicing away from the origin shifts the base
    check_view_axisgroup(sv, (1, 2, 3))
end

@testset "Strided integration: permuted AND sliced view" begin
    A = reshape(collect(1.0:120.0), 4, 5, 6)
    Av = StridedView(A)
    sv = view(Av, 2:4, 2:5, 1:2:5)
    svp = permutedims(sv, (3, 1, 2))
    check_view_axisgroup(svp, (1, 2, 3))
end

@testset "Strided integration: negative-stride input, adjusted base" begin
    # Manually construct a StridedView with a negative stride and a base
    # offset adjusted to keep every access in-bounds, per spec section 9
    # ("negative offsets are valid relative offsets and require a suitable
    # operand base"). This mirrors what a reversed view produces.
    parentvec = collect(1.0:12.0)
    # Represent a 4x3 column-major matrix, but with dim 1 reversed: element
    # (i,j) (1-based, i in 1:4) should read parentvec[(4-i) + 1 + 3*(j-1)].
    # That is base=3 (zero-based offset to row index 3, the last row),
    # stride1=-1, stride2=3.
    sv = StridedView(parentvec, (4, 3), (-1, 3), 3)
    @test offset(sv) == 3
    for i in 1:4, j in 1:3
        @test sv[i, j] == parentvec[3 + (i - 1) * (-1) + (j - 1) * 3 + 1]
    end
    check_view_axisgroup(sv, (1, 2))

    # Also drive it through group_from_views as a single-view "map".
    g = group_from_views((sv,), ((1, 2),))
    @test axis_length(g) == 12
    buf = zeros(Int, 12)
    fill_offsets!((buf,), g, 0, 12)
    par = parent(sv)
    b = offset(sv)
    for q in 0:11
        ci = CartesianIndices((4, 3))[q + 1]
        @test par[b + buf[q + 1] + 1] == sv[ci[1], ci[2]]
    end
end

@testset "Strided integration: zero-stride input" begin
    # A broadcast-like view: one dimension has stride 0, so every index
    # along it reads the same underlying element. Constructed manually since
    # this is not expressible via permutedims/slicing alone.
    parentvec = collect(1.0:5.0)
    sv = StridedView(parentvec, (5, 3), (1, 0), 0) # 5x3, column j is a repeat of parentvec
    @test offset(sv) == 0
    for i in 1:5, j in 1:3
        @test sv[i, j] == parentvec[i]
    end
    check_view_axisgroup(sv, (1, 2))

    g = group_from_views((sv,), ((1, 2),))
    Q = axis_length(g)
    @test Q == 15
    buf = zeros(Int, Q)
    fill_offsets!((buf,), g, 0, Q)
    # column strides are all zero: descriptor for a full column (count=5,
    # varying only dim 1) is regular with stride 1; for a full row-run
    # across the zero-stride dimension it must be regular with stride 0.
    d = describe_block(buf[1:5], 5)
    @test d.regular == true && d.stride == 1
end

# =====================================================================
# Test-only packer (spec section 10) -- NOT a production packing format.
# This is deliberately a simple, explicit rectangle packer to exercise the
# indexing interface end to end; it must not be confused with the separate,
# later, production packer described in
# Julia-Microkernel-Tile-Interface-Design.md.
# =====================================================================

"""
    row_addressing(desc, buf) -> Function

Select the row (or column) addressing path once per block, per spec section
10: if `desc.regular`, return a closure computing `base + (i-1)*stride`;
otherwise return a closure reading `buf[i]`. Callers must not re-branch on
`desc.regular` inside the per-element copy loop.
"""
function block_addressing(desc::BlockDescriptor, buf::Vector{Int})
    if desc.regular
        base, stride = desc.base, desc.stride
        return i -> base + (i - 1) * stride
    else
        return i -> buf[i]
    end
end

"""
    pack_rectangle!(dest, parent, base, row_desc, row_buf, col_desc, col_buf)

Test-only column-major packer. Fills `dest[1:row_desc.count, 1:col_desc.count]`
from `parent` at `parent[base + row_offset[i] + col_offset[j] + 1]`
(1-based Julia storage indexing, per spec section 9). The row/column
addressing path is selected once (via `block_addressing`), not re-branched
per element.
"""
function pack_rectangle!(
        dest::AbstractMatrix{T}, par::AbstractVector{T}, base::Int,
        row_desc::BlockDescriptor, row_buf::Vector{Int},
        col_desc::BlockDescriptor, col_buf::Vector{Int}
    ) where {T}
    nrows, ncols = row_desc.count, col_desc.count
    @assert size(dest) == (nrows, ncols)
    rowfn = block_addressing(row_desc, row_buf)
    colfn = block_addressing(col_desc, col_buf)
    for j in 1:ncols
        co = colfn(j)
        for i in 1:nrows
            dest[i, j] = par[base + rowfn(i) + co + 1]
        end
    end
    return dest
end

@testset "test-only packer: regular row/col vs direct indexing" begin
    A = reshape(collect(1.0:30.0), 3, 5, 2) # A[a,k,b]
    Av = StridedView(A)
    b_fixed = 2 # pack the (a,k) slab at fixed b=2
    par = parent(Av)
    base0 = offset(Av) + (b_fixed - 1) * Base.strides(Av)[3]

    rowg = AxisGroup((3,), (Base.strides(Av)[1:1],)) # a
    colg = AxisGroup((5,), (Base.strides(Av)[2:2],)) # k
    rbuf, cbuf = zeros(Int, 3), zeros(Int, 5)
    (rdesc,) = block_descriptors!((rbuf,), rowg, 0, 3)
    (cdesc,) = block_descriptors!((cbuf,), colg, 0, 5)
    @test rdesc.regular && cdesc.regular

    dest = zeros(3, 5)
    pack_rectangle!(dest, par, base0, rdesc, rbuf, cdesc, cbuf)
    for a in 1:3, k in 1:5
        @test dest[a, k] == A[a, k, b_fixed]
    end
end

@testset "test-only packer: irregular map vs direct indexing" begin
    # G = AxisGroup((3,2), ((1,3),(1,10))) from spec section 8: map2 is
    # irregular. Use it as the *column* map against a regular row map, on a
    # synthetic parent buffer, and compare packed output to direct
    # elementwise addressing.
    par = collect(0.0:99.0) # values equal their own index for easy checking
    rowg = AxisGroup((4,), ((1,),))     # regular row map, stride 1
    colg = AxisGroup((3, 2), ((1, 10),)) # irregular column map (single-map view)
    rbuf = zeros(Int, 4)
    cbuf = zeros(Int, 6)
    (rdesc,) = block_descriptors!((rbuf,), rowg, 0, 4)
    (cdesc,) = block_descriptors!((cbuf,), colg, 0, 6)
    @test rdesc.regular == true
    @test cdesc.regular == false
    @test cbuf == [0, 1, 2, 10, 11, 12]

    base = 5
    dest = zeros(4, 6)
    pack_rectangle!(dest, par, base, rdesc, rbuf, cdesc, cbuf)
    for i in 1:4, j in 1:6
        @test dest[i, j] == par[base + rbuf[i] + cbuf[j] + 1]
    end
end

@testset "test-only packer: tail interval (non-full-multiple count)" begin
    par = collect(0.0:49.0)
    rowg = AxisGroup((7,), ((1,),))
    # Request a tail block of 3 rows starting at row 4 (of 7): not a full
    # panel, deliberately not covering the whole group.
    rbuf = zeros(Int, 3)
    (rdesc,) = block_descriptors!((rbuf,), rowg, 4, 3)
    @test rdesc.count == 3
    @test rdesc.regular == true

    colg = AxisGroup((2,), ((7,),))
    cbuf = zeros(Int, 2)
    (cdesc,) = block_descriptors!((cbuf,), colg, 0, 2)

    dest = zeros(3, 2)
    pack_rectangle!(dest, par, 0, rdesc, rbuf, cdesc, cbuf)
    for i in 1:3, j in 1:2
        @test dest[i, j] == par[rbuf[i] + cbuf[j] + 1]
    end
end

# =====================================================================
# Small test-only scalar a,n,b,k contraction (spec section 10/11)
# =====================================================================

"""
    scalar_contract!(C, cbase, A, abase, B, bbase, M, N, K)

Test-only scalar contraction C[m,n] += sum_k A[m,k]*B[k,n], generalized to
the M/N/K AxisGroup formulation of spec section 3/8: `M`/`N`/`K` each carry
two maps, (A,C), (B,C), (A,B) respectively. Establishes agreement of paired
group enumeration across A, B, and C end to end. Not a production kernel.
"""
function scalar_contract!(
        C::AbstractVector{Float64}, cbase::Int,
        A::AbstractVector{Float64}, abase::Int,
        B::AbstractVector{Float64}, bbase::Int,
        M::AxisGroup{DM, 2}, N::AxisGroup{DN, 2}, K::AxisGroup{DK, 2}
    ) where {DM, DN, DK}
    Qm, Qn, Qk = axis_length(M), axis_length(N), axis_length(K)
    for n in 0:(Qn - 1)
        (nB, nC) = offsets(N, n)
        for m in 0:(Qm - 1)
            (mA, mC) = offsets(M, m)
            acc = 0.0
            for k in 0:(Qk - 1)
                (kA, kB) = offsets(K, k)
                acc += A[abase + mA + kA + 1] * B[bbase + kB + nB + 1]
            end
            C[cbase + mC + nC + 1] += acc
        end
    end
    return C
end

@testset "consumer integration: scalar contraction vs direct a,n,b,k loop" begin
    Random.seed!(20260908)
    A = randn(3, 5, 2)  # A[a,k,b]
    B = randn(5, 4)     # B[k,n]
    C1 = zeros(3, 4, 2) # via AxisGroup-driven scalar_contract!
    C2 = zeros(3, 4, 2) # via direct loop, independently

    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C1)
    M = group_from_views((Av, Cv), ((1, 3), (1, 3)))
    N = group_from_views((Bv, Cv), ((2,), (2,)))
    K = group_from_views((Av, Bv), ((2,), (1,)))

    scalar_contract!(
        vec(C1), offset(Cv), vec(A), offset(Av), vec(B), offset(Bv), M, N, K
    )

    for a in 1:3, n in 1:4, b in 1:2
        s = 0.0
        for k in 1:5
            s += A[a, k, b] * B[k, n]
        end
        C2[a, n, b] = s
    end

    @test C1 ≈ C2
end

@testset "consumer integration: scalar contraction on permuted views" begin
    Random.seed!(4242)
    A0 = randn(3, 5, 2) # a,k,b
    B0 = randn(5, 4)    # k,n
    # Permute A to (k,b,a) order and B to (n,k) order; still the same
    # tensors, just relabeled axis order in storage, to confirm the
    # AxisGroup-driven contraction is independent of physical axis order.
    Ap = permutedims(A0, (2, 3, 1)) # Ap[k,b,a] == A0[a,k,b]
    Bp = permutedims(B0, (2, 1))    # Bp[n,k] == B0[k,n]
    C1 = zeros(3, 4, 2) # a,n,b

    Av = StridedView(Ap)
    Bv = StridedView(Bp)
    Cv = StridedView(C1)

    M = group_from_views((Av, Cv), ((3, 2), (1, 3))) # a is axis3 of Av, axis1 of Cv; b is axis2 of Av, axis3 of Cv
    N = group_from_views((Bv, Cv), ((1,), (2,)))     # n is axis1 of Bv, axis2 of Cv
    K = group_from_views((Av, Bv), ((1,), (2,)))     # k is axis1 of Av, axis2 of Bv

    scalar_contract!(
        vec(C1), offset(Cv), vec(Ap), offset(Av), vec(Bp), offset(Bv), M, N, K
    )

    C2 = zeros(3, 4, 2)
    for a in 1:3, n in 1:4, b in 1:2
        s = 0.0
        for k in 1:5
            s += A0[a, k, b] * B0[k, n]
        end
        C2[a, n, b] = s
    end

    @test C1 ≈ C2
end
