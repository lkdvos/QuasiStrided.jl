# Independent verification of AxisGroup indexing: an oracle built from Julia's
# `CartesianIndices`, never calling offsets/fill_offsets!/block_descriptors!
# themselves.

using Test
using Random
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, normalize_group

# =====================================================================
# Independent Cartesian-coordinate oracle
# =====================================================================
#
# Per spec section 3: coordinates are x[d] = (q div prod(L[1:d-1])) mod L[d],
# first axis fastest, and offset[p](q) = sum(x[d]*S[p][d] for d). This is
# exactly what `CartesianIndices` over the length tuple enumerates (Julia's
# CartesianIndices is column-major: first dimension fastest), converted to
# zero-based coordinates.

"""
    oracle_all_offsets(lengths, strides) -> Vector{NTuple{P,Int}}

Enumerate every valid q in 0:Q-1 (Q = prod(lengths)) via `CartesianIndices`
and compute each map's offset directly from coordinates and strides. Never
touches AxisGroup.
"""
function oracle_all_offsets(lengths::NTuple{D, Int}, strides::NTuple{P, NTuple{D, Int}}) where {D, P}
    if D == 0
        # Q = 1 (empty product), single coordinate (), all offsets zero.
        return [ntuple(_ -> 0, P)]
    end
    if any(==(0), lengths)
        return NTuple{P, Int}[]
    end
    idxs = CartesianIndices(lengths)
    out = Vector{NTuple{P, Int}}(undef, length(idxs))
    for (li, ci) in enumerate(idxs)
        x = Tuple(ci) .- 1 # zero-based coordinates
        out[li] = ntuple(P) do p
            s = 0
            for d in 1:D
                s += x[d] * strides[p][d]
            end
            s
        end
    end
    return out
end

"""
    oracle_offset(lengths, strides, q) -> NTuple{P,Int}

Single-coordinate version of `oracle_all_offsets`, via direct indexing into
`CartesianIndices` (not via any AxisGroup method).
"""
function oracle_offset(lengths::NTuple{D, Int}, strides::NTuple{P, NTuple{D, Int}}, q::Int) where {D, P}
    if D == 0
        return ntuple(_ -> 0, P)
    end
    ci = CartesianIndices(lengths)[q + 1]
    x = Tuple(ci) .- 1
    return ntuple(P) do p
        s = 0
        for d in 1:D
            s += x[d] * strides[p][d]
        end
        s
    end
end

# Overflow-safe (Int128) reference helpers used to check descriptors and
# affine classification independently of AxisGroup's own overflow policy.
ref_affine_value(base::Int, stride::Int, t::Int) = Int128(base) + Int128(t) * Int128(stride)

"""
Independent affine check: does buffer[1:count] satisfy
buffer[t+1] == buffer[1] + t*stride for some Int stride, using Int128 math
throughout (never calling describe_block)? Returns (is_affine, stride_or_nothing).
"""
function independent_affine_check(buffer::Vector{Int}, count::Int)
    count <= 1 && return (true, 0)
    base128 = Int128(buffer[1])
    stride128 = Int128(buffer[2]) - base128
    for t in 0:(count - 1)
        if Int128(buffer[t + 1]) != base128 + Int128(t) * stride128
            return (false, nothing)
        end
    end
    # stride must itself be representable as Int for a true affine classification
    if stride128 < typemin(Int) || stride128 > typemax(Int)
        return (false, nothing)
    end
    return (true, Int(stride128))
end

# =====================================================================
# Oracle sanity: verify first-axis-fastest against the fixed fixtures
# =====================================================================

@testset "oracle: first-axis-fastest sanity" begin
    # D=2, lengths (3,2): first axis (length 3) must vary fastest.
    lengths = (3, 2)
    strides = ((1, 15),)
    got = oracle_all_offsets(lengths, strides)
    expected = [(0,), (1,), (2,), (15,), (16,), (17,)]
    @test got == expected

    # Single-coordinate oracle_offset agrees with oracle_all_offsets elementwise.
    for q in 0:5
        @test oracle_offset(lengths, strides, q) == expected[q + 1]
    end
end

# =====================================================================
# Fixed fixtures: worked A/B/C contraction example (spec section 8)
# =====================================================================

@testset "fixed fixture: worked A/B/C example" begin
    M = AxisGroup((3, 2), ((1, 15), (1, 12)))  # A, C
    N = AxisGroup((4,), ((5,), (3,)))          # B, C
    K = AxisGroup((5,), ((3,), (1,)))          # A, B

    @test axis_length(M) == 6
    @test axis_length(N) == 4
    @test axis_length(K) == 5

    # Table: complete offset sequences.
    exp_M_A = [0, 1, 2, 15, 16, 17]
    exp_M_C = [0, 1, 2, 12, 13, 14]
    exp_N_B = [0, 5, 10, 15]
    exp_N_C = [0, 3, 6, 9]
    exp_K_A = [0, 3, 6, 9, 12]
    exp_K_B = [0, 1, 2, 3, 4]

    for q in 0:5
        oA, oC = offsets(M, q)
        @test oA == exp_M_A[q + 1]
        @test oC == exp_M_C[q + 1]
        # Cross-check against the independent oracle too.
        @test (oA, oC) == oracle_offset((3, 2), ((1, 15), (1, 12)), q)
    end
    for q in 0:3
        oB, oC = offsets(N, q)
        @test oB == exp_N_B[q + 1]
        @test oC == exp_N_C[q + 1]
    end
    for q in 0:4
        oA, oB = offsets(K, q)
        @test oA == exp_K_A[q + 1]
        @test oB == exp_K_B[q + 1]
    end

    # fill_offsets! full-interval agreement.
    bufA, bufC = zeros(Int, 6), zeros(Int, 6)
    fill_offsets!((bufA, bufC), M, 0, 6)
    @test bufA == exp_M_A
    @test bufC == exp_M_C

    bufB2, bufC2 = zeros(Int, 4), zeros(Int, 4)
    fill_offsets!((bufB2, bufC2), N, 0, 4)
    @test bufB2 == exp_N_B
    @test bufC2 == exp_N_C

    bufA3, bufB3 = zeros(Int, 5), zeros(Int, 5)
    fill_offsets!((bufA3, bufB3), K, 0, 5)
    @test bufA3 == exp_K_A
    @test bufB3 == exp_K_B

    # "m=4, k=2 -> A relative offset 22" spot check.
    m, k = 4, 2
    oA_m, oC_m = offsets(M, m)
    oA_k, oB_k = offsets(K, k)
    @test oA_m + oA_k == 22
end

@testset "fixed fixture: M interval -> descriptor table" begin
    M = AxisGroup((3, 2), ((1, 15), (1, 12)))
    cases = [
        (0, 3, (0, 1, 3, true), (0, 1, 3, true)),
        (3, 3, (15, 1, 3, true), (12, 1, 3, true)),
        (0, 4, (0, 0, 4, false), (0, 0, 4, false)),
        (2, 2, (2, 13, 2, true), (2, 10, 2, true)),
        (6, 0, (0, 0, 0, true), (0, 0, 0, true)),
    ]
    for (first, count, expA, expC) in cases
        bufA, bufC = zeros(Int, max(count, 1)), zeros(Int, max(count, 1))
        dA, dC = block_descriptors!((bufA, bufC), M, first, count)
        @test (dA.base, dA.stride, dA.count, dA.regular) == expA
        @test (dC.base, dC.stride, dC.count, dC.regular) == expC
    end
end

@testset "fixed fixture: G = AxisGroup((3,2), ((1,3),(1,10))) non-foldable" begin
    G = AxisGroup((3, 2), ((1, 3), (1, 10)))
    @test axis_length(G) == 6
    buf1, buf2 = zeros(Int, 6), zeros(Int, 6)
    d1, d2 = block_descriptors!((buf1, buf2), G, 0, 6)
    @test d1.regular == true
    @test d1.base == 0 && d1.stride == 1 && d1.count == 6
    @test d2.regular == false
    @test buf2 == [0, 1, 2, 10, 11, 12]
    @test oracle_all_offsets((3, 2), ((1, 3), (1, 10))) ==
        [(o1, o2) for (o1, o2) in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 10, 11, 12])]

    Gn = normalize_group(G)
    @test axis_length(Gn) == 6
    # MUST NOT jointly fold: a folded rank-1 group would need a single stride
    # per map reproducing both sequences, which map 2 cannot satisfy since it
    # is not affine. So Gn must remain rank >= 2 (not folded to D=1).
    @test length(Gn.lengths) >= 2
    # And it must still reproduce the exact same offset sequences.
    buf1n, buf2n = zeros(Int, 6), zeros(Int, 6)
    fill_offsets!((buf1n, buf2n), Gn, 0, 6)
    @test buf1n == buf1
    @test buf2n == buf2
end

# =====================================================================
# Rank-zero, 1-D, and multi-map groups
# =====================================================================

@testset "rank-zero groups" begin
    g = AxisGroup((), ((),))
    @test axis_length(g) == 1
    @test offsets(g, 0) == (0,)
    @test_throws BoundsError offsets(g, 1)
    @test_throws BoundsError offsets(g, -1)

    buf = [999]
    fill_offsets!((buf,), g, 0, 1)
    @test buf == [0]

    # valid empty interval at 0 and at Q=1
    buf2 = [999]
    fill_offsets!((buf2,), g, 0, 0)
    @test buf2 == [999]
    buf3 = [999]
    fill_offsets!((buf3,), g, 1, 0)
    @test buf3 == [999]

    # multi-map rank-zero
    g3 = AxisGroup((), ((), (), ()))
    @test axis_length(g3) == 1
    @test offsets(g3, 0) == (0, 0, 0)
    d = describe_block([0], 1)
    @test (d.base, d.stride, d.count, d.regular) == (0, 0, 1, true)
end

@testset "1-D groups" begin
    g = AxisGroup((5,), ((3,), (-2,)))
    @test axis_length(g) == 5
    for q in 0:4
        @test offsets(g, q) == oracle_offset((5,), ((3,), (-2,)), q)
    end
    bufA, bufB = zeros(Int, 5), zeros(Int, 5)
    fill_offsets!((bufA, bufB), g, 0, 5)
    @test bufA == [0, 3, 6, 9, 12]
    @test bufB == [0, -2, -4, -6, -8]
end

@testset "multiple operand maps (P >= 3)" begin
    g = AxisGroup((2, 3), ((1, 2), (10, 100), (-1, -10)))
    Q = axis_length(g)
    @test Q == 6
    for q in 0:(Q - 1)
        @test offsets(g, q) == oracle_offset((2, 3), ((1, 2), (10, 100), (-1, -10)), q)
    end
    bufs = (zeros(Int, Q), zeros(Int, Q), zeros(Int, Q))
    fill_offsets!(bufs, g, 0, Q)
    expected = oracle_all_offsets((2, 3), ((1, 2), (10, 100), (-1, -10)))
    for p in 1:3
        @test [expected[i][p] for i in 1:Q] == bufs[p]
    end
end

# =====================================================================
# Zero-length dimensions
# =====================================================================

@testset "zero-length dimensions" begin
    g = AxisGroup((3, 0, 5), ((1, 10, 100),))
    @test axis_length(g) == 0
    @test_throws BoundsError offsets(g, 0)
    buf = [999]
    # only valid interval is (0,0) (and (Q,0)=(0,0) since Q=0)
    fill_offsets!((buf,), g, 0, 0)
    @test buf == [999]
    @test_throws BoundsError fill_offsets!((buf,), g, 1, 0)

    # A zero-length dim whose *other* nonzero lengths would overflow the
    # product must not throw during construction: Q is exactly 0.
    huge = typemax(Int) ÷ 2
    g2 = AxisGroup((huge, huge, 0), ((1, 1, 1),))
    @test axis_length(g2) == 0

    # zero-length dim among nonzero lengths, in different positions
    for lens in ((0, 3, 4), (3, 0, 4), (3, 4, 0))
        gg = AxisGroup(lens, ((1, 1, 1),))
        @test axis_length(gg) == 0
    end
end

# =====================================================================
# Singleton dimensions in various positions
# =====================================================================

@testset "singleton dimensions before/between/after" begin
    # singleton first
    g1 = AxisGroup((1, 3, 2), ((1000, 1, 10),))
    Q1 = axis_length(g1)
    @test Q1 == 6
    for q in 0:(Q1 - 1)
        @test offsets(g1, q) == oracle_offset((1, 3, 2), ((1000, 1, 10),), q)
    end

    # singleton in middle
    g2 = AxisGroup((3, 1, 2), ((1, 1000, 10),))
    Q2 = axis_length(g2)
    @test Q2 == 6
    for q in 0:(Q2 - 1)
        @test offsets(g2, q) == oracle_offset((3, 1, 2), ((1, 1000, 10),), q)
    end

    # singleton last
    g3 = AxisGroup((3, 2, 1), ((1, 10, 1000),))
    Q3 = axis_length(g3)
    @test Q3 == 6
    for q in 0:(Q3 - 1)
        @test offsets(g3, q) == oracle_offset((3, 2, 1), ((1, 10, 1000),), q)
    end

    # all singleton (D>0, Q=1)
    g4 = AxisGroup((1, 1, 1), ((77, 88, 99),))
    @test axis_length(g4) == 1
    @test offsets(g4, 0) == (0,) # singleton strides irrelevant to the single coordinate offset
end

# =====================================================================
# Signed / zero strides, including mixed signs across maps
# =====================================================================

@testset "signed and zero strides, mixed across maps" begin
    g = AxisGroup((4, 3), ((2, -8), (0, 5), (-3, 0)))
    Q = axis_length(g)
    @test Q == 12
    bufs = ntuple(_ -> zeros(Int, Q), 3)
    fill_offsets!(bufs, g, 0, Q)
    expected = oracle_all_offsets((4, 3), ((2, -8), (0, 5), (-3, 0)))
    for p in 1:3
        @test [expected[i][p] for i in 1:Q] == bufs[p]
    end
    # map 2 has a zero stride on dim 1: within a run of 4 (dim-1 varying),
    # offsets must be constant.
    @test all(==(bufs[2][1]), bufs[2][1:4])
    d2 = describe_block(bufs[2][1:4], 4)
    @test d2.regular == true
    @test d2.stride == 0
end

# =====================================================================
# Nonzero interval starts and carries across multiple dimensions
# =====================================================================

@testset "nonzero starts and multi-dim carries" begin
    lengths = (2, 3, 2) # Q = 12
    strides = ((1, 2, 6),)
    g = AxisGroup(lengths, strides)
    Q = axis_length(g)
    @test Q == 12
    full = oracle_all_offsets(lengths, strides)
    for first in 0:Q, count in 0:(Q - first)
        buf = zeros(Int, max(count, 1))
        fill_offsets!((buf,), g, first, count)
        @test buf[1:count] == [full[i + 1][1] for i in first:(first + count - 1)]
    end
end

# =====================================================================
# Boundary-exact and empty intervals at 0 and Q
# =====================================================================

@testset "boundary-exact and empty intervals" begin
    g = AxisGroup((3, 2), ((1, 10),))
    Q = axis_length(g)
    @test Q == 6
    # empty interval at start
    buf = fill(-1, 6)
    fill_offsets!((buf,), g, 0, 0)
    @test buf == fill(-1, 6)
    # empty interval at Q (valid: empty may start anywhere in 0..Q inclusive)
    buf2 = fill(-1, 6)
    fill_offsets!((buf2,), g, Q, 0)
    @test buf2 == fill(-1, 6)
    # positive interval ending exactly at boundary
    buf3 = zeros(Int, 6)
    fill_offsets!((buf3,), g, 0, 6)
    @test buf3 == [0, 1, 2, 10, 11, 12]
    # positive interval cannot start at Q
    @test_throws BoundsError fill_offsets!((zeros(Int, 1),), g, Q, 1)
    # count can't exceed Q - first
    @test_throws BoundsError fill_offsets!((zeros(Int, 6),), g, 1, 6)
end

# =====================================================================
# Prefix-only writes with untouched suffix canaries
# =====================================================================

@testset "prefix-only writes, suffix canaries untouched" begin
    g = AxisGroup((4,), ((1,), (2,)))
    canary = -123456789
    bufA = fill(canary, 6)
    bufB = fill(canary, 6)
    fill_offsets!((bufA, bufB), g, 0, 3)
    @test bufA[1:3] == [0, 1, 2]
    @test bufB[1:3] == [0, 2, 4]
    @test bufA[4:6] == fill(canary, 3)
    @test bufB[4:6] == fill(canary, 3)

    # Empty interval writes nothing at all, whole buffer untouched.
    bufA2 = fill(canary, 4)
    bufB2 = fill(canary, 4)
    fill_offsets!((bufA2, bufB2), g, 2, 0)
    @test bufA2 == fill(canary, 4)
    @test bufB2 == fill(canary, 4)
end

# =====================================================================
# Per-map regularity differences and nonunit affine strides
# =====================================================================

@testset "per-map regularity differences" begin
    # map 1 regular unit stride (fold condition holds: stride2==L1*stride1==2),
    # map 2 regular nonunit stride (fold condition holds: stride2==L1*stride1==4),
    # map 3 irregular (dimension-boundary crossing with a mismatched stride).
    g = AxisGroup((2, 2), ((1, 2), (2, 4), (1, 100)))
    Q = axis_length(g)
    buf1, buf2, buf3 = zeros(Int, Q), zeros(Int, Q), zeros(Int, Q)
    d1, d2, d3 = block_descriptors!((buf1, buf2, buf3), g, 0, Q)
    @test buf1 == [0, 1, 2, 3]
    @test d1.regular == true && d1.stride == 1
    @test buf2 == [0, 2, 4, 6]
    @test d2.regular == true && d2.stride == 2
    # map 3 offsets: coords (0,0)->0,(1,0)->1,(0,1)->100,(1,1)->101: diffs 1,99,1 -> irregular
    @test buf3 == [0, 1, 100, 101]
    @test d3.regular == false
end

# =====================================================================
# Normalization: equivalence, joint-fold-refusal, negative-stride folding,
# all-singleton normalization
# =====================================================================

@testset "normalization: equivalence on foldable group" begin
    # Two dims that DO jointly fold: lengths (3,2), strides such that
    # next_stride == current_length * current_stride for every map.
    g = AxisGroup((3, 2), ((1, 3), (2, 6)))
    gn = normalize_group(g)
    @test axis_length(gn) == axis_length(g)
    Q = axis_length(g)
    buf1, buf2 = zeros(Int, Q), zeros(Int, Q)
    buf1n, buf2n = zeros(Int, Q), zeros(Int, Q)
    fill_offsets!((buf1, buf2), g, 0, Q)
    fill_offsets!((buf1n, buf2n), gn, 0, Q)
    @test buf1 == buf1n
    @test buf2 == buf2n
    # A fully-folded group here should reduce to rank 1 (both maps fold).
    @test length(gn.lengths) == 1
end

@testset "normalization: joint-fold refusal (per-map affine, not jointly foldable)" begin
    # G from the fixed fixture: map1 folds (1,3)->stride1*3=3 matches next
    # stride 1? lengths (3,2): fold condition is next_stride == L1*stride1.
    # map1: L1=3, stride1=1 -> need next_stride==3, and it is 3: map1 DOES
    # fold on its own. map2: L1=3, stride1=1 -> need next_stride==3, but is 10:
    # map2 does NOT fold. Joint folding requires ALL maps to satisfy it, so
    # the group as a whole must not fold.
    G = AxisGroup((3, 2), ((1, 3), (1, 10)))
    Gn = normalize_group(G)
    @test length(Gn.lengths) == 2
    @test axis_length(Gn) == axis_length(G)
end

@testset "normalization: negative-stride folding" begin
    # Negative strides that still satisfy the fold condition exactly.
    g = AxisGroup((4, 3), ((-1, -4),))
    gn = normalize_group(g)
    @test length(gn.lengths) == 1
    Q = axis_length(g)
    buf, bufn = zeros(Int, Q), zeros(Int, Q)
    fill_offsets!((buf,), g, 0, Q)
    fill_offsets!((bufn,), gn, 0, Q)
    @test buf == bufn
end

@testset "normalization: all-singleton collapses to rank zero" begin
    g = AxisGroup((1, 1, 1), ((5, 6, 7), (8, 9, 10)))
    gn = normalize_group(g)
    @test length(gn.lengths) == 0
    @test axis_length(gn) == 1
    # same map count preserved
    @test length(gn.strides) == 2
    @test offsets(gn, 0) == (0, 0)
end

@testset "normalization: removes singletons, preserves full offset sequence" begin
    g = AxisGroup((3, 1, 2), ((1, 999, 10), (2, -999, 20)))
    gn = normalize_group(g)
    Q = axis_length(g)
    @test axis_length(gn) == Q
    bufA, bufB = zeros(Int, Q), zeros(Int, Q)
    bufAn, bufBn = zeros(Int, Q), zeros(Int, Q)
    fill_offsets!((bufA, bufB), g, 0, Q)
    fill_offsets!((bufAn, bufBn), gn, 0, Q)
    @test bufA == bufAn
    @test bufB == bufBn
end

@testset "normalization runs correctly unnormalized as well" begin
    # Explicit reminder per spec section 7: tests must run both normalized
    # and unnormalized groups. Pick an arbitrary interval start not aligned
    # to any dimension boundary and check both agree with the oracle.
    lengths = (3, 2, 2)
    strides = ((1, 3, 6), (2, 6, 12))
    g = AxisGroup(lengths, strides)
    gn = normalize_group(g)
    full = oracle_all_offsets(lengths, strides)
    for first in 0:(axis_length(g) - 1), count in 0:(axis_length(g) - first)
        bufA, bufB = zeros(Int, max(count, 1)), zeros(Int, max(count, 1))
        bufAn, bufBn = zeros(Int, max(count, 1)), zeros(Int, max(count, 1))
        fill_offsets!((bufA, bufB), g, first, count)
        fill_offsets!((bufAn, bufBn), gn, first, count)
        expectedA = [full[i + 1][1] for i in first:(first + count - 1)]
        expectedB = [full[i + 1][2] for i in first:(first + count - 1)]
        @test bufA[1:count] == expectedA
        @test bufB[1:count] == expectedB
        @test bufAn[1:count] == expectedA
        @test bufBn[1:count] == expectedB
    end
end

# =====================================================================
# Constructor overflow and invalid arguments
# =====================================================================

@testset "constructor: invalid lengths / map count" begin
    @test_throws ArgumentError AxisGroup((-1,), ((1,),))
    @test_throws ArgumentError AxisGroup((3, -2), ((1, 1),))
    @test_throws ArgumentError AxisGroup((), ())        # P == 0 not allowed
    @test_throws ArgumentError AxisGroup((3,), ())       # P == 0 not allowed
end

@testset "constructor: overflow policy" begin
    # Q itself overflows: two huge nonzero lengths.
    huge = typemax(Int) ÷ 2 + 2
    @test_throws OverflowError AxisGroup((huge, huge), ((1, 1),))

    # Q representable, but sum((L[d]-1)*abs(S[p][d])) overflows.
    @test_throws OverflowError AxisGroup((typemax(Int),), ((2,),))

    # abs(typemin(Int)) must not silently wrap into a bogus small value.
    @test_throws OverflowError AxisGroup((2,), ((typemin(Int),),))

    # Singleton dims contribute zero even with an extreme stride: must NOT throw.
    g = AxisGroup((1,), ((typemin(Int),),))
    @test axis_length(g) == 1
    @test offsets(g, 0) == (0,)

    # Empty domain skips offset-range validation even with huge strides.
    g2 = AxisGroup((0,), ((typemax(Int),),))
    @test axis_length(g2) == 0

    # A layout right at the representable boundary must succeed.
    g3 = AxisGroup((2,), ((typemax(Int) ÷ 2,),))
    @test axis_length(g3) == 2
end

# =====================================================================
# offsets(): bounds errors
# =====================================================================

@testset "offsets: bounds errors" begin
    g = AxisGroup((3, 2), ((1, 10),))
    @test_throws BoundsError offsets(g, -1)
    @test_throws BoundsError offsets(g, 6)
    @test offsets(g, 0) == (0,)
    @test offsets(g, 5) == (12,)
end

# =====================================================================
# fill_offsets!: invalid-request buffer-unchanged guarantees
# =====================================================================

@testset "fill_offsets!: invalid requests leave buffers untouched" begin
    g = AxisGroup((3, 2), ((1, 10), (2, 20)))
    canary = -777
    Q = axis_length(g)

    # negative count -> ArgumentError, buffers unchanged
    bufA, bufB = fill(canary, Q), fill(canary, Q)
    @test_throws ArgumentError fill_offsets!((bufA, bufB), g, 0, -1)
    @test bufA == fill(canary, Q)
    @test bufB == fill(canary, Q)

    # out-of-domain interval -> BoundsError, buffers unchanged
    bufA2, bufB2 = fill(canary, Q), fill(canary, Q)
    @test_throws BoundsError fill_offsets!((bufA2, bufB2), g, Q, 1)
    @test bufA2 == fill(canary, Q)
    @test bufB2 == fill(canary, Q)

    bufA2b, bufB2b = fill(canary, Q), fill(canary, Q)
    @test_throws BoundsError fill_offsets!((bufA2b, bufB2b), g, -1, 1)
    @test bufA2b == fill(canary, Q)
    @test bufB2b == fill(canary, Q)

    bufA2c, bufB2c = fill(canary, Q), fill(canary, Q)
    @test_throws BoundsError fill_offsets!((bufA2c, bufB2c), g, 1, Q)
    @test bufA2c == fill(canary, Q)
    @test bufB2c == fill(canary, Q)

    # insufficient buffer length -> DimensionMismatch, buffers unchanged
    shortA, okB = fill(canary, Q - 1), fill(canary, Q)
    @test_throws DimensionMismatch fill_offsets!((shortA, okB), g, 0, Q)
    @test shortA == fill(canary, Q - 1)
    @test okB == fill(canary, Q)
end

@testset "fill_offsets!: repeated buffer objects rejected before mutation" begin
    g = AxisGroup((3, 2), ((1, 10), (2, 20)))
    Q = axis_length(g)
    canary = -321
    buf = fill(canary, Q)
    @test_throws ArgumentError fill_offsets!((buf, buf), g, 0, Q)
    @test buf == fill(canary, Q) # untouched
end

# =====================================================================
# BlockDescriptor: overflow and edge fixtures
# =====================================================================

@testset "describe_block: canonical cases" begin
    d0 = describe_block(Int[], 0)
    @test (d0.base, d0.stride, d0.count, d0.regular) == (0, 0, 0, true)

    d = describe_block([42], 1)
    @test (d.base, d.stride, d.count, d.regular) == (42, 0, 1, true)

    d2 = describe_block([5, 8, 11, 14], 4)
    @test (d2.base, d2.stride, d2.count, d2.regular) == (5, 3, 4, true)

    d3 = describe_block([0, 5, 1, 6], 4)
    @test d3.regular == false
    @test d3.base == 0
    @test d3.stride == 0
end

@testset "describe_block: invalid arguments" begin
    @test_throws ArgumentError describe_block([1, 2, 3], -1)
    @test_throws DimensionMismatch describe_block([1, 2], 3)
end

@testset "describe_block: 3-arg (first, count) matches 2-arg on the extracted slice" begin
    rng = Random.MersenneTwister(0xDE5C812E)

    # Explicit edge cases: regular/affine, irregular/scattered, empty, singleton,
    # and first > 0 including near the end of a larger buffer.
    cases = [
        (collect(0:2:20), 0, 6),        # regular, first = 0
        (collect(0:2:20), 3, 4),        # regular, first > 0
        ([0, 5, 1, 6, 9, 2, 100], 1, 4), # irregular slice
        ([0, 5, 1, 6, 9, 2, 100], 0, 0), # empty at start
        ([0, 5, 1, 6, 9, 2, 100], 7, 0), # empty at end (first == length(buffer))
        ([42], 0, 1),                    # singleton
        ([1, 2, 42, 3], 2, 1),           # singleton, first > 0
        (collect(1:10), 8, 2),           # first near the end
        (collect(1:10), 9, 1),           # first at the very last valid start
        (collect(1:10), 10, 0),          # first == length(buffer), empty
    ]
    for (buf, first, count) in cases
        d3 = describe_block(buf, first, count)
        d2 = describe_block(copy(buf[(first + 1):(first + count)]), count)
        @test (d3.base, d3.stride, d3.count, d3.regular) ==
            (d2.base, d2.stride, d2.count, d2.regular)
    end

    # Randomized coverage across regular/irregular buffers.
    for _trial in 1:100
        n = rand(rng, 1:20)
        buf = rand(rng, Bool) ? collect(1:n) .* rand(rng, 1:5) : rand(rng, -50:50, n) # regular or irregular
        first = rand(rng, 0:n)
        count = rand(rng, 0:(n - first))
        d3 = describe_block(buf, first, count)
        d2 = describe_block(copy(buf[(first + 1):(first + count)]), count)
        @test (d3.base, d3.stride, d3.count, d3.regular) ==
            (d2.base, d2.stride, d2.count, d2.regular)
    end
end

@testset "describe_block: 3-arg invalid arguments" begin
    @test_throws ArgumentError describe_block([1, 2, 3], -1, 1)
    @test_throws ArgumentError describe_block([1, 2, 3], 0, -1)
    @test_throws DimensionMismatch describe_block([1, 2, 3], 2, 2) # first+count=4 > length=3
    @test_throws DimensionMismatch describe_block([1, 2], 3, 0)    # first > length
end

@testset "describe_block: overflow fixtures" begin
    d = describe_block([typemin(Int), typemax(Int)], 2)
    @test d.regular == false
    @test d.base == typemin(Int)
    @test d.stride == 0
    @test d.count == 2

    d2 = describe_block([typemin(Int)], 1)
    @test d2.regular == true
    @test d2.base == typemin(Int)
    @test d2.stride == 0
    @test d2.count == 1
end

# =====================================================================
# Randomized property tests (fixed seed)
# =====================================================================

@testset "randomized properties" begin
    rng = Random.MersenneTwister(0xA51CE_1)

    for trial in 1:200
        D = rand(rng, 0:4)
        P = rand(rng, 1:3)
        lengths = ntuple(_ -> rand(rng, 0:4), D)
        strides = ntuple(_ -> ntuple(_ -> rand(rng, -8:8), D), P)

        # With these bounds (D<=4, lengths<=4, |strides|<=8, P<=3) the
        # conservative overflow bound cannot be violated, so no layout here
        # is expected to be rejected; the try/catch just documents that this
        # loop's envelope is deliberately conservative rather than relying on
        # never observing a thrown error.
        local g
        try
            g = AxisGroup(lengths, strides)
        catch e
            e isa Union{ArgumentError, OverflowError} || rethrow()
            continue
        end

        Q = axis_length(g)
        expected_Q = D == 0 ? 1 : (any(==(0), lengths) ? 0 : prod(lengths))
        @test Q == expected_Q

        oracle_full = oracle_all_offsets(lengths, strides)
        @test length(oracle_full) == Q

        # offsets() vs oracle, every valid q
        for q in 0:(Q - 1)
            @test offsets(g, q) == oracle_full[q + 1]
        end
        Q > 0 && @test_throws BoundsError offsets(g, Q)
        @test_throws BoundsError offsets(g, -1)

        # fill_offsets! vs oracle over several random sub-intervals, plus concat check
        for _ in 1:5
            first = Q == 0 ? 0 : rand(rng, 0:Q)
            maxcount = Q - first
            count = rand(rng, 0:maxcount)
            bufs = ntuple(_ -> zeros(Int, max(count, 1)), P)
            fill_offsets!(bufs, g, first, count)
            for p in 1:P
                expected = [oracle_full[i + 1][p] for i in first:(first + count - 1)]
                @test bufs[p][1:count] == expected
            end

            # interval concatenation == whole-interval generation
            if count >= 2
                mid = rand(rng, 1:(count - 1))
                bufs_left = ntuple(_ -> zeros(Int, mid), P)
                bufs_right = ntuple(_ -> zeros(Int, count - mid), P)
                fill_offsets!(bufs_left, g, first, mid)
                fill_offsets!(bufs_right, g, first + mid, count - mid)
                for p in 1:P
                    @test vcat(bufs_left[p], bufs_right[p]) == bufs[p][1:count]
                end
            end

            # regular descriptor reproduces exact sequence (overflow-safe ref
            # arithmetic); irregular classification checked independently.
            descs = ntuple(p -> describe_block(bufs[p], count), P)
            for p in 1:P
                d = descs[p]
                @test d.count == count
                is_affine, stride = independent_affine_check(bufs[p], count)
                @test d.regular == is_affine
                if is_affine && count >= 1
                    @test d.base == bufs[p][1]
                    for t in 0:(count - 1)
                        @test Int128(bufs[p][t + 1]) == ref_affine_value(d.base, d.stride, t)
                    end
                    if count >= 2
                        @test d.stride == stride
                    end
                end
            end
        end

        # normalization: preserves axis_length and every offset, for
        # arbitrary interval starts (not only aligned blocks)
        gn = normalize_group(g)
        @test axis_length(gn) == Q
        for _ in 1:3
            first = Q == 0 ? 0 : rand(rng, 0:Q)
            count = rand(rng, 0:(Q - first))
            bufs = ntuple(_ -> zeros(Int, max(count, 1)), P)
            bufsn = ntuple(_ -> zeros(Int, max(count, 1)), P)
            fill_offsets!(bufs, g, first, count)
            fill_offsets!(bufsn, gn, first, count)
            for p in 1:P
                @test bufs[p][1:count] == bufsn[p][1:count]
            end
        end
    end
end
