# Verifies AffineAxis/ScatterAxis addressing, axis_from_descriptor,
# SourceTile/DestinationTile (QSTile), and pack_a!/pack_b! against direct
# tensor indexing (expected values computed by hand, not via tile_load).

using Test
using QuasiStrided
using QuasiStrided: AxisGroup, BlockDescriptor, describe_block, KernelDescriptor,
    mr, nr, scalartype, packed_a_offset, packed_b_offset, packed_a_length, packed_b_length,
    axis_from_descriptor, axis_length, axis_offset, checked_axis_offset,
    QSTile, nrows, ncols, tile_offset, checked_tile_offset,
    tile_load, tile_store!, checked_tile_load, checked_tile_store!

# =====================================================================
# AffineAxis / ScatterAxis addressing
# =====================================================================

@testset "AffineAxis: addressing and validation" begin
    ax = AffineAxis(10, 3, 5) # 10, 13, 16, 19, 22
    @test axis_length(ax) == 5
    for t in 0:4
        @test axis_offset(ax, t) == 10 + t * 3
        @test checked_axis_offset(ax, t) == 10 + t * 3
    end
    @test_throws BoundsError checked_axis_offset(ax, -1)
    @test_throws BoundsError checked_axis_offset(ax, 5)

    # negative stride
    axneg = AffineAxis(20, -4, 4) # 20, 16, 12, 8
    @test [axis_offset(axneg, t) for t in 0:3] == [20, 16, 12, 8]

    # zero stride (broadcast axis)
    axzero = AffineAxis(7, 0, 3)
    @test [axis_offset(axzero, t) for t in 0:2] == [7, 7, 7]

    # empty axis
    axempty = AffineAxis(0, 1, 0)
    @test axis_length(axempty) == 0
    @test_throws BoundsError checked_axis_offset(axempty, 0)

    @test_throws ArgumentError AffineAxis(0, 1, -1)
end

@testset "ScatterAxis: addressing, borrowing, and validation" begin
    offs = [5, -3, 100, 0, 42]
    ax = ScatterAxis(offs, 5)
    @test axis_length(ax) == 5
    for t in 0:4
        @test axis_offset(ax, t) == offs[t + 1]
        @test checked_axis_offset(ax, t) == offs[t + 1]
    end
    @test_throws BoundsError checked_axis_offset(ax, 5)
    @test_throws BoundsError checked_axis_offset(ax, -1)

    # count may be a strict prefix of offsets (populated-prefix borrowing)
    prefixed = ScatterAxis(offs, 3)
    @test axis_length(prefixed) == 3
    @test [axis_offset(prefixed, t) for t in 0:2] == offs[1:3]

    # borrowed, not copied: mutating the source vector is visible through the axis
    mutable_offs = [1, 2, 3]
    axb = ScatterAxis(mutable_offs, 3)
    mutable_offs[2] = 999
    @test axis_offset(axb, 1) == 999

    @test_throws ArgumentError ScatterAxis(offs, -1)
    @test_throws DimensionMismatch ScatterAxis(offs, 6)

    # a view of a populated prefix works directly (no copy needed)
    buf = [11, 22, 33, 44, -1, -1]
    axview = ScatterAxis(view(buf, 1:4), 4)
    @test [axis_offset(axview, t) for t in 0:3] == [11, 22, 33, 44]
end

@testset "axis_from_descriptor: regular -> AffineAxis, irregular -> ScatterAxis" begin
    # Regular (affine) buffer.
    buf = [7, 10, 13, 16]
    d = describe_block(buf, 4)
    @test d.regular
    ax = axis_from_descriptor(d, buf)
    @test ax isa AffineAxis
    @test ax.base == 7 && ax.stride == 3 && axis_length(ax) == 4

    # Irregular buffer.
    bufi = [0, 5, 1, 6]
    di = describe_block(bufi, 4)
    @test !di.regular
    axi = axis_from_descriptor(di, bufi)
    @test axi isa ScatterAxis
    @test axis_length(axi) == 4
    @test [axis_offset(axi, t) for t in 0:3] == bufi

    # ScatterAxis references the populated prefix, not a copy: mutating the
    # buffer's populated positions is visible; suffix beyond count is not read.
    bufi2 = [0, 5, 1, 6, -999, -999]
    di2 = describe_block(bufi2, 4)
    axi2 = axis_from_descriptor(di2, bufi2)
    bufi2[1] = 42
    @test axis_offset(axi2, 0) == 42

    # Empty (count == 0) is regular per describe_block -> AffineAxis, empty.
    d0 = describe_block(Int[], 0)
    ax0 = axis_from_descriptor(d0, Int[])
    @test ax0 isa AffineAxis
    @test axis_length(ax0) == 0

    # Singleton is regular -> AffineAxis of length 1.
    d1 = describe_block([42], 1)
    ax1 = axis_from_descriptor(d1, [42])
    @test ax1 isa AffineAxis
    @test axis_length(ax1) == 1
    @test axis_offset(ax1, 0) == 42

    # Nonzero interval start feeding a BlockDescriptor via AxisGroup, to
    # confirm the whole Phase 1 -> Phase 2 handoff (not just describe_block
    # in isolation).
    g = AxisGroup((3, 2), ((1, 15),))
    buf3 = zeros(Int, 3)
    fillbuf = QuasiStrided.fill_offsets!((buf3,), g, 3, 3) # second "row": 15,16,17
    dreg = describe_block(buf3, 3)
    axreg = axis_from_descriptor(dreg, buf3)
    @test axreg isa AffineAxis
    @test axreg.base == 15 && axreg.stride == 1
end

@testset "axis_from_descriptor: 3-arg (descriptor, buffer, first)" begin
    # Regular slice with first > 0: axis_offset(ax, t) for 0 <= t < count
    # must equal buf[first+1+t].
    buf = [99, 99, 7, 10, 13, 16, -1, -1]
    first = 2
    count = 4
    d = describe_block(buf, first, count)
    @test d.regular
    ax = axis_from_descriptor(d, buf, first)
    @test ax isa AffineAxis
    for t in 0:(count - 1)
        @test axis_offset(ax, t) == buf[first + 1 + t]
    end

    # Irregular slice with first > 0.
    bufi = [-1, -1, 0, 5, 1, 6, -1]
    firsti = 2
    counti = 4
    di = describe_block(bufi, firsti, counti)
    @test !di.regular
    axi = axis_from_descriptor(di, bufi, firsti)
    @test axi isa ScatterAxis
    for t in 0:(counti - 1)
        @test axis_offset(axi, t) == bufi[firsti + 1 + t]
    end

    # ScatterAxis borrows the slice: mutating the buffer at the relevant
    # position is visible through the axis.
    bufi[firsti + 1] = 555
    @test axis_offset(axi, 0) == 555

    # first = 0 (3-arg) must agree with the 2-arg form.
    buf2 = [7, 10, 13, 16]
    d2 = describe_block(buf2, 4)
    ax2a = axis_from_descriptor(d2, buf2)
    ax2b = axis_from_descriptor(d2, buf2, 0)
    @test ax2a isa AffineAxis && ax2b isa AffineAxis
    for t in 0:3
        @test axis_offset(ax2a, t) == axis_offset(ax2b, t)
    end
end

# =====================================================================
# QSTile (SourceTile / DestinationTile) addressing
# =====================================================================

@testset "SourceTile/DestinationTile share one internal type" begin
    @test SourceTile === DestinationTile
    @test SourceTile === QSTile
end

@testset "QSTile addressing: base + row_offset(i) + col_offset(j), both axis kinds" begin
    storage = collect(1.0:100.0)

    # affine rows, affine cols
    rows = AffineAxis(2, 3, 4)   # 2,5,8,11
    cols = AffineAxis(0, 10, 3) # 0,10,20
    t = SourceTile(storage, 100, rows, cols)
    @test nrows(t) == 4 && ncols(t) == 3
    for i in 0:3, j in 0:2
        expected = 100 + (2 + i * 3) + (0 + j * 10)
        @test tile_offset(t, i, j) == expected
        @test checked_tile_offset(t, i, j) == expected
    end
    @test_throws BoundsError checked_tile_offset(t, 4, 0)
    @test_throws BoundsError checked_tile_offset(t, 0, 3)
    @test_throws BoundsError checked_tile_offset(t, -1, 0)

    # scatter rows, affine cols (mixed): column addressing independent of row.
    rowoffs = [0, 100, 5, -20]
    rows2 = ScatterAxis(rowoffs, 4)
    cols2 = AffineAxis(1, 2, 3) # 1,3,5
    t2 = DestinationTile(storage, 0, rows2, cols2)
    for i in 0:3, j in 0:2
        expected = 0 + rowoffs[i + 1] + (1 + j * 2)
        @test tile_offset(t2, i, j) == expected
    end

    # affine rows, scatter cols
    coloffs = [3, -7, 42]
    rows3 = AffineAxis(5, 1, 2)
    cols3 = ScatterAxis(coloffs, 3)
    t3 = SourceTile(storage, -1, rows3, cols3)
    for i in 0:1, j in 0:2
        expected = -1 + (5 + i) + coloffs[j + 1]
        @test tile_offset(t3, i, j) == expected
    end

    # scatter rows, scatter cols
    rowoffs4 = [0, 4, 9]
    coloffs4 = [1, -1]
    rows4 = ScatterAxis(rowoffs4, 3)
    cols4 = ScatterAxis(coloffs4, 2)
    t4 = SourceTile(storage, 50, rows4, cols4)
    for i in 0:2, j in 0:1
        expected = 50 + rowoffs4[i + 1] + coloffs4[j + 1]
        @test tile_offset(t4, i, j) == expected
    end
end

@testset "QSTile load/store round-trip against hand-computed addresses" begin
    storage = zeros(Float64, 20)
    rows = AffineAxis(0, 2, 3)  # 0,2,4
    cols = AffineAxis(1, 3, 2)  # 1,4
    t = SourceTile(storage, 5, rows, cols)
    # base + row + col: (5,7,9)+(1,4) -> addresses 6,9,10,13,... let's just check each.
    for i in 0:2, j in 0:1
        v = 1000.0 * i + j
        tile_store!(t, i, j, v)
    end
    for i in 0:2, j in 0:1
        addr = 5 + (0 + i * 2) + (1 + j * 3)
        @test storage[addr + 1] == 1000.0 * i + j
        @test tile_load(t, i, j) == 1000.0 * i + j
        @test checked_tile_load(t, i, j) == 1000.0 * i + j
    end
    checked_tile_store!(t, 0, 0, -1.0)
    @test tile_load(t, 0, 0) == -1.0
    @test_throws BoundsError checked_tile_load(t, 3, 0)
    @test_throws BoundsError checked_tile_store!(t, 0, 2, 0.0)
end

# =====================================================================
# pack_a! / pack_b!: direct-indexing oracle helpers
# =====================================================================

# Build a SourceTile over a dense Vector `storage`, given per-axis affine or
# scatter descriptions as plain (base,stride) or explicit offset vectors, and
# an independent "reference" matrix computed straight from `storage` and the
# offsets, without going through tile_load, for the value-comparison tests.

# A[i,p] reference (0-based i,p), reading storage directly at
# base + rowoff(i) + coloff(p) + 1.
function _ref_matrix(storage::Vector{T}, base::Int, rowoffs::Vector{Int}, coloffs::Vector{Int}) where {T}
    m = length(rowoffs)
    k = length(coloffs)
    M = Matrix{T}(undef, m, k)
    for i in 1:m, p in 1:k
        M[i, p] = storage[base + rowoffs[i] + coloffs[p] + 1]
    end
    return M
end

_axis(base::Int, stride::Int, count::Int) = AffineAxis(base, stride, count)
_axis(offs::Vector{Int}) = ScatterAxis(offs, length(offs))

@testset "pack_a!: identity transform against direct indexing, several stride kinds" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)

    # Distinguishable deterministic values: storage[q] = q (1-based).
    storage = collect(1.0:1000.0)

    cases = [
        ("unit stride, full m", 10, AffineAxis(0, 1, 4), AffineAxis(0, 4, 5)),   # m=4=MR, kc=5
        ("nonunit stride", 3, AffineAxis(2, 7, 4), AffineAxis(1, 11, 3)),
        ("negative stride", 500, AffineAxis(0, -1, 4), AffineAxis(0, 2, 4)),
        ("zero stride (broadcast row)", 0, AffineAxis(20, 0, 4), AffineAxis(0, 1, 6)),
        ("partial m (m=2 < MR)", 0, AffineAxis(0, 3, 2), AffineAxis(0, 5, 4)),
        ("empty m (m=0)", 0, AffineAxis(0, 3, 0), AffineAxis(0, 5, 4)),
        ("scattered rows", 10, ScatterAxis([5, -3, 100, 7], 4), AffineAxis(0, 1, 3)),
        ("scattered cols (K)", 60, AffineAxis(0, 1, 3), ScatterAxis([2, 400, -50], 3)),
        ("scattered rows and cols", 10, ScatterAxis([1, 2, 3], 3), ScatterAxis([10, -10], 2)),
        ("empty kc", 0, AffineAxis(0, 1, 4), AffineAxis(0, 1, 0)),
    ]

    for (name, base, rows, cols) in cases
        m = axis_length(rows)
        kc = axis_length(cols)
        m <= MR || continue
        source = SourceTile(storage, base, rows, cols)
        packed = fill(-999.0, packed_a_length(kernel, kc) + 8) # +8 suffix canary region
        suffix_before = copy(packed[(packed_a_length(kernel, kc) + 1):end])

        pack_a!(packed, source, kernel, identity)

        for p in 0:(kc - 1), i in 0:(MR - 1)
            off = packed_a_offset(kernel, i, p)
            if i < m
                rowoff = rows isa AffineAxis ? rows.base + i * rows.stride : rows.offsets[i + 1]
                coloff = cols isa AffineAxis ? cols.base + p * cols.stride : cols.offsets[p + 1]
                expected = storage[base + rowoff + coloff + 1]
                @test packed[off + 1] == expected
            else
                @test packed[off + 1] == 0.0
            end
        end
        @test packed[(packed_a_length(kernel, kc) + 1):end] == suffix_before
    end
end

@testset "pack_b!: identity transform against direct indexing, several stride kinds" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)
    storage = collect(1.0:1000.0)

    cases = [
        ("unit stride, full n", 10, AffineAxis(0, 4, 5), AffineAxis(0, 1, 3)),   # kc=5, n=3=NR
        ("nonunit stride", 3, AffineAxis(1, 11, 4), AffineAxis(2, 7, 3)),
        ("negative stride", 500, AffineAxis(0, 2, 4), AffineAxis(0, -1, 3)),
        ("zero stride (broadcast col)", 0, AffineAxis(0, 1, 4), AffineAxis(20, 0, 2)),
        ("partial n (n=1 < NR)", 0, AffineAxis(0, 5, 4), AffineAxis(0, 3, 1)),
        ("empty n (n=0)", 0, AffineAxis(0, 5, 4), AffineAxis(0, 3, 0)),
        ("scattered cols", 10, AffineAxis(0, 1, 3), ScatterAxis([5, -3, 100], 3)),
        ("scattered rows (K)", 60, ScatterAxis([2, 400, -50], 3), AffineAxis(0, 1, 3)),
        ("scattered rows and cols", 10, ScatterAxis([1, 2, 3], 3), ScatterAxis([10, -10], 2)),
        ("empty kc", 0, AffineAxis(0, 1, 0), AffineAxis(0, 1, 3)),
    ]

    for (name, base, rows, cols) in cases
        kc = axis_length(rows)
        n = axis_length(cols)
        n <= NR || continue
        source = SourceTile(storage, base, rows, cols)
        packed = fill(-999.0, packed_b_length(kernel, kc) + 8)
        suffix_before = copy(packed[(packed_b_length(kernel, kc) + 1):end])

        pack_b!(packed, source, kernel, identity)

        for p in 0:(kc - 1), j in 0:(NR - 1)
            off = packed_b_offset(kernel, j, p)
            if j < n
                rowoff = rows isa AffineAxis ? rows.base + p * rows.stride : rows.offsets[p + 1]
                coloff = cols isa AffineAxis ? cols.base + j * cols.stride : cols.offsets[j + 1]
                expected = storage[base + rowoff + coloff + 1]
                @test packed[off + 1] == expected
            else
                @test packed[off + 1] == 0.0
            end
        end
        @test packed[(packed_b_length(kernel, kc) + 1):end] == suffix_before
    end
end

# =====================================================================
# A/B physical orientation, distinguishable deterministic values
# =====================================================================

@testset "pack_a!/pack_b!: physical orientation (A vs B layouts differ)" begin
    kernel = KernelDescriptor(Val(3), Val(2), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 4

    # A: value = 100*i + p (i = row 0..MR-1, p = K step). Distinguishable per
    # (i,p) so a transposition or offset-formula bug would show up as a
    # mismatch rather than an accidental match.
    storageA = zeros(Float64, MR * kc)
    for i in 0:(MR - 1), p in 0:(kc - 1)
        storageA[i * kc + p + 1] = 100.0 * i + p # row-major (i,p) source layout
    end
    rowsA = AffineAxis(0, kc, MR) # row i at base i*kc
    colsA = AffineAxis(0, 1, kc)  # col p at base p
    sourceA = SourceTile(storageA, 0, rowsA, colsA)
    packedA = zeros(Float64, packed_a_length(kernel, kc))
    pack_a!(packedA, sourceA, kernel, identity)
    for i in 0:(MR - 1), p in 0:(kc - 1)
        @test packedA[packed_a_offset(kernel, i, p) + 1] == 100.0 * i + p
    end
    # Physical layout check: packed_a_offset(i,p) = i + MR*p, so consecutive p
    # (fixed i) are MR apart, not 1 apart -- confirm directly.
    @test packed_a_offset(kernel, 1, 0) + MR == packed_a_offset(kernel, 1, 1)
    @test packed_a_offset(kernel, 0, 1) - packed_a_offset(kernel, 1, 1) == -1

    # B: value = 100*p + j (p = K step, j = column 0..NR-1).
    storageB = zeros(Float64, kc * NR)
    for p in 0:(kc - 1), j in 0:(NR - 1)
        storageB[p * NR + j + 1] = 100.0 * p + j
    end
    rowsB = AffineAxis(0, NR, kc) # row p at base p*NR
    colsB = AffineAxis(0, 1, NR)  # col j at base j
    sourceB = SourceTile(storageB, 0, rowsB, colsB)
    packedB = zeros(Float64, packed_b_length(kernel, kc))
    pack_b!(packedB, sourceB, kernel, identity)
    for p in 0:(kc - 1), j in 0:(NR - 1)
        @test packedB[packed_b_offset(kernel, j, p) + 1] == 100.0 * p + j
    end
    @test packed_b_offset(kernel, 1, 0) + NR == packed_b_offset(kernel, 1, 1)
end

# =====================================================================
# Padding, uninitialized-buffer detection, and canaries
# =====================================================================

@testset "pack_a!/pack_b!: padding writes literal zero, bypassing transform" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 3
    storage = fill(1.0, 100) # nonzero everywhere, so a padding bug would show a nonzero read

    transform_nonzero_at_zero = x -> x + 1000.0 # transform(0) = 1000 != 0

    # A: partial m so some rows are padding.
    m = 2
    rows = AffineAxis(0, 1, m)
    cols = AffineAxis(0, 1, kc)
    source = SourceTile(storage, 0, rows, cols)
    packed = fill(-777.0, packed_a_length(kernel, kc)) # nonzero prefill to detect uninitialized padding
    pack_a!(packed, source, kernel, transform_nonzero_at_zero)
    for p in 0:(kc - 1), i in 0:(MR - 1)
        off = packed_a_offset(kernel, i, p)
        if i < m
            @test packed[off + 1] == 1.0 + 1000.0
        else
            @test packed[off + 1] == 0.0 # literal zero, not transform(0) == 1000
        end
    end

    # B: partial n so some columns are padding.
    n = 1
    rowsB = AffineAxis(0, 1, kc)
    colsB = AffineAxis(0, 1, n)
    sourceB = SourceTile(storage, 0, rowsB, colsB)
    packedB = fill(-777.0, packed_b_length(kernel, kc))
    pack_b!(packedB, sourceB, kernel, transform_nonzero_at_zero)
    for p in 0:(kc - 1), j in 0:(NR - 1)
        off = packed_b_offset(kernel, j, p)
        if j < n
            @test packedB[off + 1] == 1.0 + 1000.0
        else
            @test packedB[off + 1] == 0.0
        end
    end
end

@testset "pack_a!/pack_b!: transform never called on padding lanes (call-counting proof)" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 3
    storage = fill(2.0, 50)

    calls = Ref(0)
    counting_transform = x -> (calls[] += 1; x)

    m = 2
    source = SourceTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, 1, kc))
    packed = zeros(Float64, packed_a_length(kernel, kc))
    pack_a!(packed, source, kernel, counting_transform)
    @test calls[] == m * kc # exactly the valid lanes, never the MR-m padding rows

    calls[] = 0
    n = 1
    sourceB = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, 1, n))
    packedB = zeros(Float64, packed_b_length(kernel, kc))
    pack_b!(packedB, sourceB, kernel, counting_transform)
    @test calls[] == n * kc
end

@testset "pack_a!/pack_b!: kc=0 reads nothing, writes nothing" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)

    read_flag = Ref(false)
    storage = fill(3.0, 10)
    # A tile whose storage access, if ever performed, would set read_flag via
    # a custom getindex wrapper is overkill; instead rely on kc==0 meaning
    # the packed buffer's declared physical length is 0, so any nonzero
    # write would be detectable via a nonempty canary-filled buffer, and any
    # read would require axis_length(cols) > 0 (contradiction with kc=0).
    canary = fill(-42.0, 8)
    packed = copy(canary)
    source = SourceTile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 1, 0)) # kc=0
    result = pack_a!(packed, source, kernel, x -> (read_flag[] = true; x))
    @test result === packed
    @test packed == canary # untouched
    @test read_flag[] == false

    packedB = copy(canary)
    sourceB = SourceTile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 1, 3)) # kc=0
    resultB = pack_b!(packedB, sourceB, kernel, x -> (read_flag[] = true; x))
    @test resultB === packedB
    @test packedB == canary
    @test read_flag[] == false
end

@testset "pack_a!/pack_b!: buffer reuse detects uninitialized padding (nonzero prefill)" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 2
    storage = fill(5.0, 20)

    m = 1
    source = SourceTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, 1, kc))
    packed = fill(123456.0, packed_a_length(kernel, kc)) # previously "used" nonzero buffer
    pack_a!(packed, source, kernel, identity)
    for p in 0:(kc - 1), i in 0:(MR - 1)
        off = packed_a_offset(kernel, i, p)
        @test packed[off + 1] == (i < m ? 5.0 : 0.0)
    end

    n = 1
    sourceB = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, 1, n))
    packedB = fill(654321.0, packed_b_length(kernel, kc))
    pack_b!(packedB, sourceB, kernel, identity)
    for p in 0:(kc - 1), j in 0:(NR - 1)
        off = packed_b_offset(kernel, j, p)
        @test packedB[off + 1] == (j < n ? 5.0 : 0.0)
    end
end

# =====================================================================
# Invalid metadata: rejected before any buffer mutation
# =====================================================================

@testset "pack_a!/pack_b!: invalid metadata rejected before mutation" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    storage = fill(9.0, 20)
    canary = fill(-1.0, 100)

    # A: m > MR
    source_bad_m = SourceTile(storage, 0, AffineAxis(0, 1, 5), AffineAxis(0, 1, 3)) # m=5 > MR=4
    packed = copy(canary)
    @test_throws ArgumentError pack_a!(packed, source_bad_m, kernel, identity)
    @test packed == canary

    # A: packed buffer too small
    source_ok = SourceTile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 1, 3)) # kc=3 needs 12
    packed_small = fill(-1.0, 11)
    @test_throws DimensionMismatch pack_a!(packed_small, source_ok, kernel, identity)
    @test packed_small == fill(-1.0, 11)

    # A: eltype mismatch
    kernel32 = KernelDescriptor(Val(4), Val(3), Float32)
    packed64 = copy(canary)
    @test_throws ArgumentError pack_a!(packed64, source_ok, kernel32, identity)
    @test packed64 == canary

    # B: n > NR
    source_bad_n = SourceTile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 1, 4)) # n=4 > NR=3
    packedB = copy(canary)
    @test_throws ArgumentError pack_b!(packedB, source_bad_n, kernel, identity)
    @test packedB == canary

    # B: packed buffer too small
    source_okB = SourceTile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 1, 3)) # kc=3 needs 9
    packedB_small = fill(-1.0, 8)
    @test_throws DimensionMismatch pack_b!(packedB_small, source_okB, kernel, identity)
    @test packedB_small == fill(-1.0, 8)

    # B: eltype mismatch
    packedB64 = copy(canary)
    @test_throws ArgumentError pack_b!(packedB64, source_okB, kernel32, identity)
    @test packedB64 == canary
end

# =====================================================================
# Nonzero interval starts: pack directly from a nonzero tile base
# =====================================================================

@testset "pack_a!/pack_b!: nonzero interval start (nonzero tile base)" begin
    kernel = KernelDescriptor(Val(3), Val(2), Float64)
    MR, NR = mr(kernel), nr(kernel)
    storage = collect(1.0:200.0)
    kc = 4

    base = 37
    rows = AffineAxis(0, 5, MR)
    cols = AffineAxis(0, 1, kc)
    source = SourceTile(storage, base, rows, cols)
    packed = zeros(Float64, packed_a_length(kernel, kc))
    pack_a!(packed, source, kernel, identity)
    for p in 0:(kc - 1), i in 0:(MR - 1)
        expected = storage[base + i * 5 + p + 1]
        @test packed[packed_a_offset(kernel, i, p) + 1] == expected
    end
end

# =====================================================================
# Non-identity transform (a real elementwise transform, not just identity)
# =====================================================================

@testset "pack_a!/pack_b!: nontrivial elementwise transform" begin
    kernel = KernelDescriptor(Val(3), Val(2), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 3
    storage = collect(1.0:100.0)

    negate = x -> -x

    source = SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, 1, kc))
    packed = zeros(Float64, packed_a_length(kernel, kc))
    pack_a!(packed, source, kernel, negate)
    for p in 0:(kc - 1), i in 0:(MR - 1)
        @test packed[packed_a_offset(kernel, i, p) + 1] == -storage[i + p + 1]
    end

    sourceB = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, 1, NR))
    packedB = zeros(Float64, packed_b_length(kernel, kc))
    pack_b!(packedB, sourceB, kernel, negate)
    for p in 0:(kc - 1), j in 0:(NR - 1)
        @test packedB[packed_b_offset(kernel, j, p) + 1] == -storage[p + j + 1]
    end
end

# =====================================================================
# Float32 support (the other required scalar type)
# =====================================================================

@testset "pack_a!/pack_b!: Float32" begin
    kernel = KernelDescriptor(Val(4), Val(2), Float32)
    MR, NR = mr(kernel), nr(kernel)
    kc = 3
    storage = Float32.(collect(1.0:100.0))

    source = SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, 1, kc))
    packed = zeros(Float32, packed_a_length(kernel, kc))
    pack_a!(packed, source, kernel, identity)
    @test eltype(packed) === Float32
    for p in 0:(kc - 1), i in 0:(MR - 1)
        @test packed[packed_a_offset(kernel, i, p) + 1] == storage[i + p + 1]
    end
end

# =====================================================================
# Steady-state allocation (Phase 2b Fable review finding 5, deferred;
# root-caused and fixed by whoever next reads this): pack_a!/pack_b!
# called directly against a bare KernelDescriptor must allocate zero
# bytes once warmed, for both affine and scattered sources, kc=0, and a
# nontrivial (non-identity) transform. Root cause was that `transform`
# (and `_pack_panel!`'s `transform`/`load`/`packed_offset`) had no type
# parameter in the signature: a `Function`-typed argument that a method
# only *forwards* (never calls directly) gets compiled against a
# widened/abstract type unless bound by an explicit `where` clause, which
# forced a dynamic call and heap-allocated the `load` closure passed
# alongside it. Fixed by giving each of those parameters its own free
# type parameter (`transform::F where {F}`, etc.) so the compiler is
# forced to specialize per concrete callable type.
#
# NOTE: this test only covers pack_a!/pack_b! called directly against a
# `KernelDescriptor`, matching this file's ownership scope. Calling
# through the `ScalarKernel`/`SIMDKernel` forwarding one-liners in
# src/kernel.jl / src/kernels/simd.jl still allocates (confirmed
# separately): those forwarding methods declare `kernel::ScalarKernel`
# (resp. `SIMDKernel`) and `transform` with no `where` clause of their
# own, so the same widening happens one layer up, in files this task
# does not own and must not edit. That is a distinct, currently
# unresolved allocation and is intentionally not asserted here.
@testset "pack_a!/pack_b!: zero steady-state allocation (direct KernelDescriptor)" begin
    nontrivial(x) = 2.0 * x + 1.0

    function run()
        kernel = KernelDescriptor(Val(8), Val(6), Float64)
        packed_a = zeros(8 * 4)
        packed_b = zeros(6 * 4)

        src_affine = SourceTile(rand(1000), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 4))
        src_b_affine = SourceTile(rand(1000), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 6))

        offs_rows = collect(0:7)
        offs_cols = [0, 8, 16, 24]
        src_scatter = SourceTile(rand(1000), 0, ScatterAxis(offs_rows, 8), ScatterAxis(offs_cols, 4))

        offs_rows_b = collect(0:3)
        offs_cols_b = [0, 4, 8, 12, 16, 20]
        src_b_scatter = SourceTile(rand(1000), 0, ScatterAxis(offs_rows_b, 4), ScatterAxis(offs_cols_b, 6))

        src_kc0 = SourceTile(rand(10), 0, AffineAxis(0, 1, 8), AffineAxis(0, 8, 0))
        src_b_kc0 = SourceTile(rand(10), 0, AffineAxis(0, 1, 0), AffineAxis(0, 0, 6))

        bytes = Int[]

        pack_a!(packed_a, src_affine, kernel, identity)
        push!(bytes, @allocated pack_a!(packed_a, src_affine, kernel, identity))

        pack_a!(packed_a, src_affine, kernel, nontrivial)
        push!(bytes, @allocated pack_a!(packed_a, src_affine, kernel, nontrivial))

        pack_b!(packed_b, src_b_affine, kernel, identity)
        push!(bytes, @allocated pack_b!(packed_b, src_b_affine, kernel, identity))

        pack_b!(packed_b, src_b_affine, kernel, nontrivial)
        push!(bytes, @allocated pack_b!(packed_b, src_b_affine, kernel, nontrivial))

        pack_a!(packed_a, src_scatter, kernel, identity)
        push!(bytes, @allocated pack_a!(packed_a, src_scatter, kernel, identity))

        pack_a!(packed_a, src_scatter, kernel, nontrivial)
        push!(bytes, @allocated pack_a!(packed_a, src_scatter, kernel, nontrivial))

        pack_b!(packed_b, src_b_scatter, kernel, identity)
        push!(bytes, @allocated pack_b!(packed_b, src_b_scatter, kernel, identity))

        pack_b!(packed_b, src_b_scatter, kernel, nontrivial)
        push!(bytes, @allocated pack_b!(packed_b, src_b_scatter, kernel, nontrivial))

        pack_a!(packed_a, src_kc0, kernel, identity)
        push!(bytes, @allocated pack_a!(packed_a, src_kc0, kernel, identity))

        pack_b!(packed_b, src_b_kc0, kernel, identity)
        push!(bytes, @allocated pack_b!(packed_b, src_b_kc0, kernel, identity))

        return bytes
    end

    @test all(iszero, run())
end

# Main-process follow-up: the diagnosis above fixed pack_a!/pack_b! called
# with a bare KernelDescriptor, but the identical missing-`where`-clause bug
# recurred one layer up in ScalarKernel's and SIMDKernel's own pack_a!/
# pack_b! forwarding methods (src/kernel.jl, src/kernels/simd.jl) — fixed
# there too (same pattern: bind the kernel's type parameters and give
# `transform` its own free type parameter). Regression-test both forwarding
# paths, not just the direct-KernelDescriptor path above.
@testset "pack_a!/pack_b!: zero steady-state allocation (ScalarKernel/SIMDKernel forwarding)" begin
    function run_forwarding(kernel)
        packed_a = zeros(scalartype(kernel), mr(kernel) * 4)
        packed_b = zeros(scalartype(kernel), nr(kernel) * 4)
        src_a = SourceTile(rand(scalartype(kernel), 1000), 0, AffineAxis(0, 1, mr(kernel)), AffineAxis(0, mr(kernel), 4))
        src_b = SourceTile(rand(scalartype(kernel), 1000), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, nr(kernel)))

        pack_a!(packed_a, src_a, kernel, identity)
        a1 = @allocated pack_a!(packed_a, src_a, kernel, identity)
        pack_b!(packed_b, src_b, kernel, identity)
        b1 = @allocated pack_b!(packed_b, src_b, kernel, identity)
        return (a1, b1)
    end

    @test run_forwarding(ScalarKernel(Val(8), Val(6), Float64)) == (0, 0)
    @test run_forwarding(SIMDKernel(Val(8), Val(6), Float64)) == (0, 0)
end

# =====================================================================
# Macro-blocking milestone: pack_a!/pack_b! widened to AbstractVector,
# packing into a panel-sliver view of a larger buffer.
# =====================================================================

@testset "pack_a!/pack_b!: packing into a SubArray view matches a fresh Vector" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 5
    storage = collect(1.0:1000.0)

    sourceA = SourceTile(storage, 10, AffineAxis(0, 1, MR), AffineAxis(0, 4, kc))
    neededA = packed_a_length(kernel, kc)
    freshA = zeros(Float64, neededA)
    pack_a!(freshA, sourceA, kernel, identity)

    bigA = zeros(Float64, neededA + 20)
    r = 6:(6 + neededA - 1)
    viewA = view(bigA, r)
    @test viewA isa SubArray
    pack_a!(viewA, sourceA, kernel, identity)
    @test viewA == freshA
    @test bigA[r] == freshA

    sourceB = SourceTile(storage, 20, AffineAxis(0, 1, kc), AffineAxis(0, kc, NR))
    neededB = packed_b_length(kernel, kc)
    freshB = zeros(Float64, neededB)
    pack_b!(freshB, sourceB, kernel, identity)

    bigB = zeros(Float64, neededB + 20)
    rB = 4:(4 + neededB - 1)
    viewB = view(bigB, rB)
    @test viewB isa SubArray
    pack_b!(viewB, sourceB, kernel, identity)
    @test viewB == freshB
    @test bigB[rB] == freshB
end

@testset "pack_a!/pack_b!: canary bytes outside a middle panel view untouched" begin
    kernel = KernelDescriptor(Val(4), Val(3), Float64)
    MR, NR = mr(kernel), nr(kernel)
    kc = 5
    storage = collect(1.0:1000.0)
    sentinel = -123456.0

    sourceA = SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, 4, kc))
    neededA = packed_a_length(kernel, kc)

    bigA = fill(sentinel, neededA + 30)
    lo, hi = 9, 9 + neededA - 1
    r = lo:hi
    pack_a!(view(bigA, r), sourceA, kernel, identity)
    @test all(==(sentinel), bigA[1:(lo - 1)])
    @test all(==(sentinel), bigA[(hi + 1):end])
    @test !any(==(sentinel), bigA[r]) # packed region actually got written

    sourceB = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, kc, NR))
    neededB = packed_b_length(kernel, kc)
    bigB = fill(sentinel, neededB + 30)
    loB, hiB = 12, 12 + neededB - 1
    rB = loB:hiB
    pack_b!(view(bigB, rB), sourceB, kernel, identity)
    @test all(==(sentinel), bigB[1:(loB - 1)])
    @test all(==(sentinel), bigB[(hiB + 1):end])
    @test !any(==(sentinel), bigB[rB])
end

@testset "pack_a!/pack_b!: zero steady-state allocation packing into a SubArray view" begin
    function run_view(kernel_ctor)
        kernel = kernel_ctor(Val(8), Val(6), Float64)
        bigA = zeros(mr(kernel) * 4 + 40)
        bigB = zeros(nr(kernel) * 4 + 40)
        viewA = view(bigA, 5:(5 + mr(kernel) * 4 - 1))
        viewB = view(bigB, 3:(3 + nr(kernel) * 4 - 1))

        src_a_affine = SourceTile(rand(1000), 0, AffineAxis(0, 1, mr(kernel)), AffineAxis(0, mr(kernel), 4))
        src_b_affine = SourceTile(rand(1000), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, nr(kernel)))

        offs_rows = collect(0:(mr(kernel) - 1))
        offs_cols = [0, 8, 16, 24]
        src_a_scatter = SourceTile(rand(1000), 0, ScatterAxis(offs_rows, mr(kernel)), ScatterAxis(offs_cols, 4))

        offs_rows_b = collect(0:3)
        offs_cols_b = collect(0:(nr(kernel) - 1)) .* 4
        src_b_scatter = SourceTile(rand(1000), 0, ScatterAxis(offs_rows_b, 4), ScatterAxis(offs_cols_b, nr(kernel)))

        pack_a!(viewA, src_a_affine, kernel, identity)
        a_affine = @allocated pack_a!(viewA, src_a_affine, kernel, identity)
        pack_b!(viewB, src_b_affine, kernel, identity)
        b_affine = @allocated pack_b!(viewB, src_b_affine, kernel, identity)

        pack_a!(viewA, src_a_scatter, kernel, identity)
        a_scatter = @allocated pack_a!(viewA, src_a_scatter, kernel, identity)
        pack_b!(viewB, src_b_scatter, kernel, identity)
        b_scatter = @allocated pack_b!(viewB, src_b_scatter, kernel, identity)

        return (a_affine, b_affine, a_scatter, b_scatter)
    end

    # Direct KernelDescriptor.
    @test run_view(KernelDescriptor) == (0, 0, 0, 0)

    # Via ScalarKernel/SIMDKernel forwarding (the exact site of the Phase 2b
    # finding-5 recurrence, now widened to AbstractVector).
    @test run_view(ScalarKernel) == (0, 0, 0, 0)
    @test run_view(SIMDKernel) == (0, 0, 0, 0)
end
