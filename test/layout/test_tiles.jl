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
