# Integration gate: exercises AxisGroup -> BlockDescriptor ->
# axis_from_descriptor -> QSTile -> pack_a!/pack_b! -> ScalarKernel
# execute_tile! end to end, including multiple K panels applying beta once.

@testset "manual pipeline: AxisGroup -> tiles -> packing -> ScalarKernel" begin
    # A[a,k,b] shape (3,5,2), B[k,n] shape (5,4), C[a,n,b] shape (3,4,2),
    # column-major, C[a,n,b] = sum_k A[a,k,b]*B[k,n].
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    B = reshape(collect(1.0:20.0), 5, 4)
    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, a in 1:3
        Cref[a, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end

    M = AxisGroup((3, 2), ((1, 15), (1, 12)))  # A, C
    N = AxisGroup((4,), ((5,), (3,)))          # B, C
    K = AxisGroup((5,), ((3,), (1,)))          # A, B

    @test axis_length(M) == 6
    @test axis_length(N) == 4
    @test axis_length(K) == 5

    kernel = ScalarKernel(Val(8), Val(6), Float64)  # MR=8 >= 6, NR=6 >= 4: one output tile

    m_buf_A = zeros(Int, axis_length(M))
    m_buf_C = zeros(Int, axis_length(M))
    n_buf_B = zeros(Int, axis_length(N))
    n_buf_C = zeros(Int, axis_length(N))
    k_buf_A = zeros(Int, axis_length(K))
    k_buf_B = zeros(Int, axis_length(K))

    (dM_A, dM_C) = block_descriptors!((m_buf_A, m_buf_C), M, 0, axis_length(M))
    (dN_B, dN_C) = block_descriptors!((n_buf_B, n_buf_C), N, 0, axis_length(N))

    C = vec(copy(Cref)) .* 0  # zero, same length as Cref's underlying storage
    Cstorage = zeros(Float64, length(Cref))
    Astorage = vec(A)
    Bstorage = vec(B)

    row_A = axis_from_descriptor(dM_A, m_buf_A)
    row_C = axis_from_descriptor(dM_C, m_buf_C)
    col_B = axis_from_descriptor(dN_B, n_buf_B)
    col_C = axis_from_descriptor(dN_C, n_buf_C)

    destination = DestinationTile(Cstorage, 0, row_C, col_C)
    @test nrows(destination) == 6
    @test ncols(destination) == 4

    packed_a = zeros(Float64, mr(kernel) * axis_length(K))
    packed_b = zeros(Float64, nr(kernel) * axis_length(K))

    # Single K panel covering all of K (kc=5) first, to check basic wiring.
    (dK_A, dK_B) = block_descriptors!((k_buf_A, k_buf_B), K, 0, axis_length(K))
    row_K_A = axis_from_descriptor(dK_A, k_buf_A)
    col_K_B = axis_from_descriptor(dK_B, k_buf_B)

    source_A = SourceTile(Astorage, 0, row_A, row_K_A)
    source_B = SourceTile(Bstorage, 0, col_K_B, col_B)

    pack_a!(packed_a, source_A, kernel, identity)
    pack_b!(packed_b, source_B, kernel, identity)
    execute_tile!(kernel, destination, packed_a, packed_b, axis_length(K), 1.0, 0.0)

    Cout = reshape(Cstorage, size(Cref))
    @test Cout ≈ Cref

    # Multiple K panels (lengths 2,2,1),
    # each packed/executed independently, beta_effective=0 on the first panel
    # and 1 on the rest -> must reproduce the same result (up to fp rounding
    # from different accumulation grouping, which is expected/allowed).
    fill!(Cstorage, 0.0)
    panel_starts = (0, 2, 4)
    panel_lengths = (2, 2, 1)
    for (idx, (first, len)) in enumerate(zip(panel_starts, panel_lengths))
        (dKp_A, dKp_B) = block_descriptors!((k_buf_A, k_buf_B), K, first, len)
        rowKp_A = axis_from_descriptor(dKp_A, k_buf_A)
        colKp_B = axis_from_descriptor(dKp_B, k_buf_B)
        srcA = SourceTile(Astorage, 0, row_A, rowKp_A)
        srcB = SourceTile(Bstorage, 0, colKp_B, col_B)
        pack_a!(packed_a, srcA, kernel, identity)
        pack_b!(packed_b, srcB, kernel, identity)
        beta_effective = idx == 1 ? 0.0 : 1.0
        execute_tile!(kernel, destination, packed_a, packed_b, len, 1.0, beta_effective)
    end
    @test reshape(Cstorage, size(Cref)) ≈ Cref

    # Nontrivial alpha/beta across the same 3 panels, starting from a nonzero
    # C, applying the original beta exactly once (on the first panel).
    Cstart = rand(MersenneTwister(1234), 3, 4, 2)
    fill!(Cstorage, 0.0)
    copyto!(Cstorage, vec(Cstart))
    alpha = 2.5
    beta = 0.75
    for (idx, (first, len)) in enumerate(zip(panel_starts, panel_lengths))
        (dKp_A, dKp_B) = block_descriptors!((k_buf_A, k_buf_B), K, first, len)
        rowKp_A = axis_from_descriptor(dKp_A, k_buf_A)
        colKp_B = axis_from_descriptor(dKp_B, k_buf_B)
        srcA = SourceTile(Astorage, 0, row_A, rowKp_A)
        srcB = SourceTile(Bstorage, 0, colKp_B, col_B)
        pack_a!(packed_a, srcA, kernel, identity)
        pack_b!(packed_b, srcB, kernel, identity)
        beta_effective = idx == 1 ? beta : 1.0
        execute_tile!(kernel, destination, packed_a, packed_b, len, alpha, beta_effective)
    end
    expected = alpha .* Cref .+ beta .* Cstart
    @test reshape(Cstorage, size(Cref)) ≈ expected
end

@testset "Phase 2b Fable review: bounds-check regressions" begin
    k = ScalarKernel(Val(2), Val(2), Float64)

    @testset "pack_a!/pack_b!: out-of-bounds source storage rejected before any read" begin
        src = SourceTile([1.0, 2.0, 3.0, 4.0], 10_000, AffineAxis(0, 1, 2), AffineAxis(0, 2, 2))
        packed = zeros(Float64, 4)
        @test_throws BoundsError pack_a!(packed, src, k, identity)
        @test all(iszero, packed)  # rejected before mutation

        src_b = SourceTile([1.0, 2.0, 3.0, 4.0], -10_000, AffineAxis(0, 1, 2), AffineAxis(0, 2, 2))
        @test_throws BoundsError pack_b!(packed, src_b, k, identity)
        @test all(iszero, packed)
    end

    @testset "execute_tile!: out-of-bounds destination storage rejected before any write" begin
        # base=3, rows visit {0,1}, cols visit {0}: max address 3+1+0=4, but
        # storage only has valid indices 0..3 (length 4) -> out of bounds.
        canary = fill(999.0, 4)
        dst = DestinationTile(canary, 3, AffineAxis(0, 1, 2), AffineAxis(0, 0, 1))
        @test_throws BoundsError execute_tile!(k, dst, zeros(4), zeros(4), 0, 1.0, 2.0)
        @test all(==(999.0), canary)  # rejected before mutation
    end

    @testset "execute_tile!: undersized packed buffers rejected before accumulation" begin
        dst = DestinationTile(zeros(4), 0, AffineAxis(0, 1, 2), AffineAxis(0, 2, 2))
        @test_throws DimensionMismatch execute_tile!(k, dst, zeros(4), zeros(4), 50, 1.0, 0.0)
        # kc=0/alpha=0 short-circuits happen before the length check and must
        # still be exempt from it (undersized-but-unused buffers are fine).
        @test execute_tile!(k, dst, zeros(1), zeros(1), 0, 1.0, 0.0) === dst
        @test execute_tile!(k, dst, zeros(1), zeros(1), 5, 0.0, 1.0) === dst
    end

    @testset "axis_offset_range / checked_tile_storage_bounds: direct unit checks" begin
        @test axis_offset_range(AffineAxis(0, 1, 2)) == (0, 1)
        @test axis_offset_range(AffineAxis(5, -2, 3)) == (1, 5)
        @test axis_offset_range(AffineAxis(7, 3, 0)) == (0, -1)  # empty
        @test axis_offset_range(ScatterAxis([4, 1, 9, 2], 3)) == (1, 9)
        @test axis_offset_range(ScatterAxis(Int[], 0)) == (0, -1)

        # A valid tile passes silently.
        @test checked_tile_storage_bounds(0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2), 6) === nothing
        # Empty tile passes regardless of base/storage_length.
        @test checked_tile_storage_bounds(1_000_000, AffineAxis(0, 1, 0), AffineAxis(0, 1, 5), 1) === nothing
        # Out of bounds on the high end.
        @test_throws BoundsError checked_tile_storage_bounds(0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2), 5)
        # Out of bounds on the low end (negative address).
        @test_throws BoundsError checked_tile_storage_bounds(-1, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2), 6)
    end

    @testset "QSTile execute_tile!: alpha/beta shortcut coverage" begin
        kernel = ScalarKernel(Val(3), Val(2), Float64)
        packed_a = Float64[1, 2, 3, 4, 5, 6]   # MR=3, kc=2
        packed_b = Float64[10, 20, 30, 40]     # NR=2, kc=2

        # alpha=0, beta=0: writes zero(T), no read of anything.
        storage = fill(NaN, 6)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
        execute_tile!(kernel, dst, packed_a, packed_b, 2, 0.0, 0.0)
        @test all(iszero, storage)

        # alpha=0, beta=1: full no-op, old values (including NaN) preserved.
        storage2 = [1.0, NaN, 3.0, 4.0, NaN, 6.0]
        dst2 = DestinationTile(storage2, 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
        execute_tile!(kernel, dst2, packed_a, packed_b, 2, 0.0, 1.0)
        @test storage2[1] == 1.0 && isnan(storage2[2]) && storage2[3] == 3.0
        @test storage2[4] == 4.0 && isnan(storage2[5]) && storage2[6] == 6.0

        # beta=0 with NaN-filled old C: result must be finite (no read of old C).
        storage3 = fill(NaN, 6)
        dst3 = DestinationTile(storage3, 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
        execute_tile!(kernel, dst3, packed_a, packed_b, 2, 1.0, 0.0)
        @test all(isfinite, storage3)

        # Empty destination: no-op.
        storage4 = [42.0]
        dst4 = DestinationTile(storage4, 0, AffineAxis(0, 1, 0), AffineAxis(0, 1, 0))
        execute_tile!(kernel, dst4, packed_a, packed_b, 2, 1.0, 0.0)
        @test storage4 == [42.0]

        # Extent-validation errors.
        oversized_rows = DestinationTile(zeros(20), 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 2))
        @test_throws ArgumentError execute_tile!(kernel, oversized_rows, packed_a, packed_b, 2, 1.0, 0.0)

        # Canaries: storage just outside the valid m x n rectangle untouched.
        storage5 = fill(-1.0, 8)  # 6 valid + 2 canary
        dst5 = DestinationTile(storage5, 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
        execute_tile!(kernel, dst5, packed_a, packed_b, 2, 1.0, 0.0)
        @test storage5[7] == -1.0 && storage5[8] == -1.0

        # Float32.
        kernel32 = ScalarKernel(Val(3), Val(2), Float32)
        pa32 = Float32.(packed_a)
        pb32 = Float32.(packed_b)
        storage32 = zeros(Float32, 6)
        dst32 = DestinationTile(storage32, 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
        execute_tile!(kernel32, dst32, pa32, pb32, 2, 1.0f0, 0.0f0)
        @test eltype(storage32) == Float32
        @test all(isfinite, storage32)

        # Affine rows, scattered columns (mixed addressing).
        colbuf = [0, 3]
        scatter_cols = ScatterAxis(colbuf, 2)
        storage6 = zeros(Float64, 6)
        dst6 = DestinationTile(storage6, 0, AffineAxis(0, 1, 3), scatter_cols)
        execute_tile!(kernel, dst6, packed_a, packed_b, 2, 1.0, 0.0)
        dst6ref = DestinationTile(zeros(Float64, 6), 0, AffineAxis(0, 1, 3), AffineAxis(0, 3, 2))
        execute_tile!(kernel, dst6ref, packed_a, packed_b, 2, 1.0, 0.0)
        @test storage6 == dst6ref.storage  # same physical layout, different axis path
    end
end
