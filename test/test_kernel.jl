# Exercises src/kernel.jl: ScalarKernel, zero_accumulator, accumulate,
# store_tile!, scale_tile!, execute_tile! (against QSTile destinations).

using Random

using QuasiStrided: ScalarKernel, scale_tile!, QSTile, AffineAxis, ScatterAxis

@testset "kernel.jl (scalar reference)" begin

    @testset "zero_accumulator" begin
        k = ScalarKernel(Val(4), Val(3), Float64)
        acc = zero_accumulator(k)
        @test size(acc) == (4, 3)
        @test all(iszero, acc)
        @test eltype(acc) == Float64
    end

    @testset "accumulate: kc=0 is a no-op, no reads" begin
        k = ScalarKernel(Val(4), Val(3), Float64)
        acc0 = [Float64(10i + j) for i in 1:4, j in 1:3]
        acc = copy(acc0)
        # Buffers deliberately too short to read from; if accumulate touched
        # them for kc=0 this would throw a BoundsError.
        packed_a = Float64[]
        packed_b = Float64[]
        result = accumulate(k, acc, packed_a, packed_b, 0)
        @test result === acc
        @test acc == acc0
    end

    @testset "accumulate: nonzero initial accumulator composes additively" begin
        k = ScalarKernel(Val(2), Val(2), Float64)
        # kc=1 panel: A column (i=0,1), B row (j=0,1)
        packed_a = [1.0, 2.0]   # a[0,0]=1, a[1,0]=2
        packed_b = [3.0, 4.0]   # b[0,0]=3, b[1,0]=4
        acc_init = [100.0 200.0; 300.0 400.0]
        acc = copy(acc_init)
        accumulate(k, acc, packed_a, packed_b, 1)
        expected = acc_init .+ [1.0 * 3.0 1.0 * 4.0; 2.0 * 3.0 2.0 * 4.0]
        @test acc == expected

        # Running two separate kc=1 accumulate calls must match one kc=2 call
        # (same p order, same inputs split across the two calls).
        packed_a2 = [1.0, 2.0, 5.0, 6.0]  # p=0: (1,2); p=1: (5,6)
        packed_b2 = [3.0, 4.0, 7.0, 8.0]  # p=0: (3,4); p=1: (7,8)
        acc_a = zero_accumulator(k)
        accumulate(k, acc_a, packed_a2, packed_b2, 2)

        acc_b = zero_accumulator(k)
        accumulate(k, acc_b, view(packed_a2, 1:2), view(packed_b2, 1:2), 1)
        accumulate(k, acc_b, view(packed_a2, 3:4), view(packed_b2, 3:4), 1)
        @test acc_a == acc_b
    end

    # --- destination helpers for the tests below ---

    # Storage with `pad` canary cells before and after the addressed region.
    function canary_storage(len::Int; pad::Int = 3, sentinel = -999.0)
        v = fill(sentinel, len + 2 * pad)
        return v, pad
    end

    @testset "execute_tile!: end-to-end mapping vs direct matmul (MR=4,NR=3)" begin
        MR, NR, kc = 4, 3, 5
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(1)

        Amat = rand(rng, MR, kc)   # Amat[i+1,p+1]
        Bmat = rand(rng, kc, NR)   # Bmat[p+1,j+1]

        packed_a = zeros(Float64, packed_a_length(k, kc))
        packed_b = zeros(Float64, packed_b_length(k, kc))
        for p in 0:(kc - 1), i in 0:(MR - 1)
            packed_a[packed_a_offset(k, i, p) + 1] = Amat[i + 1, p + 1]
        end
        for p in 0:(kc - 1), j in 0:(NR - 1)
            packed_b[packed_b_offset(k, j, p) + 1] = Bmat[p + 1, j + 1]
        end

        expected = Amat * Bmat  # (MR, NR)

        storage, pad = canary_storage(MR * NR; pad = 4, sentinel = NaN)
        base = pad
        rows = AffineAxis(0, 1, MR)   # unit-stride rows, contiguous
        cols = AffineAxis(0, MR, NR)  # column-major within the tile
        dest = QSTile(storage, base, rows, cols)

        execute_tile!(k, dest, packed_a, packed_b, kc, 1.0, 0.0)

        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = base + i * 1 + j * MR
            @test storage[addr + 1] ≈ expected[i + 1, j + 1]
        end
        # canaries untouched
        @test all(x -> isnan(x), storage[1:pad])
        @test all(x -> isnan(x), storage[(end - pad + 1):end])
    end

    @testset "output addressing: affine rows/cols (unit, nonunit, negative)" begin
        MR, NR, kc = 3, 2, 1
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        packed_a = [1.0, 2.0, 3.0]
        packed_b = [10.0, 100.0]
        acc = zero_accumulator(k)
        accumulate(k, acc, packed_a, packed_b, kc)
        # acc[i+1,j+1] = a[i]*b[j]
        expected = [1.0 * 10 1.0 * 100; 2.0 * 10 2.0 * 100; 3.0 * 10 3.0 * 100]
        @test acc == expected

        # Case 1: unit rows, unit cols but interleaved via a nonunit column stride.
        len = 100
        storage = fill(NaN, len)
        rows = AffineAxis(20, 1, MR)     # contiguous rows starting at 20
        cols = AffineAxis(0, 7, NR)      # nonunit column stride
        dest = QSTile(storage, 0, rows, cols)
        execute_tile!(k, dest, packed_a, packed_b, kc, 2.0, 0.0)
        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = 20 + i * 1 + j * 7
            @test storage[addr + 1] ≈ 2.0 * expected[i + 1, j + 1]
        end

        # Case 2: negative row stride (rows stored in reverse), nonunit cols.
        storage2 = fill(NaN, len)
        rows2 = AffineAxis(50, -3, MR)
        cols2 = AffineAxis(0, 11, NR)
        dest2 = QSTile(storage2, 0, rows2, cols2)
        execute_tile!(k, dest2, packed_a, packed_b, kc, 1.0, 0.0)
        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = 50 - 3i + 11j
            @test storage2[addr + 1] ≈ expected[i + 1, j + 1]
        end
        # Addresses for i=0..2 with stride -3 from 50 are 50,47,44 — distinct
        # from each other and from any column-shifted address in range, so no
        # aliasing corrupts the comparison above.
    end

    @testset "output addressing: scattered rows and columns" begin
        MR, NR, kc = 3, 2, 1
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        packed_a = [1.0, 2.0, 3.0]
        packed_b = [10.0, 100.0]
        acc = zero_accumulator(k)
        accumulate(k, acc, packed_a, packed_b, kc)
        expected = [1.0 * 10 1.0 * 100; 2.0 * 10 2.0 * 100; 3.0 * 10 3.0 * 100]

        row_offsets = [5, 40, 12]   # arbitrary, distinct, irregular
        col_offsets = [0, 200]
        storage = fill(NaN, 300)
        rows = ScatterAxis(row_offsets, MR)
        cols = ScatterAxis(col_offsets, NR)
        dest = QSTile(storage, 0, rows, cols)
        execute_tile!(k, dest, packed_a, packed_b, kc, 1.0, 0.0)
        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = row_offsets[i + 1] + col_offsets[j + 1]
            @test storage[addr + 1] ≈ expected[i + 1, j + 1]
        end

        # Mixed: scattered rows, affine columns.
        storage2 = fill(NaN, 300)
        cols2 = AffineAxis(0, 100, NR)
        dest2 = QSTile(storage2, 0, rows, cols2)
        execute_tile!(k, dest2, packed_a, packed_b, kc, 1.0, 0.0)
        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = row_offsets[i + 1] + j * 100
            @test storage2[addr + 1] ≈ expected[i + 1, j + 1]
        end
    end

    @testset "beta=0 never reads old C (NaN old C, finite result)" begin
        MR, NR, kc = 4, 3, 2
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        packed_a = rand(MersenneTwister(2), packed_a_length(k, kc))
        packed_b = rand(MersenneTwister(3), packed_b_length(k, kc))

        storage = fill(NaN, MR * NR)  # old C entirely NaN
        rows = AffineAxis(0, 1, MR)
        cols = AffineAxis(0, MR, NR)
        dest = QSTile(storage, 0, rows, cols)
        execute_tile!(k, dest, packed_a, packed_b, kc, 1.5, 0.0)
        @test all(isfinite, storage)

        # Textual confirmation (see src/kernel.jl): in store_tile!'s
        # `iszero(beta)` branch, and in scale_tile!'s `iszero(beta)` branch,
        # the only statement touching storage is an assignment, never a load.
    end

    @testset "alpha=0 skips accumulator arithmetic (no Inf/NaN from operands)" begin
        MR, NR, kc = 2, 2, 1
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        # These packed values would produce Inf/NaN if actually multiplied
        # and accumulated (0 * Inf = NaN).
        packed_a = [0.0, Inf]
        packed_b = [Inf, 0.0]

        storage = [1.0, 2.0, 3.0, 4.0]
        rows = AffineAxis(0, 1, MR)
        cols = AffineAxis(0, MR, NR)
        dest = QSTile(storage, 0, rows, cols)
        old = copy(storage)
        execute_tile!(k, dest, packed_a, packed_b, kc, 0.0, 2.0)
        # alpha=0, beta=2: result should be exactly 2*old C, no NaN anywhere.
        @test storage ≈ 2.0 .* old
        @test all(isfinite, storage)

        # alpha=0 && beta=0: writes zero(T), no read of C or acc.
        storage2 = fill(NaN, 4)
        dest2 = QSTile(storage2, 0, rows, cols)
        execute_tile!(k, dest2, packed_a, packed_b, kc, 0.0, 0.0)
        @test all(iszero, storage2)

        # alpha=0 && beta=1: full no-op (storage unchanged, including NaNs).
        storage3 = fill(NaN, 4)
        dest3 = QSTile(storage3, 0, rows, cols)
        execute_tile!(k, dest3, packed_a, packed_b, kc, 0.0, 1.0)
        @test all(isnan, storage3)
    end

    @testset "output canaries and only-valid-lane stores (padded lanes never propagate)" begin
        MR, NR = 4, 4
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        m, n, kc = 2, 2, 1  # valid subrectangle smaller than the register tile

        # Padding lanes (i>=m or j>=n) get Inf/NaN-inducing packed values;
        # valid lanes (i<m, j<n) get benign finite values.
        packed_a = zeros(Float64, MR)
        packed_b = zeros(Float64, NR)
        packed_a[1] = 2.0; packed_a[2] = 3.0   # valid rows i=0,1
        packed_a[3] = 0.0; packed_a[4] = Inf   # padding rows i=2,3
        packed_b[1] = 5.0; packed_b[2] = 7.0   # valid cols j=0,1
        packed_b[3] = Inf; packed_b[4] = 0.0   # padding cols j=2,3

        acc = zero_accumulator(k)
        accumulate(k, acc, packed_a, packed_b, kc)
        # Valid corner is finite and correct.
        @test acc[1, 1] ≈ 2.0 * 5.0
        @test acc[1, 2] ≈ 2.0 * 7.0
        @test acc[2, 1] ≈ 3.0 * 5.0
        @test acc[2, 2] ≈ 3.0 * 7.0
        # Some padded lanes are indeed nonfinite (proving the test actually
        # stresses the hazard, not a vacuous check).
        @test any(!isfinite, acc[3:4, :]) || any(!isfinite, acc[:, 3:4])

        pad = 5
        storage = fill(-777.0, m * n + 2 * pad)
        base = pad
        rows = AffineAxis(0, 1, m)
        cols = AffineAxis(0, m, n)
        dest = QSTile(storage, base, rows, cols)
        execute_tile!(k, dest, packed_a, packed_b, kc, 1.0, 0.0)

        # Valid rectangle: finite and correct.
        for i in 0:(m - 1), j in 0:(n - 1)
            addr = base + i * 1 + j * m
            @test isfinite(storage[addr + 1])
            @test storage[addr + 1] ≈ acc[i + 1, j + 1]
        end
        # Canaries before/after the addressed region: untouched.
        @test all(==(-777.0), storage[1:pad])
        @test all(==(-777.0), storage[(end - pad + 1):end])
    end

    @testset "nontrivial (non-0/1) alpha and beta, single execute_tile! call" begin
        MR, NR, kc = 3, 3, 4
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(7)
        Amat = rand(rng, MR, kc)
        Bmat = rand(rng, kc, NR)
        packed_a = zeros(Float64, packed_a_length(k, kc))
        packed_b = zeros(Float64, packed_b_length(k, kc))
        for p in 0:(kc - 1), i in 0:(MR - 1)
            packed_a[packed_a_offset(k, i, p) + 1] = Amat[i + 1, p + 1]
        end
        for p in 0:(kc - 1), j in 0:(NR - 1)
            packed_b[packed_b_offset(k, j, p) + 1] = Bmat[p + 1, j + 1]
        end

        alpha, beta = 2.5, -1.75
        Cold = rand(rng, MR, NR)
        storage = vec(permutedims(Cold))  # so that row-major-style affine addressing lines up
        # Use simple row-major-ish affine addressing: addr(i,j) = i*NR + j
        rows = AffineAxis(0, NR, MR)
        cols = AffineAxis(0, 1, NR)
        dest = QSTile(copy(storage), 0, rows, cols)
        execute_tile!(k, dest, packed_a, packed_b, kc, alpha, beta)

        expected = alpha .* (Amat * Bmat) .+ beta .* Cold
        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = i * NR + j
            @test dest.storage[addr + 1] ≈ expected[i + 1, j + 1] atol = 1.0e-10
        end
    end

    @testset "execute_tile!: kc=0 applies beta once, no input reads" begin
        MR, NR = 2, 2
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        storage = [1.0, 2.0, 3.0, 4.0]
        rows = AffineAxis(0, 1, MR)
        cols = AffineAxis(0, MR, NR)
        dest = QSTile(storage, 0, rows, cols)
        # Deliberately empty/undersized packed buffers: if execute_tile! read
        # from them for kc=0 this would throw a BoundsError.
        empty_a = Float64[]
        empty_b = Float64[]
        execute_tile!(k, dest, empty_a, empty_b, 0, 3.0, 2.0)
        @test storage ≈ [2.0, 4.0, 6.0, 8.0]
    end

    @testset "empty destination is a no-op" begin
        MR, NR = 2, 2
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        storage = fill(NaN, 4)
        rows = AffineAxis(0, 1, 0)
        cols = AffineAxis(0, MR, 0)
        dest = QSTile(storage, 0, rows, cols)
        execute_tile!(k, dest, Float64[], Float64[], 0, 1.0, 1.0)
        @test all(isnan, storage)  # untouched, still NaN
    end

    @testset "execute_tile! validates destination extent against kernel shape" begin
        MR, NR = 2, 2
        k = ScalarKernel(Val(MR), Val(NR), Float64)
        storage = zeros(10)
        rows = AffineAxis(0, 1, 3)  # m=3 > MR=2
        cols = AffineAxis(0, MR, 2)
        dest = QSTile(storage, 0, rows, cols)
        @test_throws ArgumentError execute_tile!(k, dest, [1.0, 2.0], [1.0, 2.0], 1, 1.0, 0.0)
    end

    @testset "Float32 works uniformly" begin
        MR, NR, kc = 2, 2, 2
        k = ScalarKernel(Val(MR), Val(NR), Float32)
        packed_a = Float32[1, 2, 3, 4]
        packed_b = Float32[1, 1, 1, 1]
        acc = zero_accumulator(k)
        @test eltype(acc) == Float32
        accumulate(k, acc, packed_a, packed_b, kc)
        storage = zeros(Float32, 4)
        rows = AffineAxis(0, 1, MR)
        cols = AffineAxis(0, MR, NR)
        dest = QSTile(storage, 0, rows, cols)
        execute_tile!(k, dest, packed_a, packed_b, kc, Float32(1), Float32(0))
        @test eltype(storage) == Float32
    end

end
