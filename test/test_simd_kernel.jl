# Exercises src/kernels/simd.jl: SIMDKernel vs. ScalarKernel on identical
# inputs (numerical tolerance, never bitwise equality).
using QuasiStrided: SIMDKernel, ScalarKernel, lanewidth, avecs_per_column
using Random
using SIMD: Vec

@testset "kernels/simd.jl (SIMD candidate)" begin

    @testset "zero_accumulator" begin
        k = SIMDKernel(Val(8), Val(6), Float64)
        acc = zero_accumulator(k)
        @test acc isa NTuple{12, Vec{4, Float64}}
        @test all(v -> all(iszero, Tuple(v)), acc)
    end

    @testset "accumulate: kc=0 is a no-op, no reads" begin
        k = SIMDKernel(Val(8), Val(6), Float64)
        acc0 = zero_accumulator(k)
        packed_a = Float64[]
        packed_b = Float64[]
        result = accumulate(k, acc0, packed_a, packed_b, 0)
        @test result === acc0
    end

    @testset "accumulate: nonzero initial accumulator composes additively, matches split kc" begin
        k = SIMDKernel(Val(8), Val(6), Float64)
        rng = MersenneTwister(11)
        packed_a2 = rand(rng, packed_a_length(k, 2))
        packed_b2 = rand(rng, packed_b_length(k, 2))

        acc_a = zero_accumulator(k)
        acc_a = accumulate(k, acc_a, packed_a2, packed_b2, 2)

        acc_b = zero_accumulator(k)
        acc_b = accumulate(k, acc_b, view(packed_a2, 1:8), view(packed_b2, 1:6), 1)
        acc_b = accumulate(k, acc_b, view(packed_a2, 9:16), view(packed_b2, 7:12), 1)

        for idx in 1:length(acc_a)
            @test all(Tuple(acc_a[idx]) .≈ Tuple(acc_b[idx]))
        end
    end

    # --- helpers ---

    function packed_from_matrices(k, Amat, Bmat, kc)
        pa = zeros(eltype(Amat), packed_a_length(k, kc))
        pb = zeros(eltype(Bmat), packed_b_length(k, kc))
        MR = size(Amat, 1)
        NR = size(Bmat, 2)
        for p in 0:(kc - 1), i in 0:(MR - 1)
            pa[packed_a_offset(k, i, p) + 1] = Amat[i + 1, p + 1]
        end
        for p in 0:(kc - 1), j in 0:(NR - 1)
            pb[packed_b_offset(k, j, p) + 1] = Bmat[p + 1, j + 1]
        end
        return pa, pb
    end

    @testset "execute_tile!: end-to-end vs direct matmul, full tile" begin
        MR, NR, kc = 8, 6, 5
        k = SIMDKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(1)
        Amat = rand(rng, MR, kc)
        Bmat = rand(rng, kc, NR)
        pa, pb = packed_from_matrices(k, Amat, Bmat, kc)
        expected = Amat * Bmat

        pad = 4
        storage = fill(NaN, MR * NR + 2pad)
        dst = DestinationTile(storage, pad, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        execute_tile!(k, dst, pa, pb, kc, 1.0, 0.0)

        for i in 0:(MR - 1), j in 0:(NR - 1)
            addr = pad + i + j * MR
            @test storage[addr + 1] ≈ expected[i + 1, j + 1]
        end
        @test all(isnan, storage[1:pad])
        @test all(isnan, storage[(end - pad + 1):end])
    end

    @testset "execute_tile!: SIMD vs scalar kernel, tolerance not bitwise, several shapes/kc/alpha/beta" begin
        MR, NR = 8, 6
        ksimd = SIMDKernel(Val(MR), Val(NR), Float64)
        kscalar = ScalarKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(42)

        for kc in (0, 1, 3, 17), (alpha, beta) in ((1.0, 0.0), (2.5, -1.75), (1.0, 1.0), (0.0, 3.0))
            Amat = rand(rng, MR, max(kc, 1))
            Bmat = rand(rng, max(kc, 1), NR)
            pa, pb = packed_from_matrices(ksimd, Amat, Bmat, kc)

            Cold = rand(rng, MR, NR)
            storage_simd = vec(permutedims(Cold))
            storage_scalar = copy(storage_simd)
            rows = AffineAxis(0, NR, MR)
            cols = AffineAxis(0, 1, NR)
            dst_simd = DestinationTile(storage_simd, 0, rows, cols)
            dst_scalar = DestinationTile(storage_scalar, 0, rows, cols)

            execute_tile!(ksimd, dst_simd, pa, pb, kc, alpha, beta)
            execute_tile!(kscalar, dst_scalar, pa, pb, kc, alpha, beta)

            @test storage_simd ≈ storage_scalar atol = 1.0e-10 rtol = 1.0e-10
        end
    end

    @testset "tail handling: m, n smaller than MR, NR (partial tile, unit-stride vectorized path)" begin
        MR, NR = 8, 6
        ksimd = SIMDKernel(Val(MR), Val(NR), Float64)
        kscalar = ScalarKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(3)

        for (m, n) in ((1, 1), (3, 2), (5, 4), (7, 6), (8, 3), (0, 0))
            kc = 4
            Amat = rand(rng, MR, kc)
            Bmat = rand(rng, kc, NR)
            pa, pb = packed_from_matrices(ksimd, Amat, Bmat, kc)

            pad = 3
            storage_simd = fill(-123.0, m * n + 2pad)
            storage_scalar = copy(storage_simd)
            rows = AffineAxis(0, 1, m)
            cols = AffineAxis(0, m, n)
            dst_simd = DestinationTile(storage_simd, pad, rows, cols)
            dst_scalar = DestinationTile(storage_scalar, pad, rows, cols)

            execute_tile!(ksimd, dst_simd, pa, pb, kc, 1.7, 0.3)
            execute_tile!(kscalar, dst_scalar, pa, pb, kc, 1.7, 0.3)

            @test storage_simd ≈ storage_scalar atol = 1.0e-10 rtol = 1.0e-10
            @test all(==(-123.0), storage_simd[1:pad])
            @test all(==(-123.0), storage_simd[(end - pad + 1):end])
        end
    end

    @testset "tail handling: scattered rows and non-Vector storage fall back correctly" begin
        MR, NR = 8, 6
        k = SIMDKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(4)
        m, n, kc = 5, 4, 3
        Amat = rand(rng, MR, kc)
        Bmat = rand(rng, kc, NR)
        pa, pb = packed_from_matrices(k, Amat, Bmat, kc)

        # Reference: unit-stride Vector storage (takes the vectorized path).
        storage_ref = zeros(Float64, m * n)
        dst_ref = DestinationTile(storage_ref, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        execute_tile!(k, dst_ref, pa, pb, kc, 1.0, 0.0)

        # Scattered rows (forces the scalar fallback branch).
        row_offsets = collect(0:(m - 1)) # same physical layout, different axis representation
        storage_scatter = zeros(Float64, m * n)
        dst_scatter = DestinationTile(storage_scatter, 0, ScatterAxis(row_offsets, m), AffineAxis(0, m, n))
        execute_tile!(k, dst_scatter, pa, pb, kc, 1.0, 0.0)
        @test storage_scatter == storage_ref

        # Non-unit affine row stride (also forces the scalar fallback branch).
        storage_stride = zeros(Float64, 2m * n)
        dst_stride = DestinationTile(storage_stride, 0, AffineAxis(0, 2, m), AffineAxis(0, 2m, n))
        execute_tile!(k, dst_stride, pa, pb, kc, 1.0, 0.0)
        for i in 0:(m - 1), j in 0:(n - 1)
            @test storage_stride[2i + 2m * j + 1] ≈ storage_ref[i + m * j + 1]
        end

        # Non-Vector (view) storage: still correct via the scalar fallback,
        # even though rows are unit-affine.
        parent_buf = zeros(Float64, m * n + 4)
        storage_view = view(parent_buf, 3:(3 + m * n - 1))
        dst_view = DestinationTile(storage_view, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        execute_tile!(k, dst_view, pa, pb, kc, 1.0, 0.0)
        @test collect(storage_view) == storage_ref
    end

    @testset "padded-lane isolation: nonfinite padding never propagates or is stored" begin
        MR, NR = 8, 6
        k = SIMDKernel(Val(MR), Val(NR), Float64)
        m, n, kc = 3, 2, 1  # small valid subrectangle, lots of padding

        packed_a = zeros(Float64, MR)
        packed_b = zeros(Float64, NR)
        packed_a[1:m] .= [2.0, 3.0, 5.0]     # valid rows
        packed_a[(m + 1):MR] .= Inf          # padding rows: deliberately nonfinite
        packed_b[1:n] .= [7.0, 11.0]         # valid cols
        packed_b[(n + 1):NR] .= Inf          # padding cols: deliberately nonfinite

        acc = zero_accumulator(k)
        acc = accumulate(k, acc, packed_a, packed_b, kc)
        # Confirm the hazard is real: some padding-lane accumulator entries
        # are indeed nonfinite (0 * Inf = NaN for the zero-padded coordinate
        # crossed with a nonfinite padding lane; padding-row * padding-col is
        # Inf*Inf = Inf).
        W = lanewidth(k)
        NVECA = avecs_per_column(k)
        any_nonfinite = false
        for i in 0:(MR - 1), j in 0:(NR - 1)
            v, lane = i ÷ W, (i % W) + 1
            r = acc[v + NVECA * j + 1][lane]
            (i >= m || j >= n) && !isfinite(r) && (any_nonfinite = true)
        end
        @test any_nonfinite

        pad = 5
        storage = fill(-999.0, m * n + 2pad)
        dst = DestinationTile(storage, pad, AffineAxis(0, 1, m), AffineAxis(0, m, n))
        execute_tile!(k, dst, packed_a, packed_b, kc, 1.0, 0.0)

        expected = [2.0 * 7.0 2.0 * 11.0; 3.0 * 7.0 3.0 * 11.0; 5.0 * 7.0 5.0 * 11.0]
        for i in 0:(m - 1), j in 0:(n - 1)
            addr = pad + i + j * m
            @test isfinite(storage[addr + 1])
            @test storage[addr + 1] ≈ expected[i + 1, j + 1]
        end
        @test all(==(-999.0), storage[1:pad])
        @test all(==(-999.0), storage[(end - pad + 1):end])
    end

    @testset "alpha/beta shortcuts" begin
        MR, NR, kc = 8, 6, 2
        k = SIMDKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(5)
        Amat = rand(rng, MR, kc)
        Bmat = rand(rng, kc, NR)
        pa, pb = packed_from_matrices(k, Amat, Bmat, kc)

        @testset "beta=0 never reads old C (NaN old C, finite result)" begin
            storage = fill(NaN, MR * NR)
            dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
            execute_tile!(k, dst, pa, pb, kc, 1.5, 0.0)
            @test all(isfinite, storage)
        end

        @testset "alpha=0 skips accumulator arithmetic (no Inf/NaN from operands)" begin
            pa_bad = zeros(Float64, packed_a_length(k, 1))
            pb_bad = zeros(Float64, packed_b_length(k, 1))
            pa_bad[1] = Inf
            pb_bad[1] = Inf

            old = rand(MersenneTwister(6), MR * NR)
            storage = copy(old)
            dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
            execute_tile!(k, dst, pa_bad, pb_bad, 1, 0.0, 2.0)
            @test storage ≈ 2.0 .* old
            @test all(isfinite, storage)

            storage2 = fill(NaN, MR * NR)
            dst2 = DestinationTile(storage2, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
            execute_tile!(k, dst2, pa_bad, pb_bad, 1, 0.0, 0.0)
            @test all(iszero, storage2)

            storage3 = fill(NaN, MR * NR)
            dst3 = DestinationTile(storage3, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
            execute_tile!(k, dst3, pa_bad, pb_bad, 1, 0.0, 1.0)
            @test all(isnan, storage3)
        end

        @testset "execute_tile!: kc=0 applies beta once, no input reads" begin
            storage = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            dst = DestinationTile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 2))
            execute_tile!(k, dst, Float64[], Float64[], 0, 3.0, 2.0)
            @test storage ≈ 2.0 .* [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        end
    end

    @testset "empty destination is a no-op" begin
        k = SIMDKernel(Val(8), Val(6), Float64)
        storage = fill(NaN, 4)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 1, 0))
        execute_tile!(k, dst, Float64[], Float64[], 0, 1.0, 1.0)
        @test all(isnan, storage)
    end

    @testset "execute_tile! validates destination extent against kernel shape" begin
        k = SIMDKernel(Val(4), Val(2), Float64)
        storage = zeros(20)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, 5), AffineAxis(0, 5, 2))  # m=5 > MR=4
        @test_throws ArgumentError execute_tile!(k, dst, zeros(8), zeros(4), 1, 1.0, 0.0)
    end

    @testset "Float32 works uniformly, custom lane width" begin
        MR, NR, kc = 16, 6, 2
        k = SIMDKernel(Val(MR), Val(NR), Float32)  # default lanewidth(Float32) = 8
        @test lanewidth(k) == 8
        @test avecs_per_column(k) == 2
        rng = MersenneTwister(9)
        Amat = rand(rng, Float32, MR, kc)
        Bmat = rand(rng, Float32, kc, NR)
        pa, pb = packed_from_matrices(k, Amat, Bmat, kc)
        acc = zero_accumulator(k)
        @test eltype(Tuple(acc[1])) == Float32
        acc = accumulate(k, acc, pa, pb, kc)

        storage = zeros(Float32, MR * NR)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        execute_tile!(k, dst, pa, pb, kc, Float32(1), Float32(0))
        @test eltype(storage) == Float32
        expected = Amat * Bmat
        for i in 0:(MR - 1), j in 0:(NR - 1)
            @test storage[i + 1 + j * MR] ≈ expected[i + 1, j + 1]
        end
    end

    @testset "SIMDKernel constructor validation" begin
        @test_throws ArgumentError SIMDKernel(Val(6), Val(4), Float64, Val(4))  # 6 not a multiple of 4
        @test_throws ArgumentError SIMDKernel(Val(6), Val(4), Float64, Val(0))  # W must be > 0
    end

    @testset "allocation: accumulate and execute_tile! are steady-state allocation-free" begin
        MR, NR, kc = 8, 6, 8
        k = SIMDKernel(Val(MR), Val(NR), Float64)
        rng = MersenneTwister(10)
        Amat = rand(rng, MR, kc)
        Bmat = rand(rng, kc, NR)
        pa, pb = packed_from_matrices(k, Amat, Bmat, kc)
        storage = zeros(Float64, MR * NR)
        dst = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))

        # Warm up (compile) before measuring.
        acc0 = zero_accumulator(k)
        accumulate(k, acc0, pa, pb, kc)
        execute_tile!(k, dst, pa, pb, kc, 1.0, 0.0)

        function run_accumulate(k, pa, pb, kc)
            acc = zero_accumulator(k)
            return accumulate(k, acc, pa, pb, kc)
        end
        bytes_acc = @allocated run_accumulate(k, pa, pb, kc)
        # Confirmed zero on Julia >= 1.11; Julia 1.10 (LTS) materializes the
        # NTuple{NV,Vec{W,T}} accumulator instead of keeping it register-
        # resident (tens of KB/call) -- a compiler capability gap, not a bug
        # here, so skip rather than hide it on older Julia.
        @test bytes_acc == 0 skip = (VERSION < v"1.11")

        function run_execute(k, dst, pa, pb, kc)
            execute_tile!(k, dst, pa, pb, kc, 1.0, 0.0)
            return nothing
        end
        bytes_exec = @allocated run_execute(k, dst, pa, pb, kc)
        @test bytes_exec == 0 skip = (VERSION < v"1.11")
    end

end
