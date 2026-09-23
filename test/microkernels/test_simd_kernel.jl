# Exercises src/microkernels/simd.jl: SIMDKernel vs. ScalarKernel on identical
# inputs (numerical tolerance, never bitwise equality).
using QuasiStrided: SIMDKernel, ScalarKernel, lanewidth, avecs_per_column,
    _vector_store_eligible
using Random
using SIMD: Vec
# `parent(::StridedView)` is how the driver obtains a destination's storage
# (src/execution/execute.jl, `Cstorage = parent(C)`): `Memory{T}` on Julia >= 1.11, a
# `Vector{T}` sharing memory on 1.10. The store-path testsets below build
# their destinations the same way rather than assuming either one.
using StridedViews: StridedView

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

    # The storage the driver actually hands `store_tile!`, holding a copy of
    # `v`: `Memory{T}` on Julia >= 1.11, `Vector{T}` on 1.10. Both are
    # `DenseVector{T}`, so the vectorized store path must take either.
    function dense_storage(v::AbstractVector{T}) where {T}
        storage = parent(StridedView(zeros(T, length(v))))
        for i in eachindex(v)
            storage[i] = v[i]
        end
        return storage
    end

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

    # ------------------------------------------------------------------------
    # Vectorized store path. Its guard admits any concrete `DenseVector{T}`
    # (exactly what SIMD.jl's array `vload`/`vstore` methods accept), not just
    # `Vector{T}`: `parent` of an Array-backed StridedView is `Memory{T}` on
    # Julia >= 1.11, so a `Vector{T}`-only guard would never fire.
    # ------------------------------------------------------------------------

    @testset "_vector_store_eligible: any 1-D dense storage, unit-stride rows only" begin
        for T in (Float64, Float32)
            m, n = 8, 8
            unit_rows = AffineAxis(0, 1, m)
            cols = AffineAxis(0, m, n)

            # What the driver hands `store_tile!`: `Memory{T}` on Julia
            # >= 1.11, `Vector{T}` on 1.10. Either way, eligible.
            mem = parent(StridedView(zeros(T, m, n)))
            @test _vector_store_eligible(DestinationTile(mem, 0, unit_rows, cols), T)
            # A plain `Vector{T}`.
            @test _vector_store_eligible(DestinationTile(zeros(T, m * n), 0, unit_rows, cols), T)

            # Excluded, and so still on the scalar fallback: a `SubArray` (a
            # contiguous view is contiguous but is not a `DenseArray`), and
            # rank-2 storage (SIMD.jl's array methods are rank-1 only).
            buf = zeros(T, m * n + 4)
            @test !_vector_store_eligible(
                DestinationTile(view(buf, 3:(2 + m * n)), 0, unit_rows, cols), T
            )
            @test !_vector_store_eligible(DestinationTile(zeros(T, m, n), 0, unit_rows, cols), T)

            # Unit-stride rows are required independently of the storage type:
            # a stride-2 affine axis or a scatter axis is never eligible, even
            # over dense rank-1 storage.
            stride2 = AffineAxis(0, 2, m)
            wide_cols = AffineAxis(0, 2m, n)
            @test !_vector_store_eligible(
                DestinationTile(zeros(T, 2m * n), 0, stride2, wide_cols), T
            )
            @test !_vector_store_eligible(
                DestinationTile(parent(StridedView(zeros(T, 2m * n))), 0, stride2, wide_cols), T
            )
            @test !_vector_store_eligible(
                DestinationTile(mem, 0, ScatterAxis(collect(0:(m - 1)), m), cols), T
            )

            # The element type must be the kernel's own.
            other = (T === Float64) ? Float32 : Float64
            @test !_vector_store_eligible(DestinationTile(mem, 0, unit_rows, cols), other)
        end
    end

    @testset "vectorized store path and scalar fallback agree (dense 1-D vs view storage)" begin
        MR, NR, kc = 8, 6, 5
        k = SIMDKernel(Val(MR), Val(NR), Float64)
        W = lanewidth(k)
        rng = MersenneTwister(2026)
        Amat = rand(rng, MR, kc)
        Bmat = rand(rng, kc, NR)
        pa, pb = packed_from_matrices(k, Amat, Bmat, kc)

        for (m, n) in ((MR, NR), (MR - 3, NR - 1), (W - 1, 1), (1, 1)),
                alpha in (1.0, 1.5), beta in (0.0, 1.0, -0.4)

            Cold = rand(rng, m * n)
            rows, cols = AffineAxis(0, 1, m), AffineAxis(0, m, n)

            # Vectorized path: dense rank-1 storage, unit-stride rows.
            fast = dense_storage(Cold)
            dst_fast = DestinationTile(fast, 0, rows, cols)
            @test _vector_store_eligible(dst_fast, Float64)
            execute_tile!(k, dst_fast, pa, pb, kc, alpha, beta)

            # Scalar fallback, forced by `SubArray` storage over the same
            # logical layout (padded, so an over-wide store would be caught).
            pad = 3
            slow_parent = fill(-77.0, m * n + 2pad)
            slow = view(slow_parent, 1:(m * n + 2pad))
            for i in eachindex(Cold)
                slow[pad + i] = Cold[i]
            end
            dst_slow = DestinationTile(slow, pad, rows, cols)
            @test !_vector_store_eligible(dst_slow, Float64)
            execute_tile!(k, dst_slow, pa, pb, kc, alpha, beta)

            @test collect(fast) ≈ slow[(pad + 1):(pad + m * n)] atol = 1.0e-10 rtol = 1.0e-10
            @test all(==(-77.0), slow_parent[1:pad])
            @test all(==(-77.0), slow_parent[(end - pad + 1):end])
        end

        @testset "beta=0 on dense 1-D storage never reads old C (NaN old C, tail rows)" begin
            m, n = MR - 3, NR - 1
            storage = dense_storage(fill(NaN, m * n))
            dst = DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
            @test _vector_store_eligible(dst, Float64)
            execute_tile!(k, dst, pa, pb, kc, 1.5, 0.0)
            @test all(isfinite, collect(storage))
        end

        @testset "padded-lane isolation on the vectorized path (nonfinite acc lanes)" begin
            # Same fixture as the fallback's own padded-lane testset above:
            # rows m..MR-1 and columns n..NR-1 of the accumulator are
            # nonfinite, and must neither be stored nor read.
            m, n = 3, 2
            packed_a = zeros(Float64, MR)
            packed_b = zeros(Float64, NR)
            packed_a[1:m] .= [2.0, 3.0, 5.0]
            packed_a[(m + 1):MR] .= Inf
            packed_b[1:n] .= [7.0, 11.0]
            packed_b[(n + 1):NR] .= Inf

            pad = 5
            storage = dense_storage(fill(-999.0, m * n + 2pad))
            dst = DestinationTile(storage, pad, AffineAxis(0, 1, m), AffineAxis(0, m, n))
            @test _vector_store_eligible(dst, Float64)
            execute_tile!(k, dst, packed_a, packed_b, 1, 1.0, 0.0)

            expected = [2.0, 3.0, 5.0] * [7.0 11.0]
            for i in 0:(m - 1), j in 0:(n - 1)
                addr = pad + i + j * m
                @test isfinite(storage[addr + 1])
                @test storage[addr + 1] ≈ expected[i + 1, j + 1]
            end
            @test all(==(-999.0), collect(storage)[1:pad])
            @test all(==(-999.0), collect(storage)[(end - pad + 1):end])
        end

        @testset "vectorized store path with unit-stride rows but SCATTERED columns" begin
            # _vector_store_eligible only inspects `tile.rows`/`tile.storage`
            # -- a ScatterAxis on the COLUMN side is untouched by the guard
            # and still takes the vectorized path (axis_offset dispatches on
            # the axis type generically). This is the case QuasiStrided exists
            # for (irregular/permuted output axes), and it depends on
            # `colbase` being computed from `axis_offset(cols, j)` only
            # inside the `j < n` guard (src/microkernels/simd.jl).
            m, n = MR - 3, NR - 1
            col_offsets = collect(0:2:(2 * (n - 1)))  # a non-affine (but here regular) permutation-style column map
            rows = AffineAxis(0, 1, m)
            cols = ScatterAxis(col_offsets, n)
            span = m * (maximum(col_offsets) + 1)

            Cold = rand(rng, span)
            fast = dense_storage(copy(Cold))
            dst_fast = DestinationTile(fast, 0, rows, cols)
            @test _vector_store_eligible(dst_fast, Float64)  # rows are unit-stride and dense; cols type is irrelevant to the guard
            execute_tile!(k, dst_fast, pa, pb, kc, 1.5, 0.5)

            slow = copy(Cold)
            dst_slow = DestinationTile(view(slow, 1:span), 0, rows, cols)
            @test !_vector_store_eligible(dst_slow, Float64)  # SubArray storage forces the fallback
            execute_tile!(k, dst_slow, pa, pb, kc, 1.5, 0.5)

            @test collect(fast) ≈ slow atol = 1.0e-10 rtol = 1.0e-10
        end
    end

    @testset "allocation: execute_tile! on dense 1-D storage WITH TAIL ROWS is allocation-free" begin
        # The vectorized store's tail must be statically indexed: a
        # dynamically indexed accumulator tuple heap-allocates above NV = 16
        # (GUARDRAIL, src/microkernels/simd.jl), and the vectorized branch is
        # reachable for real destinations on Julia >= 1.11. Covers every
        # shipped register shape plus NV = 24 and NV = 28, i.e. past the
        # cliff and up to the register budget planning/test_kernel_selection.jl allows.
        function run_execute(k, dst, pa, pb, kc)
            execute_tile!(k, dst, pa, pb, kc, 1.0, 0.5)
            return nothing
        end

        shapes = Any[]
        for T in (Float64, Float32), shape in QuasiStrided.kernel_shapes(T)
            push!(shapes, (T, shape))
        end
        push!(shapes, (Float64, (16, 6, 4)), (Float64, (16, 7, 4)))  # NV = 24, 28

        for (T, (MR, NR, W)) in shapes
            k = SIMDKernel(Val(MR), Val(NR), T, Val(W))
            kc = 4
            rng = MersenneTwister(77)
            pa, pb = packed_from_matrices(k, rand(rng, T, MR, kc), rand(rng, T, kc, NR), kc)
            # (MR - 1, NR) and (W - 1, 1) both leave a partial W-row block.
            for (m, n) in ((MR, NR), (MR - 1, NR), (max(W - 1, 1), 1))
                storage = parent(StridedView(zeros(T, m * n)))
                dst = DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
                @test _vector_store_eligible(dst, T)
                run_execute(k, dst, pa, pb, kc)  # warm up (compile) before measuring
                bytes = @allocated run_execute(k, dst, pa, pb, kc)
                @test bytes == 0 skip = (VERSION < v"1.11")
            end
        end
    end

end
