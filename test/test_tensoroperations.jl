# TO-comparison test suite for `QuasiStrided.QuasiStridedBackend`, authored
# blind against the frozen spec per docs/decisions.md's "Correctness oracle"
# section. Included by `test/runtests.jl`, which already provides
# `Test`/`Random`/`QuasiStrided`.

using TensorOperations
using TensorOperations: StridedNative, StridedBLAS
using StridedViews: StridedView, isstrided
using LinearAlgebra: Diagonal
using Bumper: Bumper, default_buffer, @no_escape

const qsbackend = QuasiStrided.QuasiStridedBackend()
const to_native = StridedNative()
const to_blas = StridedBLAS()

# Poison `C` with NaN whenever `β == 0`, exactly as tblis.jl does, so that a
# kernel that computes `0 * C` instead of ignoring `C` entirely is caught.
poison!(C) = fill!(C, convert(eltype(C), NaN))

const eltypes = (Float32, Float64)

# `pA, pB, pAB` for the plain matmul C[i,j] = A[i,k]*B[k,j], used by most of
# the edge-case and hard-reject testsets below.
const _MATMUL_PAB = ((1,), (2,)), ((1,), (2,)), ((1, 2), ())

@testset "QuasiStridedBackend export" begin
    # Frozen: QuasiStridedBackend is the *only* exported name.
    @test QuasiStrided.QuasiStridedBackend === QuasiStrided.QuasiStridedBackend
    @test QuasiStridedBackend <: TensorOperations.AbstractBackend
    @test QuasiStridedBackend() isa QuasiStridedBackend
end

@testset "tensorcontract! agrees with StridedNative/StridedBLAS (eltype = $T)" for T in eltypes
    Random.seed!(1234567)

    # Same shape/permutation family as the frozen worked example in
    # docs/decisions.md (pA = ((3,1,4),(2,5))-shaped etc.), adapted to modest
    # sizes: A has 5 axes (3 open, 2 contracted), B has 4 axes (2 contracted,
    # 2 open).
    A = randn(T, (3, 20, 5, 3, 4))
    B = randn(T, (4, 6, 20, 3))
    pA = ((3, 1, 4), (2, 5))
    pB = ((3, 1), (4, 2))
    pAB = ((3, 1, 4), (5, 2))

    for conjA in (false, true), conjB in (false, true),
            (α, β) in ((one(T), zero(T)), (rand(T), zero(T)), (rand(T), rand(T)))

        Cn = randn(T, (3, 5, 3, 6, 3))
        Cb = copy(Cn)
        Cq = copy(Cn)
        if iszero(β)
            poison!(Cn)
            poison!(Cb)
            poison!(Cq)
        end

        Rn = tensorcontract!(Cn, A, pA, conjA, B, pB, conjB, pAB, α, β, to_native)
        Rb = tensorcontract!(Cb, A, pA, conjA, B, pB, conjB, pAB, α, β, to_blas)
        Rq = tensorcontract!(Cq, A, pA, conjA, B, pB, conjB, pAB, α, β, qsbackend)

        @test all(isfinite, Rq)
        @test Rq ≈ Rn
        @test Rq ≈ Rb
    end
end

@testset "tensorcontract!: another permutation shape (eltype = $T)" for T in eltypes
    Random.seed!(7654321)

    # A different open/contracted axis ordering than the worked example, to
    # avoid pinning the adapter to a single label-mapping accident.
    A = randn(T, (4, 5, 6))
    B = randn(T, (3, 5, 6))
    pA = ((1,), (2, 3))
    pB = ((2, 3), (1,))
    pAB = ((2, 1), ())

    for (α, β) in ((one(T), zero(T)), (rand(T), rand(T)))
        Cn = randn(T, (3, 4))
        Cq = copy(Cn)
        if iszero(β)
            poison!(Cn)
            poison!(Cq)
        end
        Rn = tensorcontract!(Cn, A, pA, false, B, pB, false, pAB, α, β, to_native)
        Rq = tensorcontract!(Cq, A, pA, false, B, pB, false, pAB, α, β, qsbackend)
        @test all(isfinite, Rq)
        @test Rq ≈ Rn
    end
end

@testset "conj is a no-op for real eltype (eltype = $T)" for T in eltypes
    # LOAD-BEARING INVARIANT pin (docs/decisions.md): QuasiStrided ignores
    # conjA/conjB entirely, which is correct only for real eltypes. Real
    # operands with conjA/conjB set true must still agree with
    # StridedNative(), which itself treats conj as a no-op on reals.
    Random.seed!(2468)
    A = randn(T, (3, 4))
    B = randn(T, (4, 5))
    pA, pB, pAB = _MATMUL_PAB

    for conjA in (false, true), conjB in (false, true)
        Cn = randn(T, (3, 5))
        Cq = copy(Cn)
        Rn = tensorcontract!(Cn, A, pA, conjA, B, pB, conjB, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, A, pA, conjA, B, pB, conjB, pAB, one(T), zero(T), qsbackend)
        @test Rq ≈ Rn
    end
end

@testset "tensorcontract! edge cases (eltype = $T)" for T in eltypes
    Random.seed!(13579)

    @testset "outer product (no contracted labels)" begin
        A = randn(T, (3, 4))
        B = randn(T, (5,))
        pA, pB, pAB = ((1, 2), ()), ((), (1,)), ((3, 1), (2,))
        Cn = fill(convert(T, NaN), (5, 3, 4))
        Cq = copy(Cn)
        Rn = tensorcontract!(Cn, A, pA, true, B, pB, false, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, A, pA, true, B, pB, false, pAB, one(T), zero(T), qsbackend)
        @test all(isfinite, Rq)
        @test Rq ≈ Rn
        # Pinning test (docs/decisions.md, "Index-model coverage"): an empty
        # contracted-index group (pA[2] and pB[1] both `()`) must still
        # produce the full outer-product shape, i.e. `axis_length` of an
        # empty `AxisGroup` is 1, not 0.
        @test size(Rq) == (5, 3, 4)
    end

    @testset "full contraction (0-dimensional output)" begin
        A = randn(T, (3, 4))
        B = randn(T, (4, 3))
        pA, pB, pAB = ((), (1, 2)), ((2, 1), ()), ((), ())
        Cn = fill(convert(T, NaN))
        Cq = fill(convert(T, NaN))
        Rn = tensorcontract!(Cn, A, pA, false, B, pB, false, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)
        @test isfinite(Rq[])
        @test Rq[] ≈ Rn[]
        # Pinning test: an empty *open*-index group (both pAB tuples `()`)
        # collapses to a 0-dimensional output, i.e. `size` is `()`, not
        # erroring or producing a length-0 array.
        @test size(Rq) == ()
        @test ndims(Rq) == 0
    end

    @testset "non-contiguous (view) inputs" begin
        Av = view(randn(T, (6, 8)), 1:2:6, 1:2:8)
        Bv = view(randn(T, (8, 10)), 1:2:8, 1:2:10)
        pA, pB, pAB = _MATMUL_PAB
        Cn = fill(convert(T, NaN), (3, 5))
        Cq = copy(Cn)
        Rn = tensorcontract!(Cn, Av, pA, false, Bv, pB, true, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, Av, pA, false, Bv, pB, true, pAB, one(T), zero(T), qsbackend)
        @test all(isfinite, Rq)
        @test Rq ≈ Rn
    end

    @testset "non-contiguous StridedView-wrapped inputs" begin
        # Wrap the strided views explicitly via StridedViews, as a real user
        # constructing a StridedView directly would (per the "StridedView"
        # dependency use documented in docs/decisions.md), rather than
        # relying only on Base.SubArray.
        Afull = randn(T, (6, 8))
        Bfull = randn(T, (8, 10))
        Av = StridedView(Afull)[1:2:6, 1:2:8]
        Bv = StridedView(Bfull)[1:2:8, 1:2:10]
        pA, pB, pAB = _MATMUL_PAB
        Cn = fill(convert(T, NaN), (3, 5))
        Cq = copy(Cn)
        Rn = tensorcontract!(Cn, Av, pA, false, Bv, pB, false, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, Av, pA, false, Bv, pB, false, pAB, one(T), zero(T), qsbackend)
        @test all(isfinite, Rq)
        @test Rq ≈ Rn
    end
end

@testset "@tensor / ncon integration (eltype = $T)" for T in eltypes
    Random.seed!(112233)
    A = randn(T, (5, 5, 5, 5))
    B = randn(T, (5, 5, 5))
    C = randn(T, (5, 5, 5))

    @tensor backend = qsbackend D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    @tensor Dref[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    @test D ≈ Dref

    network = [[-1, 1, -3, 2], [3, -4, 1], [3, 2, -2]]
    @test ncon([A, B, C], network; backend = qsbackend) ≈ ncon([A, B, C], network)
end

@testset "hard-reject: ComplexF64 input" begin
    A = randn(ComplexF64, (3, 4))
    B = randn(ComplexF64, (4, 5))
    C = zeros(ComplexF64, (3, 5))
    pA, pB, pAB = _MATMUL_PAB
    @test_throws ArgumentError tensorcontract!(
        C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
    )
end

@testset "hard-reject: mixed eltype (Float32 A, Float64 B)" begin
    A = randn(Float32, (3, 4))
    B = randn(Float64, (4, 5))
    C = zeros(Float64, (3, 5))
    pA, pB, pAB = _MATMUL_PAB
    @test_throws ArgumentError tensorcontract!(
        C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
    )
end

@testset "hard-reject: Float16 input" begin
    # Per the frozen eligibility predicate, only Float32/Float64 are
    # eligible; every other eltype (real or complex) must be rejected the
    # same way, not just ComplexF64.
    A = randn(Float16, (3, 4))
    B = randn(Float16, (4, 5))
    C = zeros(Float16, (3, 5))
    pA, pB, pAB = _MATMUL_PAB
    @test_throws ArgumentError tensorcontract!(
        C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
    )
end

@testset "hard-reject: non-strided operand (Diagonal)" begin
    # `Diagonal` is not strided (StridedViews.isstrided returns false for
    # it), so it must be rejected outright rather than silently materialized
    # or routed elsewhere.
    A = Diagonal(randn(Float64, 4))
    @test !isstrided(A)
    B = randn(Float64, (4, 5))
    C = zeros(Float64, (4, 5))
    pA, pB, pAB = _MATMUL_PAB
    @test_throws ArgumentError tensorcontract!(
        C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
    )

    # Same check with the non-strided operand as B instead of A.
    B2 = Diagonal(randn(Float64, 5))
    A2 = randn(Float64, (4, 5))
    C2 = zeros(Float64, (4, 5))
    @test_throws ArgumentError tensorcontract!(
        C2, A2, pA, false, B2, pB, false, pAB, 1, 0, qsbackend
    )
end

@testset "hard-reject: aliasing between C and an input" begin
    # Frozen argument-checking order, step 4: `plan_contract`/`execute!` have
    # no aliasing check whatsoever, so the adapter itself must catch this
    # before touching the engine.
    pA, pB, pAB = _MATMUL_PAB

    @testset "C is the exact same array as A" begin
        A = randn(Float64, (4, 4))
        B = randn(Float64, (4, 4))
        @test_throws ArgumentError tensorcontract!(
            A, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end

    @testset "C is the exact same array as B" begin
        A = randn(Float64, (4, 4))
        B = randn(Float64, (4, 4))
        @test_throws ArgumentError tensorcontract!(
            B, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end

    @testset "C overlaps A via an aliasing view" begin
        M = randn(Float64, (8, 4))
        A = view(M, 1:4, :)
        Cv = view(M, 3:6, :)  # overlaps rows 3:4 of A
        B = randn(Float64, (4, 4))
        @test Base.mightalias(Cv, A)
        @test_throws ArgumentError tensorcontract!(
            Cv, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end

    @testset "C is a PermutedDimsArray of A" begin
        # Regression: `Base.mightalias` on the *raw* arrays misses this case
        # entirely (Base defines no `Base.dataids` for `PermutedDimsArray`),
        # so the adapter must test aliasing on the `StridedView`-wrapped
        # operands, which unwrap to the shared parent.
        P = randn(Float64, (5, 5))
        B = randn(Float64, (5, 5))
        Cpd = PermutedDimsArray(P, (2, 1))
        @test !Base.mightalias(Cpd, P)                              # the Base gap
        @test Base.mightalias(StridedView(Cpd), StridedView(P))     # what closes it
        @test_throws ArgumentError tensorcontract!(
            Cpd, P, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )

        # Same gap with the aliased input in the `B` slot.
        A2 = randn(Float64, (5, 5))
        @test_throws ArgumentError tensorcontract!(
            Cpd, A2, pA, false, P, pB, false, pAB, 1, 0, qsbackend
        )
    end
end

@testset "hard-reject: tensoradd! always throws" begin
    A = randn(Float64, (3, 4))
    @test_throws ArgumentError begin
        @tensor backend = qsbackend C[i, j] := A[i, j]
    end
end

@testset "hard-reject: tensortrace! always throws" begin
    A = randn(Float64, (4, 3, 3))
    @test_throws ArgumentError begin
        @tensor backend = qsbackend C[i] := A[i, j, j]
    end
end

# =====================================================================
# Workspace-pooling tests. Unlike the sections above these are written WITH
# knowledge of src/tensoroperations.jl, since they pin the task-local pooling
# behavior itself rather than externally observable `@tensor` results.
# =====================================================================

# Run `f` with a cleared pool slot on the *current* task, restoring whatever
# was there afterwards. Not run in a spawned `Task`: `@test` finds its
# enclosing `@testset` through `task_local_storage` too.
function _qs_with_clean_pool(f)
    key = QuasiStrided._QS_WORKSPACE_KEY
    tls = task_local_storage()
    had = haskey(tls, key)
    prior = had ? tls[key] : nothing
    delete!(tls, key)
    try
        return f()
    finally
        if had
            tls[key] = prior
        else
            delete!(tls, key)
        end
    end
end

@testset "workspace pooling: default path reuses a task-local ContractWorkspace" begin
    Random.seed!(90210)
    T = Float64
    A = randn(T, (20, 30))
    Bm = randn(T, (30, 25))
    run_once() = begin
        C = zeros(T, (20, 25))
        @tensor backend = qsbackend C[i, j] = A[i, k] * Bm[k, j]
        C
    end

    steady2 = _qs_with_clean_pool() do
        @test !haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)

        C1 = run_once()
        @test C1 ≈ A * Bm
        pool = task_local_storage(QuasiStrided._QS_WORKSPACE_KEY)
        @test pool isa Dict{DataType, QuasiStrided.ContractWorkspace}
        @test haskey(pool, T)
        ws = pool[T]

        # Steady-state allocation is far below a cold, unpooled workspace
        # build for the same shape/kernel (measured just below, outside the
        # pool-cleaning block since it must NOT populate the pool); the
        # SIMDKernel accumulator is not kept register-resident by Julia
        # 1.10's compiler (docs/decisions.md, Amendment 2's caveat), so this
        # needs the same skip test_driver.jl/test_simd_kernel.jl use.
        steady1 = @allocated run_once()
        steady2 = @allocated run_once()
        @test pool[T] === ws  # same workspace object, not rebuilt
        @test steady1 < 20_000 skip = (VERSION < v"1.11")
        @test steady2 < 20_000 skip = (VERSION < v"1.11")
        steady2
    end

    # A cold, unpooled `plan_contract` call for the same shape/kernel
    # allocates its own fresh `ContractWorkspace` from scratch, well above the
    # pooled steady state above (docs/decisions.md, Amendment 1's motivation).
    Cv, Av, Bv = StridedView(zeros(Float64, (20, 25))), StridedView(A), StridedView(Bm)
    cold_allocs = @allocated plan_contract(Cv, Av, (1, -1), Bv, (-1, 2), (1, 2); oracle = false)
    @test cold_allocs > steady2 skip = (VERSION < v"1.11")
    @test cold_allocs - steady2 > 5_000 skip = (VERSION < v"1.11")
end

@testset "workspace pooling: explicit allocators match the default path and never touch the pool" begin
    Random.seed!(13571113)
    T = Float64
    A = randn(T, (20, 30))
    Bm = randn(T, (30, 25))
    Cref = A * Bm

    _qs_with_clean_pool() do
        @test !haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)

        # ManualAllocator path.
        for _ in 1:5
            C = zeros(T, (20, 25))
            @tensor backend = qsbackend allocator = TensorOperations.ManualAllocator() C[i, j] = A[i, k] * Bm[k, j]
            @test C ≈ Cref
        end
        @test !haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)

        # BufferAllocator path.
        for _ in 1:5
            C = zeros(T, (20, 25))
            @tensor backend = qsbackend allocator = TensorOperations.BufferAllocator() C[i, j] = A[i, k] * Bm[k, j]
            @test C ≈ Cref
        end
        @test !haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)

        # Bumper-backed buffer path.
        for _ in 1:5
            C = zeros(T, (20, 25))
            @no_escape begin
                @tensor backend = qsbackend allocator = default_buffer() C[i, j] = A[i, k] * Bm[k, j]
            end
            @test C ≈ Cref
        end
        @test !haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)

        # The default path, exercised afterwards, still works and now (and
        # only now) populates the pool -- confirming the explicit-allocator
        # calls above really did leave it untouched rather than this task
        # simply never reaching the pooling code at all.
        C = zeros(T, (20, 25))
        @tensor backend = qsbackend C[i, j] = A[i, k] * Bm[k, j]
        @test C ≈ Cref
        @test haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)
    end
end

# Workspace reuse across shapes *through the backend path* specifically: the
# task-local pool must not corrupt results when one pooled workspace is grown
# and then reused oversized by a smaller problem. (test_driver.jl covers
# reserve!/growth at the plan_contract level directly.)
@testset "workspace pooling: correctness across differently-shaped contractions on one task" begin
    Random.seed!(2024)
    T = Float64
    # Deliberately not monotonically increasing: includes shapes both larger
    # and smaller than earlier ones in the sequence, and matrices/tensors of
    # varying rank via contracted networks, so the pooled workspace's
    # buffers must both grow (reserve!) and be safely reused while
    # oversized.
    shapes = (
        ((6, 8), (8, 5)),
        ((37, 41), (41, 29)),
        ((3, 3), (3, 3)),
        ((50, 4), (4, 61)),
        ((10, 10), (10, 10)),
        ((17, 90), (90, 2)),
    )

    _qs_with_clean_pool() do
        @test !haskey(task_local_storage(), QuasiStrided._QS_WORKSPACE_KEY)
        for (dimsA, dimsB) in shapes
            A = randn(T, dimsA)
            Bm = randn(T, dimsB)
            Cref = A * Bm
            C = zeros(T, (dimsA[1], dimsB[2]))
            @tensor backend = qsbackend C[i, j] = A[i, k] * Bm[k, j]
            @test C ≈ Cref
        end
        pool = task_local_storage(QuasiStrided._QS_WORKSPACE_KEY)
        @test haskey(pool, T)

        # A three-tensor network (distinct from a plain matmul) run last,
        # reusing the same by-now-grown workspace, must still be correct.
        A3 = randn(T, (6, 7, 8))
        B3 = randn(T, (8, 9))
        C3 = randn(T, (9, 5, 7))
        @tensor backend = qsbackend D[c, a] := A3[a, k, l] * B3[l, m] * C3[m, c, k]
        @tensor Dref[c, a] := A3[a, k, l] * B3[l, m] * C3[m, c, k]
        @test D ≈ Dref
    end
end

@testset "workspace pooling: correctness across eltypes sharing one task-local pool" begin
    # `_qs_workspace_pool` keys its `Dict` by `DataType`; alternating eltypes
    # on the same task must not let a Float32 workspace's buffers leak into
    # a Float64 contraction or vice versa.
    Random.seed!(4048)
    _qs_with_clean_pool() do
        for T in (Float32, Float64, Float32, Float64, Float32)
            A = randn(T, (12, 7))
            Bm = randn(T, (7, 9))
            Cref = A * Bm
            C = zeros(T, (12, 9))
            @tensor backend = qsbackend C[i, j] = A[i, k] * Bm[k, j]
            @test eltype(C) === T
            @test C ≈ Cref
        end
        pool = task_local_storage(QuasiStrided._QS_WORKSPACE_KEY)
        @test haskey(pool, Float32)
        @test haskey(pool, Float64)
        @test pool[Float32] isa QuasiStrided.ContractWorkspace{Float32, Vector{Float32}}
        @test pool[Float64] isa QuasiStrided.ContractWorkspace{Float64, Vector{Float64}}
    end
end
