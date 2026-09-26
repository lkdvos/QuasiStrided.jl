# execute_direct!: the opt-in unpacked scalar path (src/execution/direct.jl).
# Every case is checked against BOTH execute! and execute_tilewise! (each run
# on its own freshly built plan and operands, so no path can see another's
# output or buffers) and, where the operands are plain arrays, against an
# independent label-driven reference loop. Comparisons are `≈`, never `==`:
# the three paths accumulate K in different orders.

# Independent reference: C[indC] = alpha * sum_K opA(A[indA]) * opB(B[indB]) +
# beta * Cstart[indC], by brute force over every label assignment. Knows
# nothing about AxisGroups, strides or plans.
function _direct_ref(
        A::AbstractArray, indA, B::AbstractArray, indB, Cstart::AbstractArray, indC,
        alpha, beta; conjA::Bool = false, conjB::Bool = false
    )
    dims = Dict{Int, Int}()
    for (lbl, L) in zip(indA, size(A))
        dims[lbl] = L
    end
    for (lbl, L) in zip(indB, size(B))
        dims[lbl] = L
    end
    klabels = Tuple(l for l in indA if l in indB && !(l in indC))
    ksizes = Tuple(dims[l] for l in klabels)
    T = eltype(Cstart)
    out = similar(Cstart, T)
    for Ic in CartesianIndices(Cstart)
        at = Dict{Int, Int}(zip(indC, Tuple(Ic)))
        acc = zero(T)
        for Ik in CartesianIndices(ksizes)
            for (l, i) in zip(klabels, Tuple(Ik))
                at[l] = i
            end
            a = A[(at[l] for l in indA)...]
            b = B[(at[l] for l in indB)...]
            acc += (conjA ? conj(a) : a) * (conjB ? conj(b) : b)
        end
        out[Ic] = iszero(beta) ? alpha * acc : alpha * acc + beta * Cstart[Ic]
    end
    return out
end

# Run each of execute!/execute_tilewise!/execute_direct! on its own fresh
# fixture from `mk()` (which must return identical contents on every call) and
# return the three resulting C arrays. `mk()` returns
# (Cv, Av, indA, Bv, indB, indC); `plankw` goes to plan_contract verbatim.
function _direct_three_way(mk, alpha, beta; plankw...)
    runners = (execute!, execute_tilewise!, execute_direct!)
    return map(runners) do run!
        Cv, Av, indA, Bv, indB, indC = mk()
        plan = plan_contract(Cv, Av, indA, Bv, indB, indC; plankw...)
        ret = run!(plan, alpha, beta)
        @test ret === plan.Cstorage
        Array(Cv)
    end
end

# Plain column-major matmul fixture C[m,n] = A[m,k] * B[k,n], seeded so every
# call reproduces the same data. `Cfill === nothing` gives a random start C.
function _direct_mm_maker(::Type{T}, Ma, Ka, Na, seed; Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        Amat = randn(rng, T, Ma, Ka)
        Bmat = randn(rng, T, Ka, Na)
        Cmat = Cfill === nothing ? randn(rng, T, Ma, Na) : fill(convert(T, Cfill), Ma, Na)
        return (StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3))
    end
end

const _DIRECT_TYPES = (Float64, ComplexF64)
# (Ma, Ka, Na): tiny, sub-register-tile, around one tile, and several tiles
# (default Float64 kernel is at least 8x6-ish and ComplexF64's 16x6-ish on
# AVX2/AVX-512, but nothing here depends on the exact shape).
const _DIRECT_SHAPES = (
    (1, 1, 1), (1, 3, 5), (5, 3, 1), (3, 1, 2), (2, 7, 3), (7, 5, 4),
    (8, 6, 6), (9, 13, 7), (17, 11, 13), (33, 40, 29), (63, 63, 2),
)

@testset "execute_direct!: contiguous matmul shapes vs execute!/execute_tilewise!/reference ($T)" for T in _DIRECT_TYPES
    ab = T <: Complex ? ((1.0, 0.0), (1.5 - 0.5im, 0.75 + 0.25im), (2.0, 1.0), (-0.5, -1.25)) :
        ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0), (-0.5, 1.25))
    for (idx, (Ma, Ka, Na)) in enumerate(_DIRECT_SHAPES), (alpha, beta) in ab
        mk = _direct_mm_maker(T, Ma, Ka, Na, 5000 + idx)
        C_exec, C_tw, C_dir = _direct_three_way(mk, alpha, beta)
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
        Cv0, Av0, _, Bv0, _, _ = mk()
        @test C_dir ≈ _direct_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), alpha, beta)
    end
end

@testset "execute_direct!: M and N below one register tile of the plan's kernel ($T)" for T in _DIRECT_TYPES
    for (Ma, Ka, Na) in ((1, 4, 1), (3, 5, 2), (1, 9, 5), (5, 2, 1))
        mk = _direct_mm_maker(T, Ma, Ka, Na, 77 + Ma + 10Na)
        Cv, Av, indA, Bv, indB, indC = mk()
        plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
        # The premise of this testset: the output is smaller than one
        # register tile in both directions, whichever roles the plan chose.
        @test axis_length(plan.mgroup) < mr(plan.kernel)
        @test axis_length(plan.ngroup) < nr(plan.kernel)
        C_exec, C_tw, C_dir = _direct_three_way(mk, 1.25, 0.5)
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
    end
end

@testset "execute_direct!: ignores kernel shape and works without oracle buffers" begin
    Random.seed!(424242)
    Ma, Ka, Na = 11, 13, 7
    Amat, Bmat, Cstart = randn(Ma, Ka), randn(Ka, Na), randn(Ma, Na)
    expected = 2.0 .* (Amat * Bmat) .+ 0.5 .* Cstart
    for kw in (
            (kernel = ScalarKernel(Val(4), Val(3), Float64), kc = 5),
            (kernel = SIMDKernel(Val(8), Val(6), Float64),),
            (oracle = false,),
        )
        Cmat = copy(Cstart)
        plan = _mm_plan(Cmat, Amat, Bmat; kw...)
        # The direct path must not touch the packing workspace at all.
        # `isequal`, not `==`: the buffers are `undef`, so their (reused-heap)
        # contents may hold NaN bit patterns, and `NaN == NaN` is false.
        pa, pb = copy(plan.workspace.packed_a), copy(plan.workspace.packed_b)
        execute_direct!(plan, 2.0, 0.5)
        @test Cmat ≈ expected
        @test isequal(plan.workspace.packed_a, pa)
        @test isequal(plan.workspace.packed_b, pb)
    end
end

@testset "execute_direct!: K = 0 applies beta once, never reads A/B ($T)" for T in _DIRECT_TYPES
    Ma, Na = 5, 4
    for beta in (0.5, 1.0, 0.0, -2.0)
        Cstart = randn(MersenneTwister(31), T, Ma, Na)
        mk = () -> (
            StridedView(copy(Cstart)), StridedView(fill(convert(T, NaN), Ma, 0)), (1, 2),
            StridedView(fill(convert(T, Inf), 0, Na)), (2, 3), (1, 3),
        )
        C_exec, C_tw, C_dir = _direct_three_way(mk, 1.0, beta)
        @test C_dir ≈ beta .* Cstart
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
        @test all(isfinite, C_dir)
    end
end

@testset "execute_direct!: alpha = 0 applies beta once, never reads A/B ($T)" for T in _DIRECT_TYPES
    Ma, Ka, Na = 5, 6, 4
    for beta in (1.25, 1.0, 0.0)
        Cstart = randn(MersenneTwister(32), T, Ma, Na)
        mk = () -> (
            StridedView(copy(Cstart)), StridedView(fill(convert(T, NaN), Ma, Ka)), (1, 2),
            StridedView(fill(convert(T, Inf), Ka, Na)), (2, 3), (1, 3),
        )
        C_exec, C_tw, C_dir = _direct_three_way(mk, 0.0, beta)
        @test C_dir ≈ beta .* Cstart
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
        @test all(isfinite, C_dir)
    end
end

@testset "execute_direct!: beta = 0 never reads old C, like execute! ($T)" for T in _DIRECT_TYPES
    # The engine's convention (src/microkernels/interface.jl `_axpby_tile!`,
    # `scale_tile!`): `beta == 0` overwrites without loading, so a NaN-poisoned
    # C must not leak into the result on any of the three paths.
    for (Ma, Ka, Na) in ((1, 1, 1), (7, 5, 4), (17, 11, 13))
        mk = _direct_mm_maker(T, Ma, Ka, Na, 900 + Ma; Cfill = NaN)
        C_exec, C_tw, C_dir = _direct_three_way(mk, 1.5, 0.0)
        @test all(isfinite, C_dir)
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
    end
    # ... and the alpha = 0 / beta = 0 beta-only pass writes exact zeros.
    Cv, Av, indA, Bv, indB, indC = _direct_mm_maker(T, 4, 3, 5, 901; Cfill = NaN)()
    execute_direct!(plan_contract(Cv, Av, indA, Bv, indB, indC), 0.0, 0.0)
    @test all(iszero, Array(Cv))
end

@testset "execute_direct!: empty output is a no-op ($T)" for T in _DIRECT_TYPES
    Cmat = fill(convert(T, NaN), 3, 0)
    plan = plan_contract(
        StridedView(Cmat), StridedView(randn(T, 3, 4)), (1, 2),
        StridedView(randn(T, 4, 0)), (2, 3), (1, 3)
    )
    @test execute_direct!(plan, 1.0, 2.0) === plan.Cstorage
    @test size(Cmat) == (3, 0)

    Cmat2 = fill(convert(T, NaN), 0, 3)
    plan2 = plan_contract(
        StridedView(Cmat2), StridedView(randn(T, 0, 4)), (1, 2),
        StridedView(randn(T, 4, 3)), (2, 3), (1, 3)
    )
    @test execute_direct!(plan2, 1.0, 2.0) === plan2.Cstorage
    @test size(Cmat2) == (0, 3)
end

@testset "execute_direct!: worked fixture, plain and permuted views" begin
    A, B, Cref = _worked_fixture()
    C = zeros(Float64, 3, 4, 2)
    plan = plan_contract(StridedView(C), StridedView(A), _INDA, StridedView(B), _INDB, _INDC)
    execute_direct!(plan, 1.0, 0.0)
    @test C ≈ Cref

    # Mirrors test_execute.jl's permuted-view case: Ap[k,b,a] == A[a,k,b],
    # Bp[n,k] == B[k,n].
    Ap = permutedims(A, (2, 3, 1))
    Bp = permutedims(B, (2, 1))
    C2 = zeros(Float64, 3, 4, 2)
    plan2 = plan_contract(StridedView(C2), StridedView(Ap), (2, 3, 1), StridedView(Bp), (4, 2), _INDC)
    execute_direct!(plan2, 1.0, 0.0)
    @test C2 ≈ Cref

    # Sliced (nonzero-offset) A and C, mirroring the sliced-destination case.
    A3 = reshape(collect(1.0:40.0), 4, 5, 2)
    Cfull = zeros(Float64, 4, 4, 2)
    Av = StridedView(view(A3, 2:4, 1:5, 1:2))
    Cv = StridedView(view(Cfull, 2:4, :, :))
    @test offset(Av) != 0 && offset(Cv) != 0
    plan3 = plan_contract(Cv, Av, _INDA, StridedView(B), _INDB, _INDC)
    execute_direct!(plan3, 1.0, 0.0)
    Cref3 = [sum(A3[a, k, b] * B[k, n] for k in 1:5) for a in 2:4, n in 1:4, b in 1:2]
    @test Array(Cv) ≈ Cref3
    @test all(iszero, Cfull[1, :, :])  # row outside the slice untouched
end

@testset "execute_direct!: permuted A, negative-stride B, sliced-offset C ($T)" for T in _DIRECT_TYPES
    # helpers.jl's `scattered_fixture` (benchmark/harness.jl's `build_scattered`
    # shape): A[a,k,b] with a zero stride on b, permuted to (k,b,a); B[k,n]
    # with a negative k stride; C[a,n,b] a strided slice of a padded array.
    sentinel = convert(T, 7)
    for (a_n, k_n, b_n, n_n) in ((32, 32, 8, 32), (3, 5, 2, 4), (1, 1, 1, 1), (9, 2, 3, 1))
        for (alpha, beta) in ((1.0, 0.0), (-1.5, 0.5))
            Cstart = randn(MersenneTwister(40 + a_n), T, a_n, n_n, b_n)
            mk = function ()
                Cv, Av, indA, Bv, indB, indC = scattered_fixture(T, a_n, k_n, b_n, n_n)
                fill!(parent(Cv), sentinel)
                copyto!(Cv, Cstart)
                return (Cv, Av, indA, Bv, indB, indC)
            end
            C_exec, C_tw, C_dir = _direct_three_way(mk, alpha, beta)
            @test C_dir ≈ C_exec
            @test C_dir ≈ C_tw
            Cv, Av, indA, Bv, indB, indC = mk()
            @test C_dir ≈ _direct_ref(Array(Av), indA, Array(Bv), indB, Cstart, indC, alpha, beta)

            # Nothing outside C's slice is written: reset the slice to the
            # sentinel after a direct run and the whole parent must be uniform.
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
            @test offset(Cv) != 0
            # (StridedViews canonicalizes a length-1 axis's stride, so the
            # negative stride only survives when k_n > 1.)
            @test any(<(0), strides(Bv)) || k_n == 1
            execute_direct!(plan, alpha, beta)
            @test Array(Cv) ≈ C_dir
            fill!(Cv, sentinel)
            @test all(==(sentinel), parent(Cv))
        end
    end
end

@testset "execute_direct!: operand-role swap (C stored N-major) ($T)" for T in _DIRECT_TYPES
    # C[n,m] column-major: the M composite is not unit-stride in C but the N
    # one is, so for a real eltype plan_contract swaps roles and
    # plan.Astorage is B's parent. The swap is real-only (plan.jl), so the
    # complex case instead covers an M composite that is strided in C.
    Ma, Ka, Na = 5, 7, 19
    mk = function ()
        rng = MersenneTwister(606)
        Amat = randn(rng, T, Ma, Ka)
        Bmat = randn(rng, T, Ka, Na)
        Cmat = randn(rng, T, Na, Ma)
        return (StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (3, 1))
    end
    Cv, Av, indA, Bv, indB, indC = mk()
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
    # Premise: the swap happened exactly when the planner says it should.
    @test plan.Astorage === (T <: Real ? parent(Bv) : parent(Av))
    for (alpha, beta) in ((1.0, 0.0), (0.5, -1.0))
        C_exec, C_tw, C_dir = _direct_three_way(mk, alpha, beta)
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
        @test C_dir ≈ _direct_ref(Array(Av), indA, Array(Bv), indB, Array(Cv), indC, alpha, beta)
    end
end

@testset "execute_direct!: conjA/conjB, flags and conj views ($T)" for T in (ComplexF64, ComplexF32)
    for (Ma, Ka, Na) in ((1, 1, 1), (3, 4, 2), (17, 9, 13)),
            conjA in (false, true), conjB in (false, true)
        alpha, beta = (0.75 - 1.0im), (0.5 + 0.5im)
        mk = _direct_mm_maker(T, Ma, Ka, Na, 3000 + Ma)
        C_exec, C_tw, C_dir = _direct_three_way(mk, alpha, beta; conjA = conjA, conjB = conjB)
        @test C_dir ≈ C_exec
        @test C_dir ≈ C_tw
        Cv, Av, _, Bv, _, _ = mk()
        @test C_dir ≈ _direct_ref(
            Array(Av), (1, 2), Array(Bv), (2, 3), Array(Cv), (1, 3), alpha, beta;
            conjA = conjA, conjB = conjB
        )
        # A conj-wrapped view composes with the flag by XOR; the plan's
        # transform must come out the same, and so must the result.
        mkv = function ()
            Cv, Av, iA, Bv, iB, iC = mk()
            return (Cv, conj(Av), iA, Bv, iB, iC)
        end
        Cvv, Avv, iA, Bvv, iB, iC = mkv()
        planv = plan_contract(Cvv, Avv, iA, Bvv, iB, iC; conjA = !conjA, conjB = conjB)
        execute_direct!(planv, alpha, beta)
        @test Array(Cvv) ≈ C_dir
    end
end

@testset "execute_direct!: repeated calls on one plan, real-typed alpha into complex" begin
    # A plan is reusable: a second call accumulates onto the first's output.
    T = ComplexF64
    mk = _direct_mm_maker(T, 6, 5, 4, 1234)
    Cv, Av, indA, Bv, indB, indC = mk()
    Cstart = Array(Cv)
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
    execute_direct!(plan, 2, 1)
    execute_direct!(plan, 2, 1)
    AB = Array(Av) * Array(Bv)
    @test Array(Cv) ≈ Cstart .+ 4 .* AB
    @test (@inferred execute_direct!(plan, 1.0, 0.0)) === plan.Cstorage
    @test Array(Cv) ≈ AB
end
