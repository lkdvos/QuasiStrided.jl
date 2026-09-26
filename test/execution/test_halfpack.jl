# execute_half_packed!: the opt-in "pack B, read a one-tile A in place" path
# (src/execution/halfpack.jl). Two properties are checked:
#
#   1. `_half_pack_a_eligible` admits exactly the plans it should: real
#      eltype, SIMDKernel, dense A storage, Qm == mr(kernel), unit-stride A rows.
#   2. `execute_half_packed!` is NEVER wrong. On an eligible plan it must match
#      execute! BITWISE (`==`): the per-K-step arithmetic is identical -- same
#      `_accumulate_step`, FMA order, K blocking and store, on the same A
#      values read from a different address -- and that is what is observed.
#      On an ineligible plan it IS execute!, so `==` holds by construction.
#      execute_tilewise! (the oracle) and a brute-force reference are
#      compared too: `==` with the oracle on eligible plans (observed as well,
#      same kernel and K panels), `≈` with the reference.
#
# Every run uses its own freshly built plan and operands, so no path can see
# another's output or buffers. On the half-packed run the plan's packed-A
# buffer is poisoned with NaN first: an eligible run must leave it untouched
# (A really is not packed) and still produce a finite, correct result.

const execute_half_packed! = QuasiStrided.execute_half_packed!
const _hp_eligible = QuasiStrided._half_pack_a_eligible

# A DenseMatrix that is not a DenseVector: StridedViews keeps any non-`Array`
# parent as-is (`_normalizeparent`), so a plan over it has `Astorage` of this
# type, which the half-packed path's dense-storage clause must reject.
struct _HPDenseMat{T} <: DenseMatrix{T}
    data::Matrix{T}
end
Base.size(a::_HPDenseMat) = size(a.data)
Base.IndexStyle(::Type{<:_HPDenseMat}) = IndexLinear()
Base.getindex(a::_HPDenseMat, i::Int) = a.data[i]
Base.setindex!(a::_HPDenseMat, v, i::Int) = (a.data[i] = v)

# The default kernel a large-M plan picks for `T` (no extent demotion).
function _hp_default_kernel(::Type{T}) where {T}
    plan = plan_contract(
        StridedView(zeros(T, 512, 8)), StridedView(zeros(T, 512, 64)), (1, 2),
        StridedView(zeros(T, 64, 8)), (2, 3), (1, 3)
    )
    return plan.kernel
end

# Brute-force reference, same as test_direct.jl's `_direct_ref` but kept
# separate so this file does not depend on another test file's helpers.
function _hp_ref(A, indA, B, indB, Cstart, indC, alpha, beta; conjA = false, conjB = false)
    dims = Dict{Int, Int}()
    for (l, L) in zip(indA, size(A))
        dims[l] = L
    end
    for (l, L) in zip(indB, size(B))
        dims[l] = L
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

# Run execute!, execute_tilewise! and execute_half_packed! on three fresh
# fixtures from `mk()` (identical contents on every call). Returns
# `(C_exec, C_tw, C_half, eligible)`, where `eligible` is
# `_half_pack_a_eligible` of the half-packed run's plan.
function _hp_three_way(mk, alpha, beta; plankw...)
    out = Any[]
    eligible = false
    for run! in (execute!, execute_tilewise!, execute_half_packed!)
        Cv, Av, indA, Bv, indB, indC = mk()
        plan = plan_contract(Cv, Av, indA, Bv, indB, indC; plankw...)
        if run! === execute_half_packed!
            eligible = _hp_eligible(plan)
            fill!(plan.workspace.packed_a, NaN)
        end
        ret = run!(plan, alpha, beta)
        @test ret === plan.Cstorage
        if run! === execute_half_packed! && eligible
            # A was read in place, never packed.
            @test all(isnan, plan.workspace.packed_a)
        end
        push!(out, Array(Cv))
    end
    return (out..., eligible)
end

# Column-major matmul fixture C[m,n] = A[m,k] * B[k,n] with a choice of A
# layout. Gaps in every padded parent are NaN, so reading one poisons C.
#   :dense       plain Matrix                          (unit-stride rows)
#   :ldgap       rows 3:Ma+2 of a taller NaN-padded matrix, nonzero offset
#                                                      (unit-stride rows)
#   :transposed  a Ka-by-Ma matrix viewed transposed   (row stride Ka)
#   :rowgap      every other row of a 2Ma-row matrix   (row stride 2)
#   :wrapped     `_HPDenseMat` parent                  (non-DenseVector storage)
function _hp_mm_maker(::Type{T}, Ma, Ka, Na, seed; alayout = :dense, Cfill = nothing) where {T}
    return function ()
        rng = MersenneTwister(seed)
        Amat = randn(rng, T, Ma, Ka)
        Bmat = randn(rng, T, Ka, Na)
        Cmat = Cfill === nothing ? randn(rng, T, Ma, Na) : fill(convert(T, Cfill), Ma, Na)
        Av = if alayout === :dense
            StridedView(Amat)
        elseif alayout === :ldgap
            big = fill(convert(T, NaN), Ma + 5, Ka)
            big[3:(Ma + 2), :] .= Amat
            StridedView(view(big, 3:(Ma + 2), :))
        elseif alayout === :transposed
            permutedims(StridedView(permutedims(Amat, (2, 1))), (2, 1))
        elseif alayout === :rowgap
            big = fill(convert(T, NaN), 2Ma, Ka)
            big[1:2:(2Ma), :] .= Amat
            StridedView(view(big, 1:2:(2Ma), :))
        elseif alayout === :wrapped
            StridedView(_HPDenseMat(copy(Amat)), (Ma, Ka), (1, Ma), 0)
        else
            error("unknown alayout $alayout")
        end
        @assert Array(Av) == Amat
        return (StridedView(Cmat), Av, (1, 2), StridedView(Bmat), (2, 3), (1, 3))
    end
end

# The (alpha, beta) pairs every eligible shape runs: overwrite, general,
# accumulate, negative scale.
const _HP_AB = ((1.0, 0.0), (2.5, -0.75), (1.0, 1.0), (-0.5, 1.25))

@testset "half-pack eligibility: which plans take the in-place A path ($T)" for T in (Float64, Float32)
    kern = _hp_default_kernel(T)
    MR = mr(kern)
    planof(mk; kw...) = (f = mk(); plan_contract(f[1], f[2], f[3], f[4], f[5], f[6]; kw...))

    # Eligible: M exactly one register tile, unit-stride A rows in dense storage.
    for K in (1, 2, 7, 40), N in (1, 5, 13)
        p = planof(_hp_mm_maker(T, MR, K, N, 1))
        @test mr(p.kernel) == MR
        @test _hp_eligible(p)
        @test _hp_eligible(planof(_hp_mm_maker(T, MR, K, N, 1; alayout = :ldgap)))
    end
    # conjA/conjB on a real eltype: plan_contract folds both to `identity`
    # (src/planning/conjugation.jl, `_qs_isconj`), so the plan stays eligible.
    for conjA in (false, true), conjB in (false, true)
        p = planof(_hp_mm_maker(T, MR, 7, 5, 1); conjA = conjA, conjB = conjB)
        @test p.atransform === identity && p.btransform === identity
        @test _hp_eligible(p)
    end
    # ... including explicitly named SIMD kernels of other shapes.
    for k in (SIMDKernel(Val(8), Val(6), T), SIMDKernel(Val(2 * lanewidth(kern)), Val(3), T))
        @test _hp_eligible(planof(_hp_mm_maker(T, mr(k), 9, 7, 2); kernel = k))
        @test !_hp_eligible(planof(_hp_mm_maker(T, mr(k) + 1, 9, 7, 2); kernel = k))
    end

    # Ineligible: M one short / one over the tile (whatever kernel the planner
    # then picks, Qm != mr), non-unit-stride rows, non-DenseVector storage,
    # a ScalarKernel.
    @test !_hp_eligible(planof(_hp_mm_maker(T, MR - 1, 7, 5, 3)))
    @test !_hp_eligible(planof(_hp_mm_maker(T, MR + 1, 7, 5, 3)))
    @test !_hp_eligible(planof(_hp_mm_maker(T, MR, 7, 5, 3; alayout = :transposed)))
    @test !_hp_eligible(planof(_hp_mm_maker(T, MR, 7, 5, 3; alayout = :rowgap)))
    pw = planof(_hp_mm_maker(T, MR, 7, 5, 3; alayout = :wrapped))
    @test !(pw.Astorage isa DenseVector)
    @test !_hp_eligible(pw)
    @test !_hp_eligible(planof(_hp_mm_maker(T, 8, 7, 5, 3); kernel = ScalarKernel(Val(8), Val(4), T)))
end

@testset "half-pack eligibility: complex plans are always excluded ($T)" for T in (ComplexF64, ComplexF32)
    for conjA in (false, true), conjB in (false, true), wrap in (false, true)
        # Build once to learn the kernel's mr, then again with M == mr so that
        # the element type is the only clause that can fail.
        p0 = plan_contract(
            StridedView(zeros(T, 512, 8)), StridedView(zeros(T, 512, 8)), (1, 2),
            StridedView(zeros(T, 8, 8)), (2, 3), (1, 3)
        )
        MR = mr(p0.kernel)
        Cv, Av, iA, Bv, iB, iC = _hp_mm_maker(T, MR, 6, 5, 4)()
        p = plan_contract(Cv, wrap ? conj(Av) : Av, iA, Bv, iB, iC; conjA = conjA, conjB = conjB)
        @test axis_length(p.mgroup) == mr(p.kernel)
        @test !_hp_eligible(p)
    end
end

@testset "execute_half_packed!: eligible matmul shapes, bitwise vs execute!/oracle ($T)" for T in (Float64, Float32)
    MR = mr(_hp_default_kernel(T))
    shapes = ((1, 1), (1, 7), (5, 1), (2, 6), (7, 13), (40, 29), (63, 2), (300, 19))
    for (idx, (K, N)) in enumerate(shapes), alayout in (:dense, :ldgap), (alpha, beta) in _HP_AB
        mk = _hp_mm_maker(T, MR, K, N, 100 + idx; alayout = alayout)
        C_exec, C_tw, C_half, el = _hp_three_way(mk, alpha, beta)
        @test el
        @test C_half == C_exec
        @test C_half == C_tw
        Cv0, Av0, _, Bv0, _, _ = mk()
        @test C_half ≈ _hp_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), alpha, beta)
    end
end

@testset "execute_half_packed!: several jc/pc blocks and explicit kernels ($T)" for T in (Float64, Float32)
    kern = _hp_default_kernel(T)
    kernels = (kern, SIMDKernel(Val(8), Val(6), T), SIMDKernel(Val(2 * lanewidth(kern)), Val(3), T))
    # (kc, nc): K and N both split into several blocks, with tails.
    for k in kernels, (kc, nc) in ((3, 6), (5, 7), (64, 1), (1, 100))
        for (K, N) in ((11, 17), (4, 25), (1, 3))
            mk = _hp_mm_maker(T, mr(k), K, N, 7 * K + N)
            C_exec, C_tw, C_half, el = _hp_three_way(mk, 1.5, 0.5; kernel = k, kc = kc, nc = nc)
            @test el
            @test C_half == C_exec
            @test C_half == C_tw
            Cv0, Av0, _, Bv0, _, _ = mk()
            @test C_half ≈ _hp_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), 1.5, 0.5)
        end
    end
end

@testset "execute_half_packed!: K = 0, alpha = 0, beta = 0 never reads what it must not ($T)" for T in (Float64, Float32)
    MR = mr(_hp_default_kernel(T))
    N = 9
    # K = 0: beta applied once, A/B never read.
    for beta in (0.5, 1.0, 0.0, -2.0)
        Cstart = randn(MersenneTwister(31), T, MR, N)
        mk = () -> (
            StridedView(copy(Cstart)), StridedView(fill(convert(T, NaN), MR, 0)), (1, 2),
            StridedView(fill(convert(T, Inf), 0, N)), (2, 3), (1, 3),
        )
        C_exec, C_tw, C_half, _ = _hp_three_way(mk, 1.0, beta)
        @test C_half == C_exec
        @test C_half == C_tw
        @test C_half ≈ beta .* Cstart
    end
    # alpha = 0: beta applied once, A/B (NaN/Inf) never read.
    for beta in (1.25, 1.0, 0.0)
        Cstart = randn(MersenneTwister(32), T, MR, N)
        mk = () -> (
            StridedView(copy(Cstart)), StridedView(fill(convert(T, NaN), MR, 6)), (1, 2),
            StridedView(fill(convert(T, Inf), 6, N)), (2, 3), (1, 3),
        )
        C_exec, C_tw, C_half, el = _hp_three_way(mk, 0.0, beta)
        @test el
        @test C_half == C_exec
        @test all(isfinite, C_half)
        @test C_half ≈ beta .* Cstart
    end
    # beta = 0 into a NaN-poisoned C: old C never read.
    for (K, N2) in ((1, 1), (5, 4), (40, 13))
        mk = _hp_mm_maker(T, MR, K, N2, 900 + K; Cfill = NaN)
        C_exec, C_tw, C_half, el = _hp_three_way(mk, 1.5, 0.0)
        @test el
        @test all(isfinite, C_half)
        @test C_half == C_exec
        @test C_half == C_tw
    end
    # Empty N: a no-op.
    Cmat = fill(convert(T, NaN), MR, 0)
    plan = plan_contract(
        StridedView(Cmat), StridedView(randn(T, MR, 4)), (1, 2),
        StridedView(randn(T, 4, 0)), (2, 3), (1, 3)
    )
    @test execute_half_packed!(plan, 1.0, 2.0) === plan.Cstorage
end

@testset "execute_half_packed!: scattered K, negative-stride B, offset C, non-ramp M ($T)" for T in (Float64, Float32)
    kern = _hp_default_kernel(T)
    MR = mr(kern)

    # K composite (k1, k2) that does not fold in A (a gapped slice), so K is a
    # scattered axis on the in-place A read; B reversed along k1 (negative
    # stride); C a slice with a nonzero offset. Every gap in A's parent is NaN.
    for N in (1, 6, 23), (kc, nc) in ((64, 64), (3, 6))
        mk = function ()
            # `local`: these names are also locals of the enclosing loop, and
            # a closure would otherwise assign the outer ones.
            local Abig, Av, Bv, Cbig, Cv
            rng = MersenneTwister(50 + N)
            Abig = fill(convert(T, NaN), MR + 3, 5, 9)
            Abig[2:(MR + 1), 1:4, 2:2:9] .= randn(rng, T, MR, 4, 4)
            Av = StridedView(view(Abig, 2:(MR + 1), 1:4, 2:2:9))
            Bv = StridedView(view(randn(rng, T, 4, 4, N), 4:-1:1, :, :))
            Cbig = randn(rng, T, MR + 2, N + 3)
            Cv = StridedView(view(Cbig, 2:(MR + 1), 2:(N + 1)))
            return (Cv, Av, (1, 2, 3), Bv, (2, 3, 4), (1, 4))
        end
        Cv, Av, iA, Bv, iB, iC = mk()
        p = plan_contract(Cv, Av, iA, Bv, iB, iC; kernel = kern)
        @test !QuasiStrided.affine_ramp(p.kgroup)[1]
        @test any(<(0), strides(Bv)) && offset(Cv) != 0
        for (alpha, beta) in ((1.0, 0.0), (-1.5, 0.5))
            C_exec, C_tw, C_half, el = _hp_three_way(mk, alpha, beta; kernel = kern, kc = kc, nc = nc)
            @test el
            @test C_half == C_exec
            @test C_half == C_tw
            @test all(isfinite, C_half)
            @test C_half ≈ _hp_ref(Array(Av), iA, Array(Bv), iB, Array(Cv), iC, alpha, beta)
        end
    end

    # M composite (a1, a2) contiguous in A but gapped in C (C[a1, n, a2]):
    # not an affine ramp, so the M descriptors are materialized and C's rows
    # are a scattered axis, while A's rows are still one unit-stride run.
    for N in (2, 7)
        mk = function ()
            local Av, Bv, Cv
            rng = MersenneTwister(60 + N)
            Av = StridedView(randn(rng, T, MR ÷ 2, 2, 5))
            Bv = StridedView(randn(rng, T, 5, N))
            Cv = StridedView(randn(rng, T, MR ÷ 2, N, 2))
            return (Cv, Av, (1, 2, 3), Bv, (3, 4), (1, 4, 2))
        end
        Cv, Av, iA, Bv, iB, iC = mk()
        p = plan_contract(Cv, Av, iA, Bv, iB, iC; kernel = kern)
        @test !QuasiStrided.affine_ramp(p.mgroup)[1]
        C_exec, C_tw, C_half, el = _hp_three_way(mk, 0.75, -1.0; kernel = kern)
        @test el
        @test C_half == C_exec
        @test C_half == C_tw
        @test C_half ≈ _hp_ref(Array(Av), iA, Array(Bv), iB, Array(Cv), iC, 0.75, -1.0)
    end

    # helpers.jl's `scattered_fixture` (benchmark/harness.jl's `build_scattered`
    # shape: permuted A with a zero-stride b axis, negative-stride B, sliced
    # C) with M forced to one register tile. b_n = 1: M is `a` alone, unit
    # stride in A -> eligible. b_n = 2: M = (a, b) with b's A stride 0, so A's
    # rows repeat -> not affine -> ineligible, execute! fallback.
    sentinel = convert(T, 7)
    for (a_n, b_n, want) in ((MR, 1, true), (MR ÷ 2, 2, false)), (k_n, n_n) in ((32, 32), (3, 5), (1, 1))
        Cstart = randn(MersenneTwister(40 + a_n + k_n), T, a_n, n_n, b_n)
        mk = function ()
            local Cv, Av, indA, Bv, indB, indC
            Cv, Av, indA, Bv, indB, indC = scattered_fixture(T, a_n, k_n, b_n, n_n)
            fill!(parent(Cv), sentinel)
            copyto!(Cv, Cstart)
            return (Cv, Av, indA, Bv, indB, indC)
        end
        for (alpha, beta) in ((1.0, 0.0), (-1.5, 0.5))
            C_exec, C_tw, C_half, el = _hp_three_way(mk, alpha, beta)
            @test el == want
            @test C_half == C_exec
            want && @test C_half == C_tw
            @test C_half ≈ C_tw
            Cv, Av, indA, Bv, indB, indC = mk()
            @test C_half ≈ _hp_ref(Array(Av), indA, Array(Bv), indB, Cstart, indC, alpha, beta)
            # Nothing outside C's slice is written.
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
            execute_half_packed!(plan, alpha, beta)
            @test Array(Cv) == C_half
            fill!(Cv, sentinel)
            @test all(==(sentinel), parent(Cv))
        end
    end
end

@testset "execute_half_packed!: operand-role swap ($T)" for T in (Float64, Float32)
    # C[n,m] column-major with n of extent MR: plan_contract swaps roles, so
    # the plan's M (and its in-place "A") is the user's B. It is eligible iff
    # the user's B is unit-stride along n.
    MR = mr(_hp_default_kernel(T))
    Ma, Ka = 5, 7
    for btrans in (true, false)
        mk = function ()
            local Amat, Bmat, Cmat, Bv
            rng = MersenneTwister(606)
            Amat = randn(rng, T, Ma, Ka)
            Bmat = randn(rng, T, Ka, MR)
            Cmat = randn(rng, T, MR, Ma)
            Bv = btrans ? permutedims(StridedView(permutedims(Bmat, (2, 1))), (2, 1)) : StridedView(Bmat)
            return (StridedView(Cmat), StridedView(Amat), (1, 2), Bv, (2, 3), (3, 1))
        end
        Cv, Av, iA, Bv, iB, iC = mk()
        plan = plan_contract(Cv, Av, iA, Bv, iB, iC)
        @test plan.Astorage === parent(Bv)
        for (alpha, beta) in ((1.0, 0.0), (0.5, -1.0))
            C_exec, C_tw, C_half, el = _hp_three_way(mk, alpha, beta)
            @test el == btrans
            @test C_half == C_exec
            @test C_half ≈ C_tw
            @test C_half ≈ _hp_ref(Array(Av), iA, Array(Bv), iB, Array(Cv), iC, alpha, beta)
        end
    end
end

@testset "execute_half_packed!: ineligible plans fall back to execute! exactly ($T)" for T in (Float64, Float32)
    MR = mr(_hp_default_kernel(T))
    cases = (
        (MR - 1, 7, 5, :dense, (;)),
        (MR + 1, 7, 5, :dense, (;)),
        (2MR + 3, 9, 4, :dense, (;)),
        (MR, 7, 5, :transposed, (;)),
        (MR, 7, 5, :rowgap, (;)),
        (MR, 7, 5, :wrapped, (;)),
        (8, 7, 5, :dense, (kernel = ScalarKernel(Val(8), Val(4), T),)),
        (8, 7, 5, :dense, (kernel = ScalarKernel(Val(8), Val(4), T), kc = 3, nc = 4)),
    )
    for (idx, (M, K, N, alayout, kw)) in enumerate(cases), (alpha, beta) in _HP_AB
        mk = _hp_mm_maker(T, M, K, N, 700 + idx; alayout = alayout)
        C_exec, C_tw, C_half, el = _hp_three_way(mk, alpha, beta; kw...)
        @test !el
        @test C_half == C_exec
        @test C_half ≈ C_tw
        Cv0, Av0, _, Bv0, _, _ = mk()
        @test C_half ≈ _hp_ref(Array(Av0), (1, 2), Array(Bv0), (2, 3), Array(Cv0), (1, 3), alpha, beta)
    end
end

@testset "execute_half_packed!: complex plans fall back to execute! exactly ($T)" for T in (ComplexF64, ComplexF32)
    p0 = plan_contract(
        StridedView(zeros(T, 512, 8)), StridedView(zeros(T, 512, 8)), (1, 2),
        StridedView(zeros(T, 8, 8)), (2, 3), (1, 3)
    )
    MR = mr(p0.kernel)
    alpha, beta = (0.75 - 1.0im), (0.5 + 0.5im)
    for (M, K, N) in ((MR, 1, 1), (MR, 4, 3), (MR, 9, 13), (MR - 1, 5, 2), (3, 4, 2)),
            conjA in (false, true), conjB in (false, true)
        mk = _hp_mm_maker(T, M, K, N, 3000 + M + K)
        C_exec, C_tw, C_half, el = _hp_three_way(mk, alpha, beta; conjA = conjA, conjB = conjB)
        @test !el
        @test C_half == C_exec
        @test C_half ≈ C_tw
        Cv, Av, _, Bv, _, _ = mk()
        @test C_half ≈ _hp_ref(
            Array(Av), (1, 2), Array(Bv), (2, 3), Array(Cv), (1, 3), alpha, beta;
            conjA = conjA, conjB = conjB
        )
        # A conj-wrapped view composes with the flag by XOR.
        Cvv, Avv, iA, Bvv, iB, iC = mk()
        planv = plan_contract(Cvv, conj(Avv), iA, Bvv, iB, iC; conjA = !conjA, conjB = conjB)
        @test !_hp_eligible(planv)
        execute_half_packed!(planv, alpha, beta)
        @test Array(Cvv) == C_half
    end
end

@testset "execute_half_packed!: out-of-bounds operands are rejected like execute! ($T)" for T in (Float64, Float32)
    # StridedView does not validate its extent against the parent, and
    # plan_contract does not either; the drivers' hoisted span checks do. The
    # in-place A loads are raw-pointer reads, so the A check is load-bearing.
    MR = mr(_hp_default_kernel(T))
    K, N = 6, 5
    mkA = () -> (
        StridedView(zeros(T, MR, N)), StridedView(randn(T, MR * K), (MR, K), (1, MR), 3), (1, 2),
        StridedView(randn(T, K, N)), (2, 3), (1, 3),
    )
    mkB = () -> (
        StridedView(zeros(T, MR, N)), StridedView(randn(T, MR, K)), (1, 2),
        StridedView(randn(T, K * N), (K, N), (1, K), 1), (2, 3), (1, 3),
    )
    mkC = () -> (
        StridedView(zeros(T, MR * N), (MR, N), (1, MR), 2), StridedView(randn(T, MR, K)), (1, 2),
        StridedView(randn(T, K, N)), (2, 3), (1, 3),
    )
    for mk in (mkA, mkB, mkC), run! in (execute!, execute_half_packed!)
        Cv, Av, iA, Bv, iB, iC = mk()
        plan = plan_contract(Cv, Av, iA, Bv, iB, iC)
        @test _hp_eligible(plan)
        @test_throws BoundsError run!(plan, 1.0, 0.0)
    end
end

@testset "execute_half_packed!: allocation-free, reusable, inferred ($T)" for T in (Float64, Float32)
    kern = _hp_default_kernel(T)
    MR = mr(kern)
    # Affine K and scattered K (the `PtrScatterAxis` A view).
    Cmat = zeros(T, MR, 37)
    plan = _mm_plan(Cmat, randn(T, MR, 50), randn(T, 50, 37); kc = 16, nc = 12)
    @test _hp_eligible(plan)
    @test _steady_allocs!(execute_half_packed!, plan, Cmat) == 0

    Abig = randn(T, MR + 3, 5, 9)
    Csc = zeros(T, MR, 11)
    plan_sc = plan_contract(
        StridedView(Csc), StridedView(view(Abig, 2:(MR + 1), 1:4, 2:2:9)), (1, 2, 3),
        StridedView(randn(T, 4, 4, 11)), (2, 3, 4), (1, 4); kernel = kern, kc = 5
    )
    @test _hp_eligible(plan_sc)
    @test !QuasiStrided.affine_ramp(plan_sc.kgroup)[1]
    @test _steady_allocs!(execute_half_packed!, plan_sc, Csc) == 0

    # A plan is reusable: a second call accumulates onto the first's output.
    Amat, Bmat = randn(T, MR, 9), randn(T, 9, 8)
    Cstart = randn(T, MR, 8)
    C = copy(Cstart)
    p = _mm_plan(C, Amat, Bmat)
    @test _hp_eligible(p)
    execute_half_packed!(p, 2, 1)
    execute_half_packed!(p, 2, 1)
    @test C ≈ Cstart .+ 4 .* (Amat * Bmat)
    @test (@inferred execute_half_packed!(p, 1.0, 0.0)) === p.Cstorage
    @test C ≈ Amat * Bmat
end
