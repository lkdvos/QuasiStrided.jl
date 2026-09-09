# Independent oracle/property tests for the macro-blocking `execute!`
# (docs/decisions.md, "Macro-blocking milestone"). Nothing under test is
# exported, and test_driver.jl introduces unqualified `plan_contract`/
# `execute!` bindings into the same top-level scope, so every use here is
# written as `QuasiStrided.<name>` to avoid colliding with those.

using Test
using Random
using StridedViews: StridedView, offset

# =====================================================================
# Shared helpers
# =====================================================================

const _MACRO_KERNEL_CTORS = (ScalarKernel, SIMDKernel)
const _MACRO_SHAPES = ((Val(4), Val(3)), (Val(8), Val(6)))
const _MACRO_ELTYPES = (Float64, Float32)

# Plan for the dense matmul C[m,n] = sum_k A[m,k]*B[k,n], the shape most
# testsets below use.
function _dense_plan(Cmat, Amat, Bmat, kernel, mc, kc, nc)
    return QuasiStrided.plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        kernel = kernel, mc = mc, kc = kc, nc = nc
    )
end

"""
    _random_macro_case(rng, ctor, shape, T)

One randomized dense-matmul case: shapes in 1:37 against block sizes in 1:13
(small on purpose, to force multiple blocks in every dimension), plus random
alpha/beta and a random starting C.
"""
function _random_macro_case(rng::MersenneTwister, ctor, shape, ::Type{T}) where {T}
    Ma = rand(rng, 1:37)
    Ka = rand(rng, 1:37)
    Na = rand(rng, 1:37)
    mc = rand(rng, 1:13)
    kc = rand(rng, 1:13)
    nc = rand(rng, 1:13)
    kernel = ctor(shape[1], shape[2], T)
    alpha = T(rand(rng, -3.0:0.5:3.0))
    beta = T(rand(rng, -3.0:0.5:3.0))
    Amat = randn(rng, T, Ma, Ka)
    Bmat = randn(rng, T, Ka, Na)
    Cstart = randn(rng, T, Ma, Na)
    return (; Ma, Ka, Na, mc, kc, nc, kernel, alpha, beta, Amat, Bmat, Cstart, T)
end

# SIMDKernel needs mr(kernel) to be a multiple of its type-dependent lane
# width (4 for Float64, 8 for Float32), so (Val(4),Val(3)) is not a
# constructible SIMDKernel/Float32 combo.
_valtype(::Val{N}) where {N} = N
function _valid_shapes(ctor, ::Type{T}) where {T}
    ctor !== SIMDKernel && return _MACRO_SHAPES
    return Tuple(s for s in _MACRO_SHAPES if _valtype(s[1]) % QuasiStrided._default_lanewidth(T) == 0)
end

# Fixed-seed case stream, so testsets 1 and 6 run the *same* cases.
function _macro_random_cases(seed::Integer; per_combo::Int = 5)
    rng = MersenneTwister(seed)
    cases = Any[]
    for ctor in _MACRO_KERNEL_CTORS, T in _MACRO_ELTYPES, shape in _valid_shapes(ctor, T)
        for _ in 1:per_combo
            push!(cases, _random_macro_case(rng, ctor, shape, T))
        end
    end
    return cases
end

# Both operands and the M/N/K block partition change how the Ka terms of each
# output entry are summed, so agreement is only up to summation-reordering
# rounding: a generous multiple of the textbook Ka*eps(T) bound.
_macro_rtol(::Type{T}, Ka::Integer) where {T} = 50 * max(Ka, 1) * eps(T)

# =====================================================================
# 1. Randomized agreement vs. dense matmul
# =====================================================================

@testset "macro driver: randomized agreement vs. dense matmul" begin
    for case in _macro_random_cases(0x5A17_D817)
        (; Ka, mc, kc, nc, kernel, alpha, beta, Amat, Bmat, Cstart, T) = case
        Cmat = copy(Cstart)
        plan = _dense_plan(Cmat, Amat, Bmat, kernel, mc, kc, nc)
        QuasiStrided.execute!(plan, alpha, beta)

        expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
        @test isapprox(Cmat, expected; rtol = _macro_rtol(T, Ka))
    end
end

# =====================================================================
# 2. 3-index tensor fixture with real StridedViews: permuted A, zero-stride
# broadcast, negative-stride B, sliced (nonzero-offset) C. The reference is
# built by plain nested loops over the views, never via the driver.
# =====================================================================

@testset "macro driver: 3-index StridedViews fixture (permuted/sliced/negative/zero-stride)" begin
    a_n, k_n, b_n, n_n = 7, 11, 5, 9

    # A[a,k,b]: genuinely independent of b (zero stride there), presented as a
    # permuted (k,b,a) view. Labels: a=1, k=2, b=3, n=4.
    A2 = randn(a_n, k_n)
    Araw = StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0)
    Aperm = permutedims(Araw, (2, 3, 1))
    indA = (2, 3, 1)

    # B[k,n]: reversed-row view (negative stride along k).
    Bdata = randn(k_n * n_n)
    Bneg = StridedView(Bdata, (k_n, n_n), (-1, k_n), k_n - 1)
    indB = (2, 4)

    # C[a,n,b]: sliced out of a larger array, nonzero base offset, nonzero
    # starting value (so beta is exercised on this fixture too).
    Cbig = randn(a_n + 2, n_n + 3, b_n + 1)
    Csub = view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n)
    Cv = StridedView(Csub)
    indC = (1, 4, 3)
    @test offset(Cv) != 0

    Cstart = copy(Csub)

    alpha, beta = 1.3, 0.6
    Cref = zeros(a_n, n_n, b_n)
    for bb in 1:b_n, nn in 1:n_n, aa in 1:a_n
        s = 0.0
        for kk in 1:k_n
            s += Aperm[kk, bb, aa] * Bneg[kk, nn]
        end
        Cref[aa, nn, bb] = alpha * s + beta * Cstart[aa, nn, bb]
    end

    kernel = ScalarKernel(Val(4), Val(3), Float64)
    plan = QuasiStrided.plan_contract(Cv, Aperm, indA, Bneg, indB, indC; kernel = kernel, mc = 3, kc = 4, nc = 3)
    QuasiStrided.execute!(plan, alpha, beta)

    @test isapprox(Array(Csub), Cref; rtol = _macro_rtol(Float64, k_n))
end

# =====================================================================
# 3. Beta applied exactly once, across >= 2 blocks in M, N, and K.
# =====================================================================

@testset "macro driver: beta applied exactly once across multiple M/N/K blocks" begin
    rng = MersenneTwister(0xBE7A_0001)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 11, 10, 9
    mc = kc = nc = 4  # 3 blocks in each dimension

    Amat = randn(rng, Ma, Ka)
    Bmat = randn(rng, Ka, Na)
    Cstart = randn(rng, Ma, Na)
    Cmat = copy(Cstart)

    alpha, beta = 2.5, 0.75
    QuasiStrided.execute!(_dense_plan(Cmat, Amat, Bmat, kernel, mc, kc, nc), alpha, beta)

    expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
    @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))

    @testset "beta=0 with NaN-filled starting C: never read, result finite and correct" begin
        Cmat2 = fill(NaN, Ma, Na)
        QuasiStrided.execute!(_dense_plan(Cmat2, Amat, Bmat, kernel, mc, kc, nc), alpha, 0.0)
        @test all(isfinite, Cmat2)
        @test isapprox(Cmat2, alpha .* (Amat * Bmat); rtol = _macro_rtol(Float64, Ka))
    end
end

# =====================================================================
# 4. Short-circuits: alpha=0 and K=0 must never read poisoned A/B. Tested
# exactly, not approximately: the only arithmetic that may happen is C*beta.
# =====================================================================

@testset "macro driver: short-circuits never read poisoned A/B" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)

    @testset "alpha=0" begin
        Ma, Ka, Na = 11, 10, 9
        Cstart = randn(Ma, Na)
        Cmat = copy(Cstart)
        beta = 1.25
        plan = _dense_plan(Cmat, fill(NaN, Ma, Ka), fill(Inf, Ka, Na), kernel, 4, 4, 4)
        QuasiStrided.execute!(plan, 0.0, beta)
        @test Cmat == beta .* Cstart
        @test all(isfinite, Cmat)
    end

    @testset "K=0 (empty contracted axis)" begin
        Ma, Na = 5, 4
        Cstart = randn(Ma, Na)
        Cmat = copy(Cstart)
        beta = 0.5
        plan = _dense_plan(Cmat, fill(NaN, Ma, 0), fill(Inf, 0, Na), kernel, 4, 4, 4)
        QuasiStrided.execute!(plan, 1.0, beta)
        @test Cmat == beta .* Cstart
        @test all(isfinite, Cmat)
    end
end

# =====================================================================
# 5. Staleness: packed-panel reuse is the macro-blocking-specific risk (the
# old driver packed one sliver per tile and reused nothing across tiles).
# Buffers are discovered by name/eltype rather than hardcoded: only
# Vector{<:Integer} fields and "pack"-named float vectors are poisoned, so
# nothing that could alias C's own storage is ever touched.
# =====================================================================

# Returns the fields it poisoned, so the test can warn rather than silently
# pass if no field matches the heuristic.
function _poison_plan_scratch_buffers!(plan)
    poisoned = Symbol[]
    for fname in fieldnames(typeof(plan))
        fval = getfield(plan, fname)
        if fval isa Vector{<:Integer}
            fill!(fval, typemax(eltype(fval)))
            push!(poisoned, fname)
        elseif fval isa Vector{<:AbstractFloat} && occursin("pack", lowercase(String(fname)))
            fill!(fval, eltype(fval)(NaN))
            push!(poisoned, fname)
        end
    end
    return poisoned
end

@testset "macro driver: staleness / packed-buffer poisoning before execute!" begin
    rng = MersenneTwister(0x57A1_E000)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 13, 11, 9
    mc = kc = nc = 4  # multiple M/N/K blocks, so panel reuse actually happens

    Amat = randn(rng, Ma, Ka)
    Bmat = randn(rng, Ka, Na)
    Cstart = randn(rng, Ma, Na)
    Cmat = copy(Cstart)

    plan = _dense_plan(Cmat, Amat, Bmat, kernel, mc, kc, nc)
    poisoned = _poison_plan_scratch_buffers!(plan.workspace)
    if isempty(poisoned)
        @warn "macro driver staleness test: no plan field matched the packed/index-buffer " *
            "heuristic; fields were $(fieldnames(typeof(plan)))."
    end

    alpha, beta = 2.5, 0.75
    QuasiStrided.execute!(plan, alpha, beta)
    expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
    @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))

    # Again on the same (reused) plan: targets staleness across calls, as
    # opposed to across the blocks of one call.
    _poison_plan_scratch_buffers!(plan.workspace)
    copyto!(Cmat, Cstart)
    QuasiStrided.execute!(plan, alpha, beta)
    @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))
end

# =====================================================================
# 6. execute! vs execute_tilewise!, on testset 1's exact cases (same seed).
# The two agree only to within summation-order rounding: same arithmetic,
# different block/tile iteration order.
# =====================================================================

@testset "macro driver: execute! agrees with execute_tilewise! (old driver oracle)" begin
    for case in _macro_random_cases(0x5A17_D817)
        (; Ka, mc, kc, nc, kernel, alpha, beta, Amat, Bmat, Cstart, T) = case

        Cmat_macro = copy(Cstart)
        QuasiStrided.execute!(_dense_plan(Cmat_macro, Amat, Bmat, kernel, mc, kc, nc), alpha, beta)

        Cmat_tw = copy(Cstart)
        QuasiStrided.execute_tilewise!(_dense_plan(Cmat_tw, Amat, Bmat, kernel, mc, kc, nc), alpha, beta)

        @test isapprox(Cmat_macro, Cmat_tw; rtol = _macro_rtol(T, Ka))
    end
end

# =====================================================================
# 7. Blocking validation and effective-value clamping (via behavior).
# =====================================================================

@testset "macro driver: Blocking validation" begin
    @test_throws ArgumentError QuasiStrided.Blocking(0, 1, 1)
    @test_throws ArgumentError QuasiStrided.Blocking(1, 0, 1)
    @test_throws ArgumentError QuasiStrided.Blocking(1, 1, 0)
    @test_throws ArgumentError QuasiStrided.Blocking(-1, 1, 1)
    @test_throws ArgumentError QuasiStrided.Blocking(1, -5, 1)
    @test_throws ArgumentError QuasiStrided.Blocking(1, 1, -5)
    @test QuasiStrided.Blocking(1, 1, 1) isa QuasiStrided.Blocking  # smallest legal values

    kernel = ScalarKernel(Val(4), Val(3), Float64)
    @test QuasiStrided.default_blocking(kernel) isa QuasiStrided.Blocking
end

@testset "macro driver: effective block-size clamping, tested via behavior" begin
    rng = MersenneTwister(0xC1AB_9002)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 7, 6
    Amat = randn(rng, Ma, Ka)
    Bmat = randn(rng, Ka, Na)

    @testset "mc/kc/nc each larger than the corresponding Q: must clamp, not error" begin
        Cmat = zeros(Ma, Na)
        QuasiStrided.execute!(_dense_plan(Cmat, Amat, Bmat, kernel, 10_000, 10_000, 10_000), 1.0, 0.0)
        @test isapprox(Cmat, Amat * Bmat; rtol = _macro_rtol(Float64, Ka))
    end

    @testset "mc=nc=1: many single-row/single-column blocks" begin
        Cmat = zeros(Ma, Na)
        QuasiStrided.execute!(_dense_plan(Cmat, Amat, Bmat, kernel, 1, 3, 1), 1.0, 0.0)
        @test isapprox(Cmat, Amat * Bmat; rtol = _macro_rtol(Float64, Ka))
    end

    @testset "kc=1: many single-step K panels" begin
        Cmat = randn(rng, Ma, Na)
        Cstart = copy(Cmat)
        alpha, beta = 1.7, 0.3
        QuasiStrided.execute!(_dense_plan(Cmat, Amat, Bmat, kernel, 4, 1, 4), alpha, beta)
        expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
        @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))
    end
end

# =====================================================================
# 8. Irregular (ScatterAxis) sliver at nonzero offset within a macro block
# (Phase D review, should-fix 1). Every other testset here uses a
# single-label M/N/K group, which `describe_block` always classifies
# regular -- one dimension has no internal carry boundary to break the
# constant stride. This fixture uses a 2-label M group (a,q) where A's map
# folds across the a/q boundary but C's, deliberately padded, does not, so
# the C-side sliver at first>0 really is classified irregular. `_axis_of`
# is the one helper all three sides share, so this covers N and K too.
# =====================================================================

@testset "macro driver: irregular sliver at nonzero offset (multi-label M group)" begin
    a_n, q_n, k_n, n_n = 7, 2, 5, 3   # M = (a,q), a fastest; length 14
    pad = 3                            # breaks C's fold condition (a_n+pad != a_n)

    Amat = randn(a_n, q_n, k_n)        # dense: a-stride 1, q-stride a_n
    Av = StridedView(Amat)
    indA = (1, 2, 3)

    Bmat = randn(k_n, n_n)
    Bv = StridedView(Bmat)
    indB = (3, 4)

    # C[a,q,n] with a gap in q (q-stride = a_n+pad), so C's M-map fails the
    # fold condition at every a-boundary.
    Cbig = randn(a_n + pad, q_n, n_n)
    Csub = view(Cbig, 1:a_n, :, :)
    Cv = StridedView(Csub)
    indC = (1, 2, 4)
    Cstart = copy(Csub)

    alpha, beta = 1.7, -0.4
    Cref = zeros(a_n, q_n, n_n)
    for nn in 1:n_n, qq in 1:q_n, aa in 1:a_n
        s = 0.0
        for kk in 1:k_n
            s += Amat[aa, qq, kk] * Bmat[kk, nn]
        end
        Cref[aa, qq, nn] = alpha * s + beta * Cstart[aa, qq, nn]
    end

    kernel = ScalarKernel(Val(4), Val(3), Float64)
    # mc=8 gives the first ic block two MR=4 slivers; the second (rfirst=4,
    # coords a=4,5,6,q=0 then a=0,q=1) crosses the a/q boundary -- the
    # irregular-at-nonzero-offset case. The 6-wide tail block adds tail
    # coverage.
    plan = QuasiStrided.plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = 8, kc = 5, nc = 3)
    QuasiStrided.execute!(plan, alpha, beta)
    @test isapprox(Array(Csub), Cref; rtol = _macro_rtol(Float64, k_n))

    # execute! vs execute_tilewise! on the same irregular fixture.
    Cbig_tw = randn(a_n + pad, q_n, n_n)
    Cbig_tw[1:a_n, :, :] .= Cstart
    Csub_tw = view(Cbig_tw, 1:a_n, :, :)
    plan_tw = QuasiStrided.plan_contract(
        StridedView(Csub_tw), Av, indA, Bv, indB, indC;
        kernel = kernel, mc = 8, kc = 5, nc = 3
    )
    QuasiStrided.execute_tilewise!(plan_tw, alpha, beta)
    @test isapprox(Array(Csub_tw), Array(Csub); rtol = _macro_rtol(Float64, k_n))
end
