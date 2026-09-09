# Independent oracle/property tests for the macro-blocking `execute!` rewrite
# (docs/decisions.md, "Macro-blocking milestone" section). Written against
# the FROZEN signatures recorded there:
#
#   plan_contract(C, A, indA, B, indB, indC; kernel=default, mc=nothing,
#                 kc=nothing, nc=nothing) -> ContractPlan
#   struct Blocking; mc::Int; kc::Int; nc::Int; end
#   default_blocking(kernel) -> Blocking
#   execute!(plan, alpha, beta)                       # macro-blocking rewrite
#   QuasiStrided.execute_tilewise!(plan, alpha, beta)  # old tile-by-tile oracle (unexported)
#
# None of `plan_contract`, `execute!`, `ContractPlan`, `Blocking`,
# `default_blocking`, `execute_tilewise!` are exported, so every use below is
# fully qualified as `QuasiStrided.<name>` -- this also avoids colliding with
# the unqualified `const plan_contract = QuasiStrided.plan_contract`-style
# bindings that test_driver.jl introduces into the same top-level scope when
# both files are `include`d from test/runtests.jl.
#
# This file cannot run successfully until the concurrent Macro-C driver
# rewrite (worktree QuasiStrided.jl-macro-c1, branch macro-blocking-driver)
# lands: `Blocking`/`default_blocking`/the new `execute!`/`execute_tilewise!`
# do not exist yet in this checkout. It is written and syntax-checked, not
# executed, per the oracle-worker task instructions.

using Test
using Random
using StridedViews: StridedView, offset

# =====================================================================
# Shared helpers
# =====================================================================

# Kernel constructors x (MR,NR) shapes x element types exercised throughout.
# Two shapes per the task spec: (Val(4),Val(3)) and (Val(8),Val(6)).
const _MACRO_KERNEL_CTORS = (ScalarKernel, SIMDKernel)
const _MACRO_SHAPES = ((Val(4), Val(3)), (Val(8), Val(6)))
const _MACRO_ELTYPES = (Float64, Float32)

"""
    _random_macro_case(rng, ctor, shape, T)

Build one randomized dense-matmul case: shapes `(Ma,Ka,Na)` in 1:37, block
sizes `(mc,kc,nc)` in 1:13 (small on purpose, to force multiple blocks in
each dimension even for small operand shapes), plus random alpha/beta and a
random starting C. Returns a NamedTuple with everything needed to both run
the driver and independently compute the expected dense-matmul result.
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

"""
    _macro_random_cases(seed) -> Vector

A handful of `_random_macro_case`s per (kernel ctor, shape, eltype) combo,
built off a single fixed-seed RNG stream so the *same* sequence of cases is
reproduced whenever this is called with the same seed (used to keep testset
1 and testset 6 -- "execute! vs execute_tilewise!" on the exact same cases
-- in lockstep without duplicating the RNG draws inline in two places).
"""
# SIMDKernel requires mr(kernel) to be a multiple of its lane width, which is
# type-dependent (_default_lanewidth: 4 for Float64, 8 for Float32) -- so
# (Val(4),Val(3)) is not a valid SIMDKernel/Float32 combo (4 is not a
# multiple of 8). Filter to the combos each (ctor, T) can actually construct,
# rather than the full cartesian product.
function _valid_shapes(ctor, ::Type{T}) where {T}
    ctor !== SIMDKernel && return _MACRO_SHAPES
    return Tuple(s for s in _MACRO_SHAPES if _valtype(s[1]) % QuasiStrided._default_lanewidth(T) == 0)
end
_valtype(::Val{N}) where {N} = N

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

# Tolerance rationale (reused across testsets): random dense matmul entries
# accumulate Ka terms of order 1, and both operands *and* the M/N/K block
# partition change how those Ka terms are summed (different grouping ->
# different rounding, still mathematically exact in infinite precision). A
# textbook conditioning bound for such a sum scales like Ka*eps(T) relative
# to the entry magnitude (~sqrt(Ka) for a sum of Ka iid unit-variance
# products); we use a generous multiple of that to absorb both the summation
# reordering and BLIS-style accumulate-in-different-order effects, without
# being so loose it would hide a real bug.
_macro_rtol(::Type{T}, Ka::Integer) where {T} = 50 * max(Ka, 1) * eps(T)

# =====================================================================
# 1. Randomized agreement vs. dense matmul
# =====================================================================

@testset "macro driver: randomized agreement vs. dense matmul" begin
    for case in _macro_random_cases(0x5A17_D817)
        (; Ma, Ka, Na, mc, kc, nc, kernel, alpha, beta, Amat, Bmat, Cstart, T) = case
        Cmat = copy(Cstart)
        Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
        indA, indB, indC = (1, 2), (2, 3), (1, 3)

        plan = QuasiStrided.plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = mc, kc = kc, nc = nc)
        QuasiStrided.execute!(plan, alpha, beta)

        expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
        @test isapprox(Cmat, expected; rtol = _macro_rtol(T, Ka))
    end
end

# =====================================================================
# 2. 3-index tensor fixture with real StridedViews: permutation, sliced
#    (non-zero-offset) destination, negative stride, zero-stride broadcast.
# Reference computed with plain nested for loops over logical indices,
# indexing the StridedViews directly -- plan_contract/execute! is never
# consulted while building the reference.
# =====================================================================

@testset "macro driver: 3-index StridedViews fixture (permuted/sliced/negative/zero-stride)" begin
    a_n, k_n, b_n, n_n = 7, 11, 5, 9

    # A[a,k,b]: genuinely independent of b (zero stride on the b axis), then
    # presented to the driver as a permuted view (k,b,a) axis order.
    A2 = randn(a_n, k_n)
    Araw = StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0)  # Araw[a,k,b] == A2[a,k]
    Aperm = permutedims(Araw, (2, 3, 1))                          # Aperm[k,b,a] == A2[a,k]
    indA = (2, 3, 1)  # labels: axis1=k(2), axis2=b(3), axis3=a(1) (matches the a=1,k=2,b=3,n=4 convention)

    # B[k,n]: negative stride along k (Bneg[k,n] == Bfull[k_n - k + 1, n] for
    # the underlying flat data, i.e. a reversed-row view), built manually
    # per the strided_integration.jl idiom for negative-stride views.
    Bdata = randn(k_n * n_n)
    Bneg = StridedView(Bdata, (k_n, n_n), (-1, k_n), k_n - 1)
    indB = (2, 4)

    # C[a,n,b]: sliced destination with a nonzero base offset, embedded in a
    # larger backing array, with a nontrivial starting value (to exercise
    # beta on this same fixture).
    Cbig = randn(a_n + 2, n_n + 3, b_n + 1)
    Csub = view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n)
    Cv = StridedView(Csub)
    indC = (1, 4, 3)
    @test offset(Cv) != 0

    Cstart = copy(Csub)

    # Independent reference: plain nested loops over logical (a,k,b,n)
    # indices, indexing Aperm/Bneg/Cstart directly (never via AxisGroup,
    # BlockDescriptor, plan_contract, or execute!).
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

    # rtol scaled by k_n*eps(Float64): same summation-reordering rationale as
    # _macro_rtol, spelled out inline since this fixture builds Cref by hand
    # rather than through _random_macro_case.
    @test isapprox(Array(Csub), Cref; rtol = 50 * k_n * eps(Float64))
end

# =====================================================================
# 3. Beta applied exactly once, across >= 2 blocks in M, N, and K.
# =====================================================================

@testset "macro driver: beta applied exactly once across multiple M/N/K blocks" begin
    rng = MersenneTwister(0xBE7A_0001)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 11, 10, 9
    mc = kc = nc = 4  # ceil(11/4)=3, ceil(10/4)=3, ceil(9/4)=3 blocks: >= 2 in each dimension

    Amat = randn(rng, Ma, Ka)
    Bmat = randn(rng, Ka, Na)
    Cstart = randn(rng, Ma, Na)
    Cmat = copy(Cstart)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    alpha, beta = 2.5, 0.75
    plan = QuasiStrided.plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = mc, kc = kc, nc = nc)
    QuasiStrided.execute!(plan, alpha, beta)

    expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
    @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))

    @testset "beta=0 with NaN-filled starting C: never read, result finite and correct" begin
        Cmat2 = fill(NaN, Ma, Na)
        Cv2 = StridedView(Cmat2)
        plan2 = QuasiStrided.plan_contract(Cv2, Av, indA, Bv, indB, indC; kernel = kernel, mc = mc, kc = kc, nc = nc)
        QuasiStrided.execute!(plan2, alpha, 0.0)
        @test all(isfinite, Cmat2)
        @test isapprox(Cmat2, alpha .* (Amat * Bmat); rtol = _macro_rtol(Float64, Ka))
    end
end

# =====================================================================
# 4. Short-circuits: alpha=0 (never reads poisoned A/B) and K=0 (ditto).
# =====================================================================

@testset "macro driver: short-circuits never read poisoned A/B" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)

    @testset "alpha=0" begin
        Ma, Ka, Na = 11, 10, 9
        Apoison = fill(NaN, Ma, Ka)
        Bpoison = fill(Inf, Ka, Na)
        Cstart = randn(Ma, Na)
        Cmat = copy(Cstart)
        Av, Bv, Cv = StridedView(Apoison), StridedView(Bpoison), StridedView(Cmat)
        indA, indB, indC = (1, 2), (2, 3), (1, 3)

        beta = 1.25
        plan = QuasiStrided.plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = 4, kc = 4, nc = 4)
        QuasiStrided.execute!(plan, 0.0, beta)
        # Exact (not approximate): the only arithmetic that should occur is
        # scaling C by beta, which is the same single floating-point
        # operation performed on the LHS and RHS here.
        @test Cmat == beta .* Cstart
        @test all(isfinite, Cmat)
    end

    @testset "K=0 (empty contracted axis)" begin
        Ma, Na = 5, 4
        Apoison = fill(NaN, Ma, 0)  # K axis length 0
        Bpoison = fill(Inf, 0, Na)
        Cstart = randn(Ma, Na)
        Cmat = copy(Cstart)
        Av, Bv, Cv = StridedView(Apoison), StridedView(Bpoison), StridedView(Cmat)
        indA, indB, indC = (1, 2), (2, 3), (1, 3)

        beta = 0.5
        plan = QuasiStrided.plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = 4, kc = 4, nc = 4)
        QuasiStrided.execute!(plan, 1.0, beta)
        @test Cmat == beta .* Cstart
        @test all(isfinite, Cmat)
    end
end

# =====================================================================
# 5. Staleness / buffer-poisoning: packed-panel reuse is the macro-blocking-
# specific new risk (the old tile-by-tile driver packed one sliver per tile
# and never reused a buffer across output tiles). Written defensively: the
# real ContractPlan's field names are owned by the concurrent Macro-C
# implementation and are not available here, so this discovers candidate
# scratch-buffer fields generically via `fieldnames`/`eltype` rather than
# hardcoding names. Only Vector{<:Integer} fields (index/offset buffers) and
# Vector{<:AbstractFloat} fields whose name contains "pack" are poisoned --
# deliberately NOT every AbstractFloat vector, to avoid ever poisoning a
# field that might alias C's own output storage.
# =====================================================================

"""
    _poison_plan_scratch_buffers!(plan) -> Vector{Symbol}

Best-effort, generic buffer poisoning. Returns the field names it actually
poisoned, so the test can report (rather than silently pass) if the real
`ContractPlan` exposes no field matching the heuristic -- see the note at
the call site below.
"""
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
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    plan = QuasiStrided.plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = mc, kc = kc, nc = nc)
    poisoned = _poison_plan_scratch_buffers!(plan)
    if isempty(poisoned)
        @warn "macro driver staleness test: no plan field matched the packed/index-buffer heuristic; " *
            "field names were $(fieldnames(typeof(plan))). The main process should add a targeted " *
            "poisoning branch for the real scratch-buffer field name(s) at integration time."
    end

    alpha, beta = 2.5, 0.75
    QuasiStrided.execute!(plan, alpha, beta)
    expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
    @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))

    # Poison again and execute a *second* time on the same (reused) plan, to
    # specifically target reuse-across-calls staleness (as opposed to reuse
    # only within the many blocks of a single execute! call, already
    # exercised above).
    _poison_plan_scratch_buffers!(plan)
    copyto!(Cmat, Cstart)
    QuasiStrided.execute!(plan, alpha, beta)
    @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))
end

# =====================================================================
# 6. execute! vs execute_tilewise! agreement, on the same randomized cases
# as testset 1 (same seed -> same case sequence).
# =====================================================================

@testset "macro driver: execute! agrees with execute_tilewise! (old driver oracle)" begin
    for case in _macro_random_cases(0x5A17_D817)
        (; Ma, Ka, Na, mc, kc, nc, kernel, alpha, beta, Amat, Bmat, Cstart, T) = case

        Cmat_macro = copy(Cstart)
        Av, Bv = StridedView(Amat), StridedView(Bmat)
        indA, indB, indC = (1, 2), (2, 3), (1, 3)
        plan_macro = QuasiStrided.plan_contract(
            StridedView(Cmat_macro), Av, indA, Bv, indB, indC;
            kernel = kernel, mc = mc, kc = kc, nc = nc
        )
        QuasiStrided.execute!(plan_macro, alpha, beta)

        Cmat_tw = copy(Cstart)
        plan_tw = QuasiStrided.plan_contract(
            StridedView(Cmat_tw), Av, indA, Bv, indB, indC;
            kernel = kernel, mc = mc, kc = kc, nc = nc
        )
        QuasiStrided.execute_tilewise!(plan_tw, alpha, beta)

        # Both are implementations of the same contraction, agreeing only to
        # within summation-order rounding (same rationale/scale as
        # _macro_rtol) -- not bit-identical, per docs/decisions.md's runtime-
        # switches note that reordering work is bitwise identical only when
        # the *arithmetic* performed is unchanged, which holds here (both
        # drivers use the same kernel/complex-method-equivalent, only the
        # block/tile iteration order differs).
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
    indA, indB, indC = (1, 2), (2, 3), (1, 3)
    Av, Bv = StridedView(Amat), StridedView(Bmat)

    @testset "mc/kc/nc each larger than the corresponding Q: must clamp, not error" begin
        Cmat = zeros(Ma, Na)
        plan = QuasiStrided.plan_contract(
            StridedView(Cmat), Av, indA, Bv, indB, indC;
            kernel = kernel, mc = 10_000, kc = 10_000, nc = 10_000
        )
        QuasiStrided.execute!(plan, 1.0, 0.0)
        @test isapprox(Cmat, Amat * Bmat; rtol = _macro_rtol(Float64, Ka))
    end

    @testset "mc=nc=1: many single-row/single-column blocks" begin
        Cmat = zeros(Ma, Na)
        plan = QuasiStrided.plan_contract(
            StridedView(Cmat), Av, indA, Bv, indB, indC;
            kernel = kernel, mc = 1, kc = 3, nc = 1
        )
        QuasiStrided.execute!(plan, 1.0, 0.0)
        @test isapprox(Cmat, Amat * Bmat; rtol = _macro_rtol(Float64, Ka))
    end

    @testset "kc=1: many single-step K panels" begin
        Cmat = randn(rng, Ma, Na)
        Cstart = copy(Cmat)
        alpha, beta = 1.7, 0.3
        plan = QuasiStrided.plan_contract(
            StridedView(Cmat), Av, indA, Bv, indB, indC;
            kernel = kernel, mc = 4, kc = 1, nc = 4
        )
        QuasiStrided.execute!(plan, alpha, beta)
        expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
        @test isapprox(Cmat, expected; rtol = _macro_rtol(Float64, Ka))
    end
end
