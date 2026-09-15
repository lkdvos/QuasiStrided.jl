# Independent oracle/property tests for the macro-blocking `execute!`
# (docs/decisions.md, "Macro-blocking milestone"). Nothing under test is
# exported, and test_driver.jl introduces unqualified `plan_contract`/
# `execute!` bindings into the same top-level scope, so every use here is
# written as `QuasiStrided.<name>` to avoid colliding with those.

using Test
using Random
using StridedViews: StridedView, offset

# `test/runtests.jl` restores QuasiStrided's un-exported internal tier into the
# including scope, so the names below are already bound when this file is
# `include`d from there. Guarded so the file also runs standalone
# (`julia test/test_macro_driver.jl`-style) without shadowing or re-binding
# anything when it does not. Everything added for the complex element type is
# written `QuasiStrided.<name>` regardless, per the header comment.
using QuasiStrided
if !@isdefined(ScalarKernel)
    using QuasiStrided: ScalarKernel, SIMDKernel, Blocking, default_blocking
end

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

# =====================================================================
# Complex element type (docs/decisions.md, "Complex element-type milestone:
# Phase A direction freeze", "Verification contract" layer 4).
#
# Everything below is an *addition*: the real-eltype helpers and testsets
# above are textually unchanged, which is half the proof that the real path
# is untouched (the freeze's proof (1), "git diff shows additions, not
# edits").
#
# The oracle stays independent of the engine. In particular it does NOT call
# `QuasiStrided._qs_isconj` or `QuasiStrided._op_conjugates`: the
# conjugation table and the XOR rule are both re-derived here from the
# freeze's prose, so that an engine that folds `conjA` with `op` using `||`
# instead of `⊻`, or that mistakes `adjoint` for a non-conjugating `op`,
# fails these tests rather than agreeing with them.
# =====================================================================

const _MACRO_COMPLEX_ELTYPES = (ComplexF64, ComplexF32)

# The four `op` values `StridedViews` can put on a view. `op` is *elementwise*
# on a `Number` element (a `StridedView`'s axis permutation lives in its
# size/strides, not in `op`), so a view built with any of these has the same
# shape as its parent and only its values may differ.
const _MACRO_OPS = (identity, conj, adjoint, transpose)

# Independent re-implementation of the elementwise effect of `op` on a complex
# `Number`. Deliberately not `QuasiStrided._op_conjugates`.
_macro_op_conjugates(op) = op === conj || op === adjoint

# The frozen combining rule, re-derived here: the TO flag and the view's `op`
# are independent sources of conjugation and compose with XOR, so a
# `conj`-op'd view with the flag set is *unconjugated*. Real eltypes are never
# conjugated, whatever the flag says.
_macro_conjugated(::Type{T}, flag::Bool, op) where {T} =
    (T <: Complex) && (flag ⊻ _macro_op_conjugates(op))

# Same tolerance as the real path at the same precision -- the freeze makes
# this an acceptance criterion: "If a complex test needs a looser tolerance
# than its real counterpart, that is a bug signal". A new method, so the real
# `_macro_rtol` above is untouched (`eps` is undefined on a Complex type, so
# the generic method cannot serve both).
_macro_rtol(::Type{Complex{R}}, Ka::Integer) where {R} = _macro_rtol(R, Ka)

# Complex kernel constructors. `OneMKernel` is landing concurrently, so it is
# picked up only if it exists; this file is green either way.
function _macro_complex_kernel_ctors()
    isdefined(QuasiStrided, :OneMKernel) &&
        return (QuasiStrided.PlanarKernel, QuasiStrided.OneMKernel)
    return (QuasiStrided.PlanarKernel,)
end

# Both complex kernels take `(Val(MR), Val(NR), T)` (defaulting W to
# `_default_lanewidth(real(T))`) and both constrain MR against W: planar needs
# `MR % W == 0` (one accumulator plane pair per W logical rows) and 1m needs
# `2MR % W == 0` (its inner real kernel is `SIMDKernel{2MR,NR,real(T),W}`).
# Rather than encode either rule here -- one of the two constructors is being
# written by another worker as this is written -- ask the constructor and skip
# the combos it rejects, exactly as `_valid_shapes` does for `SIMDKernel`.
function _macro_complex_kernel(ctor, shape, ::Type{T}) where {T}
    return try
        ctor(shape[1], shape[2], T)
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
end

# The (ctor, shape) combos that exist for this eltype, as constructed kernels.
function _macro_complex_kernels(::Type{T}) where {T}
    ks = Any[]
    for ctor in _macro_complex_kernel_ctors(), shape in _MACRO_SHAPES
        k = _macro_complex_kernel(ctor, shape, T)
        k === nothing || push!(ks, k)
    end
    return ks
end

# alpha/beta must include 0 and 1 (the two short-circuit values) and must be
# *genuinely complex* the rest of the time: a real-only alpha would miss every
# bug in which the imaginary part of a scalar is dropped or mis-signed.
function _macro_complex_scalar(rng::MersenneTwister, ::Type{T}) where {T}
    r = rand(rng)
    r < 0.08 && return zero(T)
    r < 0.16 && return one(T)
    return T(rand(rng, -3.0:0.5:3.0), rand(rng, -3.0:0.5:3.0))
end

"""
    _random_macro_complex_case(rng, kernel, T; force_conj = false)

One randomized complex dense-matmul case. Same conventions as
[`_random_macro_case`](@ref) -- extents in 1:37 against block sizes in 1:13,
so every dimension is split into several blocks -- plus the two conjugation
axes, **drawn rather than enumerated** (the freeze: the full product of 2
flags x 4 ops x 2 operands x eltypes x methods x shapes is far too large to
enumerate at driver scale).

`force_conj` resamples until at least one operand really is conjugated, for
the `execute_tilewise!` comparison that must be run *with conjugation set*.

The expectation is computed here, without the engine: the XOR rule is applied
by `_macro_conjugated`, the conjugated operand is materialized with `conj.`,
and the contraction is a plain dense matmul of the materialized operands.
`alpha`/`beta` are never conjugated.
"""
function _random_macro_complex_case(
        rng::MersenneTwister, kernel, ::Type{T}; force_conj::Bool = false
    ) where {T}
    Ma = rand(rng, 1:37)
    Ka = rand(rng, 1:37)
    Na = rand(rng, 1:37)
    mc = rand(rng, 1:13)
    kc = rand(rng, 1:13)
    nc = rand(rng, 1:13)

    conjA = rand(rng, (false, true))
    conjB = rand(rng, (false, true))
    opA = rand(rng, _MACRO_OPS)
    opB = rand(rng, _MACRO_OPS)
    if force_conj && !(_macro_conjugated(T, conjA, opA) || _macro_conjugated(T, conjB, opB))
        conjA = !conjA
    end

    alpha = _macro_complex_scalar(rng, T)
    beta = _macro_complex_scalar(rng, T)

    Amat = randn(rng, T, Ma, Ka)
    Bmat = randn(rng, T, Ka, Na)
    # NaN-poisoned starting C whenever beta == 0: the driver must never read it.
    Cstart = iszero(beta) ? fill(T(NaN, NaN), Ma, Na) : randn(rng, T, Ma, Na)

    return (;
        Ma, Ka, Na, mc, kc, nc, kernel, conjA, conjB, opA, opB,
        alpha, beta, Amat, Bmat, Cstart, T,
    )
end

# `op` is elementwise, so this view has the parent's shape and the parent's
# strides; only the values it denotes may be conjugated. Built by the
# 5-argument constructor rather than by `conj`/`adjoint`/`transpose`, because
# those functions either materialize (on an `Array`) or permute dimensions (on
# a 2-d `StridedView`) instead of leaving `op` set to the value under test.
_macro_op_view(M::AbstractMatrix, op) = StridedView(M, size(M), strides(M), 0, op)

# Plan for the same dense matmul as `_dense_plan`, with the conjugation flags
# and with operands presented as `op`-carrying views. `_dense_plan` is left
# untouched for the real testsets.
function _dense_plan_conj(Cmat, Amat, Bmat, kernel, mc, kc, nc, conjA, conjB, opA, opB)
    return QuasiStrided.plan_contract(
        StridedView(Cmat), _macro_op_view(Amat, opA), (1, 2),
        _macro_op_view(Bmat, opB), (2, 3), (1, 3);
        kernel = kernel, conjA = conjA, conjB = conjB, mc = mc, kc = kc, nc = nc
    )
end

# The expected result, engine-free.
function _macro_complex_expected(case)
    (; conjA, conjB, opA, opB, alpha, beta, Amat, Bmat, Cstart, T) = case
    Aeff = _macro_conjugated(T, conjA, opA) ? conj.(Amat) : Amat
    Beff = _macro_conjugated(T, conjB, opB) ? conj.(Bmat) : Bmat
    AB = Aeff * Beff
    # beta == 0 goes with a NaN-poisoned Cstart, which must not appear in the
    # expectation at all (the driver must not read it either).
    iszero(beta) && return alpha .* AB
    return alpha .* AB .+ beta .* Cstart
end

# One fixed seed per stream, printed on failure so any failing case
# reproduces exactly.
const _MACRO_COMPLEX_SEED = 0xC047_EE01
const _MACRO_COMPLEX_TW_SEED = 0xC047_EE02

# ~300 ComplexF64 cases and ~200 ComplexF32 ones (the reference project's
# scale), spread evenly over whichever (method, shape) combos exist on this
# machine -- so the totals stay put whether or not `OneMKernel` has landed.
function _macro_complex_random_cases(
        seed::Integer; c64_total::Int = 300, c32_total::Int = 200, force_conj::Bool = false
    )
    rng = MersenneTwister(seed)
    cases = Any[]
    for T in _MACRO_COMPLEX_ELTYPES
        kernels = _macro_complex_kernels(T)
        isempty(kernels) && continue
        total = T === ComplexF64 ? c64_total : c32_total
        per_combo = cld(total, length(kernels))
        for kernel in kernels, _ in 1:per_combo
            push!(cases, _random_macro_complex_case(rng, kernel, T; force_conj = force_conj))
        end
    end
    return cases
end

# Compact, reproducible description of a failing case.
_macro_case_id(case, seed, i) = string(
    "seed = ", repr(seed), ", case ", i, ": ", nameof(typeof(case.kernel)),
    "{", QuasiStrided.mr(case.kernel), ",", QuasiStrided.nr(case.kernel), "}{",
    case.T, "} M,K,N = ", case.Ma, ",", case.Ka, ",", case.Na,
    " mc,kc,nc = ", case.mc, ",", case.kc, ",", case.nc,
    " conjA = ", case.conjA, " opA = ", case.opA,
    " conjB = ", case.conjB, " opB = ", case.opB,
    " alpha = ", case.alpha, " beta = ", case.beta
)

# =====================================================================
# 9. Randomized complex agreement vs. dense matmul, over the randomized
# conjugation/`op` cross-product.
# =====================================================================

@testset "macro driver: randomized complex agreement vs. dense matmul (conj/op randomized)" begin
    kernels64 = _macro_complex_kernels(ComplexF64)
    @test !isempty(kernels64)   # at least planar must be constructible here

    cases = _macro_complex_random_cases(_MACRO_COMPLEX_SEED)
    for (i, case) in enumerate(cases)
        (; Ka, mc, kc, nc, kernel, conjA, conjB, opA, opB, alpha, beta, Amat, Bmat, Cstart, T) = case
        Cmat = copy(Cstart)
        plan = _dense_plan_conj(Cmat, Amat, Bmat, kernel, mc, kc, nc, conjA, conjB, opA, opB)
        QuasiStrided.execute!(plan, alpha, beta)

        expected = _macro_complex_expected(case)
        ok = isapprox(Cmat, expected; rtol = _macro_rtol(T, Ka))
        # beta == 0 means Cstart was NaN-poisoned: never read.
        finite_ok = !iszero(beta) || all(isfinite, Cmat)
        (ok && finite_ok) ||
            @error "complex macro-driver case failed: " * _macro_case_id(case, _MACRO_COMPLEX_SEED, i)
        @test ok
        @test finite_ok
    end
end

# =====================================================================
# 10. execute! vs execute_tilewise!, WITH CONJUGATION SET.
#
# Mandatory, not optional (the freeze, "Where conjugation is applied"):
# `_pack_sliver!` has three call sites and the third is `execute_tilewise!`'s.
# If that one were left hardcoded to `identity`, the *oracle* would be
# silently wrong for conjugated inputs and the disagreement would present as
# an engine bug. Every case here has at least one genuinely conjugated
# operand, so a missed transform cannot pass.
# =====================================================================

@testset "macro driver: complex execute! agrees with execute_tilewise! (conjugation set)" begin
    cases = _macro_complex_random_cases(
        _MACRO_COMPLEX_TW_SEED; c64_total = 60, c32_total = 40, force_conj = true
    )
    for (i, case) in enumerate(cases)
        (; Ka, mc, kc, nc, kernel, conjA, conjB, opA, opB, alpha, beta, Amat, Bmat, Cstart, T) = case
        @test _macro_conjugated(T, conjA, opA) || _macro_conjugated(T, conjB, opB)

        Cmat_macro = copy(Cstart)
        QuasiStrided.execute!(
            _dense_plan_conj(Cmat_macro, Amat, Bmat, kernel, mc, kc, nc, conjA, conjB, opA, opB),
            alpha, beta
        )

        Cmat_tw = copy(Cstart)
        QuasiStrided.execute_tilewise!(
            _dense_plan_conj(Cmat_tw, Amat, Bmat, kernel, mc, kc, nc, conjA, conjB, opA, opB),
            alpha, beta
        )

        ok = isapprox(Cmat_macro, Cmat_tw; rtol = _macro_rtol(T, Ka))
        ok || @error "complex execute!/execute_tilewise! disagreement: " *
            _macro_case_id(case, _MACRO_COMPLEX_TW_SEED, i)
        @test ok
    end
end

# =====================================================================
# 11. Cache-crossing, per method.
#
# The randomized cases above use block sizes in 1:13, i.e. they never consult
# `default_blocking`. This testset does the opposite: it takes the *shipped*
# defaults for each (method, eltype) and picks extents that exceed all three,
# so the multi-K-panel `beta_eff = firstpanel ? betaT : one(T)` path runs at
# the real block sizes.
#
# Each method has its own `mc` -- 1m's is exactly half planar's, derived from
# `a_reals` -- so one shape does not cover both. The extents are therefore
# *derived* from `default_blocking(kernel)` rather than hardcoded, and the
# crossing is asserted against the plan's own effective blocking rather than
# assumed.
# =====================================================================

@testset "macro driver: cache-crossing at the shipped default blocking, per method" begin
    for T in _MACRO_COMPLEX_ELTYPES
        for kernel in _macro_complex_kernels(T)
            QuasiStrided.mr(kernel) == 8 || continue   # one shape per (method, eltype)
            name = string(nameof(typeof(kernel)), "{", T, "}")
            @testset "$name" begin
                b = QuasiStrided.default_blocking(kernel)
                Ma = b.mc + QuasiStrided.mr(kernel)
                Ka = b.kc + 1
                Na = b.nc + QuasiStrided.nr(kernel)

                Amat = randn(MersenneTwister(0xCAC1_0001), T, Ma, Ka)
                Bmat = randn(MersenneTwister(0xCAC1_0002), T, Ka, Na)
                Cstart = randn(MersenneTwister(0xCAC1_0003), T, Ma, Na)
                Cmat = copy(Cstart)

                alpha = T(1.5, -0.25)
                beta = T(-0.5, 0.75)
                # Conjugation on, and via both sources at once: `conjA` on A,
                # a non-`conj` conjugating `op` on B.
                plan = _dense_plan_conj(
                    Cmat, Amat, Bmat, kernel, nothing, nothing, nothing,
                    true, false, identity, adjoint
                )

                # The evidence that this shape actually crosses: the plan's
                # *effective* blocking (rounded and clamped by plan_contract)
                # is strictly smaller than the extent in every dimension, so
                # there is more than one block along each.
                @test plan.blocking.mc < Ma
                @test plan.blocking.kc < Ka
                @test plan.blocking.nc < Na

                QuasiStrided.execute!(plan, alpha, beta)

                expected = alpha .* (conj.(Amat) * conj.(Bmat)) .+ beta .* Cstart
                @test isapprox(Cmat, expected; rtol = _macro_rtol(T, Ka))
            end
        end
    end
end

# =====================================================================
# 12. A conjugated output view is rejected; a non-conjugating `op` on C is
# not. The engine addresses `parent(C)` directly, so an `op` that conjugates
# would be silently ignored -- the one case that has to be an error rather
# than a transform. `transpose` is elementwise identity on a `Number` and must
# therefore still be accepted, and on a real eltype nothing can conjugate at
# all.
# =====================================================================

@testset "macro driver: conjugated output view rejected, non-conjugating op accepted" begin
    for T in _MACRO_COMPLEX_ELTYPES
        kernel = first(_macro_complex_kernels(T))
        Ma, Ka, Na = 9, 7, 6
        Amat = randn(MersenneTwister(0xC0C0_0001), T, Ma, Ka)
        Bmat = randn(MersenneTwister(0xC0C0_0002), T, Ka, Na)

        for op in _MACRO_OPS
            Cmat = zeros(T, Ma, Na)
            mkplan() = QuasiStrided.plan_contract(
                _macro_op_view(Cmat, op), StridedView(Amat), (1, 2),
                StridedView(Bmat), (2, 3), (1, 3); kernel = kernel, mc = 4, kc = 3, nc = 4
            )
            if _macro_op_conjugates(op)
                @test_throws ArgumentError mkplan()
            else
                QuasiStrided.execute!(mkplan(), one(T), zero(T))
                @test isapprox(Cmat, Amat * Bmat; rtol = _macro_rtol(T, Ka))
            end
        end
    end

    # Real path: `conj`/`adjoint` on a real view cannot conjugate anything, so
    # nothing is rejected and `conjA`/`conjB` are inert.
    Ma, Ka, Na = 9, 7, 6
    Ar = randn(MersenneTwister(0xC0C0_0011), Ma, Ka)
    Br = randn(MersenneTwister(0xC0C0_0012), Ka, Na)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    for op in _MACRO_OPS
        Cr = zeros(Ma, Na)
        plan = QuasiStrided.plan_contract(
            _macro_op_view(Cr, op), _macro_op_view(Ar, op), (1, 2),
            _macro_op_view(Br, op), (2, 3), (1, 3);
            kernel = kernel, conjA = true, conjB = true, mc = 4, kc = 3, nc = 4
        )
        @test typeof(plan.atransform) === typeof(identity)
        @test typeof(plan.btransform) === typeof(identity)
        QuasiStrided.execute!(plan, 1.0, 0.0)
        @test isapprox(Cr, Ar * Br; rtol = _macro_rtol(Float64, Ka))
    end
end

# =====================================================================
# 13. The two conjugation sources really are independent and really do XOR.
#
# Testset 9 covers this statistically (it draws the cancelling combination),
# but the two discriminating cases are worth pinning by name, because each
# distinguishes the frozen rule from a plausible wrong one:
#
#   * `conjA = true` on a `conj`-op'd view must CANCEL to the unconjugated
#     operand. An engine combining the two sources with `||` (or applying
#     them in sequence) passes every non-cancelling case and fails only this.
#   * an `adjoint`-op'd view with no flag must conjugate. An engine testing
#     `v.op === conj` -- which is what TensorOperations' own TBLIS extension
#     does -- passes every other `op` and fails only this.
#
# Both are asserted against a *materialized* reference and its negation, so
# "agrees with the oracle" cannot be satisfied by accident.
# =====================================================================

@testset "macro driver: conj flag x op compose with XOR, not with ||" begin
    for T in _MACRO_COMPLEX_ELTYPES
        kernel = first(_macro_complex_kernels(T))
        Ma, Ka, Na = 11, 9, 7
        Amat = randn(MersenneTwister(0x0BAD_0001), T, Ma, Ka)
        Bmat = randn(MersenneTwister(0x0BAD_0002), T, Ka, Na)
        rtol = _macro_rtol(T, Ka)

        @testset "conjA = true on a conj-op view cancels ($T)" begin
            Cmat = zeros(T, Ma, Na)
            QuasiStrided.execute!(
                _dense_plan_conj(Cmat, Amat, Bmat, kernel, 4, 3, 4, true, false, conj, identity),
                one(T), zero(T)
            )
            @test isapprox(Cmat, Amat * Bmat; rtol = rtol)
            # ... and the cancellation is not vacuous: the conjugated answer
            # really is a different matrix.
            @test !isapprox(Cmat, conj.(Amat) * Bmat; rtol = rtol)
        end

        @testset "adjoint op conjugates even with no flag set ($T)" begin
            Cmat = zeros(T, Ma, Na)
            QuasiStrided.execute!(
                _dense_plan_conj(Cmat, Amat, Bmat, kernel, 4, 3, 4, false, false, adjoint, identity),
                one(T), zero(T)
            )
            @test isapprox(Cmat, conj.(Amat) * Bmat; rtol = rtol)
            @test !isapprox(Cmat, Amat * Bmat; rtol = rtol)
        end

        @testset "transpose op does not conjugate ($T)" begin
            Cmat = zeros(T, Ma, Na)
            QuasiStrided.execute!(
                _dense_plan_conj(Cmat, Amat, Bmat, kernel, 4, 3, 4, false, false, identity, transpose),
                one(T), zero(T)
            )
            @test isapprox(Cmat, Amat * Bmat; rtol = rtol)
            @test !isapprox(Cmat, Amat * conj.(Bmat); rtol = rtol)
        end
    end
end
