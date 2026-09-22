# TO-comparison test suite for `QuasiStrided.QuasiStridedBackend`, written
# against the documented adapter contract rather than its implementation
# (docs/decisions.md, "Correctness oracle: cross-package, one deliberate
# deviation from precedent"). Included by `test/runtests.jl`, which already provides
# `Test`/`Random`/`QuasiStrided`.

# Standalone-run support: `runtests.jl` supplies `Test`/`Random`/`QuasiStrided`
# before including this file, and `helpers.jl` supplies the unqualified
# `plan_contract` binding (see `runtests.jl`'s comment on why it is not restored
# there). Both are guarded, so including this file on its own works and the
# `runtests.jl` path is bit-for-bit unaffected.
if !@isdefined(QuasiStrided)
    using Test
    using Random
    using QuasiStrided
end
if !@isdefined(plan_contract)
    const plan_contract = QuasiStrided.plan_contract
end

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
# The eligible eltypes are `_QS_ELTYPES = (Float32, Float64, ComplexF32,
# ComplexF64)`. `eltypes` holds the real ones only, so that real-only
# assertions below stay real-only; the complex types are kept alongside
# rather than folded in.
const complex_eltypes = (ComplexF32, ComplexF64)
const all_eltypes = (eltypes..., complex_eltypes...)

# `pA, pB, pAB` for the plain matmul C[i,j] = A[i,k]*B[k,j], used by most of
# the edge-case and hard-reject testsets below.
const _MATMUL_PAB = ((1,), (2,)), ((1,), (2,)), ((1, 2), ())

@testset "QuasiStridedBackend export" begin
    # Frozen: QuasiStridedBackend is the *only* exported name.
    @test QuasiStrided.QuasiStridedBackend === QuasiStrided.QuasiStridedBackend
    @test QuasiStridedBackend <: TensorOperations.AbstractBackend
    @test QuasiStridedBackend() isa QuasiStridedBackend
end

@testset "tensorcontract! agrees with StridedNative/StridedBLAS (eltype = $T)" for T in all_eltypes
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

@testset "tensorcontract!: another permutation shape (eltype = $T)" for T in all_eltypes
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

@testset "conj is real conjugation for complex eltype, and still a no-op for real (eltype = $T)" for T in all_eltypes
    # Real half: the load-bearing real-path guard. Real operands with
    # conjA/conjB set true must still agree with StridedNative(), which itself
    # treats conj as a no-op on reals.
    #
    # Complex half: the same loop, non-trivial. `conjA`/`conjB` select `conj`
    # as the operand transform, and the
    # result is checked against StridedNative() *and* against an explicitly
    # conjugated matmul, so the oracle does not rest on TO alone.
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
        if T <: Complex
            @test Rq ≈ (conjA ? conj(A) : A) * (conjB ? conj(B) : B)
            # ... and the flags must actually *do* something: silently dropping
            # them would still pass `Rq ≈ Rn` above if `Rn` were computed the
            # same wrong way.
            if conjA || conjB
                @test !isapprox(Rq, A * B)
            end
        else
            # `StridedViews` defines `conj(::StridedView{<:Real}) = a`, so no
            # conjugation can ever reach the real path, whatever the flags say.
            @test Rq ≈ A * B
        end
    end
end

# =========================================================================
# Conjugation, the part the flag-only loop above cannot reach.
#
# docs/decisions.md, "Conjugation: semantics, and where each piece is
# absorbed": there are two *independent* sources of conjugation per input --
# TO's `conjA`/`conjB` flags and `StridedView.op` -- and they compose with
# **xor**:
#
#     _qs_isconj(v, flag) = (eltype(v) <: Complex) && (flag ⊻ _op_conjugates(v.op))
#
# `α`/`β` are *not* conjugated. Setting `conjA` alone never sets `op`, so no
# amount of flag-only testing distinguishes `⊻` from `||`; these testsets do.
# =========================================================================

@testset "conjugation: StridedView.op x conj flag cross-product (eltype = $T)" for T in complex_eltypes
    Random.seed!(20260914)
    pA, pB, pAB = _MATMUL_PAB
    # Square, so that every wrapper below leaves the matmul well-formed.
    M = randn(T, (4, 4))
    N = randn(T, (4, 4))

    wrappers = (identity, adjoint, transpose, conj)
    # Two families. `Base`'s wrappers around a plain `Array` are what a
    # TensorOperations user actually passes; the same wrappers around a
    # `StridedView` are what sets `op` *lazily*. The distinction matters:
    # `conj(::Matrix)` materialises (the data is conjugated and `op` stays
    # `identity`), whereas `conj(::StridedView)` only flips `op`.
    variants(X) = vcat(
        [w(X) for w in wrappers],
        [w(StridedView(X)) for w in wrappers],
    )

    for Aw in variants(M), Bw in variants(N), conjA in (false, true), conjB in (false, true)
        # Materialised equivalents: `collect` on a `StridedView` applies `op`,
        # so these carry exactly the values the wrapped operand denotes. The
        # reference therefore never depends on how any backend treats `op`.
        Am = collect(Aw)
        Bm = collect(Bw)

        Cn = fill(convert(T, NaN), (4, 4))
        Cq = copy(Cn)
        Rn = tensorcontract!(Cn, Aw, pA, conjA, Bw, pB, conjB, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, Aw, pA, conjA, Bw, pB, conjB, pAB, one(T), zero(T), qsbackend)

        @test all(isfinite, Rq)
        @test Rq ≈ Rn
        @test Rq ≈ (conjA ? conj(Am) : Am) * (conjB ? conj(Bm) : Bm)
    end

    @testset "the xor, isolated" begin
        # THE test of this file: `conj(A)` with `conjA = true` must cancel to
        # the *unconjugated* operand. A `||` in place of the `⊻` passes every
        # other test in this testset and fails this one.
        Ac = conj(StridedView(M))
        @test Ac.op === conj              # lazy, not materialised
        @test collect(Ac) == conj(M)

        C = fill(convert(T, NaN), (4, 4))
        R = tensorcontract!(C, Ac, pA, true, N, pB, false, pAB, one(T), zero(T), qsbackend)
        @test R ≈ M * N
        @test !isapprox(R, conj(M) * N)

        # And symmetrically on the B operand.
        Bc = conj(StridedView(N))
        C2 = fill(convert(T, NaN), (4, 4))
        R2 = tensorcontract!(C2, M, pA, false, Bc, pB, true, pAB, one(T), zero(T), qsbackend)
        @test R2 ≈ M * N
        @test !isapprox(R2, M * conj(N))

        # Both at once: two cancellations, not four conjugations.
        C3 = fill(convert(T, NaN), (4, 4))
        R3 = tensorcontract!(C3, Ac, pA, true, Bc, pB, true, pAB, one(T), zero(T), qsbackend)
        @test R3 ≈ M * N
    end

    @testset "α and β are not conjugated" begin
        # `alpha` and `beta` are **not** conjugated, because `conjA` applies to
        # A's *data* only. With a complex α/β and a conjugated operand, a scalar
        # that was wrongly conjugated along with the data is visible here and
        # nowhere else.
        α = convert(T, 0.75 - 1.25im)
        β = convert(T, -0.5 + 2.0im)
        Ac = conj(StridedView(M))
        C0 = randn(T, (4, 4))
        Cq = copy(C0)
        R = tensorcontract!(Cq, Ac, pA, false, N, pB, false, pAB, α, β, qsbackend)
        @test R ≈ β * C0 + α * (conj(M) * N)
    end
end

@testset "conjugation: _op_conjugates is the frozen table, with a throwing fallback" begin
    # The table as a pure function, checked on its own so that it is exercised
    # independently of any complex kernel. An `op` that is
    # not one of the four must throw rather than be silently treated as
    # unconjugated ("Hard-reject, never fall back"); TBLIS's `A.op === conj`
    # test is what this replaces.
    @test QuasiStrided._op_conjugates(identity) === false
    @test QuasiStrided._op_conjugates(conj) === true
    @test QuasiStrided._op_conjugates(transpose) === false
    @test QuasiStrided._op_conjugates(adjoint) === true
    @test_throws ArgumentError QuasiStrided._op_conjugates(-)
end

@testset "conjugation: every StridedView.op is honoured (eltype = $T)" for T in complex_eltypes
    # `StridedView(parent, size, strides, offset, op)` is directly
    # constructible for every `op` in `Union{identity, conj, adjoint,
    # transpose}`, including the two that `StridedViews`' own arithmetic never
    # produces for a `Number` eltype. TBLIS's `A.op === conj` test would treat
    # a directly-constructed `adjoint` view as unconjugated -- silently.
    # `_op_conjugates` is a total table precisely to close that.
    Random.seed!(161803)
    pA, pB, pAB = _MATMUL_PAB
    M = randn(T, (4, 4))
    N = randn(T, (4, 4))

    table = (
        (identity, false),
        (conj, true),
        (transpose, false),   # elementwise identity on a `Number`
        (adjoint, true),
    )

    for (op, op_conjugates) in table, conjA in (false, true)
        Av = StridedView(M, size(M), strides(M), 0, op)
        @test collect(Av) == (op_conjugates ? conj(M) : M)

        expected = ((op_conjugates ⊻ conjA) ? conj(M) : M) * N
        C = fill(convert(T, NaN), (4, 4))
        R = tensorcontract!(C, Av, pA, conjA, N, pB, false, pAB, one(T), zero(T), qsbackend)
        @test all(isfinite, R)
        @test R ≈ expected
    end

end

@testset "hard-reject: conjugated output view (eltype = $T)" for T in complex_eltypes
    # A conjugated output `C` is rejected, matching TO's own TBLIS extension
    # (`isconj(SV(C), false) && throw_conj_output(f)`). The rejection lives in
    # `plan_contract` rather than in the adapter (docs/decisions.md, 'Second
    # addendum to "Required argument-checking order in the adapter
    # (frozen)"'), so that a caller reaching the engine directly is protected
    # too -- but it must still surface as an `ArgumentError` from
    # `tensorcontract!`.
    Random.seed!(271828)
    pA, pB, pAB = _MATMUL_PAB
    A = randn(T, (4, 4))
    B = randn(T, (4, 4))

    Cc = conj(StridedView(zeros(T, (4, 4))))
    @test Cc.op === conj
    @test_throws ArgumentError tensorcontract!(
        Cc, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend
    )

    # Same rejection for the directly-constructed `adjoint` op, which is the
    # case a `=== conj` test would miss.
    Ca = StridedView(zeros(T, (4, 4)), (4, 4), (1, 4), 0, adjoint)
    @test_throws ArgumentError tensorcontract!(
        Ca, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend
    )

    # `transpose` does *not* conjugate, so a transposed output view is a
    # permutation and stays acceptable.
    Ct = StridedView(zeros(T, (4, 4)), (4, 4), (1, 4), 0, transpose)
    Rt = tensorcontract!(Ct, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)
    @test collect(Rt) ≈ A * B
end

@testset "real adjoint output is still accepted (eltype = $T)" for T in eltypes
    # The other half of the pair above, and the place the real path could most
    # easily break: `StridedViews` defines `conj(::StridedView{<:Real}) = a`,
    # so `op` is always `identity` for a real eltype and the conjugated-output
    # rejection must never fire on the real path, however the output is
    # wrapped.
    Random.seed!(141421)
    pA, pB, pAB = _MATMUL_PAB
    A = randn(T, (4, 4))
    B = randn(T, (4, 4))

    @test conj(StridedView(zeros(T, (4, 4)))).op === identity

    Cadj = adjoint(zeros(T, (4, 4)))
    Rq = tensorcontract!(Cadj, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)
    @test collect(Rq) ≈ A * B

    Cconj = conj(StridedView(zeros(T, (4, 4))))
    Rc = tensorcontract!(Cconj, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)
    @test collect(Rc) ≈ A * B
end

@testset "tensorcontract! edge cases (eltype = $T)" for T in all_eltypes
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

@testset "@tensor / ncon integration (eltype = $T)" for T in all_eltypes
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

@testset "accepts complex eltypes" begin
    # ComplexF32/ComplexF64 are in `_QS_ELTYPES`, so a complex contraction
    # must be *served* by this engine, neither rejected nor silently handed to
    # a fallback.
    pA, pB, pAB = _MATMUL_PAB
    @testset "eltype = $T" for T in complex_eltypes
        Random.seed!(31415)
        A = randn(T, (3, 4))
        B = randn(T, (4, 5))
        Cn = fill(convert(T, NaN), (3, 5))
        Cq = copy(Cn)
        Rn = tensorcontract!(Cn, A, pA, false, B, pB, false, pAB, one(T), zero(T), to_native)
        Rq = tensorcontract!(Cq, A, pA, false, B, pB, false, pAB, one(T), zero(T), qsbackend)
        @test all(isfinite, Rq)
        @test Rq ≈ Rn
        @test Rq ≈ A * B
        @test eltype(Rq) === T
    end
end

@testset "hard-reject: residual complex eltypes" begin
    # `Complex{Float16}`, `Complex{Int}` and `Complex{BigFloat}` are rejected
    # (not in the tuple). These are the eltypes that
    # *look* complex but are outside `_QS_ELTYPES`; a widened check that tested
    # `T <: Complex` instead of membership would wrongly accept them.
    pA, pB, pAB = _MATMUL_PAB

    @testset "Complex{Float16}" begin
        A = Complex{Float16}.(randn(ComplexF32, (3, 4)))
        B = Complex{Float16}.(randn(ComplexF32, (4, 5)))
        C = zeros(Complex{Float16}, (3, 5))
        @test_throws ArgumentError tensorcontract!(
            C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end

    @testset "Complex{Int}" begin
        A = Complex{Int}.(rand(-9:9, (3, 4)), rand(-9:9, (3, 4)))
        B = Complex{Int}.(rand(-9:9, (4, 5)), rand(-9:9, (4, 5)))
        C = zeros(Complex{Int}, (3, 5))
        @test_throws ArgumentError tensorcontract!(
            C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end

    @testset "Complex{BigFloat}" begin
        A = Complex{BigFloat}.(randn(ComplexF64, (3, 4)))
        B = Complex{BigFloat}.(randn(ComplexF64, (4, 5)))
        C = zeros(Complex{BigFloat}, (3, 5))
        @test_throws ArgumentError tensorcontract!(
            C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end
end

@testset "hard-reject: mixed complex precisions (ComplexF32 A, ComplexF64 B)" begin
    # The eltype predicate is `eltype(A) === eltype(B) === eltype(C)`; admitting
    # complex eltypes must not weaken the *shared*-eltype half of it.
    A = randn(ComplexF32, (3, 4))
    B = randn(ComplexF64, (4, 5))
    C = zeros(ComplexF64, (3, 5))
    pA, pB, pAB = _MATMUL_PAB
    @test_throws ArgumentError tensorcontract!(
        C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
    )
end

@testset "hard-reject: mixed real/complex operands (Float64 A, ComplexF64 B)" begin
    # "Mixed real-A / complex-B is out of scope" -- deliberately, not by
    # oversight: promotion belongs in TO's `promote_contract` layer, and a
    # silent materialising promotion here would break the guarantee that a
    # timing taken with this backend always measures this engine.
    A = randn(Float64, (3, 4))
    B = randn(ComplexF64, (4, 5))
    C = zeros(ComplexF64, (3, 5))
    pA, pB, pAB = _MATMUL_PAB
    @test_throws ArgumentError tensorcontract!(
        C, A, pA, false, B, pB, false, pAB, 1, 0, qsbackend
    )

    # ... and with the operands the other way round.
    A2 = randn(ComplexF64, (3, 4))
    B2 = randn(Float64, (4, 5))
    C2 = zeros(ComplexF64, (3, 5))
    @test_throws ArgumentError tensorcontract!(
        C2, A2, pA, false, B2, pB, false, pAB, 1, 0, qsbackend
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
    # Only the `_QS_ELTYPES` are eligible; every other eltype, real ones
    # included, must be rejected the same way.
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

    @testset "complex output aliasing an input through a wrapper (eltype = $T)" for T in complex_eltypes
        # The aliasing check runs on the `StridedView`-wrapped operands, which
        # unwrap `PermutedDimsArray`/`Adjoint` down to the shared parent. That
        # must hold for complex operands too -- conjugated-C rejection comes
        # *after* aliasing in the frozen order (eligibility -> argcheck ->
        # dimcheck -> wrap -> aliasing -> conjugated-C rejection), and
        # reordering it away would let a conjugated-but-aliased call through
        # with the wrong error, or none.
        P = randn(T, (5, 5))
        B = randn(T, (5, 5))

        Cpd = PermutedDimsArray(P, (2, 1))
        @test !Base.mightalias(Cpd, P)                              # the Base gap
        @test Base.mightalias(StridedView(Cpd), StridedView(P))     # what closes it
        @test_throws ArgumentError tensorcontract!(
            Cpd, P, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )

        # Adjoint of a complex parent: `StridedView(P')` unwraps to `P`'s
        # buffer (with `op === conj`), so this is an aliased *and* conjugated
        # output. Either rejection is correct; silently proceeding is not.
        Cadj = P'
        @test_throws ArgumentError tensorcontract!(
            Cadj, P, pA, false, B, pB, false, pAB, 1, 0, qsbackend
        )
    end
end

@testset "tensoradd! falls back to StridedNative (amended 2026-09-16)" begin
    A = randn(Float64, (3, 4))
    C1 = zeros(Float64, (4, 3))
    @tensor backend = qsbackend C1[j, i] = A[i, j]
    C2 = zeros(Float64, (4, 3))
    @tensor backend = StridedNative() C2[j, i] = A[i, j]
    @test C1 == C2

    # A network mixing a contraction with an add/trace step must run
    # wholesale under `qsbackend`, not throw.
    B = randn(Float64, (4, 5))
    D1 = zeros(Float64, (3, 5))
    @tensor backend = qsbackend D1[i, k] = A[i, j] * B[j, k]
    @test D1 ≈ A * B

    # The fallback forwards the allocator argument through unchanged.
    C3 = zeros(Float64, (4, 3))
    @tensor backend = qsbackend allocator = TensorOperations.ManualAllocator() C3[j, i] = A[i, j]
    @test C3 == C1
end

@testset "tensortrace! falls back to StridedNative (amended 2026-09-16)" begin
    A = randn(Float64, (4, 3, 3))
    C1 = zeros(Float64, 4)
    @tensor backend = qsbackend C1[i] = A[i, j, j]
    C2 = zeros(Float64, 4)
    @tensor backend = StridedNative() C2[i] = A[i, j, j]
    @test C1 == C2
end

# =====================================================================
# Workspace-pooling tests. Unlike the sections above these are written WITH
# knowledge of src/integrations/tensoroperations.jl, since they pin the task-local pooling
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
        # 1.10's compiler, so this
        # needs the same skip execution/test_execute.jl/microkernels/test_simd_kernel.jl use.
        steady1 = @allocated run_once()
        steady2 = @allocated run_once()
        @test pool[T] === ws  # same workspace object, not rebuilt
        @test steady1 < 20_000 skip = (VERSION < v"1.11")
        @test steady2 < 20_000 skip = (VERSION < v"1.11")
        steady2
    end

    # A cold, unpooled `plan_contract` call for the same shape/kernel
    # allocates its own fresh `ContractWorkspace` from scratch, well above the
    # pooled steady state above.
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
# and then reused oversized by a smaller problem. (execution/test_workspace.jl covers
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

@testset "workspace pooling: real and complex eltypes share one task-local pool" begin
    # The complex round of the testset above. Per "Buffer element type: the
    # `VT` bound relaxes, the arity does not": the pool is keyed by
    # `eltype(C)` (so `Float64` and `ComplexF64` get *distinct* workspaces and
    # cannot couple each other's grow-only `reserve!` footprints), while the
    # buffer element type is `real(T)` -- i.e. the complex workspace is
    # `ContractWorkspace{ComplexF64, Vector{Float64}}`.
    Random.seed!(8675309)
    _qs_with_clean_pool() do
        for T in (Float64, ComplexF64, Float64, ComplexF64, ComplexF32, Float64)
            A = randn(T, (12, 7))
            Bm = randn(T, (7, 9))
            Cref = A * Bm
            C = zeros(T, (12, 9))
            @tensor backend = qsbackend C[i, j] = A[i, k] * Bm[k, j]
            @test eltype(C) === T
            @test C ≈ Cref
        end

        pool = task_local_storage(QuasiStrided._QS_WORKSPACE_KEY)
        @test haskey(pool, Float64)
        @test haskey(pool, ComplexF64)
        @test haskey(pool, ComplexF32)
        @test pool[Float64] !== pool[ComplexF64]
        @test pool[ComplexF32] !== pool[ComplexF64]
        @test pool[Float64] isa QuasiStrided.ContractWorkspace{Float64, Vector{Float64}}
        @test pool[ComplexF64] isa QuasiStrided.ContractWorkspace{ComplexF64, Vector{Float64}}
        @test pool[ComplexF32] isa QuasiStrided.ContractWorkspace{ComplexF32, Vector{Float32}}

        # Re-run each eltype once more, after every other eltype has had its
        # turn at growing its own pooled workspace: a `Float64` result must not
        # be disturbed by the `ComplexF64` round that ran between its two
        # invocations, and vice versa.
        for T in (Float64, ComplexF64, ComplexF32)
            A = randn(T, (12, 7))
            Bm = randn(T, (7, 9))
            C = fill(convert(T, NaN), (12, 9))
            @tensor backend = qsbackend C[i, j] = A[i, k] * Bm[k, j]
            @test all(isfinite, C)
            @test C ≈ A * Bm
        end
    end
end
