# Standalone correctness check for benchmark/composite_backend.jl's
# `QuasiStridedComposite`. Run from the ROOT project environment (only needs
# QuasiStrided + TensorOperations + stdlib Test, not benchmark/'s own
# environment or the upstream TensorOperationsBenchmarks suite):
#
#   julia --project=. benchmark/check_composite_backend.jl
#
# Fixture patterns reused from test/test_tensoroperations.jl (oracle/call-form
# conventions, poison!, the frozen worked-example pA/pB/pAB shape).

using Test
using Random
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
using TensorOperations
using TensorOperations: StridedNative
using StridedViews: StridedView

include(joinpath(@__DIR__, "composite_backend.jl"))

const qscomposite = QuasiStridedComposite()
const qsbackend = QuasiStridedBackend()
const to_native = StridedNative()

poison!(C) = fill!(C, convert(eltype(C), NaN))

# `pA, pB, pAB` for the plain matmul C[i,j] = A[i,k]*B[k,j].
const _MATMUL_PAB = ((1,), (2,)), ((1,), (2,)), ((1, 2), ())

@testset "QuasiStridedComposite" begin

    @testset "tensorcontract! under the composite matches QuasiStridedBackend directly" begin
        # Same shape/permutation family as test_tensoroperations.jl's frozen
        # worked example: A has 5 axes (3 open, 2 contracted), B has 4 axes
        # (2 contracted, 2 open).
        Random.seed!(1234567)
        T = Float64
        A = randn(T, (3, 20, 5, 3, 4))
        B = randn(T, (4, 6, 20, 3))
        pA = ((3, 1, 4), (2, 5))
        pB = ((3, 1), (4, 2))
        pAB = ((3, 1, 4), (5, 2))
        α, β = rand(T), rand(T)

        Cq = randn(T, (3, 5, 3, 6, 3))
        Cc = copy(Cq)

        Rq = tensorcontract!(Cq, A, pA, false, B, pB, false, pAB, α, β, qsbackend)
        Rc = tensorcontract!(Cc, A, pA, false, B, pB, false, pAB, α, β, qscomposite)

        @test all(isfinite, Rc)
        @test Rc == Rq  # bitwise-identical: both dispatch to the identical QuasiStridedBackend() call

        # And via `@tensor`, matching the repo's call-form convention.
        M = randn(T, (6, 8))
        N = randn(T, (8, 7))
        @tensor backend = qsbackend Dref[i, j] := M[i, k] * N[k, j]
        @tensor backend = qscomposite Dcomp[i, j] := M[i, k] * N[k, j]
        @test Dcomp == Dref
    end

    @testset "tensoradd!/tensortrace! under the composite match StridedNative directly" begin
        Random.seed!(7654321)
        T = Float64

        # tensoradd! (a permutation).
        A = randn(T, (4, 5, 6))
        pA = ((2, 1, 3), ())
        Cn = randn(T, (5, 4, 6))
        Cc = copy(Cn)
        Rn = tensoradd!(Cn, A, pA, false, one(T), zero(T), to_native)
        Rc = tensoradd!(Cc, A, pA, false, one(T), zero(T), qscomposite)
        @test Rc == Rn

        # tensortrace!.
        B = randn(T, (4, 3, 3, 5))
        p = ((1,), (4,))
        q = ((2,), (3,))
        Tn = randn(T, (4, 5))
        Tc = copy(Tn)
        Rtn = tensortrace!(Tn, B, p, q, false, one(T), zero(T), to_native)
        Rtc = tensortrace!(Tc, B, p, q, false, one(T), zero(T), qscomposite)
        @test Rtc == Rtn

        # And via `@tensor`: a pure permutation, and a pure trace.
        @tensor backend = to_native Pref[j, i, k] := A[i, j, k]
        @tensor backend = qscomposite Pcomp[j, i, k] := A[i, j, k]
        @test Pcomp == Pref

        @tensor backend = to_native Tref[i, l] := B[i, j, j, l]
        @tensor backend = qscomposite Tcomp[i, l] := B[i, j, j, l]
        @test Tcomp == Tref
    end

    @testset "ineligible contraction still throws under the composite (no silent fallback)" begin
        # Mixed real/complex operands: the frozen eligibility predicate
        # (_qs_eltype_ok in src/tensoroperations.jl) rejects this outright.
        pA, pB, pAB = _MATMUL_PAB
        A = randn(Float64, (3, 4))
        B = randn(ComplexF64, (4, 5))
        C = zeros(ComplexF64, (3, 5))
        @test_throws ArgumentError tensorcontract!(
            C, A, pA, false, B, pB, false, pAB, 1, 0, qscomposite
        )

        # tensoradd!/tensortrace! also still hard-reject under the composite's
        # own QuasiStridedBackend() call for tensorcontract! (this is testing
        # the *contraction* path specifically); addtrace itself is a working
        # StridedNative() call and is not expected to reject anything.
    end

    @testset "@tensor network with both a permutation and a contraction" begin
        Random.seed!(112233)
        T = Float64
        A = randn(T, (5, 5, 5, 5))
        B = randn(T, (5, 5, 5))
        C = randn(T, (5, 5, 5))

        # A contraction network (needs tensorcontract!) ...
        @tensor backend = qscomposite D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
        @tensor Dref[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
        @test D ≈ Dref

        # ... followed by a standalone permutation of the contraction's result
        # (needs tensoradd!), in a second `@tensor` statement.
        @tensor backend = qscomposite E[b, d, a, c] := D[a, b, c, d]
        @tensor Eref[b, d, a, c] := Dref[a, b, c, d]
        @test E == Eref
    end

    @testset "both allocator paths are exercised" begin
        Random.seed!(90210)
        T = Float64
        A = randn(T, (20, 30))
        Bm = randn(T, (30, 25))
        Cref = A * Bm

        # DefaultAllocator path (implicit).
        C1 = zeros(T, (20, 25))
        @tensor backend = qscomposite C1[i, j] = A[i, k] * Bm[k, j]
        @test C1 ≈ Cref

        # Explicit-allocator path: TensorOperations' built-in ManualAllocator,
        # as exercised by test/test_tensoroperations.jl's workspace-pooling
        # tests (kept dependency-free here rather than reaching for the
        # Bumper-backed buffer that file also exercises).
        C2 = zeros(T, (20, 25))
        @tensor backend = qscomposite allocator = TensorOperations.ManualAllocator() C2[i, j] = A[i, k] * Bm[k, j]
        @test C2 ≈ Cref

        # Same pairing for the addtrace path: a plain permutation under both
        # the default and an explicit allocator.
        P = randn(T, (4, 5, 6))
        Q1 = zeros(T, (5, 4, 6))
        @tensor backend = qscomposite Q1[j, i, k] = P[i, j, k]
        @tensor Qref[j, i, k] := P[i, j, k]
        @test Q1 == Qref

        Q2 = zeros(T, (5, 4, 6))
        @tensor backend = qscomposite allocator = TensorOperations.ManualAllocator() Q2[j, i, k] = P[i, j, k]
        @test Q2 == Qref
    end
end
