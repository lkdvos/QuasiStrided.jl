# The vectorized planar complex store fast path (src/kernels/planar.jl,
# `_store_tile_planar_vector!`): Phase 2 of docs/proposals/complex-fast-paths.md,
# Section 3.3.
#
# Four separable things are pinned here, deliberately not mixed:
#
#  1. VALUES, at two different strengths, because the two comparisons answer
#     two different questions and only one of them can be exact:
#
#       (a) against an INDEPENDENT, optimization-barriered transcription of the
#           arithmetic the fast path claims to implement (Base's own `Complex`
#           `*`/`muladd` expression trees, the ones `_axpby_tile!` reaches
#           through) -- compared with `isequal`, BITWISE, so `-0.0` and NaN
#           payloads count. This is the strong pin: it catches a swapped
#           interleave lane, a dropped conjugate sign, a wrong `beta` regime.
#           It is asserted only on elements the fast path actually vectorizes
#           (full `W`-row blocks); the scalar row tail is excluded by
#           construction -- see (b).
#
#       (b) against the scalar `_store_tile_planar!` path itself, run on the
#           same fixture with a `ScatterAxis` destination the gate excludes --
#           compared with a tolerance, NOT bitwise. This is not a weakening for
#           convenience: LLVM SLP-vectorizes Base's
#           `muladd(::Complex, ::Complex, ::Complex)` into `<2 x double>` ops
#           carrying the `contract` fast-math flag (visible in `@code_llvm` of
#           `muladd(a, r, b*c)` at `ComplexF64`), which lets the backend fuse a
#           multiply the source text rounds separately. Whether that fires
#           depends on inlining context, so the scalar path is not
#           bit-reproducible against itself across call sites -- the row tail
#           inside `_store_tile_planar_vector!` and `_store_tile_planar!` run
#           character-identical source and still disagree on a handful of
#           elements. Requiring bitwise agreement here would be requiring
#           agreement with an LLVM heuristic. The proposal's Section 6.4
#           anticipated exactly this ("compare with a tolerance, never `==`").
#
#  2. CONTRACT PRESERVATION on the fast path specifically: `beta == 0` never
#     reads old `C`, `alpha == 0` never reads `acc`, nothing outside the valid
#     `m x n` rectangle is written, and `beta` is applied exactly once. These
#     mirror test_planar_kernel.jl's existing assertions, which were written
#     when only the scalar path existed and (for a dense destination) now
#     exercise the fast path instead -- so they are repeated here with the
#     path-firing assertion attached, rather than trusted to still mean what
#     they meant.
#
#  3. GATE FIRING (the dispatch-tiers.md D3 lesson, proposal Section 7 item 1):
#     assert the predicate is TRUE on a fixture that should fast-path and FALSE
#     for each single violated condition on its own.
#
#  4. ISA PORTABILITY. Decision 5 of the proposal ships this for AVX-512 only.
#     Every expectation is derived from `target_profile()` at run time, never
#     written as a literal, so `test/forced_isa_runner.jl` under
#     `avx2`/`neon`/`unknown` asserts the fast path is OFF and the outputs are
#     unchanged rather than failing by construction.

using Test
using Random
using QuasiStrided
using QuasiStrided: PlanarKernel, AffineAxis, ScatterAxis, DestinationTile, QSTile,
    store_tile!, target_profile, unknown_target, TargetProfile, CacheLevel,
    KERNEL_SHAPES_C64_PLANAR, KERNEL_SHAPES_C32_PLANAR
using SIMD: Vec
using StridedViews: StridedView

const QSS = QuasiStrided

# Whether the host (or the forced profile) is one the fast path ships for.
# Read once here so every expectation below is derived, never literal.
const STORE_FASTPATH_ON = QSS._complex_fastpath_isa_eligible()

# ---------------------------------------------------------------------------
# Independent reference for one output element.
#
# Written from Base's definitions (`complex.jl`), NOT from src/kernels/planar.jl:
#
#   *(z,w)        = Complex(zr*wr - zi*wi, zr*wi + zi*wr)
#   muladd(z,w,x) = Complex(muladd(zr, wr, -muladd(zi, wi, -xr)),
#                           muladd(zr, wi,  muladd(zi, wr,  xi)))
#
# `barrier` is a `@noinline` identity: it stops LLVM from contracting the
# separately-rounded products into an FMA, which is the very transform that
# makes the scalar path non-reproducible (see the header). `fma` is used where
# Base uses `muladd`, since on every ISA this gate admits `muladd` IS the fused
# operation and `fma` says so unambiguously.
# ---------------------------------------------------------------------------

@noinline barrier(x) = x

k_for(::Type{T}, MR::Int, NR::Int, W::Int) where {T} =
    PlanarKernel(Val(MR), Val(NR), T, Val(W))

function ref_axpby(alpha::T, rr::R, ri::R, beta::T, cold::T) where {T, R}
    ar, ai = real(alpha), imag(alpha)
    if iszero(beta)                                   # alpha * r
        return Complex(
            barrier(ar * rr) - barrier(ai * ri),
            barrier(ar * ri) + barrier(ai * rr)
        )
    end
    br, bi = real(beta), imag(beta)
    cr, ci = real(cold), imag(cold)
    if isone(beta)                                    # muladd(alpha, r, C)
        xr, xi = cr, ci
    else                                              # muladd(alpha, r, beta*C)
        xr = barrier(br * cr) - barrier(bi * ci)
        xi = barrier(br * ci) + barrier(bi * cr)
    end
    return Complex(
        fma(ar, rr, barrier(-fma(ai, ri, -xr))),
        fma(ar, ri, barrier(fma(ai, rr, xi)))
    )
end

# A deterministic accumulator of the right shape for one kernel.
function store_fp_acc(::Type{T}, MR::Int, NR::Int, W::Int, seed::Int) where {T}
    R = real(T)
    rng = MersenneTwister(seed)
    NA = 2 * (MR ÷ W) * NR
    return ntuple(_ -> Vec{W, R}(ntuple(_ -> R(2 * rand(rng) - 1), W)), NA)
end

store_fp_cold(::Type{T}, len::Int, seed::Int) where {T} =
    (rng = MersenneTwister(seed);
        T[Complex(real(T)(2 * rand(rng) - 1), real(T)(2 * rand(rng) - 1)) for _ in 1:len])

# The alpha/beta regimes: all three `_axpby_tile!` branches, plus the
# alpha == 1 / beta == 0 sub-case the proposal calls out, plus a purely
# imaginary beta (which zeroes one of the two cross terms) and a real beta
# (`isone` false, so still the general branch).
store_fp_ab(::Type{T}) where {T} = (
    (one(T), zero(T)),
    (T(-0.5, 0.25), zero(T)),
    (one(T), one(T)),
    (T(2, -1), one(T)),
    (T(2.5, -1), T(-1.75, 0.5)),
    (one(T), T(2, 0)),
    (T(1, 1), T(0, -3)),
)

const STORE_FP_MENUS = (
    (ComplexF64, KERNEL_SHAPES_C64_PLANAR),
    (ComplexF32, KERNEL_SHAPES_C32_PLANAR),
)

# ---------------------------------------------------------------------------
# 1. Values
# ---------------------------------------------------------------------------

@testset "planar store fast path: values ($T)" for (T, menu) in STORE_FP_MENUS
    R = real(T)
    # 8 eps: the measured worst case against the scalar path is one ULP on the
    # general-beta regime; this leaves room for a different LLVM version making
    # a different contraction choice without leaving room for a real bug.
    reltol = 8 * eps(R)

    @testset "($MR,$NR,$W)" for (MR, NR, W) in menu
        MV, NV = MR ÷ W, (MR ÷ W) * NR
        # Every extent class that matters: the full tile, a single element, a
        # single row/column, one short of full, an extent that is not a
        # multiple of W (so a straddling block exists), and one that is exactly
        # a multiple of W below MR (so a whole block is skipped, not split).
        extents = unique(
            (
                (MR, NR), (1, 1), (MR, 1), (1, NR), (MR - 1, NR),
                (max(1, MR ÷ 2 + 1), max(1, NR - 1)), (max(1, MV - 1) * W, NR),
                (max(1, MR - W), NR),
            )
        )
        for (m, n) in extents
            (m <= 0 || n <= 0) && continue
            acc = store_fp_acc(T, MR, NR, W, MR * 97 + NR * 13 + m * 7 + n)
            cold = store_fp_cold(T, m * n, MR * 31 + m * 5 + n)

            for (alpha, beta) in store_fp_ab(T)
                fast = copy(cold)
                dfast = DestinationTile(fast, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
                @test QSS._complex_vector_eligible(dfast, T) == STORE_FASTPATH_ON
                store_tile!(dfast, acc, alpha, beta, k_for(T, MR, NR, W))

                # Same physical layout, scattered rows -> gate excludes it, so
                # this is the scalar `_store_tile_planar!` on the same data.
                scal = copy(cold)
                dscal = DestinationTile(
                    scal, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n)
                )
                @test !QSS._complex_vector_eligible(dscal, T)
                store_tile!(dscal, acc, alpha, beta, k_for(T, MR, NR, W))

                for j in 0:(n - 1), i in 0:(m - 1)
                    v, lane = i ÷ W, (i % W) + 1
                    idx = v + MV * j + 1
                    want = ref_axpby(
                        alpha, acc[idx][lane], acc[NV + idx][lane], beta, cold[i + j * m + 1]
                    )
                    got = fast[i + j * m + 1]

                    # (1a) bitwise, but only where the fast path vectorizes:
                    # a row in the straddling block goes through the scalar
                    # tail, which inherits the scalar path's LLVM contraction.
                    in_full_block = (i < (m ÷ W) * W)
                    if in_full_block && STORE_FASTPATH_ON
                        @test isequal(want, got)
                    end
                    # (1b) tolerance, everywhere, against the shipped fallback.
                    ref = scal[i + j * m + 1]
                    @test abs(got - ref) <= reltol * max(abs(ref), one(R))
                    # ... and against the independent reference, everywhere.
                    @test abs(got - want) <= reltol * max(abs(want), one(R))
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 2. Contract preservation on the fast path
# ---------------------------------------------------------------------------

@testset "planar store fast path: contracts ($T)" for (T, menu) in STORE_FP_MENUS
    R = real(T)
    MR, NR, W = first(menu)
    k = PlanarKernel(Val(MR), Val(NR), T, Val(W))
    MV, NV = MR ÷ W, (MR ÷ W) * NR
    acc = store_fp_acc(T, MR, NR, W, 4242)

    @testset "beta == 0 never reads old C (a full, vectorized block)" begin
        # Nonfinite old C on a FULL tile, so every element goes through the
        # vector branch, not the scalar tail.
        poison = fill(T(Inf, NaN), MR * NR)
        d = DestinationTile(poison, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        @test QSS._complex_vector_eligible(d, T) == STORE_FASTPATH_ON
        store_tile!(d, acc, one(T), zero(T), k)
        @test all(isfinite, poison)
    end

    @testset "alpha == 0 never reads acc" begin
        nanacc = ntuple(_ -> Vec{W, R}(R(NaN)), Val(2NV))
        target = fill(T(2, 3), MR * NR)
        d = DestinationTile(target, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        store_tile!(d, nanacc, zero(T), T(2, 0), k)
        @test all(==(T(4, 6)), target)
    end

    @testset "nothing outside the valid rectangle is touched" begin
        # A guarded buffer: the tile occupies a sub-rectangle with a column
        # stride strictly larger than its own row count, so both the padding
        # rows (m..MR-1) and the gap between columns are observable.
        m, n = MR - 3, max(1, NR - 1)
        lda = MR + 5
        guard = T(-7, 11)
        storage = fill(guard, lda * (NR + 2) + 9)
        base = 4
        d = DestinationTile(storage, base, AffineAxis(0, 1, m), AffineAxis(0, lda, n))
        @test QSS._complex_vector_eligible(d, T) == STORE_FASTPATH_ON
        store_tile!(d, acc, T(2, -1), zero(T), k)
        for idx in eachindex(storage)
            addr = idx - 1 - base                     # zero-based tile address
            inside = false
            if 0 <= addr
                j, i = divrem(addr, lda)
                inside = (0 <= i < m) && (0 <= j < n)
            end
            inside || @test storage[idx] === guard
        end
    end

    @testset "beta is applied exactly once" begin
        # alpha == 0 short-circuits to `scale_tile!`; with a nonzero alpha and
        # a zero accumulator the result must be exactly `beta * C`, not
        # `beta^2 * C` or `beta * (beta * C)`.
        zeroacc = ntuple(_ -> Vec{W, R}(zero(R)), Val(2NV))
        cold = store_fp_cold(T, MR * NR, 99)
        got = copy(cold)
        d = DestinationTile(got, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        beta = T(-1.75, 0.5)
        store_tile!(d, zeroacc, one(T), beta, k)
        for idx in eachindex(cold)
            want = muladd(one(T), zero(T), beta * cold[idx])
            @test abs(got[idx] - want) <= 8 * eps(R) * max(abs(want), one(R))
        end
    end

    @testset "empty destination is a no-op" begin
        empty_storage = T[]
        ed = DestinationTile(empty_storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 0, 0))
        @test store_tile!(ed, acc, one(T), zero(T), k) === ed
    end
end

# ---------------------------------------------------------------------------
# 3. Gate firing: one violated condition at a time
# ---------------------------------------------------------------------------

@testset "planar store fast path: eligibility gate" begin
    T = ComplexF64
    m, n = 16, 4
    storage = zeros(T, 4 * m * n)

    ok = DestinationTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, m, n))
    @test QSS._complex_vector_eligible(ok, T) == STORE_FASTPATH_ON

    # Each of the following violates exactly one clause.
    @test !QSS._complex_vector_eligible(
        DestinationTile(storage, 0, AffineAxis(0, 2, m), AffineAxis(0, 2m, n)), T
    )                                                         # strided rows
    @test !QSS._complex_vector_eligible(
        DestinationTile(storage, m - 1, AffineAxis(0, -1, m), AffineAxis(0, m, n)), T
    )                                                         # negative stride
    @test !QSS._complex_vector_eligible(
        DestinationTile(storage, 0, ScatterAxis(collect(0:(m - 1)), m), AffineAxis(0, m, n)), T
    )                                                         # scattered rows
    @test !QSS._complex_vector_eligible(
        DestinationTile(view(storage, 1:(m * n)), 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), T
    )                                                         # non-dense storage
    @test !QSS._complex_vector_eligible(
        DestinationTile(reshape(storage, 4m, n), 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), T
    )                                                         # rank-2 storage
    @test !QSS._complex_vector_eligible(
        DestinationTile(zeros(ComplexF32, m * n), 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), T
    )                                                         # wrong element type
    @test !QSS._complex_vector_eligible(
        DestinationTile(zeros(Float64, m * n), 0, AffineAxis(0, 1, m), AffineAxis(0, m, n)), T
    )                                                         # real storage

    # The excluded fixtures must still produce the right answer through the
    # fallback -- the gate is an optimization, never a correctness condition.
    MR, NR, W = 16, 6, 8
    k = PlanarKernel(Val(MR), Val(NR), T, Val(W))
    acc = store_fp_acc(T, MR, NR, W, 7)
    mm, nn = MR, 4
    cold = store_fp_cold(T, mm * nn, 8)
    alpha, beta = T(2.5, -1), T(-1.75, 0.5)

    dense = copy(cold)
    store_tile!(
        DestinationTile(dense, 0, AffineAxis(0, 1, mm), AffineAxis(0, mm, nn)),
        acc, alpha, beta, k
    )

    subv = copy(cold)
    sv = view(subv, 1:(mm * nn))
    store_tile!(
        DestinationTile(sv, 0, AffineAxis(0, 1, mm), AffineAxis(0, mm, nn)),
        acc, alpha, beta, k
    )
    for idx in eachindex(dense)
        @test abs(dense[idx] - subv[idx]) <= 8 * eps(Float64) * max(abs(subv[idx]), 1.0)
    end

    negs = copy(cold)
    store_tile!(
        DestinationTile(
            negs, mm - 1 + mm * (nn - 1), AffineAxis(0, -1, mm), AffineAxis(0, -mm, nn)
        ),
        acc, alpha, beta, k
    )
    # The negative-stride tile addresses the same buffer in reverse; the value
    # at logical (i,j) lands at (mm-1-i) + mm*(nn-1-j), and `cold` was uniform
    # random, so compare against a freshly computed reference instead.
    for j in 0:(nn - 1), i in 0:(mm - 1)
        v, lane = i ÷ W, (i % W) + 1
        idx = v + (MR ÷ W) * j + 1
        want = ref_axpby(
            alpha, acc[idx][lane], acc[(MR ÷ W) * NR + idx][lane], beta,
            cold[(mm - 1 - i) + mm * (nn - 1 - j) + 1]
        )
        got = negs[(mm - 1 - i) + mm * (nn - 1 - j) + 1]
        @test abs(got - want) <= 8 * eps(Float64) * max(abs(want), 1.0)
    end
end

# ---------------------------------------------------------------------------
# 4. ISA portability (proposal Decision 5 / Section 7 item 5)
# ---------------------------------------------------------------------------

@testset "planar store fast path: ISA gate is a register-width question" begin
    @test QSS._complex_fastpath_isa_eligible(
        TargetProfile(:avx512, Sys.ARCH, "t", 64, 32, CacheLevel(), CacheLevel(), CacheLevel())
    )
    for (key, vb, nreg) in ((:avx2, 32, 16), (:neon, 16, 32), (:unknown, 0, 0))
        @test !QSS._complex_fastpath_isa_eligible(
            TargetProfile(key, Sys.ARCH, "t", vb, nreg, CacheLevel(), CacheLevel(), CacheLevel())
        )
    end
    @test !QSS._complex_fastpath_isa_eligible(unknown_target())
    # The live gate agrees with the live profile: this is what makes every
    # `== STORE_FASTPATH_ON` assertion above meaningful under forced_isa_runner.jl.
    @test STORE_FASTPATH_ON == (target_profile().vector_bytes == 64)
end

# ---------------------------------------------------------------------------
# 5. Zero allocation on an ELIGIBLE fixture
#
# test_planar_kernel.jl's Cliff B testset deliberately uses a SCATTERED
# destination, which the gate excludes -- so it says nothing about this path.
# ---------------------------------------------------------------------------

@testset "planar store fast path: allocation-free ($T)" for (T, menu) in STORE_FP_MENUS
    for (MR, NR, W) in menu
        k = PlanarKernel(Val(MR), Val(NR), T, Val(W))
        acc = store_fp_acc(T, MR, NR, W, MR + NR)
        storage = zeros(T, MR * NR)
        d = DestinationTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, NR))
        alpha, beta = T(2, -1), T(0.5, 0.25)
        run_store(d, acc, alpha, beta, k) = (store_tile!(d, acc, alpha, beta, k); nothing)
        run_store(d, acc, alpha, beta, k)                    # warm up
        bytes = @allocated run_store(d, acc, alpha, beta, k)
        # Same Julia 1.10 LTS caveat as test_planar_kernel.jl's Cliff B check:
        # the NTuple accumulator is not register-resident there.
        @test bytes == 0 skip = (VERSION < v"1.11")
    end
end

# ---------------------------------------------------------------------------
# 6. End to end, including conjugation.
#
# `store_tile!` itself takes no `transform` -- conjugation is a PACKING-time
# concept (`atransform`/`btransform`, src/driver.jl), which is why there is no
# `conj` argument to vary in sections 1-3 above. It is covered here instead, at
# the level where it exists, so that "conj still works with the store fast path
# live" is asserted rather than argued.
# ---------------------------------------------------------------------------

@testset "planar store fast path: end-to-end with conj ($T)" for
        T in (ComplexF64, ComplexF32)

    R = real(T)
    rng = MersenneTwister(31337)
    m, k, n = 37, 23, 19
    A = rand(rng, T, m, k)
    B = rand(rng, T, k, n)
    Cinit = rand(rng, T, m, n)

    for (ca, cb) in ((false, false), (true, false), (false, true), (true, true)),
            (alpha, beta) in ((one(T), zero(T)), (T(2.5, -1), T(-1.75, 0.5)))

        Aeff = ca ? conj.(A) : A
        Beff = cb ? conj.(B) : B
        want = alpha .* (Aeff * Beff) .+ beta .* Cinit

        C = copy(Cinit)
        plan = QuasiStrided.plan_contract(
            StridedView(C), StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3);
            conjA = ca, conjB = cb
        )
        QuasiStrided.execute!(plan, alpha, beta)
        @test maximum(abs, C .- want) <= 64 * eps(R) * k * maximum(abs, want)
    end
end
