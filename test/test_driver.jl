using StridedViews: StridedView, offset

# plan_contract/execute!/ContractPlan aren't exported (only contract! is);
# execute_tilewise! is never exported at all (it's an internal oracle).
# `test/runtests.jl` deliberately leaves these four out of its own
# name-restoring `using QuasiStrided: ...` block, because a `const` may not
# shadow an imported binding; the workspace API is reached as
# `QuasiStrided.<name>` for the same reason.
const plan_contract = QuasiStrided.plan_contract
const execute! = QuasiStrided.execute!
const ContractPlan = QuasiStrided.ContractPlan
const execute_tilewise! = QuasiStrided.execute_tilewise!

# TO names are always qualified (frozen import convention, docs/decisions.md);
# a bare `using TensorOperations` collides with QuasiStrided's `scalartype`.
import TensorOperations as TO

# Plan for the dense matmul C[m,n] = sum_k A[m,k]*B[k,n], the shape most
# testsets below use; every `plan_contract` keyword is forwarded verbatim, so
# an omitted one takes plan_contract's own default. (test_macro_driver.jl has
# its own copy: both files are included into the same scope, so the names must
# differ.)
function _mm_plan(Cmat, Amat, Bmat; kwargs...)
    return plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        kwargs...
    )
end

# Steady-state allocation of one execute!/execute_tilewise! call on a reused
# plan: warm up (compile) first, then measure.
function _steady_allocs!(run!, plan, Cmat)
    run!(plan, 1.0, 0.0)
    fill!(Cmat, 0.0)
    return @allocated run!(plan, 1.0, 0.0)
end

# Worked fixture: A[a,k,b] (3,5,2), B[k,n] (5,4), C[a,n,b] (3,4,2),
# C[a,n,b] = sum_k A[a,k,b]*B[k,n]; labels a=1,k=2,b=3,n=4.

function _worked_fixture()
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    B = reshape(collect(1.0:20.0), 5, 4)
    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, a in 1:3
        Cref[a, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end
    return A, B, Cref
end

const _INDA = (1, 2, 3)
const _INDB = (2, 4)
const _INDC = (1, 4, 3)

@testset "driver: worked fixture end to end via contract!" begin
    A, B, Cref = _worked_fixture()
    C = zeros(Float64, 3, 4, 2)
    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)

    contract!(Cv, 1.0, Av, _INDA, Bv, _INDB, 0.0, _INDC)
    @test C ≈ Cref
end

@testset "driver: worked fixture with permuted views" begin
    A, B, Cref = _worked_fixture()
    Ap = permutedims(A, (2, 3, 1)) # Ap[k,b,a] == A[a,k,b]
    Bp = permutedims(B, (2, 1))    # Bp[n,k] == B[k,n]
    C = zeros(Float64, 3, 4, 2)

    Avp = StridedView(Ap)
    Bvp = StridedView(Bp)
    Cv = StridedView(C)

    # k is axis1 of Avp, b is axis2, a is axis3 -> indA labels at those
    # positions are (k=2, b=3, a=1); n is axis1 of Bvp, k is axis2 -> (n=4, k=2).
    indAp = (2, 3, 1)
    indBp = (4, 2)

    contract!(Cv, 1.0, Avp, indAp, Bvp, indBp, 0.0, _INDC)
    @test C ≈ Cref
end

@testset "driver: worked fixture with a sliced destination and inputs" begin
    # Slice A and B down to a sub-range on their k axis, and slice C's a axis,
    # to exercise nonzero StridedView offsets end to end.
    A = reshape(collect(1.0:40.0), 4, 5, 2) # a in 1:4, k in 1:5, b in 1:2
    B = reshape(collect(1.0:20.0), 5, 4)    # k in 1:5, n in 1:4
    Cfull = zeros(Float64, 4, 4, 2)

    Asub = view(A, 2:4, 1:5, 1:2)  # a in 2:4 (3 rows), full k, full b
    Csub = view(Cfull, 2:4, :, :)

    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, (ai, a) in enumerate(2:4)
        Cref[ai, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end

    Av = StridedView(Asub)
    Bv = StridedView(B)
    Cv = StridedView(Csub)
    @test offset(Av) != 0

    contract!(Cv, 1.0, Av, _INDA, Bv, _INDB, 0.0, _INDC)
    @test Array(Csub) ≈ Cref
end

# =====================================================================
# Larger cases: M/N beyond one register tile, K beyond one panel.
# =====================================================================

@testset "driver: larger case, multiple output tiles and multiple K panels" begin
    Random.seed!(20260908)
    kernel = ScalarKernel(Val(4), Val(3), Float64)  # small shape: several tiles from modest M/N

    Ma, Ka, Na = 11, 13, 7 # deliberately not multiples of MR=4/NR=3/panel size
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)
    Cmat = zeros(Ma, Na)

    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, kc = 5)
    @test plan.blocking.kc == 5 # forces multiple K panels since Ka=13 > 5
    execute!(plan, 1.0, 0.0)

    @test Cmat ≈ Amat * Bmat
end

@testset "driver: nontrivial alpha/beta across multiple K panels, beta applied once" begin
    Random.seed!(4242)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)
    Cstart = randn(Ma, Na)
    Cmat = copy(Cstart)

    alpha, beta = 2.5, 0.75
    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, kc = 3)
    @test plan.blocking.kc == 3
    execute!(plan, alpha, beta)

    @test Cmat ≈ alpha .* (Amat * Bmat) .+ beta .* Cstart
end

# =====================================================================
# Whole-contraction short-circuits: K=0 and alpha=0. Neither may read A/B;
# verified with NaN/Inf-poisoned A/B.
# =====================================================================

@testset "driver: K=0 short-circuit applies beta once, never reads A/B" begin
    Ma, Na = 5, 4
    Apoison = fill(NaN, Ma, 0)  # K axis length 0
    Bpoison = fill(Inf, 0, Na)
    Cstart = randn(Ma, Na)
    Cmat = copy(Cstart)

    Av, Bv, Cv = StridedView(Apoison), StridedView(Bpoison), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    beta = 0.5
    contract!(Cv, 1.0, Av, indA, Bv, indB, beta, indC)
    @test Cmat ≈ beta .* Cstart
    @test all(isfinite, Cmat)
end

@testset "driver: alpha=0 short-circuit applies beta once, never reads A/B" begin
    Ma, Ka, Na = 5, 6, 4
    Apoison = fill(NaN, Ma, Ka)
    Bpoison = fill(Inf, Ka, Na)
    Cstart = randn(Ma, Na)
    Cmat = copy(Cstart)

    Av, Bv, Cv = StridedView(Apoison), StridedView(Bpoison), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    beta = 1.25
    contract!(Cv, 0.0, Av, indA, Bv, indB, beta, indC)
    @test Cmat ≈ beta .* Cstart
    @test all(isfinite, Cmat)
end

@testset "driver: empty output (M or N axis length 0) is a no-op" begin
    # N axis length 0.
    Amat = randn(3, 4)
    Bmat = randn(4, 0)
    Cmat = fill(NaN, 3, 0)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 2.0, (1, 3))
    @test size(Cmat) == (3, 0) # nothing to check elementwise; must not error

    # M axis length 0.
    Amat2 = randn(0, 4)
    Bmat2 = randn(4, 3)
    Cmat2 = fill(NaN, 0, 3)
    Av2, Bv2, Cv2 = StridedView(Amat2), StridedView(Bmat2), StridedView(Cmat2)
    contract!(Cv2, 1.0, Av2, (1, 2), Bv2, (2, 3), 2.0, (1, 3))
    @test size(Cmat2) == (0, 3)
end

# =====================================================================
# Label validation
# =====================================================================

@testset "driver: label validation errors" begin
    Amat = randn(3, 4)
    Bmat = randn(4, 5)
    Cmat = zeros(3, 5)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)

    # Mismatched shared-label length: B's k axis is 6, A's is 4.
    Bv_bad = StridedView(randn(6, 5))
    @test_throws DimensionMismatch contract!(Cv, 1.0, Av, (1, 2), Bv_bad, (2, 3), 0.0, (1, 3))

    # Diagonal: repeated label within indA, then within indC.
    Avsq = StridedView(randn(4, 4))
    @test_throws ArgumentError contract!(Cv, 1.0, Avsq, (1, 1), Bv, (2, 3), 0.0, (1, 3))
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 1, 3)[1:2])

    # Label in indC absent from both indA and indB.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 9))

    # Dangling label: only in indA, then only in indB. Distinct code paths
    # (the A loop vs. the B loop in _classify_labels).
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (5, 3), 0.0, (1, 3))
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 9), 0.0, (1, 3))

    # Label present in all three (batch-like), unsupported this milestone.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (2, 3))
end

# =====================================================================
# Planning vs execution as separate, separately measurable steps.
# =====================================================================

@testset "driver: planning and execution are separately timable; execution allocation" begin
    Random.seed!(99)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc = 4)
    planning_allocs = @allocated plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc = 4)
    @test planning_allocs > 0 # planning constructs AxisGroups/buffers

    # Execution alone, on the reused plan/workspace. This is measured
    # independently of planning; it is NOT asserted to be smaller (a
    # multi-tile case pays per-tile costs the one-shot planning does not).
    exec_allocs = _steady_allocs!(execute!, plan, Cmat)
    @test Cmat ≈ Amat * Bmat
    @test exec_allocs >= 0
    @test planning_allocs >= 0

    # A cold contract! plans AND executes every time, so it must allocate at
    # least as much as planning alone -- exactly what a reused plan avoids.
    fill!(Cmat, 0.0)
    cold_allocs = @allocated contract!(Cv, 1.0, Av, indA, Bv, indB, 0.0, indC)
    @test cold_allocs >= planning_allocs
end

@testset "driver: execution allocation through SIMDKernel is not worse than ScalarKernel" begin
    # The SIMD kernel was originally benchmarked only through a bare
    # execute_tile! call; assert that driving it through execute!'s own
    # dispatch loop adds no SIMD-specific allocation over the scalar path.
    Random.seed!(99)
    Ma, Ka, Na = 9, 10, 8
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)

    Cmat_s = zeros(Ma, Na)
    plan_s = _mm_plan(Cmat_s, Amat, Bmat; kernel = ScalarKernel(Val(4), Val(3), Float64), kc = 4)
    scalar_exec_allocs = _steady_allocs!(execute!, plan_s, Cmat_s)

    Cmat_v = zeros(Ma, Na)
    plan_v = _mm_plan(Cmat_v, Amat, Bmat; kernel = SIMDKernel(Val(4), Val(3), Float64), kc = 4)
    simd_exec_allocs = _steady_allocs!(execute!, plan_v, Cmat_v)

    @test Cmat_s ≈ Amat * Bmat
    @test Cmat_v ≈ Amat * Bmat
    @test simd_exec_allocs <= scalar_exec_allocs
end

# =====================================================================
# Blocking / default_blocking
# =====================================================================

@testset "Blocking: field validation" begin
    b = Blocking(4, 8, 16)
    @test b.mc == 4 && b.kc == 8 && b.nc == 16

    @test_throws ArgumentError Blocking(0, 8, 16)
    @test_throws ArgumentError Blocking(4, 0, 16)
    @test_throws ArgumentError Blocking(4, 8, 0)
    @test_throws ArgumentError Blocking(-1, 8, 16)
end

@testset "default_blocking: dispatches on kernel scalar type" begin
    bf64 = default_blocking(ScalarKernel(Val(8), Val(6), Float64))
    bf32 = default_blocking(ScalarKernel(Val(8), Val(6), Float32))
    @test bf64 isa Blocking
    @test bf32 isa Blocking
    @test bf64.mc >= 1 && bf64.kc >= 1 && bf64.nc >= 1
    @test bf32.mc >= 1 && bf32.kc >= 1 && bf32.nc >= 1
    # Same kernel shape, different scalar type, through SIMDKernel too.
    @test default_blocking(SIMDKernel(Val(8), Val(6), Float64)) == bf64
end

@testset "plan_contract: mc/kc/nc keywords are validated and rounded" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat, Bmat, Cmat = randn(Ma, Ka), randn(Ka, Na), zeros(Ma, Na)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    @test_throws ArgumentError plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, mc = 0)
    @test_throws ArgumentError plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc = 0)
    @test_throws ArgumentError plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, nc = -3)

    # mc=5 with MR=4 rounds up to 8, then clamps to roundup(Ma=9,4)=12 -> 8.
    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, mc = 5, kc = 100, nc = 100)
    @test plan.blocking.mc == 8
    @test plan.blocking.kc == 10  # clamped to Qk
    @test plan.blocking.nc == 9   # NR=3: roundup(8,3)=9, requested 100 clamped down to that
end

# =====================================================================
# Macro-blocking: multiple M/N blocks (loop 3 / loop 5 boundaries).
# =====================================================================

@testset "driver: forced multiple M blocks (mc = MR exactly)" begin
    Random.seed!(777)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 13, 6, 5 # Ma spans several MR=4 slivers across several mc=4 blocks
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)

    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, mc = 4)
    @test plan.blocking.mc == 4 # exactly one sliver per M block: forces several ic iterations
    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat
end

@testset "driver: forced multiple M blocks, non-multiple of MR (mc = 2*MR+1)" begin
    Random.seed!(778)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 23, 7, 5
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)

    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, mc = 2 * 4 + 1)
    @test plan.blocking.mc == 12 # rounds 9 up to a multiple of MR=4 (3 slivers/block, tail partial)
    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat
end

@testset "driver: forced multiple N blocks (nc = NR exactly, non-multiple)" begin
    Random.seed!(779)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 6, 17
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)

    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, nc = 2 * 3 + 1)
    @test plan.blocking.nc == 9 # rounds 7 up to a multiple of NR=3
    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat
end

@testset "driver: multiple M, N and K blocks simultaneously, nontrivial alpha/beta" begin
    Random.seed!(780)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 19, 23, 17 # deliberately not multiples of MR/NR or any tidy block size
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cstart = randn(Ma, Na)
    Cmat = copy(Cstart)

    alpha, beta = 1.75, -0.5
    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, mc = 8, kc = 6, nc = 7)
    @test plan.blocking.mc == 8
    @test plan.blocking.kc == 6
    @test plan.blocking.nc == 9 # roundup(7,3)

    execute!(plan, alpha, beta)
    @test Cmat ≈ alpha .* (Amat * Bmat) .+ beta .* Cstart
end

# =====================================================================
# execute! vs. execute_tilewise! on multi-block shapes. Each gets its own
# plan, so neither can be affected by the other's buffers.
# =====================================================================

@testset "execute! and execute_tilewise! agree on multi-block shapes" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)

    cases = (
        (Ma = 13, Ka = 11, Na = 10, mc = 4, kc = 5, nc = 3, alpha = 1.0, beta = 0.0),
        (Ma = 19, Ka = 23, Na = 17, mc = 8, kc = 6, nc = 7, alpha = 2.5, beta = -0.75),
    )

    for (idx, case) in enumerate(cases)
        Random.seed!(9000 + idx)
        Amat = randn(case.Ma, case.Ka)
        Bmat = randn(case.Ka, case.Na)
        Cstart = randn(case.Ma, case.Na)

        Cmat_macro = copy(Cstart)
        plan_macro = _mm_plan(Cmat_macro, Amat, Bmat; kernel = kernel, mc = case.mc, kc = case.kc, nc = case.nc)
        execute!(plan_macro, case.alpha, case.beta)

        Cmat_tw = copy(Cstart)
        plan_tw = _mm_plan(Cmat_tw, Amat, Bmat; kernel = kernel, mc = case.mc, kc = case.kc, nc = case.nc)
        execute_tilewise!(plan_tw, case.alpha, case.beta)

        @test Cmat_macro ≈ Cmat_tw
    end
end

# =====================================================================
# Allocation targets (docs/decisions.md, Phase A binding requirement): the
# macro-blocking execute! must close the QSTile-UnionAll boxing bug.
# SIMDKernel: 0 B on Julia >= 1.11 (older Julia doesn't keep the Vec-tuple
# accumulator register-resident; see test_simd_kernel.jl's own skip).
# ScalarKernel: bounded, not zero -- its zero_accumulator is a spec-accepted
# Matrix{T} allocation (176 B per execute_tile! call).
# =====================================================================

@testset "execute! allocation: SIMDKernel is zero, ScalarKernel is bounded" begin
    Random.seed!(321)
    Ma, Ka, Na = 19, 23, 17
    mc, kc, nc = 8, 6, 7
    MRk, NRk = 4, 3
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)

    tile_calls = cld(Ma, MRk) * cld(Na, NRk) * cld(Ka, kc)

    Cmat_s = zeros(Ma, Na)
    plan_s = _mm_plan(
        Cmat_s, Amat, Bmat;
        kernel = ScalarKernel(Val(MRk), Val(NRk), Float64), mc = mc, kc = kc, nc = nc
    )
    scalar_allocs = _steady_allocs!(execute!, plan_s, Cmat_s)
    @test Cmat_s ≈ Amat * Bmat
    @test scalar_allocs <= 176 * tile_calls + 1

    Cmat_v = zeros(Ma, Na)
    plan_v = _mm_plan(
        Cmat_v, Amat, Bmat;
        kernel = SIMDKernel(Val(MRk), Val(NRk), Float64), mc = mc, kc = kc, nc = nc
    )
    simd_allocs = _steady_allocs!(execute!, plan_v, Cmat_v)
    @test Cmat_v ≈ Amat * Bmat
    @test simd_allocs == 0 skip = (VERSION < v"1.11")
end

# =====================================================================
# ContractWorkspace, the `workspace`/`allocator`/`oracle` keywords, and the
# SIMDKernel default (docs/decisions.md, "Amendment 1"/"Amendment 2").
# =====================================================================

_ws_lengths(ws) = map(f -> length(getfield(ws, f)), fieldnames(typeof(ws)))

@testset "plan_contract: SIMDKernel is the engine-wide default kernel" begin
    for T in (Float64, Float32)
        Random.seed!(5150)
        Amat, Bmat = randn(T, 9, 10), randn(T, 10, 8)
        Cmat = zeros(T, 9, 8)
        plan = _mm_plan(Cmat, Amat, Bmat)

        # Amendment 2: a SIMDKernel, not a ScalarKernel. The shape itself is
        # hardware-derived (docs/decisions.md, Phase G) and demoted when M
        # cannot fill a register tile, so pin the *resolution* rather than a
        # literal shape -- `{8, 6, T}` held here only because Qm = 9 happens
        # to trigger the demotion on x86, and broke on aarch64.
        @test plan.kernel isa QuasiStrided.SIMDKernel
        @test QuasiStrided.scalartype(plan.kernel) === T
        @test plan.kernel === QuasiStrided._default_kernel(T, size(Amat, 1), size(Bmat, 2))
        execute!(plan, one(T), zero(T))
        @test Cmat ≈ Amat * Bmat

        # ... and `contract!`, which must never disagree with plan_contract
        # about what "default" means.
        Cmat2 = zeros(T, 9, 8)
        contract!(
            StridedView(Cmat2), one(T), StridedView(Amat), (1, 2),
            StridedView(Bmat), (2, 3), zero(T), (1, 3)
        )
        @test Cmat2 == Cmat
    end
end

@testset "ContractWorkspace: reuse across shapes is bitwise identical to fresh plans" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    mc, kc, nc = 8, 6, 7
    alpha, beta = 1.75, -0.5

    # Deliberately not monotone in size: the workspace is sized by the first
    # (mid) shape, grown by the second (large) one, then reused oversized by
    # every smaller one after it.
    shapes = ((13, 11, 10), (23, 19, 17), (4, 3, 2), (9, 10, 8), (1, 1, 1), (16, 5, 6))

    ws = nothing
    lengths_before = nothing
    for (idx, (Ma, Ka, Na)) in enumerate(shapes)
        Random.seed!(31_000 + idx)
        Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
        Cstart = randn(Ma, Na)

        Cfresh = copy(Cstart)
        execute!(_mm_plan(Cfresh, Amat, Bmat; kernel = kernel, mc = mc, kc = kc, nc = nc), alpha, beta)

        Creuse = copy(Cstart)
        plan = _mm_plan(
            Creuse, Amat, Bmat;
            kernel = kernel, mc = mc, kc = kc, nc = nc, workspace = ws
        )
        execute!(plan, alpha, beta)

        # Bitwise, not approximate: reuse must not perturb the arithmetic.
        @test Creuse == Cfresh

        if ws !== nothing
            @test plan.workspace === ws               # reserve!d in place, not rebuilt
            @test all(_ws_lengths(ws) .>= lengths_before)  # grow-only, never shrunk
        end
        ws = plan.workspace
        lengths_before = _ws_lengths(ws)
    end

    # Wrong-eltype workspaces are rejected rather than silently rebuilt.
    Amat32, Bmat32, Cmat32 = randn(Float32, 4, 4), randn(Float32, 4, 4), zeros(Float32, 4, 4)
    @test_throws ArgumentError _mm_plan(
        Cmat32, Amat32, Bmat32;
        kernel = ScalarKernel(Val(4), Val(3), Float32), workspace = ws
    )
end

@testset "ContractWorkspace: an oversized reused buffer is not read beyond its live region" begin
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    mc, kc, nc = 8, 6, 7
    alpha, beta = 2.5, -0.75

    # Size the workspace on a large contraction ...
    Random.seed!(606)
    Abig, Bbig = randn(23, 19), randn(19, 17)
    Cbig = zeros(23, 17)
    big = _mm_plan(Cbig, Abig, Bbig; kernel = kernel, mc = mc, kc = kc, nc = nc)
    execute!(big, 1.0, 0.0)
    ws = big.workspace

    # ... then run a much smaller one on it.
    Ma, Ka, Na = 5, 4, 3
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cstart = randn(Ma, Na)

    Cref = copy(Cstart)
    execute!(_mm_plan(Cref, Amat, Bmat; kernel = kernel, mc = mc, kc = kc, nc = nc), alpha, beta)

    # Poison every slot of every reused buffer. A packed slot read without
    # having been written this call turns the output into NaN; an offset or
    # descriptor read outside the live region addresses far outside the
    # operand and is rejected by checked_tile_storage_bounds.
    fill!(ws.packed_a, NaN)
    fill!(ws.packed_b, NaN)
    for buf in (ws.m_buf_A, ws.m_buf_C, ws.n_buf_B, ws.n_buf_C, ws.k_buf_A, ws.k_buf_B)
        fill!(buf, typemin(Int) ÷ 4)
    end
    poison = BlockDescriptor(typemin(Int) ÷ 4, 0, 1, true)
    for desc in (ws.m_desc_A, ws.m_desc_C, ws.n_desc_B, ws.n_desc_C)
        fill!(desc, poison)
    end

    lengths_before = _ws_lengths(ws)
    Cpoisoned = copy(Cstart)
    plan = _mm_plan(
        Cpoisoned, Amat, Bmat;
        kernel = kernel, mc = mc, kc = kc, nc = nc, workspace = ws
    )
    # Nothing was regrown, so this really did run on the oversized buffers.
    @test _ws_lengths(ws) == lengths_before
    @test length(ws.packed_a) > cld(Ma, 4) * 4 * min(kc, Ka)
    @test length(ws.packed_b) > cld(Na, 3) * 3 * min(kc, Ka)

    execute!(plan, alpha, beta)
    @test all(isfinite, Cpoisoned)
    @test Cpoisoned == Cref
end

@testset "plan_contract: oracle = false skips execute_tilewise!'s buffers" begin
    Random.seed!(909)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)

    Cmat = zeros(Ma, Na)
    plan = _mm_plan(Cmat, Amat, Bmat; kernel = kernel, kc = 4, oracle = false)
    ws = plan.workspace

    @test isempty(ws.tw_packed_a)
    @test isempty(ws.tw_packed_b)
    @test isempty(ws.tw_k_buf_A)
    @test isempty(ws.tw_k_buf_B)
    # The MR/NR-sized ones are NOT oracle-only: _scale_all_of_C!, the beta-only
    # pass of both drivers, uses them.
    @test length(ws.tw_m_buf_A) == 4
    @test length(ws.tw_n_buf_C) == 3

    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat

    # The beta-only short-circuit still works without the oracle buffers.
    Cstart = randn(Ma, Na)
    Cbeta = copy(Cstart)
    execute!(_mm_plan(Cbeta, Amat, Bmat; kernel = kernel, oracle = false), 0.0, 0.5)
    @test Cbeta ≈ 0.5 .* Cstart

    # ... but the oracle itself refuses to run rather than reading empty buffers.
    @test_throws ArgumentError execute_tilewise!(plan, 1.0, 0.0)

    # Reusing the same workspace with oracle = true grows them back.
    Ctw = zeros(Ma, Na)
    plan_tw = _mm_plan(Ctw, Amat, Bmat; kernel = kernel, kc = 4, workspace = ws, oracle = true)
    @test plan_tw.workspace === ws
    @test !isempty(ws.tw_packed_a)
    execute_tilewise!(plan_tw, 1.0, 0.0)
    @test Ctw ≈ Amat * Bmat
end

@testset "plan_contract: explicit allocators size the packed panels exactly once" begin
    Random.seed!(1717)
    kernel = SIMDKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 19, 23, 17
    mc, kc, nc = 8, 6, 7
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)

    Cdefault = zeros(Ma, Na)
    default_plan = _mm_plan(Cdefault, Amat, Bmat; kernel = kernel, mc = mc, kc = kc, nc = nc)
    execute!(default_plan, 1.5, 0.0)
    @test default_plan.workspace.packed_a isa Vector{Float64}

    for allocator in (TO.ManualAllocator(), TO.BufferAllocator())
        checkpoint = TO.allocator_checkpoint!(allocator)

        Cmat = zeros(Ma, Na)
        plan = _mm_plan(
            Cmat, Amat, Bmat;
            kernel = kernel, mc = mc, kc = kc, nc = nc,
            allocator = allocator, oracle = false
        )
        ws = plan.workspace

        # Concretely typed instance, exact sizing, no oracle buffers.
        @test isconcretetype(typeof(ws))
        @test all(isconcretetype, fieldtypes(typeof(ws)))
        @test length(ws.packed_a) == cld(plan.blocking.mc, 4) * 4 * plan.blocking.kc
        @test length(ws.packed_b) == cld(plan.blocking.nc, 3) * 3 * plan.blocking.kc
        @test isempty(ws.tw_packed_a)
        # Offset buffers are Val(false) requests: a plain Vector{Int} from
        # every allocator, because fill_offsets!/describe_block are frozen on
        # that concrete type.
        @test ws.m_buf_A isa Vector{Int}
        @test ws.k_buf_B isa Vector{Int}

        execute!(plan, 1.5, 0.0)
        @test Cmat == Cdefault  # same blocking, so bitwise identical

        QuasiStrided.release!(ws, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end

    # A PtrArray-backed workspace is not reserve!-able at all: the grow-upward
    # discipline and allocator-owned temporaries are mutually exclusive
    # (docs/decisions.md, "Verified allocator behavior", fact 3).
    manual = TO.ManualAllocator()
    Cmanual = zeros(Ma, Na)
    manual_plan = _mm_plan(
        Cmanual, Amat, Bmat;
        kernel = kernel, mc = mc, kc = kc, nc = nc, allocator = manual, oracle = false
    )
    @test_throws MethodError QuasiStrided.reserve!(
        manual_plan.workspace, kernel, manual_plan.blocking, false
    )
    QuasiStrided.release!(manual_plan.workspace, manual)

    # An explicit allocator and a reusable workspace are contradictory.
    @test_throws ArgumentError _mm_plan(
        zeros(Ma, Na), Amat, Bmat;
        kernel = kernel, allocator = TO.ManualAllocator(),
        workspace = default_plan.workspace
    )
end

# =====================================================================
# Type-stability regression: `ContractWorkspace`'s `VT` parameter must not
# reintroduce the boxing bug of docs/decisions.md's "Phase A findings".
# =====================================================================

_ws_union_members(t) = t isa Union ?
    (_ws_union_members(t.a)..., _ws_union_members(t.b)...) : (t,)

# Every type inference assigned in `f(argtypes...)`'s unoptimized typed IR that
# is a QSTile or ContractWorkspace and is *not* concrete, directly or as a
# union member. `Union{}` is a `throw` branch's result type, not an instability.
function _ws_nonconcrete_types(f, argtypes)
    bad = Any[]
    for (ci, rt) in Base.code_typed(f, argtypes; optimize = false)
        types = Any[rt]
        ci.slottypes isa Vector && append!(types, ci.slottypes)
        ci.ssavaluetypes isa Vector && append!(types, ci.ssavaluetypes)
        for t in types
            t isa Type || continue
            for m in _ws_union_members(t)
                (m isa Type && m !== Union{}) || continue
                if (m <: QuasiStrided.QSTile || m <: QuasiStrided.ContractWorkspace) &&
                        !isconcretetype(m)
                    push!(bad, t)
                    break
                end
            end
        end
    end
    return unique(bad)
end

@testset "plan_contract/execute!: no union-typed or partially-applied tile/workspace types" begin
    Random.seed!(2468)
    Ma, Ka, Na = 19, 23, 17
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    plan = _mm_plan(Cmat, Amat, Bmat; mc = 8, kc = 6, nc = 7)

    # Every *instance* is concretely typed, and no field is a Union or a bare
    # AbstractVector (the frozen prohibition; VT is a where-bound parameter
    # resolved at construction, like ScatterAxis{V}).
    @test isconcretetype(typeof(plan))
    @test isconcretetype(typeof(plan.workspace))
    @test all(isconcretetype, fieldtypes(typeof(plan.workspace)))
    @test !any(t -> t isa Union, fieldtypes(typeof(plan.workspace)))
    @test fieldtype(typeof(plan), :workspace) === typeof(plan.workspace)
    @test typeof(plan.workspace) === QuasiStrided.ContractWorkspace{Float64, Vector{Float64}}

    plan_argtypes = (
        typeof(Cv), typeof(Av), NTuple{2, Int}, typeof(Bv), NTuple{2, Int}, NTuple{2, Int},
    )
    @test isempty(_ws_nonconcrete_types(plan_contract, plan_argtypes))
    @test isempty(_ws_nonconcrete_types(execute!, (typeof(plan), Float64, Float64)))
    @test isempty(_ws_nonconcrete_types(execute_tilewise!, (typeof(plan), Float64, Float64)))
    @test isconcretetype(only(Base.return_types(execute!, (typeof(plan), Float64, Float64))))
end

# =====================================================================
# Zero steady-state allocation on the default (DefaultAllocator) path, with
# the default kernel. SIMDKernel's accumulator is not kept register-resident
# by Julia 1.10's compiler (docs/decisions.md, Amendment 2's caveat), so this
# is skipped there exactly as test/test_simd_kernel.jl skips its own -- never
# weakened or deleted.
# =====================================================================

@testset "execute! on the default allocator path is allocation-free" begin
    Random.seed!(1123)
    Ma, Ka, Na = 19, 23, 17
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)

    # Everything at its default: default kernel, default blocking, fresh
    # DefaultAllocator-backed workspace.
    Cmat = zeros(Ma, Na)
    plan = _mm_plan(Cmat, Amat, Bmat)
    allocs = _steady_allocs!(execute!, plan, Cmat)
    @test Cmat ≈ Amat * Bmat
    @test allocs == 0 skip = (VERSION < v"1.11")

    # ... and through a reused workspace, which is the shape the backend path
    # takes on every call.
    Cmat2 = zeros(Ma, Na)
    plan2 = _mm_plan(Cmat2, Amat, Bmat; workspace = plan.workspace, oracle = false)
    allocs2 = _steady_allocs!(execute!, plan2, Cmat2)
    @test Cmat2 ≈ Amat * Bmat
    @test allocs2 == 0 skip = (VERSION < v"1.11")
end

# =====================================================================
# Conjugation plumbing (docs/decisions.md, "Conjugation: semantics, and where
# each piece is absorbed"). No complex kernel exists yet, so what is testable
# here -- and what matters most for this worker -- is the *real-path-unchanged*
# half of that section, plus the predicates themselves.
# =====================================================================

@testset "_op_conjugates is a total table with a throwing fallback" begin
    # NOT TensorOperations' TBLIS extension's `A.op === conj` test:
    # `StridedView(p, sz, st, off, adjoint)` is directly constructible, and
    # `=== conj` would silently treat it as unconjugated.
    @test QuasiStrided._op_conjugates(identity) === false
    @test QuasiStrided._op_conjugates(conj) === true
    @test QuasiStrided._op_conjugates(transpose) === false   # elementwise identity
    @test QuasiStrided._op_conjugates(adjoint) === true
    @test_throws ArgumentError QuasiStrided._op_conjugates(sin)

    # A real element type is conjugated by nothing, whatever the flag or the
    # op: `StridedViews` collapses `conj` on a real view, so this is a
    # structural guarantee, not a convention.
    for op in (identity, conj, transpose, adjoint), flag in (false, true)
        v = StridedView(randn(16), (4, 4), (1, 4), 0, op)
        @test QuasiStrided._qs_isconj(v, flag) === false
    end
    # A complex element type: the flag and the op compose with XOR, so a
    # conj-wrapped view with conjA = true is unconjugated.
    for (op, oc) in ((identity, false), (conj, true), (transpose, false), (adjoint, true)),
            flag in (false, true)
        v = StridedView(randn(ComplexF64, 16), (4, 4), (1, 4), 0, op)
        @test QuasiStrided._qs_isconj(v, flag) === (flag ⊻ oc)
    end
end

@testset "conjA/conjB add no specialization on the real path" begin
    Random.seed!(97531)
    Ma, Ka, Na = 12, 9, 7
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    base = _mm_plan(Cmat, Amat, Bmat)

    for ca in (false, true), cb in (false, true)
        plan = _mm_plan(Cmat, Amat, Bmat; conjA = ca, conjB = cb)
        # The directly testable real-path guarantee: `_qs_isconj` is false
        # unconditionally for a real T, so TA === TB === typeof(identity) and
        # execute! gains no new specialization even with conjA = true.
        @test typeof(plan.atransform) === typeof(identity)
        @test typeof(plan.btransform) === typeof(identity)
        @test typeof(plan) === typeof(base)
        fill!(Cmat, 0.0)
        execute!(plan, 1.0, 0.0)
        @test Cmat ≈ Amat * Bmat
        fill!(Cmat, 0.0)
        execute_tilewise!(plan, 1.0, 0.0)
        @test Cmat ≈ Amat * Bmat
    end

    # ... and the same through views that carry a non-trivial `op`, which a
    # real eltype collapses to identity before the engine ever sees it.
    Av = conj(StridedView(Amat))
    @test Av.op === identity
    plan = plan_contract(
        StridedView(Cmat), Av, (1, 2), StridedView(Bmat), (2, 3), (1, 3); conjA = true
    )
    @test typeof(plan) === typeof(base)
    fill!(Cmat, 0.0)
    execute!(plan, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat
end

@testset "plan_contract rejects a conjugated output, but not a real adjoint" begin
    Random.seed!(2469)
    Amat, Bmat = randn(6, 5), randn(5, 4)

    # This is the test that pins "the real path is unchanged" exactly where it
    # could break: `adjoint(::Matrix{Float64})` has op === identity, because
    # StridedViews collapses adjoint on a real eltype, so it is still ACCEPTED.
    Cadj = adjoint(zeros(4, 6))
    Cv = StridedView(Cadj)
    @test Cv.op === identity
    plan = plan_contract(Cv, StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3))
    execute!(plan, 1.0, 0.0)
    @test Cadj ≈ Amat * Bmat

    # A conjugated COMPLEX output is rejected at the engine boundary, not in
    # the adapter, so `plan_contract`/`contract!` are protected too. Checked by
    # message: a complex plan would otherwise also throw ArgumentError from the
    # not-yet-wired kernel seam, which is a different failure.
    Ac, Bc = randn(ComplexF64, 6, 5), randn(ComplexF64, 5, 4)
    Cc = zeros(ComplexF64, 24)
    for op in (conj, adjoint)
        Ccv = StridedView(Cc, (6, 4), (1, 6), 0, op)
        err = try
            plan_contract(Ccv, StridedView(Ac), (1, 2), StridedView(Bc), (2, 3), (1, 3))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("conjugated", err.msg)
    end
    # An unrecognized op is hard-rejected rather than silently mishandled --
    # and, stronger than the freeze assumed, StridedViews makes one
    # unconstructible in the first place: its own `F` parameter is bounded by
    # exactly the four functions `_op_conjugates` tabulates. That bound is what
    # this asserts, so the throwing fallback stays correct-by-construction
    # rather than merely untested; if StridedViews ever widens it, this fails
    # here rather than silently somewhere in packing.
    @test_throws TypeError StridedView(Cc, (6, 4), (1, 6), 0, sin)
    Fbound = fieldtype(typeof(StridedView(Cc, (6, 4), (1, 6), 0, conj)), :op)
    @test Fbound === typeof(conj)
    optypes = Base.unwrap_unionall(StridedView).parameters[4].ub
    @test Set(Base.uniontypes(optypes)) ==
        Set((typeof(identity), typeof(conj), typeof(transpose), typeof(adjoint)))
end

@testset "ContractWorkspace: the relaxed VT bound keeps every old spelling" begin
    k64 = SIMDKernel(Val(8), Val(6), Float64)
    k32 = SIMDKernel(Val(8), Val(6), Float32)
    b = Blocking(16, 8, 12)

    # Every existing spelling stays valid, unedited -- the whole point of
    # relaxing the bound rather than adding a parameter.
    ws = QuasiStrided.ContractWorkspace(Float64, k64, b; oracle = true)
    @test ws isa QuasiStrided.ContractWorkspace{Float64, Vector{Float64}}
    @test eltype(ws.packed_a) === Float64

    # `T` is the STORAGE element type and the packed panels hold `real(T)`: a
    # complex storage type over Float64-packing is the new instance the relaxed
    # bound admits, at the SAME arity.
    wsc = QuasiStrided.ContractWorkspace(ComplexF64, k64, b; oracle = true)
    @test wsc isa QuasiStrided.ContractWorkspace{ComplexF64, Vector{Float64}}
    @test eltype(wsc.packed_a) === Float64 === eltype(wsc.tw_packed_b)
    @test all(isconcretetype, fieldtypes(typeof(wsc)))
    @test !any(t -> t isa Union, fieldtypes(typeof(wsc)))

    # ... and a mismatched (T, VT) pair cannot be constructed at all: the inner
    # constructor enforces eltype(VT) === real(T).
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(ComplexF64, k32, b)
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(Float64, k32, b)
    @test_throws ArgumentError QuasiStrided.ContractWorkspace(ComplexF32, k64, b)

    # Reuse still refuses a workspace of the wrong storage type, by dispatch.
    Amat, Bmat, Cmat = randn(6, 5), randn(5, 4), zeros(6, 4)
    @test_throws ArgumentError plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        workspace = wsc
    )
end

@testset "a non-identity pack transform crosses _pack_sliver! without allocating" begin
    # The real path never builds this plan -- `_qs_isconj` is false for a real
    # eltype, which is the point -- but the Phase 2b finding-5 failure mode (a
    # transform reaching `_pack_sliver!` as a Union, ~80 B/call of dynamic
    # dispatch) is a property of the `TF`/`TA`/`TB` type parameters, not of
    # complex arithmetic. Building the plan directly exercises a genuine second
    # specialization now, rather than discovering it in Phase C. `conj` is the
    # elementwise identity on a real, so the result must be unchanged.
    Random.seed!(8642)
    Ma, Ka, Na = 19, 23, 17
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    p = _mm_plan(Cmat, Amat, Bmat)
    pc = ContractPlan(
        p.kernel, p.mgroup, p.ngroup, p.kgroup, p.blocking,
        p.Astorage, p.Abase, p.Bstorage, p.Bbase, p.Cstorage, p.Cbase,
        conj, conj, p.workspace,
    )
    @test isconcretetype(typeof(pc))
    @test typeof(pc.atransform) === typeof(conj) === typeof(pc.btransform)
    @test isempty(_ws_nonconcrete_types(execute!, (typeof(pc), Float64, Float64)))

    allocs = _steady_allocs!(execute!, pc, Cmat)
    @test Cmat ≈ Amat * Bmat
    @test allocs == 0 skip = (VERSION < v"1.11")

    fill!(Cmat, 0.0)
    allocs_tw = _steady_allocs!(execute_tilewise!, pc, Cmat)
    @test Cmat ≈ Amat * Bmat
    @test allocs_tw == 0 skip = (VERSION < v"1.11")
end

# =====================================================================
# Store fast path (docs/decisions.md, "Store fast-path investigation:
# Phase A"). `store_tile!`'s vectorized path is only reachable from the
# real driver now that its guard admits any `DenseVector{T}`: the driver's
# destination storage is `parent(C)`, i.e. `Memory{T}` on Julia >= 1.11.
# That made the path's row tail a live allocation risk (a dynamically
# indexed accumulator heap-allocates above NV = 16), so the driver-level
# assertion below deliberately uses M and N extents that are NOT multiples
# of the kernel's MR/NR -- every other allocation testset in this file
# happens to be tail-free in M or exercises the scattered fallback instead.
# =====================================================================

@testset "execute! allocation: the vectorized store path with tail rows is zero" begin
    Random.seed!(20260915)
    MRk, NRk, Wk = 8, 6, 4
    kernel = SIMDKernel(Val(MRk), Val(NRk), Float64, Val(Wk))
    Ma, Ka, Na = 3 * MRk - 3, 11, 2 * NRk - 1  # tail block in both M and N
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    plan = _mm_plan(
        Cmat, Amat, Bmat;
        kernel = kernel, mc = 2 * MRk, kc = 5, nc = NRk + 2
    )

    # The destination really is the widened guard's case, and its micro-tile
    # rows really are unit-stride (C is column-major and M is its first index),
    # so `execute!` below takes the vectorized store, tail rows included.
    Cstorage = parent(StridedView(Cmat))
    @test Cstorage isa DenseVector{Float64}
    @test QuasiStrided._vector_store_eligible(
        DestinationTile(Cstorage, 0, AffineAxis(0, 1, MRk - 3), AffineAxis(0, Ma, NRk)), Float64
    )

    allocs = _steady_allocs!(execute!, plan, Cmat)
    @test Cmat ≈ Amat * Bmat
    @test allocs == 0 skip = (VERSION < v"1.11")
end

# =====================================================================
# Label order within M/N and the M/N orientation swap (docs/decisions.md,
# "Label-order milestone"). `_classify_labels` lists free labels in A's/B's
# own axis order; `plan_contract` then sorts each list by |C-stride| and may
# swap the operand roles. These are the FIRST tests that pin the composite
# order at all -- there was none before this milestone.
# =====================================================================

const _lo_order = QuasiStrided._order_free_labels
const _lo_run = QuasiStrided._leading_unit_run
const _lo_swap = QuasiStrided._prefer_swap

# A StridedView with exactly these strides and no data behind it worth
# reading: the helpers under test only look at `strides`/`size`.
_lo_view(sz::NTuple{N, Int}, st::NTuple{N, Int}) where {N} =
    StridedView(zeros(Float64, 4096), sz, st, 2048, identity)

# The four TCCG `ccsd_t_*` shapes (benchmark/bench_ccsd_t_store.jl), with a
# six-index C stored physically in (a,b,c,i,j,k) order. Labels are the
# adapter's own (`_qs_labels`), so the fixture exercises exactly the label
# assignment the TensorOperations path hands to `plan_contract`.
const _LO_IC = (:a, :b, :c, :i, :j, :k)
const _LO_CASES = (
    ("ccsd_t_1", (:i, :j, :m, :a), (:m, :k, :b, :c)),
    ("ccsd_t_2", (:i, :j, :m, :b), (:m, :k, :a, :c)),
    ("ccsd_t_3", (:i, :j, :m, :c), (:m, :k, :a, :b)),
    ("ccsd_t_4", (:i, :k, :m, :b), (:m, :j, :a, :c)),
)
function _lo_labels(IA, IB)
    pA, pB, pAB = TO.contract_indices(IA, IB, _LO_IC)
    return QuasiStrided._qs_labels(pA, pB, pAB), (pA, pB, pAB)
end

# Engine-free reference: C[indC] = alpha * sum_K conj?(A[indA]) conj?(B[indB]) + beta * C,
# by one loop over the Cartesian product of every label's range. `getindex`
# on a StridedView applies its `op`; the flag is then applied on top, which is
# the XOR rule `plan_contract` implements.
function _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA = false, conjB = false, alpha = 1, beta = 0)
    labels = unique((indA..., indB...))
    ext = Dict{Int, Int}()
    for (l, s) in zip(indA, size(Av))
        ext[l] = s
    end
    for (l, s) in zip(indB, size(Bv))
        ext[l] = s
    end
    T = eltype(Cstart)
    acc = zeros(T, size(Cstart))
    pA = map(l -> Int(findfirst(==(l), labels)), indA)
    pB = map(l -> Int(findfirst(==(l), labels)), indB)
    pC = map(l -> Int(findfirst(==(l), labels)), indC)
    dims = Tuple(ext[l] for l in labels)
    _lo_reference_loop!(acc, Av, pA, Bv, pB, pC, dims, conjA, conjB)
    return alpha .* acc .+ beta .* Cstart
end

# Function barrier: the position tuples arrive with concrete lengths, so the
# loop body is type-stable (the `map`/splat version took seconds per call).
function _lo_reference_loop!(
        acc, Av, pA::NTuple{NA, Int}, Bv, pB::NTuple{NB, Int}, pC::NTuple{NC, Int},
        dims::NTuple{NL, Int}, conjA::Bool, conjB::Bool
    ) where {NA, NB, NC, NL}
    for I in CartesianIndices(dims)
        a = Av[ntuple(d -> I[pA[d]], Val(NA))...]
        b = Bv[ntuple(d -> I[pB[d]], Val(NB))...]
        conjA && (a = conj(a))
        conjB && (b = conj(b))
        acc[ntuple(d -> I[pC[d]], Val(NC))...] += a * b
    end
    return acc
end

@testset "label order: _order_free_labels sorts by |C-stride|, stably" begin
    # C strides (12, 1, 60, 3) at indC positions carrying labels 10,20,30,40.
    Cv = _lo_view((5, 3, 2, 4), (12, 1, 60, 3))
    indC = (10, 20, 30, 40)
    labels = [10, 20, 30, 40]
    @test _lo_order(labels, indC, Cv) == [20, 40, 10, 30]
    @test labels == [10, 20, 30, 40]                 # input untouched
    @test _lo_order([40, 10], indC, Cv) == [40, 10]  # a subset: only its own members
    @test _lo_order([30, 20], indC, Cv) == [20, 30]

    # Ties keep input order, whichever way the input is given.
    Ct = _lo_view((2, 3, 4), (1, 1, 1))
    @test _lo_order([7, 8, 9], (7, 8, 9), Ct) == [7, 8, 9]
    @test _lo_order([9, 7, 8], (7, 8, 9), Ct) == [9, 7, 8]

    # Negative strides sort by magnitude.
    Cn = _lo_view((3, 4), (-1, 4))
    @test _lo_order([20, 10], (10, 20), Cn) == [10, 20]
    Cn2 = _lo_view((3, 4), (4, -1))
    @test _lo_order([10, 20], (10, 20), Cn2) == [20, 10]

    # Degenerate lengths.
    @test _lo_order(Int[], indC, Cv) == Int[]
    @test _lo_order([30], indC, Cv) == [30]
end

@testset "label order: _leading_unit_run / _prefer_swap" begin
    d = 4
    C6 = StridedView(zeros(Float64, d, d, d, d, d, d))  # strides 1, d, d^2, ...
    indC = (1, 2, 3, 4, 5, 6)                            # a,b,c,i,j,k
    a, b, c, i, j, k = indC

    @test _lo_run([a, i, j], indC, C6) == d          # ccsd_t_1's sorted M: run stops at i
    @test _lo_run([a, b, k], indC, C6) == d^2        # ccsd_t_3's sorted N: b is C-adjacent to a
    @test _lo_run([a, b, c, i, j, k], indC, C6) == d^6
    @test _lo_run([b, i, j], indC, C6) == 1          # no unit-stride head
    @test _lo_run([a, c, b], indC, C6) == d          # sorted order is the caller's job
    @test _lo_run(Int[], indC, C6) == 1

    # A descending contiguous axis is NOT a unit-stride run (`_unit_stride_rows`
    # is `stride == 1`).
    @test _lo_run([10, 20], (10, 20), _lo_view((3, 4), (-1, 3))) == 1
    # Singleton axes are skipped whatever their stride; an empty axis ends it.
    @test _lo_run([10, 20], (10, 20), _lo_view((1, 6), (5, 1))) == 6
    @test _lo_run([10, 20], (10, 20), _lo_view((6, 1), (1, 17))) == 6
    @test _lo_run([10, 20], (10, 20), _lo_view((0, 6), (1, 1))) == 0

    # The swap rule on the ccsd_t shapes at dim 4, with `mr` played by hand.
    # ccsd_t_2: sorted M = [b,i,j] (run 1), sorted N = [a,c,k] (run 4).
    @test _lo_swap([b, i, j], [a, c, k], indC, C6, 4)        # 4-wide kernel: swap
    @test !_lo_swap([b, i, j], [a, c, k], indC, C6, 8)       # 8-wide: 4 < 8, do not swap
    # ccsd_t_3: sorted N = [a,b,k] (run 16): the adjacent label extends it.
    @test _lo_swap([c, i, j], [a, b, k], indC, C6, 8)
    @test _lo_swap([c, i, j], [a, b, k], indC, C6, 16)
    @test !_lo_swap([c, i, j], [a, b, k], indC, C6, 32)
    # ccsd_t_1: M already has the run; never swap, whatever mr says.
    @test !_lo_swap([a, i, j], [b, c, k], indC, C6, 4)
    @test !_lo_swap([a, i, j], [b, c, k], indC, C6, 8)
    # Two-mr form: each orientation is judged against the kernel it would run.
    @test _lo_swap([b, i, j], [a, c, k], indC, C6, 8, 4)
    @test !_lo_swap([b, i, j], [a, c, k], indC, C6, 8, 8)
    @test !_lo_swap([a, i, j], [b, c, k], indC, C6, 4, 4)
    # Nothing to swap onto.
    @test !_lo_swap([b, i, j], Int[], indC, C6, 1)
end

@testset "label order: pinning test on the ccsd_t shapes (composite order and swap)" begin
    d = 5
    for (name, IA, IB) in _LO_CASES
        (indA, indB, indC), _ = _lo_labels(IA, IB)
        A = randn(d, d, d, d)
        B = randn(d, d, d, d)
        C = zeros(d, d, d, d, d, d)
        Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
        cst = Base.strides(Cv)
        cstride(l) = cst[findfirst(==(l), indC)]
        mlab, nlab, klab = QuasiStrided._classify_labels(indA, indB, indC)
        @test length(klab) == 1
        kA = Base.strides(Av)[findfirst(==(klab[1]), indA)]
        kB = Base.strides(Bv)[findfirst(==(klab[1]), indB)]

        # Expected: each composite ascending in |C-stride|; the same set of
        # labels as `_classify_labels` produced.
        msorted = sort(mlab; by = cstride)
        nsorted = sort(nlab; by = cstride)
        @test _lo_order(mlab, indC, Cv) == msorted
        @test _lo_order(nlab, indC, Cv) == nsorted
        mrun = _lo_run(msorted, indC, Cv)
        nrun = _lo_run(nsorted, indC, Cv)
        # ccsd_t_1 carries C's unit axis on A (run d); the others carry it on B.
        @test (name == "ccsd_t_1") == (mrun == d)
        @test nrun == (name == "ccsd_t_1" ? 1 : name == "ccsd_t_3" ? d^2 : d)

        # Machine-independent: name the kernel, so `mr` is 4 (swap for 2/3/4:
        # every N-run is >= 4 and no M-run reaches it) or 8 (swap for 3 only:
        # run 25 >= 8, runs of 5 do not).
        for (kernel, expect_swap) in (
                (ScalarKernel(Val(4), Val(3), Float64), name != "ccsd_t_1"),
                (SIMDKernel(Val(8), Val(6), Float64), name == "ccsd_t_3"),
            )
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel)
            swapped = plan.Astorage === parent(Bv)
            @test swapped == expect_swap
            @test swapped == _lo_swap(msorted, nsorted, indC, Cv, mr(kernel))
            if swapped
                # B feeds M: mgroup's maps are (B, C) over the sorted N labels.
                @test plan.mgroup.strides[2] == Tuple(cstride(l) for l in nsorted)
                @test plan.ngroup.strides[2] == Tuple(cstride(l) for l in msorted)
                @test plan.kgroup.strides == ((kB,), (kA,))
                @test plan.Bstorage === parent(Av)
                @test plan.Abase == offset(Bv) && plan.Bbase == offset(Av)
            else
                @test plan.mgroup.strides[2] == Tuple(cstride(l) for l in msorted)
                @test plan.ngroup.strides[2] == Tuple(cstride(l) for l in nsorted)
                @test plan.kgroup.strides == ((kA,), (kB,))
                @test plan.Bstorage === parent(Bv)
            end
            # The C map of each composite is ascending, whichever operand fed it.
            @test issorted(abs.(plan.mgroup.strides[2]))
            @test issorted(abs.(plan.ngroup.strides[2]))
            @test plan.mgroup.lengths == ntuple(_ -> d, 3)
            @test plan.ngroup.lengths == ntuple(_ -> d, 3)
        end

        # Default kernel: the same rule at this machine's `mr`.
        plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
        MRk = mr(plan.kernel)
        @test (plan.Astorage === parent(Bv)) == (mrun < MRk && nrun >= MRk)
    end
end

@testset "F2: run-length-aware kernel demotion (docs/decisions.md, \"F2\")" begin
    # The `ccsd_t_1` fixture (benchmark/profile_to_suite.jl's
    # `ccsd_t_1_dim16`/`ccsd_t_1_dim16_f32`, shrunk here to the minimum that
    # reproduces the identical demotion decision): C's leading unit-stride run
    # (labels a,i,j -- a is C-adjacent, i breaks it, whatever i/j/m/k/b/c's own
    # extent is) is exactly `d`, while Qm = d*extra^2 != d, so `Qm == run`
    # never saves this case; only `run % mr(kernel) == 0` can. `d` stays 16 (a
    # register-tile-sized run is the whole point); the other six axes shrink
    # to 4 -- `_leading_unit_run` only reads `a`'s own extent plus that the
    # NEXT M label's stride differs from it, so this reproduces the exact same
    # `run`/predicate outcome as the full dim=16 fixture at a small fraction of
    # the array/reference-loop cost.
    #
    # NOT hardware-derived: `mr(plan.kernel)`'s expected value below is
    # computed from `_default_kernel`/`kernel_shapes(T)` themselves, never a
    # literal -- a literal `mr` (or lack of demotion) is exactly the
    # ISA-specific hardcoding this file's own header note (`plan_contract:
    # SIMDKernel is the engine-wide default kernel`, above) warns against, and
    # is portable across avx512/avx2/neon/unknown-ISA hosts, checked via
    # `test/forced_isa_runner.jl` for avx2 and unknown/neon.
    d = 16
    extra = 4
    IA = (:i, :j, :m, :a)
    IB = (:m, :k, :b, :c)
    IC = (:a, :b, :c, :i, :j, :k)
    (indA, indB, indC), _ = _lo_labels(IA, IB)

    for T in (Float64, Float32)
        A = randn(T, extra, extra, extra, d)  # (i, j, m, a)
        B = randn(T, extra, extra, extra, extra)  # (m, k, b, c)
        C = zeros(T, d, extra, extra, extra, extra, extra)  # (a, b, c, i, j, k)
        Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)

        mlab, nlab, klab = QuasiStrided._classify_labels(indA, indB, indC)
        msorted = _lo_order(mlab, indC, Cv)
        run = _lo_run(msorted, indC, Cv)
        cpos(l) = findfirst(==(l), indC)::Int
        Qm = prod(size(Cv, cpos(l)) for l in mlab)
        Qn = prod(size(Cv, cpos(l)) for l in nlab)

        plan = plan_contract(Cv, Av, indA, Bv, indB, indC)
        # ccsd_t_1 never swaps (N's own leading run is 1, below every real
        # `mr`; pinned above, "label order: pinning test..."), so `plan.kernel`
        # is judged against A's own M composite computed here, on every ISA.
        @test plan.Astorage === parent(Av)

        default_kernel = QuasiStrided._default_kernel(T, Qm, Qn)
        default_mr = mr(default_kernel)
        if Qm == run || run % default_mr == 0
            # The predicate already holds for the shipped default: F2 must be
            # a no-op, on every ISA.
            @test plan.kernel === default_kernel
        else
            candidates = [sh[1] for sh in QuasiStrided.kernel_shapes(T) if run % sh[1] == 0]
            if isempty(candidates)
                # No menu shape fits either (mirrors the d=5 pinning fixture,
                # above): F2 falls back to leaving the kernel untouched.
                @test plan.kernel === default_kernel
            else
                # Demoted: the predicate now holds, at the LARGEST menu `mr`
                # that satisfies it -- not merely any satisfying entry.
                @test run % mr(plan.kernel) == 0
                @test mr(plan.kernel) == maximum(candidates)
            end
        end

        Cref = _lo_reference(C, Av, indA, Bv, indB, indC; alpha = 1.3, beta = -0.7)
        Ctw = copy(C)
        plan_tw = plan_contract(
            StridedView(Ctw), Av, indA, Bv, indB, indC; kernel = plan.kernel
        )
        execute_tilewise!(plan_tw, 1.3, -0.7)
        @test Ctw ≈ Cref

        Cex = copy(C)
        plan_ex = plan_contract(StridedView(Cex), Av, indA, Bv, indB, indC)
        execute!(plan_ex, 1.3, -0.7)
        @test Cex ≈ Cref
    end

    # Plain GEMM: `Qm == run` is always true (M's only label is C's own
    # unit-stride axis), so F2 must never fire regardless of dtype/mr -- the
    # kernel stays exactly `_default_kernel`'s choice.
    for T in (Float64, Float32)
        Ma, Ka, Na = 37, 11, 23
        Amat = randn(T, Ma, Ka)
        Bmat = randn(T, Ka, Na)
        Cmat = zeros(T, Ma, Na)
        Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
        plan = plan_contract(Cv, Av, (1, 2), Bv, (2, 3), (1, 3))
        @test plan.kernel === QuasiStrided._default_kernel(T, Ma, Na)

        Cref = Amat * Bmat
        execute!(plan, 1.0, 0.0)
        @test Cmat ≈ Cref
    end
end

@testset "label order: plain matmul is unchanged; a transposed output swaps" begin
    Ma, Ka, Na = 9, 4, 12
    Amat, Bmat = randn(Ma, Ka), randn(Ka, Na)
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    # Reuse ONE StridedView per operand for both `plan_contract` and the
    # `===` check below -- `parent(StridedView(x))` is not guaranteed
    # object-identical across two independently-constructed StridedViews of
    # the same `x` on Julia 1.10 (unlike 1.11+, where it resolves to the
    # Array's own `Memory{T}`), so comparing against a *fresh* StridedView
    # is a 1.10-only false failure, not a real behavior difference.
    Av, Bv = StridedView(Amat), StridedView(Bmat)

    # C[m,n] column-major: M is already C's unit axis, so nothing moves.
    Cmat = zeros(Ma, Na)
    p = plan_contract(StridedView(Cmat), Av, (1, 2), Bv, (2, 3), (1, 3); kernel = kernel)
    @test p.Astorage === parent(Av)
    @test p.mgroup.strides == ((1,), (1,))
    @test p.ngroup.strides == ((Ka,), (Ma,))
    execute!(p, 1.0, 0.0)
    @test Cmat ≈ Amat * Bmat

    # The same contraction written into C's transpose (C stored as (n, m)):
    # N carries the unit axis with a 12-wide run >= mr = 8, so B feeds M.
    Ct = zeros(Na, Ma)
    pt = plan_contract(StridedView(Ct), Av, (1, 2), Bv, (2, 3), (3, 1); kernel = kernel)
    @test pt.Astorage === parent(Bv)
    @test pt.mgroup.strides == ((Ka,), (1,))      # (B, C) maps over label 3
    @test pt.ngroup.strides == ((1,), (Na,))      # (A, C) maps over label 1
    @test pt.kgroup.strides == ((1,), (Ma,))      # (B, A) maps over label 2
    execute!(pt, 1.0, 0.0)
    @test Ct ≈ transpose(Amat * Bmat)
    # ... but not when the run is too short for the kernel.
    Ct2 = zeros(6, Ma)
    Bv6 = StridedView(Bmat[:, 1:6])
    pt2 = plan_contract(StridedView(Ct2), Av, (1, 2), Bv6, (2, 3), (3, 1); kernel = kernel)
    @test pt2.Astorage === parent(Av)
end

@testset "label order: correctness on ccsd_t shapes, permuted/sliced C, alpha/beta, conj" begin
    Random.seed!(0x1ABE_10DE)
    d = 6
    # C is a sliced, permuted view of a larger array: physical order
    # (c, k, a, j, b, i) with padding, presented as (a, b, c, i, j, k).
    perm = (3, 5, 1, 6, 4, 2)  # output axis p takes physical axis perm[p]
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        rtol = 200 * d * eps(real(T))
        for (name, IA, IB) in _LO_CASES, (conjA, conjB) in ((false, false), (true, true))
            (T <: Real) && conjA && continue  # conj is the identity on the real path
            (indA, indB, indC), (pA, pB, pAB) = _lo_labels(IA, IB)
            A = randn(T, d, d, d, d)
            B = randn(T, d, d, d, d)
            # A as a conj-wrapped view (its `op` composes with the flag by XOR),
            # B as a plain one.
            Av = conjA ? StridedView(A, size(A), strides(A), 0, conj) : StridedView(A)
            Bv = StridedView(B)
            Cbig = randn(T, d + 1, d + 2, d, d + 1, d, d + 3)
            Csub = view(Cbig, 1:d, 2:(d + 1), :, 2:(d + 1), :, 3:(d + 2))
            Cv = permutedims(StridedView(Csub), perm)
            @test offset(Cv) != 0
            @test !issorted(Base.strides(Cv))
            Cstart = copy(Cv)
            alpha = T <: Complex ? T(1.3, -0.4) : T(1.3)
            beta = T <: Complex ? T(0.7, 0.2) : T(0.7)

            Cref = _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA, conjB, alpha, beta)

            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; conjA = conjA, conjB = conjB)
            execute!(plan, alpha, beta)
            @test isapprox(copy(Cv), Cref; rtol = rtol)

            # The tile-by-tile oracle, on a fresh plan with the same swap decision.
            copyto!(Csub, permutedims(Cstart, invperm(perm)))
            plan_tw = plan_contract(Cv, Av, indA, Bv, indB, indC; conjA = conjA, conjB = conjB)
            @test (plan_tw.Astorage === parent(Bv)) == (plan.Astorage === parent(Bv))
            execute_tilewise!(plan_tw, alpha, beta)
            @test isapprox(copy(Cv), Cref; rtol = rtol)
        end
    end
end

@testset "label order: the swap never fires for complex kernels (guarded by T <: Real, deliberately deferred)" begin
    # Same shape/kernel that would trigger the swap for a real dtype at this
    # mr (ccsd_t_3, d=4: sorted N run 16 >= mr, sorted M run 1). Historically
    # PlanarKernel/OneMKernel (complex) always scatter-stored, so the swap had
    # nothing to win and measurably cost the as-is orientation's N-side
    # locality (~2-4%). PlanarKernel now has a vectorized store fast path
    # (src/kernels/planar.jl), so that rationale is stale, but the guard
    # itself (`_prefer_swap`'s call site, `T <: Real` in driver.jl) has NOT
    # been re-evaluated for the complex path yet -- extending it is a
    # deliberately deferred, unmeasured follow-up
    # (docs/proposals/complex-fast-paths.md Decision 3). This test only
    # confirms the current (unchanged) behavior: the swap still doesn't fire
    # for complex dtypes today. Also re-confirms conjugation is still correct
    # on the (now guaranteed unswapped) complex path -- an `op`-carrying A,
    # both flags exercised, checked against the loop reference.
    d = 4
    for T in (ComplexF64, ComplexF32)
        W = QuasiStrided._default_lanewidth(real(T))
        kernel = QuasiStrided.PlanarKernel(Val(W), Val(8), T, Val(W))
        @test mr(kernel) <= 16
        (indA, indB, indC), _ = _lo_labels(_LO_CASES[3][2], _LO_CASES[3][3])
        A = randn(T, d, d, d, d)
        B = randn(T, d, d, d, d)
        C = randn(T, d, d, d, d, d, d)
        Av = StridedView(A, size(A), strides(A), 0, conj)
        Bv = StridedView(B)
        Cv = StridedView(C)
        alpha, beta = T(0.5, 1.5), T(-1.0, 0.25)
        for (conjA, conjB) in ((true, true), (true, false), (false, true))
            Cstart = copy(C)
            Cref = _lo_reference(Cstart, Av, indA, Bv, indB, indC; conjA, conjB, alpha, beta)
            plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, conjA = conjA, conjB = conjB)
            @test plan.Astorage === parent(Av)   # the swap did NOT fire (complex)
            # A's view already carries `op = conj`; its effective transform is
            # conj XOR the flag, i.e. conj only when the flag is NOT set.
            @test plan.atransform === (conjA ? identity : conj)
            @test plan.btransform === (conjB ? conj : identity)
            execute!(plan, alpha, beta)
            @test isapprox(C, Cref; rtol = 200 * d * eps(real(T)))
            copyto!(C, Cstart)
            execute_tilewise!(plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, conjA = conjA, conjB = conjB), alpha, beta)
            @test isapprox(C, Cref; rtol = 200 * d * eps(real(T)))
        end
    end
end

@testset "label order: the swap threads conj/transforms correctly when forced (real kernel proxy)" begin
    # The swap branch in `plan_contract` is dtype-agnostic -- only the `T <:
    # Real` guard at the call site prevents it from firing for complex. To
    # keep direct test coverage of "transforms travel with the operands
    # under a swap" without relying solely on the code-reading argument,
    # exercise the swap on a REAL shape (conj is `identity` there, so this
    # checks storage/base/strides swap correctness, not conj folding -- the
    # conj-folding logic itself is dtype-independent and was covered by the
    # complex swap tests before this guard landed; see docs/decisions.md).
    d = 5
    (name, IA, IB) = _LO_CASES[3]  # ccsd_t_3: swaps at mr=8 (SIMDKernel(8,6))
    (indA, indB, indC), _ = _lo_labels(IA, IB)
    A, B = randn(d, d, d, d), randn(d, d, d, d)
    C = randn(d, d, d, d, d, d)
    Av, Bv, Cv = StridedView(A), StridedView(B), StridedView(C)
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    alpha, beta = 0.5, -1.0
    Cref = _lo_reference(copy(C), Av, indA, Bv, indB, indC; alpha, beta)
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel)
    @test plan.Astorage === parent(Bv)  # the swap fired
    @test plan.atransform === identity && plan.btransform === identity
    execute!(plan, alpha, beta)
    @test isapprox(C, Cref; rtol = 1.0e-10)
end

@testset "label order: the adapter path reaches the reordered plan" begin
    d = 6
    for T in (Float64, ComplexF64), (name, IA, IB) in _LO_CASES
        (indA, indB, indC), (pA, pB, pAB) = _lo_labels(IA, IB)
        A = randn(T, d, d, d, d)
        B = randn(T, d, d, d, d)
        C = randn(T, d, d, d, d, d, d)
        alpha = T <: Complex ? T(0.9, 0.3) : T(0.9)
        beta = T <: Complex ? T(-0.5, 0.1) : T(-0.5)
        Cref = _lo_reference(C, StridedView(A), indA, StridedView(B), indB, indC; alpha, beta)
        TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, alpha, beta, QuasiStrided.QuasiStridedBackend())
        @test isapprox(C, Cref; rtol = 200 * d * eps(real(T)))
    end
end
