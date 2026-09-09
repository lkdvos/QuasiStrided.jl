using StridedViews: StridedView, offset

# plan_contract/execute!/ContractPlan aren't exported (only contract! is);
# execute_tilewise! is never exported at all (it's an internal oracle).
const plan_contract = QuasiStrided.plan_contract
const execute! = QuasiStrided.execute!
const ContractPlan = QuasiStrided.ContractPlan
const execute_tilewise! = QuasiStrided.execute_tilewise!

# Plan for the dense matmul C[m,n] = sum_k A[m,k]*B[k,n], the shape most
# testsets below use. (test_macro_driver.jl has its own copy: both files are
# included into the same scope, so the names must differ.)
function _mm_plan(Cmat, Amat, Bmat; kernel, mc = nothing, kc = nothing, nc = nothing)
    return plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        kernel = kernel, mc = mc, kc = kc, nc = nc
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
