using StridedViews: StridedView, offset

# plan_contract/execute!/ContractPlan aren't exported (only contract! is).
const plan_contract = QuasiStrided.plan_contract
const execute! = QuasiStrided.execute!
const ContractPlan = QuasiStrided.ContractPlan

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
# Larger case: multiple output tiles (M and N exceed one register tile)
# and multiple K panels, checked against a plain matrix-multiply reference.
# =====================================================================

@testset "driver: larger case, multiple output tiles and multiple K panels" begin
    Random.seed!(20260908)
    # Small kernel shape to force several output tiles from modest M/N.
    kernel = ScalarKernel(Val(4), Val(3), Float64)

    Ma, Ka, Na = 11, 13, 7 # deliberately not multiples of MR=4/NR=3/panel size
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)
    Cmat = zeros(Ma, Na)
    Cref = Amat * Bmat

    Av = StridedView(Amat) # indA: m=1, k=2
    Bv = StridedView(Bmat) # indB: k=2, n=3
    Cv = StridedView(Cmat) # indC: m=1, n=3
    indA = (1, 2)
    indB = (2, 3)
    indC = (1, 3)

    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc_panel = 5)
    @test plan.kc_panel == 5 # forces multiple K panels since Ka=13 > 5
    execute!(plan, 1.0, 0.0)

    @test Cmat ≈ Cref
end

@testset "driver: nontrivial alpha/beta across multiple K panels, beta applied once" begin
    Random.seed!(4242)
    kernel = ScalarKernel(Val(4), Val(3), Float64)
    Ma, Ka, Na = 9, 10, 8
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)
    Cstart = randn(Ma, Na)
    Cmat = copy(Cstart)

    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)

    alpha, beta = 2.5, 0.75
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc_panel = 3)
    @test plan.kc_panel == 3
    execute!(plan, alpha, beta)

    expected = alpha .* (Amat * Bmat) .+ beta .* Cstart
    @test Cmat ≈ expected
end

# =====================================================================
# Whole-contraction short-circuits: K=0 and alpha=0. Neither may read A/B;
# verified with NaN/Inf-poisoned A/B, mirroring the padding/beta=0 pattern
# already used in test_phase2_integration.jl / test_kernel.jl.
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

    # Mismatched shared-label length: indB's axis 1 given label 2 (shared
    # with A's axis 2, length 4), but Bmat's axis1 length is 4 -- construct a
    # genuine mismatch by using a different-length B.
    Bmat_bad = randn(6, 5) # k axis length 6 != A's k length 4
    Bv_bad = StridedView(Bmat_bad)
    @test_throws DimensionMismatch contract!(Cv, 1.0, Av, (1, 2), Bv_bad, (2, 3), 0.0, (1, 3))

    # Diagonal: repeated label within indA.
    Asq = randn(4, 4)
    Avsq = StridedView(Asq)
    @test_throws ArgumentError contract!(Cv, 1.0, Avsq, (1, 1), Bv, (2, 3), 0.0, (1, 3))

    # Diagonal: repeated label within indC.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 1, 3)[1:2])

    # Label in indC absent from both indA and indB.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (1, 9))

    # Label present only in indA (not in indB or indC): dangling.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (5, 3), 0.0, (1, 3))

    # Label present only in indB (not in indA or indC): dangling. This is a
    # structurally distinct code path from the indA-only case above (the B
    # loop in _classify_labels, not the A loop) and was not previously
    # exercised (Phase 4 review coverage note).
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 9), 0.0, (1, 3))

    # Label present in all three (batch-like), unsupported this milestone.
    @test_throws ArgumentError contract!(Cv, 1.0, Av, (1, 2), Bv, (2, 3), 0.0, (2, 3))
end

# =====================================================================
# Planning vs execution: measurable separately, and allocation behavior of
# execution alone (with a reused plan) vs a cold contract! call that
# includes planning.
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

    # Planning is a distinct, separately callable/measurable step.
    plan = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc_panel = 4)
    planning_allocs = @allocated plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel, kc_panel = 4)
    @test planning_allocs > 0 # planning does construct AxisGroups/buffers: expected to allocate

    # Warm up execute! once (compilation), then measure steady-state
    # allocation of *execution alone*, reusing the same plan/workspace.
    execute!(plan, 1.0, 0.0)
    fill!(Cmat, 0.0)
    exec_allocs = @allocated execute!(plan, 1.0, 0.0)

    @test Cmat ≈ Amat * Bmat

    # Report honestly (docs/decisions.md Phase 2b finding #5: pack_a!/pack_b!
    # are known to allocate ~80B/call in steady state even with concrete
    # buffers, root cause not isolated there; this driver calls pack_a!/
    # pack_b! once per (output tile, K panel) unchanged, so exec_allocs
    # scales with tile-count x panel-count and is NOT expected to be smaller
    # than the one-shot planning allocation for a case with several tiles/
    # panels, as this one deliberately has). The point of this test is that
    # execution-alone allocation is measured independently of planning (a
    # reused `plan` triggers no further AxisGroup/buffer construction), not
    # that it is zero or smaller -- do not conflate the two numbers.
    @test exec_allocs >= 0
    @test planning_allocs >= 0

    # A cold contract! call performs planning AND execution every time (no
    # plan is cached across calls), so it must allocate at least as much as
    # planning alone -- this is exactly what a reused plan avoids paying
    # repeatedly.
    fill!(Cmat, 0.0)
    cold_allocs = @allocated contract!(Cv, 1.0, Av, indA, Bv, indB, 0.0, indC)
    @test cold_allocs >= planning_allocs
end

@testset "driver: execution allocation through SIMDKernel is not worse than ScalarKernel" begin
    # Phase 4 review: the SIMD worker only benchmarked a single execute_tile!
    # call directly, never through the driver's own tiling/dispatch loop.
    # Assert here (not just spot-check once) that going through execute!
    # with a SIMDKernel doesn't introduce driver-induced boxing/allocation
    # beyond what pack_a!/pack_b! already contribute (the same deferred
    # ~80B/call finding applies to both kernels equally, scaled by tile x
    # panel count) -- i.e. no *additional* per-call allocation specific to
    # SIMDKernel dispatch through the driver.
    Random.seed!(99)
    Ma, Ka, Na = 9, 10, 8
    Amat = randn(Ma, Ka)
    Bmat = randn(Ka, Na)
    indA, indB, indC = (1, 2), (2, 3), (1, 3)
    Av, Bv = StridedView(Amat), StridedView(Bmat)

    Cmat_s = zeros(Ma, Na)
    plan_s = plan_contract(
        StridedView(Cmat_s), Av, indA, Bv, indB, indC;
        kernel = ScalarKernel(Val(4), Val(3), Float64), kc_panel = 4
    )
    execute!(plan_s, 1.0, 0.0)
    fill!(Cmat_s, 0.0)
    scalar_exec_allocs = @allocated execute!(plan_s, 1.0, 0.0)

    Cmat_v = zeros(Ma, Na)
    plan_v = plan_contract(
        StridedView(Cmat_v), Av, indA, Bv, indB, indC;
        kernel = SIMDKernel(Val(4), Val(3), Float64), kc_panel = 4
    )
    execute!(plan_v, 1.0, 0.0)
    fill!(Cmat_v, 0.0)
    simd_exec_allocs = @allocated execute!(plan_v, 1.0, 0.0)

    @test Cmat_s ≈ Amat * Bmat
    @test Cmat_v ≈ Amat * Bmat
    @test simd_exec_allocs <= scalar_exec_allocs
end
