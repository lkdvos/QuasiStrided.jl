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
