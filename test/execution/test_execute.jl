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
    # Driving the SIMD kernel through execute!'s own dispatch loop (not just a
    # bare execute_tile! call) adds no SIMD-specific allocation over the
    # scalar path.
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
# Allocation targets: the macro-blocking execute! must not box a QSTile
# UnionAll.
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
# Type stability: `ContractWorkspace`'s `VT` parameter must not reintroduce
# the QSTile-UnionAll boxing described above.
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
# by Julia 1.10's compiler, so this
# is skipped there exactly as test/microkernels/test_simd_kernel.jl skips its own -- never
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

@testset "a non-identity pack transform crosses _pack_sliver! without allocating" begin
    # The real path never builds this plan -- `_qs_isconj` is false for a real
    # eltype, which is the point -- but the failure mode pinned here (a
    # transform reaching `_pack_sliver!` as a Union, ~80 B/call of dynamic
    # dispatch) is a property of the `TF`/`TA`/`TB` type parameters, not of
    # complex arithmetic. Building the plan directly exercises a genuine second
    # specialization on the real path. `conj` is the elementwise identity on a
    # real, so the result must be unchanged.
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
# Store fast path. `store_tile!`'s vectorized path is reachable from the real
# driver because its guard admits any `DenseVector{T}`: the driver's
# destination storage is `parent(C)`, i.e. `Memory{T}` on Julia >= 1.11.
# That makes the path's row tail a live allocation risk (a dynamically
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

    # The destination really is the guard's `DenseVector` case, and its micro-tile
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
