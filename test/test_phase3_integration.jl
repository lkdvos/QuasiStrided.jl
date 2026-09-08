# Integration gate: plan_contract/execute! accept SIMDKernel as a drop-in
# swap for ScalarKernel, agreeing numerically through the real driver.

using StridedViews: StridedView

@testset "Phase 3 integration: driver + SIMDKernel swap" begin
    rng = MersenneTwister(20260908)
    Amat = rand(rng, 11, 13)
    Bmat = rand(rng, 13, 9)
    ref = Amat * Bmat

    A = StridedView(Amat)
    B = StridedView(Bmat)

    Cscalar = zeros(11, 9)
    plan_s = plan_contract(StridedView(Cscalar), A, (1, 2), B, (2, 3), (1, 3);
                            kernel=ScalarKernel(Val(8), Val(6), Float64), kc_panel=5)
    execute!(plan_s, 1.0, 0.0)

    Csimd = zeros(11, 9)
    plan_v = plan_contract(StridedView(Csimd), A, (1, 2), B, (2, 3), (1, 3);
                            kernel=SIMDKernel(Val(8), Val(6), Float64), kc_panel=5)
    execute!(plan_v, 1.0, 0.0)

    @test isapprox(Cscalar, ref)
    @test isapprox(Csimd, ref)
    @test isapprox(Cscalar, Csimd; atol=1e-10)

    # Reused plan, nontrivial alpha/beta, second execute! call (workspace
    # reuse across calls with a different kernel instance).
    Cstart = rand(rng, 11, 9)
    Cscalar2 = copy(Cstart)
    Csimd2 = copy(Cstart)
    plan_s2 = plan_contract(StridedView(Cscalar2), A, (1, 2), B, (2, 3), (1, 3);
                             kernel=ScalarKernel(Val(4), Val(3), Float64), kc_panel=4)
    plan_v2 = plan_contract(StridedView(Csimd2), A, (1, 2), B, (2, 3), (1, 3);
                             kernel=SIMDKernel(Val(4), Val(3), Float64), kc_panel=4)
    execute!(plan_s2, 2.5, 0.75)
    execute!(plan_v2, 2.5, 0.75)
    expected = 2.5 .* ref .+ 0.75 .* Cstart
    @test isapprox(Cscalar2, expected)
    @test isapprox(Csimd2, expected; atol=1e-8)
end
