# Empirical crossover sweep for `execute_direct!` (src/execution/direct.jl):
# the unpacked, scalar, BLIS-SUP-style path for small contractions, versus the
# existing packed `execute!`.
#
#   julia --project=. benchmark/bench_skip_packing.jl
#
# Point: `plan_contract`/`execute!` always pack A/B into register-format
# panels before running the SIMD microkernel, even for tiny shapes where
# packing's O(MK+NK) cost isn't repaid by the microkernel's cache reuse.
# `execute_direct!` skips packing and the microkernel entirely (a plain scalar
# triple loop over the same AxisGroup-derived offsets `execute!` uses). This
# script sweeps shapes from tiny (M=N=K=1) up past where the packed path
# clearly wins, to find the empirical crossover -- per the project's
# "no claim without a measurement" rule, no threshold is hardcoded anywhere in
# src/ off the back of this file; a human reads the CSV/ranking and decides.
#
# Shapes deliberately include genuinely small M *and* N *and* K together (not
# just one small free extent, as `SMALL_SHAPES` in harness.jl does), plus a
# small-K family around the `dim63_2_1_2` regression shape (K=63) at a few
# M/N sizes, since that is the shape that motivated this work.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: execute_direct!

const REPS_TINY = 51   # small shapes need more reps: timer-resolution noise
const REPS_MAIN = 21   # matches bench_complex_efficiency.jl's REPS

# (M, K, N) triples. Doubling ladder from 1 up to past the packed crossover,
# plus asymmetric small-K cases (the dim63_2_1_2 family) and small-M/N cases
# (mirroring SMALL_SHAPES's motivation) at a few different absolute scales.
const SKIP_PACKING_SHAPES = [
    # --- every dim tiny: doubling ladder ---
    (1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16),
    (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256),
    # --- small-K family (dim63_2_1_2's K=63 is the motivating shape) ---
    (32, 63, 32), (64, 63, 64), (128, 63, 128), (256, 63, 256),
    (32, 1, 32), (64, 1, 64), (256, 1, 256),
    (32, 8, 32), (64, 8, 64), (256, 8, 256),
    # --- small-M or small-N, K not tiny ---
    (4, 64, 256), (256, 64, 4), (4, 256, 256), (256, 256, 4),
    (12, 64, 256), (256, 64, 12),
    # --- small M and N together, K larger ---
    (8, 256, 8), (16, 128, 16), (4, 512, 4),
]

reps_for(M::Int, K::Int, N::Int) = (M * K * N <= 64^3) ? REPS_TINY : REPS_MAIN

function time_packed(::Type{T}, fx; reps::Int) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute!(plan, one(T), zero(T))
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
    return t
end

function time_direct(::Type{T}, fx; reps::Int) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute_direct!(plan, one(T), zero(T))
    t = median_time_s(() -> execute_direct!(plan, one(T), zero(T)); reps = reps)
    return t
end

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "skip_packing.csv")
const PROV_PATH = joinpath(OUTDIR, "skip_packing_PROVENANCE.txt")

function main()
    print_env_header(stdout, "bench_skip_packing.jl")
    csv = open(CSV_PATH, "w")
    println(csv, "dtype,M,K,N,reps,t_packed,t_direct,gf_packed,gf_direct,speedup_direct_over_packed")

    canaries = Float64[]
    rng = Random.MersenneTwister(0xDEC0DE01)
    push!(canaries, run_canary(rng, "start"))

    for T in DTYPES  # Float64, Float32 only -- execute_direct! is exercised
        # for complex types in test/execution/test_direct.jl; this sweep is
        # about the packed-vs-direct crossover, which the real kernels'
        # measured menu already anchors, so keep the sweep itself real-only.
        for (M, K, N) in SKIP_PACKING_SHAPES
            spec = ShapeSpec("$(M)x$(K)x$(N)", M, K, N)
            fx = build_plain(T, spec, rng)
            reps = reps_for(M, K, N)

            t_packed = time_packed(T, fx; reps = reps)
            t_direct = time_direct(T, fx; reps = reps)
            gf_packed = gflops(T, M, K, N, t_packed)
            gf_direct = gflops(T, M, K, N, t_direct)
            speedup = t_packed / t_direct

            println(
                csv,
                "$T,$M,$K,$N,$reps,$t_packed,$t_direct,$gf_packed,$gf_direct,$speedup"
            )
            @printf(
                "%-10s %4dx%4dx%4d  packed %10.3e s (%6.2f GF/s)  direct %10.3e s (%6.2f GF/s)  direct/packed speedup %6.3fx\n",
                string(T), M, K, N, t_packed, gf_packed, t_direct, gf_direct, speedup
            )
        end
    end

    push!(canaries, run_canary(rng, "end"))
    close(csv)

    spread = relative_spread(canaries)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    if spread > 0.1
        @warn "canary spread exceeds 10%: the machine was not quiet enough for " *
            "a few-percent conclusion. Re-run before believing any crossover read " *
            "off this CSV."
    end

    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_skip_packing.jl")
        println(io, "git commit = ", git_commit())
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
        println(
            io,
            "\nNo threshold is derived automatically here -- read speedup_direct_over_packed",
        )
        println(
            io,
            "in skip_packing.csv against M,K,N and pick the crossover by hand, per shape",
        )
        println(io, "family (all-tiny vs small-K vs small-M/N).")
    end
    println("\nwrote ", CSV_PATH)
    return nothing
end

main()
