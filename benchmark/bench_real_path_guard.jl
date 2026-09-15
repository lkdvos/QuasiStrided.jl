# Real-path regression guard for the complex element-type milestone.
#
# The milestone's first acceptance criterion is that adding complex support did
# not slow the real path down. The test suite proves the real path is
# *correct*, and `git diff` proves the hot functions are textually unedited,
# but neither can see a regression caused by (say) a new `ContractPlan` type
# parameter defeating a specialization. Only a measurement can, and this is it.
#
# It is NOT `bench_default_vs_legacy.jl`, which compares two configurations
# *within* one tree. This measures the same real default configuration so that
# it can be compared *across two trees* -- the working tree against the
# milestone's base commit -- which cannot be done in a single process, because
# both trees define a module named `QuasiStrided`.
#
# Usage: run once per tree, back to back, then diff the two CSVs.
#
#   julia --project=. benchmark/bench_real_path_guard.jl
#
# Sequencing matters more than usual. ccqlin038 is not reliably exclusive,
# canary spreads of 4-15% are normal, and an 11-rep comparison in the
# panel-addressing milestone invented two regressions that 21 reps erased
# (STATUS.md, "Measurement hygiene"). So: check for other users' processes
# first, run the two trees with nothing else on the machine, and read the
# canary bracket before reading any ratio.
#
# How to read the result. A *systematic one-sided* shift across every shape is
# a regression even if each individual point sits inside the noise band -- that
# is what a lost specialization looks like. A single point outside the band
# with the others clean is noise, not a finding.

include(joinpath(@__DIR__, "harness.jl"))

const REPS = 21
const GUARD_SHAPES = vcat(MAIN_SHAPES, SMALL_SHAPES)

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "real_path_guard.csv")
const PROV_PATH = joinpath(OUTDIR, "real_path_guard_PROVENANCE.txt")

# Pass no kernel and no blocking: this must take the real default path,
# including the extent-aware demotion, exactly as a user gets it. Naming either
# would bypass the thing under test.
function time_real_default(::Type{T}, fx; reps::Int = REPS) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute!(plan, one(T), zero(T))
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
    return (t, plan.kernel, plan.blocking)
end

shape_tag(k) = "$(mr(k))x$(nr(k))/W$(lanewidth(k))"

function main()
    print_env_header(stdout, "bench_real_path_guard.jl")
    println()

    csv = open(CSV_PATH, "w")
    println(csv, "dtype,shape,Ma,Ka,Na,kernel,mc,kc,nc,reps,median_seconds,gflops")

    canaries = Float64[]
    crng = Random.MersenneTwister(0x0CA9A121)
    push!(canaries, run_canary(crng, "start"))

    for (di, T) in enumerate(DTYPES)
        rng = Random.MersenneTwister(0x00C0FFEE + di)
        for spec in GUARD_SHAPES
            fx = build_plain(T, spec, rng)
            t, kernel, b = time_real_default(T, fx)
            gf = gflops(T, spec.Ma, spec.Ka, spec.Na, t)
            println(
                csv,
                "$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),",
                "$(shape_tag(kernel)),$(b.mc),$(b.kc),$(b.nc),$REPS,$t,$gf"
            )
            @printf(
                "%-10s %-22s %9.3f ms  %7.2f GF/s  %s\n",
                T, spec.name, 1000t, gf, shape_tag(kernel)
            )
        end
        di == 1 && push!(canaries, run_canary(crng, "middle"))
    end

    # The scattered fixture too: the real default's zero-allocation scattered
    # path is where a lost specialization would surface first, and every
    # allocation regression this project has had was invisible on plain shapes.
    for (di, T) in enumerate(DTYPES)
        rng = Random.MersenneTwister(0x5CA7 + di)
        fx = build_scattered(T, rng)
        t, kernel, b = time_real_default(T, fx)
        println(
            csv,
            "$T,scattered_a64k64b16n64,64,64,1024,",
            "$(shape_tag(kernel)),$(b.mc),$(b.kc),$(b.nc),$REPS,$t,NaN"
        )
        @printf("%-10s %-22s %9.3f ms  (scattered)\n", T, "scattered", 1000t)
    end

    push!(canaries, run_canary(crng, "end"))
    close(csv)

    spread = (maximum(canaries) - minimum(canaries)) / minimum(canaries)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    if spread > 0.1
        @warn "canary spread exceeds 10%: the machine was not quiet enough for " *
            "a few-percent conclusion. Re-run before believing any ratio."
    end

    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_real_path_guard.jl")
        println(io, "git commit = ", git_commit())
        println(io, "reps = ", REPS)
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
    end

    println("\nwrote ", CSV_PATH)
    return nothing
end

main()
