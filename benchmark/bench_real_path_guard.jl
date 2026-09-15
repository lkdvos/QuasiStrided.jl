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
# with the others clean is noise, not a finding. And a pattern that does not
# reproduce between two rounds is noise no matter how tidy it looks in one.
#
# Known artefact, measured: `canary[start]` reads systematically ~10% FASTER
# than `canary[middle]`/`canary[end]`, in every run of this script so far, on
# both trees. That is a warm-up effect in the canary itself, not machine drift
# -- the middle-to-end spread is 0.2-3.4% in the same runs. So the reported
# "canary spread" here reads ~11% and trips its own warning while the machine
# is in fact quiet. Judge quietness from the middle/end pair; the start canary
# is useful only as a fixed reference across runs.

include(joinpath(@__DIR__, "harness.jl"))

const REPS = 21
const GUARD_SHAPES = vcat(MAIN_SHAPES, SMALL_SHAPES)

const OUTDIR = results_dir()
mkpath(OUTDIR)

# The filename carries the commit and a run counter, which matters more here
# than for any other script in this directory: this one exists to be run
# against TWO trees and diffed, and `results_dir()` is keyed only by host and
# date. Without this, the second run of the day silently overwrites the first
# -- which happened on the first use of this script and destroyed a round of
# data before it was noticed. A run counter as well as the commit, because the
# interesting comparison is sometimes the *same* tree twice (that is how the
# run-to-run noise floor is established, and a noise floor measured from one
# run is not a noise floor).
const RUN_TAG = let
    c = git_commit()
    # `git_commit()` returns a human sentence on failure ("unknown (git
    # rev-parse failed)"), which is how a tree extracted with `git archive`
    # reports -- exactly the situation this script is built for. Keep the
    # filename filesystem-safe rather than assuming a hash.
    safe = replace(c, r"[^A-Za-z0-9]" => "")
    short = length(safe) >= 8 ? safe[1:8] : safe
    n = 1
    while isfile(joinpath(OUTDIR, "real_path_guard_$(short)_run$(n).csv"))
        n += 1
    end
    "$(short)_run$(n)"
end
const CSV_PATH = joinpath(OUTDIR, "real_path_guard_$(RUN_TAG).csv")
const PROV_PATH = joinpath(OUTDIR, "real_path_guard_$(RUN_TAG)_PROVENANCE.txt")

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
