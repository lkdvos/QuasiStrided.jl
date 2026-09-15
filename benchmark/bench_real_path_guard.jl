# Real-path regression guard for the complex element-type milestone.
#
# The milestone's first acceptance criterion is that adding complex support did
# not slow the real path down. The suite proves the real path is *correct* and
# `git diff` proves the hot functions are textually unedited, but neither can
# see a regression caused by (say) a new `ContractPlan` type parameter
# defeating a specialization. Only a measurement can, and this is it.
#
# It is NOT `bench_default_vs_legacy.jl`, which compares two configurations
# *within* one tree. This measures the same real default configuration so it
# can be compared *across two trees* -- the working tree against the
# milestone's base commit -- which cannot be done in a single process, because
# both trees define a module named `QuasiStrided`.
#
# Usage: run once per tree, back to back, then diff the two CSVs.
#
#   julia --project=. benchmark/bench_real_path_guard.jl
#
# HOW TO RUN AND READ IT -- all three points cost this project time to find,
# and the full account is in docs/decisions.md, "The real-path regression
# guard: no regression, and the resolution is ~5%":
#
#   * Sequence in ABBA order, with nothing else on the machine. Straight
#     base-then-new twice is confounded by wall-clock drift, and an 11-rep
#     comparison once invented two regressions that 21 reps erased.
#   * A *systematic one-sided* shift across every shape is a regression even
#     inside the noise band -- that is what a lost specialization looks like. A
#     lone outlier, or a pattern that does not reproduce across rounds, is
#     noise. This instrument resolves ~5-6% on geomean, no finer.
#   * Judge machine quietness from the middle/end canary pair only.
#     `canary[start]` reads ~10% FASTER than the others in every run of this
#     script, on both trees -- a warm-up effect in the canary itself, not drift
#     -- so the reported "canary spread" trips its own warning at ~11% while
#     the machine is in fact quiet.

include(joinpath(@__DIR__, "harness.jl"))

const REPS = 21
const GUARD_SHAPES = vcat(MAIN_SHAPES, SMALL_SHAPES)

const OUTDIR = results_dir()
mkpath(OUTDIR)

# The filename carries the commit and a run counter. `results_dir()` is keyed
# only by host and date, so without the counter the second run of the day
# silently overwrites the first -- which happened, and destroyed a round of
# data. The counter matters because the interesting comparison is sometimes the
# *same* tree twice: that is how the run-to-run noise floor gets established.
const RUN_TAG = let
    # `QS_GUARD_LABEL`: the tree that is NOT the working copy is usually
    # extracted with `git archive`, where `git_commit()` cannot work, so both
    # sides would otherwise carry the same tag and a reviewer could not tell a
    # two-tree comparison from a same-tree noise run. That happened too.
    c = get(ENV, "QS_GUARD_LABEL", git_commit())
    # `git_commit()` returns a human sentence on failure, so keep the filename
    # filesystem-safe rather than assuming a hash.
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
