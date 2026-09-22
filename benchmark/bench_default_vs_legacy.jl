# The configuration this package ships now against the one it shipped before
# hardware detection existed. Both measured as a user would get them -- no
# pinned mc/kc/nc, no grid. This is the number quoted in docs/decisions.md,
# Phases G and H, and it doubles as a regression guard.
#
#   julia --project=. benchmark/bench_default_vs_legacy.jl

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: _default_kernel, default_blocking, _fallback_blocking, _fallback_shape

# The pre-detection configuration, reconstructed.
function legacy_config(::Type{T}) where {T}
    MR, NR, W = _fallback_shape(T)
    return (SIMDKernel(Val(MR), Val(NR), T, Val(W)), _fallback_blocking(T))
end

# What `contract!` / `plan_contract` / QuasiStridedBackend use by default now.
function derived_config(::Type{T}) where {T}
    kernel = _default_kernel(T)
    return (kernel, default_blocking(kernel))
end

# The derived side passes NOTHING, so it takes the real default path including
# the extent-aware demotion; naming the kernel would bypass that.
function time_legacy(::Type{T}, fx, kernel, b; reps = 21) where {T}
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = kernel, mc = b.mc, kc = b.kc, nc = b.nc
    )
    execute!(plan, one(T), zero(T))
    return median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
end

function time_default(::Type{T}, fx; reps = 21) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute!(plan, one(T), zero(T))
    return (median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps), plan.kernel)
end

print_env_header(stdout, "bench_default_vs_legacy.jl")
println("target = ", QuasiStrided.target_profile())

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "default_vs_legacy.csv")

rng = MersenneTwister(0xB3_C4_0003)
csv = open(CSV_PATH, "w")
println(csv, "dtype,shape,legacy_MR,legacy_NR,legacy_W,new_MR,new_NR,new_W,legacy_s,new_s,speedup")

canaries = Float64[]
push!(canaries, run_canary(rng, "A (start)"))

for T in DTYPES
    kold, bold = legacy_config(T)
    knew, bnew = derived_config(T)
    println("\n== ", T, " ==")
    println("  legacy : MR=", mr(kold), " NR=", nr(kold), " W=", lanewidth(kold), "  ", bold)
    println("  derived: MR=", mr(knew), " NR=", nr(knew), " W=", lanewidth(knew), "  ", bnew)
    println("           (per-contraction demotion may pick a smaller shape; see last column)")
    println(
        rpad("  shape", 30), lpad("legacy s", 14), lpad("new s", 14),
        lpad("speedup", 10), lpad("shape used", 12)
    )

    speedups = Float64[]
    for spec in vcat(MAIN_SHAPES, SMALL_SHAPES)
        fx = build_plain(T, spec, rng)
        told = time_legacy(T, fx, kold, bold)
        tnew, kused = time_default(T, fx)
        push!(speedups, told / tnew)
        println(
            rpad("  " * spec.name, 30), lpad(@sprintf("%.6e", told), 14),
            lpad(@sprintf("%.6e", tnew), 14), lpad(@sprintf("%.3f", told / tnew), 10),
            lpad(string(mr(kused), "x", nr(kused), "/W", lanewidth(kused)), 12)
        )
        println(
            csv, "$T,$(spec.name),$(mr(kold)),$(nr(kold)),$(lanewidth(kold)),",
            "$(mr(kused)),$(nr(kused)),$(lanewidth(kused)),",
            @sprintf("%.9f,%.9f,%.4f", told, tnew, told / tnew)
        )
    end

    fxs = build_scattered(T, rng)
    told = time_legacy(T, fxs, kold, bold)
    tnew, kused = time_default(T, fxs)
    push!(speedups, told / tnew)
    println(
        rpad("  scattered_a64k64b16n64", 30), lpad(@sprintf("%.6e", told), 14),
        lpad(@sprintf("%.6e", tnew), 14), lpad(@sprintf("%.3f", told / tnew), 10),
        lpad(string(mr(kused), "x", nr(kused), "/W", lanewidth(kused)), 12)
    )
    println(
        csv, "$T,scattered_a64k64b16n64,$(mr(kold)),$(nr(kold)),$(lanewidth(kold)),",
        "$(mr(kused)),$(nr(kused)),$(lanewidth(kused)),",
        @sprintf("%.9f,%.9f,%.4f", told, tnew, told / tnew)
    )

    println(
        "  geomean speedup = ", @sprintf("%.3f", geomean(speedups)),
        "   worst = ", @sprintf("%.3f", minimum(speedups)),
        "   best = ", @sprintf("%.3f", maximum(speedups))
    )
    push!(canaries, run_canary(rng, "after $T"))
end

close(csv)
println("\ncanary spread (max-min)/min = ", @sprintf("%.4f", relative_spread(canaries)))
println("Results in ", CSV_PATH)
