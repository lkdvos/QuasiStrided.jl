# The complex element-type milestone's headline metric, plus the planar-vs-1m
# head-to-head.
#
#   julia --project=. benchmark/bench_complex_efficiency.jl
#
# The metric is the **complex efficiency ratio**: one engine's complex
# throughput divided by its own real throughput at the same shape, with complex
# charged 8 flops per multiply-accumulate (`flops_per_mac`, deliberately the
# textbook count -- see its docstring for why it is not reduced for 1m).
#
# `1.0` means complex is treated exactly as well as real. It should exceed 1:
# complex is 4x the flops on 2x the bytes, i.e. twice the arithmetic intensity,
# so packing and per-call overheads amortise *better*. The reference project
# measures 1.42-1.47 and replicates that on two microarchitectures. Below ~0.9
# indicates a structural overhead specific to complex -- a packing cost or an
# accumulator spill -- and is a finding, not a result to publish.
#
# Being a WITHIN-run, within-engine ratio is what makes it worth reporting at
# all: it survives machine-to-machine variation where an absolute GFLOP/s
# number does not. It is still a single-machine measurement and is labelled as
# one.
#
# On the planar-vs-1m arm, read `docs/decisions.md`'s "Method ranking does not
# transfer between machines" first. The reference project measured four
# different orderings on four machines; this arm produces a ccqlin038 number
# and **must not** become an auto-dispatch rule. Its more interesting output is
# arguably the register-pressure question Phase C left open: whether the
# menu-head shape that spills (16x6 planar, 26 stores per K step) actually
# loses to the clean alternative (24x3, zero spills), which spill counts alone
# cannot answer.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: PlanarMethod, OneMMethod, complex_method, kernel_shapes,
    _kernel_from_shape, _default_kernel, default_blocking

const REPS = 21
const SWEEP_SHAPES = vcat(MAIN_SHAPES, SMALL_SHAPES)

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "complex_efficiency.csv")
const METHOD_CSV = joinpath(OUTDIR, "complex_method_shapes.csv")
const PROV_PATH = joinpath(OUTDIR, "complex_efficiency_PROVENANCE.txt")

shape_tag(k) = "$(mr(k))x$(nr(k))/W$(lanewidth(k))"

# Times the default path (no kernel, no blocking named), i.e. what a user gets.
function time_default(::Type{T}, fx; reps::Int = REPS) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute!(plan, one(T), zero(T))
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
    return (t, plan.kernel, plan.blocking)
end

# Times one explicitly named kernel at its own default blocking, so each method
# and shape is measured with the cache budget it actually ships with.
function time_kernel(::Type{T}, fx, kernel; reps::Int = REPS) where {T}
    b = default_blocking(kernel)
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = kernel, mc = b.mc, kc = b.kc, nc = b.nc
    )
    execute!(plan, one(T), zero(T))
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
    return (t, b)
end

# ---------------------------------------------------------------------------
# Arm 1: complex efficiency, at the shipped defaults
# ---------------------------------------------------------------------------
function arm_efficiency(csv, canaries, crng)
    println("\n== arm 1: complex efficiency at the shipped defaults ==")
    println("(ratio > 1 means complex amortises overhead better than real, as expected)\n")
    ratios64 = Float64[]
    ratios32 = Float64[]
    for (Treal, Tcplx, acc) in (
            (Float64, ComplexF64, ratios64), (Float32, ComplexF32, ratios32),
        )
        for spec in SWEEP_SHAPES
            rr = Random.MersenneTwister(0x0EFF1C1E)
            fxr = build_plain(Treal, spec, rr)
            fxc = build_plain(Tcplx, spec, rr)
            # Adjacent: the two arms of every ratio are timed back to back, so
            # a drift between them cannot masquerade as an effect.
            tr, kr, br = time_default(Treal, fxr)
            tc, kc, bc = time_default(Tcplx, fxc)
            gr = gflops(Treal, spec.Ma, spec.Ka, spec.Na, tr)
            gc = gflops(Tcplx, spec.Ma, spec.Ka, spec.Na, tc)
            eff = complex_efficiency(gc, gr)
            push!(acc, eff)
            println(
                csv,
                "efficiency,$Treal,$Tcplx,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),",
                "$(shape_tag(kr)),$(shape_tag(kc)),$tr,$tc,$gr,$gc,$eff,$REPS"
            )
            @printf(
                "%-10s %-22s real %7.2f GF/s   cplx %7.2f GF/s   eff %5.3f\n",
                Tcplx, spec.name, gr, gc, eff
            )
        end
        push!(canaries, run_canary(crng, "after-$(Tcplx)"))
    end
    @printf(
        "\ngeomean complex efficiency: ComplexF64 %.3f   ComplexF32 %.3f\n",
        geomean(ratios64), geomean(ratios32)
    )
    for (T, g) in ((ComplexF64, geomean(ratios64)), (ComplexF32, geomean(ratios32)))
        if g < 0.9
            @warn "$T complex efficiency geomean $(round(g, digits = 3)) < 0.9: " *
                "that indicates a structural overhead specific to complex " *
                "(packing cost or accumulator spill), and is a finding rather " *
                "than a result to publish. See docs/decisions.md."
        end
    end
    return (geomean(ratios64), geomean(ratios32))
end

# ---------------------------------------------------------------------------
# Arm 2: planar vs 1m, and the shape question Phase C left open
# ---------------------------------------------------------------------------
function arm_methods(csv, canaries, crng)
    println("\n== arm 2: complex methods x register shapes (ccqlin038 only) ==")
    println("(NO auto-dispatch rule is derived from this; see docs/decisions.md)\n")
    for T in CDTYPES
        rows = NamedTuple[]
        for method in (PlanarMethod(), OneMMethod())
            for shape in kernel_shapes(T, method)
                kernel = try
                    _kernel_from_shape(shape, T, method)
                catch err
                    @info "skipping $T $(method) $(shape): $(sprint(showerror, err))"
                    continue
                end
                for spec in SWEEP_SHAPES
                    rng = Random.MersenneTwister(0x5EED)
                    fx = build_plain(T, spec, rng)
                    t, b = time_kernel(T, fx, kernel)
                    gf = gflops(T, spec.Ma, spec.Ka, spec.Na, t)
                    push!(
                        rows, (
                            method = string(method), shape = shape_tag(kernel),
                            case = spec.name, t = t, gflops = gf,
                            kernel = shape_tag(kernel),
                            reals = panel_reals_per_element(kernel),
                        )
                    )
                    println(
                        csv,
                        "method,$T,$(method),$(shape_tag(kernel)),$(spec.name),",
                        "$(spec.Ma),$(spec.Ka),$(spec.Na),$t,$gf,",
                        "$(panel_reals_per_element(kernel)),$(b.mc),$(b.kc),$(b.nc),$REPS"
                    )
                end
                @printf(
                    "  %-12s %-14s  geomean GF/s %7.2f   panel reals/elt %d\n",
                    string(method), shape_tag(kernel),
                    geomean(
                        [
                            r.gflops for r in rows if r.kernel == shape_tag(kernel) &&
                                r.method == string(method)
                        ]
                    ),
                    panel_reals_per_element(kernel)
                )
            end
        end
        push!(canaries, run_canary(crng, "after-methods-$T"))
        # Rank by geomean of per-case time normalized by the best at that case,
        # so shapes of different absolute cost weigh equally.
        best = Dict{String, Float64}()
        for r in rows
            best[r.case] = min(get(best, r.case, Inf), r.t)
        end
        byconf = Dict{Tuple{String, String}, Vector{Float64}}()
        for r in rows
            push!(get!(byconf, (r.method, r.kernel), Float64[]), r.t / best[r.case])
        end
        ranked = sort([(k, geomean(v)) for (k, v) in byconf]; by = x -> x[2])
        println("\n  $T ranking (1.000 = best; ccqlin038, $(REPS) reps, this run only):")
        for (k, g) in ranked
            @printf("    %-12s %-14s %.3f\n", k[1], k[2], g)
        end
    end
    return nothing
end

function main()
    print_env_header(stdout, "bench_complex_efficiency.jl")
    csv = open(CSV_PATH, "w")
    println(
        csv,
        "arm,dtype_a,dtype_b,shape,Ma,Ka,Na,kernel_a,kernel_b,t_a,t_b,gf_a,gf_b,ratio,reps"
    )
    canaries = Float64[]
    crng = Random.MersenneTwister(0x0CA9A121)
    push!(canaries, run_canary(crng, "start"))

    g64, g32 = arm_efficiency(csv, canaries, crng)
    arm_methods(csv, canaries, crng)

    push!(canaries, run_canary(crng, "end"))
    close(csv)

    spread = (maximum(canaries) - minimum(canaries)) / minimum(canaries)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    if spread > 0.1
        @warn "canary spread exceeds 10%: the machine was not quiet enough for " *
            "a few-percent conclusion. Re-run before believing any ranking."
    end

    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_complex_efficiency.jl")
        println(io, "git commit = ", git_commit())
        println(io, "reps = ", REPS)
        @printf(io, "geomean complex efficiency: ComplexF64 %.4f  ComplexF32 %.4f\n", g64, g32)
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
        println(io, "\nMachine: ccqlin038 only. No ranking here transfers to another")
        println(io, "microarchitecture, and none of it is wired into dispatch.")
    end
    println("\nwrote ", CSV_PATH)
    return nothing
end

main()
