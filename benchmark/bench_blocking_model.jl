# Block-size plateau sweep, and the analytical model against the constants.
#
#   julia --project=. benchmark/bench_blocking_model.jl [--scalar]
#
# Times `execute!` at the engine's own real SIMD kernel (`_default_kernel(T)`)
# over, per real dtype:
#
#   * named points: the analytical model (`_modelled_blocking`), the shipped
#     fallback row (`_fallback_blocking`), and the AVX-512 measured row --
#     whatever ISA this host is;
#   * `orig`: bench_driver.jl's 36-point grid, the grid behind
#     `default_blocking`'s "9%/11% best-to-worst" figure;
#   * `wide`: a log-spaced factorial, wide enough that model variants can be
#     scored offline from the same CSV;
#   * `nslice`/`mslice`: 1-D `nc` and `mc` slices at the model's other two
#     factors on a 2048x256x2048 shape, the only one large enough for `nc`
#     to bind.
#
# and, for the complex dtypes, the three named rows scaled by
# `_scale_blocking` at the default planar kernel.
#
# Every configuration of one shape is timed back to back, 21 reps, median,
# with the start/middle/end canary bracket. Ranking is harness.jl's `rank_by`:
# geomean over shapes of time normalized by that shape's best.
#
# `--scalar` also runs `orig` with the (8,6) ScalarKernel and ranks both
# kernels jointly, which is exactly how the 9%/11% figure was computed.
#
# writes blocking_model.csv + summary.txt + PROVENANCE.txt to
# benchmark/results/<hostname>-<date>/.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: Blocking, target_profile, _default_kernel, _modelled_blocking,
    _fallback_blocking, default_blocking, _scale_blocking, complex_method

const SMOKE = hasflag("smoke")  # 1 rep, a few points: checks the script runs
const REPS = SMOKE ? 1 : 21
const WITH_SCALAR = hasflag("scalar")
thin(v) = SMOKE ? v[1:min(end, 3)] : v

const GRID_SHAPES = vcat(MAIN_SHAPES, EXTRA_SHAPES, SMALL_SHAPES)
const BIG_SHAPE = ShapeSpec("2048x256x2048", 2048, 256, 2048)

const ORIG = Dict(
    Float64 => full_grid((64, 128, 256, 512), (128, 256, 512), (768, 1536, 3072)),
    Float32 => full_grid((96, 192, 384, 768), (192, 384, 768), (1152, 2304, 4608)),
)
const WIDE = Dict(
    Float64 => full_grid(
        (48, 96, 192, 384, 768), (64, 128, 192, 256, 384, 512, 768, 1024), (192, 768, 3072)
    ),
    Float32 => full_grid(
        (48, 96, 192, 384, 768), (128, 256, 384, 512, 768, 1024, 1536, 2048), (192, 768, 3072)
    ),
)
const NSLICE = (48, 96, 192, 384, 768, 1536, 3072, 6144)
const MSLICE = (24, 48, 96, 192, 384, 768, 1536)

const PROFILE = target_profile()
const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "blocking_model.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary.txt")
const PROV_PATH = joinpath(OUTDIR, "PROVENANCE.txt")

tup(b::Blocking) = (b.mc, b.kc, b.nc)
named_rows(::Type{T}) where {T <: Real} = (
    model = _modelled_blocking(PROFILE, T),
    fallback = _fallback_blocking(T),
    avx512row = default_blocking(Val(:avx512), T),
)

csv = open(CSV_PATH, "w")
println(csv, "set,kernel,dtype,shape,Ma,Ka,Na,mc,kc,nc,mc_eff,kc_eff,nc_eff,reps,median_seconds,gflops")

function time_point(set, kname, kernel, ::Type{T}, fx, spec, combo) where {T}
    mc, kc, nc = combo
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = kernel, mc = mc, kc = kc, nc = nc
    )
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS)
    b = plan.blocking
    gf = gflops(T, spec.Ma, spec.Ka, spec.Na, t)
    println(
        csv, "$set,$kname,$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),$mc,$kc,$nc,",
        "$(b.mc),$(b.kc),$(b.nc),$REPS,", @sprintf("%.9f,%.4f", t, gf)
    )
    flush(csv)
    return (set = set, kernel = kname, dtype = T, shape = spec.name, mc = mc, kc = kc, nc = nc, t = t)
end

# Every (set, combo) for one kernel at one shape, back to back.
function sweep_shape!(raw, kname, kernel, ::Type{T}, spec, points) where {T}
    fx = build_plain(T, spec, Random.MersenneTwister(0xB10C))
    for (set, combo) in points
        push!(raw, time_point(set, kname, kernel, T, fx, spec, combo))
    end
    return nothing
end

function main()
    print_env_header(stdout, "bench_blocking_model.jl")
    println("target = ", PROFILE)
    raw = NamedTuple[]
    canaries = Float64[]
    crng = Random.MersenneTwister(0x0CA9A121)
    push!(canaries, run_canary(crng, "start"))

    for T in DTYPES
        kernel = _default_kernel(T)
        rows = named_rows(T)
        println("\n$T kernel $(mr(kernel))x$(nr(kernel))/W$(lanewidth(kernel))  ", rows)
        rows.model === nothing && @warn "no cache geometry detected: model undefined"
        named = [(string(k), tup(v)) for (k, v) in pairs(rows) if v !== nothing]
        points = vcat(
            named, [("orig", c) for c in thin(ORIG[T])], [("wide", c) for c in thin(WIDE[T])]
        )
        for spec in thin(GRID_SHAPES)
            sweep_shape!(raw, "SIMDKernel", kernel, T, spec, points)
            println("  $T $(spec.name) done")
        end
        m = something(rows.model, rows.fallback)
        slices = vcat(
            named,
            [("nslice", (m.mc, m.kc, n)) for n in NSLICE],
            [("mslice", (x, m.kc, m.nc)) for x in MSLICE],
        )
        sweep_shape!(raw, "SIMDKernel", kernel, T, BIG_SHAPE, slices)
        if WITH_SCALAR
            sk = ScalarKernel(Val(8), Val(6), T)
            for spec in MAIN_SHAPES
                sweep_shape!(raw, "ScalarKernel", sk, T, spec, [("orig", c) for c in ORIG[T]])
            end
        end
        push!(canaries, run_canary(crng, "after-$T"))
    end

    # Complex: the named real rows through `_scale_blocking`, default planar kernel.
    for T in CDTYPES
        kernel = _default_kernel(T)
        meth = complex_method(kernel)
        rows = named_rows(real(T))
        named = [
            (string(k), tup(_scale_blocking(v, meth))) for (k, v) in pairs(rows) if v !== nothing
        ]
        println("\n$T kernel $(mr(kernel))x$(nr(kernel))/W$(lanewidth(kernel))  ", named)
        for spec in MAIN_SHAPES
            sweep_shape!(raw, "PlanarKernel", kernel, T, spec, named)
        end
    end
    push!(canaries, run_canary(crng, "end"))
    close(csv)

    spread = relative_spread(canaries)
    open(SUMMARY_PATH, "w") do io
        for out in (stdout, io)
            summarize(out, raw)
            @printf(out, "\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
        end
    end
    spread > 0.1 && @warn "canary spread exceeds 10%: re-run before believing any ranking."

    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_blocking_model.jl")
        println(io, "target = ", PROFILE)
        println(io, "git commit = ", git_commit())
        println(io, "reps = ", REPS, "  scalar = ", WITH_SCALAR)
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
    end
    println("\nwrote ", CSV_PATH)
    return nothing
end

# Geomean of time over `shapes`, each normalized by the best time any of
# `sets` reached at that shape (per kernel). Returns combo => geomean, sorted.
function rank_sets(raw, T, sets, shapes; kernels = ("SIMDKernel",))
    rows = filter(r -> r.dtype == T && r.set in sets && r.shape in shapes && r.kernel in kernels, raw)
    best = Dict{Tuple{String, String}, Float64}()
    for r in rows
        best[(r.kernel, r.shape)] = min(get(best, (r.kernel, r.shape), Inf), r.t)
    end
    g = Dict{Tuple{Int, Int, Int}, Vector{Float64}}()
    for r in rows
        push!(get!(g, (r.mc, r.kc, r.nc), Float64[]), r.t / best[(r.kernel, r.shape)])
    end
    return sort([(k, geomean(v)) for (k, v) in g]; by = last)
end

# Geomean over `shapes` of `combo`'s time normalized by the best time at each
# shape over every configuration measured there (all sets); a combo measured
# in several sets contributes the mean of its repeats at a shape.
function score(raw, T, combo, shapes)
    rows = filter(r -> r.dtype == T && r.kernel != "ScalarKernel" && r.shape in shapes, raw)
    best = Dict{String, Float64}()
    mine = Dict{String, Vector{Float64}}()
    for r in rows
        best[r.shape] = min(get(best, r.shape, Inf), r.t)
        (r.mc, r.kc, r.nc) == combo && push!(get!(mine, r.shape, Float64[]), r.t)
    end
    isempty(mine) && return NaN
    return geomean([sum(ts) / length(ts) / best[s] for (s, ts) in mine])
end

function summarize(io, raw)
    main_names = [s.name for s in MAIN_SHAPES]
    grid_names = [s.name for s in GRID_SHAPES]
    println(io, "\n# host ", gethostname(), "  cpu ", Sys.CPU_NAME, "  isa ", PROFILE.isa)
    for T in DTYPES
        rows = named_rows(T)
        println(io, "\n## $T")
        o = rank_sets(raw, T, ("orig",), main_names)
        @printf(io, "orig 36-grid, MAIN_SHAPES, SIMD only: best %.4f %s  worst %.4f %s  spread %.1f%%\n",
            o[1][2], o[1][1], o[end][2], o[end][1], 100 * (o[end][2] / o[1][2] - 1))
        if WITH_SCALAR
            os = rank_sets(raw, T, ("orig",), main_names; kernels = ("SIMDKernel", "ScalarKernel"))
            @printf(io, "orig 36-grid, MAIN_SHAPES, SIMD+Scalar jointly (the 9%%/11%% method): spread %.1f%%\n",
                100 * (os[end][2] / os[1][2] - 1))
        end
        w = rank_sets(raw, T, ("wide", "orig", "model", "fallback", "avx512row"), grid_names)
        @printf(io, "all grid points, GRID_SHAPES: best %.4f %s  worst %.4f %s  spread %.1f%%\n",
            w[1][2], w[1][1], w[end][2], w[end][1], 100 * (w[end][2] / w[1][2] - 1))
        within(tol) = count(x -> x[2] <= w[1][2] * (1 + tol), w)
        println(io, "points within 3%/6%/10% of best: ", within(0.03), "/", within(0.06), "/", within(0.10), " of ", length(w))
        println(io, "top 5: ", w[1:min(5, end)])
        for (k, v) in pairs(rows)
            v === nothing && continue
            @printf(io, "  %-10s %-18s GRID_SHAPES %.4f   MAIN_SHAPES %.4f   big %.4f\n", k, tup(v),
                score(raw, T, tup(v), grid_names), score(raw, T, tup(v), main_names),
                score(raw, T, tup(v), [BIG_SHAPE.name]))
        end
        for s in ("nslice", "mslice")
            pts = sort([r for r in raw if r.dtype == T && r.set == s]; by = r -> s == "nslice" ? r.nc : r.mc)
            isempty(pts) && continue
            tb = minimum(r.t for r in pts)
            println(io, "  $s @ $(BIG_SHAPE.name): ", join([@sprintf("%d:%.3f", s == "nslice" ? r.nc : r.mc, r.t / tb) for r in pts], "  "))
        end
    end
    for T in CDTYPES
        println(io, "\n## $T (planar default kernel, scaled rows)")
        rows = filter(r -> r.dtype == T, raw)
        isempty(rows) && continue
        best = Dict{String, Float64}()
        for r in rows
            best[r.shape] = min(get(best, r.shape, Inf), r.t)
        end
        for set in unique(r.set for r in rows)
            v = [r.t / best[r.shape] for r in rows if r.set == set]
            c = first(r for r in rows if r.set == set)
            @printf(io, "  %-10s %-18s MAIN_SHAPES %.4f\n", set, (c.mc, c.kc, c.nc), geomean(v))
        end
    end
    return nothing
end

main()
