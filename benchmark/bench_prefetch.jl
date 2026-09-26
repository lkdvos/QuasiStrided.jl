# Software-prefetch experiment: does any of the three EXPERIMENTAL prefetch
# sites (src/hardware/prefetch.jl) pay, and does any of them cost the plain
# shapes anything?
#
#   julia --project=. benchmark/bench_prefetch.jl
#   julia --project=. benchmark/bench_prefetch.jl --variants base,pack_b,pack_b@32 \
#       --dtypes Float64 --reps 21
#   julia --project=. benchmark/bench_prefetch.jl --smoke     # 1 rep, quick check
#
# `--tag NAME` suffixes the output files (`prefetch_NAME.csv`, ...).
#
# Two shape families, reported apart:
#
#   CONTROL   -- MAIN_SHAPES + SMALL_SHAPES, plain column-major GEMMs. A packs
#                through its contiguous fast path, so only `pack_b` and `macro`
#                even reach these; they must NOT regress.
#   SCATTERED -- `SCATTER_VARIANTS` (benchmark/harness.jl): negative strides,
#                transposed views, interleaved multi-axis groups, a
#                tensor-network 4-index case. These push both packers onto
#                their gather loops, which is where a prefetch could matter.
#
# `--family` (comma-separated, default `control,scattered`) picks the shape
# families; the two below are opt-in because of their walltime and memory:
#
#   LARGE     -- `LARGE_VARIANTS`: operands of 128-256 MB each (Float64), so
#                the packers stream from DRAM -- small-M/small-N shapes where
#                packing is a large share of the time, and 4096^3 square ones,
#                plain and scattered.
#   IRREGULAR -- `IRREGULAR_VARIANTS`: the most irregular gathers the engine
#                can express (many short axes with large pseudo-random
#                strides; it has no arbitrary-offset input -- see
#                `build_irregular` in harness.jl).
#
# Variants (`--variants`, comma-separated; `base` is always measured; the
# default is every single site below, i.e. all but `all`):
#
#   base         every site off (the shipped configuration)
#   pack_b       B gather loop, per lane, distance `--pack-dist` K steps (16)
#   pack_a       A gather loop (non-contiguous case only), per lane
#   pack_b_line  B gather loop, one prefetch per distinct cache line
#   pack_a_line  A gather loop, one prefetch per distinct cache line
#   macro        macro-kernel panel-ahead, `--macro-lines` lines (default 4)
#   ctile        C micro-tile before its K loop, `prefetcht0`
#   ctile_w      C micro-tile before its K loop, `prefetchw`
#   all          pack_a, pack_b and macro together (round 1's "all")
#   <site>@<d>   one site at an explicit distance, e.g. `pack_b@32`, `macro@8`
#
# Methodology, following bench_complex_efficiency.jl: single thread,
# warm-up-then-median over `REPS` (21) calls -- fewer for a case whose single
# call is slow (at least 3, and about `--budget` seconds (default 4) of timed
# calls per timing; the per-row `reps` column says how many), and a start/middle/end canary
# (always timed with every site off) whose spread is printed and warned on
# above 10%. In addition, within each (dtype, shape) the variants are timed
# ADJACENTLY -- base, then each variant, then base again -- so a drift between
# a variant and its own baseline cannot masquerade as an effect; the two base
# timings' spread (`base_spread`) is that shape's in-situ noise floor, and a
# variant ratio inside it is noise, not a finding. Ratios are
# `t_variant / mean(t_base1, t_base2)`: < 1 means the variant is faster.
#
# Switching a site recompiles the engine (it is a method redefinition; see
# `set_prefetch!`); the warm-up call of each timing absorbs that compile.
#
# This produces a per-machine measurement and nothing here is wired into any
# default. Run it on several microarchitectures
# (benchmark/submit_prefetch.sh) before concluding anything.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: set_prefetch!, prefetch_distance, PREFETCH_SITES

const SMOKE = hasflag("smoke")
const REPS = SMOKE ? 1 : argopt("reps", 21)
const PACK_DIST = argopt("pack-dist", 16)
const MACRO_LINES = argopt("macro-lines", 4)
const DTYPES_RUN = parse_dtypes(argopt("dtypes", SMOKE ? "Float64" : "Float64,Float32,ComplexF64,ComplexF32"))
const VARIANT_NAMES = let v = split(argopt("variants", "pack_b,pack_a,pack_b_line,pack_a_line,macro,ctile,ctile_w"), ',')
    filter(!=("base"), strip.(v))
end

const BUDGET_S = parse(Float64, argopt("budget", "4"))
const FAMILIES = Tuple(strip.(split(argopt("family", "control,scattered"), ',')))
for f in FAMILIES
    f in ("control", "scattered", "large", "irregular") ||
        throw(ArgumentError("unknown --family $(repr(f))"))
end

const CONTROL_SHAPES = SMOKE ? [MAIN_SHAPES[2], SMALL_SHAPES[1]] : vcat(MAIN_SHAPES, SMALL_SHAPES)
const SCATTERED = SMOKE ? (:default, :interleaved_groups) : SCATTER_VARIANTS

const OUTDIR = results_dir() * (haskey(ENV, "SLURM_JOB_ID") ? "-job" * ENV["SLURM_JOB_ID"] : "")
mkpath(OUTDIR)
# `--tag` names this run's files, so several runs can share one results dir.
const TAG = let t = argopt("tag", ""); isempty(t) ? "" : "_" * t end
const CSV_PATH = joinpath(OUTDIR, "prefetch$(TAG).csv")
const SUMMARY_PATH = joinpath(OUTDIR, "prefetch$(TAG)_summary.txt")
const PROV_PATH = joinpath(OUTDIR, "prefetch$(TAG)_PROVENANCE.txt")

# Variant name -> the (site, distance) settings it switches on; every other
# site is off.
const ROUND1_ALL = () -> ((:pack_a, PACK_DIST), (:pack_b, PACK_DIST), (:macro, MACRO_LINES))

default_distance(site::Symbol) =
    site === :macro ? MACRO_LINES : site in (:ctile, :ctile_w) ? 1 : PACK_DIST

function variant_settings(name::AbstractString)
    name == "all" && return ROUND1_ALL()
    if occursin('@', name)
        site, d = split(name, '@'; limit = 2)
        Symbol(site) in PREFETCH_SITES || throw(ArgumentError("unknown site in variant $(repr(name))"))
        return ((Symbol(site), parse(Int, d)),)
    end
    site = Symbol(name)
    site in PREFETCH_SITES || throw(ArgumentError("unknown variant $(repr(name))"))
    return ((site, default_distance(site)),)
end

function apply_variant!(settings)
    for site in PREFETCH_SITES
        set_prefetch!(site, 0)
    end
    for (site, d) in settings
        set_prefetch!(site, d)
    end
    return nothing
end

# `invokelatest`: `set_prefetch!` has just redefined a method from inside this
# function's own call tree, and only a latest-world call sees the new code.
function time_case(::Type{T}, fx) where {T}
    return Base.invokelatest() do
        plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
        execute!(plan, one(T), zero(T))           # compile (after a switch) + warm
        t1 = @elapsed execute!(plan, one(T), zero(T))
        reps = clamp(floor(Int, BUDGET_S / t1), min(3, REPS), REPS)
        (median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps), reps)
    end
end

canary(crng, label) = (apply_variant!(()); Base.invokelatest(run_canary, crng, label))

# One (dtype, shape): base, every variant, base again -- adjacent.
function measure_shape!(rows, csv, ::Type{T}, family, name, fx, M, K, N) where {T}
    apply_variant!(())
    tb1, _ = time_case(T, fx)
    tv = Float64[]
    rv = Int[]
    for v in VARIANT_NAMES
        apply_variant!(variant_settings(v))
        t, r = time_case(T, fx)
        push!(tv, t); push!(rv, r)
    end
    apply_variant!(())
    tb2, _ = time_case(T, fx)
    tbase = (tb1 + tb2) / 2
    bspread = abs(tb2 - tb1) / min(tb1, tb2)
    @printf(
        "%-10s %-9s %-24s base %9.3f us (%5.1f GF/s, spread %4.1f%%) ",
        T, family, name, 1.0e6tbase, gflops(T, M, K, N, tbase), 100bspread
    )
    for (v, t, nreps) in zip(VARIANT_NAMES, tv, rv)
        r = t / tbase
        @printf(" %s %.3f", v, r)
        println(
            csv, "$T,$family,$name,$M,$K,$N,$v,", @sprintf("%.9f,%.9f,%.9f,%.5f,%.5f", t, tb1, tb2, r, bspread), ",$nreps"
        )
        push!(rows, (dtype = T, family = family, shape = name, variant = v, ratio = r, bspread = bspread))
    end
    println()
    return nothing
end

function summarize(io, rows)
    println(io, "\n== geomean ratio t_variant / t_base per (dtype, family); < 1 = faster ==")
    println(io, "   (max base_spread in that group shown: ratios inside it are noise)")
    for T in DTYPES_RUN, family in FAMILIES
        sel = filter(r -> r.dtype == T && r.family == family, rows)
        isempty(sel) && continue
        maxspread = maximum(r.bspread for r in sel)
        @printf(io, "%-10s %-9s max base spread %4.1f%% :", T, family, 100maxspread)
        for v in VARIANT_NAMES
            rv = [r.ratio for r in sel if r.variant == v]
            @printf(io, "  %s %.3f [%.3f..%.3f]", v, geomean(rv), minimum(rv), maximum(rv))
        end
        println(io)
    end
    return nothing
end

function main()
    print_env_header(stdout, "bench_prefetch.jl")
    println("families = ", join(FAMILIES, ","), "   variants = base + ", join(VARIANT_NAMES, ", "), "   reps <= ", REPS,
        "   pack-dist = ", PACK_DIST, "   macro-lines = ", MACRO_LINES)
    for v in VARIANT_NAMES
        variant_settings(v)  # validate every name before spending any time
    end
    csv = open(CSV_PATH, "w")
    println(csv, "dtype,family,shape,M,K,N,variant,t_variant,t_base1,t_base2,ratio,base_spread,reps")
    rows = NamedTuple[]
    canaries = Float64[]
    crng = Random.MersenneTwister(0x0CA9A121)
    push!(canaries, canary(crng, "start"))

    for (i, T) in enumerate(DTYPES_RUN)
        if "control" in FAMILIES
            for spec in CONTROL_SHAPES
                fx = build_plain(T, spec, Random.MersenneTwister(0x9F7E))
                measure_shape!(rows, csv, T, "control", spec.name, fx, spec.Ma, spec.Ka, spec.Na)
            end
        end
        for (family, variants, build) in (
                ("scattered", SCATTERED, build_scattered),
                ("large", LARGE_VARIANTS, build_large),
                ("irregular", IRREGULAR_VARIANTS, build_irregular),
            )
            family in FAMILIES || continue
            for v in variants
                fx = build(T, Random.MersenneTwister(0x9F7E), v)
                measure_shape!(rows, csv, T, family, string(v), fx, fx.M, fx.K, fx.N)
                fx = nothing
                GC.gc()  # the large fixtures are hundreds of MB each
            end
        end
        flush(csv)
        i < length(DTYPES_RUN) && push!(canaries, canary(crng, "after-$T"))
    end
    push!(canaries, canary(crng, "end"))
    apply_variant!(())
    close(csv)

    spread = relative_spread(canaries)
    summarize(stdout, rows)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    spread > 0.1 && @warn "canary spread exceeds 10%: the machine was not quiet enough for " *
        "a few-percent conclusion. Re-run before believing any ratio."

    open(SUMMARY_PATH, "w") do io
        summarize(io, rows)
        @printf(io, "\ncanary spread: %.1f%%\n", 100spread)
    end
    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_prefetch.jl")
        println(io, "git commit = ", git_commit())
        println(io, "target = ", QuasiStrided.target_profile())
        println(io, "reps = ", REPS, "  pack-dist = ", PACK_DIST, "  macro-lines = ", MACRO_LINES)
        println(io, "variants = ", join(VARIANT_NAMES, ","))
        println(io, "families = ", join(FAMILIES, ","), "  budget = ", BUDGET_S, " s")
        println(io, "slurm job = ", get(ENV, "SLURM_JOB_ID", "none"), "  node = ", gethostname())
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
        println(io, "\nOne machine only; nothing here is wired into any default.")
    end
    println("\nwrote ", CSV_PATH)
    return nothing
end

main()
