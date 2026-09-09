# Phase E block-size sweep (docs/decisions.md, "Macro-blocking milestone").
# Times `execute!` over a grid of shapes x (mc,kc,nc) for
# ScalarKernel/SIMDKernel x Float64/Float32, against `execute_tilewise!` and
# a `LinearAlgebra.mul!` line, and reports a geomean ranking. No cache model
# or probing by design -- just measured grid points.
#
# `mul!` is BLAS on a plain dense matmul of the same M/K/N: a reference
# point, NOT a target -- a "slower than mul!" ratio is not a problem.
#
#   julia --project=. benchmark/bench_driver.jl
#
# writes results + PROVENANCE.txt to benchmark/results/<hostname>-<date>/.

using QuasiStrided
using QuasiStrided: execute_tilewise!
using StridedViews: StridedView
using LinearAlgebra
using Statistics: median
using Random
using Dates
using Printf

# Single-core measurement discipline (this project's standing rule).
LinearAlgebra.BLAS.set_num_threads(1)
const NTHREADS = Threads.nthreads()
const BLAS_THREADS = LinearAlgebra.BLAS.get_num_threads()
if NTHREADS != 1
    @warn "Threads.nthreads() = $NTHREADS != 1 -- this is NOT the pinned " *
        "single-core measurement this project's rules require. Results " *
        "below should not be trusted as the reference-machine numbers."
end

# Warm up once (discarded), then `reps` timed calls; median, not mean.
function median_time_s(f!::Function; reps::Int = 5)
    f!()  # warm-up, discarded
    ts = Vector{Float64}(undef, reps)
    for r in 1:reps
        t0 = time_ns()
        f!()
        t1 = time_ns()
        ts[r] = (t1 - t0) / 1.0e9
    end
    return median(ts)
end

# Shapes.
struct ShapeSpec
    name::String
    Ma::Int
    Ka::Int
    Na::Int
end

const MAIN_SHAPES = [
    ShapeSpec("64^3", 64, 64, 64),
    ShapeSpec("128^3", 128, 128, 128),
    ShapeSpec("256^3", 256, 256, 256),
    ShapeSpec("512^3", 512, 512, 512),
    ShapeSpec("shallowK_256x24x256", 256, 24, 256),
]
const EXTRA_SHAPES = [
    ShapeSpec("1024x256x1024", 1024, 256, 1024),
]

function build_plain(::Type{T}, spec::ShapeSpec, rng) where {T}
    Amat = randn(rng, T, spec.Ma, spec.Ka)
    Bmat = randn(rng, T, spec.Ka, spec.Na)
    Cmat = zeros(T, spec.Ma, spec.Na)
    Av, Bv, Cv = StridedView(Amat), StridedView(Bmat), StridedView(Cmat)
    return (
        Av = Av, indA = (1, 2), Bv = Bv, indB = (2, 3), Cv = Cv, indC = (1, 3),
        Amat = Amat, Bmat = Bmat, Cmat = Cmat,
    )
end

# 3-index / scattered-C fixture (permuted A, negative-stride B,
# sliced-with-offset C), sized up from test/test_macro_driver.jl's version.
function build_scattered(::Type{T}, rng) where {T}
    a_n, k_n, b_n, n_n = 64, 64, 16, 64
    A2 = randn(rng, T, a_n, k_n)
    Araw = StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0)
    Aperm = permutedims(Araw, (2, 3, 1))  # k,b,a order
    indA = (2, 3, 1)
    Bdata = randn(rng, T, k_n * n_n)
    Bneg = StridedView(Bdata, (k_n, n_n), (-1, k_n), k_n - 1)
    indB = (2, 4)
    Cbig = zeros(T, a_n + 2, n_n + 3, b_n + 1)
    Csub = view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n)
    Cv = StridedView(Csub)
    indC = (1, 4, 3)
    return (Av = Aperm, indA = indA, Bv = Bneg, indB = indB, Cv = Cv, indC = indC)
end

# (mc,kc,nc) grids, 36 combos each. Float32's is Float64's scaled ~1.5x:
# it packs more per cache line, so it can profitably use larger blocks.
function full_grid(mcs, kcs, ncs)
    combos = Tuple{Int, Int, Int}[]
    for kc in kcs, mc in mcs, nc in ncs
        push!(combos, (mc, kc, nc))
    end
    return combos
end

const GRID_F64 = full_grid((64, 128, 256, 512), (128, 256, 512), (768, 1536, 3072))
const GRID_F32 = full_grid((96, 192, 384, 768), (192, 384, 768), (1152, 2304, 4608))

grid_for(::Type{Float64}) = GRID_F64
grid_for(::Type{Float32}) = GRID_F32

const KERNEL_CTORS = (ScalarKernel = ScalarKernel, SIMDKernel = SIMDKernel)
const DTYPES = (Float64, Float32)

# Header.
function print_header(io::IO)
    println(io, "# QuasiStrided.jl benchmark/bench_driver.jl")
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "nthreads = ", NTHREADS, "  blas_threads = ", BLAS_THREADS)
    for T in DTYPES
        sk = ScalarKernel(Val(8), Val(6), T)
        simdk = SIMDKernel(Val(8), Val(6), T)
        println(
            io, "kernel(", T, "): MR=", mr(sk), " NR=", nr(sk),
            " W(SIMD)=", lanewidth(simdk)
        )
    end
    return println(io, "date = ", now())
end
print_header(stdout)

# Output location.
const OUTDIR = joinpath(
    @__DIR__, "results", "$(gethostname())-$(Dates.format(now(), "yyyy-mm-dd"))"
)
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "bench.csv")
const CANARY_PATH = joinpath(OUTDIR, "canary.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary.txt")
const PROVENANCE_PATH = joinpath(OUTDIR, "PROVENANCE.txt")

csv_io = open(CSV_PATH, "w")
println(
    csv_io,
    "kernel,dtype,shape,Ma,Ka,Na,method,mc,kc,nc,reps,median_seconds"
)

function log_row(kernel_name, T, shapename, Ma, Ka, Na, method, mc, kc, nc, reps, t)
    println(
        csv_io,
        "$kernel_name,$T,$shapename,$Ma,$Ka,$Na,$method,$mc,$kc,$nc,$reps,",
        @sprintf("%.9f", t)
    )
    return flush(csv_io)
end

# Canary: a small, fixed case run at the start / middle / end of the sweep
# (A, B, A' pattern) to catch drift (thermal throttling, background load).
const CANARY_SHAPE = ShapeSpec("canary_64^3", 64, 64, 64)
const CANARY_COMBO = (128, 256, 1536)

function run_canary(rng, label::String)
    fx = build_plain(Float64, CANARY_SHAPE, rng)
    kernel = SIMDKernel(Val(8), Val(6), Float64)
    mc, kc, nc = CANARY_COMBO
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = kernel, mc = mc, kc = kc, nc = nc
    )
    t = median_time_s(() -> execute!(plan, 1.0, 0.0); reps = 15)
    println("canary[$label] median = $(t) s")
    return t
end

# Main grid sweep over the small/medium shapes.
rng = MersenneTwister(0xB3_C4_0001)

raw_execute = Vector{NamedTuple}()  # for ranking

canary_results = Float64[]
push!(canary_results, run_canary(rng, "A (start)"))

for (kname, kctor) in pairs(KERNEL_CTORS)
    for T in DTYPES
        kernel = kctor(Val(8), Val(6), T)
        combos = grid_for(T)
        for (mc, kc, nc) in combos
            for spec in MAIN_SHAPES
                fx = build_plain(T, spec, rng)
                plan = plan_contract(
                    fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
                    kernel = kernel, mc = mc, kc = kc, nc = nc
                )
                t = median_time_s(() -> execute!(plan, 1.0, 0.0); reps = 9)
                log_row(
                    kname, T, spec.name, spec.Ma, spec.Ka, spec.Na,
                    "execute!", mc, kc, nc, 9, t
                )
                push!(
                    raw_execute,
                    (
                        kernel = String(kname), dtype = T, shape = spec.name,
                        mc = mc, kc = kc, nc = nc, t = t,
                    )
                )
            end
            @info "grid progress" kernel = kname dtype = T combo = (mc, kc, nc)
        end

        # execute_tilewise!, once per (kernel,dtype,shape) at the center
        # combo -- the old driver has no macro blocking to sweep, so
        # repeating it at every grid point would be pure waste.
        center = combos[cld(length(combos), 2)]
        mc, kc, nc = center
        for spec in MAIN_SHAPES
            fx = build_plain(T, spec, rng)
            plan = plan_contract(
                fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
                kernel = kernel, mc = mc, kc = kc, nc = nc
            )
            t = median_time_s(() -> execute_tilewise!(plan, 1.0, 0.0); reps = 9)
            log_row(
                kname, T, spec.name, spec.Ma, spec.Ka, spec.Na,
                "execute_tilewise!", mc, kc, nc, 9, t
            )
        end
    end
end

push!(canary_results, run_canary(rng, "B (middle)"))

# mul! reference line (once per dtype/shape -- it doesn't depend on
# kernel/blocking at all).
for T in DTYPES
    for spec in MAIN_SHAPES
        fx = build_plain(T, spec, rng)
        t = median_time_s(() -> mul!(fx.Cmat, fx.Amat, fx.Bmat); reps = 9)
        log_row("mul!", T, spec.name, spec.Ma, spec.Ka, spec.Na, "mul!", 0, 0, 0, 9, t)
    end
end

# Extra shapes (large cuboid + scattered-C fixture): only at each dtype's
# center combo, both kernels, to check the grid-derived winner generalizes
# without paying for the full grid at these more expensive shapes.
for (kname, kctor) in pairs(KERNEL_CTORS)
    for T in DTYPES
        kernel = kctor(Val(8), Val(6), T)
        mc, kc, nc = grid_for(T)[cld(length(grid_for(T)), 2)]

        for spec in EXTRA_SHAPES
            fx = build_plain(T, spec, rng)
            plan = plan_contract(
                fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
                kernel = kernel, mc = mc, kc = kc, nc = nc
            )
            t_exec = median_time_s(() -> execute!(plan, 1.0, 0.0); reps = 9)
            log_row(
                kname, T, spec.name, spec.Ma, spec.Ka, spec.Na,
                "execute!", mc, kc, nc, 9, t_exec
            )
            plan_tw = plan_contract(
                fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
                kernel = kernel, mc = mc, kc = kc, nc = nc
            )
            t_tw = median_time_s(() -> execute_tilewise!(plan_tw, 1.0, 0.0); reps = 9)
            log_row(
                kname, T, spec.name, spec.Ma, spec.Ka, spec.Na,
                "execute_tilewise!", mc, kc, nc, 9, t_tw
            )
            if kname == :ScalarKernel  # mul! doesn't depend on kernel; log once
                t_mul = median_time_s(() -> mul!(fx.Cmat, fx.Amat, fx.Bmat); reps = 9)
                log_row("mul!", T, spec.name, spec.Ma, spec.Ka, spec.Na, "mul!", 0, 0, 0, 9, t_mul)
            end
        end

        # Scattered-C fixture: no plain-matmul mul! reference (it's a
        # genuinely 3-index case; a 2-D mul! isn't a fair comparison point).
        fx = build_scattered(T, rng)
        plan = plan_contract(
            fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
            kernel = kernel, mc = mc, kc = kc, nc = nc
        )
        t_exec = median_time_s(() -> execute!(plan, 1.0, 0.0); reps = 9)
        log_row(kname, T, "scattered_a64k64b16n64", 64, 64, 64, "execute!", mc, kc, nc, 9, t_exec)
        plan_tw = plan_contract(
            fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
            kernel = kernel, mc = mc, kc = kc, nc = nc
        )
        t_tw = median_time_s(() -> execute_tilewise!(plan_tw, 1.0, 0.0); reps = 9)
        log_row(
            kname, T, "scattered_a64k64b16n64", 64, 64, 64,
            "execute_tilewise!", mc, kc, nc, 9, t_tw
        )
    end
end

push!(canary_results, run_canary(rng, "A' (end)"))
close(csv_io)

# Canary spread report.
canary_spread = (maximum(canary_results) - minimum(canary_results)) / minimum(canary_results)
open(CANARY_PATH, "w") do io
    println(io, "label,median_seconds")
    for (lbl, t) in zip(("A_start", "B_middle", "Aprime_end"), canary_results)
        println(io, "$lbl,", @sprintf("%.9f", t))
    end
    println(io, "# relative spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
end
println("canary spread (max-min)/min = ", @sprintf("%.4f", canary_spread))

# Ranking: geomean ratio of execute! time across all (kernel,shape) pairs
# in the main grid, per dtype, at each (mc,kc,nc) combo. Each shape/kernel
# time is first normalized by that (kernel,dtype,shape)'s own minimum
# across the combo grid, so shapes of very different absolute cost weigh
# equally in the geomean.
function rank_combos(raw, T::DataType)
    rows = filter(r -> r.dtype == T, raw)
    # minimum time per (kernel,shape) across all combos
    mins = Dict{Tuple{String, String}, Float64}()
    for r in rows
        key = (r.kernel, r.shape)
        mins[key] = min(get(mins, key, Inf), r.t)
    end
    combo_ratios = Dict{Tuple{Int, Int, Int}, Vector{Float64}}()
    for r in rows
        key = (r.mc, r.kc, r.nc)
        push!(get!(combo_ratios, key, Float64[]), r.t / mins[(r.kernel, r.shape)])
    end
    geo(v) = exp(sum(log, v) / length(v))
    ranked = sort(
        [(combo, geo(ratios)) for (combo, ratios) in combo_ratios];
        by = x -> x[2]
    )
    return ranked
end

open(SUMMARY_PATH, "w") do io
    println(io, "# Phase E ranking summary")
    println(io, "canary median times (s): ", canary_results)
    println(io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", canary_spread))
    chosen = Dict{DataType, Tuple{Int, Int, Int}}()
    for T in DTYPES
        ranked = rank_combos(raw_execute, T)
        println(io, "\n== $T ranking (combo => geomean ratio, best first) ==")
        for (combo, g) in ranked
            println(io, "  $combo => ", @sprintf("%.4f", g))
        end
        best_combo, best_g = ranked[1]
        best_size = prod(best_combo)
        # Prefer the smallest-footprint combo within 6% of the best geomean
        # (this project's own noise-floor convention: sub-6% differences
        # are treated as noise, not a real ranking signal).
        within_noise = filter(x -> x[2] <= best_g * 1.06, ranked)
        winner = first(sort(within_noise; by = x -> prod(x[1])))
        chosen[T] = winner[1]
        println(
            io,
            "chosen $T default: mc,kc,nc = ", winner[1],
            " (geomean ratio ", @sprintf("%.4f", winner[2]), "; unconstrained best was ",
            best_combo, " at ", @sprintf("%.4f", best_g), ")"
        )
        runner_up = length(ranked) > 1 ? ranked[2] : nothing
        println(io, "runner-up: ", runner_up)
    end
    println(io, "\nchosen defaults: ", chosen)
    global CHOSEN = chosen
end
println(read(SUMMARY_PATH, String))

# Provenance.
commit = try
    strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse HEAD`, String))
catch
    "unknown (git rev-parse failed)"
end
open(PROVENANCE_PATH, "w") do io
    println(io, "git_commit = ", commit)
    println(io, "command = julia --project=. benchmark/bench_driver.jl")
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "nthreads = ", NTHREADS, " blas_threads = ", BLAS_THREADS)
    println(io, "date = ", now())
    println(io, "grid_f64 = ", GRID_F64)
    println(io, "grid_f32 = ", GRID_F32)
    println(io, "main_shapes = ", [s.name for s in MAIN_SHAPES])
    println(io, "extra_shapes = ", [s.name for s in EXTRA_SHAPES], " + scattered_a64k64b16n64")
    println(io, "chosen_defaults = ", CHOSEN)
end

println("\nDone. Results in ", OUTDIR)
