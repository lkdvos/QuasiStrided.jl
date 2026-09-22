# Block-size sweep.
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

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: execute_tilewise!

const GRID_F64 = full_grid((64, 128, 256, 512), (128, 256, 512), (768, 1536, 3072))
const GRID_F32 = full_grid((96, 192, 384, 768), (192, 384, 768), (1152, 2304, 4608))

grid_for(::Type{Float64}) = GRID_F64
grid_for(::Type{Float32}) = GRID_F32

# ScalarKernel stays at the historical (8,6) reference shape; SIMDKernel uses
# whatever `_default_kernel` derives for this machine, so this sweep validates
# mc/nc at the shape the engine actually ships.
kernels_for(::Type{T}) where {T} = (
    ScalarKernel = ScalarKernel(Val(8), Val(6), T),
    SIMDKernel = QuasiStrided._default_kernel(T),
)

function print_header(io::IO)
    print_env_header(io, "bench_driver.jl")
    println(io, "target = ", QuasiStrided.target_profile())
    for T in DTYPES
        for (kname, k) in pairs(kernels_for(T))
            println(
                io, "kernel(", T, ", ", kname, "): MR=", mr(k), " NR=", nr(k),
                k isa SIMDKernel ? "  W=" * string(lanewidth(k)) : ""
            )
        end
        println(io, "default_blocking(", T, ") = ", QuasiStrided.default_blocking(kernels_for(T).SIMDKernel))
    end
    return nothing
end
print_header(stdout)

# Output location.
const OUTDIR = results_dir()
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

for T in DTYPES
    for (kname, kernel) in pairs(kernels_for(T))
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
for T in DTYPES
    for (kname, kernel) in pairs(kernels_for(T))
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
rank_combos(raw, T::DataType) = rank_by(raw, T, r -> (r.mc, r.kc, r.nc))

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
commit = git_commit()
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
