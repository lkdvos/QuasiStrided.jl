# OpenBLAS reference: `BLAS.gemm!` (the OpenBLAS_jll that LinearAlgebra loads,
# pinned to ONE thread by harness.jl) next to QuasiStrided at the same shapes.
#
#   julia --project=. benchmark/bench_vs_openblas.jl
#
# Per shape and dtype, timed back to back: OpenBLAS gemm!, QuasiStrided's
# default path, and the best menu shape of each complex method (planar, 1m,
# fmaddsub) at its own default blocking. Real dtypes are included so the
# complex gap can be read against the real gap on the same machine. Same
# timing convention as the suite: warm-up + median of REPS, canary bracket,
# spread must be < 10% before any few-percent conclusion.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: PlanarMethod, OneMMethod, FMAddSubMethod, kernel_shapes,
    _kernel_from_shape, default_blocking

const REPS = 21
const SHAPES = vcat(MAIN_SHAPES, EXTRA_SHAPES, SMALL_SHAPES)
const OUTDIR = get(ENV, "QS_RESULTS_DIR", results_dir())
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "vs_openblas.csv")
const PROV_PATH = joinpath(OUTDIR, "vs_openblas_PROVENANCE.txt")
const METHODS = (PlanarMethod(), OneMMethod(), FMAddSubMethod())

tag(k) = "$(mr(k))x$(nr(k))/W$(lanewidth(k))"

function time_blas(fx, ::Type{T}) where {T}
    A, B, C = fx.Amat, fx.Bmat, fx.Cmat
    return median_time_s(() -> BLAS.gemm!('N', 'N', one(T), A, B, zero(T), C); reps = REPS)
end

function time_qs(fx, ::Type{T}, kernel) where {T}
    plan = if kernel === nothing
        plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    else
        b = default_blocking(kernel)
        plan_contract(
            fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
            kernel = kernel, mc = b.mc, kc = b.kc, nc = b.nc
        )
    end
    execute!(plan, one(T), zero(T))
    return median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS)
end

function main()
    print_env_header(stdout, "bench_vs_openblas.jl")
    blascfg = string(BLAS.get_config())
    corename = try
        lib = BLAS.get_config().loaded_libs[1].libname
        f = Libc.Libdl.dlsym(Libc.Libdl.dlopen(lib), :openblas_get_corename64_)
        unsafe_string(ccall(f, Cstring, ()))
    catch
        "?"
    end
    println("BLAS = ", blascfg, "  core = ", corename, "  blas_threads = ", BLAS.get_num_threads())
    csv = open(CSV_PATH, "w")
    println(csv, "dtype,shape,Ma,Ka,Na,engine,kernel,t,gflops,frac_of_openblas,reps")
    canaries = Float64[]
    crng = Random.MersenneTwister(0x0CA9A121)
    push!(canaries, run_canary(crng, "start"))
    summary = Dict{Tuple{DataType, String}, Vector{Float64}}()
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        println("\n== $T ==")
        for spec in SHAPES
            fx = build_plain(T, spec, Random.MersenneTwister(0x5EED))
            tb = time_blas(fx, T)
            gb = gflops(T, spec.Ma, spec.Ka, spec.Na, tb)
            # correctness guard: QS must agree with BLAS on this fixture
            ref = fx.Amat * fx.Bmat
            rows = Tuple{String, String, Float64}[("openblas", corename, tb)]
            td = time_qs(fx, T, nothing)
            @assert isapprox(fx.Cmat, ref; rtol = sqrt(eps(real(T))))
            push!(rows, ("qs_default", "-", td))
            if T <: Complex
                for m in METHODS
                    best = (Inf, "")
                    for shape in kernel_shapes(T, m)
                        k = try
                            _kernel_from_shape(shape, T, m)
                        catch
                            continue
                        end
                        t = time_qs(fx, T, k)
                        t < best[1] && (best = (t, tag(k)))
                    end
                    isfinite(best[1]) && push!(rows, ("qs_best_" * string(m), best[2], best[1]))
                end
            end
            line = @sprintf("%-22s OpenBLAS %7.2f GF/s |", spec.name, gb)
            for (eng, kt, t) in rows
                g = gflops(T, spec.Ma, spec.Ka, spec.Na, t)
                println(csv, "$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),$eng,$kt,$t,$g,$(tb / t),$REPS")
                push!(get!(summary, (T, eng), Float64[]), tb / t)
                eng == "openblas" || (line *= @sprintf(" %s %.2f (%.0f%%)", eng, g, 100tb / t))
            end
            println(line)
        end
        push!(canaries, run_canary(crng, "after-$T"))
    end
    push!(canaries, run_canary(crng, "end"))
    close(csv)
    spread = relative_spread(canaries)
    println("\ngeomean fraction of OpenBLAS throughput (1.0 = parity):")
    for ((T, eng), v) in sort(collect(summary); by = x -> (string(x[1][1]), x[1][2]))
        eng == "openblas" && continue
        @printf("  %-11s %-22s %.3f\n", T, eng, geomean(v))
    end
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    spread > 0.1 && @warn "canary spread > 10%: not quiet enough for a few-percent conclusion"
    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_vs_openblas.jl")
        println(io, "git commit = ", git_commit())
        println(io, "BLAS = ", blascfg, "  core = ", corename)
        println(io, "reps = ", REPS)
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
    end
    println("\nwrote ", CSV_PATH)
    return nothing
end

main()
