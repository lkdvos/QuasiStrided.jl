# Head-to-head timing of `@tensor` under `StridedNative()`, `StridedBLAS()`
# and `QuasiStridedBackend()` on identical inputs, over the same shape grid as
# `benchmark/bench_driver.jl`. An honest comparison, not a search for a win:
# see docs/decisions.md, "Not hooked into `select_backend`", which these
# numbers are the evidence base for.
#
#   julia --project=. benchmark/bench_tensoroperations.jl
#
# writes results + PROVENANCE.txt to benchmark/results/<hostname>-<date>/.

using TensorOperations
using TensorOperations: StridedNative, StridedBLAS
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
using LinearAlgebra
using Statistics: median
using Random
using Dates
using Printf

# Single-core measurement discipline (this project's standing rule; see
# benchmark/bench_driver.jl).
LinearAlgebra.BLAS.set_num_threads(1)
const NTHREADS = Threads.nthreads()
const BLAS_THREADS = LinearAlgebra.BLAS.get_num_threads()
if NTHREADS != 1
    @warn "Threads.nthreads() = $NTHREADS != 1 -- this is NOT the pinned " *
        "single-core measurement this project's rules require. Results " *
        "below should not be trusted as the reference-machine numbers."
end

# Warm up once (discarded), then `reps` timed calls; median, not mean.
# Identical convention to benchmark/bench_driver.jl's `median_time_s`.
function median_time_s(f!::Function; reps::Int = 9)
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

# Same shape grid as benchmark/bench_driver.jl's MAIN_SHAPES + EXTRA_SHAPES
# (matrix-shaped contraction C[m,n] = A[m,k]*B[k,n]), reproduced here verbatim
# rather than included: bench_driver.jl is not meant to be used as a library.
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
const ALL_SHAPES = vcat(MAIN_SHAPES, EXTRA_SHAPES)

const DTYPES = (Float64, Float32)

const BACKENDS = (
    StridedNative = StridedNative(),
    StridedBLAS = StridedBLAS(),
    QuasiStrided = QuasiStridedBackend(),
)

function build(::Type{T}, spec::ShapeSpec, rng) where {T}
    A = randn(rng, T, spec.Ma, spec.Ka)
    B = randn(rng, T, spec.Ka, spec.Na)
    C = zeros(T, spec.Ma, spec.Na)
    return A, B, C
end

# One in-place contraction call under a given backend: `C = A*B` (beta = 0),
# identical expression for all three backends -- only `backend` varies.
function run_contract!(backend, C, A, B)
    @tensor backend = backend C[i, j] = A[i, k] * B[k, j]
    return C
end

# Output location.
const OUTDIR = joinpath(
    @__DIR__, "results", "$(gethostname())-$(Dates.format(now(), "yyyy-mm-dd"))"
)
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "bench_tensoroperations.csv")
const CANARY_PATH = joinpath(OUTDIR, "canary_tensoroperations.csv")
const PROVENANCE_PATH = joinpath(OUTDIR, "PROVENANCE.txt")

csv_io = open(CSV_PATH, "w")
println(csv_io, "backend,dtype,shape,Ma,Ka,Na,reps,median_seconds")
function log_row(backend_name, T, spec::ShapeSpec, reps, t)
    println(
        csv_io,
        "$backend_name,$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),$reps,",
        @sprintf("%.9f", t)
    )
    return flush(csv_io)
end

println("# QuasiStrided.jl benchmark/bench_tensoroperations.jl")
println("cpu = ", Sys.CPU_NAME)
println("julia = ", VERSION)
println("nthreads = ", NTHREADS, "  blas_threads = ", BLAS_THREADS)
println("date = ", now())

# Canary: a small, fixed case run at the start / middle / end of the sweep
# (A, B, A' pattern), StridedBLAS on a 64^3 shape -- same spirit as
# bench_driver.jl's canary (drift detection: thermal throttling, background
# load), scoped to a single backend since this script's job is a
# cross-backend comparison, not a per-backend sweep.
const CANARY_SHAPE = ShapeSpec("canary_64^3", 64, 64, 64)
function run_canary(rng, label::String)
    A, B, C = build(Float64, CANARY_SHAPE, rng)
    t = median_time_s(() -> run_contract!(StridedBLAS(), C, A, B); reps = 15)
    println("canary[$label] median = $(t) s")
    return t
end

rng = MersenneTwister(0xB3_C4_0002)

canary_results = Float64[]
push!(canary_results, run_canary(rng, "A (start)"))

const REPS = 9
raw = Vector{NamedTuple}()

for T in DTYPES
    for spec in ALL_SHAPES
        # One shared pair of inputs per (dtype, shape): all three backends
        # see byte-identical A/B, and each gets its own freshly zeroed C
        # (beta = 0 in run_contract! means C's initial content is irrelevant
        # to the result, but a fresh buffer per backend avoids any
        # in-place-mutation surprises across backends sharing memory).
        A, B, _ = build(T, spec, rng)
        for (bname, backend) in pairs(BACKENDS)
            C = zeros(T, spec.Ma, spec.Na)
            t = median_time_s(() -> run_contract!(backend, C, A, B); reps = REPS)
            log_row(bname, T, spec, REPS, t)
            push!(raw, (backend = String(bname), dtype = T, shape = spec.name, t = t))
        end
        @info "grid progress" dtype = T shape = spec.name
    end
end

push!(canary_results, run_canary(rng, "B (middle)"))
push!(canary_results, run_canary(rng, "A' (end)"))
close(csv_io)

canary_spread = (maximum(canary_results) - minimum(canary_results)) / minimum(canary_results)
open(CANARY_PATH, "w") do io
    println(io, "label,median_seconds")
    for (lbl, t) in zip(("A_start", "B_middle", "Aprime_end"), canary_results)
        println(io, "$lbl,", @sprintf("%.9f", t))
    end
    println(io, "# relative spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
end
println("canary spread (max-min)/min = ", @sprintf("%.4f", canary_spread))

# Per-shape/dtype summary: ratio of each backend's time to the fastest backend
# at that point, with a QuasiStrided win reported exactly like a loss.
open(joinpath(OUTDIR, "summary_tensoroperations.txt"), "w") do io
    println(io, "# TensorOperations backend benchmark summary")
    println(io, "canary median times (s): ", canary_results)
    println(io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", canary_spread))
    for T in DTYPES
        println(io, "\n== ", T, " ==")
        for spec in ALL_SHAPES
            rows = filter(r -> r.dtype == T && r.shape == spec.name, raw)
            isempty(rows) && continue
            best = minimum(r.t for r in rows)
            println(io, "  ", spec.name, ":")
            for r in sort(collect(rows); by = r -> r.t)
                println(
                    io, "    ", rpad(r.backend, 14), @sprintf("%.6e s", r.t),
                    "  (", @sprintf("%.3fx", r.t / best), " of fastest)"
                )
            end
        end
    end
end
println(read(joinpath(OUTDIR, "summary_tensoroperations.txt"), String))

# Provenance -- format matched to
# benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/PROVENANCE.txt.
commit = try
    strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse HEAD`, String))
catch
    "unknown (git rev-parse failed)"
end
open(PROVENANCE_PATH, "w") do io
    println(io, "git_commit = ", commit)
    println(io, "command = julia --project=. benchmark/bench_tensoroperations.jl")
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "nthreads = ", NTHREADS, " blas_threads = ", BLAS_THREADS)
    println(io, "date = ", now())
    println(io, "backends = ", collect(keys(BACKENDS)))
    println(io, "dtypes = ", collect(DTYPES))
    println(io, "shapes = ", [s.name for s in ALL_SHAPES])
    println(io, "reps = ", REPS)
    println(io, "caveat = single machine ($(gethostname())), single measurement session; ")
    println(
        io,
        "  not averaged across machines or repeated sessions. Numbers are indicative of ",
        "this reference machine only, matching the caveat in the macro-blocking ",
        "milestone's benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/PROVENANCE.txt."
    )
end

println("\nDone. Results in ", OUTDIR)
