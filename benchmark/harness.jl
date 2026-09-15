# Shared measurement harness, factored out of bench_driver.jl: single-threaded,
# warm-up-then-median timing, a start/middle/end canary bracket, and geomean
# ranking normalized per (kernel, shape). `include`d, not a module.

using QuasiStrided
using QuasiStrided: ScalarKernel, SIMDKernel, mr, nr, lanewidth, plan_contract, execute!
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

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

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

# Small free extents: a larger MR/NR pads more of every micro-tile away, and
# small bond dimensions are the tensor-network common case. Reported apart.
const SMALL_SHAPES = [
    ShapeSpec("smallN_256x256x12", 256, 256, 12),
    ShapeSpec("smallM_12x256x256", 12, 256, 256),
    ShapeSpec("smallMN_16x256x16", 16, 256, 16),
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

function full_grid(mcs, kcs, ncs)
    combos = Tuple{Int, Int, Int}[]
    for kc in kcs, mc in mcs, nc in ncs
        push!(combos, (mc, kc, nc))
    end
    return combos
end

const DTYPES = (Float64, Float32)

# Complex dtypes are a SEPARATE constant, deliberately. `DTYPES` is what
# `bench_driver.jl`, `bench_kernel_shape.jl` and `bench_axis_group.jl` sweep,
# and every committed real measurement was taken over it; widening it in place
# would silently change what those scripts mean and make the real baseline
# incomparable with its own history.
const CDTYPES = (ComplexF64, ComplexF32)
const ALL_DTYPES = (DTYPES..., CDTYPES...)

"""
    flops_per_mac(::Type{T}) -> Int

Real floating-point operations per multiply-accumulate: 2 for a real type, 8
for a complex one.

**8 is the textbook count and is deliberately not reduced for induced
methods**: charging 1m or 3m its own lower multiply count would flatter its
throughput and make a method comparison meaningless.
"""
flops_per_mac(::Type{T}) where {T} = T <: Complex ? 8 : 2

"""
    gflops(::Type{T}, Ma, Ka, Na, seconds) -> Float64

Throughput of an `Ma x Ka x Na` contraction in GFLOP/s, charging
[`flops_per_mac`](@ref) per multiply-accumulate.
"""
gflops(::Type{T}, Ma::Int, Ka::Int, Na::Int, seconds::Float64) where {T} =
    flops_per_mac(T) * Ma * Ka * Na / seconds / 1.0e9

"""
    panel_reals_per_element(kernel) -> Int

Reals a *packed panel* holds per complex element, summed over both operands:
2 + 2 = 4 for planar, 4 + 2 = 6 for 1m, so 1m/planar = **1.5x**. A property of
the *formats alone* -- independent of blocking, shape, or machine -- and
therefore the number to quote when comparing methods.

**Not** the same quantity as [`packed_bytes_per_flop`](@ref); see there.
"""
panel_reals_per_element(kernel) =
    (QuasiStrided.packed_a_per_k(kernel) ÷ QuasiStrided.mr(kernel)) +
    (QuasiStrided.packed_b_per_k(kernel) ÷ QuasiStrided.nr(kernel))

"""
    packed_bytes_per_flop(kernel, blocking) -> Float64

Packed-panel bytes streamed per useful real flop **for one macro block at this
method's own shipped blocking**. Reported alongside GFLOP/s because the complex
methods differ mainly in bytes moved per useful flop, not in flop count -- 3m
does 25% fewer FMAs and still loses -- which a flops column alone cannot see.

**Do not read this as the 1.5x format figure**; use
[`panel_reals_per_element`](@ref) for that. This one uses each method's *own*
`mc` (which `default_blocking` halves for 1m to hold the L2 byte budget equal),
and at the shipped `nc` the B term dominates, so it comes out near 2x and is
insensitive to exactly the A-side difference that distinguishes the methods.
Conflating the two would credit a method for its blocking rather than its
format (docs/decisions.md, "Harness defects fixed en route").
"""
function packed_bytes_per_flop(kernel, blocking)
    T = QuasiStrided.scalartype(kernel)
    R = QuasiStrided.realtype(kernel)
    mc, kc, nc = blocking.mc, blocking.kc, blocking.nc
    a_reals = QuasiStrided.packed_a_per_k(kernel) * kc * cld(mc, QuasiStrided.mr(kernel))
    b_reals = QuasiStrided.packed_b_per_k(kernel) * kc * cld(nc, QuasiStrided.nr(kernel))
    bytes = (a_reals + b_reals) * sizeof(R)
    flops = flops_per_mac(T) * mc * kc * nc
    return bytes / flops
end

"""
    complex_efficiency(gf_complex, gf_real) -> Float64

The milestone's headline metric: one engine's complex throughput divided by its
own real throughput at the same shape, with complex charged 8 flops/MAC.

`1.0` means complex is treated exactly as well as real, and it should exceed 1
-- complex is 4x the flops on 2x the bytes, i.e. twice the arithmetic
intensity, so packing and per-call overheads amortise *better*. Below ~0.9
indicates a structural overhead specific to complex (a packing cost or an
accumulator spill) and is a finding, not a result to publish. Measured values:
docs/decisions.md, "The headline metric".
"""
complex_efficiency(gf_complex::Float64, gf_real::Float64) = gf_complex / gf_real

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

geomean(v) = exp(sum(log, v) / length(v))

# Rank rows for dtype `T` by geomean of time normalized per (kernel, shape) by
# that pair's own minimum, so shapes of different absolute cost weigh equally.
function rank_by(raw, T::DataType, keyfn)
    rows = filter(r -> r.dtype == T, raw)
    mins = Dict{Tuple{String, String}, Float64}()
    for r in rows
        key = (r.kernel, r.shape)
        mins[key] = min(get(mins, key, Inf), r.t)
    end
    ratios = Dict{Any, Vector{Float64}}()
    for r in rows
        push!(get!(ratios, keyfn(r), Float64[]), r.t / mins[(r.kernel, r.shape)])
    end
    return sort([(k, geomean(v)) for (k, v) in ratios]; by = x -> x[2])
end

# Of the points within `tol` of the best geomean (this project's 6% noise-floor
# convention), the one with the smallest `footprint(key)`.
function choose_within_noise(ranked, footprint; tol::Float64 = 0.06)
    best_g = ranked[1][2]
    within = filter(x -> x[2] <= best_g * (1 + tol), ranked)
    return first(sort(within; by = x -> footprint(x[1])))
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

results_dir() = joinpath(
    @__DIR__, "results", "$(gethostname())-$(Dates.format(now(), "yyyy-mm-dd"))"
)

function git_commit()
    return try
        strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse HEAD`, String))
    catch
        "unknown (git rev-parse failed)"
    end
end

function print_env_header(io::IO, script::String)
    println(io, "# QuasiStrided.jl benchmark/", script)
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "nthreads = ", NTHREADS, "  blas_threads = ", BLAS_THREADS)
    return println(io, "date = ", now())
end

# Fixed case timed at the start/middle/end of a sweep to catch drift. 15 reps:
# a 7-rep trial once showed 43% spread on this ~25 us shape, a
# timer-resolution artefact (docs/decisions.md, Phase E).
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

relative_spread(ts) = isempty(ts) ? 0.0 : (maximum(ts) - minimum(ts)) / minimum(ts)
