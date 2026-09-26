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
# sliced-with-offset C), sized up from test/execution/test_macro_blocking.jl's version.
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

# More scattered/gathered fixtures (first used by the software-prefetch
# experiment on branch `prefetch`): each forces the packers off their
# contiguous fast paths in a different way, so
# the gather loops -- not the microkernel -- carry a real share of the time.
# `build_scattered(T, rng)` above is unchanged and is the `:default` entry.
# Every entry returns the same fields as `build_scattered`, plus `M`/`K`/`N`
# (the free and contracted extents, for GFLOP/s).
const SCATTER_VARIANTS = (
    :default, :negstride_big, :transposed_AB, :interleaved_groups, :tn4,
)

function build_scattered(::Type{T}, rng, variant::Symbol) where {T}
    if variant === :default
        # M = a*b = 64*16, K = 64, N = 64 (see `build_scattered` above).
        return (; build_scattered(T, rng)..., M = 64 * 16, K = 64, N = 64)
    elseif variant === :negstride_big
        # Large negative strides on both operands: A[m,k] with M stride -2
        # (so A's unit-stride fast path is ineligible), B[k,n] with K stride
        # -3 and N stride -6K. C plain.
        M, K, N = 192, 192, 192
        Abig = randn(rng, T, 2M, K)
        Bbig = randn(rng, T, 3K, 2N)
        Av = StridedView(view(Abig, (2M):-2:1, :))
        Bv = StridedView(view(Bbig, (3K):-3:1, (2N):-2:1))
        Cv = StridedView(zeros(T, M, N))
        return (Av = Av, indA = (1, 2), Bv = Bv, indB = (2, 3), Cv = Cv, indC = (1, 3), M = M, K = K, N = N)
    elseif variant === :transposed_AB
        # Both operands transposed views: A's M axis has stride K and B's K
        # axis has stride N, i.e. every packed lane of A is its own cache line
        # and B's K steps are N elements apart. Still one plain GEMM.
        M, K, N = 256, 256, 256
        Av = permutedims(StridedView(randn(rng, T, K, M)), (2, 1))
        Bv = permutedims(StridedView(randn(rng, T, N, K)), (2, 1))
        Cv = StridedView(zeros(T, M, N))
        return (Av = Av, indA = (1, 2), Bv = Bv, indB = (2, 3), Cv = Cv, indC = (1, 3), M = M, K = K, N = N)
    elseif variant === :interleaved_groups
        # Six small axes per operand with the M, K and N legs interleaved in
        # memory, so no group merges into one affine axis: every M/K/N group
        # is a genuine 3-axis composite whose offsets come in runs of 6, and
        # the engine takes the scattered (`PtrScatterAxis`) path everywhere.
        d = 6
        # labels: a1..a3 = 1..3 (M), k1..k3 = 4..6 (K), n1..n3 = 7..9 (N)
        Av = StridedView(randn(rng, T, d, d, d, d, d, d))   # (a1,k1,a2,k2,a3,k3)
        Bv = StridedView(randn(rng, T, d, d, d, d, d, d))   # (k1,n1,k2,n2,k3,n3)
        Cv = StridedView(zeros(T, d, d, d, d, d, d))        # (a1,n1,a2,n2,a3,n3)
        return (
            Av = Av, indA = (1, 4, 2, 5, 3, 6), Bv = Bv, indB = (4, 7, 5, 8, 6, 9),
            Cv = Cv, indC = (1, 7, 2, 8, 3, 9), M = d^3, K = d^3, N = d^3,
        )
    elseif variant === :tn4
        # A tensor-network-style 4-index contraction
        # C[a,b,c,d] = A[a,k,b,l] * B[l,c,k,d]: A's M legs (a,b) and K legs
        # (k,l) interleave, and B's K legs arrive in the opposite order.
        e = 16
        Av = StridedView(randn(rng, T, e, e, e, e))   # (a,k,b,l)
        Bv = StridedView(randn(rng, T, e, e, e, e))   # (l,c,k,d)
        Cv = StridedView(zeros(T, e, e, e, e))        # (a,b,c,d)
        # labels: a=1 b=2 c=3 d=4 k=5 l=6
        return (
            Av = Av, indA = (1, 5, 2, 6), Bv = Bv, indB = (6, 3, 5, 4),
            Cv = Cv, indC = (1, 2, 3, 4), M = e^2, K = e^2, N = e^2,
        )
    end
    throw(ArgumentError("unknown scattered variant $(repr(variant)); expected one of $SCATTER_VARIANTS"))
end

# ---------------------------------------------------------------------------
# DRAM-resident and irregular fixtures, for packing work where memory latency
# matters. Same fields as `build_scattered(T, rng, variant)`.
# Sizes are for Float64; a Float32 operand is half the bytes.
# ---------------------------------------------------------------------------

# Operands of 128-256 MB each (Float64): well beyond any node's L3, so the
# packers stream from DRAM. The small-M/small-N ones make packing a large share
# of the time (each packed element feeds only 16 microkernel rows/columns).
const LARGE_VARIANTS = (
    :smallM_plain, :smallM_negB, :smallN_transA, :square_plain, :square_scattered,
)

function build_large(::Type{T}, rng, variant::Symbol) where {T}
    plain(M, K, N) = (
        Av = StridedView(randn(rng, T, M, K)), indA = (1, 2),
        Bv = StridedView(randn(rng, T, K, N)), indB = (2, 3),
        Cv = StridedView(zeros(T, M, N)), indC = (1, 3), M = M, K = K, N = N,
    )
    if variant === :smallM_plain
        # B = 4096 x 8192 (256 MB), plain: B packs through its gather loop
        # anyway (B has no contiguous fast path), A is tiny.
        return plain(16, 4096, 8192)
    elseif variant === :smallM_negB
        # The same B, reversed along both axes (negative K and N strides).
        M, K, N = 16, 4096, 8192
        B = randn(rng, T, K, N)
        return (
            Av = StridedView(randn(rng, T, M, K)), indA = (1, 2),
            Bv = StridedView(view(B, K:-1:1, N:-1:1)), indB = (2, 3),
            Cv = StridedView(zeros(T, M, N)), indC = (1, 3), M = M, K = K, N = N,
        )
    elseif variant === :smallN_transA
        # A = 8192 x 4096 stored transposed (256 MB): A's M axis has stride K,
        # so A packs through its gather loop from DRAM.
        M, K, N = 8192, 4096, 16
        return (
            Av = permutedims(StridedView(randn(rng, T, K, M)), (2, 1)), indA = (1, 2),
            Bv = StridedView(randn(rng, T, K, N)), indB = (2, 3),
            Cv = StridedView(zeros(T, M, N)), indC = (1, 3), M = M, K = K, N = N,
        )
    elseif variant === :square_plain
        return plain(4096, 4096, 4096)   # 128 MB per operand
    elseif variant === :square_scattered
        # 4096^3 with A transposed and B reversed: both gathers, from DRAM.
        n = 4096
        B = randn(rng, T, n, n)
        return (
            Av = permutedims(StridedView(randn(rng, T, n, n)), (2, 1)), indA = (1, 2),
            Bv = StridedView(view(B, n:-1:1, :)), indB = (2, 3),
            Cv = StridedView(zeros(T, n, n)), indC = (1, 3), M = n, K = n, N = n,
        )
    end
    throw(ArgumentError("unknown large variant $(repr(variant)); expected one of $LARGE_VARIANTS"))
end

# As irregular as the engine can express. It has NO arbitrary-offset input:
# every operand is a `StridedView` (sizes + strides + offset) and every
# `AxisGroup` is a stride table, so a random-permutation index vector cannot be
# passed in. The closest it gets: each free/contracted group is many short
# axes with large, pseudo-random (odd, non-power-of-two) strides, so a group's
# offset table -- which the engine then gathers through, `PtrScatterAxis` --
# is a sum of unrelated jumps with no locality beyond each short axis. Inputs
# may alias (a read-only operand is allowed to repeat elements), which is what
# lets the strides be chosen freely inside a buffer of a few hundred MB.
const IRREGULAR_VARIANTS = (:hypercube_A, :hypercube_AB)

# `n` pseudo-random odd strides in `[lo, hi]` from `rng`.
_random_strides(rng, n, lo, hi) = Tuple(rand(rng, lo:hi) | 1 for _ in 1:n)

function _hypercube_view(rng, ::Type{T}, dims::NTuple{D, Int}, strides::NTuple{D, Int}) where {T, D}
    len = 1 + sum((dims[d] - 1) * strides[d] for d in 1:D)
    return StridedView(randn(rng, T, len), dims, strides, 0)
end

function build_irregular(::Type{T}, rng, variant::Symbol) where {T}
    # M = K = 4^6 = 4096 (six length-4 axes each), N = 64.
    L, nax, N = 4, 6, 64
    M = K = L^nax
    # labels: M axes 1..6, K axes 7..12, N axis 13
    mlab = ntuple(identity, nax)
    klab = ntuple(d -> nax + d, nax)
    nlab = 2nax + 1
    # A: M and K axes interleaved, strides up to ~2^21 elements (16 MB):
    # buffer ~ 3 * 12 * 2^20 elements (~300 MB Float64).
    sA = _random_strides(rng, 2nax, 1 << 18, 1 << 21)
    Av = _hypercube_view(rng, T, ntuple(_ -> L, 2nax), sA)
    indA = ntuple(d -> isodd(d) ? mlab[(d + 1) >> 1] : klab[d >> 1], 2nax)
    if variant === :hypercube_A
        # B plain: K axes as a dense (4,...,4,64) column-major array.
        Bv = StridedView(randn(rng, T, ntuple(_ -> L, nax)..., N))
    elseif variant === :hypercube_AB
        # B irregular too: its six K axes and N get their own random strides.
        sB = _random_strides(rng, nax + 1, 1 << 16, 1 << 20)
        Bv = _hypercube_view(rng, T, (ntuple(_ -> L, nax)..., N), sB)
    else
        throw(ArgumentError("unknown irregular variant $(repr(variant)); expected one of $IRREGULAR_VARIANTS"))
    end
    indB = (klab..., nlab)
    Cv = StridedView(zeros(T, ntuple(_ -> L, nax)..., N))
    indC = (mlab..., nlab)
    return (Av = Av, indA = indA, Bv = Bv, indB = indB, Cv = Cv, indC = indC, M = M, K = K, N = N)
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
format.
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

The headline complex metric: one engine's complex throughput divided by its
own real throughput at the same shape, with complex charged 8 flops/MAC.

`1.0` means complex is treated exactly as well as real, and it should exceed 1
-- complex is 4x the flops on 2x the bytes, i.e. twice the arithmetic
intensity, so packing and per-call overheads amortise *better*. Below ~0.9
indicates a structural overhead specific to complex (a packing cost or an
accumulator spill) and is a finding, not a result to publish.
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
    root = joinpath(@__DIR__, "..")
    return try
        sha = strip(read(`git -C $root rev-parse HEAD`, String))
        dirty = !isempty(strip(read(`git -C $root status --porcelain`, String)))
        dirty ? sha * "-dirty" : sha
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
# timer-resolution artefact.
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

# ---------------------------------------------------------------------------
# Minimal CLI arg parsing: `--name value` or `--name=value`.
# ---------------------------------------------------------------------------

function argval(name::String)
    for (i, a) in enumerate(ARGS)
        if a == "--$name" && i < length(ARGS)
            return ARGS[i + 1]
        elseif startswith(a, "--$name=")
            return split(a, '='; limit = 2)[2]
        end
    end
    return nothing
end

hasflag(name::String) = "--$name" in ARGS

argopt(name::String, default::AbstractString) = something(argval(name), default)
argopt(name::String, default::Integer) = something(tryparse(Int, something(argval(name), "")), default)

const DTYPE_BY_NAME = Dict(
    "Float64" => Float64, "Float32" => Float32,
    "ComplexF64" => ComplexF64, "ComplexF32" => ComplexF32,
)
parse_dtypes(s::AbstractString) = Tuple(DTYPE_BY_NAME[strip(t)] for t in split(s, ','))
parse_ints(s::AbstractString) = Tuple(parse(Int, strip(t)) for t in split(s, ','))
