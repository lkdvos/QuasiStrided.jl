# Empirical measurement for `execute_half_packed!` (src/execution/halfpack.jl):
# B packed as usual, A read in place when the whole M extent is exactly one
# full SIMDKernel register tile (`Qm == mr(kernel)`) -- versus the fully
# packed `execute!`.
#
#   julia --project=. benchmark/bench_half_packing.jl
#
# Point: profiling (see benchmark/profile_buckets.jl and the investigation
# behind this file) found packing to be a genuine 30-38% of `execute!`'s own
# time in the 8-32-element range, most of it attributable to a redundant
# copy-then-immediately-reread round trip through a freshly packed A panel
# when A already has the layout the microkernel could read directly. This
# script sweeps M fixed at exactly `mr(kernel)` (the only shape
# `execute_half_packed!` takes the in-place path for) across a range of N/K
# from small to large, to see where -- if anywhere -- skipping A's packing
# pays off and by how much, and whether that shrinks or grows with N/K.

# This script's first sweep (below) only ever measured the simplest eligible
# case: a plain, contiguous, dense M-by-K/K-by-N matrix pair. Two follow-up
# sweeps extend that:
#
#   - "non-ramp K": A's own rows are still unit-stride (eligible), but the K
#     composite as a WHOLE (both operands' maps) is not a single affine ramp,
#     so `_axis_of` hands the in-place A read a `PtrScatterAxis` -- a per-K-step
#     indexed lookup instead of a compile-time stride multiply. This is the
#     realistic case for a genuine multi-label tensor contraction (as opposed
#     to a plain two-index GEMM), and it was previously only correctness-
#     tested (test/execution/test_halfpack.jl), never benchmarked.
#   - "alternate kernel shape": the first sweep only ever measured M pinned to
#     the DEFAULT kernel's `mr` on this machine. This sweeps a second,
#     non-default menu shape too.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: execute_half_packed!, _default_kernel, kernel_shapes, _kernel_from_shape,
    affine_ramp

const REPS_TINY = 51
const REPS_MAIN = 21

# (K, N) pairs at M = mr(kernel) exactly. Doubling ladder from tiny K/N (where
# the packing-avoidance premise should matter most, relative to a near-zero
# baseline cost) up through shapes large enough that B's own packing/compute
# dominates and any A-side saving should vanish into noise.
const HALF_PACKING_KN = [
    (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (63, 63),
    (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024),
    (2048, 2048), (4096, 512), (512, 4096),
]

reps_for(K::Int, N::Int) = (K * N <= 64 * 64) ? REPS_TINY : REPS_MAIN

function time_packed(::Type{T}, fx; reps::Int) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute!(plan, one(T), zero(T))
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
    return t
end

function time_half_packed(::Type{T}, fx; reps::Int) where {T}
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    execute_half_packed!(plan, one(T), zero(T))
    t = median_time_s(() -> execute_half_packed!(plan, one(T), zero(T)); reps = reps)
    return (t, plan)
end

# ---------------------------------------------------------------------------
# "Non-ramp K" fixture: A[m,k1,k2] dense (m fastest, then k1, then k2) so A's
# OWN map is a clean ramp; B[k2,k1,n] dense (k2 fastest) so B's map, over the
# SAME (k1,k2) label order, is not -- stride(k2 in B) = 1 is far smaller than
# stride(k1 in B) = k2n, the opposite of what folding k1,k2 into one ramp
# needs. `affine_ramp` requires EVERY map to fold, so the whole K composite
# (shared by both operands) classifies as non-ramp, even though A's own
# addresses alone would trivially ramp -- exactly the generic multi-label
# case, and the in-place A read pays for it (indexed `PtrScatterAxis` lookups,
# not a compile-time stride multiply) regardless of which operand "caused" it.
function build_nonramp_k(::Type{T}, M::Int, k1n::Int, k2n::Int, N::Int, rng) where {T}
    Amat = randn(rng, T, M, k1n, k2n)
    Bmat = randn(rng, T, k2n, k1n, N)
    Cmat = zeros(T, M, N)
    Av = StridedView(Amat)
    Bv = StridedView(Bmat)
    Cv = StridedView(Cmat)
    # indA labels: m=1, k1=2, k2=3. indB labels B's own physical axis order
    # (k2, k1, n) = (3, 2, 4). indC: (m, n) = (1, 4).
    return (Av = Av, indA = (1, 2, 3), Bv = Bv, indB = (3, 2, 4), Cv = Cv, indC = (1, 4))
end

const NONRAMP_K1 = 4  # fixed; k2 scales to reach each target K = k1n*k2n below
const NONRAMP_TOTAL_K_N = [
    (4, 1), (8, 2), (16, 8), (32, 32), (64, 64), (128, 128),
    (256, 256), (512, 512), (1024, 1024),
]

function run_nonramp_sweep(csv, ::Type{T}, rng) where {T}
    default_kernel = _default_kernel(T)
    M = mr(default_kernel)
    for (K, N) in NONRAMP_TOTAL_K_N
        k1n = NONRAMP_K1
        k2n = cld(K, k1n)
        fx = build_nonramp_k(T, M, k1n, k2n, N, rng)
        plan_check = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
        if affine_ramp(plan_check.kgroup)[1]
            @warn "non-ramp K fixture unexpectedly ramps; skipping" T K N
            continue
        end
        reps = reps_for(k1n * k2n, N)

        t_packed = time_packed(T, fx; reps = reps)
        (t_half, plan) = time_half_packed(T, fx; reps = reps)
        gf_packed = gflops(T, M, k1n * k2n, N, t_packed)
        gf_half = gflops(T, M, k1n * k2n, N, t_half)
        speedup = t_packed / t_half

        println(
            csv,
            "$T,$M,$(k1n * k2n),$N,$reps,$t_packed,$t_half,$gf_packed,$gf_half,$speedup,nonramp_k"
        )
        @printf(
            "[nonramp-K] %-10s M=%-4d K=%-5d(=%dx%d) N=%-5d  packed %10.3e s (%6.2f GF/s)  half %10.3e s (%6.2f GF/s)  speedup %6.3fx\n",
            string(T), M, k1n * k2n, k1n, k2n, N, t_packed, gf_packed, t_half, gf_half, speedup
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Alternate (non-default) kernel shapes: the first and non-ramp sweeps above
# only ever pin M to mr() of the DEFAULT kernel this machine's ISA resolves
# to. Sweep every OTHER menu shape too (kernel_shapes(T), real method), at a
# fixed mid-range (K, N), to check the win isn't an artifact of one specific
# register shape.
const ALT_SHAPE_K, ALT_SHAPE_N = 128, 128

function run_alt_shape_sweep(csv, ::Type{T}, rng) where {T}
    default_kernel = _default_kernel(T)
    for shape in kernel_shapes(T)
        kernel = _kernel_from_shape(shape, T)
        M = mr(kernel)
        M == mr(default_kernel) && shape == (mr(default_kernel), nr(default_kernel), lanewidth(default_kernel)) && continue
        spec = ShapeSpec("alt_$(M)x$(ALT_SHAPE_K)x$(ALT_SHAPE_N)", M, ALT_SHAPE_K, ALT_SHAPE_N)
        fx = build_plain(T, spec, rng)
        reps = REPS_MAIN

        plan_p = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC; kernel = kernel)
        execute!(plan_p, one(T), zero(T))
        t_packed = median_time_s(() -> execute!(plan_p, one(T), zero(T)); reps = reps)
        plan_h = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC; kernel = kernel)
        execute_half_packed!(plan_h, one(T), zero(T))
        t_half = median_time_s(() -> execute_half_packed!(plan_h, one(T), zero(T)); reps = reps)

        gf_packed = gflops(T, M, ALT_SHAPE_K, ALT_SHAPE_N, t_packed)
        gf_half = gflops(T, M, ALT_SHAPE_K, ALT_SHAPE_N, t_half)
        speedup = t_packed / t_half

        println(
            csv,
            "$T,$M,$ALT_SHAPE_K,$ALT_SHAPE_N,$reps,$t_packed,$t_half,$gf_packed,$gf_half,$speedup,alt_shape_$(shape[1])x$(shape[2])W$(shape[3])"
        )
        @printf(
            "[alt-shape] %-10s kernel=%dx%d/W%d  packed %10.3e s (%6.2f GF/s)  half %10.3e s (%6.2f GF/s)  speedup %6.3fx\n",
            string(T), shape[1], shape[2], shape[3], t_packed, gf_packed, t_half, gf_half, speedup
        )
    end
    return nothing
end

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "half_packing.csv")
const PROV_PATH = joinpath(OUTDIR, "half_packing_PROVENANCE.txt")

function main()
    print_env_header(stdout, "bench_half_packing.jl")
    csv = open(CSV_PATH, "w")
    println(csv, "dtype,M,K,N,reps,t_packed,t_half_packed,gf_packed,gf_half_packed,speedup_half_over_packed,fixture")

    canaries = Float64[]
    rng = Random.MersenneTwister(0x4A1FBACC)
    push!(canaries, run_canary(rng, "start"))

    for T in DTYPES  # Float64, Float32 -- execute_half_packed! is real-only by design
        # M is fixed at exactly mr(kernel) for the DEFAULT kernel this dtype
        # resolves to on this machine -- the only shape the in-place path takes.
        default_kernel = _default_kernel(T)
        M = mr(default_kernel)
        for (K, N) in HALF_PACKING_KN
            spec = ShapeSpec("$(M)x$(K)x$(N)", M, K, N)
            fx = build_plain(T, spec, rng)
            reps = reps_for(K, N)

            t_packed = time_packed(T, fx; reps = reps)
            (t_half, plan) = time_half_packed(T, fx; reps = reps)
            gf_packed = gflops(T, M, K, N, t_packed)
            gf_half = gflops(T, M, K, N, t_half)
            speedup = t_packed / t_half

            println(
                csv,
                "$T,$M,$K,$N,$reps,$t_packed,$t_half,$gf_packed,$gf_half,$speedup,plain"
            )
            @printf(
                "%-10s M=%-4d (mr) K=%-5d N=%-5d  packed %10.3e s (%6.2f GF/s)  half %10.3e s (%6.2f GF/s)  half/packed speedup %6.3fx  kernel=%dx%d/W%d\n",
                string(T), M, K, N, t_packed, gf_packed, t_half, gf_half, speedup,
                mr(plan.kernel), nr(plan.kernel), lanewidth(plan.kernel)
            )
        end

        run_nonramp_sweep(csv, T, rng)
        run_alt_shape_sweep(csv, T, rng)
    end

    push!(canaries, run_canary(rng, "end"))
    close(csv)

    spread = relative_spread(canaries)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    if spread > 0.1
        @warn "canary spread exceeds 10%: the machine was not quiet enough for " *
            "a few-percent conclusion. Re-run before believing any of these ratios."
    end

    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_half_packing.jl")
        println(io, "git commit = ", git_commit())
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
        println(
            io,
            "\nM is fixed per dtype at mr(default_kernel(T)) on THIS machine -- see the",
        )
        println(
            io,
            "printed kernel= shape per row. speedup_half_over_packed > 1 means the",
        )
        println(io, "in-place-A path won; read half_packing.csv for the full sweep.")
    end
    println("\nwrote ", CSV_PATH)
    return nothing
end

main()
