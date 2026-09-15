# Profiling tool for triaging QuasiStridedComposite() vs TensorOperations'
# StridedBLAS() on individual tensor-contraction cases drawn from the upstream
# `TensorOperationsBenchmarks` suite's `:pairwise`/`:tccg` categories.
#
# This is a STANDALONE profiling script, not a benchmark: it does not depend
# on `benchmark/bench_to_suite.jl` (a sibling task, concurrently under
# development) at all. Instead it profiles a small, hand-edited `CASES` list
# defined below. Once `bench_to_suite.jl` has produced real timing results,
# the coordinator/triage task should REPLACE the 4 example cases below with
# the specific cases found interesting there (e.g. worst QuasiStrided/BLAS
# slowdown, a near-parity case, etc.) -- the 4 shipped here are PLACEHOLDER
# EXAMPLES only, chosen to exercise a plain-matmul-like :pairwise shape, a
# multi-index :tccg shape (mirroring TCCG's `ccsd_2`), a larger plain-matmul
# shape, and one Float32 case.
#
# Usage:
#   julia --project=benchmark benchmark/profile_to_suite.jl
#
# Edit the `CASES` vector below to add/replace cases. Each entry is a
# `NamedTuple` with fields:
#   id::String            -- short identifier, used in output file names
#   IA::Vector{Symbol}     -- labels of tensor A, in A's axis order
#   IB::Vector{Symbol}     -- labels of tensor B, in B's axis order
#   IC::Vector{Symbol}     -- labels of the output C, in C's axis order
#   dims::Dict{Symbol,Int} -- extent of every label appearing in IA/IB/IC
#   dtype::Type            -- element type shared by A, B, C (Float64/Float32/...)
#
# KNOWN CAVEAT (do not "fix" by filtering -- it's real, not a parsing bug): with
# `C = true`, Julia's sampling profiler captures backtraces on every live OS thread
# each tick, including this process's persistently-idle helper thread(s) (GC/IO/etc)
# sitting in `__futex_abstimed_wait_common`. On this machine that shows up as a large,
# roughly workload-size-independent share of "other" in the bucket tables below (~50%
# for StridedBLAS, higher for QuasiStridedComposite) -- confirmed present identically in
# `Profile.print`'s own `.flat.txt` output, not an artifact of this script's bucket
# parsing. It does not affect the sanity check that matters (a nonzero `blas` bucket for
# StridedBLAS): that bucket is computed only from LEAF/self-time samples classified by
# substring match, and idle-thread futex samples never match any bucket's substrings, so
# they fall into "other" rather than stealing from `blas`/`microkernel`/etc.
#
# For each case, both `QuasiStridedComposite()` and `TensorOperations.StridedBLAS()`
# execute the identical `tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1, 0, backend)`
# call, where `pA, pB, pAB = TensorOperations.contract_indices(IA, IB, IC)` -- the exact
# same helper `TensorOperationsBenchmarks/src/lowering.jl`'s `maketensors(::ContractSpec, ...)`
# uses to build its own `pA`/`pB`/`pAB`, so this script's contraction-index convention is
# guaranteed to match both `benchmark/composite_backend.jl`'s worked examples and
# `bench_to_suite.jl`'s (T3's) cases without needing to reimplement the label-set
# arithmetic (`intersect`/`setdiff` over `IA`/`IB`/`IC`) by hand.

using Profile
using Printf
using Random
using Dates
using TensorOperations
using TensorOperations: StridedBLAS

include(joinpath(@__DIR__, "composite_backend.jl"))
include(joinpath(@__DIR__, "profile_buckets.jl"))

# ---------------------------------------------------------------------------
# CASES -- PLACEHOLDER EXAMPLES, replace once bench_to_suite.jl results exist.
# ---------------------------------------------------------------------------

const CASES = [
    # 1. Small :pairwise-style plain matmul: C[a1,b1] = A[a1,c1] * B[c1,b1].
    (
        id = "pairwise_small_63",
        IA = [:a1, :c1], IB = [:c1, :b1], IC = [:a1, :b1],
        dims = Dict(:a1 => 63, :b1 => 63, :c1 => 63),
        dtype = Float64,
    ),
    # 2. :tccg-style multi-index case, mirroring TCCG's `ccsd_2`
    #    (C[i,j] = A[i,k,l] * B[l,j,k]).
    (
        id = "tccg_ccsd2_16",
        IA = [:i, :k, :l], IB = [:l, :j, :k], IC = [:i, :j],
        dims = Dict(:i => 16, :j => 16, :k => 16, :l => 16),
        dtype = Float64,
    ),
    # 3. Larger :pairwise-style plain matmul.
    (
        id = "pairwise_large_256",
        IA = [:a1, :c1], IB = [:c1, :b1], IC = [:a1, :b1],
        dims = Dict(:a1 => 256, :b1 => 256, :c1 => 256),
        dtype = Float64,
    ),
    # 4. Float32 example (same shape family as case 1, single precision).
    (
        id = "pairwise_small_63_f32",
        IA = [:a1, :c1], IB = [:c1, :b1], IC = [:a1, :b1],
        dims = Dict(:a1 => 63, :b1 => 63, :c1 => 63),
        dtype = Float32,
    ),
]

const BACKENDS = [
    ("QuasiStridedComposite", QuasiStridedComposite()),
    ("StridedBLAS", StridedBLAS()),
]

# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------

const RESULTS_ROOT = joinpath(
    @__DIR__, "results", "$(gethostname())-$(Dates.format(Dates.today(), "yyyy-mm-dd"))"
)
const PROFILE_DIR = joinpath(RESULTS_ROOT, "profiles")
mkpath(PROFILE_DIR)

# ---------------------------------------------------------------------------
# Case -> tensors / call args
# ---------------------------------------------------------------------------

function build_case_tensors(case, rng)
    T = case.dtype
    dimsA = ntuple(i -> case.dims[case.IA[i]], length(case.IA))
    dimsB = ntuple(i -> case.dims[case.IB[i]], length(case.IB))
    dimsC = ntuple(i -> case.dims[case.IC[i]], length(case.IC))
    A = randn(rng, T, dimsA)
    B = randn(rng, T, dimsB)
    C = zeros(T, dimsC)
    pA, pB, pAB = TensorOperations.contract_indices(case.IA, case.IB, case.IC)
    return A, B, C, pA, pB, pAB
end

# ---------------------------------------------------------------------------
# ProfileCanvas: best-effort, optional.
# ---------------------------------------------------------------------------

const _HAVE_PROFILECANVAS = try
    @eval import ProfileCanvas
    true
catch err
    @warn "ProfileCanvas not available/loadable; skipping HTML flamegraphs" exception = err
    false
end

function try_profilecanvas_html(path)
    if !_HAVE_PROFILECANVAS
        return false
    end
    try
        ProfileCanvas.html_file(path)
        return true
    catch err
        @warn "ProfileCanvas.html_file failed; skipping" path exception = err
        return false
    end
end

# ---------------------------------------------------------------------------
# Profiling one (case, backend) pair
# ---------------------------------------------------------------------------

"""
    profile_one!(io_summary, case, backendname, backend)

Warm up, profile >=2s of repeated `tensorcontract!` calls for `case` under
`backend`, write flat/tree profile text, an `@allocated` figure, an optional
ProfileCanvas HTML flamegraph, and a per-case/backend bucket table -- and
append that bucket table into the shared `io_summary` stream.
"""
function profile_one!(io_summary, case, backendname, backend)
    rng = Random.Xoshiro(0x5eed_5eed)
    A, B, C, pA, pB, pAB = build_case_tensors(case, rng)
    T = case.dtype
    α, β = one(T), zero(T)

    call!() = tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, backend)

    # Warm-up (JIT).
    call!()

    # Representative fresh-call allocation figure (separate from the profiled loop).
    call!()  # extra warm-up in case of first-call effects on this exact signature
    allocated_bytes = @allocated call!()

    # Determine how many reps are needed to accumulate >= 2s of wall time.
    t0 = time_ns()
    call!()
    t1 = time_ns()
    per_call_s = max((t1 - t0) / 1.0e9, 1.0e-6)
    reps = max(1, ceil(Int, 2.0 / per_call_s))

    Profile.init(n = 10^7, delay = 1.0e-4)
    Profile.clear()
    Profile.@profile begin
        for _ in 1:reps
            call!()
        end
    end

    # `include_meta = false`: strip threadid/taskid/etc metadata entries so
    # every remaining value in `data` is a genuine instruction pointer (see
    # profile_buckets.jl's `compute_buckets` docstring for why this matters).
    data = Profile.fetch(include_meta = false)
    total_samples = count(iszero, data)  # number of per-sample backtraces (sentinel count)

    base = joinpath(PROFILE_DIR, "$(case.id)-$(backendname)")

    open(base * ".flat.txt", "w") do io
        Profile.print(io, format = :flat, sortedby = :count, C = true)
    end

    mincount = max(1, round(Int, 0.01 * total_samples))
    open(base * ".tree.txt", "w") do io
        Profile.print(io, format = :tree, C = true, mincount = mincount)
    end

    canvas_ok = try_profilecanvas_html(base * ".html")

    buckets = compute_buckets(data, backendname)

    open(base * ".buckets.txt", "w") do io
        print_bucket_table(io, case.id, backendname, buckets, total_samples, allocated_bytes, reps)
    end
    print_bucket_table(stdout, case.id, backendname, buckets, total_samples, allocated_bytes, reps)
    print_bucket_table(io_summary, case.id, backendname, buckets, total_samples, allocated_bytes, reps)

    return (
        case_id = case.id, backend = backendname, total_samples = total_samples,
        allocated_bytes = allocated_bytes, reps = reps, canvas_ok = canvas_ok,
        buckets = buckets,
    )
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function main()
    summary_path = joinpath(PROFILE_DIR, "buckets_summary.txt")
    results = []
    open(summary_path, "w") do io_summary
        println(io_summary, "# Bucketed cost-attribution summary")
        println(io_summary, "# generated $(Dates.now()) on $(gethostname())")
        println(io_summary)
        for case in CASES
            for (backendname, backend) in BACKENDS
                @info "Profiling" case = case.id backend = backendname
                r = profile_one!(io_summary, case, backendname, backend)
                push!(results, r)
            end
        end
    end
    @info "Done" summary_path profile_dir = PROFILE_DIR
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
