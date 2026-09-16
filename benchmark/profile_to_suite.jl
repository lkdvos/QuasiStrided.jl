# Profiling tool for triaging QuasiStridedBackend() vs TensorOperations'
# StridedBLAS() on individual tensor-contraction cases drawn from the upstream
# `TensorOperationsBenchmarks` suite's `:pairwise`/`:tccg` categories.
#
# This is a STANDALONE profiling script, not a benchmark: it does not depend
# on `benchmark/bench_to_suite.jl` at build/run time. Instead it profiles a
# small, hand-edited `CASES` list defined below. The 4 cases currently listed
# are the T5 triage set, chosen from `bench_to_suite.jl`'s (T3's) three-way
# run results (see benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/
# summary_to_suite.txt): the worst substantive QuasiStrided/BLAS loss
# (tccg/ccsd_t_1_dim16), the best QuasiStrided win (tccg/ao2mo_2_dim16), the
# known small-shape overhead pattern (pairwise/dim15_2_2_2), and a Float32
# repeat of the worst-loss shape to separate dtype- from shape-sensitivity.
# Every IA/IB/IC/dims triple was read out of the upstream generators
# themselves, not hand-written -- see the CASES comments below.
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
# for StridedBLAS, higher for QuasiStridedBackend) -- confirmed present identically in
# `Profile.print`'s own `.flat.txt` output, not an artifact of this script's bucket
# parsing. It does not affect the sanity check that matters (a nonzero `blas` bucket for
# StridedBLAS): that bucket is computed only from LEAF/self-time samples classified by
# substring match, and idle-thread futex samples never match any bucket's substrings, so
# they fall into "other" rather than stealing from `blas`/`microkernel`/etc.
#
# For each case, both `QuasiStridedBackend()` and `TensorOperations.StridedBLAS()`
# execute the identical `tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1, 0, backend)`
# call, where `pA, pB, pAB = TensorOperations.contract_indices(IA, IB, IC)` -- the exact
# same helper `TensorOperationsBenchmarks/src/lowering.jl`'s `maketensors(::ContractSpec, ...)`
# uses to build its own `pA`/`pB`/`pAB`, so this script's contraction-index convention is
# guaranteed to match `bench_to_suite.jl`'s cases without needing to reimplement the
# label-set arithmetic (`intersect`/`setdiff` over `IA`/`IB`/`IC`) by hand. (Every case
# below is a contraction, so QuasiStridedBackend()'s tensoradd!/tensortrace! fallback --
# docs/decisions.md, "Amendment 7" -- is never exercised here.)

using Profile
using Printf
using Random
using Dates
using TensorOperations
using TensorOperations: StridedBLAS
using QuasiStrided: QuasiStridedBackend

include(joinpath(@__DIR__, "profile_buckets.jl"))

# ---------------------------------------------------------------------------
# CASES -- the T5 triage set (see header comment above).
# ---------------------------------------------------------------------------

const CASES = [
    # The 4 cases below are the T5 triage set, chosen from T3's three-way run
    # (benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/
    # summary_to_suite.txt). Every IA/IB/IC/dims triple was read out of the
    # upstream generators themselves --
    #   TensorOperationsBenchmarks._tccg_cases((16,)) / ._pairwise_cases((15,))
    # then `case.spec.IA` / `.IB` / `.IC` / `.dims` -- not hand-written, so the
    # shapes here are byte-identical to the ones bench_to_suite.jl timed.
    #
    # 1. tccg/ccsd_t_1_dim16 -- worst substantive throughput loss in T3:
    #    QS/BLAS = 10.110, Native/QS = 0.280 (the only class where QuasiStrided
    #    loses to StridedNative at a non-trivial absolute size).
    #    C[a,b,c,i,j,k] = A[i,j,m,a] * B[m,k,b,c]; single contracted index m.
    (
        id = "ccsd_t_1_dim16",
        IA = [:i, :j, :m, :a], IB = [:m, :k, :b, :c],
        IC = [:a, :b, :c, :i, :j, :k],
        dims = Dict(
            :i => 16, :j => 16, :m => 16, :a => 16, :k => 16, :b => 16, :c => 16
        ),
        dtype = Float64,
    ),
    # 2. tccg/ao2mo_2_dim16 -- best QuasiStrided win in T3: QS/BLAS = 0.315.
    #    C[a,b,r,s] = A[q,b] * B[a,q,r,s]; single contracted index q.
    (
        id = "ao2mo_2_dim16",
        IA = [:q, :b], IB = [:a, :q, :r, :s], IC = [:a, :b, :r, :s],
        dims = Dict(:q => 16, :b => 16, :a => 16, :r => 16, :s => 16),
        dtype = Float64,
    ),
    # 3. pairwise/dim15_2_2_2 -- the known small-shape overhead pattern
    #    (QS/BLAS = 1.710 Float64), GEMM-like with rank-4 operands.
    #    C[a1,a2,b1,b2] = A[a1,a2,c1,c2] * B[c1,c2,b1,b2].
    (
        id = "dim15_2_2_2",
        IA = [:a1, :a2, :c1, :c2], IB = [:c1, :c2, :b1, :b2],
        IC = [:a1, :a2, :b1, :b2],
        dims = Dict(:a1 => 15, :a2 => 15, :c1 => 15, :c2 => 15, :b1 => 15, :b2 => 15),
        dtype = Float64,
    ),
    # 4. tccg/ccsd_t_1_dim16 again, at Float32 (QS/BLAS = 11.509 in T3) --
    #    same shape as case 1, so the pair isolates dtype-sensitivity from
    #    shape-sensitivity.
    (
        id = "ccsd_t_1_dim16_f32",
        IA = [:i, :j, :m, :a], IB = [:m, :k, :b, :c],
        IC = [:a, :b, :c, :i, :j, :k],
        dims = Dict(
            :i => 16, :j => 16, :m => 16, :a => 16, :k => 16, :b => 16, :c => 16
        ),
        dtype = Float32,
    ),
]

const BACKENDS = [
    ("QuasiStridedBackend", QuasiStridedBackend()),
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
