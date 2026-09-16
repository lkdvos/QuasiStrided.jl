# Profiles QuasiStridedBackend() vs StridedBLAS() on tensor-contraction
# cases drawn from the upstream TensorOperationsBenchmarks suite, bucketing
# sampled cost into adapter/planning/packing/microkernel/store/blas/etc.
#
#   julia --project=benchmark benchmark/profile_to_suite.jl [caseid ...]
#
# With no case ids, profiles every case in CASES below; edit that list to
# add/replace cases. Writes flat/tree profiles, an optional ProfileCanvas
# HTML flamegraph, and a bucket table per (case, backend) to
# benchmark/results/<hostname>-<date>/profiles/.
#
# CAVEAT: with `C=true`, Julia's profiler also samples idle helper threads
# (GC/IO, sitting in `__futex_abstimed_wait_common`), inflating "other" --
# this does not steal samples from named buckets (leaf/self-time only), so
# named-bucket percentages stay reliable even when "other" is large.

using Profile
using TensorOperations
using TensorOperations: StridedBLAS
using QuasiStrided: QuasiStridedBackend

include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "profile_buckets.jl"))

const CASES = [
    # C[a,b,c,i,j,k] = A[i,j,m,a] * B[m,k,b,c] -- 6-index output, 1 contracted index.
    (
        id = "ccsd_t_1_dim16",
        IA = [:i, :j, :m, :a], IB = [:m, :k, :b, :c],
        IC = [:a, :b, :c, :i, :j, :k],
        dims = Dict(
            :i => 16, :j => 16, :m => 16, :a => 16, :k => 16, :b => 16, :c => 16
        ),
        dtype = Float64,
    ),
    # C[a,b,r,s] = A[q,b] * B[a,q,r,s].
    (
        id = "ao2mo_2_dim16",
        IA = [:q, :b], IB = [:a, :q, :r, :s], IC = [:a, :b, :r, :s],
        dims = Dict(:q => 16, :b => 16, :a => 16, :r => 16, :s => 16),
        dtype = Float64,
    ),
    # C[a1,a2,b1,b2] = A[a1,a2,c1,c2] * B[c1,c2,b1,b2] -- GEMM-like, rank-4.
    (
        id = "dim15_2_2_2",
        IA = [:a1, :a2, :c1, :c2], IB = [:c1, :c2, :b1, :b2],
        IC = [:a1, :a2, :b1, :b2],
        dims = Dict(:a1 => 15, :a2 => 15, :c1 => 15, :c2 => 15, :b1 => 15, :b2 => 15),
        dtype = Float64,
    ),
    # Same shape as ccsd_t_1_dim16, at Float32.
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

const SELECTED_IDS = filter(a -> !startswith(a, "--"), ARGS)
const SELECTED_CASES = isempty(SELECTED_IDS) ? CASES : filter(c -> c.id in SELECTED_IDS, CASES)

const BACKENDS = [
    ("QuasiStridedBackend", QuasiStridedBackend()),
    ("StridedBLAS", StridedBLAS()),
]

const RESULTS_ROOT = results_dir()
const PROFILE_DIR = joinpath(RESULTS_ROOT, "profiles")
mkpath(PROFILE_DIR)

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

const _HAVE_PROFILECANVAS = try
    @eval import ProfileCanvas
    true
catch err
    @warn "ProfileCanvas not available; skipping HTML flamegraphs" exception = err
    false
end

function try_profilecanvas_html(path)
    _HAVE_PROFILECANVAS || return false
    try
        ProfileCanvas.html_file(path)
        return true
    catch err
        @warn "ProfileCanvas.html_file failed" path exception = err
        return false
    end
end

"""
    profile_one!(io_summary, case, backendname, backend)

Warm up, profile >=2s of repeated `tensorcontract!` calls under `backend`,
write flat/tree profiles, an `@allocated` figure, an optional ProfileCanvas
flamegraph, and a bucket table (also appended to `io_summary`).
"""
function profile_one!(io_summary, case, backendname, backend)
    rng = Random.Xoshiro(0x5eed_5eed)
    A, B, C, pA, pB, pAB = build_case_tensors(case, rng)
    T = case.dtype
    α, β = one(T), zero(T)

    call!() = tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, backend)

    call!()  # warm up
    call!()
    allocated_bytes = @allocated call!()

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

    # `include_meta = false` strips threadid/taskid entries, so every
    # remaining value is a genuine instruction pointer.
    data = Profile.fetch(include_meta = false)
    total_samples = count(iszero, data)

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

function main()
    summary_path = joinpath(PROFILE_DIR, "buckets_summary.txt")
    results = []
    open(summary_path, "w") do io_summary
        println(io_summary, "# Bucketed cost-attribution summary")
        println(io_summary, "# generated $(Dates.now()) on $(gethostname())")
        println(io_summary)
        for case in SELECTED_CASES
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
