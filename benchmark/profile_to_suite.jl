# Profiles QuasiStridedBackend() vs StridedBLAS() on tensor-contraction
# cases drawn from the upstream TensorOperationsBenchmarks suite plus this
# project's own benchmark/harness.jl shapes (plain square GEMM, small-skewed,
# and the permuted/negative-stride/sliced "scattered" fixture), bucketing
# sampled cost into adapter/planning/packing/microkernel/store/blas/etc.
#
# Extended 2026-09-21 ("profile-grid" pass, see docs/decisions.md's "T4"
# section) from the original 7 CASES + 1 DIRECT_CASES entry to the full
# MAIN_SHAPES/SMALL_SHAPES/EXTRA_SHAPES x 4-dtype grid (38 CASES + 5
# DIRECT_CASES, including two explicit OneMKernel entries) -- see the
# "Full-grid extension" comment below for the additive rules.
#
#   julia --project=benchmark benchmark/profile_to_suite.jl [caseid ...]
#
# With no case ids, profiles every case in CASES and DIRECT_CASES below; edit
# those lists to add/replace cases. CASES goes through the TensorOperations
# adapter, profiled under both QuasiStridedBackend() and StridedBLAS();
# DIRECT_CASES calls `plan_contract`/`execute!` directly (no adapter, no
# backend choice) for fixtures `harness.jl` builds directly rather than as
# plain labels/dims. Writes flat/tree profiles, an optional ProfileCanvas
# HTML flamegraph, and a bucket table per (case, backend) to
# benchmark/results/<hostname>-<date>/profiles/.
#
# CAVEAT: with `C=true`, Julia's profiler also samples idle helper threads
# (GC/IO, sitting in `__futex_abstimed_wait_common`). `profile_buckets.jl`'s
# `compute_buckets` classifies by walking each sample's whole backtrace (not
# just its leaf frame -- see that file's module docstring for why), so an
# idle thread's samples still land in "other" rather than stealing from a
# named bucket, but they are not filtered out by thread id, so a busy
# machine can inflate "other" here. Re-run on a quiet machine if "other" is
# large and unexplained.
#
# NOTE: each run OVERWRITES the previous one's artefacts for the same case
# (fixed filenames under `results_dir()`); pass `--tag <label>` to suffix
# every output file for this run instead, so repeated runs (e.g. for a
# reproducibility check) don't clobber each other.

using Profile
using TensorOperations
using TensorOperations: StridedBLAS
using QuasiStrided: QuasiStridedBackend, OneMKernel, kernel_shapes, OneMMethod

include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "profile_buckets.jl"))

# Look shapes up by `ShapeSpec.name` so the extended grid below can reference
# `harness.jl`'s MAIN_SHAPES/SMALL_SHAPES/EXTRA_SHAPES constants without
# hardcoding dims a second time (and without editing harness.jl itself).
const SHAPES_BY_NAME = Dict(
    s.name => s for s in vcat(MAIN_SHAPES, SMALL_SHAPES, EXTRA_SHAPES)
)

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
    # Same shape as ccsd_t_1_dim16, at ComplexF64 -- added 2026-09-22 to
    # investigate why QuasiStrided's ComplexF64/Float64 GFLOP/s ratio on
    # :tccg (bench_to_suite.jl, job 7087420) sits at ~0.29 median vs
    # StridedBLAS's ~0.68 (docs/decisions.md, "ComplexF64 tccg slowdown").
    (
        id = "ccsd_t_1_dim16_c64",
        IA = [:i, :j, :m, :a], IB = [:m, :k, :b, :c],
        IC = [:a, :b, :c, :i, :j, :k],
        dims = Dict(
            :i => 16, :j => 16, :m => 16, :a => 16, :k => 16, :b => 16, :c => 16
        ),
        dtype = ComplexF64,
    ),
    # TCCG's ccsd_6: C[i,j,k] = A[i,l,m,k] * B[m,j,l] -- 3-index output, 2
    # contracted indices. QS/BLAS GFLOP/s ratio in job 7087420's data: 36.02
    # (Float64) vs 10.46 (ComplexF64), ratio 0.29 -- squarely at the observed
    # median, added here as the other half of the same investigation.
    (
        id = "ccsd_6_dim16",
        IA = [:i, :l, :m, :k], IB = [:m, :j, :l], IC = [:i, :j, :k],
        dims = Dict(:i => 16, :l => 16, :m => 16, :k => 16, :j => 16),
        dtype = Float64,
    ),
    (
        id = "ccsd_6_dim16_c64",
        IA = [:i, :l, :m, :k], IB = [:m, :j, :l], IC = [:i, :j, :k],
        dims = Dict(:i => 16, :l => 16, :m => 16, :k => 16, :j => 16),
        dtype = ComplexF64,
    ),
    # C[m,n] = A[m,k] * B[k,n] -- plain square GEMM, large and compute-bound
    # by construction; the reference point for "what does the microkernel
    # share look like when there's nothing else to do."
    (
        id = "plain_256",
        IA = [:m, :k], IB = [:k, :n], IC = [:m, :n],
        dims = Dict(:m => 256, :k => 256, :n => 256),
        dtype = Float64,
    ),
    (
        id = "plain_512",
        IA = [:m, :k], IB = [:k, :n], IC = [:m, :n],
        dims = Dict(:m => 512, :k => 512, :n => 512),
        dtype = Float64,
    ),
    # Same GEMM shape, but N=12: STATUS.md's "Next task" flags this as the
    # regime where packing/per-call overhead, not the microkernel, dominates.
    (
        id = "smallN_256x256x12",
        IA = [:m, :k], IB = [:k, :n], IC = [:m, :n],
        dims = Dict(:m => 256, :k => 256, :n => 12),
        dtype = Float64,
    ),
]

# ---------------------------------------------------------------------------
# Full-grid extension (2026-09-21 "profile-grid" pass): every
# MAIN_SHAPES/SMALL_SHAPES/EXTRA_SHAPES shape from harness.jl, at both real
# dtypes and (MAIN_SHAPES union SMALL_SHAPES only) both complex dtypes.
# Additive only -- the seven case ids above are untouched; `_EXISTING_CASE_IDS`
# below guards against ever emitting a duplicate id for the three shapes
# (`plain_256`, `plain_512`, `smallN_256x256x12`) already covered at Float64.
# ---------------------------------------------------------------------------

# (ShapeSpec name in harness.jl, case id base for Float64 / `_f32` suffix).
const _REAL_GRID = [
    ("64^3", "plain_64"),
    ("128^3", "plain_128"),
    ("256^3", "plain_256"),
    ("512^3", "plain_512"),
    ("shallowK_256x24x256", "shallowK_256x24x256"),
    ("smallN_256x256x12", "smallN_256x256x12"),
    ("smallM_12x256x256", "smallM_12x256x256"),
    ("smallMN_16x256x16", "smallMN_16x256x16"),
    ("1024x256x1024", "big_1024x256x1024"),
]

function _plain_case(shapename::String, idbase::String, dtype::DataType)
    spec = SHAPES_BY_NAME[shapename]
    id = dtype == Float64 ? idbase : idbase * "_f32"
    return (
        id = id, IA = [:m, :k], IB = [:k, :n], IC = [:m, :n],
        dims = Dict(:m => spec.Ma, :k => spec.Ka, :n => spec.Na),
        dtype = dtype,
    )
end

const _EXISTING_CASE_IDS = Set(c.id for c in CASES)
for (shapename, idbase) in _REAL_GRID, dtype in DTYPES
    c = _plain_case(shapename, idbase, dtype)
    c.id in _EXISTING_CASE_IDS && continue
    push!(CASES, c)
    push!(_EXISTING_CASE_IDS, c.id)
end

# Complex dtypes: MAIN_SHAPES union SMALL_SHAPES only (no EXTRA_SHAPES --
# 1024x256x1024 at ComplexF64 is a >1 GB fixture and not needed for coverage
# here). These go through the default complex kernel the driver picks for
# a plain label/dims contraction -- the "planar" method
# (`_default_method` in src/planning/kernel_selection.jl always returns `PlanarMethod()`
# unless a kernel is explicitly named, which only the DIRECT_CASES 1m entries
# below do). `_c64`/`_c32` suffix distinguishes these from the real-dtype ids.
const _COMPLEX_GRID = [
    ("64^3", "plain_64"),
    ("128^3", "plain_128"),
    ("256^3", "plain_256"),
    ("512^3", "plain_512"),
    ("shallowK_256x24x256", "shallowK_256x24x256"),
    ("smallN_256x256x12", "smallN_256x256x12"),
    ("smallM_12x256x256", "smallM_12x256x256"),
    ("smallMN_16x256x16", "smallMN_16x256x16"),
]
const _COMPLEX_SUFFIX = Dict(ComplexF64 => "_c64", ComplexF32 => "_c32")

for (shapename, idbase) in _COMPLEX_GRID, dtype in CDTYPES
    spec = SHAPES_BY_NAME[shapename]
    push!(
        CASES, (
            id = idbase * _COMPLEX_SUFFIX[dtype], IA = [:m, :k], IB = [:k, :n],
            IC = [:m, :n], dims = Dict(:m => spec.Ma, :k => spec.Ka, :n => spec.Na),
            dtype = dtype,
        )
    )
end

# Cases that go through `plan_contract`/`execute!` directly (bypassing the
# TensorOperations adapter entirely), for fixtures that aren't expressible as
# a plain label/dims dict -- e.g. `harness.jl`'s permuted-A/negative-stride-B/
# sliced-C fixture. `builder(T, rng)` must return the `(Av, indA, Bv, indB,
# Cv, indC)` tuple `plan_contract` expects (see `benchmark/harness.jl`,
# `build_plain`/`build_scattered`). Profiled once, unbucketed by backend
# (there is no backend choice on this path), under the pseudo-backend name
# `"QuasiStridedDirect"` so `profile_buckets.jl`'s QuasiStrided bucket set
# applies (its `adapter/prepare`/`TO overhead` buckets simply read 0 here,
# since no TensorOperations frame is ever on the stack).
const DIRECT_CASES = Any[
    (
        id = "scattered_64", dtype = Float64,
        builder = (T, rng) -> build_scattered(T, rng),
        # F[a,b,n] = A[a,b,k] * B[k,n] in `harness.jl`'s naming (a_n=64,
        # k_n=64, b_n=16, n_n=64): flops_per_mac(Float64) * a_n*b_n*n_n
        # (output) * k_n (contracted).
        flops = flops_per_mac(Float64) * 64 * 16 * 64 * 64,
    ),
]

# 1m (`OneMKernel`) is reachable ONLY by explicitly naming the kernel to
# `plan_contract` (src/planning/kernel_selection.jl: `_default_method` always returns
# `PlanarMethod()`), so the label/dims `CASES` above -- which all go through
# the default kernel -- never exercise it. Exercise it here via
# `plan_contract(...; kernel = ...)` on a couple of MAIN_SHAPES, both complex
# dtypes, at 1m's own shipped default register shape (mirrors
# `bench_complex_efficiency.jl`'s `time_kernel`, which does the same thing
# through the `kernel_shapes`/`_kernel_from_shape` menu).
_onem_default_kernel(::Type{T}) where {T} = ((MR, NR, W) = kernel_shapes(T, OneMMethod())[end]; OneMKernel(Val(MR), Val(NR), T, Val(W)))

const _ONEM_GRID = [("256^3", "onem_256"), ("512^3", "onem_512")]
for (shapename, idbase) in _ONEM_GRID, dtype in CDTYPES
    spec = SHAPES_BY_NAME[shapename]
    push!(
        DIRECT_CASES, (
            id = idbase * _COMPLEX_SUFFIX[dtype], dtype = dtype,
            builder = (T, rng) -> build_plain(T, spec, rng),
            flops = flops_per_mac(dtype) * spec.Ma * spec.Ka * spec.Na,
            kernel = _onem_default_kernel(dtype),
        )
    )
end

function _argopt(name::String, default::String)
    pfx = "--$(name)="
    for (i, a) in enumerate(ARGS)
        startswith(a, pfx) && return a[(length(pfx) + 1):end]
        if a == "--$(name)" && i < length(ARGS)
            return ARGS[i + 1]
        end
    end
    return default
end

const TAG = _argopt("tag", "")
_tagged(base::String) = isempty(TAG) ? base : base * "-" * TAG

const SELECTED_IDS = filter(
    a -> !startswith(a, "--") && !(a == TAG && !isempty(TAG)), ARGS
)
const SELECTED_CASES = isempty(SELECTED_IDS) ? CASES : filter(c -> c.id in SELECTED_IDS, CASES)
const SELECTED_DIRECT_CASES = isempty(SELECTED_IDS) ? DIRECT_CASES : filter(c -> c.id in SELECTED_IDS, DIRECT_CASES)

# 2 * (product of C's extents) * (product of the contracted extents) --
# the standard GEMM-equivalent flop count for one tensor contraction.
function case_flops(case)
    contracted = [l for l in case.IA if l in case.IB && !(l in case.IC)]
    nC = prod(case.dims[l] for l in case.IC; init = 1)
    nK = prod(case.dims[l] for l in contracted; init = 1)
    return flops_per_mac(case.dtype) * nC * nK
end

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
flamegraph, and a bucket table (also appended to `io_summary`). Also takes
a separate, harness-standard (>=15-rep median) timing measurement -- used
for the printed GFLOP/s figure, so it does not depend on the single
untimed call used only to size the profiling loop's rep count.
"""
function profile_one!(io_summary, case, backendname, backend)
    rng = Random.Xoshiro(0x5eed_5eed)
    A, B, C, pA, pB, pAB = build_case_tensors(case, rng)
    T = case.dtype
    α, β = one(T), zero(T)

    call!() = tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, backend)

    median_s = median_time_s(call!; reps = 15)
    gflops = case_flops(case) / median_s / 1.0e9

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

    base = joinpath(PROFILE_DIR, _tagged("$(case.id)-$(backendname)"))

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
        @printf(io, "measured (median of 15 reps): %.9f s/call  %.3f GFLOP/s\n", median_s, gflops)
        print_bucket_table(io, case.id, backendname, buckets, total_samples, allocated_bytes, reps)
    end
    @printf(stdout, "measured (median of 15 reps): %.9f s/call  %.3f GFLOP/s\n", median_s, gflops)
    print_bucket_table(stdout, case.id, backendname, buckets, total_samples, allocated_bytes, reps)
    print_bucket_table(io_summary, case.id, backendname, buckets, total_samples, allocated_bytes, reps)

    return (
        case_id = case.id, backend = backendname, total_samples = total_samples,
        allocated_bytes = allocated_bytes, reps = reps, canvas_ok = canvas_ok,
        buckets = buckets, median_s = median_s, gflops = gflops,
    )
end

"""
    profile_one_direct!(io_summary, case)

Same as `profile_one!`, but for a `DIRECT_CASES` entry: builds the fixture
via `case.builder`, plans once with `plan_contract` (default kernel/blocking),
and profiles repeated `execute!` calls -- no TensorOperations adapter, no
backend choice.
"""
function profile_one_direct!(io_summary, case)
    backendname = "QuasiStridedDirect"
    rng = Random.Xoshiro(0x5eed_5eed)
    T = case.dtype
    fx = case.builder(T, rng)
    Av, indA, Bv, indB, Cv, indC = fx.Av, fx.indA, fx.Bv, fx.indB, fx.Cv, fx.indC
    # Optional `kernel` field (see the 1m entries appended to DIRECT_CASES
    # above): `hasproperty`, not `get`, since these are heterogeneous
    # NamedTuples (not every DIRECT_CASES entry carries a `kernel` field).
    kernel = hasproperty(case, :kernel) ? case.kernel : nothing
    plan = kernel === nothing ?
        plan_contract(Cv, Av, indA, Bv, indB, indC) :
        plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = kernel)
    α, β = one(T), zero(T)

    call!() = execute!(plan, α, β)

    median_s = median_time_s(call!; reps = 15)
    gflops = case.flops / median_s / 1.0e9

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

    data = Profile.fetch(include_meta = false)
    total_samples = count(iszero, data)

    base = joinpath(PROFILE_DIR, _tagged("$(case.id)-$(backendname)"))

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
        @printf(io, "measured (median of 15 reps): %.9f s/call  %.3f GFLOP/s\n", median_s, gflops)
        print_bucket_table(io, case.id, backendname, buckets, total_samples, allocated_bytes, reps)
    end
    @printf(stdout, "measured (median of 15 reps): %.9f s/call  %.3f GFLOP/s\n", median_s, gflops)
    print_bucket_table(stdout, case.id, backendname, buckets, total_samples, allocated_bytes, reps)
    print_bucket_table(io_summary, case.id, backendname, buckets, total_samples, allocated_bytes, reps)

    return (
        case_id = case.id, backend = backendname, total_samples = total_samples,
        allocated_bytes = allocated_bytes, reps = reps, canvas_ok = canvas_ok,
        buckets = buckets, median_s = median_s, gflops = gflops,
    )
end

function main()
    summary_path = joinpath(PROFILE_DIR, _tagged("buckets_summary") * ".txt")
    results = []
    open(summary_path, "w") do io_summary
        println(io_summary, "# Bucketed cost-attribution summary")
        print_env_header(io_summary, "profile_to_suite.jl")
        println(io_summary, "git_commit = ", git_commit())
        println(io_summary)
        for case in SELECTED_CASES
            for (backendname, backend) in BACKENDS
                @info "Profiling" case = case.id backend = backendname
                r = profile_one!(io_summary, case, backendname, backend)
                push!(results, r)
            end
        end
        for case in SELECTED_DIRECT_CASES
            @info "Profiling" case = case.id backend = "QuasiStridedDirect"
            r = profile_one_direct!(io_summary, case)
            push!(results, r)
        end
    end
    @info "Done" summary_path profile_dir = PROFILE_DIR
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
