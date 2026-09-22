# Head-to-head timing of StridedBLAS() and QuasiStridedBackend() on the
# upstream TensorOperations.jl benchmark suite's :pairwise/:tccg/:mps/:ctmrg/
# :trg cases.
#
# StridedNative() was dropped from BACKENDS (2026-09-22): it was already
# excluded from plot_bench_to_suite.jl's plots (see that file's header), so
# timing it here only cost walltime -- and on :ctmrg/:trg specifically it has
# a severe, size-growing slowdown (see the --trg-chis note below) that made a
# ComplexF64 run risk blowing the job's walltime budget for numbers nobody
# looks at.
#
#   julia --project=benchmark benchmark/bench_to_suite.jl [options]
#
# Options (all optional):
#   --categories pairwise,tccg,mps,ctmrg,trg
#   --dtypes Float64,Float32
#   --pairwise-sizes 15,63,128
#   --tccg-sizes 8,16
#   --mps-bonddims 32,64,128     # MPS/MPO effective-Hamiltonian bond dim D
#   --ctmrg-chis 16,32,64        # CTMRG environment bond dim chi
#   --trg-chis 16,32,48          # TRG plaquette bond dim chi
#   --reps 21
#   --max-bytes 2147483648      # per-case skip ceiling
#
# :mps/:ctmrg/:trg are multi-tensor `NetworkSpec` cases (run via `ncon`, not
# `tensorcontract!` directly), unlike :pairwise/:tccg's two-tensor
# `ContractSpec` cases -- see build_case/run_case! dispatch below. `ncon` has
# no in-place variant, so timing for these three categories includes output
# allocation (this matches upstream's own accepted discipline, see
# TensorOperationsBenchmarks/src/lowering.jl).
#
# Writes bench_to_suite.csv / canary_to_suite.csv / summary_to_suite.txt /
# mismatches_to_suite.txt / PROVENANCE_to_suite.txt to
# benchmark/results/<hostname>-<date>/.
#
# bench_to_suite.csv's `gflops` column is the median-time-based rate
# (unchanged); `min_gflops`/`std_gflops` are the min and standard deviation
# of the REPS per-rep GFLOP/s samples (not derived from min/max *time*
# converted to a rate -- computed directly on the per-rep throughput array so
# they describe the throughput distribution plot_bench_to_suite.jl's violin
# plots are built from).

using TensorOperations
using TensorOperations: StridedBLAS
using TensorOperationsBenchmarks
using TensorOperationsBenchmarks: BenchmarkCase, ContractSpec, NetworkSpec, flops, bytes,
    ArrayProvider, randtensor
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
using Statistics: std
import Pkg

include(joinpath(@__DIR__, "harness.jl"))

const TOB = TensorOperationsBenchmarks

const REPS = argopt("reps", 21)
const DTYPES = parse_dtypes(argopt("dtypes", "Float64,Float32"))
const CATEGORIES = Symbol.(split(argopt("categories", "pairwise,tccg"), ','))
const PAIRWISE_SIZES = parse_ints(argopt("pairwise-sizes", "15,63,128"))
const TCCG_SIZES = parse_ints(argopt("tccg-sizes", "8,16"))
const MPS_BONDDIMS = parse_ints(argopt("mps-bonddims", "32,64,128"))
const CTMRG_CHIS = parse_ints(argopt("ctmrg-chis", "16,32,64"))
const TRG_CHIS = parse_ints(argopt("trg-chis", "16,32,48"))
const MAX_CASE_BYTES = argopt("max-bytes", 2 * 2^30)

const BACKENDS = (
    StridedBLAS = StridedBLAS(),
    QuasiStrided = QuasiStridedBackend(),
)

# `REGISTRY[:pairwise]`/`[:tccg]`/`[:mps]`/`[:ctmrg]`/`[:trg]` are called
# directly, bypassing `build_suite`/BenchmarkTools, for this project's own
# timing discipline.
@assert TOB.REGISTRY[:pairwise] === TOB._pairwise_cases
@assert TOB.REGISTRY[:tccg] === TOB._tccg_cases
@assert TOB.REGISTRY[:mps] === TOB._mps_cases
@assert TOB.REGISTRY[:ctmrg] === TOB._ctmrg_cases
@assert TOB.REGISTRY[:trg] === TOB._trg_cases

const CASES = vcat(
    :pairwise in CATEGORIES ? TOB._pairwise_cases(PAIRWISE_SIZES) : BenchmarkCase[],
    :tccg in CATEGORIES ? TOB._tccg_cases(TCCG_SIZES) : BenchmarkCase[],
    :mps in CATEGORIES ? TOB._mps_cases(MPS_BONDDIMS) : BenchmarkCase[],
    :ctmrg in CATEGORIES ? TOB._ctmrg_cases(CTMRG_CHIS) : BenchmarkCase[],
    :trg in CATEGORIES ? TOB._trg_cases(TRG_CHIS) : BenchmarkCase[],
)

_nelem(spec::ContractSpec, I) = prod((spec.dims[l] for l in I); init = 1)
_nelem_network(spec::NetworkSpec, il) = prod((spec.dims[abs(l)] for l in il); init = 1)

# `bytes(spec)` assumes Float64 elements; this is dtype-generic.
function case_bytes(spec::ContractSpec, ::Type{T}) where {T}
    n = _nelem(spec, spec.IA) + _nelem(spec, spec.IB) + _nelem(spec, spec.IC)
    return n * sizeof(T)
end

# All input tensors plus the (fresh, `ncon`-allocated) output.
function case_bytes(spec::NetworkSpec, ::Type{T}) where {T}
    n = sum(_nelem_network(spec, il) for il in spec.indexlists)
    n += _nelem_network(spec, spec.output)
    return n * sizeof(T)
end

params_string(params::NamedTuple) =
    join(("$k=$(getfield(params, k))" for k in keys(params)), ";")

# :pairwise/:tccg cases sweep a leg dimension `dim`; :mps sweeps bond dim `D`;
# :ctmrg/:trg sweep environment/plaquette bond dim `chi`. This picks whichever
# is present so the CSV/summary can report a single generic sweep-parameter
# column across all categories without renaming pairwise/tccg's own `dim`.
function case_sweepparam(case::BenchmarkCase)
    p = case.params
    hasproperty(p, :dim) && return p.dim
    hasproperty(p, :D) && return p.D
    hasproperty(p, :chi) && return p.chi
    error("case $(case.category)/$(case.id) has no known sweep-parameter field (dim/D/chi)")
end

function build_case(spec::ContractSpec, provider, ::Type{T}) where {T}
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    dimsC = ntuple(i -> spec.dims[spec.IC[i]], length(spec.IC))
    A = randtensor(provider, spec.IA, dimsA, T)
    B = randtensor(provider, spec.IB, dimsB, T)
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    return (; A, B, pA, pB, pAB, dimsC)
end

function build_case(spec::NetworkSpec, provider, ::Type{T}) where {T}
    tensors = map(spec.indexlists) do il
        dims = ntuple(i -> spec.dims[abs(il[i])], length(il))
        return randtensor(provider, il, dims, T)
    end
    return (; tensors)
end

alloc_output(spec::ContractSpec, ctx, ::Type{T}) where {T} = zeros(T, ctx.dimsC)
# `ncon` has no in-place variant -- it allocates its own output every call;
# this placeholder is ignored by `run_case!` below.
alloc_output(::NetworkSpec, ctx, ::Type{T}) where {T} = nothing

function run_case!(backend, spec::ContractSpec, ctx, C)
    return TensorOperations.tensorcontract!(
        C, ctx.A, ctx.pA, spec.conjA, ctx.B, ctx.pB, spec.conjB, ctx.pAB,
        one(eltype(C)), zero(eltype(C)), backend
    )
end

function run_case!(backend, spec::NetworkSpec, ctx, C)
    return TensorOperations.ncon(
        ctx.tensors, spec.indexlists, spec.conjlist;
        order = spec.order, output = spec.output, backend = backend
    )
end

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "bench_to_suite.csv")
const CANARY_PATH = joinpath(OUTDIR, "canary_to_suite.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_to_suite.txt")
const MISMATCH_PATH = joinpath(OUTDIR, "mismatches_to_suite.txt")
const PROVENANCE_PATH = joinpath(OUTDIR, "PROVENANCE_to_suite.txt")

csv_io = open(CSV_PATH, "w")
println(
    csv_io,
    "backend,dtype,category,case_id,dim,params,reps,median_seconds,gflops,gbytes,min_gflops,std_gflops"
)
function log_row(backend_name, T, case::BenchmarkCase, reps, t, gf, gb, min_gf, std_gf)
    println(
        csv_io,
        "$backend_name,$T,$(case.category),$(case.id),$(case_sweepparam(case)),",
        params_string(case.params), ",$reps,",
        @sprintf("%.9f,%.4f,%.4f,%.4f,%.4f", t, gf, gb, min_gf, std_gf)
    )
    return flush(csv_io)
end

# Per-rep timing samples (not just the median) -- same warm-up-then-timed
# discipline as harness.jl's `median_time_s`, but returns every sample so the
# caller can compute min/std of the derived throughput, not just its median.
function timed_samples_s(f!::Function; reps::Int)
    f!()  # warm-up, discarded
    ts = Vector{Float64}(undef, reps)
    for r in 1:reps
        t0 = time_ns()
        f!()
        t1 = time_ns()
        ts[r] = (t1 - t0) / 1.0e9
    end
    return ts
end

print_env_header(stdout, "bench_to_suite.jl")
println("cases = ", length(CASES), " per dtype (before per-dtype byte skips)")

# Canary: StridedBLAS, 64^3 Float64 matmul-shaped @tensor, 15 reps, at
# start/middle/end -- drift detection (thermal throttling, background load).
function canary_contract!(backend, C, A, B)
    @tensor backend = backend C[i, j] = A[i, k] * B[k, j]
    return C
end

function run_canary(rng, label::String)
    A = randn(rng, Float64, 64, 64)
    B = randn(rng, Float64, 64, 64)
    C = zeros(Float64, 64, 64)
    t = median_time_s(() -> canary_contract!(StridedBLAS(), C, A, B); reps = 15)
    println("canary[$label] median = $(t) s")
    return t
end

canary_rng = MersenneTwister(0xB3_C4_0003)
canary_results = Float64[]
push!(canary_results, run_canary(canary_rng, "A (start)"))

raw = Vector{NamedTuple}()          # successful timings
mismatches = Vector{NamedTuple}()   # QuasiStrided result != StridedBLAS result
failures = Vector{NamedTuple}()     # a backend threw
skipped = Vector{NamedTuple}()      # over --max-bytes

const MIDPOINT = cld(length(CASES) * length(DTYPES), 2)
progress = 0

for T in DTYPES
    provider = ArrayProvider{T}()
    rtol = T === Float64 ? 1.0e-10 : 1.0e-5
    for case in CASES
        global progress += 1
        spec = case.spec
        cb = case_bytes(spec, T)
        if cb > MAX_CASE_BYTES
            push!(skipped, (dtype = T, category = case.category, id = case.id, bytes = cb))
            @info "skipped (over byte ceiling)" dtype = T id = case.id bytes = cb
            continue
        end

        ctx = build_case(spec, provider, T)
        fl = flops(spec)

        # Correctness gate before any timing: StridedBLAS is the reference,
        # QuasiStrided must match to `rtol` or its timing is skipped.
        results = Dict{Symbol, Any}()
        for (bname, backend) in pairs(BACKENDS)
            C = alloc_output(spec, ctx, T)
            try
                results[bname] = run_case!(backend, spec, ctx, C)
            catch e
                msg = sprint(showerror, e)
                push!(
                    failures,
                    (
                        dtype = T, category = case.category, id = case.id,
                        backend = String(bname), message = first(msg, 400),
                    )
                )
                @warn "backend threw" dtype = T id = case.id backend = bname msg
            end
        end

        ref = get(results, :StridedBLAS, nothing)
        qs = get(results, :QuasiStrided, nothing)
        qs_ok = true
        if ref !== nothing && qs !== nothing
            if !isapprox(qs, ref; rtol = rtol)
                nref = norm(vec(ref))
                disc = nref == 0 ? norm(vec(qs)) : norm(vec(qs) - vec(ref)) / nref
                qs_ok = false
                push!(
                    mismatches,
                    (
                        dtype = T, category = case.category, id = case.id,
                        rtol = rtol, discrepancy = disc,
                    )
                )
                @error "MISMATCH: QuasiStrided != StridedBLAS -- timing skipped" dtype = T id = case.id discrepancy = disc
            end
        end

        for (bname, backend) in pairs(BACKENDS)
            haskey(results, bname) || continue           # threw above
            bname === :QuasiStrided && !qs_ok && continue # mismatched above
            C = alloc_output(spec, ctx, T)
            times = timed_samples_s(() -> run_case!(backend, spec, ctx, C); reps = REPS)
            t = median(times)
            gf = fl / t / 1.0e9
            gb = cb / t / 1.0e9
            gflops_samples = (fl ./ times) ./ 1.0e9
            min_gf = minimum(gflops_samples)
            std_gf = std(gflops_samples)
            log_row(bname, T, case, REPS, t, gf, gb, min_gf, std_gf)
            push!(
                raw,
                (
                    backend = String(bname), dtype = T, category = case.category,
                    id = case.id, dim = case_sweepparam(case), t = t, gflops = gf, gbytes = gb,
                    min_gflops = min_gf, std_gflops = std_gf,
                )
            )
        end

        if progress == MIDPOINT
            push!(canary_results, run_canary(canary_rng, "B (middle)"))
        end
    end
    @info "dtype done" dtype = T rows = length(raw)
end

push!(canary_results, run_canary(canary_rng, "A' (end)"))
close(csv_io)

canary_spread = relative_spread(canary_results)
open(CANARY_PATH, "w") do io
    println(io, "label,median_seconds")
    labels = length(canary_results) == 3 ?
        ("A_start", "B_middle", "Aprime_end") :
        Tuple("c$i" for i in 1:length(canary_results))
    for (lbl, t) in zip(labels, canary_results)
        println(io, "$lbl,", @sprintf("%.9f", t))
    end
    println(io, "# relative spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
end
println("canary spread (max-min)/min = ", @sprintf("%.4f", canary_spread))

const NOISE_FLOOR = max(0.1, canary_spread)

open(SUMMARY_PATH, "w") do io
    println(io, "# TensorOperations upstream-suite backend benchmark summary")
    print_env_header(io, "bench_to_suite.jl")
    println(io, "reps = ", REPS, " (median of ", REPS, ", one discarded warm-up)")
    println(io, "canary median times (s): ", canary_results)
    println(io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", canary_spread))
    println(
        io,
        "NOISE: read any difference smaller than max(10%, canary spread) = ",
        @sprintf("%.1f%%", 100 * NOISE_FLOOR), " as noise, not as a result."
    )
    if !isempty(mismatches)
        println(io, "\n!! ", length(mismatches), " CORRECTNESS MISMATCH(ES) -- see mismatches_to_suite.txt")
    end
    if !isempty(failures)
        println(io, "!! ", length(failures), " BACKEND REJECTION(S)/ERROR(S) -- see mismatches_to_suite.txt")
    end

    for T in DTYPES
        for cat in CATEGORIES
            println(io, "\n===== ", T, " / ", cat, " =====")
            ids = unique(r.id for r in raw if r.dtype == T && r.category == cat)
            for id in ids
                rows = filter(r -> r.dtype == T && r.category == cat && r.id == id, raw)
                isempty(rows) && continue
                best = minimum(r.t for r in rows)
                println(io, "  ", id, ":")
                for r in sort(collect(rows); by = r -> r.t)
                    println(
                        io, "    ", rpad(r.backend, 14), @sprintf("%.6e s", r.t),
                        @sprintf("  %8.2f GFLOP/s  %7.2f GB/s", r.gflops, r.gbytes),
                        "  (", @sprintf("%.3fx", r.t / best), " of fastest)"
                    )
                end
                tof(b) = (i = findfirst(r -> r.backend == b, rows); i === nothing ? nothing : rows[i].t)
                tb, tq = tof("StridedBLAS"), tof("QuasiStrided")
                qs_blas = (tq === nothing || tb === nothing) ? "n/a" :
                    @sprintf("%.3f", tq / tb)
                println(
                    io, "    -> QS/BLAS = ", qs_blas,
                    "   (>1 = QuasiStrided slower than BLAS)"
                )
            end

            gq = Float64[]
            for id in ids
                rows = filter(r -> r.dtype == T && r.category == cat && r.id == id, raw)
                tof(b) = (i = findfirst(r -> r.backend == b, rows); i === nothing ? nothing : rows[i].t)
                tb, tq = tof("StridedBLAS"), tof("QuasiStrided")
                (tq !== nothing && tb !== nothing) && push!(gq, tq / tb)
            end
            geomean(v) = isempty(v) ? NaN : exp(sum(log, v) / length(v))
            println(
                io, "  [", T, "/", cat, "] geomean QS/BLAS = ",
                @sprintf("%.3f", geomean(gq)), " over ", length(gq), " cases"
            )
        end
    end
end
println(read(SUMMARY_PATH, String))

open(MISMATCH_PATH, "w") do io
    println(io, "# Correctness mismatches, backend rejections and skips")
    println(io, "# benchmark/bench_to_suite.jl -- ", gethostname(), " ", now())
    println(io, "\n## QuasiStrided-vs-StridedBLAS mismatches (timing NOT taken for these)")
    if isempty(mismatches)
        println(io, "none found")
    else
        for m in mismatches
            println(
                io, "  MISMATCH ", m.category, "/", m.id, " dtype=", m.dtype,
                " rtol=", m.rtol,
                @sprintf(" norm(diff)/norm(ref)=%.3e", m.discrepancy)
            )
        end
    end
    println(io, "\n## Backend errors / rejections (backend threw; other backends still timed)")
    if isempty(failures)
        println(io, "none found")
    else
        for f in failures
            println(
                io, "  ERROR ", f.category, "/", f.id, " dtype=", f.dtype,
                " backend=", f.backend, ": ", replace(f.message, "\n" => " | ")
            )
        end
    end
    println(io, "\n## Cases skipped by --max-bytes=", MAX_CASE_BYTES)
    if isempty(skipped)
        println(io, "none (upstream's own 256 MiB within_memory_budget is stricter and ran first)")
    else
        for s in skipped
            println(io, "  SKIP ", s.category, "/", s.id, " dtype=", s.dtype, " bytes=", s.bytes)
        end
    end
end
println(read(MISMATCH_PATH, String))

commit = git_commit()

to_version, tob_rev = try
    deps = Pkg.dependencies()
    tov = tobr = "unknown"
    for (_, info) in deps
        if info.name == "TensorOperations"
            tov = string(info.version)
        elseif info.name == "TensorOperationsBenchmarks"
            tobr = string(
                something(info.git_revision, "?"), " (tree ",
                something(info.tree_hash, "?"), ")"
            )
        end
    end
    tov, tobr
catch e
    "unknown ($(sprint(showerror, e)))", "unknown"
end

machine_load = try
    strip(read(`uptime`, String))
catch
    "unknown (uptime failed)"
end
top_procs = try
    strip(read(pipeline(`ps -eo pcpu,comm --sort=-pcpu`, `head -6`), String))
catch
    "unknown (ps failed)"
end

open(PROVENANCE_PATH, "w") do io
    println(io, "git_commit = ", commit)
    println(io, "command = julia --project=benchmark benchmark/bench_to_suite.jl")
    print_env_header(io, "bench_to_suite.jl")
    println(io, "logical_cpus = ", Sys.CPU_THREADS)
    println(io, "blas_config = ", LinearAlgebra.BLAS.get_config())
    println(io, "TensorOperations = ", to_version)
    println(io, "TensorOperationsBenchmarks = ", tob_rev)
    println(io, "backends = ", collect(keys(BACKENDS)), " (QuasiStrided = QuasiStridedBackend() directly)")
    println(io, "dtypes = ", collect(DTYPES))
    println(io, "reps = ", REPS, " (median, one discarded warm-up)")
    println(io, "categories = ", CATEGORIES)
    println(io, "pairwise sizes = ", PAIRWISE_SIZES)
    println(io, "tccg sizes = ", TCCG_SIZES)
    println(io, "mps bonddims = ", MPS_BONDDIMS)
    println(io, "ctmrg chis = ", CTMRG_CHIS)
    println(io, "trg chis = ", TRG_CHIS)
    println(io, "cases generated = ", length(CASES), " per dtype")
    println(io, "cases actually timed = ", length(raw), " backend-rows total")
    println(io, "case list = ")
    for case in CASES
        println(io, "  ", case.category, "/", case.id, "  ", params_string(case.params))
    end
    println(io, "mismatches = ", length(mismatches), " (see mismatches_to_suite.txt)")
    for m in mismatches
        println(
            io, "  ", m.category, "/", m.id, " dtype=", m.dtype,
            @sprintf(" norm(diff)/norm(ref)=%.3e", m.discrepancy)
        )
    end
    println(io, "backend_rejections = ", length(failures), " (see mismatches_to_suite.txt)")
    for f in failures
        println(io, "  ", f.category, "/", f.id, " dtype=", f.dtype, " backend=", f.backend)
    end
    println(io, "skipped_cases = ", length(skipped))
    println(io, "canary_medians_s = ", canary_results)
    println(io, "canary_relative_spread = ", @sprintf("%.4f", canary_spread))
    println(io, "noise_floor_used = ", @sprintf("%.4f", NOISE_FLOOR))
    println(io, "machine_load_at_run = ", machine_load)
    println(io, "top_processes_at_run =")
    for line in split(top_procs, '\n')
        println(io, "  ", line)
    end
    println(io, "caveat = single machine ($(gethostname())), single measurement session.")
end

println("\nDone. Results in ", OUTDIR)
