# Head-to-head timing of `StridedNative()`, `StridedBLAS()` and
# `QuasiStridedBackend()` on the *upstream* TensorOperations.jl benchmark
# suite's own cases -- the `:pairwise` and `:tccg` categories of
# `TensorOperationsBenchmarks` -- rather than on this repo's hand-written
# matmul-shaped grid. (This script previously ran QuasiStrided through a
# benchmark-only `QuasiStridedComposite()` wrapper so the suite's
# one-backend-per-provider interface could also exercise `:permute`/`:trace`
# categories against it; `QuasiStridedBackend` now falls back to
# `StridedNative()` for those two operations itself -- see
# docs/decisions.md, "Amendment 7" -- so the wrapper is retired and every
# case here runs against the real backend directly.)
#
#   julia --project=benchmark benchmark/bench_to_suite.jl
#   # or, warm: jld --project=benchmark run benchmark/bench_to_suite.jl
#
# writes bench_to_suite.csv / canary_to_suite.csv / summary_to_suite.txt /
# mismatches_to_suite.txt / PROVENANCE_to_suite.txt to
# benchmark/results/<hostname>-<date>/.
#
# Why this is a sibling of benchmark/bench_tensoroperations.jl and not an
# extension of it: that script is the evidence base quoted in the README and
# every committed number under benchmark/results/ for it was taken against its
# own literal shape grid (see the comment there). This script keeps that
# script's measurement conventions verbatim -- warm-up-then-median timing,
# reps >= 15, a StridedBLAS 64^3 Float64 canary at start/middle/end, CSV +
# summary + PROVENANCE -- while taking its CASES from upstream, reusing
# benchmark/harness.jl's shared `median_time_s`/`results_dir`/`git_commit`/
# `print_env_header`/`relative_spread` rather than a local copy (StridedViews
# was added to benchmark/Project.toml so harness.jl can be `include`d here).
#
# TRIMMING APPLIED: none beyond upstream's own filter. The design contract
# allowed dropping Float32 and/or reducing `:tccg` if a dry run projected more
# than 45 minutes of wall clock; a dry run over the three most expensive cases
# projected well under that, so the full bounded grid below is run as
# specified: dtypes (Float64, Float32); `:pairwise` at dims {15, 63, 128};
# `:tccg` (all 24 chemistry specs) at dims {8, 16}. Note that
# `_pairwise_cases` yields 11 -- not 15 -- cases for those dims because
# upstream's own `within_memory_budget` (registry.jl, MAX_CASE_BYTES = 256 MiB
# assuming Float64 elements) drops dim63_2_2_2, dim128_2_1_2, dim128_2_2_2 and
# dim128_1_3_1. That is upstream's filter, not a trim by this script, and it is
# left in place deliberately: overriding it would change what the upstream
# suite's `:pairwise` category means.

using TensorOperations
using TensorOperations: StridedNative, StridedBLAS
using TensorOperationsBenchmarks
using TensorOperationsBenchmarks: BenchmarkCase, ContractSpec, flops, bytes,
    ArrayProvider, randtensor
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
import Pkg

include(joinpath(@__DIR__, "harness.jl"))  # median_time_s, results_dir, git_commit,
# print_env_header, relative_spread, and the single-core measurement pinning
# (LinearAlgebra/Random/Printf/Dates already `using`d there too).

const TOB = TensorOperationsBenchmarks

# 21 reps, not the 15-rep floor: STATUS.md "Measurement hygiene" records that
# an 11-rep comparison once invented two regressions that 21 reps erased, and
# this grid is cheap enough (see the dry-run note above) to afford the margin.
const REPS = 21

const DTYPES = (Float64, Float32)

const PAIRWISE_SIZES = (15, 63, 128)
const TCCG_SIZES = (8, 16)

# Per-case memory ceiling for *this* script, applied per dtype before any
# allocation. Upstream's own `within_memory_budget` (256 MiB, Float64-assumed)
# is stricter and has already run inside the generators, so this is a belt-and-
# braces guard that is expected never to fire; a case it does reject is logged
# and skipped, never an error.
const MAX_CASE_BYTES = 2 * 2^30  # 2 GiB

const BACKENDS = (
    StridedNative = StridedNative(),
    StridedBLAS = StridedBLAS(),
    QuasiStrided = QuasiStridedBackend(),
)

# ---------------------------------------------------------------------------
# Cases: pulled from the upstream suite's own category generators
# ---------------------------------------------------------------------------
#
# The generators are plain `sizes -> Vector{BenchmarkCase}` functions
# (TensorOperationsBenchmarks/src/registry.jl, `register_category!`), and
# `REGISTRY[:pairwise]`/`REGISTRY[:tccg]` hold exactly those two functions --
# asserted below so a rename upstream is a loud failure here rather than a
# silent divergence. They are called directly, bypassing
# `build_suite`/BenchmarkTools, because this script needs this project's own
# timing discipline (explicit reps, median, canary bracket) and not
# BenchmarkTools' statistics.
@assert TOB.REGISTRY[:pairwise] === TOB._pairwise_cases
@assert TOB.REGISTRY[:tccg] === TOB._tccg_cases

const CASES = vcat(TOB._pairwise_cases(PAIRWISE_SIZES), TOB._tccg_cases(TCCG_SIZES))

# Element counts per operand, from the spec's labels/dims alone.
_nelem(spec::ContractSpec, I) = prod((spec.dims[l] for l in I); init = 1)

# Total operand bytes at element type `T`. `bytes(spec)` itself always assumes
# Float64-sized elements (cost.jl's `_elsize(::Nothing)`), so it cannot be used
# directly for the Float32 rows; this reduces to `bytes(spec)` for Float64.
function case_bytes(spec::ContractSpec, ::Type{T}) where {T}
    n = _nelem(spec, spec.IA) + _nelem(spec, spec.IB) + _nelem(spec, spec.IC)
    return n * sizeof(T)
end

params_string(params::NamedTuple) =
    join(("$k=$(getfield(params, k))" for k in keys(params)), ";")

# Build the operands for one case. `pA`/`pB`/`pAB` come from
# `TensorOperations.contract_indices(IA, IB, IC)` -- TO's own label-to-position
# resolver, the very call the upstream suite's `maketensors(::ContractSpec, ...)`
# (lowering.jl) and TO's own `@tensor`/`ncon` lowering use. Deriving the
# `Index2Tuple`s by hand here would be a needless reimplementation of exactly
# that function and the obvious place for a silent transposition bug, so it is
# not done. Operands are built through the upstream `ArrayProvider{T}` (plain
# `Array{T}` filled by `randn!` from the provider's stored Xoshiro), so the
# data is the upstream suite's own; `C` is allocated here, pre-zeroed and
# fresh per backend.
function build_case(spec::ContractSpec, provider, ::Type{T}) where {T}
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    dimsC = ntuple(i -> spec.dims[spec.IC[i]], length(spec.IC))
    A = randtensor(provider, spec.IA, dimsA, T)
    B = randtensor(provider, spec.IB, dimsB, T)
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    return A, B, pA, pB, pAB, dimsC
end

# One in-place contraction under a given backend, alpha = 1, beta = 0 --
# the same call the upstream suite's `execute(::ContractSpec, ...)` makes,
# with only `backend` varying across the three columns.
function run_case!(backend, C, A, pA, conjA, B, pB, conjB, pAB)
    return TensorOperations.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB,
        one(eltype(C)), zero(eltype(C)), backend
    )
end

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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
    "backend,dtype,category,case_id,dim,params,reps,median_seconds,gflops,gbytes"
)
function log_row(backend_name, T, case::BenchmarkCase, reps, t, gf, gb)
    println(
        csv_io,
        "$backend_name,$T,$(case.category),$(case.id),$(case.params.dim),",
        params_string(case.params), ",$reps,",
        @sprintf("%.9f,%.4f,%.4f", t, gf, gb)
    )
    return flush(csv_io)
end

print_env_header(stdout, "bench_to_suite.jl")
println("cases = ", length(CASES), " per dtype (before per-dtype byte skips)")

# ---------------------------------------------------------------------------
# Canary -- byte-for-byte the same case as benchmark/bench_tensoroperations.jl:
# StridedBLAS, 64^3 Float64, matmul-shaped `@tensor`, 15 reps, run at the
# start / middle / end of the sweep to catch drift (thermal throttling,
# background load).
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

raw = Vector{NamedTuple}()          # successful timings
mismatches = Vector{NamedTuple}()   # QuasiStrided result != StridedBLAS result
failures = Vector{NamedTuple}()     # a backend threw
skipped = Vector{NamedTuple}()      # over this script's byte ceiling

const MIDPOINT = cld(length(CASES) * length(DTYPES), 2)
progress = 0

for T in DTYPES
    # One fresh provider per dtype: its Xoshiro is seeded deterministically
    # (0x5eed5eed5eed5eed) and stateful, so the whole dtype sweep is
    # reproducible while individual tensors still differ.
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

        A, B, pA, pB, pAB, dimsC = build_case(spec, provider, T)
        fl = flops(spec)

        # -------------------------------------------------------------------
        # Correctness gate, BEFORE any timing: one untimed call per backend.
        # StridedBLAS is the reference; QuasiStrided must match it to `rtol`
        # or its timing for this case is not taken at all.
        # -------------------------------------------------------------------
        results = Dict{Symbol, Any}()
        for (bname, backend) in pairs(BACKENDS)
            C = zeros(T, dimsC)
            try
                run_case!(backend, C, A, pA, spec.conjA, B, pB, spec.conjB, pAB)
                results[bname] = C
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

        # -------------------------------------------------------------------
        # Timing
        # -------------------------------------------------------------------
        for (bname, backend) in pairs(BACKENDS)
            haskey(results, bname) || continue           # threw above
            bname === :QuasiStrided && !qs_ok && continue # mismatched above
            C = zeros(T, dimsC)
            t = median_time_s(
                () -> run_case!(backend, C, A, pA, spec.conjA, B, pB, spec.conjB, pAB);
                reps = REPS
            )
            gf = fl / t / 1.0e9
            gb = cb / t / 1.0e9
            log_row(bname, T, case, REPS, t, gf, gb)
            push!(
                raw,
                (
                    backend = String(bname), dtype = T, category = case.category,
                    id = case.id, dim = case.params.dim, t = t, gflops = gf, gbytes = gb,
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

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

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
    println(
        io,
        "NOTE: the QuasiStrided column is QuasiStridedBackend() directly -- ",
        "every case here IS a contraction, so no timing below measures its ",
        "StridedNative tensoradd!/tensortrace! fallback (see docs/decisions.md, ",
        "\"Amendment 7\")."
    )
    if !isempty(mismatches)
        println(io, "\n!! ", length(mismatches), " CORRECTNESS MISMATCH(ES) -- see mismatches_to_suite.txt")
    end
    if !isempty(failures)
        println(io, "!! ", length(failures), " BACKEND REJECTION(S)/ERROR(S) -- see mismatches_to_suite.txt")
    end

    for T in DTYPES
        for cat in (:pairwise, :tccg)
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
                tn, tb, tq = tof("StridedNative"), tof("StridedBLAS"), tof("QuasiStrided")
                qs_blas = (tq === nothing || tb === nothing) ? "n/a" :
                    @sprintf("%.3f", tq / tb)
                nat_qs = (tn === nothing || tq === nothing) ? "n/a" :
                    @sprintf("%.3f", tn / tq)
                println(
                    io, "    -> QS/BLAS = ", qs_blas,
                    "   (>1 = QuasiStrided slower than BLAS)",
                    "   Native/QS = ", nat_qs,
                    "   (>1 = QuasiStrided faster than Native)"
                )
            end

            # Category-level geometric means of the two headline ratios.
            gq, gn = Float64[], Float64[]
            for id in ids
                rows = filter(r -> r.dtype == T && r.category == cat && r.id == id, raw)
                tof(b) = (i = findfirst(r -> r.backend == b, rows); i === nothing ? nothing : rows[i].t)
                tn, tb, tq = tof("StridedNative"), tof("StridedBLAS"), tof("QuasiStrided")
                (tq !== nothing && tb !== nothing) && push!(gq, tq / tb)
                (tn !== nothing && tq !== nothing) && push!(gn, tn / tq)
            end
            geomean(v) = isempty(v) ? NaN : exp(sum(log, v) / length(v))
            println(
                io, "  [", T, "/", cat, "] geomean QS/BLAS = ",
                @sprintf("%.3f", geomean(gq)), " over ", length(gq), " cases; ",
                "geomean Native/QS = ", @sprintf("%.3f", geomean(gn)),
                " over ", length(gn), " cases"
            )
        end
    end
end
println(read(SUMMARY_PATH, String))

# ---------------------------------------------------------------------------
# Mismatches / rejections
# ---------------------------------------------------------------------------

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
    println(io, "\n## Cases skipped by this script's ", MAX_CASE_BYTES, "-byte ceiling")
    if isempty(skipped)
        println(io, "none (upstream's own 256 MiB within_memory_budget is stricter and ran first)")
    else
        for s in skipped
            println(io, "  SKIP ", s.category, "/", s.id, " dtype=", s.dtype, " bytes=", s.bytes)
        end
    end
end
println(read(MISMATCH_PATH, String))

# ---------------------------------------------------------------------------
# Provenance -- format matched to benchmark/bench_tensoroperations.jl's.
# ---------------------------------------------------------------------------

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
    println(
        io, "TensorOperationsBenchmarks pinned rev (benchmark/Project.toml) = ",
        "528dd85d8bf886c734a207732a7cb591a3691dd3 ",
        "(QuantumKitHub/TensorOperations.jl, branch \"benchmark\", subdir benchmark, PR #303)"
    )
    println(io, "backends = ", collect(keys(BACKENDS)))
    println(io, "  QuasiStrided column = QuasiStridedBackend() directly (docs/decisions.md, \"Amendment 7\").")
    println(io, "dtypes = ", collect(DTYPES))
    println(io, "reps = ", REPS, " (median, one discarded warm-up)")
    println(io, "case source = TensorOperationsBenchmarks._pairwise_cases / ._tccg_cases,")
    println(io, "  called directly (asserted identical to REGISTRY[:pairwise]/[:tccg]);")
    println(io, "  build_suite/BenchmarkTools deliberately bypassed for this project's timing discipline.")
    println(io, "pairwise sizes = ", PAIRWISE_SIZES)
    println(io, "tccg sizes = ", TCCG_SIZES)
    println(io, "cases generated = ", length(CASES), " per dtype")
    println(io, "cases actually timed (rows/3 nominal) = ", length(raw), " backend-rows total")
    println(io, "case list = ")
    for case in CASES
        println(io, "  ", case.category, "/", case.id, "  ", params_string(case.params))
    end
    println(io, "trimming = none applied beyond upstream's own within_memory_budget")
    println(io, "  (registry.jl MAX_CASE_BYTES = 256 MiB, Float64-assumed), which drops")
    println(io, "  dim63_2_2_2, dim128_2_1_2, dim128_2_2_2 and dim128_1_3_1 from :pairwise,")
    println(io, "  leaving 11 of a nominal 15. A dry run over the three most expensive")
    println(io, "  cases projected total wall time far under the 45-minute budget, so")
    println(io, "  neither the Float32 drop nor the :tccg reduction the design contract")
    println(io, "  allowed was needed.")
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
    println(io, "canary = StridedBLAS, 64^3 Float64 matmul-shaped @tensor, 15 reps,")
    println(io, "  identical to benchmark/bench_tensoroperations.jl's canary; run at start/middle/end.")
    println(io, "canary_medians_s = ", canary_results)
    println(io, "canary_relative_spread = ", @sprintf("%.4f", canary_spread))
    println(io, "noise_floor_used = ", @sprintf("%.4f", NOISE_FLOOR))
    println(io, "machine_load_at_run = ", machine_load)
    println(io, "top_processes_at_run =")
    for line in split(top_procs, '\n')
        println(io, "  ", line)
    end
    println(
        io,
        "measurement_hygiene_caveat = this machine is NOT guaranteed exclusive; the load ",
        "average and process list above were captured by this run. Any difference under ",
        @sprintf("%.1f%%", 100 * NOISE_FLOOR), " must be read as noise."
    )
    println(io, "caveat = single machine ($(gethostname())), single measurement session; ")
    println(
        io,
        "  not averaged across machines or repeated sessions. Numbers are indicative of ",
        "this reference machine only, matching the caveat in the macro-blocking ",
        "milestone's benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/PROVENANCE.txt."
    )
end

println("\nDone. Results in ", OUTDIR)
