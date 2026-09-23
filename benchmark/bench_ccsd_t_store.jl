# Times four TCCG quantum-chemistry contractions (6-index output, 1
# contracted index) across several "arms":
#
#   ccsd_t_1: C[a,b,c,i,j,k] = A[i,j,m,a] * B[m,k,b,c]
#   ccsd_t_2: C[a,b,c,i,j,k] = A[i,j,m,b] * B[m,k,a,c]
#   ccsd_t_3: C[a,b,c,i,j,k] = A[i,j,m,c] * B[m,k,a,b]
#   ccsd_t_4: C[a,b,c,i,j,k] = A[i,k,m,b] * B[m,j,a,c]
#
#   Arm 1:     TO.tensorcontract! under StridedNative()/StridedBLAS()/QuasiStridedBackend().
#   Arm 2:     QuasiStrided.plan_contract/execute! directly, with the same label
#              order TO.tensorcontract! derives internally (via _qs_labels).
#   Arm 3:     same as Arm 2, but with whichever operand carries `a` (C's
#              stride-1 axis) permuted so `a` is that operand's own first
#              physical axis. Only `a` moves; the other labels keep their
#              relative order.
#   Arm 4-M:   same as Arm 2, but operand A's free (M) axes are reordered by
#              increasing |C-stride| of their label ("M sorted by |C-stride|").
#              Contracted axes keep their positions.
#   Arm 4-N:   same for operand B's free (N) axes.
#   Arm 4both: 4-M and 4-N together.
#   Arm 5:     4both, plus an M/N *orientation swap* whenever C's stride-1
#              label is not on A: B is fed into plan_contract's A (M-role)
#              slot and vice versa, so C's stride-1 label lands in the M
#              composite for every case. For ccsd_t_1 no swap is needed and
#              Arm 5 is a repeat of Arm 4both.
#
# NOTE: `plan_contract` performs the label sort and orientation
# swap that Arms 3/4-*/5 emulate by hand INTERNALLY, unconditionally, on
# every call -- so on current `src/planning/labels.jl`, Arm 2's own label order is
# already re-sorted by `_order_free_labels` before these arms' extra operand
# permutations are even applied. These arms describe PRE-FIX semantics; they
# are kept as a working prototype/regression record of what motivated the
# fix, not as arms that still change engine behaviour today.
#
# All operand permutations (Arms 3, 4-*, 5) are `permutedims` on a
# `StridedView`, i.e. lazy views with permuted strides -- no data is copied.
# They emulate a planning-time label-order change with zero extra data
# movement.
#
#   julia --project=. benchmark/bench_ccsd_t_store.jl [options]
#
# Options:
#   --smoke              dims=(8,) only, quick check
#   --dims 8,16
#   --dtypes Float64,Float32
#
# Writes benchmark/results/<hostname>-<date>/{bench_ccsd_t_store.csv,
# summary_ccsd_t_store.txt,PROVENANCE_ccsd_t_store.txt}.

using TensorOperations
import TensorOperations as TO
using TensorOperations: StridedNative, StridedBLAS
using QuasiStrided
using QuasiStrided: QuasiStridedBackend
using StridedViews
using StridedViews: StridedView
using Random

include(joinpath(@__DIR__, "harness.jl"))

const SMOKE = hasflag("smoke")
const DIMS = SMOKE ? (8,) : parse_ints(argopt("dims", "8,16"))
const CORRECTNESS_DTYPES = parse_dtypes(argopt("dtypes", "Float64,Float32"))

# `arm3_target` names which operand carries `a` (C's stride-1 axis) and
# therefore gets permuted for Arm 3; `arm3_perm` is that operand's own
# 4-axis permutation moving `a` to the front.
struct CaseSpec
    name::String
    IA::NTuple{4, Symbol}
    IB::NTuple{4, Symbol}
    IC::NTuple{6, Symbol}
    arm3_target::Symbol
    arm3_perm::NTuple{4, Int}
end

const CASES = [
    CaseSpec(
        "ccsd_t_1", (:i, :j, :m, :a), (:m, :k, :b, :c), (:a, :b, :c, :i, :j, :k),
        :A, (4, 1, 2, 3)  # A is (i,j,m,a); move a (axis 4) to front -> (a,i,j,m)
    ),
    CaseSpec(
        "ccsd_t_2", (:i, :j, :m, :b), (:m, :k, :a, :c), (:a, :b, :c, :i, :j, :k),
        :B, (3, 1, 2, 4)  # B is (m,k,a,c); move a (axis 3) to front -> (a,m,k,c)
    ),
    CaseSpec(
        "ccsd_t_3", (:i, :j, :m, :c), (:m, :k, :a, :b), (:a, :b, :c, :i, :j, :k),
        :B, (3, 1, 2, 4)  # B is (m,k,a,b); move a (axis 3) to front -> (a,m,k,b)
    ),
    CaseSpec(
        "ccsd_t_4", (:i, :k, :m, :b), (:m, :j, :a, :c), (:a, :b, :c, :i, :j, :k),
        :B, (3, 1, 2, 4)  # B is (m,j,a,c); move a (axis 3) to front -> (a,m,j,c)
    ),
]

# Arm identifiers, in run order. Arm 1 is the adapter path (three backends);
# everything else is the direct plan_contract/execute! API.
const DIRECT_ARMS = ("2", "3", "4M", "4N", "4both", "5")

# dim=16 QuasiStrided calls run ~1-2s each; keep those at 15 reps, everything
# else (dim=8, and dim=16 StridedNative/StridedBLAS) at 21.
function reps_for(dim::Int, arm::AbstractString, backend_name::AbstractString)
    dim == 8 && return 21
    (arm == "1" && backend_name != "QuasiStrided") && return 21
    return 15
end

# Tallies M/N register-slivers of a plan's mgroup/ngroup by whether their
# C-side BlockDescriptor is `regular && stride == 1`.
function tally_group_slivers(group, reg_tile::Int)
    Q = QuasiStrided.axis_length(group)
    buf1 = Vector{Int}(undef, reg_tile)
    buf2 = Vector{Int}(undef, reg_tile)
    n_unit = 0
    n_other = 0
    first_desc = nothing
    first = 0
    while first < Q
        count = min(reg_tile, Q - first)
        (_, dC) = QuasiStrided.block_descriptors!((buf1, buf2), group, first, count)
        if dC.regular && dC.stride == 1
            n_unit += 1
        else
            n_other += 1
        end
        first_desc === nothing && (first_desc = dC)
        first += count
    end
    return (n_unit = n_unit, n_other = n_other, first_desc = first_desc)
end

function print_layout_diagnostic(io::IO, label::String, plan)
    println(io, "  [$label] typeof(plan.Cstorage) = ", typeof(plan.Cstorage))
    MRk = mr(plan.kernel)
    NRk = nr(plan.kernel)
    mt = tally_group_slivers(plan.mgroup, MRk)
    nt = tally_group_slivers(plan.ngroup, NRk)
    println(
        io, "  [$label] M-slivers: regular&&unit-stride = ", mt.n_unit,
        "  other = ", mt.n_other, "  (first C-descriptor: ", mt.first_desc, ")"
    )
    println(
        io, "  [$label] N-slivers: regular&&unit-stride = ", nt.n_unit,
        "  other = ", nt.n_other, "  (first C-descriptor: ", nt.first_desc, ")"
    )
    return (mt = mt, nt = nt)
end

# ----------------------------------------------------------------------------
# Label-order helpers for Arms 4/5
# ----------------------------------------------------------------------------

# Symbol names for a QuasiStrided label tuple, for readable logs.
function label_syms(ind, indA, IA, indB, IB)
    sym = Dict{Int, Symbol}()
    for (l, s) in zip(indA, IA)
        sym[l] = s
    end
    for (l, s) in zip(indB, IB)
        sym[l] = s
    end
    return Tuple(sym[l] for l in ind)
end

# Permutation of one operand's axes that reorders its *free* labels (positive,
# i.e. present in C) by increasing |C-stride|, leaving contracted (negative)
# labels at their original positions. `cstride` maps label -> C stride.
function csorted_perm(ind::NTuple{N, Int}, cstride::Dict{Int, Int}) where {N}
    free_pos = [p for p in 1:N if ind[p] > 0]
    free_sorted = sort(free_pos; by = p -> abs(cstride[ind[p]]))
    perm = collect(1:N)
    for (slot, src) in zip(free_pos, free_sorted)
        perm[slot] = src
    end
    return Tuple(perm)
end

# `permutedims` convention: output axis c takes input axis perm[c].
permute_labels(ind::NTuple{N, Int}, perm::NTuple{N, Int}) where {N} = ntuple(i -> ind[perm[i]], N)

# ----------------------------------------------------------------------------
# Correctness gate
# ----------------------------------------------------------------------------

rtol_for(::Type{Float64}) = 1.0e-10
rtol_for(::Type{Float32}) = 1.0e-4

# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "bench_ccsd_t_store.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_ccsd_t_store.txt")
const PROVENANCE_PATH = joinpath(OUTDIR, "PROVENANCE_ccsd_t_store.txt")

csv_io = open(CSV_PATH, "w")
println(
    csv_io,
    "arm,backend,dtype,case,dim,reps,median_seconds,gflops,rows_unit_stride_frac,cols_stride,indA,indB,swapped"
)
function log_row(
        arm::AbstractString, backend::AbstractString, T::Type, case::AbstractString, dim::Int,
        reps::Int, t::Float64, gf::Float64,
        rows_unit_stride_frac, cols_stride, indA, indB, swapped
    )
    println(
        csv_io,
        "$arm,$backend,$T,$case,$dim,$reps,",
        @sprintf("%.9f", t), ",",
        @sprintf("%.6f", gf), ",",
        rows_unit_stride_frac === nothing ? "" : @sprintf("%.6f", rows_unit_stride_frac), ",",
        cols_stride === nothing ? "" : string(cols_stride), ",",
        indA === nothing ? "" : "\"$(indA)\"", ",",
        indB === nothing ? "" : "\"$(indB)\"", ",",
        swapped === nothing ? "" : string(swapped)
    )
    return flush(csv_io)
end

summary_io = open(SUMMARY_PATH, "w")
print_env_header(summary_io, "bench_ccsd_t_store.jl")
println(summary_io, "smoke = ", SMOKE)
println(summary_io, "dims = ", collect(DIMS))

print_env_header(stdout, "bench_ccsd_t_store.jl")
println("smoke = ", SMOKE)
println("dims = ", collect(DIMS))

function both(msg)
    println(msg)
    return println(summary_io, msg)
end

const rng = MersenneTwister(0xC7_5D_00_03)

canary_results = Float64[]
push!(canary_results, run_canary(rng, "A (start)"))

n_mismatches = 0
n_bitwise_diffs = 0

# (case, dim, dtype, arm-key) -> median seconds, for the end-of-run ratio table.
const RESULTS = Dict{Tuple{String, Int, DataType, String}, Float64}()

# Plans, checks, prints layout diagnostics, times and logs one direct-API arm.
# Returns the output array on success, `nothing` on a correctness mismatch.
function run_direct_arm!(
        arm::AbstractString, case::CaseSpec, dim::Int, ::Type{T},
        Cbase, Cref, Aop::StridedView, indA, Bop::StridedView, indB, indC,
        swapped::Bool, note::AbstractString
    ) where {T}
    both("  [arm$arm] indA=$indA indB=$indB indC=$indC  $note")
    C = copy(Cbase)
    plan = plan_contract(StridedView(C), Aop, indA, Bop, indB, indC)
    execute!(plan, one(T), zero(T))
    ok = isapprox(C, Cref; rtol = rtol_for(T))
    diag = print_layout_diagnostic(stdout, "arm$arm", plan)
    print_layout_diagnostic(summary_io, "arm$arm", plan)
    if !ok
        global n_mismatches += 1
        both("MISMATCH: arm=$arm case=$(case.name) dim=$dim dtype=$T -- ABORTED (not timed)")
        return nothing
    end
    reps = reps_for(dim, arm, "QuasiStrided")
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = reps)
    gf = gflops(T, dim^3, dim, dim^3, t)
    mfrac = diag.mt.n_unit / (diag.mt.n_unit + diag.mt.n_other)
    ncol_stride = diag.nt.first_desc.regular ? diag.nt.first_desc.stride : "irregular"
    log_row(arm, "QuasiStrided-direct", T, case.name, dim, reps, t, gf, mfrac, ncol_stride, indA, indB, swapped)
    RESULTS[(case.name, dim, T, arm)] = t
    both("  arm=$(rpad(arm, 5)) QuasiStrided-direct   reps=$reps  median=$(@sprintf("%.6e", t)) s  gflops=$(@sprintf("%.3f", gf))")
    return C
end

for T in CORRECTNESS_DTYPES
    for dim in DIMS
        for case in CASES
            pA, pB, pAB = TO.contract_indices(case.IA, case.IB, case.IC)

            Aarr = randn(rng, T, dim, dim, dim, dim)
            Barr = randn(rng, T, dim, dim, dim, dim)
            Cbase = zeros(T, dim, dim, dim, dim, dim, dim)  # physical order a,b,c,i,j,k

            # Reference, computed once per (case,dim,dtype) with StridedBLAS.
            Cref = copy(Cbase)
            TO.tensorcontract!(Cref, Aarr, pA, false, Barr, pB, false, pAB, one(T), zero(T), StridedBLAS())

            rtol = rtol_for(T)

            both("\n== case=$(case.name) dim=$dim dtype=$T ==")

            # -----------------------------------------------------------
            # Arm 1: adapter path, one backend at a time.
            # -----------------------------------------------------------
            for (bname, backend) in (
                    ("StridedNative", StridedNative()),
                    ("StridedBLAS", StridedBLAS()),
                    ("QuasiStrided", QuasiStridedBackend()),
                )
                C = copy(Cbase)
                TO.tensorcontract!(C, Aarr, pA, false, Barr, pB, false, pAB, one(T), zero(T), backend)
                ok = isapprox(C, Cref; rtol = rtol)
                if !ok
                    global n_mismatches += 1
                    both("MISMATCH: arm=1 backend=$bname case=$(case.name) dim=$dim dtype=$T -- ABORTED (not timed)")
                    continue
                end
                reps = reps_for(dim, "1", bname)
                t = median_time_s(() -> TO.tensorcontract!(C, Aarr, pA, false, Barr, pB, false, pAB, one(T), zero(T), backend); reps = reps)
                gf = gflops(T, dim^3, dim, dim^3, t)
                log_row("1", bname, T, case.name, dim, reps, t, gf, nothing, nothing, nothing, nothing, nothing)
                RESULTS[(case.name, dim, T, "1-" * bname)] = t
                both("  arm=1 backend=$(rpad(bname, 14)) reps=$reps  median=$(@sprintf("%.6e", t)) s  gflops=$(@sprintf("%.3f", gf))")
            end

            # -----------------------------------------------------------
            # Arm 2: direct API, adapter's own label order (via _qs_labels,
            # the exact function TO.tensorcontract! for QuasiStridedBackend
            # calls internally -- see src/integrations/tensoroperations.jl).
            # -----------------------------------------------------------
            indA2, indB2, indC2 = QuasiStrided._qs_labels(pA, pB, pAB)
            Av = StridedView(Aarr)
            Bv = StridedView(Barr)
            C2 = run_direct_arm!(
                "2", case, dim, T, Cbase, Cref, Av, indA2, Bv, indB2, indC2, false,
                "(adapter's own label order)"
            )

            # -----------------------------------------------------------
            # Arm 3: "C-local" label order control -- permute whichever
            # operand carries `a` (C's stride-1 axis) so that `a` is that
            # operand's own first physical axis. Lazy view, no copy.
            # -----------------------------------------------------------
            if case.arm3_target === :A
                Aperm3 = permutedims(Av, case.arm3_perm)
                indA3 = permute_labels(indA2, case.arm3_perm)
                Bperm3 = Bv
                indB3 = indB2
            else
                Aperm3 = Av
                indA3 = indA2
                Bperm3 = permutedims(Bv, case.arm3_perm)
                indB3 = permute_labels(indB2, case.arm3_perm)
            end
            C3 = run_direct_arm!(
                "3", case, dim, T, Cbase, Cref, Aperm3, indA3, Bperm3, indB3, indC2, false,
                "(permuted operand: $(case.arm3_target), `a` moved to front)"
            )

            # -----------------------------------------------------------
            # Arms 4-M / 4-N / 4both: reorder each operand's free axes by
            # increasing |C-stride| of their label (lazy views).
            # -----------------------------------------------------------
            Cstr = strides(Cbase)
            cstride = Dict{Int, Int}(indC2[i] => Cstr[i] for i in 1:6)
            permA4 = csorted_perm(indA2, cstride)
            permB4 = csorted_perm(indB2, cstride)
            Aperm4 = permutedims(Av, permA4)
            indA4 = permute_labels(indA2, permA4)
            Bperm4 = permutedims(Bv, permB4)
            indB4 = permute_labels(indB2, permB4)
            symsA4 = label_syms(indA4, indA2, case.IA, indB2, case.IB)
            symsB4 = label_syms(indB4, indA2, case.IA, indB2, case.IB)
            both("  [arm4] C strides by label: " * join(("$(label_syms((l,), indA2, case.IA, indB2, case.IB)[1])=$(cstride[l])" for l in indC2), " "))
            both("  [arm4] permA=$permA4 -> A axes $symsA4 ; permB=$permB4 -> B axes $symsB4")

            C4M = run_direct_arm!(
                "4M", case, dim, T, Cbase, Cref, Aperm4, indA4, Bv, indB2, indC2, false,
                "(A's M-axes sorted by |C-stride|)"
            )
            C4N = run_direct_arm!(
                "4N", case, dim, T, Cbase, Cref, Av, indA2, Bperm4, indB4, indC2, false,
                "(B's N-axes sorted by |C-stride|)"
            )
            C4both = run_direct_arm!(
                "4both", case, dim, T, Cbase, Cref, Aperm4, indA4, Bperm4, indB4, indC2, false,
                "(both sorted)"
            )

            # -----------------------------------------------------------
            # Arm 5: 4both + orientation swap when C's stride-1 label is
            # not on A. Swapping the operand slots leaves the contraction
            # itself unchanged (A*B summed over the shared label; real
            # dtypes, so no conjugation to track).
            # -----------------------------------------------------------
            lbl_unit = indC2[argmin(abs.(Cstr))]
            sym_unit = label_syms((lbl_unit,), indA2, case.IA, indB2, case.IB)[1]
            swap5 = !(lbl_unit in indA2)
            if swap5
                C5 = run_direct_arm!(
                    "5", case, dim, T, Cbase, Cref, Bperm4, indB4, Aperm4, indA4, indC2, true,
                    "(both sorted + SWAPPED: original B in M-role slot; C's stride-1 label `$sym_unit` was on B)"
                )
            else
                C5 = run_direct_arm!(
                    "5", case, dim, T, Cbase, Cref, Aperm4, indA4, Bperm4, indB4, indC2, false,
                    "(both sorted, no swap needed: C's stride-1 label `$sym_unit` already on A; repeat of 4both)"
                )
            end

            # Bonus: bitwise equality between the direct-API outputs.
            outs = (("2", C2), ("3", C3), ("4M", C4M), ("4N", C4N), ("4both", C4both), ("5", C5))
            if C2 !== nothing
                parts = String[]
                for (nm, Cx) in outs[2:end]
                    Cx === nothing && continue
                    eq = (Cx == C2)
                    eq || (global n_bitwise_diffs += 1)
                    push!(parts, "arm$nm==arm2:$(eq)")
                end
                both("  [bitwise] " * join(parts, "  "))
            end
        end
    end
end

push!(canary_results, run_canary(rng, "B (middle)"))
push!(canary_results, run_canary(rng, "A' (end)"))
close(csv_io)

canary_spread = relative_spread(canary_results)
println("\ncanary spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
println(summary_io, "\ncanary median times (s): ", canary_results)
println(summary_io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", canary_spread))
both("\ncorrectness mismatches (aborted, not timed): $n_mismatches")
both("bitwise inequalities vs arm 2 (informational): $n_bitwise_diffs")

# ----------------------------------------------------------------------------
# Ratio table: median seconds per arm and ratio vs Arm 1 (QuasiStrided
# backend, the adapter path) and vs Arm 1 StridedBLAS.
# ----------------------------------------------------------------------------

const TABLE_ARMS = ("1-StridedBLAS", "1-QuasiStrided", DIRECT_ARMS...)

function print_ratio_table(io::IO)
    println(io, "\n# median seconds per arm; in brackets: time / Arm-1-QuasiStrided time (lower is better)")
    hdr = rpad("case", 9) * rpad("dim", 4) * rpad("dtype", 8)
    for a in TABLE_ARMS
        hdr *= rpad(a, 22)
    end
    println(io, hdr)
    for T in CORRECTNESS_DTYPES, dim in DIMS, case in CASES
        base = get(RESULTS, (case.name, dim, T, "1-QuasiStrided"), NaN)
        row = rpad(case.name, 9) * rpad(string(dim), 4) * rpad(string(T), 8)
        for a in TABLE_ARMS
            t = get(RESULTS, (case.name, dim, T, a), NaN)
            row *= isnan(t) ? rpad("n/a", 22) : rpad(@sprintf("%.3e [%.3f]", t, t / base), 22)
        end
        println(io, row)
    end
    println(io, "\n# speedup of each arm over Arm-1-QuasiStrided (Arm1-QS time / arm time; >1 is faster), and Arm-1-StridedBLAS time / arm time")
    hdr = rpad("case", 9) * rpad("dim", 4) * rpad("dtype", 8)
    for a in TABLE_ARMS
        hdr *= rpad(a, 22)
    end
    println(io, hdr)
    for T in CORRECTNESS_DTYPES, dim in DIMS, case in CASES
        base = get(RESULTS, (case.name, dim, T, "1-QuasiStrided"), NaN)
        blas = get(RESULTS, (case.name, dim, T, "1-StridedBLAS"), NaN)
        row = rpad(case.name, 9) * rpad(string(dim), 4) * rpad(string(T), 8)
        for a in TABLE_ARMS
            t = get(RESULTS, (case.name, dim, T, a), NaN)
            row *= isnan(t) ? rpad("n/a", 22) : rpad(@sprintf("%.2fx / blas %.2fx", base / t, blas / t), 22)
        end
        println(io, row)
    end
    return nothing
end

print_ratio_table(stdout)
print_ratio_table(summary_io)
close(summary_io)

# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------

function safe_run(cmd)
    return try
        read(cmd, String)
    catch e
        "unavailable ($(sprint(showerror, e)))"
    end
end

open(PROVENANCE_PATH, "w") do io
    println(io, "git_commit = ", git_commit())
    println(io, "command = julia --project=. benchmark/bench_ccsd_t_store.jl ", join(ARGS, " "))
    println(io, "hostname = ", gethostname())
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "TensorOperations = ", pkgversion(TensorOperations))
    println(io, "nthreads = ", NTHREADS, " blas_threads = ", BLAS_THREADS)
    println(io, "dtypes = ", collect(CORRECTNESS_DTYPES))
    println(io, "dims = ", collect(DIMS))
    println(io, "arms = 1 (StridedNative/StridedBLAS/QuasiStrided adapter), ", join(DIRECT_ARMS, ", "), " (direct API)")
    println(io, "reps = 21 (dim=8, and dim=16 StridedNative/StridedBLAS); 15 (dim=16 QuasiStrided-engine rows, noted per-row)")
    println(io, "smoke = ", SMOKE)
    println(io, "date = ", now())
    println(io, "correctness mismatches (aborted, not timed) = ", n_mismatches)
    println(io, "bitwise inequalities vs arm 2 (informational) = ", n_bitwise_diffs)
    println(io, "canary median times (s) = ", canary_results)
    println(io, "canary relative spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
    println(
        io,
        "caveat = single machine ($(gethostname())), single measurement session; ",
        "not averaged across machines or repeated sessions."
    )
    println(io, "\n# Machine-load check at provenance-write time (uptime):")
    println(io, safe_run(`uptime`))
    println(io, "# Machine-load check at provenance-write time (top -bn1 | head -15):")
    println(io, safe_run(pipeline(`top -bn1`, `head -15`)))
end

println("\nDone. Results in ", OUTDIR)
