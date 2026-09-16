# Times four TCCG quantum-chemistry contractions (6-index output, 1
# contracted index) across three "arms":
#
#   ccsd_t_1: C[a,b,c,i,j,k] = A[i,j,m,a] * B[m,k,b,c]
#   ccsd_t_2: C[a,b,c,i,j,k] = A[i,j,m,b] * B[m,k,a,c]
#   ccsd_t_3: C[a,b,c,i,j,k] = A[i,j,m,c] * B[m,k,a,b]
#   ccsd_t_4: C[a,b,c,i,j,k] = A[i,k,m,b] * B[m,j,a,c]
#
#   Arm 1: TO.tensorcontract! under StridedNative()/StridedBLAS()/QuasiStridedBackend().
#   Arm 2: QuasiStrided.plan_contract/execute! directly, with the same label
#          order TO.tensorcontract! derives internally (via _qs_labels).
#   Arm 3: same as Arm 2, but with whichever operand carries `a` (C's
#          stride-1 axis) permuted so `a` is that operand's own first
#          physical axis -- isolates whether label ordering, not the store
#          fast-path, is the lever for this shape class. Effect's sign
#          depends on dim vs the kernel's MR/NR (see per-case perm below).
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

# dim=16 QuasiStrided calls run ~1-2s each; keep those at 15 reps, everything
# else (dim=8, and dim=16 StridedNative/StridedBLAS) at 21.
function reps_for(dim::Int, arm::Int, backend_name::AbstractString)
    dim == 8 && return 21
    (arm == 1 && backend_name != "QuasiStrided") && return 21
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
    "arm,backend,dtype,case,dim,reps,median_seconds,gflops,rows_unit_stride_frac,cols_stride"
)
function log_row(
        arm::Int, backend::AbstractString, T::Type, case::AbstractString, dim::Int,
        reps::Int, t::Float64, gf::Float64,
        rows_unit_stride_frac, cols_stride
    )
    println(
        csv_io,
        "$arm,$backend,$T,$case,$dim,$reps,",
        @sprintf("%.9f", t), ",",
        @sprintf("%.6f", gf), ",",
        rows_unit_stride_frac === nothing ? "" : @sprintf("%.6f", rows_unit_stride_frac), ",",
        cols_stride === nothing ? "" : string(cols_stride)
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

const rng = MersenneTwister(0xC7_5D_00_03)

canary_results = Float64[]
push!(canary_results, run_canary(rng, "A (start)"))

n_mismatches = 0

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

            println(
                "\n== case=$(case.name) dim=$dim dtype=$T =="
            )
            println(summary_io, "\n== case=$(case.name) dim=$dim dtype=$T ==")

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
                    msg = "MISMATCH: arm=1 backend=$bname case=$(case.name) dim=$dim dtype=$T -- ABORTED (not timed)"
                    println(msg)
                    println(summary_io, msg)
                    continue
                end
                reps = reps_for(dim, 1, bname)
                t = median_time_s(() -> TO.tensorcontract!(C, Aarr, pA, false, Barr, pB, false, pAB, one(T), zero(T), backend); reps = reps)
                gf = gflops(T, dim^3, dim, dim^3, t)
                log_row(1, bname, T, case.name, dim, reps, t, gf, nothing, nothing)
                line = "  arm=1 backend=$(rpad(bname, 14)) reps=$reps  median=$(@sprintf("%.6e", t)) s  gflops=$(@sprintf("%.3f", gf))"
                println(line)
                println(summary_io, line)
            end

            # -----------------------------------------------------------
            # Arm 2: direct API, adapter's own label order (via _qs_labels,
            # the exact function TO.tensorcontract! for QuasiStridedBackend
            # calls internally -- see src/tensoroperations.jl).
            # -----------------------------------------------------------
            indA2, indB2, indC2 = QuasiStrided._qs_labels(pA, pB, pAB)
            println("  [arm2] indA=$indA2 indB=$indB2 indC=$indC2")
            println(summary_io, "  [arm2] indA=$indA2 indB=$indB2 indC=$indC2")

            C2 = copy(Cbase)
            plan2 = plan_contract(StridedView(C2), StridedView(Aarr), indA2, StridedView(Barr), indB2, indC2)
            execute!(plan2, one(T), zero(T))
            ok2 = isapprox(C2, Cref; rtol = rtol)
            diag2 = print_layout_diagnostic(stdout, "arm2", plan2)
            print_layout_diagnostic(summary_io, "arm2", plan2)
            if !ok2
                global n_mismatches += 1
                msg = "MISMATCH: arm=2 case=$(case.name) dim=$dim dtype=$T -- ABORTED (not timed)"
                println(msg)
                println(summary_io, msg)
            else
                reps = reps_for(dim, 2, "QuasiStrided")
                t = median_time_s(() -> execute!(plan2, one(T), zero(T)); reps = reps)
                gf = gflops(T, dim^3, dim, dim^3, t)
                mfrac = diag2.mt.n_unit / (diag2.mt.n_unit + diag2.mt.n_other)
                ncol_stride = diag2.nt.first_desc.regular ? diag2.nt.first_desc.stride : "irregular"
                log_row(2, "QuasiStrided-direct", T, case.name, dim, reps, t, gf, mfrac, ncol_stride)
                line = "  arm=2 QuasiStrided-direct   reps=$reps  median=$(@sprintf("%.6e", t)) s  gflops=$(@sprintf("%.3f", gf))"
                println(line)
                println(summary_io, line)
            end

            # -----------------------------------------------------------
            # Arm 3: "C-local" label order control -- permute whichever
            # operand carries `a` (C's stride-1 axis) so that `a` is that
            # operand's own first physical axis.
            # -----------------------------------------------------------
            if case.arm3_target === :A
                Aperm3 = permutedims(StridedView(Aarr), case.arm3_perm)
                indA3 = ntuple(i -> indA2[case.arm3_perm[i]], 4)
                Bperm3 = StridedView(Barr)
                indB3 = indB2
            else
                Aperm3 = StridedView(Aarr)
                indA3 = indA2
                Bperm3 = permutedims(StridedView(Barr), case.arm3_perm)
                indB3 = ntuple(i -> indB2[case.arm3_perm[i]], 4)
            end
            indC3 = indC2

            println("  [arm3] indA=$indA3 indB=$indB3 indC=$indC3  (permuted operand: $(case.arm3_target))")
            println(summary_io, "  [arm3] indA=$indA3 indB=$indB3 indC=$indC3  (permuted operand: $(case.arm3_target))")

            C3 = copy(Cbase)
            plan3 = plan_contract(StridedView(C3), Aperm3, indA3, Bperm3, indB3, indC3)
            execute!(plan3, one(T), zero(T))
            ok3 = isapprox(C3, Cref; rtol = rtol)
            diag3 = print_layout_diagnostic(stdout, "arm3", plan3)
            print_layout_diagnostic(summary_io, "arm3", plan3)
            if !ok3
                global n_mismatches += 1
                msg = "MISMATCH: arm=3 case=$(case.name) dim=$dim dtype=$T -- ABORTED (not timed)"
                println(msg)
                println(summary_io, msg)
            else
                reps = reps_for(dim, 3, "QuasiStrided")
                t = median_time_s(() -> execute!(plan3, one(T), zero(T)); reps = reps)
                gf = gflops(T, dim^3, dim, dim^3, t)
                mfrac = diag3.mt.n_unit / (diag3.mt.n_unit + diag3.mt.n_other)
                ncol_stride = diag3.nt.first_desc.regular ? diag3.nt.first_desc.stride : "irregular"
                log_row(3, "QuasiStrided-direct", T, case.name, dim, reps, t, gf, mfrac, ncol_stride)
                line = "  arm=3 QuasiStrided-direct   reps=$reps  median=$(@sprintf("%.6e", t)) s  gflops=$(@sprintf("%.3f", gf))"
                println(line)
                println(summary_io, line)
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
println(summary_io, "\ncorrectness mismatches (aborted, not timed): ", n_mismatches)
println("\ncorrectness mismatches (aborted, not timed): ", n_mismatches)
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
    println(io, "reps = 21 (dim=8, and dim=16 StridedNative/StridedBLAS); 15 (dim=16 QuasiStrided-engine rows, noted per-row)")
    println(io, "smoke = ", SMOKE)
    println(io, "date = ", now())
    println(io, "correctness mismatches (aborted, not timed) = ", n_mismatches)
    println(io, "canary median times (s) = ", canary_results)
    println(io, "canary relative spread (max-min)/min = ", @sprintf("%.4f", canary_spread))
    println(
        io,
        "caveat = single machine ($(gethostname())), single measurement session; ",
        "not averaged across machines or repeated sessions. See docs/decisions.md, ",
        "\"Store fast-path investigation: Phase A\"."
    )
    println(io, "\n# Machine-load check at provenance-write time (uptime):")
    println(io, safe_run(`uptime`))
    println(io, "# Machine-load check at provenance-write time (top -bn1 | head -15):")
    println(io, safe_run(pipeline(`top -bn1`, `head -15`)))
end

println("\nDone. Results in ", OUTDIR)
