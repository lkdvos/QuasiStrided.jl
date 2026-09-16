# For the four `ccsd_t_*` TCCG quantum-chemistry cases, tallies how many M/N
# micro-tile slivers on the C side are `regular && stride==1` (i.e. would
# satisfy the store fast-path's `_unit_stride_rows` guard) versus not.
#
# Usage: julia --project=. benchmark/probes/probe_ccsd_t_layout.jl [--dim N]
# Writes benchmark/results/<hostname>-<date>/probes_ccsd_t_layout.txt.

using QuasiStrided
using QuasiStrided: plan_contract, mr, nr, block_descriptors!
using StridedViews
using Random

include(joinpath(@__DIR__, "..", "harness.jl"))

const OUTPATH = joinpath(results_dir(), "probes_ccsd_t_layout.txt")
mkpath(dirname(OUTPATH))

io = open(OUTPATH, "w")
out(args...) = (println(io, args...); println(args...))

out("git_commit = ", git_commit())
out("hostname = ", gethostname())
out("julia_version = ", VERSION)
out("date = ", now())
out()

# Manual label assignment matching QuasiStrided's convention (positive =
# open axis matching C's own label numbering in C's order, negative =
# contracted pair). Any consistent labeling is valid; we choose one that
# mirrors C's own axis order 1:1 so indC is just (1,2,...,rank(C)).
#
#   a=1 b=2 c=3 i=4 j=5 k=6   (all open, C = C[a,b,c,i,j,k])
#   m=-1                       (the one contracted label in every equation)
const CASES = [
    (id = "ccsd_t_1", indA = (4, 5, -1, 1), indB = (-1, 6, 2, 3)),  # A=ijma B=mkbc
    (id = "ccsd_t_2", indA = (4, 5, -1, 2), indB = (-1, 6, 1, 3)),  # A=ijmb B=mkac
    (id = "ccsd_t_3", indA = (4, 5, -1, 3), indB = (-1, 6, 1, 2)),  # A=ijmc B=mkab
    (id = "ccsd_t_4", indA = (4, 6, -1, 2), indB = (-1, 5, 1, 3)),  # A=ikmb B=mjac
]
const indC = (1, 2, 3, 4, 5, 6)
const DIM = argopt("dim", 16)

global refuted = false

for T in (Float64, Float32)
    rng = MersenneTwister(0)
    for case in CASES
        A = randn(rng, T, DIM, DIM, DIM, DIM)
        B = randn(rng, T, DIM, DIM, DIM, DIM)
        C = zeros(T, DIM, DIM, DIM, DIM, DIM, DIM)

        plan = plan_contract(
            StridedView(C), StridedView(A), case.indA,
            StridedView(B), case.indB, indC
        )

        out("case=$(case.id) dtype=$T: typeof(plan.Cstorage) = ", typeof(plan.Cstorage))
        MR, NR = mr(plan.kernel), nr(plan.kernel)
        out("case=$(case.id) dtype=$T: mr=$MR nr=$NR")

        Qm = QuasiStrided.axis_length(plan.mgroup)
        Qn = QuasiStrided.axis_length(plan.ngroup)

        m_bufs = (zeros(Int, MR), zeros(Int, MR))
        n_unit_rows = 0
        m_total = 0
        mfirst = 0
        while mfirst < Qm
            mcount = min(MR, Qm - mfirst)
            (dM_A, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
            m_total += 1
            if dM_C.regular && dM_C.stride == 1
                n_unit_rows += 1
            end
            mfirst += mcount
        end
        out(
            "case=$(case.id) dtype=$T: M-slivers checked=$m_total, ",
            "unit-stride (regular && stride==1) = $n_unit_rows"
        )
        if n_unit_rows > 0
            out("!!! unit-stride M-sliver found for case=$(case.id) dtype=$T !!!")
            global refuted = true
        end

        n_bufs = (zeros(Int, NR), zeros(Int, NR))
        n_unit_cols = 0
        n_total = 0
        nfirst = 0
        while nfirst < Qn
            ncount = min(NR, Qn - nfirst)
            (dN_B, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            n_total += 1
            if dN_C.regular && dN_C.stride == 1
                n_unit_cols += 1
            end
            nfirst += ncount
        end
        out(
            "case=$(case.id) dtype=$T: N-slivers checked=$n_total, ",
            "unit-stride (regular && stride==1) = $n_unit_cols"
        )
        if n_unit_cols > 0
            out("!!! unit-stride N-sliver found for case=$(case.id) dtype=$T !!!")
            global refuted = true
        end

        # Also print the actual first M-sliver's C-side descriptor verbatim,
        # so the raw evidence (not just the pass/fail tally) is on record.
        (dM_A0, dM_C0) = block_descriptors!(m_bufs, plan.mgroup, 0, min(MR, Qm))
        out("case=$(case.id) dtype=$T: first M-sliver C descriptor = ", dM_C0)
        (dN_B0, dN_C0) = block_descriptors!(n_bufs, plan.ngroup, 0, min(NR, Qn))
        out("case=$(case.id) dtype=$T: first N-sliver C descriptor = ", dN_C0)
        out()
    end
end

out("="^72)
if refuted
    out("VERDICT: at least one unit-stride sliver found -- the store fast-path")
    out("guard could reach some of these cases.")
else
    out("VERDICT: zero unit-stride M/N slivers across all cases/dtypes -- the")
    out("store fast-path guard cannot help any of them.")
end

close(io)
println("\nWrote ", OUTPATH)
