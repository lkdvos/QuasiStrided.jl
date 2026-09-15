# T1 probe: for the four `ccsd_t_*` regression cases (from the upstream TCCG
# quantum-chemistry benchmark, dim=16), tally how many M/N micro-tile slivers
# on the C side are `regular && stride==1` (i.e. would satisfy
# `src/kernels/simd.jl:145` `_unit_stride_rows`) versus not. Prediction E3
# (docs/decisions.md, "Store fast-path investigation: Phase A") is that this
# count is ZERO for every case/dtype -- i.e. even a fully-working vectorized
# store fast path could not help these cases, because their destination rows
# are never unit-stride to begin with.
#
# Usage: julia --project=. benchmark/probes/probe_ccsd_t_layout.jl
# Writes benchmark/results/<hostname>-<date>/probes_T1_ccsd_t_layout.txt.

using QuasiStrided
using QuasiStrided: plan_contract, mr, nr, block_descriptors!
using StridedViews
using Random
using Dates

const OUTDIR = joinpath(
    @__DIR__, "..", "results", "$(gethostname())-$(Dates.format(now(), "yyyy-mm-dd"))"
)
mkpath(OUTDIR)
const OUTPATH = joinpath(OUTDIR, "probes_T1_ccsd_t_layout.txt")

io = open(OUTPATH, "w")
out(args...) = (println(io, args...); println(args...))

commit = try
    strip(read(`git -C $(joinpath(@__DIR__, "..", "..")) rev-parse HEAD`, String))
catch
    "unknown"
end
out("git_commit = ", commit)
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
const DIM = 16

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
            out("!!! E3 REFUTED (M side) for case=$(case.id) dtype=$T !!!")
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
            out("!!! E3 REFUTED (N side) for case=$(case.id) dtype=$T !!!")
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

out("=" ^ 72)
if refuted
    out("VERDICT: E3 REFUTED -- at least one unit-stride sliver found. The")
    out("attribution must be redone: some of the regression may be reachable")
    out("by fixing the store fast-path guard.")
else
    out("VERDICT: E3 CONFIRMED -- zero unit-stride M/N slivers found across")
    out("all 4 equations x 2 dtypes. Fixing the store fast-path guard cannot")
    out("account for any part of the ccsd_t_* regression.")
end

close(io)
println("\nWrote ", OUTPATH)
