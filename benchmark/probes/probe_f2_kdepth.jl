# T2b evidence-gathering probe: how big is the F2 run-length demotion's
# regression (`_demote_for_run`, src/planning/kernel_selection.jl, fired unconditionally
# from its two call sites at src/planning/kernel_selection.jl/1074) across a range of
# `Qk` ("K-depth", the contracted extent `m` in the fixture below)?
#
# T1 (this milestone, earlier in this session) already confirmed, on this
# exact fixture family, that F2 fires at EVERY `(T, a, m)` combination
# regardless of `m` -- i.e. it never declines to demote. This script does not
# add a K-depth guard and draws no go/no-go conclusion; it only measures the
# three arms below so a downstream gate can read `r = t_demoted / t_default`
# against fixed rules.
#
#   julia -t 1 --project=benchmark benchmark/probes/probe_f2_kdepth.jl
#
# Fixture: A is (i,j,m,a), B is (m,k,b,c), C is (a,b,c,i,j,k), with
# i=j=k=b=c=6 fixed and (a, m) swept. Free extents seen by `plan_contract`
# are therefore Qm = a*36 (i*j=36) and Qn = 216 (b*c*k=6*6*6=216), both
# independent of `m` -- only Qk = m depends on `m`. That in turn means the
# swap decision (`_prefer_swap`) and the demoted shape are themselves
# independent of `m` for a fixed `(T, a)`; this is verified explicitly below
# (per-point, not assumed) by re-deriving `plan.kernel`'s type at every `m`
# and comparing it against the type recorded at the first `m` of the sweep.
#
# Three arms per `(T, a, m)` point, timed interleaved (not
# all-of-arm-1-then-all-of-arm-2), each on its own warmed `ContractPlan` via
# `execute!` (never `plan_contract` itself, which is comparatively cheap and
# would just add noise):
#   1. auto           -- plan_contract(...), no `kernel=`; F2 fires as today.
#   2. forced-default -- plan_contract(...; kernel = _default_kernel(T, Qm,
#                         Qn)); explicit `kernel=` makes both call sites use
#                         this kernel verbatim (see src/planning/plan.jl,
#                         1065-1067/1073-1075), bypassing F2 entirely.
#   3. forced-demoted -- plan_contract(...; kernel = <whatever `auto` chose>),
#                         i.e. F2's own pick, forced so its cost can be
#                         measured without re-deciding the swap/demotion.

include(joinpath(@__DIR__, "..", "harness.jl"))

using QuasiStrided: _default_kernel
using StridedViews: StridedView
using Statistics: median
using Printf
using Random

const REPS = 15
const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "f2_kdepth.csv")
const PROV_PATH = joinpath(OUTDIR, "f2_kdepth_PROVENANCE.txt")

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

function fixture(::Type{T}, a::Int, m::Int, rng) where {T}
    i = j = k = b = c = 6
    A = rand(rng, T, i, j, m, a)
    B = rand(rng, T, m, k, b, c)
    C = zeros(T, a, b, c, i, j, k)
    indA = (1, 2, 3, 4)         # i,j,m,a
    indB = (3, 5, 6, 7)         # m,k,b,c
    indC = (4, 6, 7, 1, 2, 5)   # a,b,c,i,j,k
    return StridedView(C), StridedView(A), indA, StridedView(B), indB, indC
end

kernel_shape_str(kernel) = "($(mr(kernel)),$(nr(kernel)),$(lanewidth(kernel)))"

# ---------------------------------------------------------------------------
# One measured point: (T, a, m). `default_kernel`/`demoted_kernel` are passed
# in (computed once per (T,a), reused across the m-sweep, per the docstring
# above); `verify_kernel_type` is the type recorded for `demoted_kernel` at
# the first m of the sweep, used to check m-independence at every point.
# ---------------------------------------------------------------------------

function measure_point(
        T::Type, a::Int, m::Int, default_kernel, demoted_kernel,
        verify_kernel_type, rng; check_numeric::Bool = false
    )
    Cv, Av, indA, Bv, indB, indC = fixture(T, a, m, rng)

    plan_auto = plan_contract(Cv, Av, indA, Bv, indB, indC)
    plan_default = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = default_kernel)
    plan_demoted = plan_contract(Cv, Av, indA, Bv, indB, indC; kernel = demoted_kernel)

    kernel_type_ok = typeof(plan_auto.kernel) === verify_kernel_type
    matches_default = typeof(plan_auto.kernel) === typeof(plan_default.kernel)
    matches_demoted = typeof(plan_auto.kernel) === typeof(plan_demoted.kernel)

    # Warm up (compile + discard) before any timing.
    execute!(plan_auto, 1.0, 0.0)
    execute!(plan_default, 1.0, 0.0)
    execute!(plan_demoted, 1.0, 0.0)

    numeric_ok = true
    if check_numeric
        C1 = copy(parent(Cv))
        execute!(plan_auto, 1.0, 0.0)
        Cref = copy(parent(Cv))
        execute!(plan_default, 1.0, 0.0)
        Cdef = copy(parent(Cv))
        execute!(plan_demoted, 1.0, 0.0)
        Cdem = copy(parent(Cv))
        numeric_ok = isapprox(Cref, Cdef) && isapprox(Cref, Cdem)
    end

    ts_auto = Vector{Float64}(undef, REPS)
    ts_default = Vector{Float64}(undef, REPS)
    ts_demoted = Vector{Float64}(undef, REPS)
    for r in 1:REPS
        t0 = time_ns()
        execute!(plan_auto, 1.0, 0.0)
        t1 = time_ns()
        execute!(plan_default, 1.0, 0.0)
        t2 = time_ns()
        execute!(plan_demoted, 1.0, 0.0)
        t3 = time_ns()
        ts_auto[r] = (t1 - t0) / 1.0e9
        ts_default[r] = (t2 - t1) / 1.0e9
        ts_demoted[r] = (t3 - t2) / 1.0e9
    end

    t_auto = median(ts_auto)
    t_default = median(ts_default)
    t_demoted = median(ts_demoted)
    r_ratio = t_demoted / t_default
    matched_arm = abs(t_auto - t_default) < abs(t_auto - t_demoted) ? "default" : "demoted"

    return (
        dtype = string(T), a = a, m = m, Qm = a * 36, Qn = 216, Qk = m,
        default_shape = kernel_shape_str(default_kernel),
        demoted_shape = kernel_shape_str(demoted_kernel),
        kernel_type_matches_first_m = kernel_type_ok,
        auto_matches_default_type = matches_default,
        auto_matches_demoted_type = matches_demoted,
        t_auto = t_auto, t_default = t_default, t_demoted = t_demoted,
        r = r_ratio, auto_matches = matched_arm,
        numeric_checked = check_numeric, numeric_ok = numeric_ok,
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    print_env_header(stdout, "probes/probe_f2_kdepth.jl")
    println("REPS = ", REPS)

    csv = open(CSV_PATH, "w")
    println(
        csv,
        "dtype,a,m,Qm,Qn,Qk,default_shape,demoted_shape,kernel_type_matches_first_m," *
            "auto_matches_default_type,auto_matches_demoted_type,t_auto_s,t_default_s," *
            "t_demoted_s,r,auto_matches,numeric_checked,numeric_ok"
    )

    canaries = Float64[]
    crng = Random.MersenneTwister(0x0F2_1CADE)
    push!(canaries, run_canary(crng, "start"))

    ms = [8, 16, 32, 64, 128, 256, 512]
    dtype_specs = [
        (Float64, [8, 40]),
        (Float32, [16, 80]),
    ]

    all_rows = NamedTuple[]
    extra_rows = NamedTuple[]  # the 2 extra reps of Float64,a=8,m=512

    rng = Random.MersenneTwister(0x0F2B00B5)

    for (T, avals) in dtype_specs
        for a in avals
            Qm = a * 36
            Qn = 216
            default_kernel = _default_kernel(T, Qm, Qn)

            # Reference m to establish the demoted kernel + its type, per the
            # m-independence claim in the header comment.
            m0 = ms[1]
            Cv0, Av0, indA0, Bv0, indB0, indC0 = fixture(T, a, m0, rng)
            plan0 = plan_contract(Cv0, Av0, indA0, Bv0, indB0, indC0)
            demoted_kernel = plan0.kernel
            verify_kernel_type = typeof(plan0.kernel)

            println(
                "\n=== T=$T a=$a  Qm=$Qm Qn=$Qn  default_shape=$(kernel_shape_str(default_kernel))" *
                    "  demoted_shape=$(kernel_shape_str(demoted_kernel)) ==="
            )
            if kernel_shape_str(default_kernel) == kernel_shape_str(demoted_kernel)
                @warn "T=$T a=$a: auto's kernel at m=$m0 has the SAME shape as " *
                    "_default_kernel(T,Qm,Qn) -- F2 did NOT demote here, contrary " *
                    "to the T1 finding. Flagging loudly per task instructions."
            end

            for (i, m) in enumerate(ms)
                check_numeric = i == 1  # once per (T,a), per task ask
                row = measure_point(
                    T, a, m, default_kernel, demoted_kernel, verify_kernel_type, rng;
                    check_numeric = check_numeric
                )
                if !row.kernel_type_matches_first_m
                    @warn "T=$T a=$a m=$m: auto's kernel TYPE differs from the " *
                        "one recorded at m=$m0 -- the m-independence assumption " *
                        "is FALSE at this point. Flagging loudly."
                end
                if check_numeric && !row.numeric_ok
                    @warn "T=$T a=$a m=$m: numeric mismatch between arms!"
                end
                @printf(
                    "  m=%-4d  t_auto=%.3e  t_default=%.3e  t_demoted=%.3e  r=%.4f  auto~%s  num_ok=%s\n",
                    m, row.t_auto, row.t_default, row.t_demoted, row.r,
                    row.auto_matches, check_numeric ? string(row.numeric_ok) : "n/a"
                )
                push!(all_rows, row)
                println(
                    csv,
                    "$(row.dtype),$(row.a),$(row.m),$(row.Qm),$(row.Qn),$(row.Qk)," *
                        "$(row.default_shape),$(row.demoted_shape),$(row.kernel_type_matches_first_m)," *
                        "$(row.auto_matches_default_type),$(row.auto_matches_demoted_type)," *
                        "$(row.t_auto),$(row.t_default),$(row.t_demoted),$(row.r)," *
                        "$(row.auto_matches),$(row.numeric_checked),$(row.numeric_ok)"
                )

                # The specific regression point (Float64, a=8, m=512): 2 EXTRA
                # independent reps, per task ask.
                if T === Float64 && a == 8 && m == 512
                    for rep in 1:2
                        erow = measure_point(
                            T, a, m, default_kernel, demoted_kernel, verify_kernel_type, rng;
                            check_numeric = false
                        )
                        push!(extra_rows, erow)
                        @printf(
                            "  [extra rep %d] m=%-4d  t_auto=%.3e  t_default=%.3e  t_demoted=%.3e  r=%.4f  auto~%s\n",
                            rep, m, erow.t_auto, erow.t_default, erow.t_demoted, erow.r,
                            erow.auto_matches
                        )
                        println(
                            csv,
                            "$(erow.dtype),$(erow.a),$(erow.m)_extra$(rep),$(erow.Qm),$(erow.Qn),$(erow.Qk)," *
                                "$(erow.default_shape),$(erow.demoted_shape),$(erow.kernel_type_matches_first_m)," *
                                "$(erow.auto_matches_default_type),$(erow.auto_matches_demoted_type)," *
                                "$(erow.t_auto),$(erow.t_default),$(erow.t_demoted),$(erow.r)," *
                                "$(erow.auto_matches),$(erow.numeric_checked),$(erow.numeric_ok)"
                        )
                    end
                end
            end
            push!(canaries, run_canary(crng, "after-$(T)-a$(a)"))
        end
    end

    close(csv)

    spread = relative_spread(canaries)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    if spread > 0.15
        @warn "canary spread exceeds 15%: the machine was not quiet enough for " *
            "a confident ratio; re-run before trusting these numbers as final."
    end

    machine_load = try
        strip(read(`uptime`, String))
    catch
        "unknown (uptime failed)"
    end
    println("machine load (uptime): ", machine_load)

    open(PROV_PATH, "w") do io
        print_env_header(io, "probes/probe_f2_kdepth.jl")
        println(io, "reps = ", REPS)
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
        println(io, "machine load (uptime, at provenance-write time) = ", machine_load)
        println(io, "\n--- table (see f2_kdepth.csv for the machine-readable form) ---")
        for row in all_rows
            @printf(
                io,
                "%-10s a=%-4d m=%-4d Qm=%-5d Qn=%-4d  default=%-12s demoted=%-12s  t_auto=%.4e t_default=%.4e t_demoted=%.4e  r=%.4f  auto~%s  kernel_type_ok=%s  numeric_checked=%s numeric_ok=%s\n",
                row.dtype, row.a, row.m, row.Qm, row.Qn, row.default_shape, row.demoted_shape,
                row.t_auto, row.t_default, row.t_demoted, row.r, row.auto_matches,
                row.kernel_type_matches_first_m, row.numeric_checked, row.numeric_ok
            )
        end
        println(io, "\n--- extra reps: Float64, a=8, m=512 (the specific prior-pass regression point) ---")
        for row in extra_rows
            @printf(
                io,
                "%-10s a=%-4d m=%-4d  t_auto=%.4e t_default=%.4e t_demoted=%.4e  r=%.4f  auto~%s\n",
                row.dtype, row.a, row.m, row.t_auto, row.t_default, row.t_demoted, row.r,
                row.auto_matches
            )
        end
    end

    println("\nWrote ", CSV_PATH)
    println("Wrote ", PROV_PATH)
    return nothing
end

main()
