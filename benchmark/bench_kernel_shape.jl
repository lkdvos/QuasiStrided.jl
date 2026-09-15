# Register-blocking sweep: (MR, NR, W) x kc for SIMDKernel.
#
# kc is swept jointly with the shape, never held fixed: MR*kc*sizeof(T) is the
# A-micropanel L1 footprint, so doubling MR at fixed kc measures the wrong
# thing. Rationale and results in docs/decisions.md, Phases G and H.
#
#   julia --project=. benchmark/bench_kernel_shape.jl

include(joinpath(@__DIR__, "harness.jl"))

using InteractiveUtils: code_native

# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

# (MR, NR, W). Budget: NV = (MR/W)*NR accumulators + MR/W live A vectors,
# against 32 registers on AVX-512 (the count comes from AVX512VL, not the lane
# width in use). `spill_report` below is the ground truth. BLIS's own choices,
# harvested from the blis_jll 2.0.0+2 artifact: skx MR_d=16 NR_d=14,
# MR_s=32 NR_s=12; haswell MR_d=8 NR_d=6 (= this package's old default).
const CANDIDATES_F64 = [
    (8, 6, 4),    # control: today's default (= BLIS haswell dgemm 8x6)
    # (16,6,4): NV=24 at 256-bit lanes, kept to separate "more accumulators"
    # from "wider accumulators". It does NOT spill here -- 32 ymm registers
    # via AVX512VL, see the note above.
    (16, 6, 4),
    (8, 12, 4),   # bigger tile, 256-bit lanes: isolates tile size from width
    (8, 6, 8),    # 512-bit lanes, small tile: isolates width from tile size
    (16, 6, 8),
    (16, 8, 8),
    (16, 12, 8),
    (16, 14, 8),  # BLIS skx dgemm family
    (24, 8, 8),
    (32, 6, 8),
    (8, 14, 8),   # wide/short: better padding behaviour at small M
]

const CANDIDATES_F32 = [
    (8, 6, 8),    # control: today's default
    (16, 12, 8),  # 256-bit, bigger tile
    (16, 6, 16),
    (16, 12, 16),
    (16, 14, 16),
    (32, 12, 16), # BLIS skx sgemm family
    (32, 6, 16),
    (48, 8, 16),
]

candidates_for(::Type{Float64}) = CANDIDATES_F64
candidates_for(::Type{Float32}) = CANDIDATES_F32

# mc/nc pinned at the Stage 0 re-measured defaults; only kc moves with MR.
const PINNED_F64 = (mc = 64, nc = 1536)
const PINNED_F32 = (mc = 96, nc = 1152)
pinned_for(::Type{Float64}) = PINNED_F64
pinned_for(::Type{Float32}) = PINNED_F32

const KC_BASE_F64 = 128
const KC_BASE_F32 = 384
kc_base(::Type{Float64}) = KC_BASE_F64
kc_base(::Type{Float32}) = KC_BASE_F32

# The measured default; one holding MR*kc*sizeof(T) at the control's 8 KiB A
# micropanel; and 2x, since top-ranked points sit at the largest kc.
function kc_values(::Type{T}, MR::Int) where {T}
    base = kc_base(T)
    panel = max(8, round(Int, base * 8 / MR) & ~7)
    return unique((base, panel, 2 * base))
end

# ---------------------------------------------------------------------------
# Spill detection
# ---------------------------------------------------------------------------

# A stack store has the [rsp/rbp] operand first, i.e. followed by a comma.
# The match must be pinned to the accumulator's own width: matching any width
# counts the 16-byte GC-frame store every one of these functions emits, and
# reports a phantom spill even at 6 accumulators.
# Matches BOTH `vfmadd*` and `vfnmadd*`. The negated form is not an
# alternative spelling: a planar complex kernel issues exactly `MV*NR` of them
# per K step (the `-Ai*Bi` term of the real output plane) against `3*MV*NR`
# plain ones, so a bare `r"vfmadd"` -- which does NOT match the substring
# `vfnmadd` -- would undercount planar's FMAs by a quarter and make the
# "fmas should equal nv" check below read as a spurious shortfall.
const FMA_RE = r"vfn?madd"

function stack_store_re(vector_bytes::Int)
    word = vector_bytes == 64 ? "zmmword" :
        vector_bytes == 32 ? "ymmword" : "xmmword"
    return Regex(
        "v(?:mov(?:up|ap)[sd]|movdq[ua])\\s+" * word *
            "\\s+ptr\\s+\\[r[sb]p[^\\]]*\\]\\s*,"
    )
end

# `spills == 0` passes; `fmas` should equal `nv` (one FMA per accumulator per
# K step) -- a shortfall means LLVM restructured the body.
function spill_report(kernel::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W}
    nv = (MR ÷ W) * NR
    acc = QuasiStrided.zero_accumulator(kernel)
    io = IOBuffer()
    try
        code_native(
            io, Base.accumulate,
            (typeof(kernel), typeof(acc), Vector{T}, Vector{T}, Int);
            debuginfo = :none, syntax = :intel
        )
    catch err
        return (spills = -1, fmas = -1, nv = nv, ok = false, err = string(err))
    end
    asm = String(take!(io))
    spills = count(stack_store_re(W * sizeof(T)), asm)
    fmas = count(FMA_RE, asm)
    return (spills = spills, fmas = fmas, nv = nv, ok = true, err = "")
end

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

function print_header(io::IO)
    print_env_header(io, "bench_kernel_shape.jl")
    for T in DTYPES
        println(io, "candidates(", T, ") = ", candidates_for(T))
        println(io, "pinned(", T, ") = ", pinned_for(T), "  kc_base = ", kc_base(T))
    end
    return nothing
end
print_header(stdout)

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "bench_kernel_shape.csv")
const SPILL_PATH = joinpath(OUTDIR, "kernel_shape_spills.csv")
const SUMMARY_PATH = joinpath(OUTDIR, "summary_kernel_shape.txt")
const CANARY_PATH = joinpath(OUTDIR, "canary_kernel_shape.csv")
const PROVENANCE_PATH = joinpath(OUTDIR, "PROVENANCE_kernel_shape.txt")

const REPS = 9

csv_io = open(CSV_PATH, "w")
println(csv_io, "dtype,shape,Ma,Ka,Na,MR,NR,W,NV,mc,kc,nc,reps,median_seconds,allocated_bytes")

function log_row(T, spec, MR, NR, W, mc, kc, nc, t, alloc)
    nv = (MR ÷ W) * NR
    println(
        csv_io,
        "$T,$(spec.name),$(spec.Ma),$(spec.Ka),$(spec.Na),$MR,$NR,$W,$nv,",
        "$mc,$kc,$nc,$REPS,", @sprintf("%.9f", t), ",$alloc"
    )
    return flush(csv_io)
end

# ---------------------------------------------------------------------------
# Spill pass (no timing; cheap and decides how to read the timings)
# ---------------------------------------------------------------------------

# "0 spills" only means something if the detector can see one at all.
function assert_spill_detector_works()
    for (MR, NR, T, W) in ((16, 16, Float64, 4), (32, 32, Float32, 16))
        rep = spill_report(SIMDKernel(Val(MR), Val(NR), T, Val(W)))
        rep.spills > 0 || error(
            "spill detector is broken: ($MR,$NR,$T,W=$W) has NV=$(rep.nv) " *
                "accumulators, which cannot fit 32 registers, yet reported " *
                "$(rep.spills) spills"
        )
        @info "spill detector self-test" shape = (MR, NR, T, W) NV = rep.nv spills = rep.spills
    end
    return nothing
end
assert_spill_detector_works()

spill_rows = NamedTuple[]
open(SPILL_PATH, "w") do io
    println(io, "dtype,MR,NR,W,NV,spills,fmas,codegen_ok,note")
    for T in DTYPES, (MR, NR, W) in candidates_for(T)
        kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
        rep = spill_report(kernel)
        push!(spill_rows, (dtype = T, MR = MR, NR = NR, W = W, rep = rep))
        println(
            io, "$T,$MR,$NR,$W,$(rep.nv),$(rep.spills),$(rep.fmas),",
            "$(rep.ok),", replace(rep.err, ',' => ';')
        )
        @info "spill check" dtype = T shape = (MR, NR, W) NV = rep.nv spills = rep.spills fmas = rep.fmas
    end
end

# ---------------------------------------------------------------------------
# Timing sweep
# ---------------------------------------------------------------------------

rng = MersenneTwister(0xB3_C4_0002)
raw = NamedTuple[]
canary_results = Float64[]
push!(canary_results, run_canary(rng, "A (start)"))

const SWEEP_SHAPES = vcat(MAIN_SHAPES, SMALL_SHAPES)

function measure!(raw, T, spec, fx, kernel, MR, NR, W, mc, kc, nc)
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = kernel, mc = mc, kc = kc, nc = nc
    )
    execute!(plan, one(T), zero(T))             # warm up before @allocated
    alloc = @allocated execute!(plan, one(T), zero(T))
    t = median_time_s(() -> execute!(plan, one(T), zero(T)); reps = REPS)
    log_row(T, spec, MR, NR, W, mc, kc, nc, t, alloc)
    push!(
        raw, (
            kernel = "SIMDKernel", dtype = T, shape = spec.name,
            MR = MR, NR = NR, W = W, kc = kc, t = t, alloc = alloc,
        )
    )
    return t
end

for T in DTYPES
    pin = pinned_for(T)
    for (MR, NR, W) in candidates_for(T)
        kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
        for kc in kc_values(T, MR)
            for spec in SWEEP_SHAPES
                fx = build_plain(T, spec, rng)
                measure!(raw, T, spec, fx, kernel, MR, NR, W, pin.mc, kc, pin.nc)
            end
            # scattered / 3-index fixture: the store path that cannot
            # vectorize, so a larger register tile may behave differently.
            fxs = build_scattered(T, rng)
            spec_s = ShapeSpec("scattered_a64k64b16n64", 64, 64, 64)
            measure!(raw, T, spec_s, fxs, kernel, MR, NR, W, pin.mc, kc, pin.nc)
            @info "sweep progress" dtype = T shape = (MR, NR, W) kc = kc
        end
    end
    push!(canary_results, run_canary(rng, "mid ($T done)"))
end

push!(canary_results, run_canary(rng, "A' (end)"))
close(csv_io)

spread = relative_spread(canary_results)
open(CANARY_PATH, "w") do io
    println(io, "index,median_seconds")
    for (i, t) in enumerate(canary_results)
        println(io, "$i,", @sprintf("%.9f", t))
    end
    println(io, "# relative spread (max-min)/min = ", @sprintf("%.4f", spread))
end
println("canary spread (max-min)/min = ", @sprintf("%.4f", spread))

# ---------------------------------------------------------------------------
# Ranking and summary
# ---------------------------------------------------------------------------

# MAIN_SHAPES decide; SMALL_SHAPES are reported apart so a large-shape win
# cannot hide a small-extent padding regression.
const MAIN_NAMES = Set(s.name for s in MAIN_SHAPES)
const SMALL_NAMES = Set(s.name for s in SMALL_SHAPES)

shape_key(r) = (r.MR, r.NR, r.W, r.kc)
nv_of(key) = (key[1] ÷ key[3]) * key[2]

control_for(::Type{Float64}) = (8, 6, 4)
control_for(::Type{Float32}) = (8, 6, 8)

open(SUMMARY_PATH, "w") do io
    print_header(io)
    println(io, "\ncanary median times (s): ", canary_results)
    println(io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", spread))

    println(io, "\n== spill report (spills must be 0; fmas should equal NV) ==")
    for r in spill_rows
        println(
            io, "  ", r.dtype, " (MR,NR,W)=", (r.MR, r.NR, r.W),
            " NV=", r.rep.nv, " spills=", r.rep.spills, " fmas=", r.rep.fmas,
            r.rep.ok ? "" : "  CODEGEN FAILED: " * r.rep.err
        )
    end

    for T in DTYPES
        main_rows = filter(r -> r.dtype == T && r.shape in MAIN_NAMES, raw)
        ranked = rank_by(main_rows, T, shape_key)

        println(io, "\n== $T ranking on MAIN_SHAPES ((MR,NR,W,kc) => geomean ratio) ==")
        for (key, g) in ranked
            println(io, "  ", key, " NV=", nv_of(key), " => ", @sprintf("%.4f", g))
        end

        winner, wg = choose_within_noise(ranked, nv_of)
        best_key, best_g = ranked[1]
        println(
            io, "chosen $T shape: (MR,NR,W,kc) = ", winner,
            " (geomean ", @sprintf("%.4f", wg), "; unconstrained best ",
            best_key, " at ", @sprintf("%.4f", best_g), ")"
        )

        # Speedup of the best candidate over the control, per shape, so the
        # decision is reported in wall-clock terms and not only as a ratio.
        ctrl = control_for(T)
        println(io, "\n-- $T per-shape time vs control $(ctrl) (>1 means candidate is faster) --")
        for name in vcat(
                [s.name for s in MAIN_SHAPES], [s.name for s in SMALL_SHAPES],
                ["scattered_a64k64b16n64"]
            )
            rows = filter(r -> r.dtype == T && r.shape == name, raw)
            isempty(rows) && continue
            cbest = minimum(
                r.t for r in rows
                    if (r.MR, r.NR, r.W) == ctrl; init = Inf
            )
            wbest = minimum(
                r.t for r in rows
                    if (r.MR, r.NR, r.W) == (winner[1], winner[2], winner[3]); init = Inf
            )
            tag = name in SMALL_NAMES ? " [small-extent]" :
                (name in MAIN_NAMES ? "" : " [scattered]")
            println(
                io, "  ", rpad(name, 26), " control ", @sprintf("%.6e", cbest),
                "  chosen ", @sprintf("%.6e", wbest),
                "  speedup ", @sprintf("%.3f", cbest / wbest), tag
            )
        end

        println(io, "\n-- $T allocation per execute! (Julia $(VERSION)) --")
        for (MR, NR, W) in candidates_for(T)
            rows = filter(r -> r.dtype == T && (r.MR, r.NR, r.W) == (MR, NR, W), raw)
            isempty(rows) && continue
            println(
                io, "  (MR,NR,W)=", (MR, NR, W), " NV=", (MR ÷ W) * NR,
                " max allocated = ", maximum(r.alloc for r in rows), " B"
            )
        end
    end
end
println(read(SUMMARY_PATH, String))

open(PROVENANCE_PATH, "w") do io
    println(io, "git_commit = ", git_commit())
    println(io, "command = julia --project=. benchmark/bench_kernel_shape.jl")
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "nthreads = ", NTHREADS, " blas_threads = ", BLAS_THREADS)
    println(io, "date = ", now())
    println(io, "reps = ", REPS)
    println(io, "candidates_f64 = ", CANDIDATES_F64)
    println(io, "candidates_f32 = ", CANDIDATES_F32)
    println(io, "pinned_f64 = ", PINNED_F64, "  kc_base_f64 = ", KC_BASE_F64)
    println(io, "pinned_f32 = ", PINNED_F32, "  kc_base_f32 = ", KC_BASE_F32)
    println(io, "main_shapes = ", [s.name for s in MAIN_SHAPES])
    println(io, "small_shapes = ", [s.name for s in SMALL_SHAPES])
    println(io, "canary_spread = ", @sprintf("%.4f", spread))
    println(io, "blis_reference = blis_jll 2.0.0+2 artifact share/blis/config/*/bli_kernel_defs_*.h")
end

println("\nDone. Results in ", OUTDIR)
