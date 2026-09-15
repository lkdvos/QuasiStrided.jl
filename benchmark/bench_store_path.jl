# Store fast-path investigation (docs/decisions.md, "Store fast-path
# investigation: Phase A") -- task T2.
#
# Tile-level cost of the scattered-store path (`_store_tile_scattered!`,
# src/kernels/simd.jl) vs. the vectorized fast path in `store_tile!`
# (src/kernels/simd.jl:217, `_unit_stride_rows(destination.rows) &&
# destination.storage isa Vector{T}` -- analytically dead on Julia >= 1.11
# per E1, since the real driver hands the kernel `Memory{T}`), across four
# destination-storage variants plus MR-tail variants of the two that matter
# for the guard (`Vector{T}` vs `Memory{T}`, otherwise-identical geometry).
#
# Also attempts to reproduce the "SIMDKernel reaches 101-103 GFLOP/s, ~88% of
# peak" accumulate-only claim (docs/decisions.md, "NV is held at 12
# deliberately", Float64, kc=256, packed panel "as plain Vector").
#
# Report-only: builds no fix. See docs/decisions.md's "Decision boundaries"
# for what this feeds into.
#
#   julia --project=. benchmark/bench_store_path.jl

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: SIMDKernel, mr, nr, lanewidth, scalartype,
    packed_a_length, packed_b_length, packed_a_offset, packed_b_offset,
    AffineAxis, DestinationTile, zero_accumulator, accumulate, store_tile!,
    execute_tile!, PackedPanel, packed_panel
using InteractiveUtils: code_native

# ---------------------------------------------------------------------------
# Measurement constants
# ---------------------------------------------------------------------------

const REPS = 21
const BATCH = 50_000          # calls per timed rep, for median_time_s
const ALLOC_BATCH = 100       # calls per @allocated probe (after warm-up)

# "Cold" destination: cycle the tile's base across COLD_NPOS positions spaced
# COLD_STEP elements apart.
#
# DEVIATION FROM THE TASK TEXT'S LITERAL "e.g. cycle through 4096 distinct
# tile positions" reading: the task's own D-strided-hot/cold geometry (row
# stride 4096, col stride 4096*MR, mimicking the real ccsd_t_* stride
# pattern, E3) already makes one tile's own address footprint ~0.4-0.9M
# elements (~3-7 MB for Float64) at the shapes swept here. 4096 *fully
# disjoint* copies of that footprint would need tens of GB per shape, which
# is not a reasonable thing for a benchmark script to allocate on a shared
# machine. Instead this sweeps COLD_NPOS=4096 positions spaced COLD_STEP=8191
# elements apart (coprime-ish to the 4096-element row stride, so consecutive
# positions land on different pages), spanning ~270 MB of address space per
# shape -- comfortably larger than one core's L2 (1 MiB) and most of one
# socket's L3 (~24.75 MiB) on the reference machine, while keeping total
# script memory in the hundreds-of-MB range. This is an approximation of "a
# large, mostly-cold output", not a reproduction of the exact regression
# case's address arithmetic (that is out of scope -- E3/Cause B, non-goals).
const COLD_NPOS = 4096
const COLD_STEP = 8191

# ---------------------------------------------------------------------------
# Kernel shapes
# ---------------------------------------------------------------------------

# (dtype, (MR,NR,W), label). Shipped defaults confirmed by reading
# src/driver.jl's `_derived_shape`/`_legacy_shape`/`KERNEL_SHAPES_F64`/
# `KERNEL_SHAPES_F32` directly (AVX-512, this machine): Float64 -> (16,6,8),
# Float32 -> (32,6,16). (16,14,8) and (32,6,8) are the two Phase G/H "NV is
# held at 12 deliberately" / "panel addressing" swept shapes (docs/
# decisions.md), reused here verbatim for comparability.
const SHAPES = [
    (Float64, (16, 6, 8), "F64_16x6x8 (shipped default)"),
    (Float64, (16, 14, 8), "F64_16x14x8 (Phase G/H skx dgemm family)"),
    (Float64, (32, 6, 8), "F64_32x6x8 (Phase G/H swept)"),
    (Float32, (32, 6, 16), "F32_32x6x16 (shipped default)"),
]

const KC_VALUES = (16, 256)

shape_label(MR, NR, W) = "$(MR)x$(NR)x$(W)"

# ---------------------------------------------------------------------------
# Packed-panel construction (mirrors test/test_simd_kernel.jl's
# `packed_from_matrices` helper -- same offset formulas, same construction
# order).
# ---------------------------------------------------------------------------

function build_packed(::Type{T}, kernel, kc::Int, rng) where {T}
    MR, NR = mr(kernel), nr(kernel)
    Amat = rand(rng, T, MR, max(kc, 1))
    Bmat = rand(rng, T, max(kc, 1), NR)
    pa = zeros(T, packed_a_length(kernel, kc))
    pb = zeros(T, packed_b_length(kernel, kc))
    for p in 0:(kc - 1), i in 0:(MR - 1)
        pa[packed_a_offset(kernel, i, p) + 1] = Amat[i + 1, p + 1]
    end
    for p in 0:(kc - 1), j in 0:(NR - 1)
        pb[packed_b_offset(kernel, j, p) + 1] = Bmat[p + 1, j + 1]
    end
    return pa, pb
end

# ---------------------------------------------------------------------------
# Destination variants
# ---------------------------------------------------------------------------

# `dests` is a concretely-typed Vector (length 1 except D-strided-cold) so
# `bench_execute!`'s inner loop never dynamically dispatches.
struct DestVariant{V}
    name::String
    dests::V
    m::Int
    n::Int
end

function dvec_variant(::Type{T}, MR, NR) where {T}
    storage = zeros(T, MR * NR)
    rows, cols = AffineAxis(0, 1, MR), AffineAxis(0, MR, NR)
    dest = DestinationTile(storage, 0, rows, cols)
    return DestVariant("D-vec", [dest], MR, NR)
end

function dmem_variant(::Type{T}, MR, NR) where {T}
    storage = parent(StridedView(zeros(T, MR, NR)))
    rows, cols = AffineAxis(0, 1, MR), AffineAxis(0, MR, NR)
    dest = DestinationTile(storage, 0, rows, cols)
    return DestVariant("D-mem", [dest], MR, NR)
end

function dvec_tail_variant(::Type{T}, MR, NR) where {T}
    m = MR - 3
    storage = zeros(T, m * NR)
    rows, cols = AffineAxis(0, 1, m), AffineAxis(0, m, NR)
    dest = DestinationTile(storage, 0, rows, cols)
    return DestVariant("D-vec-tail", [dest], m, NR)
end

function dmem_tail_variant(::Type{T}, MR, NR) where {T}
    m = MR - 3
    storage = parent(StridedView(zeros(T, m, NR)))
    rows, cols = AffineAxis(0, 1, m), AffineAxis(0, m, NR)
    dest = DestinationTile(storage, 0, rows, cols)
    return DestVariant("D-mem-tail", [dest], m, NR)
end

max_tile_offset(MR, NR) = (MR - 1) * 4096 + (NR - 1) * 4096 * MR

function dstrided_hot_variant(::Type{T}, MR, NR) where {T}
    storage = zeros(T, max_tile_offset(MR, NR) + 1)
    rows, cols = AffineAxis(0, 4096, MR), AffineAxis(0, 4096 * MR, NR)
    dest = DestinationTile(storage, 0, rows, cols)
    return DestVariant("D-strided-hot", [dest], MR, NR)
end

function dstrided_cold_variant(::Type{T}, MR, NR) where {T}
    tile_span = max_tile_offset(MR, NR) + 1
    storage = zeros(T, COLD_NPOS * COLD_STEP + tile_span)
    rows, cols = AffineAxis(0, 4096, MR), AffineAxis(0, 4096 * MR, NR)
    dests = [DestinationTile(storage, (p - 1) * COLD_STEP, rows, cols) for p in 1:COLD_NPOS]
    return DestVariant("D-strided-cold", dests, MR, NR)
end

variant_builders(::Type{T}, MR, NR) where {T} = (
    dvec_variant(T, MR, NR), dmem_variant(T, MR, NR),
    dstrided_hot_variant(T, MR, NR), dstrided_cold_variant(T, MR, NR),
    dvec_tail_variant(T, MR, NR), dmem_tail_variant(T, MR, NR),
)

# ---------------------------------------------------------------------------
# Timed inner loops
# ---------------------------------------------------------------------------

# `accumulate` alone (FMA only, no store): destination-independent. Sinks
# into a `Ref` so the compiler cannot hoist/CSE the whole batch away (with no
# real side effect and identical inputs every iteration, that is a genuine
# risk for a pure function called in a tight loop).
function bench_accumulate(kernel, packed_a, packed_b, kc::Int, batch::Int)
    T = scalartype(kernel)
    sink = Ref(zero(T))
    @inbounds for _ in 1:batch
        acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
        sink[] += first(Tuple(acc[1]))
    end
    return sink[]
end

# `execute_tile!`, including the store. Cycles through `dests` (length 1
# except D-strided-cold) so the array write itself is inside the timed loop.
function bench_execute(kernel, dests::Vector, packed_a, packed_b, kc::Int, alpha, beta, batch::Int)
    n = length(dests)
    @inbounds for k in 1:batch
        d = dests[mod1(k, n)]
        execute_tile!(kernel, d, packed_a, packed_b, kc, alpha, beta)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Native-code inspection (report-only; mirrors bench_kernel_shape.jl's own
# stack-store regex approach, reimplemented here since that file is out of
# this task's edit scope)
# ---------------------------------------------------------------------------

# A stack store has the [rsp/rbp] operand first, i.e. followed by a comma --
# same pattern bench_kernel_shape.jl's `stack_store_re` uses for spill
# detection.
const VEC_STORE_RE = r"v(?:mov(?:up|ap)[sd]|movdq[ua])\s+(?:zmmword|ymmword|xmmword)\s+ptr\s+\[([^\]]*)\]\s*,"

function native_store_counts(f, argtypes::Tuple)
    io = IOBuffer()
    try
        code_native(io, f, argtypes; debuginfo = :none, syntax = :intel)
    catch err
        return (ok = false, err = string(err), total = 0, stack = 0, other = 0)
    end
    asm = String(take!(io))
    total = 0
    stack = 0
    for m in eachmatch(VEC_STORE_RE, asm)
        total += 1
        occursin(r"r[sb]p", m.captures[1]) && (stack += 1)
    end
    return (ok = true, err = "", total = total, stack = stack, other = total - stack)
end

# ---------------------------------------------------------------------------
# Output setup
# ---------------------------------------------------------------------------

const OUTDIR = results_dir()
mkpath(OUTDIR)

function unique_path(path::String)
    isfile(path) || return path
    base, ext = splitext(path)
    i = 1
    while isfile("$(base)_$(i)$(ext)")
        i += 1
    end
    chosen = "$(base)_$(i)$(ext)"
    @warn "output path already existed; using a suffixed name instead" original = path chosen
    return chosen
end

const CSV_PATH = unique_path(joinpath(OUTDIR, "bench_store_path.csv"))
const SUMMARY_PATH = unique_path(joinpath(OUTDIR, "summary_store_path.txt"))
const PROVENANCE_PATH = unique_path(joinpath(OUTDIR, "PROVENANCE_store_path.txt"))
const CANARY_PATH = unique_path(joinpath(OUTDIR, "canary_store_path.csv"))
native_path(tag) = unique_path(joinpath(OUTDIR, "store_path_native_$(tag).txt"))

csv_io = open(CSV_PATH, "w")
println(
    csv_io,
    "kernel_shape,dtype,kc,destination_variant,call,reps,median_seconds,",
    "gflops,allocated_bytes,storage_type_string"
)

const ROWS = NamedTuple[]

function log_row!(kernel_shape, T, kc, variant, call, t, gf, alloc, storagetype)
    println(
        csv_io,
        kernel_shape, ",", T, ",", kc, ",", variant, ",", call, ",", REPS, ",",
        @sprintf("%.9e", t), ",", (gf === nothing ? "" : @sprintf("%.4f", gf)), ",",
        alloc, ",", storagetype
    )
    flush(csv_io)
    push!(
        ROWS, (
            shape = kernel_shape, dtype = T, kc = kc, variant = variant,
            call = call, t = t, gf = gf, alloc = alloc, storagetype = storagetype,
        )
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Machine-load check (this project's standing measurement discipline: the
# reference machine is not assumed exclusive)
# ---------------------------------------------------------------------------

function capture_cmd(cmd)
    return try
        read(cmd, String)
    catch err
        "unavailable: $err"
    end
end

const UPTIME_OUT = capture_cmd(`uptime`)
const TOP_OUT = capture_cmd(`top -bn1`)
const TOP_HEAD = join(split(TOP_OUT, '\n')[1:min(15, end)], '\n')

println("machine load at start:\n", UPTIME_OUT)

# ---------------------------------------------------------------------------
# Canary bracket
# ---------------------------------------------------------------------------

# Fixed cell timed at start/middle/end: Float64 (16,6,8) shipped default,
# kc=256, D-vec, beta=0.
function run_store_path_canary(rng, label::String)
    T = Float64
    MR, NR, W = 16, 6, 8
    kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
    kc = 256
    pa, pb = build_packed(T, kernel, kc, rng)
    variant = dvec_variant(T, MR, NR)
    t = GC.@preserve pa pb begin
        panel_a = packed_panel(pa, 1, length(pa))
        panel_b = packed_panel(pb, 1, length(pb))
        bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 0.0, 1) # warm-up
        median_time_s(
            () -> bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 0.0, BATCH);
            reps = 15
        ) / BATCH
    end
    println("canary[$label] median per-call = $(t) s")
    return t
end

rng = MersenneTwister(0x5701_0002)
canary_results = Float64[]
push!(canary_results, run_store_path_canary(rng, "A (start)"))

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

for (T, (MR, NR, W), label) in SHAPES
    kshape = shape_label(MR, NR, W)
    kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
    for kc in KC_VALUES
        pa, pb = build_packed(T, kernel, kc, rng)
        GC.@preserve pa pb begin
            panel_a = packed_panel(pa, 1, length(pa))
            panel_b = packed_panel(pb, 1, length(pb))

            # --- accumulate (destination-independent) ---
            bench_accumulate(kernel, panel_a, panel_b, kc, 1) # warm-up
            t_acc = median_time_s(
                () -> bench_accumulate(kernel, panel_a, panel_b, kc, BATCH);
                reps = REPS
            ) / BATCH
            alloc_acc = @allocated bench_accumulate(kernel, panel_a, panel_b, kc, ALLOC_BATCH)
            gf_acc = gflops(T, MR, kc, NR, t_acc)

            for variant in variant_builders(T, MR, NR)
                log_row!(kshape, T, kc, variant.name, "accumulate", t_acc, gf_acc, alloc_acc, "n/a (accumulate)")

                storagetype = string(typeof(variant.dests[1].storage))

                bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 0.0, 1) # warm-up
                t0 = median_time_s(
                    () -> bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 0.0, BATCH);
                    reps = REPS
                ) / BATCH
                alloc0 = @allocated bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 0.0, ALLOC_BATCH)
                log_row!(kshape, T, kc, variant.name, "execute_tile_beta0", t0, nothing, alloc0, storagetype)

                bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 1.0, 1) # warm-up
                t1 = median_time_s(
                    () -> bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 1.0, BATCH);
                    reps = REPS
                ) / BATCH
                alloc1 = @allocated bench_execute(kernel, variant.dests, panel_a, panel_b, kc, 1.0, 1.0, ALLOC_BATCH)
                log_row!(kshape, T, kc, variant.name, "execute_tile_beta1", t1, nothing, alloc1, storagetype)
            end
        end
        @info "sweep progress" dtype = T shape = (MR, NR, W) kc = kc
    end
    push!(canary_results, run_store_path_canary(rng, "mid ($label done)"))
end

push!(canary_results, run_store_path_canary(rng, "A' (end)"))
close(csv_io)

const CANARY_SPREAD = relative_spread(canary_results)
open(CANARY_PATH, "w") do io
    println(io, "index,median_seconds_per_call")
    for (i, t) in enumerate(canary_results)
        println(io, "$i,", @sprintf("%.9e", t))
    end
    println(io, "# relative spread (max-min)/min = ", @sprintf("%.4f", CANARY_SPREAD))
end
println("canary spread (max-min)/min = ", @sprintf("%.4f", CANARY_SPREAD))

# ---------------------------------------------------------------------------
# Reproduction attempt: "SIMDKernel reaches 101-103 GFLOP/s, ~88% of peak"
# (docs/decisions.md, "NV is held at 12 deliberately"; that table's own
# "as plain Vector" column at Float64, kc=256, listed (16,14,8) => 101.8 and
# (32,6,8) => 102.6). Uses a plain packed `Vector`, not a `PackedPanel`, to
# match the cited table's own column, and additionally reports the same cell
# using `PackedPanel` (the real driver's actual panel type today) for
# comparison.
# ---------------------------------------------------------------------------

const REPRO_SHAPES = [
    (Float64, (16, 6, 8)),
    (Float64, (16, 14, 8)),
    (Float64, (32, 6, 8)),
    (Float32, (32, 6, 16)),
]
const REPRO_KC = 256

repro_rows = NamedTuple[]
for (T, (MR, NR, W)) in REPRO_SHAPES
    kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
    pa, pb = build_packed(T, kernel, REPRO_KC, rng)

    bench_accumulate(kernel, pa, pb, REPRO_KC, 1) # warm-up
    t_vec = median_time_s(() -> bench_accumulate(kernel, pa, pb, REPRO_KC, BATCH); reps = REPS) / BATCH
    gf_vec = gflops(T, MR, REPRO_KC, NR, t_vec)

    t_panel = GC.@preserve pa pb begin
        panel_a = packed_panel(pa, 1, length(pa))
        panel_b = packed_panel(pb, 1, length(pb))
        bench_accumulate(kernel, panel_a, panel_b, REPRO_KC, 1) # warm-up
        median_time_s(() -> bench_accumulate(kernel, panel_a, panel_b, REPRO_KC, BATCH); reps = REPS) / BATCH
    end
    gf_panel = gflops(T, MR, REPRO_KC, NR, t_panel)

    push!(
        repro_rows, (
            dtype = T, shape = (MR, NR, W), kc = REPRO_KC,
            gf_vec = gf_vec, gf_panel = gf_panel,
        )
    )
    @info "reproduction attempt" dtype = T shape = (MR, NR, W) gf_as_vector = gf_vec gf_as_packedpanel = gf_panel
end

# ---------------------------------------------------------------------------
# Native-code inspection: shipped shapes only, D-vec vs D-mem
# ---------------------------------------------------------------------------

native_rows = NamedTuple[]
for (T, (MR, NR, W), label) in SHAPES
    occursin("shipped default", label) || continue
    kernel = SIMDKernel(Val(MR), Val(NR), T, Val(W))
    kc = 4
    pa, pb = build_packed(T, kernel, kc, rng)
    GC.@preserve pa pb begin
        panel_a = packed_panel(pa, 1, length(pa))
        panel_b = packed_panel(pb, 1, length(pb))
        for (vname, variant) in (
                ("D-vec", dvec_variant(T, MR, NR)), ("D-mem", dmem_variant(T, MR, NR)),
            )
            dest = variant.dests[1]
            argtypes = (
                typeof(kernel), typeof(dest), typeof(panel_a), typeof(panel_b), Int, T, T,
            )
            rep = native_store_counts(execute_tile!, argtypes)
            tag = "$(shape_label(MR, NR, W))_$(vname)"
            path = native_path(tag)
            io = IOBuffer()
            try
                code_native(io, execute_tile!, argtypes; debuginfo = :none, syntax = :intel)
            catch err
                println(io, "code_native FAILED: ", err)
            end
            open(path, "w") do fio
                write(fio, take!(io))
            end
            push!(native_rows, (dtype = T, shape = (MR, NR, W), variant = vname, rep = rep, path = path))
            @info "native code dumped" shape = (MR, NR, W) variant = vname total_vec_stores = rep.total stack_stores = rep.stack other_stores = rep.other path = path
        end
    end
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

function per_element_ns(rows, shape, T, kc, variant, call)
    for r in rows
        if r.shape == shape && r.dtype == T && r.kc == kc && r.variant == variant && r.call == call
            return r.t
        end
    end
    return NaN
end

open(SUMMARY_PATH, "w") do io
    print_env_header(io, "bench_store_path.jl")
    println(io, "reps = ", REPS, "  batch = ", BATCH, "  alloc_batch = ", ALLOC_BATCH)
    println(io, "cold_npos = ", COLD_NPOS, "  cold_step = ", COLD_STEP)
    println(io, "\nmachine load (uptime): ", strip(UPTIME_OUT))
    println(io, "\ncanary median per-call times (s): ", canary_results)
    println(io, "canary relative spread (max-min)/min: ", @sprintf("%.4f", CANARY_SPREAD))

    println(io, "\n== Reproduction attempt: \"101-103 GFLOP/s, ~88% of peak\" (accumulate-only) ==")
    println(io, "(docs/decisions.md, \"NV is held at 12 deliberately\", Float64 kc=256, \"as plain Vector\" column)")
    for r in repro_rows
        println(
            io, "  ", r.dtype, " ", r.shape, " kc=", r.kc,
            "  as-plain-Vector: ", @sprintf("%.2f", r.gf_vec), " GFLOP/s",
            "  as-PackedPanel: ", @sprintf("%.2f", r.gf_panel), " GFLOP/s"
        )
    end
    cited_lo, cited_hi = 101.0, 103.0
    any_match = any(r -> r.dtype == Float64 && (cited_lo <= r.gf_vec <= cited_hi || cited_lo <= r.gf_panel <= cited_hi), repro_rows)
    println(
        io, any_match ?
            "  MATCH: at least one Float64 cell fell inside the cited 101-103 GFLOP/s band." :
            "  NO MATCH: no measured cell fell inside the cited 101-103 GFLOP/s band on this run " *
            "(reported honestly rather than forced -- see the numbers above for the actual gap)."
    )

    println(io, "\n== Store cost table: per-element store cost (ns) = (median(execute_tile_beta) - median(accumulate)) / (m*n) ==")
    for (T, (MR, NR, W), label) in SHAPES
        kshape = shape_label(MR, NR, W)
        for kc in KC_VALUES
            println(io, "\n-- ", label, "  kc=", kc, " --")
            for variant in ("D-vec", "D-mem", "D-strided-hot", "D-strided-cold", "D-vec-tail", "D-mem-tail")
                t_acc = per_element_ns(ROWS, kshape, T, kc, variant, "accumulate")
                t0 = per_element_ns(ROWS, kshape, T, kc, variant, "execute_tile_beta0")
                t1 = per_element_ns(ROWS, kshape, T, kc, variant, "execute_tile_beta1")
                mn = variant in ("D-vec-tail", "D-mem-tail") ? (MR - 3) * NR : MR * NR
                cost0 = (t0 - t_acc) / mn * 1.0e9
                cost1 = (t1 - t_acc) / mn * 1.0e9
                println(
                    io, "  ", rpad(variant, 16),
                    "  accumulate=", @sprintf("%.3e", t_acc), " s/call",
                    "  beta0=", @sprintf("%.3e", t0), " s/call (", @sprintf("%7.2f", cost0), " ns/elem store cost)",
                    "  beta1=", @sprintf("%.3e", t1), " s/call (", @sprintf("%7.2f", cost1), " ns/elem store cost)"
                )
            end
        end
    end

    println(io, "\n== D-vec vs D-mem: the core deliverable ==")
    println(io, "(today's real driver hands the kernel D-mem's storage type; D-vec is what would take the")
    println(io, " fast path IF the guard were fixed to also accept Memory{T} -- this is what a fix could recover)")
    for (T, (MR, NR, W), label) in SHAPES
        kshape = shape_label(MR, NR, W)
        for kc in KC_VALUES
            t_acc = per_element_ns(ROWS, kshape, T, kc, "D-vec", "accumulate")
            t0_vec = per_element_ns(ROWS, kshape, T, kc, "D-vec", "execute_tile_beta0")
            t0_mem = per_element_ns(ROWS, kshape, T, kc, "D-mem", "execute_tile_beta0")
            t1_vec = per_element_ns(ROWS, kshape, T, kc, "D-vec", "execute_tile_beta1")
            t1_mem = per_element_ns(ROWS, kshape, T, kc, "D-mem", "execute_tile_beta1")
            mn = MR * NR
            c0v = (t0_vec - t_acc) / mn * 1.0e9
            c0m = (t0_mem - t_acc) / mn * 1.0e9
            c1v = (t1_vec - t_acc) / mn * 1.0e9
            c1m = (t1_mem - t_acc) / mn * 1.0e9
            println(
                io, "  ", label, " kc=", kc,
                "  beta0: D-vec=", @sprintf("%7.2f", c0v), " ns/elem  D-mem=", @sprintf("%7.2f", c0m),
                " ns/elem  delta(mem-vec)=", @sprintf("%7.2f", c0m - c0v), " ns/elem"
            )
            println(
                io, "  ", " "^length(label), "        ",
                "  beta1: D-vec=", @sprintf("%7.2f", c1v), " ns/elem  D-mem=", @sprintf("%7.2f", c1m),
                " ns/elem  delta(mem-vec)=", @sprintf("%7.2f", c1m - c1v), " ns/elem"
            )
        end
    end

    println(io, "\n== Allocation (bytes over ", ALLOC_BATCH, " calls; should be 0 everywhere -- Cliff B's static-index fix applies to BOTH code paths) ==")
    for r in ROWS
        r.alloc == 0 && continue
        println(io, "  NONZERO ALLOCATION: ", r.shape, " ", r.dtype, " kc=", r.kc, " ", r.variant, " ", r.call, " => ", r.alloc, " B")
    end
    if all(r -> r.alloc == 0, ROWS)
        println(io, "  none found -- every (shape,kc,variant,call) cell measured 0 B.")
    end

    println(io, "\n== Native-code stack-store counts (shipped shapes, D-vec vs D-mem; report-only, no fix) ==")
    println(io, "(VEC_STORE_RE counts, ", "vmovup(s|d)/vmovap(s|d)/vmovdqu/vmovdqa to a zmmword/ymmword/xmmword ptr; \"stack\" = operand references r[sb]p)")
    for r in native_rows
        println(
            io, "  ", r.dtype, " ", r.shape, " ", r.variant,
            "  total_vector_stores=", r.rep.ok ? r.rep.total : -1,
            "  stack(rsp/rbp)=", r.rep.ok ? r.rep.stack : -1,
            "  other(non-stack, e.g. destination/panel pointers)=", r.rep.ok ? r.rep.other : -1,
            r.rep.ok ? "" : "  CODEGEN FAILED: " * r.rep.err,
            "  (dump: ", r.path, ")"
        )
    end
end
println(read(SUMMARY_PATH, String))

open(PROVENANCE_PATH, "w") do io
    println(io, "git_commit = ", git_commit())
    println(io, "command = julia --project=. benchmark/bench_store_path.jl")
    println(io, "cpu = ", Sys.CPU_NAME)
    println(io, "julia = ", VERSION)
    println(io, "nthreads = ", NTHREADS, "  blas_threads = ", BLAS_THREADS)
    println(io, "date = ", now())
    println(io, "reps = ", REPS, "  batch = ", BATCH, "  alloc_batch = ", ALLOC_BATCH)
    println(io, "shapes = ", [(T, s) for (T, s, _) in SHAPES])
    println(io, "kc_values = ", KC_VALUES)
    println(io, "canary_spread = ", @sprintf("%.4f", CANARY_SPREAD))
    println(io, "\n--- machine load at script start (uptime) ---")
    println(io, UPTIME_OUT)
    println(io, "--- machine load at script start (top -bn1, first 15 lines) ---")
    println(io, TOP_HEAD)
    println(
        io, "\nSINGLE-MACHINE CAVEAT: as with every other benchmark in this project, this is one",
        "\nCascade Lake / AVX-512 machine (see docs/decisions.md's standing caveat), not a portable claim.",
        "\nThe reference machine was NOT confirmed exclusive for this run (see uptime/top above);",
        "\nthe canary bracket above is the check for drift during the run."
    )
end

println("\nDone. Results in ", OUTDIR)
