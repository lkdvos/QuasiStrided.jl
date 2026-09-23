# Packing-loop microbenchmark: per-call cost of the single `_pack_panel!` loop
# (src/packing/pack.jl), which packs every format (real, planar, 1e, 1m),
# dispatched via `_pack_emit!`/`_pack_emit_zero!`. Times real and complex
# formats on full and tail slivers (ns per element and per real written), and
# its `code_llvm` section dumps `_pack_panel!` for a complex format to check
# its branch structure.
#
#   julia -t 1 --project=benchmark benchmark/bench_pack_loops.jl
#
# Every case calls `pack_a!`/`pack_b!` **through `plan.kernel`** (never the
# `_pack_panel!` loop function directly): a
# `plan_contract` call builds a real plan first (default kernel selection for
# the real/complex default cases, an explicitly named `OneMKernel` for the 1m
# case, mirroring `benchmark/profile_to_suite.jl`'s `_onem_default_kernel`),
# and `plan.kernel` is what every timed `pack_a!`/`pack_b!` call is dispatched
# on (src/microkernels/interface.jl forwards `DescriptorKernel` to its `.descriptor`,
# which is where src/packing/pack.jl's real/complex methods live).
#
# Fixtures are built directly against `QSTile`/`PackedPanel` (as
# `test/packing/test_pack*.jl` do), NOT via `execute!`, so each case isolates one
# pack call's cost instead of a whole contraction's. "Full sliver" means the
# tile's valid row/column count equals the kernel's MR/NR (the loop's fast
# branch in the real path, no padding at all); "tail sliver" means valid ==
# MR-1/NR-1 (the padding branch), the case a non-multiple extent produces at
# the last block of an M/N sweep. `kc` (the K depth packed per call) is fixed
# at 256 for every case.
#
# Real-dtype A is packed from a deliberately NON-contiguous (stride-2 rows)
# source, so `_pack_a_contiguous_eligible` (src/packing/pack_contiguous.jl) can never
# fire -- `_unit_stride_rows` (src/microkernels/simd.jl) requires stride == 1
# exactly -- and `pack_a!` falls into `_pack_panel!`'s scalar fallback body,
# the real baseline the complex formats are compared against. Real B
# has no contiguous fast path to dodge either way, so it is packed from an
# ordinary column-major-pitch tile.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: PlanarKernel, OneMKernel, kernel_shapes, OneMMethod, default_blocking,
    PackedPanel, packed_panel, AffineAxis, SourceTile, pack_a!, pack_b!,
    packed_a_length, packed_b_length, a_format, b_format, reals_per_element, realtype,
    ComplexKernelDescriptor, PlanarFormat, packed_a_plane_offset, packed_b_plane_offset,
    tile_load, _pack_panel!
using InteractiveUtils

const REPS = 21
const KC = 256                    # fixed K depth for every packing call
const SHAPE_FOR_PLAN = ShapeSpec("plan_256^3", 256, 256, 256)  # only to pick plan.kernel

const OUTDIR = results_dir()
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "pack_loops.csv")
const PROV_PATH = joinpath(OUTDIR, "pack_loops_PROVENANCE.txt")

# ---------------------------------------------------------------------------
# Fixture builders: plain QSTile sources, no `execute!` involved.
# ---------------------------------------------------------------------------

# Non-contiguous A source (stride-2 rows): defeats
# `_pack_a_contiguous_eligible` unconditionally, so `pack_a!` always runs
# `_pack_panel!`'s scalar body for the real dtypes below.
function noncontig_a_tile(::Type{T}, m::Int, kc::Int) where {T}
    rowstride = 2
    colstride = 4 * max(m, 1)   # generous pitch; no address overlap with rows
    len = (m == 0 || kc == 0) ? 1 : (m - 1) * rowstride + (kc - 1) * colstride + 1
    storage = rand(T, len)
    return SourceTile(storage, 0, AffineAxis(0, rowstride, m), AffineAxis(0, colstride, kc))
end

# Ordinary column-major-pitch B source (unit-stride K rows): B has no
# contiguous fast path either way, so this is simply "packed normally".
function plain_b_tile(::Type{T}, kc::Int, n::Int) where {T}
    storage = rand(T, kc * max(n, 1))
    return SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, kc, n))
end

# Ordinary contiguous complex A/B tiles (no fast path exists for complex
# packing at all, so contiguity here is just the common case, not a dodge).
function plain_a_tile(::Type{T}, m::Int, kc::Int) where {T}
    storage = rand(T, max(m, 1) * kc)
    return SourceTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, max(m, 1), kc))
end

# ---------------------------------------------------------------------------
# Plan builders: get a real `plan.kernel` for each configuration.
# ---------------------------------------------------------------------------

# Default kernel (real SIMDKernel, or complex PlanarKernel -- the driver's
# unconditional complex default; src/execution/execute.jl) at a plain 256^3 shape.
function default_plan_kernel(::Type{T}) where {T}
    rng = Random.MersenneTwister(0x0007A616)
    fx = build_plain(T, SHAPE_FOR_PLAN, rng)
    plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
    return plan.kernel
end

# Explicitly-named OneMKernel (1e format on A), mirroring
# `benchmark/profile_to_suite.jl`'s `_onem_default_kernel`.
_onem_default_kernel(::Type{T}) where {T} =
    ((MR, NR, W) = kernel_shapes(T, OneMMethod())[end]; OneMKernel(Val(MR), Val(NR), T, Val(W)))

function onem_plan_kernel(::Type{T}) where {T}
    kernel = _onem_default_kernel(T)
    b = default_blocking(kernel)
    rng = Random.MersenneTwister(0x0007A616)
    fx = build_plain(T, SHAPE_FOR_PLAN, rng)
    plan = plan_contract(
        fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC;
        kernel = kernel, mc = b.mc, kc = b.kc, nc = b.nc
    )
    return plan.kernel
end

# ---------------------------------------------------------------------------
# One timed case
# ---------------------------------------------------------------------------

struct PackCase
    label::String
    dtype::DataType
    format::String     # "real" / "planar" / "1e"
    operand::Symbol     # :A / :B
    sliver::String      # "full" / "tail"
end

function run_case!(csv, kernel, case::PackCase)
    T = case.dtype
    R = realtype(kernel)
    MR, NR = mr(kernel), nr(kernel)
    pd = case.operand === :A ? MR : NR
    valid = case.sliver == "full" ? pd : pd - 1
    valid >= 0 || error("MR/NR too small for a tail sliver: $pd")

    if case.operand === :A
        tile = case.format == "real" ? noncontig_a_tile(T, valid, KC) : plain_a_tile(T, valid, KC)
        needed = packed_a_length(kernel, KC)
        rpe = reals_per_element(a_format(kernel))
        pack! = pack_a!
    else
        tile = plain_b_tile(T, KC, valid)
        needed = packed_b_length(kernel, KC)
        rpe = reals_per_element(b_format(kernel))
        pack! = pack_b!
    end
    buffer = Vector{R}(undef, needed)
    packed = packed_panel(buffer, 1, needed)

    f!() = pack!(packed, tile, kernel, identity)
    t = median_time_s(f!; reps = REPS)

    n_slots = KC * pd                       # register-tile slots touched (valid + padding)
    ns_per_element = 1.0e9 * t / n_slots
    ns_per_real = ns_per_element / rpe

    @printf(
        "%-28s %-10s %-7s %-4s %-6s MR=%-3d NR=%-3d valid=%-3d kc=%-4d  t=%.9e s  ns/elt=%.4f  ns/real=%.4f  (rpe=%d)\n",
        case.label, string(T), case.format, string(case.operand), case.sliver,
        MR, NR, valid, KC, t, ns_per_element, ns_per_real, rpe
    )
    println(
        csv,
        "$(case.label),$T,$(case.format),$(case.operand),$(case.sliver),",
        "$MR,$NR,$valid,$KC,$t,$ns_per_element,$ns_per_real,$rpe,$REPS"
    )
    return (case = case, t = t, ns_per_element = ns_per_element, ns_per_real = ns_per_real, rpe = rpe)
end

# ---------------------------------------------------------------------------
# Code inspection: real and complex packing share one loop, `_pack_panel!`
# (src/packing/pack.jl), dispatched on `PackFormat` via
# `_pack_emit!`/`_pack_emit_zero!`. This dumps it at the Planar-A call's own
# (complex) argument types, i.e. `format = PlanarFormat()`,
# `Val(PD) = Val(MR)`. `_pack_panel!` branches once per `p` (`if valid == PD`
# outside the K loop), not once per element, so check that this coarse branch
# is present and that no per-element `t < valid ? load : zero` conditional
# load appears in the IR.
# ---------------------------------------------------------------------------

function dump_pack_panel_llvm_complex(io, ::Type{T}) where {T}
    kernel = default_plan_kernel(T)           # PlanarKernel{MR,NR,T,W}
    d = kernel.descriptor
    tile = plain_a_tile(T, mr(kernel), KC)
    load = (i, p) -> tile_load(tile, i, p)
    plane_offset = (plane, i, p) -> packed_a_plane_offset(d, plane, i, p)
    R = realtype(kernel)
    needed = packed_a_length(kernel, KC)
    buffer = Vector{R}(undef, needed)
    packed = packed_panel(buffer, 1, needed)
    MR = mr(kernel)

    println(io, "\n--- @code_llvm _pack_panel! at the Planar-A call's argument types ($T) ---")
    println(io, "kernel = ", typeof(kernel), "  MR=", MR, " format=", PlanarFormat())
    io2 = IOBuffer()
    InteractiveUtils.code_llvm(
        io2, _pack_panel!,
        Tuple{
            typeof(packed), Type{T}, PlanarFormat, Val{MR}, Int, Int,
            typeof(identity), typeof(load), typeof(plane_offset),
        };
        debuginfo = :none, optimize = true
    )
    llvm = String(take!(io2))
    println(io, llvm)
    return llvm
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    using_io = stdout
    print_env_header(using_io, "bench_pack_loops.jl")
    println("REPS = ", REPS, "  kc = ", KC)

    csv = open(CSV_PATH, "w")
    println(csv, "label,dtype,format,operand,sliver,MR,NR,valid,kc,t_s,ns_per_element,ns_per_real,reals_per_element,reps")

    canaries = Float64[]
    crng = Random.MersenneTwister(0x0CA9A121)
    push!(canaries, run_canary(crng, "start"))

    results = NamedTuple[]

    # --- complex default (PlanarKernel), A and B separately ---
    for T in CDTYPES
        kernel = default_plan_kernel(T)
        for sliver in ("full", "tail")
            push!(results, run_case!(csv, kernel, PackCase("planar_A_$(T)", T, "planar", :A, sliver)))
            push!(results, run_case!(csv, kernel, PackCase("planar_B_$(T)", T, "planar", :B, sliver)))
        end
    end
    push!(canaries, run_canary(crng, "after-planar"))

    # --- complex OneMKernel (1e format), A only ---
    for T in CDTYPES
        kernel = onem_plan_kernel(T)
        for sliver in ("full", "tail")
            push!(results, run_case!(csv, kernel, PackCase("onem_A_$(T)", T, "1e", :A, sliver)))
        end
    end
    push!(canaries, run_canary(crng, "after-onem"))

    # --- real fallback loop: A non-contiguous, B normal ---
    for T in DTYPES
        kernel = default_plan_kernel(T)
        for sliver in ("full", "tail")
            push!(results, run_case!(csv, kernel, PackCase("real_A_$(T)", T, "real", :A, sliver)))
            push!(results, run_case!(csv, kernel, PackCase("real_B_$(T)", T, "real", :B, sliver)))
        end
    end
    push!(canaries, run_canary(crng, "end"))

    close(csv)

    spread = relative_spread(canaries)
    @printf("\ncanary spread: %.1f%%  %s\n", 100spread, canaries)
    if spread > 0.1
        @warn "canary spread exceeds 10%: the machine was not quiet enough for " *
            "a few-percent conclusion. Re-run before believing any ranking."
    end

    machine_load = try
        strip(read(`uptime`, String))
    catch
        "unknown (uptime failed)"
    end

    open(PROV_PATH, "w") do io
        print_env_header(io, "bench_pack_loops.jl")
        println(io, "git_commit = ", git_commit())
        println(io, "reps = ", REPS, "  kc = ", KC)
        println(io, "canaries = ", canaries)
        @printf(io, "canary spread = %.2f%%\n", 100spread)
        println(io, "machine load (uptime, at provenance-write time) = ", machine_load)
        println(io)
        println(io, "--- microbenchmark table (see pack_loops.csv for the machine-readable form) ---")
        for r in results
            c = r.case
            @printf(
                io, "%-28s %-10s %-7s %-4s %-6s t=%.9e s  ns/elt=%.4f  ns/real=%.4f  (rpe=%d)\n",
                c.label, string(c.dtype), c.format, string(c.operand), c.sliver,
                r.t, r.ns_per_element, r.ns_per_real, r.rpe
            )
        end
    end

    println("\n--- code_llvm: _pack_panel! branch structure (complex formats) ---")
    for T in (ComplexF64, ComplexF32)
        dump_pack_panel_llvm_complex(stdout, T)
    end

    println("\nwrote ", CSV_PATH)
    println("wrote ", PROV_PATH)
    return results
end

main()
