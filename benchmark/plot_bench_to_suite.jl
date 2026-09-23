# Plots benchmark/bench_to_suite.jl's results: per (dtype, category), a
# GFLOP/s comparison (StridedBLAS vs QuasiStrided only), a log-scaled
# QuasiStrided / StridedBLAS time-ratio chart (< 1 = QuasiStrided faster),
# and a separate per-case throughput violin plot ("..._violin.png").
#
#   julia --project=benchmark benchmark/plot_bench_to_suite.jl [csv_path]
#
# With no `csv_path`, uses the most recently modified
# benchmark/results/*/bench_to_suite.csv. Writes PNGs next to that CSV.
#
# Only Float64/ComplexF64 rows are plotted (Float32/ComplexF32, if present in
# the CSV, are skipped) and StridedNative is dropped from both panels -- this
# is a two-backend comparison (StridedBLAS vs QuasiStrided) by design.
#
# The violin plot is NOT built from raw per-rep samples -- bench_to_suite.jl
# only logs min/median/std of each case's REPS-sample throughput, to keep the
# CSV small. `synth_samples` below turns those three numbers into a plausible
# distribution (a normal(median, std) reflected below the observed min, so
# nothing falls under the recorded floor) purely for visual shape; it is a
# MODELED approximation, not the empirical distribution -- treat the violin's
# location/spread as real, its exact tail shape as illustrative only. Reading
# an old-format CSV without min_gflops/std_gflops columns skips the violin
# plot for that file (see `read_rows`).

using CairoMakie
using Printf
using Random
using TensorOperationsBenchmarks: TensorOperationsBenchmarks

const PLOTTED_DTYPES = ("Float64", "ComplexF64")
const ELSIZE = Dict("Float64" => 8, "Float32" => 4, "ComplexF64" => 16, "ComplexF32" => 8)

# TCCG_CONTRACTIONS isn't exported, but bench_to_suite.jl already treats this
# package as a first-class dependency (see benchmark/Project.toml); looking
# the id up here instead of re-deriving A/B/C strings some other way keeps
# this in sync with categories/tccg.jl by construction.
const TCCG_BY_ID = Dict(eq.id => eq for eq in TensorOperationsBenchmarks.TCCG_CONTRACTIONS)

function latest_csv()
    root = joinpath(@__DIR__, "results")
    candidates = String[]
    for d in readdir(root; join = true)
        p = joinpath(d, "bench_to_suite.csv")
        isfile(p) && push!(candidates, p)
    end
    isempty(candidates) && error("no benchmark/results/*/bench_to_suite.csv found")
    return candidates[argmax(mtime.(candidates))]
end

const CSV_PATH = isempty(ARGS) ? latest_csv() : ARGS[1]
const OUTDIR = dirname(CSV_PATH)

struct Row
    backend::String
    dtype::String
    category::String
    case_id::String
    dim::Int
    params::Dict{String, String}
    reps::Int
    t::Float64
    gflops::Float64
    gbytes::Float64
    min_gflops::Float64
    std_gflops::Float64
end

# Whether the CSV being read has the min_gflops/std_gflops columns (older
# CSVs lack them); set by `read_rows`, checked before attempting the violin
# plot.
HAS_SPREAD_COLUMNS = false

# `params` is the free-form "k1=v1;k2=v2;..." field bench_to_suite.jl writes
# from `params_string`; values are kept as strings since categories other
# than :pairwise (e.g. :tccg's "contraction=ccsd_1") are not integers --
# `pairwise_shape_label` parses the (integer) keys it needs itself.
function parse_params(s::AbstractString)
    d = Dict{String, String}()
    isempty(s) && return d
    for kv in split(s, ';')
        k, v = split(kv, '=')
        d[k] = v
    end
    return d
end

function read_rows(path)
    lines = readlines(path)
    has_spread = occursin("min_gflops", lines[1])
    global HAS_SPREAD_COLUMNS = has_spread
    rows = Row[]
    for line in lines[2:end]
        f = split(line, ',')
        gf = parse(Float64, f[9])
        min_gf, std_gf = has_spread ? (parse(Float64, f[11]), parse(Float64, f[12])) : (gf, 0.0)
        push!(
            rows, Row(
                f[1], f[2], f[3], f[4], parse(Int, f[5]), parse_params(f[6]), parse(Int, f[7]),
                parse(Float64, f[8]), gf, parse(Float64, f[10]), min_gf, std_gf
            )
        )
    end
    return rows
end

const ROWS = read_rows(CSV_PATH)

# QuasiStrided / StridedBLAS time ratio: < 1 = QuasiStrided faster. Log-scaled
# on the plot (not linear) since case-to-case values span orders of magnitude
# (some ~150x) -- a linear axis makes everything but the most extreme case
# collapse to an indistinguishable sliver near zero.
function ratio_for(rows, case_id, num, den)
    t_num = only(r.t for r in rows if r.case_id == case_id && r.backend == num)
    t_den = only(r.t for r in rows if r.case_id == case_id && r.backend == den)
    return t_num / t_den
end

# Whether `seq` can be split into the two given label sets with each set
# occupying one contiguous run of `seq` (in either order) -- i.e. whether
# `seq`'s own physical index order already groups the two sets without
# interleaving them. This is the exact condition under which merging each
# set into one composite index is a plain reshape (no data movement): the
# labels do not need to already be sorted or in any canonical order *within*
# a set, only the two sets must not interleave.
function _contiguous_partition(seq::AbstractString, groupA::AbstractSet{Char})
    length(seq) <= 1 && return true
    tags = [c in groupA for c in seq]
    transitions = count(i -> tags[i] != tags[i - 1], 2:length(tags))
    return transitions <= 1
end

# "Permute-free" = the whole contraction reduces to a bare BLAS `gemm!` via
# reshapes alone: A's M/K labels are each contiguous in IA, B's K/N labels
# are each contiguous in IB, AND C's M/N labels are each contiguous in IC (a
# permute-free A/B pack alone isn't enough if the output still needs
# permuting into IC's label order). This is a property of the label ORDER
# only -- true for every :pairwise case by construction (`_pairwise_cases`
# always builds `IA = openA++contract`, `IB = contract++openB`,
# `IC = openA++openB`), generally false for :tccg's realistic index orders.
function is_permute_free(IA::AbstractString, IB::AbstractString, IC::AbstractString)
    contracted = Set(IA) ∩ Set(IB)
    _contiguous_partition(IA, contracted) || return false
    _contiguous_partition(IB, contracted) || return false
    openA = setdiff(Set(IA), contracted)
    return _contiguous_partition(IC, openA)
end

# Shared by :pairwise and :tccg: an einsum-style index expression ("IA,IB->IC")
# plus the case's arithmetic intensity (FLOP/byte moved), instead of the bare
# `case_id` (e.g. "dim15_2_1_2"/"ccsd_6_dim16"). Intensity is
# backend-independent (same problem for every backend), computed straight
# from the shape rather than the per-backend GFLOP/s|GB/s rate columns --
# mirrors bench_to_suite.jl's own `flops(spec)`/`case_bytes(spec, T)`,
# specialized to the case where every leg has the same dimension `dim` (true
# for both categories) so element counts reduce to
# `dim^(number of distinct letters)`. A leading "*" marks a permute-free case
# (see `is_permute_free`).
function shape_label(IA::AbstractString, IB::AbstractString, IC::AbstractString, dim::Int, dtype::String)
    contracted = intersect(Set(IA), Set(IB))
    nA_open, nB_open, nc = length(IA) - length(contracted), length(IB) - length(contracted), length(contracted)
    expr = isempty(contracted) ? "$IA,$IB->$IC (outer)" : "$IA,$IB->$IC"
    star = is_permute_free(IA, IB, IC) ? "*" : ""

    flops = 2.0 * Float64(dim)^(nA_open + nc + nB_open)
    esz = ELSIZE[dtype]
    bytes = esz * (Float64(dim)^length(IA) + Float64(dim)^length(IB) + Float64(dim)^length(IC))
    intensity = flops / bytes
    return @sprintf("%s%s  dim=%d  %.3g FLOP/B", star, expr, dim, intensity)
end

function pairwise_shape_label(dim::Int, params::Dict{String, String}, dtype::String)
    nA, nc, nB = parse(Int, params["nopenA"]), parse(Int, params["ncontract"]), parse(Int, params["nopenB"])
    letters = collect('a':'z')
    i = 1
    IA_open, i = letters[i:(i + nA - 1)], i + nA
    contract, i = letters[i:(i + nc - 1)], i + nc
    IB_open, i = letters[i:(i + nB - 1)], i + nB
    IA, IB, IC = join(vcat(IA_open, contract)), join(vcat(contract, IB_open)), join(vcat(IA_open, IB_open))
    return shape_label(IA, IB, IC, dim, dtype)
end

function tccg_shape_label(dim::Int, params::Dict{String, String}, dtype::String)
    eq = TCCG_BY_ID[params["equation"]]
    return shape_label(eq.A, eq.B, eq.C, dim, dtype)
end

# See the file header for why this is a modeled approximation, not real
# per-rep data: normal(median, std), reflected below `min_v` (mirrored back
# up rather than clipped, so the reflected mass still contributes density
# instead of piling up at the boundary). `seed` is deterministic in the
# inputs so replotting the same CSV always draws the same shape.
function synth_samples(median_v::Float64, min_v::Float64, std_v::Float64; n::Int = 300)
    std_v <= 0 && return fill(median_v, n)
    rng = Random.Xoshiro(hash((median_v, min_v, std_v)))
    raw = median_v .+ std_v .* randn(rng, n)
    return map(x -> x < min_v ? 2 * min_v - x : x, raw)
end

for dtype in PLOTTED_DTYPES, category in unique(r.category for r in ROWS)
    subset = filter(r -> r.dtype == dtype && r.category == category, ROWS)
    isempty(subset) && continue
    ids = unique(r.case_id for r in subset)
    length(ids) < 2 && continue

    ratios = [ratio_for(subset, id, "QuasiStrided", "StridedBLAS") for id in ids]
    order = sortperm(ratios)
    ids, ratios = ids[order], ratios[order]

    labeler = category == "pairwise" ? pairwise_shape_label :
        category == "tccg" ? tccg_shape_label : nothing
    labels = labeler === nothing ? ids : [
            labeler(
                first(r.dim for r in subset if r.case_id == id),
                first(r.params for r in subset if r.case_id == id),
                dtype
            ) for id in ids
        ]

    fig = Figure(size = (1150, max(400, 26 * length(ids) + 150) + (labeler === nothing ? 0 : 24)))
    if labeler !== nothing
        Label(
            fig[0, 1:2],
            "* = permute-free (A/B pack and the C store all reduce to a bare BLAS gemm! via reshapes -- no permutation anywhere)";
            fontsize = 12
        )
    end

    ax1 = Axis(
        fig[1, 1]; xscale = log10, yticks = (1:length(ids), labels),
        xlabel = "GFLOP/s (log scale)", title = "$dtype / $category -- throughput"
    )
    for backend in ("StridedBLAS", "QuasiStrided")
        ys = [only(r.gflops for r in subset if r.case_id == id && r.backend == backend) for id in ids]
        scatter!(ax1, ys, 1:length(ids); label = backend, markersize = 10)
    end
    axislegend(ax1; position = :rb)

    ax2 = Axis(
        fig[1, 2]; xscale = log10,
        xlabel = "QuasiStrided / StridedBLAS time (log scale)",
        title = "ratio (< 1 = QuasiStrided faster)"
    )
    hideydecorations!(ax2)
    colors = [r <= 1 ? :seagreen : :firebrick for r in ratios]
    barplot!(ax2, 1:length(ids), ratios; direction = :x, color = colors)
    vlines!(ax2, [1.0]; color = :black, linestyle = :dash)

    path = joinpath(OUTDIR, "bench_to_suite_$(dtype)_$(category).png")
    save(path, fig)
    println("wrote ", path)

    HAS_SPREAD_COLUMNS || continue

    vfig = Figure(size = (1150, max(400, 26 * length(ids) + 150) + (labeler === nothing ? 0 : 24)))
    if labeler !== nothing
        Label(
            vfig[0, 1],
            "* = permute-free (A/B pack and the C store all reduce to a bare BLAS gemm! via reshapes -- no permutation anywhere)";
            fontsize = 12
        )
    end
    vax = Axis(
        vfig[1, 1]; yticks = (1:length(ids), labels),
        xlabel = "GFLOP/s -- modeled from median/min/std (see file header)",
        title = "$dtype / $category -- throughput spread"
    )
    offset = 0.18
    for (boff, backend, color) in ((-offset, "StridedBLAS", :dodgerblue), (offset, "QuasiStrided", :orange))
        ys = Float64[]
        positions = Float64[]
        for (i, id) in enumerate(ids)
            row = only(r for r in subset if r.case_id == id && r.backend == backend)
            samples = synth_samples(row.gflops, row.min_gflops, row.std_gflops)
            append!(ys, samples)
            append!(positions, fill(Float64(i) + boff, length(samples)))
        end
        violin!(
            vax, positions, ys; orientation = :horizontal, side = boff < 0 ? :left : :right,
            width = 2 * abs(offset) * 1.8, color = color, label = backend
        )
    end
    axislegend(vax; position = :rb)
    vpath = joinpath(OUTDIR, "bench_to_suite_$(dtype)_$(category)_violin.png")
    save(vpath, vfig)
    println("wrote ", vpath)
end
