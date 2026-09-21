# Plots benchmark/bench_to_suite.jl's results: per (dtype, category), a
# GFLOP/s comparison across backends and a QS/BLAS ratio chart.
#
#   julia --project=benchmark benchmark/plot_bench_to_suite.jl [csv_path]
#
# With no `csv_path`, uses the most recently modified
# benchmark/results/*/bench_to_suite.csv. Writes PNGs next to that CSV.

using CairoMakie

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
    reps::Int
    t::Float64
    gflops::Float64
    gbytes::Float64
end

function read_rows(path)
    lines = readlines(path)
    rows = Row[]
    for line in lines[2:end]
        f = split(line, ',')
        push!(
            rows, Row(
                f[1], f[2], f[3], f[4], parse(Int, f[5]), parse(Int, f[7]),
                parse(Float64, f[8]), parse(Float64, f[9]), parse(Float64, f[10])
            )
        )
    end
    return rows
end

const ROWS = read_rows(CSV_PATH)

function ratio_for(rows, case_id, num, den)
    tn = only(r.t for r in rows if r.case_id == case_id && r.backend == num)
    td = only(r.t for r in rows if r.case_id == case_id && r.backend == den)
    return tn / td
end

for dtype in unique(r.dtype for r in ROWS), category in unique(r.category for r in ROWS)
    subset = filter(r -> r.dtype == dtype && r.category == category, ROWS)
    isempty(subset) && continue
    ids = unique(r.case_id for r in subset)
    length(ids) < 2 && continue

    ratios = [ratio_for(subset, id, "QuasiStrided", "StridedBLAS") for id in ids]
    order = sortperm(ratios)
    ids, ratios = ids[order], ratios[order]

    fig = Figure(size = (900, max(400, 22 * length(ids) + 150)))

    ax1 = Axis(
        fig[1, 1]; xscale = log10, yticks = (1:length(ids), ids),
        xlabel = "GFLOP/s (log scale)", title = "$dtype / $category -- throughput"
    )
    for (i, backend) in enumerate(("StridedNative", "StridedBLAS", "QuasiStrided"))
        ys = [only(r.gflops for r in subset if r.case_id == id && r.backend == backend) for id in ids]
        scatter!(ax1, ys, 1:length(ids); label = backend, markersize = 10)
    end
    axislegend(ax1; position = :rb)

    ax2 = Axis(
        fig[1, 2]; xscale = log10,
        xlabel = "QuasiStrided / StridedBLAS (log scale)",
        title = "ratio (< 1 = QuasiStrided faster)"
    )
    hideydecorations!(ax2)
    colors = [r <= 1 ? :seagreen : :firebrick for r in ratios]
    barplot!(ax2, 1:length(ids), ratios; direction = :x, color = colors)
    vlines!(ax2, [1.0]; color = :black, linestyle = :dash)

    path = joinpath(OUTDIR, "bench_to_suite_$(dtype)_$(category).png")
    save(path, fig)
    println("wrote ", path)
end
