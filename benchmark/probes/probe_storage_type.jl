# T1 probe: is `parent(StridedView(::Array{T}))` a `Vector{T}` (satisfying
# `src/kernels/simd.jl:217`'s fast-path guard `destination.storage isa
# Vector{T}`) or a `Memory{T}` (never satisfying it), on this Julia version?
# And does `SIMD.vload`/`vstore` work correctly and allocation-free on
# whichever storage type `parent` actually returns?
#
# Usage: julia --project=. benchmark/probes/probe_storage_type.jl
# Writes benchmark/results/<hostname>-<date>/probes_T1_storage_type.txt
# (results/ is gitignored; only this script is committed).

using StridedViews
using SIMD
using Dates

const OUTDIR = joinpath(
    @__DIR__, "..", "results", "$(gethostname())-$(Dates.format(now(), "yyyy-mm-dd"))"
)
mkpath(OUTDIR)
const OUTPATH = joinpath(OUTDIR, "probes_T1_storage_type.txt")

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
out("isdefined(Core, :Memory) = ", isdefined(Core, :Memory))
out()

for T in (Float64, Float32)
    for nd in (2, 6)
        p = parent(StridedView(zeros(T, ntuple(_ -> 4, nd))))
        out("T=$T nd=$nd: typeof(parent(StridedView(...))) = ", typeof(p))
        out("T=$T nd=$nd: isa Vector{$T} = ", p isa Vector{T})
        out("T=$T nd=$nd: isa DenseVector{$T} = ", p isa DenseVector{T})
    end
end
out()

for T in (Float64, Float32)
    W = 64 ÷ sizeof(T)
    m = parent(StridedView(collect(T, 1:(4 * W))))
    out("T=$T: W=$W, storage type for vload/vstore probe = ", typeof(m))

    @noinline function do_vload(m, W::Int, ::Type{T}) where {T}
        return SIMD.vload(SIMD.Vec{W, T}, m, W + 1)
    end
    @noinline function alloc_of_vload(m, W::Int, ::Type{T}) where {T}
        return @allocated do_vload(m, W, T)
    end
    v = do_vload(m, W, T)  # warm up (compile)
    v = do_vload(m, W, T)  # measured
    expected = SIMD.Vec{W, T}(ntuple(i -> T(W + i), W))
    out("T=$T: vload result == expected (all lanes): ", all(SIMD.Vec{W, Bool}(Tuple(v == expected))))
    alloc_of_vload(m, W, T)  # warm up the measuring function itself
    alloc_load = alloc_of_vload(m, W, T)
    out("T=$T: @allocated do_vload (post-warmup, measured inside a compiled function) = ", alloc_load)

    @noinline function do_vstore!(m, v, W::Int)
        SIMD.vstore(v, m, 1)
        return nothing
    end
    @noinline function alloc_of_vstore(m, v, W::Int)
        return @allocated do_vstore!(m, v, W)
    end
    newvals = SIMD.Vec{W, T}(ntuple(i -> T(1000 + i), W))
    do_vstore!(m, newvals, W)  # warm up
    alloc_of_vstore(m, newvals, W)  # warm up the measuring function itself
    alloc_store = alloc_of_vstore(m, newvals, W)
    readback = [m[i] for i in 1:W]
    out("T=$T: vstore readback == written values: ", readback == [T(1000 + i) for i in 1:W])
    out("T=$T: @allocated do_vstore! (post-warmup, measured inside a compiled function) = ", alloc_store)

    out("T=$T: pointer(m,3) == pointer(m) + 2*sizeof(T): ", pointer(m, 3) == pointer(m) + 2 * sizeof(T))
    out()
end

close(io)
println("\nWrote ", OUTPATH)
