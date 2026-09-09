# Benchmark harness for the Phase 1 grouped-axis indexing layer
# (`src/axis_group.jl`). See Julia-Tensor-Indexing-Agent-Spec.md section 12
# for what this is meant to measure and why.
#
# Run with:
#     julia --project=. benchmark/bench_axis_group.jl
#
# This measures, separately from setup/compilation:
#   1. Repeated reference `offsets` calls (per-element divrem decode).
#   2. Interval generation with preallocated buffers (`fill_offsets!`).
#   3. Generation plus classification (`block_descriptors!`).
#   4. Classification alone on already-filled buffers (`describe_block`).
#
# across affine and irregular layouts, one and several maps, small
# microtile-like counts and larger panel-like counts, and aligned and
# nonaligned interval starts — both for an unnormalized group and its
# `normalize_group` result, where that is meaningful (an irregular layout
# does not normalize away).
#
# This is a correctness-first indexing layer, not a tuned microkernel: no
# absolute speedup threshold is asserted here. The numbers are reported for
# the record (Julia version, CPU, parameters), per the spec's requirement
# to measure rather than assume.

using QuasiStrided
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, describe_block,
    block_descriptors!, normalize_group
using Printf

# ------------------------------------------------------------------------
# Layouts under test
# ------------------------------------------------------------------------

# Affine, single map, unit stride: the maximally regular case.
affine1_unnorm = AxisGroup((8, 8), ((1, 8),))     # foldable: normalize_group -> rank 1
affine1_norm = normalize_group(affine1_unnorm)     # (64,), stride (1,)

# Affine, two maps (as in a paired A/C group), unit + nonunit stride.
affine2_unnorm = AxisGroup((8, 8), ((1, 8), (1, 24)))
affine2_norm = normalize_group(affine2_unnorm)

# Irregular for at least one map: mismatched strides across dims prevent a
# single affine run (this is the section 8 `G` fixture's shape, extended).
irregular2 = AxisGroup((8, 8), ((1, 3), (1, 40)))  # map 2 irregular over the join
# irregular2 does not jointly fold (per-map strides disagree), so
# normalize_group(irregular2) == irregular2 up to singleton removal (none
# here); we do not separately benchmark a "normalized" arm for it.

layouts = [
    ("affine1_unnorm", affine1_unnorm),
    ("affine1_norm", affine1_norm),
    ("affine2_unnorm", affine2_unnorm),
    ("affine2_norm", affine2_norm),
    ("irregular2", irregular2),
]

# Interval shapes: (label, count, aligned_start_fn)
# "microtile-like" (e.g. MR=8) and "panel-like" (bigger) counts, at both an
# aligned (block-boundary) and a nonaligned start.
interval_shapes = [
    ("micro_aligned", 8, g -> 0),
    ("micro_unaligned", 8, g -> 3),
    ("panel_aligned", 48, g -> 0),
    ("panel_unaligned", 48, g -> 5),
]

# ------------------------------------------------------------------------
# Timing helpers: separate warmup/compilation from steady-state execution.
# ------------------------------------------------------------------------

function time_offsets_reference(g, first, count; reps = 2000)
    # Warmup (forces compilation before timing).
    s = 0
    for t in 0:(count - 1)
        s += sum(offsets(g, first + t))
    end
    t0 = time_ns()
    for _ in 1:reps
        for t in 0:(count - 1)
            s += sum(offsets(g, first + t))
        end
    end
    elapsed = (time_ns() - t0) / 1.0e9
    return elapsed / reps, s  # seconds per rep; return s to prevent DCE
end

function time_fill_offsets!(bufs, g, first, count; reps = 2000)
    fill_offsets!(bufs, g, first, count)  # warmup
    allocs = @allocated fill_offsets!(bufs, g, first, count)
    t0 = time_ns()
    for _ in 1:reps
        fill_offsets!(bufs, g, first, count)
    end
    elapsed = (time_ns() - t0) / 1.0e9
    return elapsed / reps, allocs
end

function time_block_descriptors!(bufs, g, first, count; reps = 2000)
    block_descriptors!(bufs, g, first, count)  # warmup
    allocs = @allocated block_descriptors!(bufs, g, first, count)
    t0 = time_ns()
    for _ in 1:reps
        block_descriptors!(bufs, g, first, count)
    end
    elapsed = (time_ns() - t0) / 1.0e9
    return elapsed / reps, allocs
end

function time_describe_block(bufs, count; reps = 2000)
    P = length(bufs)
    descs = ntuple(p -> describe_block(bufs[p], count), P)  # warmup
    allocs = @allocated ntuple(p -> describe_block(bufs[p], count), P)
    t0 = time_ns()
    for _ in 1:reps
        ntuple(p -> describe_block(bufs[p], count), P)
    end
    elapsed = (time_ns() - t0) / 1.0e9
    return elapsed / reps, allocs
end

# ------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------

println("Julia version: ", VERSION)
println("CPU: ", Sys.cpu_info()[1].model, " (", length(Sys.cpu_info()), " logical cores reported)")
println()

@printf(
    "%-16s %-16s %-10s %10s %12s %12s %10s\n",
    "layout", "interval", "count", "offsets/el", "fill/call", "descr/call", "classify"
)
@printf(
    "%-16s %-16s %-10s %10s %12s %12s %10s\n",
    "", "", "", "(ns)", "(ns, alloc)", "(ns, alloc)", "(ns)"
)

for (glabel, g) in layouts
    P = length(g.strides)
    Q = axis_length(g)
    for (ilabel, count, startfn) in interval_shapes
        count > Q && continue
        first = startfn(g)
        first + count > Q && (first = Q - count)  # clamp into domain

        t_offsets, _ = time_offsets_reference(g, first, count)
        t_offsets_per_el = t_offsets / count

        bufs = ntuple(_ -> Vector{Int}(undef, count), P)
        t_fill, a_fill = time_fill_offsets!(bufs, g, first, count)
        t_bd, a_bd = time_block_descriptors!(bufs, g, first, count)

        fill_offsets!(bufs, g, first, count)
        t_cls, _ = time_describe_block(bufs, count)

        @printf(
            "%-16s %-16s %-10d %10.2f %8.1f/%-3d %8.1f/%-3d %10.2f\n",
            glabel, ilabel, count,
            t_offsets_per_el * 1.0e9,
            t_fill * 1.0e9, a_fill,
            t_bd * 1.0e9, a_bd,
            t_cls * 1.0e9
        )
    end
end
