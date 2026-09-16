# Bucketed cost-attribution helper for benchmark/profile_to_suite.jl.
# `include`d, not a module, consistent with benchmark/harness.jl.
#
# Parses `Profile.fetch()`'s raw backtrace data directly (rather than relying
# on `Profile.print`'s text output) so we can compute, per frame, which named
# cost bucket it belongs to and tally sample counts ourselves.

using Profile
using Printf

# Ordered bucket definitions per backend "family". Each entry is
# `(bucketname, [substring, ...])`; a frame matches a bucket if ANY of its
# substrings appears in either the frame's function name or file path
# (case-sensitive, as the source identifiers themselves are). The first
# matching bucket wins, in listed order -- `"other"` (and, for QuasiStrided,
# the qualifying "TensorOperations package path but not already bucketed"
# rule) always exists as a fallback so percentages never silently drop
# samples.

const QS_BUCKETS = [
    ("gc/alloc", ["gc", "jl_gc", "tensoralloc"]),
    ("adapter/prepare", ["_qs_prepare", "StridedView", "argcheck", "dimcheck", "mightalias", "tensoroperations.jl"]),
    ("planning", ["plan_contract", "_plan_contract", "AxisGroup", "fill_offsets!", "block_descriptors!", "driver.jl", "axis_group.jl"]),
    ("packing", ["pack_a!", "pack_b!", "_pack_panel!", "_pack_sliver!", "packing.jl"]),
    # Must precede "microkernel": the store path's own frames live in the
    # same file (kernels/simd.jl) as the FMA microkernel, so a bare
    # "kernels/" substring on "microkernel" would swallow them first-match-wins.
    (
        "store",
        [
            "store_tile!", "scale_tile!", "_store_tile_scattered!",
            "tile_store!", "tile_offset", "_axpby_tile!",
        ],
    ),
    ("microkernel", ["accumulate", "_execute_micro_tile!", "execute_tile!", "kernels/"]),
]

const BLAS_BUCKETS = [
    ("gc/alloc", ["gc", "jl_gc", "tensoralloc"]),
    ("blas", ["dgemm", "sgemm", "gemm", "openblas", "cblas"]),
    ("permute/copy", ["Strided", "_mapreduce", "permutedims", "copy"]),
]

const BUCKET_ORDER_QS = ["adapter/prepare", "planning", "packing", "microkernel", "store", "gc/alloc", "TO overhead", "other"]
const BUCKET_ORDER_BLAS = ["blas", "permute/copy", "gc/alloc", "TO overhead", "other"]

# Cache StackTraces lookups: the same instruction pointer recurs across many
# backtraces/samples, and `StackTraces.lookup` is not free.
const _LOOKUP_CACHE = Dict{UInt, Vector{Base.StackTraces.StackFrame}}()

function _lookup_cached(ip)
    key = UInt(ip)
    return get!(_LOOKUP_CACHE, key) do
        Base.StackTraces.lookup(convert(Ptr{Cvoid}, ip))
    end
end

_frame_haystack(frame) = string(frame.func) * "|" * string(frame.file)

function _is_qs_or_kernels_path(frame)
    f = string(frame.file)
    return occursin("QuasiStrided", f) &&
        (
        occursin("driver.jl", f) || occursin("axis_group.jl", f) || occursin("kernels/", f) ||
            occursin("packing.jl", f) || occursin("tensoroperations.jl", f)
    )
end

function _is_tensoroperations_path(frame)
    f = string(frame.file)
    fn = string(frame.func)
    return occursin("TensorOperations", f) || occursin("tensorcontract", fn) ||
        occursin("ncon", fn) || occursin("promote", fn)
end

function classify_frame(frame, backendname::AbstractString)
    hay = _frame_haystack(frame)
    buckets = backendname == "StridedBLAS" ? BLAS_BUCKETS : QS_BUCKETS
    for (name, substrs) in buckets
        for s in substrs
            if occursin(s, hay)
                return name
            end
        end
    end
    # "TO overhead" fallback: anything in the TensorOperations package path
    # (or the generic entry-point function names) not already caught above.
    if _is_tensoroperations_path(frame)
        return "TO overhead"
    end
    return "other"
end

"""
    compute_buckets(data, backendname) -> Dict{String,Int}

`data` must be the raw backtrace stream from `Profile.fetch(include_meta =
false)` -- metadata (threadid/taskid/etc, included by `fetch`'s default
`include_meta = true`) is NOT instruction-pointer data and must be stripped
first, or `Base.StackTraces.lookup` will resolve garbage frames from it.

Backtraces are stored leaf-first (the actively-executing frame comes first)
and each per-sample backtrace is terminated by a `0` sentinel. This function
splits on those sentinels and classifies only the LEAF frame of each sample
-- i.e. "self time" attribution, the same quantity `Profile.print`'s
`Overhead`/self column reports, and the standard notion of "where time is
spent" for a flamegraph-style summary. (Classifying every frame in every
backtrace, i.e. inclusive/"Count"-column semantics, would double- and
triple-count nested calls and make generic runtime/dispatch frames that sit
between every named layer -- present in nearly every stack purely because of
call depth -- spuriously dominate every bucket.)

Returns raw leaf-sample counts per bucket name (NOT yet normalized to
percentages); callers should divide by the sum of all bucket counts (not by
`length(data)`, which also counts non-leaf frames and sentinels) to get
percentages that sum to 100%.
"""
function compute_buckets(data, backendname::AbstractString)
    order = backendname == "StridedBLAS" ? BUCKET_ORDER_BLAS : BUCKET_ORDER_QS
    counts = Dict{String, Int}(name => 0 for name in order)
    at_leaf = true
    for ip in data
        if iszero(ip)
            # Sentinel: end of one sample's backtrace, next entry starts a
            # new sample's leaf frame.
            at_leaf = true
            continue
        end
        if at_leaf
            frames = _lookup_cached(ip)
            if !isempty(frames)
                # A single IP can resolve to multiple inlined frames; the
                # innermost (first) is the true leaf.
                name = classify_frame(frames[1], backendname)
                counts[name] = get(counts, name, 0) + 1
            end
        end
        at_leaf = false
    end
    return counts
end

"""
    print_bucket_table(io, caseid, backendname, buckets, total_samples, allocated_bytes, reps)

Print a percentage-of-samples table for one (case, backend) pair. Percentages
are computed against the sum of all bucket counts (not the raw `data`
length, which also includes sentinel/zero entries), so they sum to ~100%
regardless of how many samples were sentinels.
"""
function print_bucket_table(io, caseid, backendname, buckets, total_samples, allocated_bytes, reps)
    denom = sum(values(buckets))
    println(io, "="^72)
    @printf(
        io, "case=%s backend=%s reps=%d total_raw_samples=%d classified_samples=%d\n",
        caseid, backendname, reps, total_samples, denom
    )
    @printf(io, "allocated (representative single call): %d bytes\n", allocated_bytes)
    println(io, "-"^72)
    order = backendname == "StridedBLAS" ? BUCKET_ORDER_BLAS : BUCKET_ORDER_QS
    pct_sum = 0.0
    for name in order
        c = get(buckets, name, 0)
        pct = denom == 0 ? 0.0 : 100.0 * c / denom
        pct_sum += pct
        @printf(io, "  %-18s %8d samples  %6.2f%%\n", name, c, pct)
    end
    @printf(io, "  %-18s %8s          %6.2f%% (sanity check, should be ~100.00%%)\n", "TOTAL", "", pct_sum)
    println(io)
    return nothing
end
