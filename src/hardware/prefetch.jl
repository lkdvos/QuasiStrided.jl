# Software prefetch: a raw `llvm.prefetch` intrinsic, and the compile-time
# switches that decide whether any site in the engine issues one.
#
# Nothing else in the stack provides this: SIMD.jl exports no prefetch, and
# Base/Core.Intrinsics have none. `llvm.prefetch` takes its three trailing
# operands as `immarg`s, so they must be literals in the IR text; that is why
# `prefetch` is `@generated` over `Val`s rather than taking runtime integers.
#
# Verified at the runtime optimization level the package actually runs at
# (`-O2`, Julia's default), on Julia 1.12.7/LLVM 18 and 1.13.0/LLVM 20: the
# call lowers to exactly one `prefetcht0`/`prefetcht1`/`prefetcht2`/
# `prefetchnta`/`prefetchw` instruction, it survives inlining into a caller,
# and inside a gather loop it stays in the loop body at the per-iteration
# position it was written at (neither hoisted nor eliminated).
#
# The pointer goes through LLVM as an `i64` plus `inttoptr` rather than as a
# `ptr` argument: `llvmcall` has lowered `Ptr{T}` differently across Julia
# versions, while an integer argument means the same thing on all of them.
# Under opaque pointers (LLVM >= 15, every Julia this package supports that
# has LLVM >= 15) the intrinsic's name is `llvm.prefetch.p0`.

"""
    prefetch(ptr::Ptr, ::Val{RW} = Val(0), ::Val{LOCALITY} = Val(3))

Issue one data-cache software prefetch of the line containing `ptr`. A hint
only: it never faults, whatever `ptr` is, and never changes a result.

`RW` is `0` (read) or `1` (write; `prefetchw` on x86). `LOCALITY` is `0`-`3`,
from no temporal locality (`prefetchnta`) to keep in every cache level
(`prefetcht0`); `1` and `2` are `prefetcht2` and `prefetcht1`.
"""
@generated function prefetch(
        ptr::Ptr, ::Val{RW} = Val(0), ::Val{LOCALITY} = Val(3)
    ) where {RW, LOCALITY}
    if !(RW isa Int && RW in (0, 1))
        msg = "prefetch: RW must be 0 or 1, got $(repr(RW))"
        return :(throw(ArgumentError($msg)))
    end
    if !(LOCALITY isa Int && 0 <= LOCALITY <= 3)
        msg = "prefetch: LOCALITY must be 0:3, got $(repr(LOCALITY))"
        return :(throw(ArgumentError($msg)))
    end
    ir = """
    declare void @llvm.prefetch.p0(ptr nocapture readonly, i32 immarg, i32 immarg, i32 immarg)

    define void @entry(i64 %0) #0 {
    top:
      %p = inttoptr i64 %0 to ptr
      call void @llvm.prefetch.p0(ptr %p, i32 $RW, i32 $LOCALITY, i32 1)
      ret void
    }

    attributes #0 = { alwaysinline }
    """
    return :(Base.llvmcall(($ir, "entry"), Cvoid, Tuple{UInt}, UInt(ptr)))
end

# ----------------------------------------------------------------------------
# Per-site switches
# ----------------------------------------------------------------------------
#
# EXPERIMENTAL insertion points, each off by default and independently
# switchable:
#
#   :pack_b  -- the B gather loop (`_pack_b_sliver!`'s scalar fallback,
#               src/packing/pack.jl). `distance` is in K steps.
#   :pack_a  -- the A gather loop, only where A's contiguous fast path is NOT
#               eligible. `distance` is in K steps.
#   :macro   -- the macro-kernel's panel-ahead prefetch
#               (src/execution/execute.jl, loop 1): while the microkernel
#               consumes one packed A/B micropanel, prefetch the head of the
#               next one. `distance` is the number of cache lines of that head.
#   :pack_b_line, :pack_a_line
#            -- cache-line-granular versions of `:pack_b`/`:pack_a` (same
#               loops, same `distance` in K steps): one prefetch per distinct
#               line a K step's lanes touch, issued only when those lines
#               differ from the previous step's, instead of one per lane per
#               step. When both the per-lane and the line site of one operand
#               are on, the line site wins.
#   :ctile   -- the C micro-tile (src/execution/macrokernel.jl,
#               `unsafe_execute_micro_tile!`): every line of the tile the
#               microkernel is about to write, prefetched (`prefetcht0`) right
#               before its K loop starts. Any `distance > 0` means on.
#   :ctile_w -- the same, with write intent (`prefetchw`).
#
# `_prefetch_distance(Val(site))` returns a literal `Int`, `0` meaning off. Each
# call site branches on it with `> 0`, which folds at compile time, so a
# disabled site contributes no instruction at all -- test/execution/
# test_prefetch.jl checks that by counting `prefetch` in the native code of the
# packers and of `_execute_nest!`.
#
# Changing a setting is a method redefinition (`set_prefetch!`), the same
# mechanism as `TimerOutputs.enable_debug_timings`: every compiled method that
# inlined the old constant is invalidated through its backedges and recompiled
# on its next call, so the new value takes effect everywhere with no runtime
# flag in any loop. It is a process-wide, compile-time setting, meant for
# benchmarks and tests -- a redefinition costs a recompile of the engine.
#
# Measured: every site stays OFF by default because none pays for itself
# (`benchmark/bench_prefetch.jl`, Float64/Float32/ComplexF64/ComplexF32, 21
# reps, ratio = t_site_on / t_off per shape, base-before/after spread and
# canary spread reported, all canaries <= 2.9%; `benchmark/perf_prefetch.jl`
# for the counters). Round 1 (2026-09-25): jobs 7109952 (Ice Lake-SP), 7109953
# (Genoa/Zen4), 7109954 (Rome/Zen2). Round 2 (2026-09-26, on main 5242652):
# jobs 7111536 / 7111537 / 7111538, same three node types.
#
#   :pack_b, :pack_b_line -- slower everywhere. Plain/scattered geomean
#     1.01-1.16, worst 1.45 (Float32 12x256x256, Ice Lake); 128-256 MB
#     operands 1.00-1.06. Going per-line cuts executed prefetches ~8x but
#     `instructions` still rise 17-44% and loads 16-30%: the cost is the hook
#     (the offset-table read for step p+D, address math), not the prefetch uops,
#     in a loop that is already load/store bound. Distances 4/16/64: no change.
#   :pack_a, :pack_a_line -- neutral to slightly slower (0.99-1.05) everywhere
#     EXCEPT the `--family irregular` gathers (axes of length 4 with strides up
#     to 2^21, DRAM resident): 0.87 on Ice Lake (cycles 0.82x, L3-miss stalls
#     0.07x, demand L3 misses ~0.05x -- the prefetch really hides DRAM
#     latency), but 1.08-1.11 on Genoa and 1.24-1.27 on Rome (cycles
#     +13-18%, DRAM fills unchanged). Too narrow and too ISA-split for a
#     default; the AMD distance was never swept on those shapes.
#   :macro -- neutral everywhere (0.98-1.02, 1-16 lines).
#   :ctile, :ctile_w -- slower on real types (1.05-1.08, worst 1.33),
#     neutral on complex; write intent changes nothing. The 16x6 Float64 C
#     tile is ~12 lines the out-of-order core already hides behind the K loop.
#
# Cause, overall: hardware stream/stride prefetchers already cover every
# stride-predictable pattern this engine can express (it only takes
# StridedViews, so truly random-index gathers are not expressible), so a
# software prefetch only adds instructions.
const PREFETCH_SITES = (:pack_a, :pack_b, :macro, :pack_a_line, :pack_b_line, :ctile, :ctile_w)

@inline _prefetch_distance(::Val{:pack_a}) = 0
@inline _prefetch_distance(::Val{:pack_b}) = 0
@inline _prefetch_distance(::Val{:macro}) = 0
@inline _prefetch_distance(::Val{:pack_a_line}) = 0
@inline _prefetch_distance(::Val{:pack_b_line}) = 0
@inline _prefetch_distance(::Val{:ctile}) = 0
@inline _prefetch_distance(::Val{:ctile_w}) = 0

# ----------------------------------------------------------------------------
# Cache-line helpers, shared by the line-granular packing sites and `:ctile`
# ----------------------------------------------------------------------------

# Line size assumed for line-granular prefetch. 64 bytes on every x86 this
# package targets; on a 128-byte-line machine the only cost is prefetching
# each line twice.
const PREFETCH_LINE_BYTES = 64
const _LINE_SHIFT = 6

# Prefetch every line overlapping the inclusive byte range `[a, b]`.
@inline function _prefetch_lines!(a::UInt, b::UInt, rw::Val)
    l = a >> _LINE_SHIFT
    lend = b >> _LINE_SHIFT
    while l <= lend
        prefetch(Ptr{Cvoid}(l << _LINE_SHIFT), rw, Val(3))
        l += 1
    end
    return nothing
end

"""
    set_prefetch!(site::Symbol, distance::Integer) -> Int

Switch prefetch site `site` (one of `$(PREFETCH_SITES)`) on with the given
`distance` (> 0), or off (`0`). Returns the previous distance. Takes effect for
every call made after it returns (from the top level; it recompiles the
engine). Experimental: all sites are off by default.
"""
function set_prefetch!(site::Symbol, distance::Integer)
    site in PREFETCH_SITES ||
        throw(ArgumentError("set_prefetch!: unknown site $(repr(site)); expected one of $PREFETCH_SITES"))
    distance >= 0 || throw(ArgumentError("set_prefetch!: distance must be >= 0, got $distance"))
    old = prefetch_distance(site)
    d = Int(distance)
    d == old && return old
    @eval @inline _prefetch_distance(::Val{$(QuoteNode(site))}) = $d
    return old
end

"""
    prefetch_distance(site::Symbol) -> Int

Current setting of prefetch site `site`; `0` is off. See [`set_prefetch!`](@ref).
"""
prefetch_distance(site::Symbol) = Base.invokelatest(_prefetch_distance, Val(site))
