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
# Three EXPERIMENTAL insertion points, each off by default and independently
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
#
# `_prefetch_distance(Val(site))` returns a literal `Int`, `0` meaning off. Each
# call site branches on it with `> 0`, which folds at compile time, so a
# disabled site contributes no instruction at all -- test/hardware/
# test_prefetch.jl checks that by counting `prefetch` in the native code of the
# packers and of `_execute_nest!`.
#
# Changing a setting is a method redefinition (`set_prefetch!`), the same
# mechanism as `TimerOutputs.enable_debug_timings`: every compiled method that
# inlined the old constant is invalidated through its backedges and recompiled
# on its next call, so the new value takes effect everywhere with no runtime
# flag in any loop. It is a process-wide, compile-time setting, meant for
# benchmarks and tests -- a redefinition costs a recompile of the engine.
const PREFETCH_SITES = (:pack_a, :pack_b, :macro)

@inline _prefetch_distance(::Val{:pack_a}) = 0
@inline _prefetch_distance(::Val{:pack_b}) = 0
@inline _prefetch_distance(::Val{:macro}) = 0

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
