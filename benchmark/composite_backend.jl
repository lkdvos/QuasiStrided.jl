# Benchmark-only composite backend, so the upstream `TensorOperationsBenchmarks`
# suite's `AbstractProvider` interface -- which takes exactly one `backend` --
# can be pointed at QuasiStrided even though that suite's category list
# includes `:permute`/`:trace` (needing `tensoradd!`/`tensortrace!`) alongside
# `:pairwise`/`:tccg` (needing `tensorcontract!`). `include`d, not a module or
# package, consistent with benchmark/harness.jl.
using TensorOperations
using TensorOperations: AbstractBackend, StridedNative
using QuasiStrided: QuasiStridedBackend

"""
Benchmark-only composite backend: NOT part of the shipped QuasiStrided package, and does
NOT change QuasiStridedBackend's frozen "hard-reject, never fall back" contraction
invariant (see docs/decisions.md). `tensorcontract!` always dispatches to
`QuasiStridedBackend()` -- an ineligible contraction still throws exactly as it would
directly. `tensoradd!`/`tensortrace!` dispatch to `addtrace` (default `StridedNative()`)
because QuasiStridedBackend does not implement them at all, not because of any
performance choice.

WARNING for anyone reading benchmark results: any timing taken under this composite for
an add/trace/permute operation measures `addtrace` (StridedNative by default), NOT
QuasiStrided. Label results accordingly. This type must never migrate into `src/` -- that
would reverse a frozen product decision.
"""
struct QuasiStridedComposite{F <: AbstractBackend} <: AbstractBackend
    addtrace::F
end
QuasiStridedComposite() = QuasiStridedComposite(StridedNative())

# ----------------------------------------------------------------------------
# Method forwarding
#
# Signatures verified against the installed TensorOperations 5.8.1 (checked
# 2026-09-15, ~/.julia/packages/TensorOperations/aLeSt/):
#   - generic 8/9-arg backend-dispatch signatures declared in src/interface.jl
#     (tensoradd! lines 40-45, tensortrace! lines 99-104, tensorcontract! lines
#     169-176: `(C, A, ..., α, β, backend, allocator)`, no default arguments at
#     that arity), and
#   - the exact argument order/keywords a concrete backend fills those with,
#     taken from this repo's own src/tensoroperations.jl (QuasiStridedBackend's
#     `TO.tensorcontract!` methods at lines 304-323 and 332-357, `TO.tensoradd!`
#     at lines 373-380, `TO.tensortrace!` at lines 392-399) and cross-checked
#     against TensorOperations/src/implementation/strided.jl's own
#     `StridedNative`/`StridedBLAS` methods (tensoradd! line 15, tensortrace!
#     line 38, tensorcontract! lines 61 and 94), which both give `allocator` a
#     `= DefaultAllocator()` default at that same arity -- so, like
#     src/tensoroperations.jl's own QuasiStridedBackend methods, tensorcontract!
#     is split into two methods here (one pinned to `DefaultAllocator`, one
#     generic) to mirror that existing dispatch structure, while
#     tensoradd!/tensortrace! each stay a single method with a default
#     `allocator` argument.
# ----------------------------------------------------------------------------

function TensorOperations.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::TensorOperations.Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::TensorOperations.Index2Tuple, conjB::Bool,
        pAB::TensorOperations.Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedComposite,
        allocator::TensorOperations.DefaultAllocator = TensorOperations.DefaultAllocator()
    )
    return TensorOperations.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, QuasiStridedBackend(), allocator
    )
end

function TensorOperations.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::TensorOperations.Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::TensorOperations.Index2Tuple, conjB::Bool,
        pAB::TensorOperations.Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedComposite, allocator
    )
    return TensorOperations.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, QuasiStridedBackend(), allocator
    )
end

function TensorOperations.tensoradd!(
        C,
        A, pA::TensorOperations.Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedComposite, allocator = TensorOperations.DefaultAllocator()
    )
    return TensorOperations.tensoradd!(C, A, pA, conjA, α, β, backend.addtrace, allocator)
end

function TensorOperations.tensortrace!(
        C,
        A, p::TensorOperations.Index2Tuple, q::TensorOperations.Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedComposite, allocator = TensorOperations.DefaultAllocator()
    )
    return TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, backend.addtrace, allocator)
end
