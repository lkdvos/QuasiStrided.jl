# TensorOperations adapter: `QuasiStridedBackend`. Frozen contract:
# docs/decisions.md, "TensorOperations integration milestone: Phase A direction
# freeze". This is the whole TensorOperations-facing surface; the engine
# (src/driver.jl and everything it includes) knows nothing about TO.
#
# Frozen import convention: every TensorOperations name is written `TO.<name>`
# (a bare `using TensorOperations` would collide on `scalartype`), except the
# TO-exported `Index2Tuple` and `linearize`.
import TensorOperations as TO
using TensorOperations: Index2Tuple, linearize
import TupleTools
using StridedViews: StridedView, isstrided

# ----------------------------------------------------------------------------
# Backend marker
# ----------------------------------------------------------------------------

"""
    QuasiStridedBackend()

TensorOperations backend routing a contraction to QuasiStrided's BLIS-style
macro-blocking engine ([`QuasiStrided.contract!`](@ref)):

    @tensor backend = QuasiStridedBackend() C[i, j] := A[i, k] * B[k, j]

Contraction only, and only for strided operands sharing a single element type
out of `Float32`/`Float64`; everything else (`TensorOperations.tensoradd!`,
`TensorOperations.tensortrace!`, mixed or unsupported eltypes, a non-strided
operand, an output aliased with an input) throws an `ArgumentError`. It is not
registered with `TensorOperations.select_backend` and never falls back to
another backend. Rationale is frozen in docs/decisions.md, "TensorOperations
integration milestone: Phase A direction freeze".
"""
struct QuasiStridedBackend <: TO.AbstractBackend end

# ----------------------------------------------------------------------------
# Task-local workspace pooling
# ----------------------------------------------------------------------------

# One `task_local_storage` slot (not `threadid()`) holding a
# `Dict{DataType,ContractWorkspace}` keyed by scalar type; see
# docs/decisions.md, Amendment 1 and its workspace/allocator design-constraints
# section.
const _QS_WORKSPACE_KEY = :quasistrided_contract_workspaces

@inline function _qs_workspace_pool()
    return get!(task_local_storage(), _QS_WORKSPACE_KEY) do
        return Dict{DataType, ContractWorkspace}()
    end::Dict{DataType, ContractWorkspace}
end

# Fetch (or lazily build) this task's persistent, `reserve!`-able
# `ContractWorkspace{T,Vector{T}}` for scalar type `T`. Built once per
# `(task, T)` at `T`'s own default kernel/blocking; `plan_contract` grows it
# to whatever blocking the actual call needs via `reserve!`.
@inline function _qs_task_workspace(::Type{T}) where {T}
    pool = _qs_workspace_pool()
    ws = get(pool, T, nothing)
    ws === nothing || return ws::ContractWorkspace{T, Vector{T}}
    kernel = _default_kernel(T)
    new_ws = ContractWorkspace(T, kernel, default_blocking(kernel), false, TO.DefaultAllocator())
    pool[T] = new_ws
    return new_ws
end

# ----------------------------------------------------------------------------
# Label mapping
# ----------------------------------------------------------------------------

"""
    _qs_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple) -> (indA, indB, indC)

Translate TensorOperations' `pA`/`pB`/`pAB` index specification into
QuasiStrided's label convention (one `Int` label per axis, in axis order).

TensorOperations computes `C = β*C + α*permutedims(contract(opA(A), opB(B)), pAB)`,
contracting the axes `pA[2]` of `A` with `pB[1]` of `B` and permuting the
remaining `(pA[1]..., pB[2]...)` by `pAB`. The produced alphabet is `1:NoA` for
`A`'s open axes (in `pA[1]` order), `NoA+1 : NoA+NoB` for `B`'s open axes (in
`pB[2]` order), and `-1:-1:-Nk` for the contracted pairs (in the positionally
matched `pA[2]`/`pB[1]` order). `indC` is `linearize(pAB)` verbatim: the
intermediate tensor's slot `j` carries label `j` by construction, and
`permutedims`' convention is "output axis `c` takes input axis `perm[c]`".

Worked example (frozen in docs/decisions.md, verified numerically):

```
pA  = ((3,1,4),(2,5))  ->  indA = ( 2, -1,  1,  3, -2)
pB  = ((3,1),(2,4))    ->  indB = (-2,  4, -1,  5)
pAB = ((4,2),(5,1,3))  ->  indC = ( 4,  2,  5,  1,  3)
```

Performs **no** validation of its own: `numout(pA) + numin(pB) == numind(pAB)`
and `numin(pA) == numout(pB)` are `TO.argcheck_tensorcontract`'s job, and the
adapter calls that first.
"""
function _qs_labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple)
    NoA, Nk = TO.numout(pA), TO.numin(pA)
    qA = TupleTools.invperm(linearize(pA))
    qB = TupleTools.invperm(linearize(pB))
    indA = map(s -> s <= NoA ? s : -(s - NoA), qA)
    indB = map(s -> s <= Nk ? -s : NoA + (s - Nk), qB)
    return indA, indB, linearize(pAB)
end

# ----------------------------------------------------------------------------
# Eligibility and argument checking
# ----------------------------------------------------------------------------

# The two clauses of the frozen eligibility predicate, split out only so the
# rejection message can name the one that failed. `_qs_eligible` below is the
# predicate itself; nothing else in this file re-states it.
_qs_eltype_ok(C, A, B) =
    eltype(A) === eltype(B) === eltype(C) && eltype(C) ∈ (Float32, Float64)
_qs_strided_ok(C, A, B) = all(isstrided, (A, B, C))

"""
    _qs_eligible(C, A, B) -> Bool

Whether `QuasiStridedBackend` can serve `tensorcontract!(C, A, ..., B, ...)`:
a single shared element type out of `Float32`/`Float64`, and all three operands
strided. Ineligible inputs are rejected outright (see [`QuasiStridedBackend`](@ref)),
never routed to another backend.
"""
_qs_eligible(C, A, B) = _qs_eltype_ok(C, A, B) && _qs_strided_ok(C, A, B)

@noinline _qs_throw(msg::AbstractString) = throw(ArgumentError(msg))

@noinline function _qs_throw_unsupported(f)
    return _qs_throw(
        "QuasiStridedBackend implements contraction only, so $f is not " *
            "supported. Use a different backend (e.g. backend=StridedNative()) for " *
            "networks that need an addition or trace step."
    )
end

# Step 1 of the frozen argument-checking order: hard-reject every ineligible
# input class before any TensorOperations check runs. `_qs_eligible` is the
# gate; the clause checks below it only exist to name the failure.
@noinline function _qs_check_eligible(f, C, A, B)
    _qs_eligible(C, A, B) && return nothing
    _qs_eltype_ok(C, A, B) || _qs_throw(
        "QuasiStridedBackend requires all tensors of $f to share a single " *
            "element type out of Float32 and Float64, got " *
            join(map(eltype, (C, A, B)), ", ")
    )
    _qs_strided_ok(C, A, B) || _qs_throw(
        "QuasiStridedBackend requires strided arrays for $f, got " *
            join(map(typeof, (C, A, B)), ", ")
    )
    return nothing
end

# ----------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------

# Shared prefix of both `tensorcontract!` methods below, in the frozen order
# (docs/decisions.md, "Required argument-checking order in the adapter", plus
# its `StridedView`-based-aliasing addendum): eligibility, argcheck, dimcheck,
# wrap, aliasing. The engine itself performs no aliasing check at all.
#
# LOAD-BEARING: `conjA`/`conjB` are dropped and `StridedView.op` ignored,
# correct only because the eligibility gate pins the element type to real
# Float32/Float64, on which every `op` and a conjugated α/β are the identity
# (docs/decisions.md, "Eligibility predicate, and the conjugation invariant").
@inline function _qs_prepare(C, A, pA, B, pB, pAB, α, β)
    _qs_check_eligible(TO.tensorcontract!, C, A, B)
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    Cv, Av, Bv = StridedView(C), StridedView(A), StridedView(B)

    # Aliasing is tested on the *wrapped* views: Base has no `dataids` for
    # `PermutedDimsArray`, but `StridedView` unwraps to the shared parent and
    # forwards `dataids` to it, so this catches cases raw arrays would miss.
    (Base.mightalias(Cv, Av) || Base.mightalias(Cv, Bv)) && _qs_throw(
        "output tensor must not be aliased with an input tensor in $(TO.tensorcontract!)"
    )

    # Equivalent to `TO.standardize_scalartype` here. Discarding `Zero()`/
    # `One()`'s strong semantics is safe: every kernel branches on
    # `iszero(alpha)`/`iszero(beta)` rather than evaluating `β * C`.
    T = eltype(C)
    indA, indB, indC = _qs_labels(pA, pB, pAB)
    return Cv, Av, Bv, indA, indB, indC, convert(T, α), convert(T, β)
end

"""
    TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β,
                                     ::QuasiStridedBackend, allocator)

Compute `C = β*C + α*permutedims(contract(A, B), pAB)` with QuasiStrided's
macro-blocking engine.

Throws an `ArgumentError` unless every operand is strided and they share a
single element type out of `Float32`/`Float64`, and unless `C` is unaliased
with both `A` and `B`; there is no fallback to another backend. See
[`QuasiStridedBackend`](@ref) for the full eligibility contract, and
[`_qs_labels`](@ref) for the index translation.

Under `TensorOperations.DefaultAllocator` the buffers come from a persistent,
`reserve!`-grown task-local [`ContractWorkspace`](@ref); any other allocator
gets a workspace scoped to the single call. Dispatching on the allocator type,
rather than branching on its value, mirrors `_resolve_workspace` in
src/driver.jl and keeps both paths concretely typed.
"""
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend,
        allocator::TO.DefaultAllocator = TO.DefaultAllocator()
    )
    Cv, Av, Bv, indA, indB, indC, α′, β′ = _qs_prepare(C, A, pA, B, pB, pAB, α, β)
    # `plan_contract` `reserve!`s the pooled workspace itself. oracle=false:
    # the backend path never needs `execute_tilewise!`'s buffers.
    plan = plan_contract(
        Cv, Av, indA, Bv, indB, indC;
        workspace = _qs_task_workspace(eltype(C)), allocator = allocator, oracle = false
    )
    execute!(plan, α′, β′)
    return C
end

function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator
    )
    Cv, Av, Bv, indA, indB, indC, α′, β′ = _qs_prepare(C, A, pA, B, pB, pAB, α, β)
    # Explicit allocator: a workspace scoped to this call, never the pool,
    # bracketed with checkpoint!/reset! exactly as TO's own `blas_contract!`
    # brackets its temporaries so it composes with the surrounding network.
    checkpoint = TO.allocator_checkpoint!(allocator)
    plan = plan_contract(
        Cv, Av, indA, Bv, indB, indC;
        workspace = nothing, allocator = allocator, oracle = false
    )
    try
        execute!(plan, α′, β′)
    finally
        release!(plan.workspace, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end
    return C
end

# `tensoradd!`/`tensortrace!` hard-reject: QuasiStrided has no analog of
# either. Deliberately untyped in `C`/`A` so every call is rejected with this
# message rather than TO's generic "unknown backend" error.

"""
    TensorOperations.tensoradd!(C, A, pA, conjA, α, β, ::QuasiStridedBackend, allocator)

Always throws `ArgumentError`: `QuasiStridedBackend` implements contraction
only ([`TensorOperations.tensorcontract!`](@ref)). QuasiStrided has no
analog of a standalone add/permute step, so there is nothing to delegate to,
and this backend never falls back to another one (see
[`QuasiStridedBackend`](@ref)). Use a different `backend=` for a `@tensor`
network that needs this step.
"""
function TO.tensoradd!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator = TO.DefaultAllocator()
    )
    return _qs_throw_unsupported(TO.tensoradd!)
end

"""
    TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, ::QuasiStridedBackend, allocator)

Always throws `ArgumentError`: `QuasiStridedBackend` implements contraction
only ([`TensorOperations.tensorcontract!`](@ref)). QuasiStrided has no
trace/diagonal support at all (`_classify_labels` in `src/driver.jl` rejects
repeated labels), so there is nothing to delegate to, and this backend never
falls back to another one (see [`QuasiStridedBackend`](@ref)). Use a
different `backend=` for a `@tensor` network that needs a trace step.
"""
function TO.tensortrace!(
        C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::QuasiStridedBackend, allocator = TO.DefaultAllocator()
    )
    return _qs_throw_unsupported(TO.tensortrace!)
end
