# ----------------------------------------------------------------------------
# Conjugation: folding `conjA`/`conjB` with each view's `.op`
# ----------------------------------------------------------------------------
# These live in the ENGINE, not the adapter, because the invariant is an engine
# invariant: the engine never indexes through a `StridedView` (`_plan_contract`
# takes `parent`/`offset`), so a view's `.op` is silently dropped on all three
# operands unless folded in here -- and a caller reaching `plan_contract`
# directly, with no adapter in sight, is exposed to the same silent wrongness.
#
# GUARDRAIL: a TOTAL table with a throwing fallback, NOT TensorOperations'
# TBLIS extension's `A.op === conj` test. `StridedView(p, sz, st, off,
# adjoint)` is directly constructible, and `=== conj` classifies it as
# *unconjugated* -- exactly the silent wrong answer this table exists to
# prevent. The fallback is `@noinline` and unreachable for every `op`
# `StridedViews` itself constructs, so totality costs nothing.
_op_conjugates(::typeof(identity)) = false
_op_conjugates(::typeof(conj)) = true
# Elementwise identity on a `Number`: these permute axes, they do not touch
# values, and the engine has already resolved axes into `AxisGroup`s.
_op_conjugates(::typeof(transpose)) = false
_op_conjugates(::typeof(adjoint)) = true
@noinline _op_conjugates(f) = throw(
    ArgumentError(
        "unsupported StridedView.op $f: QuasiStrided folds a view's `op` into the " *
            "packing transform and recognizes only identity/conj/transpose/adjoint"
    )
)

# GUARDRAIL: `⊻`, not `||`. The flag and the view's `op` are two INDEPENDENT
# requests to conjugate the same data, and `conj` is involutive, so applying
# both is the identity and only the parity survives. `false` unconditionally
# for a real element type -- `StridedViews` defines
# `conj(::StridedView{<:Real}) = a`, so a real view's `op` can never conjugate
# anyway, and the real path therefore always gets `identity` and no new
# `execute!` specialization, even with `conjA = true`.
_qs_isconj(v::StridedView{T}, flag::Bool) where {T} =
    (T <: Complex) && (flag ⊻ _op_conjugates(v.op))
