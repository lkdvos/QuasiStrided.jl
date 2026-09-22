using Test
using Aqua
using QuasiStrided

# Quality gate for the package as a whole. This is deliberately kept
# separate from the functional testsets above: it checks package hygiene
# (stale/undeclared deps, missing [compat] bounds, method ambiguities,
# undefined exports, type piracy, etc.) rather than behavior.
#
# Context (docs/decisions.md, "TensorOperations integration milestone:
# Phase A direction freeze"): QuasiStrided takes a hard dependency on
# TensorOperations and adds methods to TO's own generic functions
# (`TO.tensorcontract!`, `TO.tensoradd!`, `TO.tensortrace!`) dispatching on
# a QuasiStrided-owned type (`QuasiStridedBackend`). That is not type
# piracy (the dispatch type is ours), but Aqua's piracy check is exactly
# the mechanism that would catch it if it ever accidentally became piracy,
# so it is left enabled below.
@testset "Aqua" begin
    Aqua.test_all(
        QuasiStrided;
        # Every flagged pair is between methods of upstream packages
        # (TensorOperations, StridedViews, SIMD, TupleTools) and touches
        # nothing QuasiStrided defines; re-litigating those is not our job.
        ambiguities = false,
        # False positive of `Test.detect_unbound_args` on
        # `src/kernels/simd.jl`'s `_acc_lane(::NTuple{NV,Vec{W,T}}, ...,
        # ::Val{NVECA})`: with a `Vararg`-shaped argument in play the detector
        # stops seeing that `NVECA` is pinned by a separate, ordinary
        # argument, which it always is at every call site.
        unbound_args = false,
    )
end
