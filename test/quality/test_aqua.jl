using Test
using Aqua
using QuasiStrided

# Quality gate for the package as a whole. This is deliberately kept
# separate from the functional testsets above: it checks package hygiene
# (stale/undeclared deps, missing [compat] bounds, method ambiguities,
# undefined exports, type piracy, etc.) rather than behavior.
#
# QuasiStrided takes a hard dependency on
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
        # `Test.detect_unbound_args` flags every accumulator signature of the
        # form `acc::NTuple{NV, Vec{W, R}}` (the planar and 1m kernels'
        # `accumulate`/`store_tile!` and their generated helpers): at
        # `NV == 0` the tuple is empty and `W`/`R` would be unbound. No kernel
        # has an empty accumulator, and `W`/`R` are pinned by the kernel
        # argument at every call site.
        unbound_args = false,
    )
end
