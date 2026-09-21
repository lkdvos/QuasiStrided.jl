using Test
using Random
using QuasiStrided
# QuasiStrided.jl un-exports its internal/public-unexported tiers (see
# docs/decisions.md, "Public / internal API split: three tiers"); this test
# suite still references all of them unqualified across its included files,
# so restore them here once rather than editing each file individually.
using QuasiStrided: AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor,
    describe_block, block_descriptors!, normalize_group, KernelDescriptor, mr, nr,
    scalartype, packed_a_offset, packed_b_offset, packed_a_length, packed_b_length,
    AffineAxis, ScatterAxis, SourceTile, DestinationTile, axis_from_descriptor, nrows,
    ncols, axis_offset_range, checked_tile_storage_bounds, pack_a!, pack_b!,
    zero_accumulator, accumulate, scale_tile!, store_tile!, execute_tile!, lanewidth,
    avecs_per_column, contract!, Blocking, default_blocking, ScalarKernel, SIMDKernel
# plan_contract, execute! and ContractPlan are intentionally NOT restored here:
# test_driver.jl (included below) introduces its own unqualified `plan_contract`/
# `execute!`/`ContractPlan` bindings via `const ... = QuasiStrided.<name>`, and
# test_macro_driver.jl deliberately writes `QuasiStrided.<name>` to avoid
# colliding with those (see its header comment).

@testset "QuasiStrided.jl" begin
    include("test_target.jl")
    include("test_axis_group.jl")
    include("strided_integration.jl")
    include("test_kernel_descriptor.jl")
    include("test_packing.jl")
    include("test_packing_complex.jl")
    include("test_kernel.jl")
    include("test_simd_kernel.jl")
    include("test_planar_kernel.jl")
    include("test_onem_kernel.jl")
    include("test_phase2_integration.jl")
    include("test_driver.jl")
    include("test_phase3_integration.jl")
    include("test_macro_driver.jl")
    include("test_per_call_floor.jl")
    # Included last: it does a bare `using TensorOperations`, which exports
    # its own `scalartype` and would otherwise conflict with the `scalartype`
    # restored from QuasiStrided above for any file included after it.
    include("test_tensoroperations.jl")
    # Quality gate (Aqua): runs last since it audits the *whole* loaded
    # package/dependency graph after every other testset has already forced
    # everything to load, rather than gating functional tests on it.
    include("test_quality.jl")
end
