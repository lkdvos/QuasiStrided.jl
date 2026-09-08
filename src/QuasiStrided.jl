module QuasiStrided

using LinearAlgebra
using StridedViews: StridedView, offset

# --- Phase 1: indexing ---
include("axis_group.jl")
export AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor, describe_block,
    block_descriptors!, normalize_group

# --- Phase 2: tiles, packing, scalar kernel ---
include("kernel_descriptor.jl")
export KernelDescriptor, mr, nr, scalartype, packed_a_offset, packed_b_offset,
    packed_a_length, packed_b_length

include("tiles.jl")
export AffineAxis, ScatterAxis, SourceTile, DestinationTile,
    axis_from_descriptor, nrows, ncols

include("packing.jl")
export pack_a!, pack_b!

include("kernel.jl")
export ScalarKernel, zero_accumulator, accumulate, scale_tile!, store_tile!, execute_tile!

# --- Phase 3: SIMD kernel + serial driver ---
include("kernels/simd.jl")

include("driver.jl")
export contract!

end # module QuasiStrided
