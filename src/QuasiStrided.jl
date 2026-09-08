module QuasiStrided

using LinearAlgebra
using StridedViews: StridedView, offset

# --- Phase 1: indexing ---
include("axis_group.jl")
export AxisGroup, axis_length, offsets, fill_offsets!, BlockDescriptor, describe_block,
    block_descriptors!, normalize_group

# --- Phase 2: tiles, packing, scalar kernel ---
include("tiles.jl")
export AffineAxis, ScatterAxis, SourceTile, DestinationTile

include("packing.jl")
export pack_a!, pack_b!

include("kernel.jl")
export zero_accumulator, accumulate, store_tile!, execute_tile!

# --- Phase 3: SIMD kernel + serial driver ---
include("kernels/simd.jl")

include("driver.jl")
export contract!

end # module QuasiStrided
