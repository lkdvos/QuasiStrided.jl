module QuasiStrided

using LinearAlgebra
using StridedViews: StridedView, offset

# --- Phase 1: indexing ---
include("axis_group.jl")

# --- Phase 2: tiles, packing, scalar kernel ---
include("kernel_descriptor.jl")

include("tiles.jl")

include("packing.jl")

include("kernel.jl")

# --- Phase 3: SIMD kernel + serial driver ---
include("kernels/simd.jl")

# --- Macro-blocking milestone: BLIS five-loop driver ---
include("blocking.jl")

include("workspace.jl")

include("driver.jl")

# --- TensorOperations integration milestone: QuasiStridedBackend adapter ---
include("tensoroperations.jl")

export QuasiStridedBackend

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public, :contract!, :plan_contract, :execute!, :ContractPlan,
            :ContractWorkspace, :Blocking, :default_blocking,
            :ScalarKernel, :SIMDKernel
        )
    )
end

end # module QuasiStrided
