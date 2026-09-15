module QuasiStrided

using LinearAlgebra
using StridedViews: StridedView, offset

# --- Hardware detection (pure; no kernel or blocking dependencies) ---
include("target.jl")

# --- Phase 1: indexing ---
include("axis_group.jl")

# --- Phase 2: tiles, packing, scalar kernel ---
include("kernel_descriptor.jl")

# --- Complex milestone: packed formats, methods, the complex descriptor ---
include("complex_format.jl")

include("panel.jl")

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

# Hardware detection runs once per process, never at precompile time: a .ji
# cached on one node class of a shared depot must not carry another node's
# feature set (src/target.jl).
function __init__()
    _init_target!()
    return nothing
end

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public, :contract!, :plan_contract, :execute!, :ContractPlan,
            :ContractWorkspace, :Blocking, :default_blocking,
            :ScalarKernel, :SIMDKernel, :target_profile, :cache_topology,
            :TargetProfile, :CacheLevel
        )
    )
end

end # module QuasiStrided
