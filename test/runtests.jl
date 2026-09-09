using Test
using Random
using QuasiStrided

@testset "QuasiStrided.jl" begin
    include("test_axis_group.jl")
    include("strided_integration.jl")
    include("test_kernel_descriptor.jl")
    include("test_packing.jl")
    include("test_kernel.jl")
    include("test_simd_kernel.jl")
    include("test_phase2_integration.jl")
    include("test_driver.jl")
    include("test_phase3_integration.jl")
    include("test_macro_driver.jl")
end
