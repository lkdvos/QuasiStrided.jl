using Test
using QuasiStrided

@testset "QuasiStrided.jl" begin
    include("test_axis_group.jl")
    include("strided_integration.jl")
    include("test_kernel_descriptor.jl")
    include("test_packing.jl")
    include("test_kernel.jl")
    include("test_driver.jl")
end
