@testset "KernelDescriptor" begin
    k = KernelDescriptor(Val(8), Val(6), Float64)
    @test mr(k) == 8
    @test nr(k) == 6
    @test scalartype(k) == Float64

    @test packed_a_offset(k, 0, 0) == 0
    @test packed_a_offset(k, 3, 0) == 3
    @test packed_a_offset(k, 3, 2) == 3 + 8 * 2
    @test packed_b_offset(k, 0, 0) == 0
    @test packed_b_offset(k, 5, 0) == 5
    @test packed_b_offset(k, 5, 2) == 5 + 6 * 2

    @test packed_a_length(k, 0) == 0
    @test packed_a_length(k, 4) == 32
    @test packed_b_length(k, 4) == 24

    @test_throws ArgumentError KernelDescriptor(Val(0), Val(6), Float64)
    @test_throws ArgumentError KernelDescriptor(Val(8), Val(-1), Float64)
    @test_throws ArgumentError KernelDescriptor(Val(8), Val(6), Int)
end
