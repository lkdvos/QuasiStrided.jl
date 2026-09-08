# OWNER: main process. Phase 2 integration gate: exercises the real
# AxisGroup -> BlockDescriptor -> axis_from_descriptor -> QSTile ->
# pack_a!/pack_b! -> ScalarKernel execute_tile! path end to end, on the
# worked A[a,k,b]/B[k,n]/C[a,n,b] fixture from both specs (indexing spec
# section 8, microkernel spec section 11), including multiple K panels
# applying beta once (microkernel spec section 10's contract, without the
# not-yet-written driver: this test drives the loop by hand once to prove
# the pieces compose before Phase 3 builds the general version).

@testset "Phase 2 integration: AxisGroup -> tiles -> packing -> ScalarKernel" begin
    # A[a,k,b] shape (3,5,2), B[k,n] shape (5,4), C[a,n,b] shape (3,4,2),
    # column-major, C[a,n,b] = sum_k A[a,k,b]*B[k,n].
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    B = reshape(collect(1.0:20.0), 5, 4)
    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, a in 1:3
        Cref[a, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end

    M = AxisGroup((3, 2), ((1, 15), (1, 12)))  # A, C
    N = AxisGroup((4,), ((5,), (3,)))          # B, C
    K = AxisGroup((5,), ((3,), (1,)))          # A, B

    @test axis_length(M) == 6
    @test axis_length(N) == 4
    @test axis_length(K) == 5

    kernel = ScalarKernel(Val(8), Val(6), Float64)  # MR=8 >= 6, NR=6 >= 4: one output tile

    m_buf_A = zeros(Int, axis_length(M))
    m_buf_C = zeros(Int, axis_length(M))
    n_buf_B = zeros(Int, axis_length(N))
    n_buf_C = zeros(Int, axis_length(N))
    k_buf_A = zeros(Int, axis_length(K))
    k_buf_B = zeros(Int, axis_length(K))

    (dM_A, dM_C) = block_descriptors!((m_buf_A, m_buf_C), M, 0, axis_length(M))
    (dN_B, dN_C) = block_descriptors!((n_buf_B, n_buf_C), N, 0, axis_length(N))

    C = vec(copy(Cref)) .* 0  # zero, same length as Cref's underlying storage
    Cstorage = zeros(Float64, length(Cref))
    Astorage = vec(A)
    Bstorage = vec(B)

    row_A = axis_from_descriptor(dM_A, m_buf_A)
    row_C = axis_from_descriptor(dM_C, m_buf_C)
    col_B = axis_from_descriptor(dN_B, n_buf_B)
    col_C = axis_from_descriptor(dN_C, n_buf_C)

    destination = DestinationTile(Cstorage, 0, row_C, col_C)
    @test nrows(destination) == 6
    @test ncols(destination) == 4

    packed_a = zeros(Float64, mr(kernel) * axis_length(K))
    packed_b = zeros(Float64, nr(kernel) * axis_length(K))

    # Single K panel covering all of K (kc=5) first, to check basic wiring.
    (dK_A, dK_B) = block_descriptors!((k_buf_A, k_buf_B), K, 0, axis_length(K))
    row_K_A = axis_from_descriptor(dK_A, k_buf_A)
    col_K_B = axis_from_descriptor(dK_B, k_buf_B)

    source_A = SourceTile(Astorage, 0, row_A, row_K_A)
    source_B = SourceTile(Bstorage, 0, col_K_B, col_B)

    pack_a!(packed_a, source_A, kernel, identity)
    pack_b!(packed_b, source_B, kernel, identity)
    execute_tile!(kernel, destination, packed_a, packed_b, axis_length(K), 1.0, 0.0)

    Cout = reshape(Cstorage, size(Cref))
    @test Cout ≈ Cref

    # Multiple K panels (lengths 2,2,1 per microkernel spec section 10/11),
    # each packed/executed independently, beta_effective=0 on the first panel
    # and 1 on the rest -> must reproduce the same result (up to fp rounding
    # from different accumulation grouping, which is expected/allowed).
    fill!(Cstorage, 0.0)
    panel_starts = (0, 2, 4)
    panel_lengths = (2, 2, 1)
    for (idx, (first, len)) in enumerate(zip(panel_starts, panel_lengths))
        (dKp_A, dKp_B) = block_descriptors!((k_buf_A, k_buf_B), K, first, len)
        rowKp_A = axis_from_descriptor(dKp_A, k_buf_A)
        colKp_B = axis_from_descriptor(dKp_B, k_buf_B)
        srcA = SourceTile(Astorage, 0, row_A, rowKp_A)
        srcB = SourceTile(Bstorage, 0, colKp_B, col_B)
        pack_a!(packed_a, srcA, kernel, identity)
        pack_b!(packed_b, srcB, kernel, identity)
        beta_effective = idx == 1 ? 0.0 : 1.0
        execute_tile!(kernel, destination, packed_a, packed_b, len, 1.0, beta_effective)
    end
    @test reshape(Cstorage, size(Cref)) ≈ Cref

    # Nontrivial alpha/beta across the same 3 panels, starting from a nonzero
    # C, applying the original beta exactly once (on the first panel).
    Cstart = rand(MersenneTwister(1234), 3, 4, 2)
    fill!(Cstorage, 0.0)
    copyto!(Cstorage, vec(Cstart))
    alpha = 2.5
    beta = 0.75
    for (idx, (first, len)) in enumerate(zip(panel_starts, panel_lengths))
        (dKp_A, dKp_B) = block_descriptors!((k_buf_A, k_buf_B), K, first, len)
        rowKp_A = axis_from_descriptor(dKp_A, k_buf_A)
        colKp_B = axis_from_descriptor(dKp_B, k_buf_B)
        srcA = SourceTile(Astorage, 0, row_A, rowKp_A)
        srcB = SourceTile(Bstorage, 0, colKp_B, col_B)
        pack_a!(packed_a, srcA, kernel, identity)
        pack_b!(packed_b, srcB, kernel, identity)
        beta_effective = idx == 1 ? beta : 1.0
        execute_tile!(kernel, destination, packed_a, packed_b, len, alpha, beta_effective)
    end
    expected = alpha .* Cref .+ beta .* Cstart
    @test reshape(Cstorage, size(Cref)) ≈ expected
end
