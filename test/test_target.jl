# Hardware detection (src/target.jl) and the register shape / blocking it
# selects. Two properties matter most: detection never throws, and every
# failure path resolves to exactly the pre-detection constants.

using StridedViews: StridedView
using QuasiStrided: TargetProfile, CacheLevel, target_profile, cache_topology,
    unknown_target, _detect_isa, _detect_target, _derived_shape, _legacy_shape,
    _shape_override, _kernel_for, _default_kernel, _legacy_blocking, kernel_shapes,
    _parse_size, _count_cpu_list, NR_DEFAULT

const VALID_ISAS = (:avx512, :avx2, :neon, :unknown)
synthetic(isakey, vb) = TargetProfile(
    isakey, Sys.ARCH, "synthetic", vb, 32, CacheLevel(), CacheLevel(), CacheLevel()
)

# Permuted A / negative-stride B / sliced-with-offset C, the fixture shape
# benchmark/harness.jl uses.
function scattered_fixture(::Type{T}, a_n = 32, k_n = 32, b_n = 8, n_n = 32) where {T}
    A2 = randn(MersenneTwister(11), T, a_n, k_n)
    Aperm = permutedims(StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0), (2, 3, 1))
    Bneg = StridedView(randn(MersenneTwister(12), T, k_n * n_n), (k_n, n_n), (-1, k_n), k_n - 1)
    Cbig = zeros(T, a_n + 2, n_n + 3, b_n + 1)
    Cv = StridedView(view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n))
    return (Cv, Aperm, (2, 3, 1), Bneg, (2, 4), (1, 4, 3))
end

@testset "target detection" begin
    @testset "runs on this host without throwing" begin
        p = target_profile()
        @test p isa TargetProfile
        @test p.isa in VALID_ISAS
        @test p.arch === Sys.ARCH
        @test _detect_target().isa === p.isa   # detection is per-process
        @test _detect_isa() in VALID_ISAS
        # vector_bytes/nregisters are 0 exactly when the ISA is unknown.
        if p.isa === :unknown
            @test (p.vector_bytes, p.nregisters) == (0, 0)
        else
            @test p.vector_bytes > 0
            @test p.nregisters > 0
            @test ispow2(p.vector_bytes)
        end
    end

    @testset "CPUID probe is an independent fallback, not decoration" begin
        # The uarch table is an optimization; the probe is what makes an
        # *unlisted* CPU still get the right shape -- the portability claim.
        probe = QuasiStrided._isa_from_cpuid()
        @test probe in (:avx512, :avx2, :unknown)
        if Sys.ARCH === :x86_64
            table = get(QuasiStrided._UARCH_ISA, Sys.CPU_NAME, :miss)
            (table === :miss || probe === :unknown) || @test probe === table
            # Base's CPUID submodule is undocumented: if these names move,
            # `_isa_from_cpuid` silently returns :unknown and every unlisted
            # x86 CPU quietly loses the derived shape. Fail loudly instead.
            C = Base.BinaryPlatforms.CPUID
            @test isdefined(C, :JL_X86_avx512f)
            @test isdefined(C, :JL_X86_avx2)
        end
    end

    @testset "cache topology is optional and non-negative" begin
        for _ in 1:2   # must not throw when called repeatedly (shells out on macOS)
            topo = cache_topology()
            @test topo === nothing || topo isa NamedTuple
            topo === nothing && continue
            for lvl in (topo.l1d, topo.l2, topo.l3)
                @test lvl isa CacheLevel
                # Any field may be 0 for "not detected", never negative.
                @test all(>=(0), (lvl.bytes, lvl.ways, lvl.line, lvl.sharing))
            end
        end
    end

    @testset "sysfs parsing helpers" begin
        for (str, want) in (
                "32K" => 32 * 1024, "1024K" => 1024 * 1024,
                "25344K" => 25344 * 1024, "2M" => 2 * 1024 * 1024,
                "512" => 512, "" => 0, "garbage" => 0,
            )
            @test _parse_size(str) == want
        end
        # "0,16" is this machine's L2 (SMT pair); "0-7,16-23" its shared L3.
        for (str, want) in ("0,16" => 2, "0-7,16-23" => 16, "3" => 1, "" => 0, "0-3" => 4)
            @test _count_cpu_list(str) == want
        end
    end

    @testset "unknown target is bit-identical to the pre-detection defaults" begin
        u = unknown_target()
        @test u.isa === :unknown
        for T in (Float64, Float32)
            @test _derived_shape(u, T) === _legacy_shape(T)
            k = _kernel_for(u, T)
            @test k isa SIMDKernel
            @test (mr(k), nr(k), lanewidth(k)) === _legacy_shape(T)
        end
        # The exact constants from docs/decisions.md Phase E.
        @test _legacy_blocking(Float64) === Blocking(64, 128, 768)
        @test _legacy_blocking(Float32) === Blocking(96, 384, 1152)
        for key in (:unknown, :avx2, :neon, :somethingelse), T in (Float64, Float32)
            @test default_blocking(Val(key), T) === _legacy_blocking(T)
        end
        # The documented type-argument form is unchanged by detection.
        @test default_blocking(Float64) === _legacy_blocking(Float64)
        @test default_blocking(Float32) === _legacy_blocking(Float32)
    end

    @testset "derivation rule: MR = 2W, NR = NR_DEFAULT, NV = 12" begin
        # NV = 12 everywhere keeps the rule safe on a 16-register AVX2 machine
        # and on Julia 1.10, and after Phase H it is also the measured optimum.
        for (isakey, vb) in ((:avx512, 64), (:avx2, 32)), T in (Float64, Float32)
            MR, NR, W = _derived_shape(synthetic(isakey, vb), T)
            @test W == vb ÷ sizeof(T)
            @test (MR, NR) == (2 * W, NR_DEFAULT)
            @test (MR ÷ W) * NR == 12
            @test MR % W == 0   # the packing/kernel contract
        end
        # AVX2 reproduces the old hardcoded shape, so an AVX2 machine sees no
        # change from this work.
        @test _derived_shape(synthetic(:avx2, 32), Float64) === (8, 6, 4)
        @test _derived_shape(synthetic(:avx2, 32), Float64) === _legacy_shape(Float64)
    end

    @testset "the rule applies only where it was measured" begin
        # :neon is detected but unmeasured, so it gets the legacy shape rather
        # than an invented one -- the rule would pick MR = 2W = 4 on 128-bit
        # lanes, narrower and smaller than legacy with nothing to justify it.
        # This is also what keeps the shipped default identical on aarch64,
        # which test_driver.jl's "SIMDKernel is the default" testset pins.
        for T in (Float64, Float32)
            @test _derived_shape(synthetic(:neon, 16), T) === _legacy_shape(T)
            @test _derived_shape(synthetic(:unknown, 0), T) === _legacy_shape(T)
            for vb in (16, 32, 64)   # width must not matter for an unmeasured ISA
                @test _derived_shape(synthetic(:neon, vb), T) === _legacy_shape(T)
            end
        end
    end

    @testset "the override hook is empty, and the rule is what ships" begin
        # After Phase H the rule is at the measured optimum, so no override row
        # is justified. Update this deliberately rather than deleting it: an
        # override fitting one machine's noise is what it guards against.
        for T in (Float64, Float32)
            for key in VALID_ISAS
                @test _shape_override(Val(key), T) === nothing
            end
            W = 64 ÷ sizeof(T)
            @test _derived_shape(synthetic(:avx512, 64), T) === (2 * W, NR_DEFAULT, W)
            @test _derived_shape(synthetic(:avx512, 64), T) in kernel_shapes(T)
        end
    end

    @testset "every shipped shape is constructible" begin
        for T in (Float64, Float32), (MR, NR, W) in kernel_shapes(T)
            @test MR % W == 0
            k = SIMDKernel(Val(MR), Val(NR), T, Val(W))
            @test (mr(k), nr(k), lanewidth(k)) == (MR, NR, W)
            # Register-budget sanity only: NV is capped by neither panel
            # addressing nor scattered-axis boxing any more (Phase H).
            @test (MR ÷ W) * NR + (MR ÷ W) <= 32
        end
        for T in (Float64, Float32)
            k = _default_kernel(T)
            @test (mr(k), nr(k), lanewidth(k)) in kernel_shapes(T)
            @test scalartype(k) === T
        end
    end

    @testset "shipped defaults are allocation-free on a SCATTERED destination" begin
        # The gap that let a 24576 B/call regression ship: every other
        # allocation assertion uses plain regular contractions, and shapes with
        # MR > 16 allocated only on an irregular destination (Phase H). A
        # scattered 3-index case is what this engine exists for.
        for T in (Float64, Float32)
            plan = QuasiStrided.plan_contract(scattered_fixture(T)...)
            QuasiStrided.execute!(plan, one(T), zero(T))
            QuasiStrided.execute!(plan, one(T), zero(T))
            allocated = @allocated QuasiStrided.execute!(plan, one(T), zero(T))
            @test allocated == 0 skip = (VERSION < v"1.11")
        end
    end

    @testset "extent-aware demotion when M cannot fill a register tile" begin
        for T in (Float64, Float32)
            MR = mr(_default_kernel(T))
            @test mr(_default_kernel(T, 4 * MR, 256)) == MR   # Qm >= MR keeps it
            small = _default_kernel(T, 1, 256)                # Qm < MR demotes
            @test (mr(small), nr(small), lanewidth(small)) === _legacy_shape(T)
            @test mr(_default_kernel(T, 0, 256)) == MR        # empty must not demote
        end
    end
end
