# Hardware detection (src/target.jl) and the register shape / blocking it
# selects. Two properties matter most: detection never throws, and every
# failure path resolves to exactly the pre-detection constants.

using StridedViews: StridedView
using QuasiStrided: TargetProfile, CacheLevel, target_profile, cache_topology,
    unknown_target, _detect_isa, _detect_target, _derived_shape, _legacy_shape,
    _shape_override, _kernel_for, _default_kernel, _legacy_blocking, kernel_shapes,
    _parse_size, _count_cpu_list, NR_DEFAULT, _rule_applies, _rule_applies_complex,
    _isa_nregisters, packed_a_per_k, packed_b_per_k, realtype, complex_method,
    RealMethod, PlanarMethod, OneMMethod, accumulator_planes, a_reals, b_reals


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

end
