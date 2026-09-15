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

# ============================================================================
# Complex element types: the shape rule, the menus and the blocking derivation.
#
# Nothing here constructs a complex kernel -- PlanarKernel/OneMKernel do not
# exist yet, and `_default_kernel(::Type{<:Complex})` deliberately throws until
# they do. Everything below is a pure function of a `TargetProfile`, an element
# type and a `ComplexMethod`, which is exactly what makes it testable now.
# ============================================================================

@testset "packed_*_per_k is the identity on every shipped real kernel" begin
    # The pin behind substituting `packed_a_per_k`/`packed_b_per_k` for
    # `mr`/`nr` at the driver's four `_sliver_panel` call sites: for a real
    # kernel the two are the same number, so that substitution cannot change
    # the real path. Argued in docs/decisions.md, "Three meanings of `T`,
    # pinned"; pinned here rather than argued.
    for T in (Float64, Float32), (MR, NR, W) in kernel_shapes(T)
        k = SIMDKernel(Val(MR), Val(NR), T, Val(W))
        @test packed_a_per_k(k) === mr(k) === MR
        @test packed_b_per_k(k) === nr(k) === NR
        @test realtype(k) === T === scalartype(k)
        @test complex_method(k) === RealMethod()
    end
    for T in (Float64, Float32)
        for k in (_default_kernel(T), ScalarKernel(Val(4), Val(3), T), SIMDKernel(Val(8), Val(6), T))
            @test packed_a_per_k(k) === mr(k)
            @test packed_b_per_k(k) === nr(k)
            @test realtype(k) === scalartype(k)
            # Panel length in reals equals the logical count, for the same reason.
            @test packed_a_length(k, 7) === mr(k) * 7
            @test packed_b_length(k, 7) === nr(k) * 7
        end
    end
end

@testset "complex derivation rule: MR = 2W, with W taken from the REAL type" begin
    # The same one-line Phase G rule; only the `sizeof` argument moves to
    # `real(T)`, because W is a count of real lanes. Tested here as the *rule*,
    # separately from the override that Phase F layered on top of it -- the two
    # are independent and conflating them is what made the original version of
    # this testset fail when the override landed.
    for (T, W, swept) in ((ComplexF64, 8, (24, 3, 8)), (ComplexF32, 16, (48, 3, 16)))
        @test W == 64 ÷ sizeof(real(T))

        # The rule itself, with the override factored out.
        @test QuasiStrided._complex_rule_shape(64, T) === (2 * W, NR_DEFAULT, W)

        # `_shape_override` on AVX-512 carries the Phase F sweep's winner, and
        # is what `_derived_shape` therefore returns. This is the ONE swept row
        # in the package: measured on ccqlin038 at 21 reps and a 0.4% canary
        # spread, where the derived shape was the worst planar configuration by
        # 38-41%. See `_shape_override`'s comment for the table.
        @test _shape_override(Val(:avx512), T) === swept
        @test _derived_shape(synthetic(:avx512, 64), T) === swept

        # The resolved default and the menu head agree, so a reader of either
        # sees the same shape.
        @test _derived_shape(synthetic(:avx512, 64), T) ===
            first(kernel_shapes(T, PlanarMethod()))
        @test _derived_shape(synthetic(:avx512, 64), T) in kernel_shapes(T, PlanarMethod())

        # Reordering the menu must not change the SET, or the compiled
        # specialization count moves with it.
        @test Set(kernel_shapes(T, PlanarMethod())) ==
            Set(((2 * W, NR_DEFAULT, W), swept, (2 * W ÷ 2, 8, W)))

        # The override is AVX-512-only; every other ISA takes the rule, and in
        # fact does not even reach it (`_rule_applies_complex` is false there),
        # so an unmeasured machine is never handed a ccqlin038 constant.
        for key in VALID_ISAS
            key === :avx512 && continue
            @test _shape_override(Val(key), T) === nothing
        end
    end
    # The REAL override stays empty on every ISA: the real rule was within
    # noise of its own sweep's best, so a row there would encode noise. That
    # asymmetry is deliberate, not an oversight.
    for T in (Float64, Float32), key in VALID_ISAS
        @test _shape_override(Val(key), T) === nothing
    end
    # The REAL rule is untouched: `_derived_shape` gained a method, it was not
    # edited, so a real element type still reaches exactly the old code.
    for T in (Float64, Float32)
        W = 64 ÷ sizeof(T)
        @test _derived_shape(synthetic(:avx512, 64), T) === (2 * W, NR_DEFAULT, W)
    end
end

@testset "the complex rule applies on :avx512 only, and falls back elsewhere" begin
    @test _rule_applies_complex(Val(:avx512))
    for T in (ComplexF64, ComplexF32)
        # AVX2's 16 ymm registers leave planar zero spare, and the reference's
        # AVX2 complex shapes are explicitly unmeasured: fall back, never guess.
        for key in (:avx2, :neon, :unknown, :somethingelse)
            @test !_rule_applies_complex(Val(key))
            for vb in (0, 16, 32, 64)   # width must not matter off :avx512
                @test _derived_shape(synthetic(key, vb), T) === _legacy_shape(T)
            end
        end
        # ... and on :avx512 with nothing detected, likewise.
        @test _derived_shape(synthetic(:avx512, 0), T) === _legacy_shape(T)
        # The complex legacy shape is the real one's (8, 6) tile in LOGICAL
        # complex rows, with the lane width of the real type.
        @test _legacy_shape(T) === (8, NR_DEFAULT, _legacy_shape(real(T))[3])
    end
    # The real rule still applies on AVX2 -- this is the `:neon` precedent, not
    # a narrowing of anything that already shipped.
    @test _rule_applies(Val(:avx2))
    @test _rule_applies(Val(:avx512))
end

@testset "register budget: the real assertion verbatim, the complex one generalized" begin
    # Unchanged and verbatim, as it appears in "every shipped shape is
    # constructible" above. Generalised, not widened.
    for T in (Float64, Float32), (MR, NR, W) in kernel_shapes(T)
        @test (MR ÷ W) * NR + (MR ÷ W) <= 32
    end

    # Method-aware complex form. A complex MRxNR tile is 2*MR*NR reals however
    # it is split, so one accumulator plane holds `2MR/planes` real rows: MR
    # for planar (separate real and imaginary planes) and 2MR for 1m (one plane
    # over a doubled real row count -- which is why a 1m menu entry need not
    # have MR % W == 0). Live state per K step is then
    #   planes*MV*NR accumulators + planes*MV A vectors + planes B broadcasts
    # which at planar (16,6,8) is 24 + 4 + 2 = 30 (docs/decisions.md, Cliff A).
    #
    # These are AVX-512 menus -- `_rule_applies_complex` ships them nowhere
    # else -- so they are checked against the AVX-512 register file, read from
    # the ISA table rather than written as a literal 32. On an AVX-512 host the
    # detected profile must agree with that table.
    nreg = _isa_nregisters(Val(:avx512))
    p = target_profile()
    p.isa === :avx512 && @test p.nregisters == nreg
    for T in (ComplexF64, ComplexF32), m in (PlanarMethod(), OneMMethod())
        planes = accumulator_planes(m)
        @test planes == (m === PlanarMethod() ? 2 : 1)
        for (MR, NR, W) in kernel_shapes(T, m)
            rows = (2 * MR) ÷ planes
            @test rows % W == 0
            mv = rows ÷ W
            @test planes * mv * NR + planes * mv + planes <= nreg
        end
    end
    # Menus are bounded so the compiled specialization set is.
    for T in (ComplexF64, ComplexF32), m in (PlanarMethod(), OneMMethod())
        @test length(kernel_shapes(T, m)) <= 3
    end
    # `RealMethod` forwards to the one-argument form: the real menus are
    # reached by exactly the code they always were.
    for T in (Float64, Float32)
        @test kernel_shapes(T, RealMethod()) === kernel_shapes(T)
    end
end

@testset "complex blocking derives from the measured real row, never a new table" begin
    for v in (Val(:avx512), Val(:avx2), Val(:unknown)), T in (ComplexF64, ComplexF32)
        base = default_blocking(v, real(T))     # the MEASURED real row
        for m in (PlanarMethod(), OneMMethod())
            b = default_blocking(v, T, m)
            @test b.kc === base.kc              # kc is per-K-step, not per-element
            @test b.mc === max(1, base.mc ÷ a_reals(m))
            @test b.nc === max(1, base.nc ÷ b_reals(m))
        end
        planar = default_blocking(v, T, PlanarMethod())
        onem = default_blocking(v, T, OneMMethod())
        # 1m's mc is exactly half planar's -- derived from a_reals, never
        # tabulated -- and both methods get the same packed byte budget.
        @test 2 * onem.mc === planar.mc
        @test onem.nc === planar.nc
        @test planar.mc * a_reals(PlanarMethod()) === onem.mc * a_reals(OneMMethod())
    end
    # The worked example from docs/decisions.md.
    @test default_blocking(Val(:avx512), Float64) === Blocking(128, 256, 768)
    @test default_blocking(Val(:avx512), ComplexF64, PlanarMethod()) === Blocking(64, 256, 384)
    @test default_blocking(Val(:avx512), ComplexF64, OneMMethod()) === Blocking(32, 256, 384)
    # Every real row is reached by exactly the method it always was.
    for key in VALID_ISAS, T in (Float64, Float32)
        @test default_blocking(Val(key), T, RealMethod()) === default_blocking(Val(key), T)
        @test default_blocking(Val(key), T) === default_blocking(Val(key), T)
    end
    # A direct `_legacy_blocking` call degrades through the same formula rather
    # than throwing a MethodError.
    @test _legacy_blocking(ComplexF64) === Blocking(32, 128, 384)
    @test _legacy_blocking(ComplexF64, OneMMethod()) === Blocking(16, 128, 384)
    @test _legacy_blocking(ComplexF32) === Blocking(48, 384, 576)
    # The real rows are untouched.
    @test _legacy_blocking(Float64) === Blocking(64, 128, 768)
    @test _legacy_blocking(Float32) === Blocking(96, 384, 1152)
end

@testset "the complex kernel constructor is an explicit, single seam" begin
    # Phase C wired `_complex_kernel_from_shape`'s PlanarMethod arm and Phase D
    # its OneMMethod arm; the method-generic arm still throws, so an
    # unimplemented method can never silently fall back to one that is
    # implemented -- a substitution that would make a planar-vs-1m measurement
    # meaningless.
    for T in (ComplexF64, ComplexF32)
        kernel = _default_kernel(T)
        @test kernel isa QuasiStrided.PlanarKernel
        @test scalartype(kernel) === T
        @test QuasiStrided.realtype(kernel) === real(T)
        @test QuasiStrided.complex_method(kernel) === PlanarMethod()
        @test QuasiStrided._default_complex_method(T) === PlanarMethod()
        # The default resolves to the head of that method's own menu.
        @test (mr(kernel), nr(kernel), lanewidth(kernel)) ===
            first(QuasiStrided.kernel_shapes(T, PlanarMethod()))
    end
    # 1m is now constructible (Phase D) -- but only by asking for it by name.
    # It is still not the default, and no rule may make it one.
    for T in (ComplexF64, ComplexF32)
        shape = first(QuasiStrided.kernel_shapes(T, OneMMethod()))
        kernel = QuasiStrided._complex_kernel_from_shape(shape, T, OneMMethod())
        @test kernel isa QuasiStrided.OneMKernel
        @test scalartype(kernel) === T
        @test QuasiStrided.complex_method(kernel) === OneMMethod()
        @test (mr(kernel), nr(kernel), lanewidth(kernel)) === shape
        # ... and the engine's own choice is still planar.
        @test _default_kernel(T) isa QuasiStrided.PlanarKernel
    end
    # A method with no constructor still throws, naming itself; it does not
    # degrade to a method that does have one. `RealMethod` stands in for any
    # such method here -- it is a `ComplexMethod` with no complex kernel, so it
    # reaches exactly the generic arm an unimplemented complex method would.
    for T in (ComplexF64, ComplexF32)
        shape = first(QuasiStrided.kernel_shapes(T, OneMMethod()))
        err = try
            QuasiStrided._complex_kernel_from_shape(shape, T, QuasiStrided.RealMethod())
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("RealMethod", err.msg)
        @test occursin(string(T), err.msg)
    end
    # The real path's default kernel is unaffected.
    for T in (Float64, Float32)
        @test _default_kernel(T) isa SIMDKernel
    end
end

@testset "the engine picks no complex kernel on an unmeasured ISA" begin
    # The complex shapes are budgeted for a 32-register AVX-512 file; every
    # candidate needs more vector registers than AVX2's 16 ymm provide. The
    # shape *rule* already refuses to derive off :avx512, but the legacy
    # fallback would still hand back a shape, so kernel construction is gated
    # too: an error beats a guaranteed-spilling default. Naming a kernel
    # explicitly is unaffected -- this governs only what the engine picks.
    @test QuasiStrided._complex_default_supported(Val(:avx512))
    for isakey in (:avx2, :neon, :unknown)
        @test !QuasiStrided._complex_default_supported(Val(isakey))
        err = try
            QuasiStrided._complex_unsupported_isa(ComplexF64, synthetic(isakey, 32))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(isakey), err.msg)
    end
    # The real path derives or falls back on every ISA, never throws.
    for isakey in (:avx512, :avx2, :neon, :unknown), T in (Float64, Float32)
        @test QuasiStrided._derived_shape(synthetic(isakey, 32), T) isa Tuple{Int, Int, Int}
    end
end
