# Register shape, kernel and blocking selection from a `TargetProfile`.

using StridedViews: StridedView
using QuasiStrided: TargetProfile, CacheLevel, target_profile, cache_topology,
    unknown_target, _detect_isa, _detect_target, _derived_shape, _fallback_shape,
    _shape_override, _kernel_for, _default_kernel, _fallback_blocking, kernel_shapes,
    _parse_size, _count_cpu_list, NR_DEFAULT, _rule_applies,
    _isa_nregisters, packed_a_per_k, packed_b_per_k, realtype, complex_method,
    RealMethod, PlanarMethod, OneMMethod, accumulator_planes, a_reals, b_reals

@testset "kernel shape selection" begin

    @testset "unknown target is bit-identical to the pre-detection defaults" begin
        u = unknown_target()
        @test u.isa === :unknown
        for T in (Float64, Float32)
            @test _derived_shape(u, T) === _fallback_shape(T)
            k = _kernel_for(u, T)
            @test k isa SIMDKernel
            @test (mr(k), nr(k), lanewidth(k)) === _fallback_shape(T)
        end
        # The exact constants from docs/decisions.md Phase E.
        @test _fallback_blocking(Float64) === Blocking(64, 128, 768)
        @test _fallback_blocking(Float32) === Blocking(96, 384, 1152)
        for key in (:unknown, :avx2, :neon, :somethingelse), T in (Float64, Float32)
            @test default_blocking(Val(key), T) === _fallback_blocking(T)
        end
        # The documented type-argument form is unchanged by detection.
        @test default_blocking(Float64) === _fallback_blocking(Float64)
        @test default_blocking(Float32) === _fallback_blocking(Float32)
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
        @test _derived_shape(synthetic(:avx2, 32), Float64) === _fallback_shape(Float64)
    end

    @testset "the rule applies only where it was measured" begin
        # :neon is detected but unmeasured, so it gets the legacy shape rather
        # than an invented one -- the rule would pick MR = 2W = 4 on 128-bit
        # lanes, narrower and smaller than legacy with nothing to justify it.
        # This is also what keeps the shipped default identical on aarch64,
        # which execution/test_workspace.jl's "SIMDKernel is the default" testset pins.
        for T in (Float64, Float32)
            @test _derived_shape(synthetic(:neon, 16), T) === _fallback_shape(T)
            @test _derived_shape(synthetic(:unknown, 0), T) === _fallback_shape(T)
            for vb in (16, 32, 64)   # width must not matter for an unmeasured ISA
                @test _derived_shape(synthetic(:neon, vb), T) === _fallback_shape(T)
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
            @test (mr(small), nr(small), lanewidth(small)) === _fallback_shape(T)
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
        @test QuasiStrided._rule_shape(64, T) === (2 * W, NR_DEFAULT, W)

        # `_shape_override` on AVX-512 carries the Phase F sweep's winner, and
        # is what `_derived_shape` therefore returns. This is the only row in
        # the package swept on the host that runs it: measured on ccqlin038 at
        # 21 reps and a 0.4% canary spread, where the derived shape was the
        # worst planar configuration by 38-41%. See `_shape_override`'s
        # comment for the table, and for the NEON and AVX2 rows' provenance.
        @test _shape_override(Val(:avx512), T) === swept
        @test _derived_shape(synthetic(:avx512, 64), T) === swept

        # The resolved default and the menu head agree, so a reader of either
        # sees the same shape.
        @test _derived_shape(synthetic(:avx512, 64), T) ===
            first(kernel_shapes(T, PlanarMethod()))
        @test _derived_shape(synthetic(:avx512, 64), T) in kernel_shapes(T, PlanarMethod())

        # The menu is the three AVX-512 shapes, the AVX2 row, the NEON row,
        # and a narrow floor entry that guarantees `_fitted_shape`
        # always finds something on an ISA with no row of its own. Pinned as a
        # SET so a reorder cannot change the compiled specialization count
        # silently, and so adding a shape is a deliberate edit here rather than
        # a side effect.
        @test Set(kernel_shapes(T, PlanarMethod())) == Set(
            (
                (2 * W, NR_DEFAULT, W), swept, (2 * W ÷ 2, 8, W),
                _shape_override(Val(:avx2), T), _shape_override(Val(:neon), T),
                (W ÷ 4, NR_DEFAULT, W ÷ 4),
            )
        )

        # There is now an override row per *measured or modelled* ISA, each
        # with its own provenance (see `_shape_override`): AVX-512 swept on
        # ccqlin038, NEON measured on an M3 Max by the sibling project, AVX2
        # adopted from that project's model. What must hold for all of them is
        # that they fit their ISA's register file with room to spare, and are
        # in the menu -- no ISA may be handed another ISA's constant, and none
        # may be handed a zero-spare shape.
        for (key, nreg) in ((:avx512, 32), (:avx2, 16), (:neon, 32))
            ovr = _shape_override(Val(key), T)
            @test ovr !== nothing
            @test QuasiStrided._planar_pressure(ovr...) <= nreg
            @test QuasiStrided._planar_pressure(ovr...) < nreg  # scratch left over
            @test ovr in kernel_shapes(T, PlanarMethod())
        end
        # `:unknown` gets no row: there is nothing to base one on, so it takes
        # the register-budget fit.
        @test _shape_override(Val(:unknown), T) === nothing
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

@testset "the complex rule applies on :avx512 only, and fits elsewhere" begin
    @test _rule_applies(Val(:avx512), QuasiStrided.PlanarMethod())
    for T in (ComplexF64, ComplexF32)
        # The measured *rule* (and its swept override) is :avx512-only. Off it,
        # the shape is fitted to the register file rather than derived -- and,
        # since CI, rather than refused. What must hold is that the result fits
        # the budget and is in the menu; the exact shape is an implementation
        # detail of `_fitted_shape` and is deliberately not pinned here.
        # `:unknown` and an unrecognised key have no override row, so they
        # exercise the fit itself: whatever it returns must fit the budget it
        # was given and be in the menu.
        for key in (:unknown, :somethingelse)
            @test !_rule_applies(Val(key), QuasiStrided.PlanarMethod())
            @test _shape_override(Val(key), T) === nothing
            for vb in (0, 16, 32, 64), nreg in (0, 16, 32)
                shape = _derived_shape(synthetic(key, vb; nregisters = nreg), T)
                MR, NR, W = shape
                @test QuasiStrided._planar_pressure(MR, NR, W) <=
                    (nreg > 0 ? nreg : 16)
                @test shape in kernel_shapes(T, PlanarMethod())
            end
        end
        # `:avx2` and `:neon` do have rows, so the override short-circuits the
        # rule and the fit alike -- and, unlike the fit, is width-independent.
        for key in (:avx2, :neon)
            @test !_rule_applies(Val(key), QuasiStrided.PlanarMethod())
            for vb in (0, 16, 32, 64), nreg in (0, 16, 32)
                @test _derived_shape(synthetic(key, vb; nregisters = nreg), T) ===
                    _shape_override(Val(key), T)
            end
        end
        # On :avx512 with no width detected, the rule cannot apply either, so
        # this also takes the fitted path.
        @test _derived_shape(synthetic(:avx512, 0), T) in
            kernel_shapes(T, PlanarMethod())
        # The complex legacy shape is still defined -- the real one's (8, 6)
        # tile in LOGICAL complex rows -- but is no longer what the resolver
        # falls back to. For `ComplexF64` that is because it does not fit:
        # (8, 6, 4) is `MV = 2`, pressure 30, against AVX2's 16. For
        # `ComplexF32` (8, 6, 8) is `MV = 1` and lands exactly on 16, so it
        # would have been admissible -- the resolver does not special-case
        # either, it just asks the budget.
        @test _fallback_shape(T) === (8, NR_DEFAULT, _fallback_shape(real(T))[3])
    end
    @test QuasiStrided._planar_pressure(_fallback_shape(ComplexF64)...) == 30
    @test QuasiStrided._planar_pressure(_fallback_shape(ComplexF32)...) == 16
    for T in (ComplexF64, ComplexF32)
    end
    # The real rule still applies on AVX2 -- this is the `:neon` precedent, not
    # a narrowing of anything that already shipped.
    @test _rule_applies(Val(:avx2), QuasiStrided.RealMethod())
    @test _rule_applies(Val(:avx512), QuasiStrided.RealMethod())
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
    # Menus stay bounded so the compiled specialization set does. Planar's is
    # six: three measured AVX-512 shapes plus one `MV = 1` entry per lane width
    # for `_fitted_shape` to land on off `:avx512`. 1m's is three --
    # it is never selected automatically, so it needs no fitted entries.
    for T in (ComplexF64, ComplexF32)
        @test length(kernel_shapes(T, PlanarMethod())) <= 6
        @test length(kernel_shapes(T, OneMMethod())) <= 3
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
    # A direct `_fallback_blocking` call degrades through the same formula rather
    # than throwing a MethodError.
    @test _fallback_blocking(ComplexF64) === Blocking(32, 128, 384)
    @test _fallback_blocking(ComplexF64, OneMMethod()) === Blocking(16, 128, 384)
    @test _fallback_blocking(ComplexF32) === Blocking(48, 384, 576)
    # The real rows are untouched.
    @test _fallback_blocking(Float64) === Blocking(64, 128, 768)
    @test _fallback_blocking(Float32) === Blocking(96, 384, 1152)
end

@testset "the complex kernel constructor is an explicit, single seam" begin
    # Phase C wired `_kernel_from_shape`'s PlanarMethod arm and Phase D
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
        @test QuasiStrided._default_method(T) === PlanarMethod()

        # What holds on EVERY host: the resolved shape is in the menu (so the
        # `@generated` dispatch cannot have fallen through to the menu tail)
        # and fits the detected register file.
        shape = (mr(kernel), nr(kernel), lanewidth(kernel))
        @test shape in QuasiStrided.kernel_shapes(T, PlanarMethod())
        profile = QuasiStrided.target_profile()
        budget = profile.nregisters > 0 ? profile.nregisters : 16
        @test QuasiStrided._planar_pressure(shape...) <= budget

        # Menu HEADSHIP holds only on `:avx512`, where the swept override
        # applies. Asserting it unconditionally is what made an earlier
        # revision of this testset fail on CI's AVX2 and NEON runners -- the
        # engine was right and the test was machine-dependent. Third instance
        # of that pattern in this project; see Amendment 5.
        if profile.isa === :avx512
            @test shape === first(QuasiStrided.kernel_shapes(T, PlanarMethod()))
        end
    end
    # 1m is now constructible (Phase D) -- but only by asking for it by name.
    # It is still not the default, and no rule may make it one.
    for T in (ComplexF64, ComplexF32)
        shape = first(QuasiStrided.kernel_shapes(T, OneMMethod()))
        kernel = QuasiStrided._kernel_from_shape(shape, T, OneMMethod())
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
            QuasiStrided._kernel_from_shape(shape, T, QuasiStrided.RealMethod())
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

@testset "the engine fits a complex shape to the register file on every ISA" begin
    # Phase C gated complex kernel *construction* to :avx512 and threw
    # elsewhere, on the argument that an error beats a guaranteed spill. CI
    # refuted that: every runner is AVX2 (Linux) or NEON (macOS), so complex
    # support was unavailable through `@tensor` on every machine the project
    # tests on. A slow-but-correct kernel beats no complex support.
    #
    # So off :avx512 the shape is fitted to the detected register file, and the
    # two things that must hold are (1) it fits, and (2) it is in the menu --
    # `_kernel_from_shape` is `@generated` over the menu and falls
    # through to the LAST entry on no match, so a fitted shape absent from the
    # menu would silently build a different kernel than was asked for.
    for (isakey, vb, nreg) in (
            (:avx512, 64, 32), (:avx2, 32, 16), (:neon, 16, 32), (:unknown, 0, 0),
        )
        profile = synthetic(isakey, vb; nregisters = nreg)
        for T in (ComplexF64, ComplexF32)
            shape = _derived_shape(profile, T)
            MR, NR, W = shape
            budget = nreg > 0 ? nreg : 16
            @test QuasiStrided._planar_pressure(MR, NR, W) <= budget
            @test shape in kernel_shapes(T, PlanarMethod())
            # Constructible, and at the shape asked for -- not the menu tail.
            k = QuasiStrided._kernel_from_shape(shape, T, PlanarMethod())
            @test (mr(k), nr(k), lanewidth(k)) === shape
        end
    end
    # And the engine now picks a complex kernel on the *actual* host, whatever
    # it is, rather than throwing. This is the assertion CI was failing.
    for T in (ComplexF64, ComplexF32)
        @test _default_kernel(T) isa QuasiStrided.PlanarKernel
        @test _default_kernel(T, 1024, 1024) isa QuasiStrided.PlanarKernel
        @test _default_kernel(T, 1, 1) isa QuasiStrided.PlanarKernel   # demotion path
    end
    # The real path derives or falls back on every ISA, never throws.
    for isakey in (:avx512, :avx2, :neon, :unknown), T in (Float64, Float32)
        @test QuasiStrided._derived_shape(synthetic(isakey, 32), T) isa Tuple{Int, Int, Int}
    end
end

@testset "_demote_for_run: largest menu shape whose mr divides the run" begin
    demote = QuasiStrided._demote_for_run
    for T in (Float64, Float32), (MR, NR, W) in kernel_shapes(T)
        k = SIMDKernel(Val(MR), Val(NR), T, Val(W))
        # Every sliver is already unit-stride: the run is a multiple of `mr`,
        # or it covers all of M.
        @test demote(T, k, 3 * MR, 1000, 1) === k
        @test demote(T, k, 7, 7, 1) === k
        # Deeper than the K cutoff: never demoted.
        kmax = T === Float64 ? QuasiStrided.F2_DEMOTE_KMAX_F64 : QuasiStrided.F2_DEMOTE_KMAX_F32
        @test demote(T, k, 1, 1000, kmax + 1) === k
        for run in (1, 4, 8, 12, 16, 20, 24, 48)
            run % MR == 0 && continue
            fits = [s for s in kernel_shapes(T) if run % s[1] == 0]
            d = demote(T, k, run, 1000, 1)
            if isempty(fits)
                @test d === k
            else
                @test (mr(d), nr(d), lanewidth(d)) === fits[argmax(first.(fits))]
            end
        end
    end
    # Complex kernels are never demoted (a deliberately deferred extension).
    for T in (ComplexF64, ComplexF32)
        k = _default_kernel(T)
        @test demote(T, k, 1, 1000, 1) === k
    end
end
