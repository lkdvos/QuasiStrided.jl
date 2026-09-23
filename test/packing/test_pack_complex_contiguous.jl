# The vectorized complex packing fast path (src/packing/pack_contiguous.jl,
# `_pack_complex_contiguous!`), serving `PlanarFormat`, `OneEFormat`'s A panel,
# and 1m's B panel (which is planar's).
#
# Three separable things are pinned here, deliberately not mixed:
#
#  1. VALUES. The fast path must produce the bytes the scalar `_pack_panel!`
#     loop produces -- not "within a tolerance". There is
#     no arithmetic on this path beyond a sign flip, so bitwise `isequal` (which
#     also separates `+0.0` from `-0.0`) is the right comparison, matching
#     test_pack_complex.jl's own justification for using `==` there. Each
#     case is checked against BOTH the scalar path (run on the same fixture with
#     a plain `Vector` destination, which the gate excludes) and an
#     independently written reference layout, so the test is not vacuous when
#     the ISA gate is closed and the "fast" call is itself the scalar path.
#
#  2. GATE FIRING: assert the predicate is TRUE on a fixture that should
#     fast-path, and FALSE for each single violated condition on its own, so
#     a gate that never opens cannot pass unnoticed.
#
#  3. ISA PORTABILITY. The fast path ships for AVX-512 only.
#     The expectation is derived from `target_profile()` at run time, never
#     written as a literal, so `test/forced_isa_runner.jl` under
#     `avx2`/`neon`/`unknown` asserts the fast path is OFF and the outputs are
#     unchanged, rather than failing by construction.

using Test
using QuasiStrided
using QuasiStrided: ComplexKernelDescriptor, PlanarFormat, OneEFormat, RealFormat,
    AffineAxis, ScatterAxis, PtrScatterAxis, SourceTile, pack_a!, pack_b!,
    packed_a_length, packed_b_length, packed_panel, PackedPanel,
    target_profile, unknown_target, TargetProfile, CacheLevel,
    KERNEL_SHAPES_C64_PLANAR, KERNEL_SHAPES_C32_PLANAR,
    KERNEL_SHAPES_C64_ONEM, KERNEL_SHAPES_C32_ONEM

const QS = QuasiStrided

# Whether the host (or the forced profile) is one the fast path ships for.
# Read once here so every expectation below is derived, never literal.
const FASTPATH_ON = QS._complex_fastpath_isa_eligible()

# ---------------------------------------------------------------------------
# Independently written reference layouts (same shape as the ones in
# test_pack_complex.jl, repeated rather than imported so this file does not
# depend on that file's include order).
# ---------------------------------------------------------------------------

function fp_ref_planar(::Type{T}, vr::Int, kc::Int, valid::Int, g, f) where {T}
    out = zeros(real(T), 2 * vr * kc)
    for p in 0:(kc - 1)
        base = p * 2 * vr
        for t in 0:(valid - 1)
            z = f(g(t, p))
            out[base + t + 1] = real(z)
            out[base + vr + t + 1] = imag(z)
        end
    end
    return out
end

function fp_ref_onee(::Type{T}, vr::Int, kc::Int, valid::Int, g, f) where {T}
    out = zeros(real(T), 4 * vr * kc)
    for p in 0:(kc - 1)
        base = p * 4 * vr
        for t in 0:(valid - 1)
            z = f(g(t, p))
            re = real(z)
            im = imag(z)
            out[base + 2 * t + 1] = re
            out[base + 2 * t + 2] = im
            out[base + 2 * vr + 2 * t + 1] = -im
            out[base + 2 * vr + 2 * t + 2] = re
        end
    end
    return out
end

fp_ref(::PlanarFormat, ::Type{T}, vr, kc, valid, g, f) where {T} =
    fp_ref_planar(T, vr, kc, valid, g, f)
fp_ref(::OneEFormat, ::Type{T}, vr, kc, valid, g, f) where {T} =
    fp_ref_onee(T, vr, kc, valid, g, f)

# Values that are exactly representable in Float32 as well as Float64, plus a
# deliberate sprinkling of signed and unsigned zeros -- `conj` and 1e's `-im`
# are sign flips, and `+0.0` vs `-0.0` is precisely where a `*(-1.0)`-based
# implementation would diverge from the scalar `-x`.
function fp_value(::Type{T}, i::Int) where {T}
    R = real(T)
    if i % 17 == 0
        return T(R(0), R(0))
    elseif i % 19 == 0
        return T(R(3), R(0))
    elseif i % 23 == 0
        return T(R(0), R(-5))
    else
        return T(R(10 * i + 1), R(-(10 * i + 2)))
    end
end
fp_storage(::Type{T}, n) where {T} = [fp_value(T, i) for i in 1:n]

# Run `pack_*!` into a `PackedPanel` (the only destination the gate admits) and
# return the buffer contents.
function fp_pack_a_panel(kernel, source, kc, transform)
    buf = fill(realtype_of(kernel)(-777), packed_a_length(kernel, kc))
    GC.@preserve buf begin
        pack_a!(packed_panel(buf, 1, length(buf)), source, kernel, transform)
    end
    return buf
end

function fp_pack_b_panel(kernel, source, kc, transform)
    buf = fill(realtype_of(kernel)(-777), packed_b_length(kernel, kc))
    GC.@preserve buf begin
        pack_b!(packed_panel(buf, 1, length(buf)), source, kernel, transform)
    end
    return buf
end

realtype_of(kernel) = QS.realtype(kernel)

# Scalar reference run: a plain `Vector` destination is excluded by the gate
# (`packed isa PackedPanel{R}`), so this is the scalar `_pack_panel!` loop by
# construction.
function fp_pack_a_scalar(kernel, source, kc, transform)
    buf = fill(realtype_of(kernel)(-777), packed_a_length(kernel, kc))
    pack_a!(buf, source, kernel, transform)
    return buf
end

function fp_pack_b_scalar(kernel, source, kc, transform)
    buf = fill(realtype_of(kernel)(-777), packed_b_length(kernel, kc))
    pack_b!(buf, source, kernel, transform)
    return buf
end

# ---------------------------------------------------------------------------
# Shape coverage: every logical register-tile extent the shipped AVX-512
# complex menus can hand the packer, read off the menus rather than transcribed
# (no expectation may be hardcoded to one shape), plus three hand-added edge
# cases the menus do not contain.
# ---------------------------------------------------------------------------

const MENU_MRS = sort(
    unique(
        vcat(
            [s[1] for s in KERNEL_SHAPES_C64_PLANAR],
            [s[1] for s in KERNEL_SHAPES_C32_PLANAR],
            [s[1] for s in KERNEL_SHAPES_C64_ONEM],
            [s[1] for s in KERNEL_SHAPES_C32_ONEM],
        )
    )
)
const MENU_NRS = sort(
    unique(
        vcat(
            [s[2] for s in KERNEL_SHAPES_C64_PLANAR],
            [s[2] for s in KERNEL_SHAPES_C32_PLANAR],
            [s[2] for s in KERNEL_SHAPES_C64_ONEM],
            [s[2] for s in KERNEL_SHAPES_C32_ONEM],
        )
    )
)

# `1` is the single-element tile; `3`/`7` are deliberately not multiples of any
# shipped lane width (8 and 16 reals on AVX-512), so the `2PD`-wide load and the
# `PD`-wide halves of the shuffle are exercised at non-register-multiple sizes.
const EXTRA_DIMS = (1, 3, 7)

@testset "complex pack fast path: menus supply shapes to cover" begin
    # Sanity on the derivation above, so a menu edit that empties these lists
    # turns the loops below into silent no-ops noisily instead.
    @test 24 in MENU_MRS && 48 in MENU_MRS      # the two shipped AVX-512 defaults
    @test maximum(MENU_MRS) == 48
    @test 3 in MENU_NRS && maximum(MENU_NRS) == 8
end

# ---------------------------------------------------------------------------
# 1. Values: fast path == scalar path == independent reference, bitwise
# ---------------------------------------------------------------------------

@testset "complex pack fast path: A panel, $T / $fmtname / MR=$MR / $fname" for
    T in (ComplexF64, ComplexF32),
        (fmtname, fa) in (("planar", PlanarFormat()), ("1e", OneEFormat())),
        MR in sort(unique(vcat(MENU_MRS, collect(EXTRA_DIMS)))),
        (fname, f) in (("identity", identity), ("conj", conj))

    NR, kc = 3, 5
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, PlanarFormat())
    storage = fp_storage(T, 40000)
    base, lda = 17, 997
    source = SourceTile(storage, base, AffineAxis(0, 1, MR), AffineAxis(0, lda, kc))
    g = (t, p) -> storage[base + t + lda * p + 1]

    fast = fp_pack_a_panel(kernel, source, kc, f)
    scalar = fp_pack_a_scalar(kernel, source, kc, f)
    reference = fp_ref(fa, T, MR, kc, MR, g, f)

    @test all(isequal.(fast, scalar))
    @test all(isequal.(fast, reference))
    # The gate really did fire (or really was closed) on this fixture.
    @test QS._pack_complex_contiguous_eligible(
        packed_panel(scalar, 1, length(scalar)), storage, source.rows, f, fa,
        MR, Val(MR), T
    ) == FASTPATH_ON
end

@testset "complex pack fast path: B panel, $T / $fmtname / NR=$NR / $fname" for
    T in (ComplexF64, ComplexF32),
        (fmtname, fb) in (("planar", PlanarFormat()), ("1e", OneEFormat())),
        NR in sort(unique(vcat(MENU_NRS, collect(EXTRA_DIMS)))),
        (fname, f) in (("identity", identity), ("conj", conj))

    MR, kc = 4, 5
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, PlanarFormat(), fb)
    storage = fp_storage(T, 40000)
    base, ldb = 17, 997
    # B's tile is (K rows) x (N cols) and its PACKED index runs along N, so the
    # unit-stride axis here is `cols`, not `rows` -- the mirror image of A.
    source = SourceTile(storage, base, AffineAxis(0, ldb, kc), AffineAxis(0, 1, NR))
    g = (j, p) -> storage[base + ldb * p + j + 1]

    fast = fp_pack_b_panel(kernel, source, kc, f)
    scalar = fp_pack_b_scalar(kernel, source, kc, f)
    reference = fp_ref(fb, T, NR, kc, NR, g, f)

    @test all(isequal.(fast, scalar))
    @test all(isequal.(fast, reference))
    @test QS._pack_complex_contiguous_eligible(
        packed_panel(scalar, 1, length(scalar)), storage, source.cols, f, fb,
        NR, Val(NR), T
    ) == FASTPATH_ON
end

# `kc == 0`, and a scattered K axis over a unit-stride lane axis -- the fast
# path's lane axis must be unit-stride but its STEP axis is free to scatter,
# which is what makes it usable on this engine's irregular contractions.
@testset "complex pack fast path: scattered K steps, kc==0 ($T, $fmtname)" for
    T in (ComplexF64, ComplexF32),
        (fmtname, fa) in (("planar", PlanarFormat()), ("1e", OneEFormat()))

    MR, NR, kc = 8, 6, 6
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, PlanarFormat())
    storage = fp_storage(T, 40000)
    base = 123
    co = [((p * 5) % 7) * 1013 + 3 * p for p in 0:(kc - 1)]
    source = SourceTile(storage, base, AffineAxis(0, 1, MR), ScatterAxis(co, kc))
    g = (t, p) -> storage[base + t + co[p + 1] + 1]

    for f in (identity, conj)
        fast = fp_pack_a_panel(kernel, source, kc, f)
        @test all(isequal.(fast, fp_pack_a_scalar(kernel, source, kc, f)))
        @test all(isequal.(fast, fp_ref(fa, T, MR, kc, MR, g, f)))
    end

    empty_src = SourceTile(storage, base, AffineAxis(0, 1, MR), ScatterAxis(co, 0))
    buf = fill(real(T)(-777), 8)
    GC.@preserve buf begin
        pk = packed_panel(buf, 1, length(buf))
        @test pack_a!(pk, empty_src, kernel, identity) === pk
    end
    @test all(isequal.(buf, fill(real(T)(-777), 8)))   # kc == 0 writes nothing
end

# The transform contract, re-pinned THROUGH the fast path: `conj`
# applies to the complex element and the split happens afterwards. A packer
# that conjugated per real half (or that used a `*(-1.0)` sign vector and got
# `-0.0` wrong) fails here.
@testset "complex pack fast path: conj == identity on a pre-conjugated source ($T, $fmtname)" for
    T in (ComplexF64, ComplexF32),
        (fmtname, fa) in (("planar", PlanarFormat()), ("1e", OneEFormat()))

    MR, NR, kc = 16, 8, 4
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, PlanarFormat())
    storage = fp_storage(T, 40000)
    cstorage = conj.(storage)
    base, lda = 17, 997
    src = SourceTile(storage, base, AffineAxis(0, 1, MR), AffineAxis(0, lda, kc))
    csrc = SourceTile(cstorage, base, AffineAxis(0, 1, MR), AffineAxis(0, lda, kc))

    pconj = fp_pack_a_panel(kernel, src, kc, conj)
    pident = fp_pack_a_panel(kernel, csrc, kc, identity)
    @test all(isequal.(pconj, pident))          # signed zeros included
    # ... and the pin has teeth: the two transforms do not agree on one source.
    @test fp_pack_a_panel(kernel, src, kc, conj) != fp_pack_a_panel(kernel, src, kc, identity)
end

# ---------------------------------------------------------------------------
# 2. Gate firing: one violated condition at a time
# ---------------------------------------------------------------------------

@testset "complex pack fast path: eligibility gate, one violation at a time" begin
    T = ComplexF64
    R = Float64
    MR, NR, kc = 8, 6, 4
    fa = PlanarFormat()
    storage = fp_storage(T, 4000)
    rows = AffineAxis(0, 1, MR)
    buf = fill(R(0), packed_a_length(ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fa), kc))
    ptr_scatter_offs = collect(0:(MR - 1))

    GC.@preserve buf ptr_scatter_offs begin
        pk = packed_panel(buf, 1, length(buf))
        elig(;
            packed = pk, store = storage, ax = rows, tr = identity, fmt = fa,
            valid = MR, pd = Val(MR), et = T
        ) =
            QS._pack_complex_contiguous_eligible(packed, store, ax, tr, fmt, valid, pd, et)

        # Baseline: everything satisfied -> tracks the ISA gate and nothing else.
        @test elig() == FASTPATH_ON

        # Each of these must be `false` regardless of ISA.
        @test elig(packed = buf) == false                        # Vector, not PackedPanel
        @test elig(packed = packed_panel(Float32[0.0f0], 1, 1)) == false   # wrong real type
        @test elig(store = view(storage, 1:100)) == false        # not a DenseVector
        @test elig(store = collect(reshape(storage, 40, 100))) == false    # rank-2
        @test elig(ax = AffineAxis(0, 2, MR)) == false           # strided lanes
        @test elig(ax = AffineAxis(0, -1, MR)) == false          # negative unit stride
        @test elig(ax = ScatterAxis(collect(0:(MR - 1)), MR)) == false     # scattered lanes
        @test elig(ax = PtrScatterAxis(pointer(ptr_scatter_offs), MR)) == false   # scattered lanes, `Ptr`-backed (`_axis_of`'s irregular-axis type)
        @test elig(valid = MR - 1) == false                      # tail sliver (padding)
        @test elig(valid = 0) == false
        @test elig(tr = (z -> 2z)) == false                      # transform not id/conj
        @test elig(tr = real) == false
        @test elig(fmt = RealFormat()) == false                  # format without a packer

        # ... and `conj` and `OneEFormat` are both admitted, not accidents of
        # the baseline's choices.
        @test elig(tr = conj) == FASTPATH_ON
        @test elig(fmt = OneEFormat()) == FASTPATH_ON
    end
end

# The ineligible shapes must still produce the right bytes -- the fallback is
# the thing the gate protects, and it must remain reachable.
@testset "complex pack fast path: ineligible shapes take the scalar path and stay correct ($T)" for
    T in (ComplexF64, ComplexF32)

    MR, NR, kc = 8, 6, 4
    for fa in (PlanarFormat(), OneEFormat())
        kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, PlanarFormat())
        storage = fp_storage(T, 40000)
        base, lda = 3000, 997
        cases = (
            (
                "negative stride",
                SourceTile(storage, base, AffineAxis(0, -1, MR), AffineAxis(0, -lda, kc)),
                (t, p) -> storage[base - t - lda * p + 1],
                MR,
            ),
            (
                "strided lanes",
                SourceTile(storage, 17, AffineAxis(0, 3, MR), AffineAxis(0, lda, kc)),
                (t, p) -> storage[17 + 3 * t + lda * p + 1],
                MR,
            ),
            (
                "tail sliver",
                SourceTile(storage, 17, AffineAxis(0, 1, MR - 3), AffineAxis(0, lda, kc)),
                (t, p) -> storage[17 + t + lda * p + 1],
                MR - 3,
            ),
        )
        for (name, source, g, valid) in cases, f in (identity, conj)
            @test QS._pack_complex_contiguous_eligible(
                packed_panel(zeros(real(T), 1), 1, 1), storage, source.rows, f, fa,
                valid, Val(MR), T
            ) == false
            got = fp_pack_a_panel(kernel, source, kc, f)
            @test all(isequal.(got, fp_ref(fa, T, MR, kc, valid, g, f)))
        end
    end
end

# ---------------------------------------------------------------------------
# 3. ISA portability
# ---------------------------------------------------------------------------

@testset "complex pack fast path: ISA gate is a register-width question" begin
    # Derived from the same table the target detector uses, so this stays true
    # if a width ever changes; `:avx512` is the only key that opens the gate.
    @test QS._complex_fastpath_isa_eligible(
        TargetProfile(:avx512, Sys.ARCH, "t", 64, 32, CacheLevel(), CacheLevel(), CacheLevel())
    )
    for (key, vb, nreg) in ((:avx2, 32, 16), (:neon, 16, 32), (:unknown, 0, 0))
        @test !QS._complex_fastpath_isa_eligible(
            TargetProfile(key, Sys.ARCH, "t", vb, nreg, CacheLevel(), CacheLevel(), CacheLevel())
        )
    end
    @test !QS._complex_fastpath_isa_eligible(unknown_target())
    # The live gate agrees with the live profile: this is what makes every
    # `== FASTPATH_ON` assertion above meaningful under forced_isa_runner.jl.
    @test FASTPATH_ON == (target_profile().vector_bytes == QS._isa_vector_bytes(Val(:avx512)))
end

# ---------------------------------------------------------------------------
# 4. Zero allocation (Cliff B discipline: the fast path must not box anything)
# ---------------------------------------------------------------------------

@testset "complex pack fast path: allocation-free ($T, $fmtname)" for
    T in (ComplexF64, ComplexF32),
        (fmtname, fa) in (("planar", PlanarFormat()), ("1e", OneEFormat()))

    MR, NR, kc = 24, 3, 8
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, PlanarFormat())
    storage = fp_storage(T, 40000)
    source = SourceTile(storage, 17, AffineAxis(0, 1, MR), AffineAxis(0, 997, kc))
    buf = fill(real(T)(0), packed_a_length(kernel, kc))
    GC.@preserve buf begin
        pk = packed_panel(buf, 1, length(buf))
        for f in (identity, conj)
            pack_a!(pk, source, kernel, f)          # warm up
            @test @allocated(pack_a!(pk, source, kernel, f)) == 0
        end
    end
end
