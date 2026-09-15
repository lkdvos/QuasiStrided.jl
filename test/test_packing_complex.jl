# Complex packing: pack_a!/pack_b! against a ComplexKernelDescriptor, verified
# against test-local, independently written reference layouts (never against
# the implementation's own offset helpers for the value expectations).
#
# The highest-risk item this file pins is the `transform` contract frozen in
# docs/decisions.md ("Complex element-type milestone"): `transform` applies to
# the loaded *complex* element and the result is split afterwards, never per
# real half -- where `conj` would be a silent no-op. The pin is bitwise:
# `pack_*!(..., conj)` equals `pack_*!(..., identity)` on a pre-conjugated
# source, in every format. Bitwise `==` is correct here because these are
# exact (a sign flip and a copy); it remains wrong for SIMD-vs-scalar
# comparisons elsewhere in this suite.

using Test
using QuasiStrided
using QuasiStrided: ComplexKernelDescriptor, PlanarFormat, OneEFormat,
    realtype, reals_per_element, packed_a_per_k, packed_b_per_k,
    packed_a_length, packed_b_length,
    AffineAxis, ScatterAxis, SourceTile, pack_a!, pack_b!

# =====================================================================
# Layout pin: the Julia analogue of the reference's `offset_of!` test.
# Every planar and 1e packer depends on re-then-im, adjacent, unit stride.
# =====================================================================

@testset "complex packing: memory layout pin (re then im, unit stride)" begin
    @test reinterpret(Float64, [ComplexF64(1, 2)]) == [1.0, 2.0]
    @test reinterpret(Float32, [ComplexF32(1, 2)]) == [1.0f0, 2.0f0]
    @test sizeof(ComplexF64) == 2 * sizeof(Float64)
    # ... and the accessors this file (and src/packing.jl) actually use agree.
    z = ComplexF64(1, 2)
    @test real(z) === 1.0 && imag(z) === 2.0
end

# =====================================================================
# Test-local reference layouts, written independently of src/packing.jl.
# `g(t, p)` returns the logical source element at lane `t`, K step `p`.
# `vr` is the register-tile extent (MR for A, NR for B). Lanes `>= valid`
# are padding and are left at literal zero.
# =====================================================================

function ref_planar(::Type{T}, vr::Int, kc::Int, valid::Int, g, f) where {T}
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

function ref_onee(::Type{T}, vr::Int, kc::Int, valid::Int, g, f) where {T}
    out = zeros(real(T), 4 * vr * kc)
    for p in 0:(kc - 1)
        base = p * 4 * vr
        for t in 0:(valid - 1)
            z = f(g(t, p))
            re = real(z)
            im = imag(z)
            out[base + 2 * t + 1] = re             # [[re, -im],
            out[base + 2 * t + 2] = im             #  [im,  re]]
            out[base + 2 * vr + 2 * t + 1] = -im
            out[base + 2 * vr + 2 * t + 2] = re
        end
    end
    return out
end

ref_pack(::PlanarFormat, ::Type{T}, vr, kc, valid, g, f) where {T} =
    ref_planar(T, vr, kc, valid, g, f)
ref_pack(::OneEFormat, ::Type{T}, vr, kc, valid, g, f) where {T} =
    ref_onee(T, vr, kc, valid, g, f)

# Shrink an axis to `count` valid lanes, keeping its addressing identical.
resized(ax::AffineAxis, count::Int) = AffineAxis(ax.base, ax.stride, count)
resized(ax::ScatterAxis, count::Int) = ScatterAxis(ax.offsets, count)

# Irregular offsets: the scattered case is this engine's reason to exist, and
# regular-only fixtures have hidden a real bug in this project before.
scatter_lane_offsets(n) = [(t * 7) % 11 + 13 * (t % 3) for t in 0:(n - 1)]
scatter_step_offsets(k, span) = [((p * 5) % 7) * span + 3 * p for p in 0:(k - 1)]

# Each fixture returns the two axes, the base, and a hand-written address
# function `g` that indexes `storage` directly -- never via `tile_load`.
function fixture_a(storage, fixname::String, vr::Int, kc::Int)
    if fixname == "affine"
        base, lda = 17, 97
        return (
            rows = AffineAxis(0, 1, vr), cols = AffineAxis(0, lda, kc), base = base,
            g = (t, p) -> storage[base + t + lda * p + 1],
        )
    elseif fixname == "negative-stride"
        base, lda = 3000, 97
        return (
            rows = AffineAxis(0, -1, vr), cols = AffineAxis(0, -lda, kc), base = base,
            g = (t, p) -> storage[base - t - lda * p + 1],
        )
    else
        base = 500
        ro = scatter_lane_offsets(vr)
        co = scatter_step_offsets(kc, 101)
        return (
            rows = ScatterAxis(ro, vr), cols = ScatterAxis(co, kc), base = base,
            g = (t, p) -> storage[base + ro[t + 1] + co[p + 1] + 1],
        )
    end
end

# B's tile is (K rows) x (N cols): the lane axis is the *column* axis.
function fixture_b(storage, fixname::String, vr::Int, kc::Int)
    if fixname == "affine"
        base, ldb = 17, 97
        return (
            rows = AffineAxis(0, 1, kc), cols = AffineAxis(0, ldb, vr), base = base,
            g = (j, p) -> storage[base + p + ldb * j + 1],
        )
    elseif fixname == "negative-stride"
        base, ldb = 3000, 97
        return (
            rows = AffineAxis(0, -1, kc), cols = AffineAxis(0, -ldb, vr), base = base,
            g = (j, p) -> storage[base - p - ldb * j + 1],
        )
    else
        base = 500
        ro = scatter_step_offsets(kc, 101)
        co = scatter_lane_offsets(vr)
        return (
            rows = ScatterAxis(ro, kc), cols = ScatterAxis(co, vr), base = base,
            g = (j, p) -> storage[base + ro[p + 1] + co[j + 1] + 1],
        )
    end
end

# Exactly representable in Float32 as well as Float64, so `==` is meaningful.
sample_storage(::Type{T}, n) where {T} = [T(10 * i + 1, 10 * i + 2) for i in 1:n]

const FIXTURES = ("affine", "negative-stride", "scatter")

# =====================================================================
# Per-format correctness against the test-local reference
# =====================================================================

@testset "pack_a!: $aname A-format, $T, $fixname" for T in (ComplexF64, ComplexF32),
        (aname, fa, fb) in (
            ("planar", PlanarFormat(), PlanarFormat()),
            ("1e", OneEFormat(), PlanarFormat()),
        ),
        fixname in FIXTURES

    MR, NR, kc = 4, 3, 5
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fb)
    R = real(T)
    @test realtype(kernel) === R

    storage = sample_storage(T, 4000)
    fix = fixture_a(storage, fixname, MR, kc)

    for m in (MR, 2, 0), f in (identity, conj, z -> 2 * z + one(T))
        source = SourceTile(storage, fix.base, resized(fix.rows, m), fix.cols)
        packed = fill(R(-777), packed_a_length(kernel, kc))   # nonzero prefill
        @test pack_a!(packed, source, kernel, f) === packed
        @test eltype(packed) === R
        @test packed == ref_pack(fa, T, MR, kc, m, fix.g, f)
    end
end

@testset "pack_b!: $bname B-format, $T, $fixname" for T in (ComplexF64, ComplexF32),
        (bname, fa, fb) in (
            ("planar", PlanarFormat(), PlanarFormat()),
            # B under 1m is bit-identical to planar's (their D14); 1e on B is
            # not a shipped combination but the emit method is format-generic,
            # so cover it rather than leave it unexercised.
            ("1e", OneEFormat(), OneEFormat()),
        ),
        fixname in FIXTURES

    MR, NR, kc = 4, 3, 5
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fb)
    R = real(T)

    storage = sample_storage(T, 4000)
    fix = fixture_b(storage, fixname, NR, kc)

    for n in (NR, 1, 0), f in (identity, conj, z -> 2 * z + one(T))
        source = SourceTile(storage, fix.base, fix.rows, resized(fix.cols, n))
        packed = fill(R(-777), packed_b_length(kernel, kc))
        @test pack_b!(packed, source, kernel, f) === packed
        @test packed == ref_pack(fb, T, NR, kc, n, fix.g, f)
    end
end

# The 1m claim from the reference's D14, at the level of bytes rather than
# lengths: B packed under the 1m descriptor is bit-identical to B packed under
# the planar descriptor, not merely the same size.
@testset "pack_b!: 1m's B panel is bit-identical to planar's ($T)" for
    T in (ComplexF64, ComplexF32)

    MR, NR, kc = 4, 3, 5
    planar = ComplexKernelDescriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    onem = ComplexKernelDescriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())
    storage = sample_storage(T, 4000)
    fix = fixture_b(storage, "scatter", NR, kc)

    for n in (NR, 1), f in (identity, conj)
        source = SourceTile(storage, fix.base, fix.rows, resized(fix.cols, n))
        p1 = fill(real(T)(-1), packed_b_length(planar, kc))
        p2 = fill(real(T)(-2), packed_b_length(onem, kc))
        pack_b!(p1, source, planar, f)
        pack_b!(p2, source, onem, f)
        @test p1 == p2
    end
end

# =====================================================================
# The conjugation pin: transform applies to the complex element, and the
# result is split afterwards -- NOT per real half, where `conj` is a no-op.
# =====================================================================

@testset "pack_*!(.., conj) == pack_*!(.., identity) on a pre-conjugated source ($T, $name)" for
    T in (ComplexF64, ComplexF32),
        (name, fa, fb) in (
            ("planar/planar", PlanarFormat(), PlanarFormat()),
            ("1e/planar", OneEFormat(), PlanarFormat()),
            ("1e/1e", OneEFormat(), OneEFormat()),
        )

    MR, NR, kc = 4, 3, 5
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fb)
    R = real(T)

    storage = sample_storage(T, 4000)
    cstorage = conj.(storage)

    # Scattered axes, so the pin covers the path the engine exists for.
    ro = scatter_lane_offsets(MR)
    co = scatter_step_offsets(kc, 101)

    for m in (MR, 2)
        src = SourceTile(storage, 500, ScatterAxis(ro, m), ScatterAxis(co, kc))
        csrc = SourceTile(cstorage, 500, ScatterAxis(ro, m), ScatterAxis(co, kc))
        pconj = fill(R(-777), packed_a_length(kernel, kc))
        pident = fill(R(-999), packed_a_length(kernel, kc))
        pack_a!(pconj, src, kernel, conj)
        pack_a!(pident, csrc, kernel, identity)
        @test pconj == pident                 # bitwise: both sides are exact
        @test all(isequal.(pconj, pident))    # and agree on signed zeros too
    end

    ro_b = scatter_lane_offsets(NR)
    for n in (NR, 1)
        src = SourceTile(storage, 500, ScatterAxis(co, kc), ScatterAxis(ro_b, n))
        csrc = SourceTile(cstorage, 500, ScatterAxis(co, kc), ScatterAxis(ro_b, n))
        pconj = fill(R(-777), packed_b_length(kernel, kc))
        pident = fill(R(-999), packed_b_length(kernel, kc))
        pack_b!(pconj, src, kernel, conj)
        pack_b!(pident, csrc, kernel, identity)
        @test pconj == pident
        @test all(isequal.(pconj, pident))
    end
end

# A guard on the guard: the pin above must be able to fail. A packer that
# applied `conj` per real half would make the two packings identical.
@testset "conjugation pin has teeth (conj packing differs from identity packing)" begin
    for (fa, fb) in ((PlanarFormat(), PlanarFormat()), (OneEFormat(), PlanarFormat()))
        kernel = ComplexKernelDescriptor(Val(4), Val(3), ComplexF64, fa, fb)
        storage = [ComplexF64(i, i + 1) for i in 1:100]
        src = SourceTile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 4, 3))
        pc = zeros(Float64, packed_a_length(kernel, 3))
        pid = zeros(Float64, packed_a_length(kernel, 3))
        pack_a!(pc, src, kernel, conj)
        pack_a!(pid, src, kernel, identity)
        @test pc != pid
    end
end

# =====================================================================
# Packed-length ratios (the reference's D14): 1m's packed A is exactly twice
# planar's, and 1m's packed B is exactly equal -- 1m's cost over planar is
# entirely on the A side.
# =====================================================================

@testset "packed-length ratios: 1m's A is 2x planar's, 1m's B is equal ($T)" for
    T in (ComplexF64, ComplexF32)

    MR, NR = 8, 6
    planar = ComplexKernelDescriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    onem = ComplexKernelDescriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())

    for kc in (0, 1, 7, 256)
        @test packed_a_length(onem, kc) == 2 * packed_a_length(planar, kc)
        @test packed_b_length(onem, kc) == packed_b_length(planar, kc)
        # The lengths are counts of REALS at a LOGICAL (complex) kc.
        @test packed_a_length(planar, kc) == 2 * MR * kc
        @test packed_a_length(onem, kc) == 4 * MR * kc
        @test packed_b_length(planar, kc) == 2 * NR * kc
    end

    @test packed_a_per_k(planar) == 2 * MR
    @test packed_a_per_k(onem) == 4 * MR
    @test packed_b_per_k(planar) == packed_b_per_k(onem) == 2 * NR
    @test reals_per_element(PlanarFormat()) == 2
    @test reals_per_element(OneEFormat()) == 4
end

# =====================================================================
# Padding: literal zero in every real of the lane; transform never called.
# =====================================================================

@testset "complex packing: padding writes literal zero, bypassing transform ($name)" for
    (name, fa) in (("planar", PlanarFormat()), ("1e", OneEFormat()))

    MR, NR, kc = 4, 3, 3
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), ComplexF64, fa, fa)
    rpe = reals_per_element(fa)
    # transform(0) != 0, so applying it to a padding lane would show up.
    nonzero_at_zero = z -> z + ComplexF64(1000, 2000)

    storage = fill(ComplexF64(1, 1), 200)   # nonzero everywhere
    m = 2
    src = SourceTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, 8, kc))
    packed = fill(-777.0, packed_a_length(kernel, kc))   # nonzero prefill
    pack_a!(packed, src, kernel, nonzero_at_zero)

    for p in 0:(kc - 1), t in m:(MR - 1)
        base = p * rpe * MR
        if fa isa PlanarFormat
            @test packed[base + t + 1] === 0.0         # literal +0.0, not -0.0
            @test packed[base + MR + t + 1] === 0.0
        else
            # OneEFormat: ALL FOUR reals of an edge lane are zeroed.
            @test packed[base + 2 * t + 1] === 0.0
            @test packed[base + 2 * t + 2] === 0.0
            @test packed[base + 2 * MR + 2 * t + 1] === 0.0
            @test packed[base + 2 * MR + 2 * t + 2] === 0.0
        end
    end

    n = 1
    srcB = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, 8, n))
    packedB = fill(-777.0, packed_b_length(kernel, kc))
    pack_b!(packedB, srcB, kernel, nonzero_at_zero)
    for p in 0:(kc - 1), j in n:(NR - 1)
        base = p * rpe * NR
        if fa isa PlanarFormat
            @test packedB[base + j + 1] === 0.0
            @test packedB[base + NR + j + 1] === 0.0
        else
            @test packedB[base + 2 * j + 1] === 0.0
            @test packedB[base + 2 * j + 2] === 0.0
            @test packedB[base + 2 * NR + 2 * j + 1] === 0.0
            @test packedB[base + 2 * NR + 2 * j + 2] === 0.0
        end
    end
end

@testset "complex packing: transform never called on padding lanes (call-counting) ($name)" for
    (name, fa) in (("planar", PlanarFormat()), ("1e", OneEFormat()))

    MR, NR, kc = 4, 3, 3
    kernel = ComplexKernelDescriptor(Val(MR), Val(NR), ComplexF64, fa, fa)
    storage = fill(ComplexF64(2, 3), 200)

    calls = Ref(0)
    counting = z -> (calls[] += 1; z)

    m = 2
    src = SourceTile(storage, 0, AffineAxis(0, 1, m), AffineAxis(0, 8, kc))
    packed = zeros(Float64, packed_a_length(kernel, kc))
    pack_a!(packed, src, kernel, counting)
    # Exactly the valid lanes, once each -- not 2*m*kc, which is what a packer
    # that applied `transform` per real half would do.
    @test calls[] == m * kc

    calls[] = 0
    n = 1
    srcB = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, 8, n))
    packedB = zeros(Float64, packed_b_length(kernel, kc))
    pack_b!(packedB, srcB, kernel, counting)
    @test calls[] == n * kc
end

# =====================================================================
# Validation discipline, and kc == 0
# =====================================================================

@testset "complex packing: validation before any write, kc == 0 is a no-op" begin
    kernel = ComplexKernelDescriptor(Val(4), Val(3), ComplexF64, PlanarFormat(), PlanarFormat())
    storage = fill(ComplexF64(3, 4), 100)
    src = SourceTile(storage, 0, AffineAxis(0, 1, 4), AffineAxis(0, 8, 2))

    # The buffer holds realtype(kernel), NOT scalartype(kernel): a complex
    # buffer is the conflation this milestone is most exposed to.
    @test_throws ArgumentError pack_a!(zeros(ComplexF64, 1000), src, kernel, identity)
    @test_throws ArgumentError pack_b!(zeros(ComplexF64, 1000), src, kernel, identity)
    @test_throws ArgumentError pack_a!(zeros(Float32, 1000), src, kernel, identity)

    # Too many rows / columns.
    toowide = SourceTile(storage, 0, AffineAxis(0, 1, 5), AffineAxis(0, 8, 2))
    @test_throws ArgumentError pack_a!(zeros(Float64, 1000), toowide, kernel, identity)
    toowideB = SourceTile(storage, 0, AffineAxis(0, 1, 2), AffineAxis(0, 8, 4))
    @test_throws ArgumentError pack_b!(zeros(Float64, 1000), toowideB, kernel, identity)

    # Short buffer: lengths count reals at a LOGICAL kc, so a buffer sized as
    # if it held complex elements is exactly half as long and must throw.
    @test_throws DimensionMismatch pack_a!(
        zeros(Float64, packed_a_length(kernel, 2) - 1), src, kernel, identity
    )
    @test_throws DimensionMismatch pack_a!(zeros(Float64, 4 * 2), src, kernel, identity)

    # All validation happens before any write.
    canary = fill(-42.0, packed_a_length(kernel, 2))
    @test_throws ArgumentError pack_a!(canary, toowide, kernel, identity)
    @test all(==(-42.0), canary)

    # kc == 0: no read, no write, returns the buffer.
    read_flag = Ref(false)
    zerokc = SourceTile(storage, 0, AffineAxis(0, 1, 3), AffineAxis(0, 8, 0))
    packed = fill(-42.0, 8)
    @test pack_a!(packed, zerokc, kernel, z -> (read_flag[] = true; z)) === packed
    @test all(==(-42.0), packed)
    zerokcB = SourceTile(storage, 0, AffineAxis(0, 1, 0), AffineAxis(0, 8, 3))
    packedB = fill(-42.0, 8)
    @test pack_b!(packedB, zerokcB, kernel, z -> (read_flag[] = true; z)) === packedB
    @test all(==(-42.0), packedB)
    @test read_flag[] == false

    # Storage bounds are checked before the @inbounds loop.
    tiny = fill(ComplexF64(1, 1), 4)
    oob = SourceTile(tiny, 0, AffineAxis(0, 1, 4), AffineAxis(0, 8, 2))
    @test_throws BoundsError pack_a!(zeros(Float64, 1000), oob, kernel, identity)
end

# =====================================================================
# Zero steady-state allocation, regular and scattered, in every format.
# Each probe is its own function so `transform` is a bound type parameter at
# the call site rather than a Union over a heterogeneous tuple.
# =====================================================================

probe_a(packed, src, kernel, f) =
    (pack_a!(packed, src, kernel, f); @allocated pack_a!(packed, src, kernel, f))
probe_b(packed, src, kernel, f) =
    (pack_b!(packed, src, kernel, f); @allocated pack_b!(packed, src, kernel, f))

@testset "complex packing: zero steady-state allocation" begin
    nontrivial(z) = 2 * z + oneunit(z)

    function run(::Type{T}, fa, fb) where {T}
        MR, NR, kc = 8, 6, 4
        kernel = ComplexKernelDescriptor(Val(MR), Val(NR), T, fa, fb)
        R = real(T)
        packed_a = zeros(R, packed_a_length(kernel, kc))
        packed_b = zeros(R, packed_b_length(kernel, kc))
        storage = rand(T, 4000)

        src_a = SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, kc))
        src_b = SourceTile(storage, 0, AffineAxis(0, 1, kc), AffineAxis(0, kc, NR))

        ro = collect(0:(MR - 1)) .* 3
        co = [0, 100, 250, 400]
        ro_b = collect(0:(NR - 1)) .* 5
        src_a_s = SourceTile(storage, 0, ScatterAxis(ro, MR), ScatterAxis(co, kc))
        src_b_s = SourceTile(storage, 0, ScatterAxis(co, kc), ScatterAxis(ro_b, NR))

        src_kc0 = SourceTile(storage, 0, AffineAxis(0, 1, MR), AffineAxis(0, MR, 0))

        return (
            probe_a(packed_a, src_a, kernel, identity),
            probe_a(packed_a, src_a, kernel, conj),
            probe_a(packed_a, src_a, kernel, nontrivial),
            probe_b(packed_b, src_b, kernel, identity),
            probe_b(packed_b, src_b, kernel, conj),
            probe_b(packed_b, src_b, kernel, nontrivial),
            probe_a(packed_a, src_a_s, kernel, identity),
            probe_a(packed_a, src_a_s, kernel, conj),
            probe_a(packed_a, src_a_s, kernel, nontrivial),
            probe_b(packed_b, src_b_s, kernel, identity),
            probe_b(packed_b, src_b_s, kernel, conj),
            probe_b(packed_b, src_b_s, kernel, nontrivial),
            probe_a(packed_a, src_kc0, kernel, identity),
        )
    end

    @test all(iszero, run(ComplexF64, PlanarFormat(), PlanarFormat()))
    @test all(iszero, run(ComplexF64, OneEFormat(), PlanarFormat()))
    @test all(iszero, run(ComplexF32, PlanarFormat(), PlanarFormat()))
    @test all(iszero, run(ComplexF32, OneEFormat(), PlanarFormat()))
end

# =====================================================================
# Packing into a SubArray sliver of a macro panel, as the driver does.
# =====================================================================

@testset "complex packing: SubArray sliver matches a fresh Vector" begin
    for (fa, fb) in ((PlanarFormat(), PlanarFormat()), (OneEFormat(), PlanarFormat()))
        MR, NR, kc = 4, 3, 5
        kernel = ComplexKernelDescriptor(Val(MR), Val(NR), ComplexF64, fa, fb)
        storage = [ComplexF64(i, -i) for i in 1:2000]
        src = SourceTile(storage, 10, AffineAxis(0, 1, MR), AffineAxis(0, 64, kc))

        needed = packed_a_length(kernel, kc)
        fresh = zeros(Float64, needed)
        pack_a!(fresh, src, kernel, conj)

        big = fill(-1.0, needed + 20)
        sliver = view(big, 8:(7 + needed))
        pack_a!(sliver, src, kernel, conj)
        @test collect(sliver) == fresh
        @test all(==(-1.0), big[1:7])
        @test all(==(-1.0), big[(8 + needed):end])
    end
end
