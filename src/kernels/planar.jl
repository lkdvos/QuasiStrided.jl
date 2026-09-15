# Planar (split-complex, BLIS "1r") microkernel: the default complex method.
#
# Both packed panels hold `[re_0 .. re_{n-1} | im_0 .. im_{n-1}]` per logical K
# step (`PlanarFormat`), so the data is already in the right lanes and the body
# is four real FMAs per (A-vector, B-scalar) pair -- no shuffles, no `fmaddsub`,
# no duplicated lanes. Everything below this boundary works exclusively in
# `real(T)`: packed buffers, `SIMD.Vec` lanes and the accumulator.
#
# Two *independent* performance cliffs run through this file and must not be
# conflated (docs/decisions.md, "The planar microkernel: accumulator, body, and
# two independent cliffs"):
#
#   Cliff A -- architectural register spill. Hardware. See
#   `planar_register_pressure`.
#
#   Cliff B -- Julia's tuple lowering. Dynamic `NTuple` indexing above NV = 16
#   makes the compiler heap-allocate the accumulator: 24576 B per `execute!`,
#   measured in Phase H. Planar at 16x6 is NV_total = 24, so the cliff is live
#   from the first line of code. Therefore *every* accumulate and store here is
#   `@generated` with literal tuple indices, including the lane tail -- the real
#   path's runtime-indexed `_acc_lane` helper (src/kernels/simd.jl) must not be
#   used here.

using SIMD: Vec, vload, vstore

"""
    PlanarKernel{MR,NR,T,W}(descriptor)
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Split-complex microkernel over `SIMD.Vec{W,real(T)}` lanes, implementing
[`PlanarMethod`](@ref). `MR`/`NR` are the **logical** (complex) register-tile
extents and `T` is the **storage** element type (`ComplexF32`/`ComplexF64`);
`W` is the lane count of the *real* type, and `MR` must be a multiple of it
(checked at construction). `NR` need not be — B contributes one complex scalar
per column per K step.

The field is named `descriptor`, so `mr`/`nr`/`scalartype`/`packed_a_length`/
`packed_b_length`/`realtype`/`packed_a_per_k` all forward through the existing
`DescriptorKernel` methods in src/kernel.jl and src/complex_format.jl; this
kernel adds no forwarding of its own beyond the two plane-offset accessors.

The 3-argument form defaults `W` via `_default_lanewidth(real(T))`.
"""
struct PlanarKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, PlanarFormat, PlanarFormat}

    function PlanarKernel{MR, NR, T, W}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, PlanarFormat, PlanarFormat}
        ) where {MR, NR, T, W}
        _check_lanewidth("PlanarKernel", W)
        _check_mr_multiple("PlanarKernel", MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

function PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    return PlanarKernel{MR, NR, T, W}(
        ComplexKernelDescriptor(Val(MR), Val(NR), T, PlanarFormat(), PlanarFormat())
    )
end
function PlanarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("PlanarKernel requires a complex element type, got $T"))
    return PlanarKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::PlanarKernel) = PlanarMethod()

"""
    lanewidth(kernel::PlanarKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`, counted in **reals**.
"""
lanewidth(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::PlanarKernel) -> Int

`mr(kernel) ÷ lanewidth(kernel)`: A vectors (and accumulator vectors) per
output column **per plane**. The accumulator holds twice this many per column,
one set for each of the real and imaginary planes.
"""
avecs_per_column(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} = MR ÷ W

# Plane-offset forwarding, scoped to this kernel type. `packed_a_offset` (the
# single-plane accessor) deliberately has no method for a complex descriptor:
# no complex kernel should ever be asked for one, and the MethodError is the
# intended outcome.
@inline packed_a_plane_offset(k::PlanarKernel, plane::Int, i::Int, p::Int) =
    packed_a_plane_offset(k.descriptor, plane, i, p)
@inline packed_b_plane_offset(k::PlanarKernel, plane::Int, j::Int, p::Int) =
    packed_b_plane_offset(k.descriptor, plane, j, p)

"""
    planar_register_pressure(kernel::PlanarKernel) -> Int

Vector registers live at the bottom of the K loop:

    2*MV*NR accumulators + 2*MV A vectors + 2 B broadcasts

with `MV = mr(kernel) ÷ lanewidth(kernel)`. **Cliff A**: this must be `<=` the
architectural register count (`target_profile().nregisters`; 32 zmm under
AVX-512, 16 ymm under AVX2) or the microkernel spills, costing 30-50%.

**The `<= nregisters` bound is measured to be optimistic, and is not a
predictor.** At the reference shape `(MV, NR) = (2, 6)` it is `24 + 4 + 2 = 30`
and that shape *does* spill (26 stack stores per K step against 48 FMAs, mostly
store-port traffic rather than a load-use chain), while `(24,3,8)` at pressure
26 is clean and `(8,8,8)` at pressure 20 is not -- so spilling is not monotone
in this number and aspect ratio matters independently. Full tables, both
instruments, and the Phase D correction: docs/decisions.md, "Cliff A bites at
the shipped shape" and "Correcting the Phase C planar spill table". Treat this
as a necessary condition, never a ranking; shapes are ranked on measured
throughput (Phase F) or not at all.
"""
planar_register_pressure(::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W} =
    2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::PlanarKernel{MR,NR,T,W}) -> NTuple{2NV,SIMD.Vec{W,real(T)}}

A logical `MR`-by-`NR` complex zero accumulator, held as one **flat** immutable
tuple of `2NV = 2*(MR÷W)*NR` zero `Vec{W,real(T)}` values: the real plane at
tuple indices `1:NV` and the imaginary plane at `NV+1:2NV`, each using the real
path's `(v, j) -> v + (MR÷W)*j + 1` convention. Physical row `i` of vector `v`
is lane `i - v*W + 1` (1-based `SIMD.Vec` indexing).

Flat rather than nested: it keeps every signature the same *shape* as the real
path, which is the pattern Phase H proved keeps the accumulator
register-resident.
"""
function zero_accumulator(kernel::PlanarKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, real(T)})
    return ntuple(_ -> z, Val(2 * (MR ÷ W) * NR))
end

# Fully unrolled, closure-free K-step body, mirroring `_accumulate_step` in
# src/kernels/simd.jl: per logical K step, MV A vector loads per plane, NR B
# scalar loads per plane, 4*MV*NR FMAs, literal tuple indices throughout
# (Cliff B).
#
# GUARDRAIL: the real part is `muladd(-ai, bi, muladd(ar, br, c))`. `c - ai*bi`
# is REJECTED and must not be reintroduced: Julia does not set LLVM's
# `contract` fast-math flag by default, so that form does NOT fuse -- two
# instructions, and different rounding from the other three terms.
#
# `nai_v = -ai_v` is hoisted out of the `j` loop so a declined fold would cost
# MV extra ops per K step rather than MV*NR. Verified by `@code_native` at
# every menu shape: exactly `MV*NR` `vfnmadd231` + `3*MV*NR` `vfmadd231` and
# ZERO separate negations, so the hoist is free rather than merely cheap
# (per-shape table in docs/decisions.md, "the per-shape `vfnmadd` count").
@generated function _accumulate_step_planar(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    R === real(T) ||
        throw(
        ArgumentError(
            "_accumulate_step_planar: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    MV = MR ÷ W
    NV = MV * NR
    2NV == NA ||
        throw(
        ArgumentError(
            "_accumulate_step_planar: accumulator length $NA does not match " *
                "2*(mr÷W)*nr = $(2NV) for MR=$MR, NR=$NR, W=$W"
        )
    )

    arv = [Symbol(:ar, v) for v in 0:(MV - 1)]
    aiv = [Symbol(:ai, v) for v in 0:(MV - 1)]
    naiv = [Symbol(:nai, v) for v in 0:(MV - 1)]
    brv = [Symbol(:br, j) for j in 0:(NR - 1)]
    biv = [Symbol(:bi, j) for j in 0:(NR - 1)]

    load_a = Any[]
    for v in 0:(MV - 1)
        push!(
            load_a,
            :(
                $(arv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_plane_offset(kernel, 0, $(v * W), p)
                )
            )
        )
        push!(
            load_a,
            :(
                $(aiv[v + 1]) = panel_vload(
                    Vec{$W, $R}, packed_a, packed_a_plane_offset(kernel, 1, $(v * W), p)
                )
            )
        )
        push!(load_a, :($(naiv[v + 1]) = -$(aiv[v + 1])))
    end

    load_b = Any[]
    for j in 0:(NR - 1)
        push!(load_b, :($(brv[j + 1]) = panel_load(packed_b, packed_b_plane_offset(kernel, 0, $j, p))))
        push!(load_b, :($(biv[j + 1]) = panel_load(packed_b, packed_b_plane_offset(kernel, 1, $j, p))))
    end

    acc_exprs = Vector{Any}(undef, NA)
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        # re: ar*br - ai*bi, as two chained fused ops with a negated operand.
        acc_exprs[idx] = :(
            muladd(
                $(naiv[v + 1]), $(biv[j + 1]),
                muladd($(arv[v + 1]), $(brv[j + 1]), acc[$idx])
            )
        )
        # im: ar*bi + ai*br.
        acc_exprs[NV + idx] = :(
            muladd(
                $(aiv[v + 1]), $(brv[j + 1]),
                muladd($(arv[v + 1]), $(biv[j + 1]), acc[$(NV + idx)])
            )
        )
    end

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(load_a...)
            $(load_b...)
            return $(Expr(:tuple, acc_exprs...))
        end
    end
end

"""
    accumulate(kernel::PlanarKernel, acc, packed_a, packed_b, kc::Int) -> acc

Planar counterpart of `SIMDKernel`'s `accumulate`. `kc` is the **logical**
(complex) K depth; `kc == 0` returns `acc` unchanged without reading the packed
buffers. Both panels must be in [`PlanarFormat`](@ref). Not bitwise identical
to a scalar complex dot product (FMA grouping differs) — compare with a
tolerance, never `==`.

`packed_a`/`packed_b` may be a [`PackedPanel`](@ref) of `real(T)` — what the
driver passes, and the only form that keeps a large accumulator
register-resident — or any contiguous `AbstractVector{real(T)}`.
"""
function Base.accumulate(
        kernel::PlanarKernel{MR, NR, T, W}, acc::NTuple{NA, Vec{W, R}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, R, NA, PA, PB}
    kc == 0 && return acc
    kc > 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    @inbounds for p in 0:(kc - 1)
        acc = _accumulate_step_planar(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# ----------------------------------------------------------------------------
# store_tile!
# ----------------------------------------------------------------------------

# Scattered/scalar store, `@generated` so every `acc[...]` is a compile-time
# index (Cliff B; see src/kernels/simd.jl's `_store_tile_scattered!` for the
# measured number). Only the lane index inside a single `Vec` is a runtime
# value.
#
# The fused form is kept deliberately: `store_tile!` is already specialised per
# `(QSTile{S,R,C}, kernel)` by dispatch, so the reference project's
# kernel-writes-a-stack-tile / separate-writeback split would buy nothing and
# would re-introduce the memory round-trip Phase H removed at ~4x
# (docs/decisions.md, "The fused `store_tile!` is kept"). The unit-stride
# plane-to-interleave fast path is a deliberately deferred, measurement-gated
# follow-on and is NOT built here.
@generated function _store_tile_planar!(
        destination::QSTile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::PlanarKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NA}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_planar!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    MV = MR ÷ W
    NV = MV * NR
    2NV == NA ||
        throw(
        ArgumentError(
            "_store_tile_planar!: accumulator length $NA does not match " *
                "2*(mr÷W)*nr = $(2NV) for MR=$MR, NR=$NR, W=$W"
        )
    )

    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        push!(
            blocks, quote
                if $j < n
                    revec = acc[$idx]
                    imvec = acc[$(NV + idx)]
                    for lane in 1:$W
                        i = $(v * W) + lane - 1
                        i < m || break
                        _axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(revec[lane], imvec[lane]), beta
                        )
                    end
                end
            end
        )
    end
    return quote
        @inbounds begin
            $(blocks...)
        end
        return destination
    end
end

"""
    store_tile!(destination::QSTile, acc::NTuple{2NV,SIMD.Vec{W,real(T)}}, alpha::T, beta::T, kernel::PlanarKernel) -> destination

Planar counterpart of `SIMDKernel`'s `store_tile!`, with the **same contract**:
`C[i,j] = alpha*R[i,j] + beta*C[i,j]` over the valid rectangle only;
`alpha == 0` never reads `acc`; `beta == 0` never reads old `C`; `beta == 1`
skips the multiplication; padding lanes (`i >= m`, `j >= n`) are never read, so
nonfinite padding in `acc` cannot propagate; an empty destination is a no-op in
every branch.

Ships the scattered/scalar path only, recombining `Complex(re[lane], im[lane])`
and delegating to the existing generic `_axpby_tile!` (src/kernel.jl), which is
already generic in the value type and provides those `beta` shortcuts.
"""
function store_tile!(
        destination::QSTile, acc::NTuple{NA, Vec{W, R}},
        alpha::T, beta::T, kernel::PlanarKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NA}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    return _store_tile_planar!(destination, acc, alpha, beta, kernel, m, n)
end

"""
    execute_tile!(kernel::PlanarKernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

Planar counterpart of `SIMDKernel`'s `execute_tile!`; same validation order and
short-circuits (both are `_execute_tile_prologue!`'s). `kc` is the **logical**
(complex) K depth, and the buffer-length checks go through
`packed_a_length`/`packed_b_length`, which take a logical `kc` and return a
count of **reals**.
"""
function execute_tile!(
        kernel::PlanarKernel{MR, NR, T, W}, destination::QSTile,
        packed_a::PA, packed_b::PB, kc::Int, alpha, beta
    ) where {MR, NR, T, W, PA, PB}
    run, alphaT, betaT =
        _execute_tile_prologue!(kernel, destination, packed_a, packed_b, kc, alpha, beta)
    run || return destination
    acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end
