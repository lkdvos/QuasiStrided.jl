# Contiguous packing fast paths: a straight copy for real A, and a
# deinterleave-and-copy for complex A/B, each used only for the one sliver
# shape it can serve.

# `transform` is `identity` or `conj` (src/planning/plan.jl, `plan_contract`). On a
# real element type `conj` is the identity, so both admit a straight copy.
# This is only the *value* half of the eligibility test: a straight copy is
# vectorizable only in conjunction with the eltype/storage/destination
# conditions in `_pack_a_contiguous_eligible` below.
@inline _copies_unchanged(::typeof(identity), ::Type) = true
@inline _copies_unchanged(::typeof(conj), ::Type{T}) where {T <: Real} = true
@inline _copies_unchanged(::Any, ::Type) = false

# Gate for `_pack_a_contiguous!`, kept as its own function so a test can
# assert it fires for the driver's argument types and stays off for every
# ineligible shape (test/packing/test_pack_real.jl). All but `m == MR` and the stride
# test fold at compile time (they inspect types only). `_unit_stride_rows`
# (src/layout/tiles.jl) has methods for exactly the three `Axis` kinds and
# deliberately NO fallback: an unknown axis type must be a MethodError here,
# never a silent `true`/`false`.
@inline function _pack_a_contiguous_eligible(
        packed::V, source::QSTile, transform::F, m::Int, ::Val{MR}, ::Type{T}
    ) where {V, F, MR, T}
    return packed isa PackedPanel{T} && source.storage isa DenseVector{T} &&
        _copies_unchanged(transform, T) && m == MR && _unit_stride_rows(source.rows)
end

# Fast path for the common A sliver: unit-stride rows filling the whole
# register tile, straight copy, `PackedPanel` destination, dense storage --
# every M-sliver of every column-major (or unit-stride-fastest multi-index) A.
# Each K step is then `MR` contiguous source elements
# landing at `MR` contiguous packed offsets (`i + MR*p`), i.e. one
# `Vec{MR,T}` load/store per K step. Padding never arises here (`m == MR`),
# and `_check_pack_a` has already validated every address `base + rows.base +
# i + col_offset(p)`, `0 <= i < MR`, against `length(storage)` -- exactly the
# span each `vload` reads -- so nothing is read that the scalar path would not
# have read. Same eligibility shape as `_vector_store_eligible`
# (src/microkernels/simd.jl).
@inline function _pack_a_contiguous!(
        packed::PackedPanel{T}, storage::DenseVector{T}, rowbase::Int, cols::C,
        ::Val{MR}, kc::Int
    ) where {T, C, MR}
    GC.@preserve storage begin
        sp = pointer(storage)
        dp = packed.ptr
        for p in 0:(kc - 1)
            v = vload(Vec{MR, T}, sp + sizeof(T) * (rowbase + axis_offset(cols, p)))
            vstore(v, dp + sizeof(T) * (MR * p))
        end
    end
    return packed
end

# ---------------------------------------------------------------------------
# Complex packing fast path (deinterleave-and-copy)
#
# Design: docs/proposals/complex-fast-paths.md, Sections 4.3 (PlanarFormat)
# and 4.4 (OneEFormat's A panel). It is the complex counterpart of
# `_pack_a_contiguous!` above and nothing more: a leaf-level alternative inside
# `_pack_a!`/`_pack_b!` for the ONE sliver shape it can serve, with the shared
# `_pack_panel!` loop (src/packing/pack.jl) as the fallback for everything
# else. It uses the same format, offset formula and transform contract, and
# must produce the same bytes the scalar loop would.
#
# Why this is the easy half of the complex round trip (proposal Section 4.3):
# packing applies no `alpha`/`beta`, never reads its destination, and its only
# `transform`s are `identity` and `conj` -- and `conj` on a complex element
# touches the imaginary half alone. So there is no cross-plane arithmetic here
# at all, only a deinterleave of the source's native `[re,im,re,im,...]` layout
# and, for `conj`, a sign flip on the lanes that carry `im`.
#
# NOT an `unsafe_` path: it skips no validation a caller would otherwise get.
# `_check_pack_a`/`_check_pack_b` run first and unchanged (including the
# storage-bounds check, in the `Val(true)` mode), and the addresses the `vload`
# below reads are EXACTLY the `PD` elements the scalar loop would have read at
# the same K step -- `m == PD` (no padding lanes) and unit-stride lanes make
# the two address sets identical, element for element.
# ---------------------------------------------------------------------------

# The value half of the gate, mirroring `_copies_unchanged` above: the two
# transforms the driver can produce (src/planning/plan.jl, `plan_contract`) are the
# two the shuffle patterns below cover. Anything else -- including the
# arbitrary closures test/packing/test_pack_complex.jl packs with -- is a MUST-fall-
# back, not a MAY: `_pack_alt` has no method for it.
@inline _complex_pack_transform_eligible(::typeof(identity)) = true
@inline _complex_pack_transform_eligible(::typeof(conj)) = true
@inline _complex_pack_transform_eligible(::Any) = false

# Total over `PackFormat`, so the gate answers for `RealFormat` rather than
# leaving a MethodError as the contract.
@inline _complex_pack_format_eligible(::PlanarFormat) = true
@inline _complex_pack_format_eligible(::OneEFormat) = true
@inline _complex_pack_format_eligible(::PackFormat) = false

"""
    _pack_complex_contiguous_eligible(packed, storage, lane_axis, transform, format, valid, ::Val{PD}, ::Type{T}) -> Bool

Whether a complex sliver can take the vectorized pack path. The complex
counterpart of [`_pack_a_contiguous_eligible`](@ref), and the same shape of
predicate: every clause except `valid == PD`, the stride test and the ISA test
inspects types only, so those three are all that survive to run time at each
specialization.

`lane_axis` is the axis the PACKED index runs along: `source.rows` for A (the
`MR` logical rows of one K step) and `source.cols` for B (the `NR` logical
columns). It must be unit-stride `AffineAxis`, because that is what makes `PD`
consecutive source elements `PD` consecutive `Complex{T}` values in storage and
hence `2PD` consecutive `real(T)`s -- the bitcast the fast path performs is
sound for exactly that case and for no other (proposal Section 3.3; the
"Complex packing" header in src/packing/pack.jl says why a `QSTile` is never
`reinterpret`ed in general).

`valid == PD` excludes every partial sliver, so the fast path never has to
write a padding lane and the "padding is a literal zero, never `-0.0`" contract
stays entirely with `_pack_emit_zero!`.
"""
@inline function _pack_complex_contiguous_eligible(
        packed::V, storage::S, lane_axis::AX, transform::F, format::FMT,
        valid::Int, ::Val{PD}, ::Type{T}
    ) where {V, S, AX, F, FMT, PD, T}
    return packed isa PackedPanel{real(T)} && storage isa DenseVector{T} &&
        _complex_pack_format_eligible(format) &&
        _complex_pack_transform_eligible(transform) &&
        valid == PD && _unit_stride_rows(lane_axis) &&
        _complex_fastpath_isa_eligible()
end

# The second shuffle operand. Every pattern below reads the lanes that carry
# `im` out of this vector and the lanes that carry `re` out of `src`, so the
# whole of `transform` is the choice made here -- `identity` takes `im`
# unchanged, `conj` takes it from `-src`. `-src` is an `fneg` on every lane,
# i.e. a sign-bit flip, which is bit-for-bit what the scalar path's
# `imag(conj(z))` computes (including on `-0.0` and on NaN payloads; a multiply
# by `-1.0` would NOT be, which is why no sign-pattern constant appears here).
@inline _pack_alt(src::Vec, ::typeof(identity)) = src
@inline _pack_alt(src::Vec, ::typeof(conj)) = -src

# OneEFormat's second `2PD`-real region wants the OPPOSITE choice from its
# first (proposal Section 4.4: `conj` "just swaps which plane is free"),
# because that region stores `-im` where the first stores `+im`.
@inline _pack_alt_flipped(src::Vec, ::typeof(identity)) = -src
@inline _pack_alt_flipped(src::Vec, ::typeof(conj)) = src

# --- shuffle primitives ----------------------------------------------------
#
# GUARDRAIL: every index tuple below is built HERE, from `PD`, at specialization
# time -- never hardcoded to one register shape, and never a runtime gather. `PD` is `mr(kernel)`/`nr(kernel)`, which the driver
# derives from `kernel_shapes`; the patterns therefore follow the shipped menus
# automatically (proposal Section 6.2). These are `@generated` for the same
# reason the store paths in src/microkernels/simd.jl are: `shufflevector` needs a
# literal `Val` index tuple, and `Val(ntuple(...))` is not reliably one.
#
# `src` is `[re_0, im_0, ..., re_{PD-1}, im_{PD-1}]`, the `PD` source elements
# of one K step read through their native `Complex{T}` layout.

# PlanarFormat ("1r"), one whole K step: `[re_0 .. re_{PD-1} | im_0 .. im_{PD-1}]`.
# Both halves are one contiguous `2PD`-real run at `p * packed_*_per_k`
# (src/microkernels/interface.jl, `packed_a_plane_offset`: plane 0 at `+0`, plane 1 at
# `+PD`, adjacent), so the whole K step is a single store -- the destination
# needs no shuffle of its own, only the source needs deinterleaving.
@generated function _planar_pack_shuffle(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_planar_pack_shuffle: expected N == 2PD")))
    # lane j < PD  -> re_j      = src[2j]
    # lane PD + t  -> (+/-)im_t = alt[2t + 1]   (offset N: second shuffle source)
    idx = ntuple(k -> (k - 1) < PD ? 2 * (k - 1) : N + 2 * ((k - 1) - PD) + 1, 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# OneEFormat ("1e"), first of the two `2PD`-real regions of a K step
# (`plane_offset(0, ..)`): `[re_0, s*im_0, re_1, s*im_1, ...]`, with `s = +1`
# under `identity` and `-1` under `conj`. At `identity` this is a pure copy of
# `src` -- the pattern is the identity permutation over the first source -- and
# LLVM folds the shuffle away entirely, leaving a plain memory copy.
@generated function _onee_pack_shuffle_a(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_onee_pack_shuffle_a: expected N == 2PD")))
    # even lane 2t -> re_t      = src[2t]
    # odd  lane    -> (+/-)im_t = alt[2t + 1]
    idx = ntuple(k -> iseven(k) ? N + (k - 1) : (k - 1), 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# OneEFormat, second region (`plane_offset(2, ..)`, i.e. `+2PD`):
# `[-s*im_0, re_0, -s*im_1, re_1, ...]` -- the adjacent-pair swap, with the
# sign carried by whichever of `src`/`-src` the caller passes as `alt`.
@generated function _onee_pack_shuffle_b(
        src::Vec{N, R}, alt::Vec{N, R}, ::Val{PD}
    ) where {N, R, PD}
    N == 2 * PD || return :(throw(ArgumentError("_onee_pack_shuffle_b: expected N == 2PD")))
    # even lane 2t   -> (-/+)im_t = alt[2t + 1]
    # odd  lane 2t+1 -> re_t      = src[2t]
    idx = ntuple(k -> isodd(k) ? N + k : k - 2, 2 * PD)
    return :(shufflevector(src, alt, Val($idx)))
end

# --- the two packers -------------------------------------------------------
#
# `elembase` is the zero-based storage address of lane 0 at K step 0 minus the
# step axis's contribution, i.e. `source.base + lane_axis.base`; `steps` is the
# OTHER axis (`source.cols` for A, `source.rows` for B) and may be scattered --
# only the lane axis has to be unit-stride. Lane `t` of K step `p` therefore
# lives at element `elembase + axis_offset(steps, p) + t`, exactly the address
# `tile_load` would compute, and its two reals at twice that.
#
# `storage::DenseVector{Complex{R}}` is pinned in the signature (rather than a
# free `T`) so that a `packed`/`storage` element-type mismatch is a MethodError
# here instead of a bitcast to the wrong width.

@inline function _pack_complex_contiguous!(
        ::PlanarFormat, packed::PackedPanel{R}, storage::DenseVector{Complex{R}},
        elembase::Int, steps::C, ::Val{PD}, kc::Int, transform::F
    ) where {R, C, PD, F}
    GC.@preserve storage begin
        sp = reinterpret(Ptr{R}, pointer(storage))
        dp = packed.ptr
        for p in 0:(kc - 1)
            src = vload(
                Vec{2 * PD, R},
                sp + sizeof(R) * (2 * (elembase + axis_offset(steps, p)))
            )
            vstore(
                _planar_pack_shuffle(src, _pack_alt(src, transform), Val(PD)),
                dp + sizeof(R) * (2 * PD * p)
            )
        end
    end
    return packed
end

@inline function _pack_complex_contiguous!(
        ::OneEFormat, packed::PackedPanel{R}, storage::DenseVector{Complex{R}},
        elembase::Int, steps::C, ::Val{PD}, kc::Int, transform::F
    ) where {R, C, PD, F}
    GC.@preserve storage begin
        sp = reinterpret(Ptr{R}, pointer(storage))
        dp = packed.ptr
        for p in 0:(kc - 1)
            src = vload(
                Vec{2 * PD, R},
                sp + sizeof(R) * (2 * (elembase + axis_offset(steps, p)))
            )
            # 1e writes twice planar's destination bandwidth by construction
            # (reals_per_element(OneEFormat) == 4); that is the format's cost,
            # not the fast path's.
            at = 4 * PD * p
            vstore(
                _onee_pack_shuffle_a(src, _pack_alt(src, transform), Val(PD)),
                dp + sizeof(R) * at
            )
            vstore(
                _onee_pack_shuffle_b(src, _pack_alt_flipped(src, transform), Val(PD)),
                dp + sizeof(R) * (at + 2 * PD)
            )
        end
    end
    return packed
end
