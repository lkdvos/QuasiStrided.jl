# contract!'s frozen signature/label semantics: see docs/decisions.md.
# Planning (labels, AxisGroups, buffers) is split from execution so a
# ContractPlan can be built once and reused:
#   plan_contract(...) -> ContractPlan; execute!(plan, alpha, beta); contract! = both.
# execute! is a BLIS five-loop (NC/KC/MC) nest with packed-panel reuse; the
# pre-macro-blocking tile-by-tile driver is kept unexported as
# `execute_tilewise!`, an independent correctness oracle for it.
# The buffers themselves live in a separate, reusable `ContractWorkspace`
# (src/workspace.jl; docs/decisions.md, "Amendment 1").

# Frozen module-import convention (docs/decisions.md): TO names are always
# qualified; a bare `using TensorOperations` would collide on `scalartype`.
import TensorOperations as TO

# Classify every label in indA ∪ indB ∪ indC into M/N/K. Returns
# (mlabels, nlabels, klabels) in indA/indB appearance order. Per (inA,inB,inC):
#   (T,F,T)->M  (F,T,T)->N  (T,T,F)->K  everything else -> ArgumentError
# (labeled in C only, present in all three, or dangling in just A or B).
function _classify_labels(
        indA::NTuple{NA, Int}, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int}
    ) where {NA, NB, NC}
    allunique(indA) ||
        throw(ArgumentError("indA has a repeated label (diagonal), not supported: $indA"))
    allunique(indB) ||
        throw(ArgumentError("indB has a repeated label (diagonal), not supported: $indB"))
    allunique(indC) ||
        throw(ArgumentError("indC has a repeated label (diagonal), not supported: $indC"))

    setA = Set(indA)
    setB = Set(indB)
    setC = Set(indC)

    mlabels = Int[]
    klabels = Int[]
    for lbl in indA
        inB = lbl in setB
        inC = lbl in setC
        if inB && inC
            throw(
                ArgumentError(
                    "label $lbl appears in indA, indB, and indC: labels present in all " *
                        "three operands (batch-like) are out of scope for this milestone"
                )
            )
        elseif inB && !inC
            push!(klabels, lbl)
        elseif !inB && inC
            push!(mlabels, lbl)
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indA (not in indB or indC): not a valid " *
                        "free (M) or contracted (K) label"
                )
            )
        end
    end

    nlabels = Int[]
    for lbl in indB
        inA = lbl in setA
        inC = lbl in setC
        if inA && inC
            continue  # already rejected while scanning indA, above.
        elseif inA && !inC
            continue  # already classified as K, above.
        elseif !inA && inC
            push!(nlabels, lbl)
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indB (not in indA or indC): not a valid " *
                        "free (N) or contracted (K) label"
                )
            )
        end
    end

    for lbl in indC
        inA = lbl in setA
        inB = lbl in setB
        (inA || inB) ||
            throw(ArgumentError("label $lbl appears in indC but not in indA or indB"))
    end

    return mlabels, nlabels, klabels
end

# Build the two-map AxisGroup for one of M/N/K: (v1,v2) is (A,C)/(B,C)/(A,B).
# Raises DimensionMismatch on a matched-label length mismatch.
function _build_pair_group(
        labels::Vector{Int},
        ind1::NTuple, v1::StridedView,
        ind2::NTuple, v2::StridedView
    )
    D = length(labels)
    pos1 = ntuple(d -> findfirst(==(labels[d]), ind1)::Int, D)
    pos2 = ntuple(d -> findfirst(==(labels[d]), ind2)::Int, D)
    lens = ntuple(D) do d
        l1 = size(v1, pos1[d])
        l2 = size(v2, pos2[d])
        l1 == l2 || throw(
            DimensionMismatch(
                "label $(labels[d]) has mismatched axis length: $l1 vs $l2"
            )
        )
        l1
    end
    s1 = ntuple(d -> Base.strides(v1)[pos1[d]], D)
    s2 = ntuple(d -> Base.strides(v2)[pos2[d]], D)
    return AxisGroup(lens, (s1, s2))
end

# Engine-wide default kernel: shape from ONE detected capability, the vector
# register width -- `W = vector_bytes/sizeof(T)`, `MR = 2W`, `NR = NR_DEFAULT`,
# so `NV = 12`. Reproduces the swept optimum for both dtypes on AVX-512 and
# reduces to the previous hardcoded (8,6,4) on AVX2; NV = 12 is also the only
# setting that fits a 16-register AVX2 machine and Julia 1.10's register
# allocation (docs/decisions.md, Amendment 2 and Phases G/H).
const NR_DEFAULT = 6

_legacy_shape(::Type{T}) where {T} = (8, 6, _default_lanewidth(T))

# Complex counterpart: same `(8, 6)` register tile in LOGICAL (complex) rows,
# with the lane width taken from the real type, since every packed buffer and
# every `SIMD.Vec` below the kernel boundary is made of `real(T)`.
_legacy_shape(::Type{T}) where {T <: Complex} = (8, 6, _default_lanewidth(real(T)))

# Deliberately empty for the REAL types: after Phase H the rule above is
# already the measured optimum (1st of 24 for Float32, 2nd of 33 by 0.01% for
# Float64), so a row here would only pin this package to one machine's noise.
_shape_override(::Val, ::Type) = nothing

# For COMPLEX types the rule IS overridden, and these two rows are the only
# swept constants in the package. The derived `MR = 2W, NR = 6` shape measured
# as the *worst* planar configuration -- 38% (`ComplexF64`) / 41%
# (`ComplexF32`) off the best -- and `24x3`/`48x3` won outright. Ranking table,
# canary spread and the spill mechanism that predicted it: docs/decisions.md,
# "The register shape: the derived rule was wrong for complex by 38-41%".
#
# These are ccqlin038 measurements and are reached only on `:avx512`
# (`_rule_applies_complex`), so no other microarchitecture is handed them.
_shape_override(::Val{:avx512}, ::Type{ComplexF64}) = (24, 3, 8)
_shape_override(::Val{:avx512}, ::Type{ComplexF32}) = (48, 3, 16)

# NEON: **measured**, on an Apple M3 Max, by the sibline `tensorcontract-rs`
# project (`crates/tensorcontract/src/kernel/aarch64.rs`, `cfg_neon_f64` /
# `cfg_neon_f32`, three arms at `kc = 384`). Its planar winner is
# `(MV, NR) = (2, 6)` for both precisions, which converts to these two rows --
# `MR = MV * lanes`, lanes being 2 for `ComplexF64` and 4 for `ComplexF32` on
# 128-bit vectors.
#
# The register-budget fit already selected exactly these shapes, so this is a
# pin rather than a change. Pinned anyway, because reaching a measured optimum
# by coincidence is fragile: a later menu edit could move it silently, and
# nothing would notice.
_shape_override(::Val{:neon}, ::Type{ComplexF64}) = (4, 6, 2)
_shape_override(::Val{:neon}, ::Type{ComplexF32}) = (8, 6, 4)

# AVX2: **modelled, not measured** -- by anyone. Adopted from the same sibling
# project (`cfg_avx2_f64`, explicitly labelled "provisional and unmeasured"),
# whose planar choice is `(MV, NR) = (1, 5)`.
#
# It is adopted over what the budget fit picks -- `NR = 6` -- for one reason:
# `NR = 6` at `MV = 1` costs `2*6 + 2 + 2 = 16` registers out of AVX2's 16,
# leaving LLVM nothing for address arithmetic or loop counters, so it will
# spill something. `NR = 5` costs 14 and leaves two. The sibling's table
# records the same figure as `live 14`. The headroom argument is sound
# independently of whether 5 is the exact optimum, which is the part nobody
# has measured.
#
# This cannot be measured on `ccqlin038`: it has 32 registers, so forcing an
# AVX2 *shape* there would not exercise the 16-register constraint that
# motivates the row. It needs AVX2-only hardware, and is cheap to revisit --
# `benchmark/bench_complex_efficiency.jl` arm 2 is the sweep.
_shape_override(::Val{:avx2}, ::Type{ComplexF64}) = (4, 5, 4)
_shape_override(::Val{:avx2}, ::Type{ComplexF32}) = (8, 5, 8)

# The rule applies only to the ISAs it was validated on. `:neon` is detected
# but deliberately gets the legacy shape: no aarch64 measurement exists, and
# the rule would pick MR = 2W = 4 on 128-bit lanes -- narrower and smaller than
# the legacy (8,6,4), not obviously better. Derive where measured, fall back
# everywhere else.
_rule_applies(::Val{:avx512}) = true
_rule_applies(::Val{:avx2}) = true
_rule_applies(::Val) = false

# Complex counterpart, `:avx512` only, for the same kind of reason as the
# `:neon` fallback above: AVX2 has 16 ymm and planar holds separate real and
# imaginary accumulator planes, so even (MV,NR) = (1,6) leaves zero spare
# there (docs/decisions.md, "Cliff A").
_rule_applies_complex(::Val{:avx512}) = true
_rule_applies_complex(::Val) = false

function _derived_shape(profile::TargetProfile, ::Type{T}) where {T}
    vb = profile.vector_bytes
    key = Val(profile.isa)
    (_rule_applies(key) && vb > 0 && vb % sizeof(T) == 0) || return _legacy_shape(T)
    ovr = _shape_override(key, T)
    return ovr === nothing ? (2 * (vb ÷ sizeof(T)), NR_DEFAULT, vb ÷ sizeof(T)) : ovr
end

# A NEW METHOD, not an edit of the one above, so that "the real path is
# bit-identical" is a `git diff` fact rather than an argument about whether
# `real(Float64) === Float64`.
#
# The Phase G rule survives unchanged; only the `sizeof` argument moves to the
# real type, because `W` is a count of REAL lanes -- the whole complex
# adaptation. Factored out of `_derived_shape` so it can be tested
# independently of the override layered on top of it (see `_shape_override`,
# which unlike the real path is NOT empty here).
_complex_rule_shape(vb::Int, ::Type{T}) where {T <: Complex} =
    (2 * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

# Precedence, uniform across ISAs: an explicit override row, then the derived
# rule where it is validated, then a register-budget fit. The override is
# consulted FIRST and for every ISA -- an earlier revision checked
# `_rule_applies_complex` before it, which meant an AVX2 or NEON row could
# never be reached, so the non-AVX-512 rows below would have been dead code.
function _derived_shape(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    key = Val(profile.isa)
    ovr = _shape_override(key, T)
    ovr === nothing || return ovr
    R = real(T)
    vb = profile.vector_bytes
    (_rule_applies_complex(key) && vb > 0 && vb % sizeof(R) == 0) ||
        return _complex_fitted_shape(profile, T)
    return _complex_rule_shape(vb, T)
end

# Closed set, so compiled SIMDKernel (and driver) specializations are bounded.
const KERNEL_SHAPES_F64 = ((8, 6, 4), (16, 6, 8))
const KERNEL_SHAPES_F32 = ((8, 6, 8), (32, 6, 16), (16, 6, 8))

# Complex menus, seeded from the reference's measured AVX-512 shapes, with `MR`
# in LOGICAL complex rows and `W` in real lanes; at most three each, so the
# compiled specialization set stays bounded. The 1m menus look "unaligned"
# (MR = 12 at W = 8) only because 1m runs a real microkernel of `2MR` rows, so
# it is `2*12 = 24` that must be a multiple of `W`. Ordered with the Phase F
# winner first, so the menu head and the shape the engine resolves to agree
# (see `_shape_override`); the SET is unchanged, only the order, so no
# specialization is added or removed (pinned by a test).
#
# The last three entries of each PLANAR menu exist for
# `_complex_fitted_shape` to select off `:avx512`: an `MV = 1` tile at each
# lane width the package compiles, so that for any (lane count, register
# budget >= 16) pair at least one entry fits. They are here
# because `_complex_kernel_from_shape` is `@generated` over this menu and falls
# through to the LAST entry on no match: a resolver that returned a shape
# absent from the menu would silently get a different kernel than it asked for,
# which is worse than either a spill or an error. A test pins that every ISA's
# fitted shape is present. They are unmeasured and are not claimed to be good,
# only to fit.
const KERNEL_SHAPES_C64_PLANAR = (
    (24, 3, 8), (16, 6, 8), (8, 8, 8), (4, 5, 4), (4, 6, 2), (2, 6, 2),
)
const KERNEL_SHAPES_C64_ONEM = ((12, 8, 8), (16, 6, 8), (8, 8, 8))
const KERNEL_SHAPES_C32_PLANAR = (
    (48, 3, 16), (32, 6, 16), (16, 8, 16), (8, 5, 8), (8, 6, 4), (4, 6, 4),
)
const KERNEL_SHAPES_C32_ONEM = ((24, 8, 16), (32, 6, 16), (16, 8, 16))

kernel_shapes(::Type{Float64}) = KERNEL_SHAPES_F64
kernel_shapes(::Type{Float32}) = KERNEL_SHAPES_F32
kernel_shapes(::Type{T}) where {T} = (_legacy_shape(T),)

"""
    kernel_shapes(T, method::ComplexMethod) -> NTuple{<:Any,NTuple{3,Int}}

Method-aware menu. Each complex method has its own measured menu because each
has its own packed A format and therefore its own register budget; `RealMethod`
forwards to the one-argument form so the real path is reached by exactly the
code it always was.
"""
kernel_shapes(::Type{T}, ::RealMethod) where {T} = kernel_shapes(T)
kernel_shapes(::Type{ComplexF64}, ::PlanarMethod) = KERNEL_SHAPES_C64_PLANAR
kernel_shapes(::Type{ComplexF64}, ::OneMMethod) = KERNEL_SHAPES_C64_ONEM
kernel_shapes(::Type{ComplexF32}, ::PlanarMethod) = KERNEL_SHAPES_C32_PLANAR
kernel_shapes(::Type{ComplexF32}, ::OneMMethod) = KERNEL_SHAPES_C32_ONEM
# Any other element type / method pair: the legacy shape, never a guess.
kernel_shapes(::Type{T}, ::ComplexMethod) where {T} = (_legacy_shape(T),)

# Unrolled over `kernel_shapes(T)` so every branch builds a concrete kernel
# from literal `Val`s (the last shape is the fallback); a plain loop would
# construct `Val(cand[1])` dynamically and widen to `Any`. Costs one dynamic
# dispatch per `plan_contract`, none per tile or K step -- `_plan_contract`
# specializes, so `execute!` sees no abstract type.
@generated function _kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T}
    shapes = kernel_shapes(T)
    ex = :(SIMDKernel(Val($(shapes[end][1])), Val($(shapes[end][2])), T, Val($(shapes[end][3]))))
    for (MR, NR, W) in reverse(shapes[1:(end - 1)])
        ex = :(
            shape === ($MR, $NR, $W) ? SIMDKernel(Val($MR), Val($NR), T, Val($W)) : $ex
        )
    end
    return ex
end

@noinline _kernel_for(profile::TargetProfile, ::Type{T}) where {T} =
    _kernel_from_shape(_derived_shape(profile, T), T)

_default_kernel(::Type{T}) where {T} = _kernel_for(target_profile(), T)

# ----------------------------------------------------------------------------
# Complex kernel construction
# ----------------------------------------------------------------------------
# The complex counterpart of `_kernel_from_shape`, generated for the same
# reason: every branch must build a concrete kernel from literal `Val`s.
# An unimplemented method's arm throws rather than silently falling back to
# planar, since a silent method substitution would make a planar-vs-1m
# measurement meaningless.
@generated function _complex_kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method::PlanarMethod
    ) where {T}
    shapes = kernel_shapes(T, PlanarMethod())
    ex = :(PlanarKernel(Val($(shapes[end][1])), Val($(shapes[end][2])), T, Val($(shapes[end][3]))))
    for (MR, NR, W) in reverse(shapes[1:(end - 1)])
        ex = :(
            shape === ($MR, $NR, $W) ? PlanarKernel(Val($MR), Val($NR), T, Val($W)) : $ex
        )
    end
    return ex
end

# The 1m arm, over 1m's OWN menu (each method has its own packed A format and
# therefore its own register budget). Deliberately a second method rather than
# a shared generic over `M <: ComplexMethod`, so the planar arm stays
# byte-identical.
#
# Nothing in the engine reaches this arm: `_default_complex_method` always
# returns `PlanarMethod()` and 1m is selectable ONLY by naming the kernel (no
# auto-dispatch, no env var, no shape rule -- docs/decisions.md, "Method
# ranking does not transfer between machines"). It exists so the function is
# total for callers that name the method themselves.
@generated function _complex_kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method::OneMMethod
    ) where {T}
    shapes = kernel_shapes(T, OneMMethod())
    ex = :(OneMKernel(Val($(shapes[end][1])), Val($(shapes[end][2])), T, Val($(shapes[end][3]))))
    for (MR, NR, W) in reverse(shapes[1:(end - 1)])
        ex = :(
            shape === ($MR, $NR, $W) ? OneMKernel(Val($MR), Val($NR), T, Val($W)) : $ex
        )
    end
    return ex
end

@noinline function _complex_kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method
    ) where {T}
    throw(
        ArgumentError(
            "no microkernel is available for $T at shape $shape under $(method). " *
                "Only $(PlanarMethod()) and $(OneMMethod()) are implemented. Pass an " *
                "explicit `kernel = ...` to plan_contract to use a kernel this " *
                "engine does not pick itself."
        )
    )
end

"""
    _planar_pressure(MR, NR, W) -> Int

Vector registers a planar kernel holds live per K step: `2*MV*NR`
accumulators (two planes) + `2*MV` A vectors (two planes) + 2 B broadcasts,
with `MV = MR ÷ W`.

A *necessary* condition only, not a predictor. Phase D measured that spilling
is not monotone in this number -- planar `(24,3,8)` at 26 is clean while
`(8,8,8)` at 20 spills -- so it is used below to *exclude* shapes that cannot
possibly fit, never to rank the ones that can.
"""
_planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# Largest MENU shape that this host can actually run, for use off `:avx512`.
#
# AMENDS Phase C's decision to refuse a complex default off `:avx512`
# entirely. That was wrong, and CI said so: every runner this package has is
# AVX2 (Linux) or NEON (macOS), so refusing made complex support unavailable
# through `@tensor` on every machine the project tests on, and on most machines
# anyone would run it on. The priority was backwards -- a slow-but-correct
# kernel beats no complex support at all, and "an error beats a guaranteed
# spill" is only defensible when the user has an alternative, which they did
# not.
#
# It selects **from the menu** rather than computing a shape freely, so
# "whatever this returns is in the menu" holds by construction rather than by
# having enumerated the right hardware. That matters because
# `_complex_kernel_from_shape` is `@generated` over the menu and falls through
# to its LAST entry on no match: a freely computed shape absent from the menu
# would silently build a different kernel than was asked for. An earlier
# revision did compute freely, and the test that sweeps synthetic
# `(vector_bytes, nregisters)` pairs caught exactly that.
#
# Two constraints, both necessary:
#
#   * `W <= hardware lanes` -- a `Vec{8,Float64}` on 128-bit NEON is emulated
#     across four registers, so a shape whose pressure "fits" on paper would
#     not fit at all.
#   * `_planar_pressure <= nregisters` -- see that function; a necessary
#     condition, never used here to rank.
#
# `nregisters == 0` (unrecognised CPU) assumes 16, the conservative x86
# baseline, matching the conservatism `_legacy_shape` already applies.
#
# The selected shapes are UNMEASURED off `:avx512` and are not claimed to be
# good, only to run without spilling by the budget's own reckoning.
function _complex_fitted_shape(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    R = real(T)
    vb = profile.vector_bytes
    lanes = (vb > 0 && vb % sizeof(R) == 0) ? vb ÷ sizeof(R) : _default_lanewidth(R)
    budget = profile.nregisters > 0 ? profile.nregisters : 16
    best = nothing
    for shape in kernel_shapes(T, PlanarMethod())
        MR, NR, W = shape
        (W <= lanes && MR % W == 0) || continue
        _planar_pressure(MR, NR, W) <= budget || continue
        # Largest logical tile wins; ties by the wider vector.
        if best === nothing || (MR * NR, W) > (best[1] * best[2], best[3])
            best = shape
        end
    end
    # Unreachable for any budget >= 16 at any lane width the package compiles:
    # each menu carries an `MV = 1` entry per width, whose pressure is
    # `2*NR + 4 = 16` at `NR = 6`. Kept as a total fallback rather than an
    # assertion because returning a correct-but-slow kernel is always better
    # than throwing here -- this is the code path that CI proved must not
    # refuse.
    return best === nothing ? last(kernel_shapes(T, PlanarMethod())) : best
end

# The unconditional default; `OneMMethod` is selected only by naming the kernel
# (docs/decisions.md, "Method ranking does not transfer between machines").
_default_complex_method(::Type{<:Complex}) = PlanarMethod()

@noinline function _kernel_for(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    return _complex_kernel_from_shape(
        _derived_shape(profile, T), T, _default_complex_method(T)
    )
end

# Extent-aware variant, used only when the caller did not name a kernel: a
# contraction whose M extent cannot fill one register tile pads every
# micro-tile away, so fall back to the legacy shape. Keys on padding waste -- a
# countable quantity known at plan time -- not on a cache estimate, which is
# what distinguishes it from the refuted depth-adaptive MC.
@noinline function _default_kernel(::Type{T}, Qm::Int, Qn::Int) where {T}
    kernel = _kernel_for(target_profile(), T)
    (Qm > 0 && Qm < mr(kernel)) || return kernel
    legacy = _legacy_shape(T)
    return SIMDKernel(Val(legacy[1]), Val(legacy[2]), T, Val(legacy[3]))
end

# Structurally identical to the real method above, so the demotion rule has one
# definition in two places rather than two rules; it routes through the same
# seam.
@noinline function _default_kernel(::Type{T}, Qm::Int, Qn::Int) where {T <: Complex}
    profile = target_profile()
    kernel = _kernel_for(profile, T)
    (Qm > 0 && Qm < mr(kernel)) || return kernel
    # Demote to the budget-fitted shape, NOT to `_legacy_shape`: the latter is
    # `(8, 6, W)` at pressure 30, which is over AVX2's 16 ymm. The fitted shape
    # is the smallest thing guaranteed to be constructible on this host.
    small = _complex_fitted_shape(profile, T)
    return _complex_kernel_from_shape(small, T, _default_complex_method(T))
end

# GUARDRAIL, load-bearing (docs/decisions.md, macro-blocking Phase A
# findings): `_axis_of` returns a `Union{AffineAxis,PtrScatterAxis}`, and each
# consumer below is a `where {R<:Axis, C<:Axis}` barrier method that Julia
# specializes per concrete (R,C), so no partially-applied -- heap-boxed --
# `QSTile` is ever built. **Do not** collapse these helpers into their call
# sites, and do not let a union cross any other boundary.
# Both arms are isbits, so this Union itself needs no heap box (see
# PtrScatterAxis).
@inline function _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    return d.regular ? AffineAxis(d.base, d.stride, d.count) :
        PtrScatterAxis(pointer(buffer, first + 1), d.count)
end

# `pack!` is pack_a! or pack_b! (a plain function, specialized on, never a
# closure); A and B differ only in which of rows/cols is the k axis, which the
# caller has already resolved. `transform` is the plan's per-operand
# `identity`/`conj` singleton.
#
# GUARDRAIL: every argument here has its OWN bound type parameter, `transform`
# included. Leaving `TF` unbound reintroduces the Phase 2b finding-5 ~80 B/call
# dynamic dispatch, for the reason spelled out at `pack_a!` in src/kernel.jl.
# And all THREE call sites -- `_execute_nest!`'s two and `execute_tilewise!`'s
# one -- must pass the matching operand's transform: missing the third makes
# the in-tree ORACLE silently wrong for conjugated inputs.
@inline function _pack_sliver!(
        pack!::PF, packed::PK, storage::S, base::Int,
        rows::R, cols::C, kernel, transform::TF
    ) where {PF, PK, S, R <: Axis, C <: Axis, TF}
    pack!(packed, SourceTile(storage, base, rows, cols), kernel, transform)
    return nothing
end

@inline function _execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, kc_len::Int, alpha, beta
    ) where {PA, PB, S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    execute_tile!(kernel, destination, packed_a, packed_b, kc_len, alpha, beta)
    return nothing
end

@inline function _scale_micro_tile!(
        storage::S, base::Int, rows::R, cols::C, beta
    ) where {S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    scale_tile!(destination, beta)
    return nothing
end

# Sliver `s`'s region within a shared packed panel, at the CURRENT block's
# depth `kc_len` (< the buffer's per-sliver capacity on a tail K block, since
# the buffer is sized for kc_eff). Used by both the packing and the consuming
# step of a (jc,pc,ic) iteration, so the two cannot disagree.
@inline function _sliver_range(reg_tile::Int, kc_len::Int, s::Int)
    stride = reg_tile * kc_len
    lo = s * stride + 1
    return lo:(lo + stride - 1)
end

# Same addressing as `_sliver_range`, as a borrowed pointer. Keep in step.
#
# GUARDRAIL: `reg_tile` here is a count of REALS per logical K step --
# `packed_a_per_k`/`packed_b_per_k`, NOT `mr`/`nr`. They coincide for every
# real kernel (pinned in test/test_target.jl), so this is the identity on the
# real path; for a complex kernel only the packed count addresses the panel
# correctly.
@inline function _sliver_panel(buffer, reg_tile::Int, kc_len::Int, s::Int)
    stride = reg_tile * kc_len
    return packed_panel(buffer, s * stride + 1, stride)
end

# Classify each register sliver of a just-filled macro block. Shared by the
# N side (jc: B/C) and the M side (ic: A/C), which are structurally identical.
@inline function _classify_slivers!(
        desc1::Vector{BlockDescriptor}, desc2::Vector{BlockDescriptor},
        buf1::Vector{Int}, buf2::Vector{Int},
        blocklen::Int, reg_tile::Int, nslivers::Int
    )
    for s in 0:(nslivers - 1)
        sfirst = s * reg_tile
        scount = min(reg_tile, blocklen - sfirst)
        desc1[s + 1] = describe_block(buf1, sfirst, scount)
        desc2[s + 1] = describe_block(buf2, sfirst, scount)
    end
    return nothing
end

# Apply beta once to every element of C at MR x NR granularity, without
# reading A or B. Shared by both drivers' Qk==0/alpha==0 short-circuit; uses
# the tw_* (MR/NR-sized) buffers, since a beta-only pass needs no blocking --
# which is why those four, unlike the kc-sized tw_k_buf_*/tw_packed_* ones, are
# allocated even under `oracle = false`.
function _scale_all_of_C!(plan, betaT::T, MRk::Int, NRk::Int, Qm::Int, Qn::Int) where {T}
    ws = plan.workspace
    m_bufs = (ws.tw_m_buf_A, ws.tw_m_buf_C)
    n_bufs = (ws.tw_n_buf_B, ws.tw_n_buf_C)
    mfirst = 0
    while mfirst < Qm
        mcount = min(MRk, Qm - mfirst)
        (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
        rowsC = _axis_of(dM_C, ws.tw_m_buf_C, 0)
        nfirst = 0
        while nfirst < Qn
            ncount = min(NRk, Qn - nfirst)
            (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            colsC = _axis_of(dN_C, ws.tw_n_buf_C, 0)
            _scale_micro_tile!(plan.Cstorage, plan.Cbase, rowsC, colsC, betaT)
            nfirst += ncount
        end
        mfirst += mcount
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Conjugation: folding `conjA`/`conjB` with each view's `.op`
# ----------------------------------------------------------------------------
# These live in the ENGINE, not the adapter, because the invariant is an engine
# invariant: the engine never indexes through a `StridedView` (`_plan_contract`
# takes `parent`/`offset`), so a view's `.op` is silently dropped on all three
# operands unless folded in here -- and a caller reaching `plan_contract`
# directly, with no adapter in sight, is exposed to the same silent wrongness.
# Semantics and rejected alternatives: docs/decisions.md, "Conjugation:
# semantics, and where each piece is absorbed".
#
# GUARDRAIL: a TOTAL table with a throwing fallback, NOT TensorOperations'
# TBLIS extension's `A.op === conj` test. `StridedView(p, sz, st, off,
# adjoint)` is directly constructible, and `=== conj` classifies it as
# *unconjugated* -- the silent-wrong-answer class this milestone exists to
# close. The fallback is `@noinline` and unreachable for every `op`
# `StridedViews` itself constructs, so totality costs nothing.
_op_conjugates(::typeof(identity)) = false
_op_conjugates(::typeof(conj)) = true
# Elementwise identity on a `Number`: these permute axes, they do not touch
# values, and the engine has already resolved axes into `AxisGroup`s.
_op_conjugates(::typeof(transpose)) = false
_op_conjugates(::typeof(adjoint)) = true
@noinline _op_conjugates(f) = throw(
    ArgumentError(
        "unsupported StridedView.op $f: QuasiStrided folds a view's `op` into the " *
            "packing transform and recognizes only identity/conj/transpose/adjoint"
    )
)

# GUARDRAIL: `⊻`, not `||`. The flag and the view's `op` are two INDEPENDENT
# requests to conjugate the same data, and `conj` is involutive, so applying
# both is the identity and only the parity survives. `false` unconditionally
# for a real element type -- `StridedViews` defines
# `conj(::StridedView{<:Real}) = a`, so a real view's `op` can never conjugate
# anyway, and the real path therefore always gets `identity` and no new
# `execute!` specialization, even with `conjA = true`.
_qs_isconj(v::StridedView{T}, flag::Bool) where {T} =
    (T <: Complex) && (flag ⊻ _op_conjugates(v.op))

"""
    ContractPlan

Reusable plan from [`plan_contract`](@ref): resolved M/N/K `AxisGroup`s,
kernel, operand storage/base, the effective [`Blocking`](@ref), and the
[`ContractWorkspace`](@ref) holding every buffer
[`execute!`](@ref)/[`execute_tilewise!`](@ref) need -- sized once during
planning, never (re)allocated during execution, plus the per-operand packing
transforms `atransform`/`btransform`. `VT` is the workspace's packed-panel
vector type (`Vector{real(T)}` on the default allocator path, so `Vector{T}`
on the real path), a `where`-bound parameter resolved at construction, so
every plan instance is concretely typed. Field layout is an implementation
detail, not part of the frozen interface.
"""
struct ContractPlan{
        T, Kern, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC,
        TA, TB, VT <: AbstractVector,
    }
    kernel::Kern
    mgroup::GM
    ngroup::GN
    kgroup::GK
    blocking::Blocking
    Astorage::SA
    Abase::Int
    Bstorage::SB
    Bbase::Int
    Cstorage::SC
    Cbase::Int

    # `identity` or `conj`, as singleton function VALUES with their own type
    # parameters. GUARDRAIL: not a `Bool` field and not a `Val{Bool}` -- either
    # would cross `_pack_sliver!` as a `Union` or need mapping to a function at
    # the pack site, i.e. Phase 2b finding 5 and its ~80 B/call.
    atransform::TA
    btransform::TB

    # Every buffer both drivers use (docs/decisions.md, "Amendment 1").
    workspace::ContractWorkspace{T, VT}
end

"""
    plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                  B::StridedView, indB::NTuple{NB,Int},
                  indC::NTuple{NC,Int};
                  kernel = nothing,
                  conjA = false, conjB = false,
                  mc = nothing, kc = nothing, nc = nothing,
                  workspace = nothing,
                  allocator = TensorOperations.DefaultAllocator(),
                  oracle = true) -> ContractPlan

`kernel = nothing` (the default) resolves the kernel *after* the M/N/K groups
are built, via `_default_kernel(T, Qm, Qn)`, because the extent-aware demotion
needs `Qm`. For a real element type that is a [`SIMDKernel`](@ref) at the
hardware-derived shape; for a complex one a [`PlanarKernel`](@ref) at the
swept shape (and on a vector ISA with no complex measurement, an
`ArgumentError` rather than a guaranteed-spilling default -- pass `kernel`
explicitly to override). [`OneMKernel`](@ref) is never selected automatically;
naming it is the only way to use 1m.

Planning phase of [`contract!`](@ref): resolves labels into M/N/K
`AxisGroup`s, validates matched axis lengths and eltypes, and preallocates
every buffer [`execute!`](@ref) needs. `mc`/`kc`/`nc` are the macro-blocking
factors (see [`Blocking`](@ref)); a `nothing` keyword takes the corresponding
field of `default_blocking(kernel)`. Each must be `>= 1`, and is then rounded
and clamped into the *effective* blocking stored on the plan: `mc`/`nc` round
up to a whole `mr(kernel)`/`nr(kernel)` multiple, then cap at the M/N extent
(likewise rounded up); `kc` caps at the K extent. Throws
`ArgumentError`/`DimensionMismatch` on invalid input.

Buffers (docs/decisions.md, "Amendment 1"):

  * `workspace = nothing` builds a fresh [`ContractWorkspace`](@ref); passing
    an existing one reuses it, grown as needed by [`reserve!`](@ref), even
    across differently shaped contractions.
  * `allocator` is a TensorOperations allocator. `DefaultAllocator` gives a
    plain, GC-owned, `reserve!`-able `ContractWorkspace{T,Vector{T}}`; any
    other one sizes the packed panels exactly once via
    `TensorOperations.tensoralloc`, forbids `workspace` as well, and leaves
    [`release!`](@ref) to the caller.
  * `oracle = false` skips `execute_tilewise!`'s own buffers entirely, making
    that oracle unavailable for this plan. `execute!` is unaffected.

Conjugation: `conjA`/`conjB` request `conj` on A's/B's elements, TensorOperations'
semantics (`alpha`/`beta` are never conjugated). Each flag is folded here with
the corresponding view's `op` -- they compose with XOR, so a `conj`-wrapped view
with `conjA = true` is unconjugated -- and the result is stored on the plan as a
singleton transform applied during packing. A conjugated *output* view is
rejected: the engine addresses `parent(C)` directly and would silently ignore
it. For a real element type both transforms are `identity` no matter what the
flags say, so the real path gains no specialization.
"""
function plan_contract(
        C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int};
        kernel = nothing,
        conjA::Bool = false,
        conjB::Bool = false,
        mc::Union{Int, Nothing} = nothing,
        kc::Union{Int, Nothing} = nothing,
        nc::Union{Int, Nothing} = nothing,
        workspace::Union{Nothing, ContractWorkspace} = nothing,
        allocator = TO.DefaultAllocator(),
        oracle::Bool = true
    ) where {NA, NB, NC}
    T = eltype(C)
    eltype(A) === T ||
        throw(ArgumentError("eltype(A) = $(eltype(A)) does not match eltype(C) = $T"))
    eltype(B) === T ||
        throw(ArgumentError("eltype(B) = $(eltype(B)) does not match eltype(C) = $T"))

    # Fold each flag with its view's `op`. GUARDRAIL: a conjugated `C` is
    # REJECTED, not supported -- there is nowhere to absorb its `op` (the
    # engine writes through to the parent), so it would be silently wrong;
    # supporting it would also thread a flag through `store_tile!` and force
    # re-deriving the beta-applied-once argument (docs/decisions.md, "A
    # conjugated output `C` is rejected this milestone"). The two transforms
    # are `Union{typeof(identity),typeof(conj)}` here and die at the
    # `_plan_contract` barrier below, as `_default_kernel`'s Union already does.
    _qs_isconj(C, false) && throw(
        ArgumentError(
            "plan_contract: cannot write into a conjugated view (C has op $(C.op)); " *
                "writing a conjugated output is not supported"
        )
    )
    atransform = _qs_isconj(A, conjA) ? conj : identity
    btransform = _qs_isconj(B, conjB) ? conj : identity

    mlabels, nlabels, klabels = _classify_labels(indA, indB, indC)

    mgroup = _build_pair_group(mlabels, indA, A, indC, C)  # maps: (A, C)
    ngroup = _build_pair_group(nlabels, indB, B, indC, C)  # maps: (B, C)
    kgroup = _build_pair_group(klabels, indA, A, indB, B)  # maps: (A, B)

    Qm = axis_length(mgroup)
    Qn = axis_length(ngroup)
    Qk = axis_length(kgroup)

    # Resolved here, not in the signature default: the demotion needs Qm.
    # Nothing between the eltype checks and here reads `kernel`.
    resolved = kernel === nothing ? _default_kernel(T, Qm, Qn) : kernel
    return _plan_contract(
        C, A, B, indC, mgroup, ngroup, kgroup, Qm, Qn, Qk,
        resolved, atransform, btransform, mc, kc, nc, workspace, allocator, oracle
    )
end

# Function barrier: the small Unions from `_default_kernel` and from the
# `conj`/`identity` transforms die here, so `ContractPlan`'s `Kern`, `TA` and
# `TB` are concrete and `execute!` sees no abstract type. `TA`/`TB` each get
# their own bound parameter for the same reason `K` does.
function _plan_contract(
        C::StridedView, A::StridedView, B::StridedView, indC::NTuple{NC, Int},
        mgroup, ngroup, kgroup, Qm::Int, Qn::Int, Qk::Int,
        kernel::K, atransform::TA, btransform::TB,
        mc::Union{Int, Nothing}, kc::Union{Int, Nothing},
        nc::Union{Int, Nothing}, workspace::Union{Nothing, ContractWorkspace},
        allocator, oracle::Bool
    ) where {NC, K, TA, TB}
    T = eltype(C)
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match eltype(C) = $T"))

    defaults = default_blocking(kernel)
    # Blocking's own constructor validates all three >= 1.
    requested = Blocking(
        mc === nothing ? defaults.mc : mc,
        kc === nothing ? defaults.kc : kc,
        nc === nothing ? defaults.nc : nc
    )

    MRk = mr(kernel)
    NRk = nr(kernel)

    mc_rounded = _roundup(requested.mc, MRk)
    nc_rounded = _roundup(requested.nc, NRk)

    # On an empty extent both drivers short-circuit before reading these, so
    # the floor here only keeps Blocking's >=1 invariant and the buffers
    # well-formed.
    mc_eff = Qm == 0 ? MRk : min(mc_rounded, _roundup(Qm, MRk))
    nc_eff = Qn == 0 ? NRk : min(nc_rounded, _roundup(Qn, NRk))
    kc_eff = Qk == 0 ? 1 : min(requested.kc, Qk)

    blocking = Blocking(mc_eff, kc_eff, nc_eff)

    Astorage = parent(A)
    Abase = offset(A)
    Bstorage = parent(B)
    Bbase = offset(B)
    Cstorage = parent(C)
    Cbase = offset(C)

    # Buffers are `undef`-initialized, not zeroed; see `ContractWorkspace`.
    ws = _resolve_workspace(T, workspace, kernel, blocking, oracle, allocator)

    return ContractPlan(
        kernel, mgroup, ngroup, kgroup, blocking,
        Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
        atransform, btransform, ws
    )
end

# Build or reuse the plan's workspace. Dispatching on the allocator type (not
# an `isa` branch on a value) keeps both paths concretely typed and makes the
# unreachable one disappear at compile time. Default path: a plain, GC-owned
# workspace, reused via `reserve!` when one is handed in -- the
# zero-steady-state-allocation fast path (docs/decisions.md, Amendment 1).
function _resolve_workspace(
        ::Type{T}, workspace, kernel, blocking::Blocking, oracle::Bool,
        allocator::TO.DefaultAllocator
    ) where {T}
    workspace === nothing &&
        return ContractWorkspace(T, kernel, blocking, oracle, allocator)
    return _reuse_workspace(T, workspace, kernel, blocking, oracle)
end

# Explicit-allocator path: sized exactly once from the effective blocking, no
# `reserve!`, no resizing. The caller owns `release!`.
function _resolve_workspace(
        ::Type{T}, workspace, kernel, blocking::Blocking, oracle::Bool, allocator
    ) where {T}
    workspace === nothing || throw(
        ArgumentError(
            "plan_contract: `workspace` cannot be combined with a non-default " *
                "`allocator` ($(typeof(allocator))); an allocator-provided workspace is " *
                "sized once at construction and must not be resized or reused"
        )
    )
    return ContractWorkspace(T, kernel, blocking, oracle, allocator)
end

# `R` is the PACKED element type, `real(T)` by `ContractWorkspace`'s own
# invariant. The extra guard stops a pooled workspace of one precision serving
# a plan of another: the pool is keyed by `eltype(C)` alone, so `T` matching is
# not enough once the packed type is a separate notion. Both sides are
# compile-time constants, so this folds away entirely.
@inline function _reuse_workspace(
        ::Type{T}, ws::ContractWorkspace{T, Vector{R}}, kernel, blocking::Blocking,
        oracle::Bool
    ) where {T, R}
    R === realtype(kernel) || _throw_packed_eltype_mismatch(T, ws, kernel)
    return reserve!(ws, kernel, blocking, oracle)
end

@noinline function _throw_packed_eltype_mismatch(::Type{T}, ws, kernel) where {T}
    throw(
        ArgumentError(
            "plan_contract: cannot reuse a $(typeof(ws)) whose packed panels hold " *
                "$(eltype(ws.packed_a)) for a kernel packing $(realtype(kernel))"
        )
    )
end

@noinline function _reuse_workspace(
        ::Type{T}, ws::ContractWorkspace, kernel, blocking::Blocking, oracle::Bool
    ) where {T}
    throw(
        ArgumentError(
            "plan_contract: cannot reuse a $(typeof(ws)) for an eltype-$T contraction " *
                "on the default allocator; only a " *
                "ContractWorkspace{$T,Vector{$(realtype(kernel))}} -- storage element " *
                "type $T, packed panels of $(realtype(kernel)) -- is `reserve!`-able"
        )
    )
end

"""
    execute!(plan::ContractPlan, alpha::Number, beta::Number)

Execution phase of [`contract!`](@ref): a BLIS five-loop macro-blocking nest
over `plan.blocking` (`nc`/loop 5, `kc`/loop 4, `mc`/loop 3), packing the
whole B panel once per `(jc,pc)` and the whole A panel once per `(jc,pc,ic)`,
then running `execute_tile!` over every micro-tile of that block (loops 2/1).
`beta` applies exactly once per output element (on the first K block only;
later ones accumulate with `beta = one(T)`). Empty output is a no-op; empty K
or `alpha == 0` applies `beta` once without reading `A`/`B`. Allocation-free.
Returns `plan.Cstorage`. See [`execute_tilewise!`](@ref) for the independent
tile-by-tile oracle this is checked against.
"""
function execute!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    kernel = plan.kernel
    MRk = mr(kernel)
    NRk = nr(kernel)

    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    Qk = axis_length(plan.kgroup)

    # Empty output: no-op, nothing read or written at all.
    (Qm == 0 || Qn == 0) && return plan.Cstorage

    # Nothing to contract: beta-only pass, A and B never read.
    if Qk == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, MRk, NRk, Qm, Qn)
        return plan.Cstorage
    end

    mc_eff = plan.blocking.mc
    kc_eff = plan.blocking.kc
    nc_eff = plan.blocking.nc

    # A reused workspace may be oversized: every extent below comes from the
    # *current* block, never from a buffer's length.
    ws = plan.workspace

    # Panels below borrow pointers into ws.packed_a/_b (src/panel.jl), as do
    # the PtrScatterAxes from `_axis_of`; this is their lifetime.
    return GC.@preserve ws begin
        _execute_nest!(
            plan, ws, kernel, MRk, NRk, Qm, Qn, Qk,
            mc_eff, kc_eff, nc_eff, alphaT, betaT
        )
    end
end

# Split out so the `GC.@preserve` above has one obvious scope.
function _execute_nest!(
        plan::ContractPlan{T}, ws, kernel::K, MRk::Int, NRk::Int,
        Qm::Int, Qn::Int, Qk::Int, mc_eff::Int, kc_eff::Int, nc_eff::Int,
        alphaT::T, betaT::T
    ) where {T, K}
    # GUARDRAIL: reals per sliver per LOGICAL K step, which is what addresses
    # the packed panels. NOT interchangeable with `MRk`/`NRk`, which keep their
    # meaning everywhere else here (sliver counts, block extents,
    # `_classify_slivers!`): one counts register-tile rows, the other reals.
    # `MRp === MRk` for every real kernel (pinned in test/test_target.jl), so
    # the substitution below is provably the identity on the real path.
    MRp = packed_a_per_k(kernel)
    NRp = packed_b_per_k(kernel)

    atransform = plan.atransform
    btransform = plan.btransform

    # --- loop 5: jc over N in steps of nc_eff ---
    jc = 0
    while jc < Qn
        nblock = min(nc_eff, Qn - jc)
        n_slivers = cld(nblock, NRk)
        fill_offsets!((ws.n_buf_B, ws.n_buf_C), plan.ngroup, jc, nblock)
        _classify_slivers!(
            ws.n_desc_B, ws.n_desc_C, ws.n_buf_B, ws.n_buf_C,
            nblock, NRk, n_slivers
        )

        # --- loop 4: pc over K in steps of kc_eff ---
        pc = 0
        firstpanel = true
        while pc < Qk
            kblock = min(kc_eff, Qk - pc)
            fill_offsets!((ws.k_buf_A, ws.k_buf_B), plan.kgroup, pc, kblock)
            dK_A = describe_block(ws.k_buf_A, 0, kblock)
            dK_B = describe_block(ws.k_buf_B, 0, kblock)
            colsA_k = _axis_of(dK_A, ws.k_buf_A, 0)
            rowsB_k = _axis_of(dK_B, ws.k_buf_B, 0)

            beta_eff = firstpanel ? betaT : one(T)

            # Pack the whole B panel for this (jc, pc): every N-sliver.
            for s in 0:(n_slivers - 1)
                sfirst = s * NRk
                colsB = _axis_of(ws.n_desc_B[s + 1], ws.n_buf_B, sfirst)
                bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                _pack_sliver!(
                    pack_b!, bpanel, plan.Bstorage, plan.Bbase, rowsB_k, colsB,
                    kernel, btransform
                )
            end

            # --- loop 3: ic over M in steps of mc_eff ---
            ic = 0
            while ic < Qm
                mblock = min(mc_eff, Qm - ic)
                m_slivers = cld(mblock, MRk)
                fill_offsets!((ws.m_buf_A, ws.m_buf_C), plan.mgroup, ic, mblock)
                _classify_slivers!(
                    ws.m_desc_A, ws.m_desc_C, ws.m_buf_A, ws.m_buf_C,
                    mblock, MRk, m_slivers
                )

                # Pack the whole A panel for this (jc, pc, ic): every M-sliver.
                for r in 0:(m_slivers - 1)
                    rfirst = r * MRk
                    rowsA = _axis_of(ws.m_desc_A[r + 1], ws.m_buf_A, rfirst)
                    apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                    _pack_sliver!(
                        pack_a!, apanel, plan.Astorage, plan.Abase, rowsA, colsA_k,
                        kernel, atransform
                    )
                end

                # --- loop 2: jr over N-slivers; loop 1: ir over M-slivers ---
                for s in 0:(n_slivers - 1)
                    sfirst = s * NRk
                    colsC = _axis_of(ws.n_desc_C[s + 1], ws.n_buf_C, sfirst)
                    bpanel = _sliver_panel(ws.packed_b, NRp, kblock, s)
                    for r in 0:(m_slivers - 1)
                        rfirst = r * MRk
                        rowsC = _axis_of(ws.m_desc_C[r + 1], ws.m_buf_C, rfirst)
                        apanel = _sliver_panel(ws.packed_a, MRp, kblock, r)
                        _execute_micro_tile!(
                            kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                            apanel, bpanel, kblock, alphaT, beta_eff
                        )
                    end
                end

                ic += mblock
            end

            firstpanel = false
            pc += kblock
        end

        jc += nblock
    end

    return plan.Cstorage
end

"""
    execute_tilewise!(plan::ContractPlan, alpha::Number, beta::Number)

Unexported. The pre-macro-blocking driver, kept as the independent
correctness oracle for [`execute!`](@ref) (docs/decisions.md,
"Macro-blocking milestone", item 5): tiles M/N in steps of
`mr(plan.kernel)`/`nr(plan.kernel)` and, per output tile, K in
`plan.blocking.kc`-sized panels, packing one sliver per tile and calling
`execute_tile!` with `beta` on the first panel and `one(T)` on later ones.
Uses only its own `tw_*` buffers, so it shares no mutable state with
`execute!`. Allocation-free.

Requires a plan built with `oracle = true` (the default); throws
`ArgumentError` otherwise, since `oracle = false` is exactly the request not
to allocate these buffers.
"""
function execute_tilewise!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    ws = plan.workspace
    _has_oracle(ws) || throw(
        ArgumentError(
            "execute_tilewise! needs the oracle buffers, which this plan was built " *
                "without; re-plan with `oracle = true`"
        )
    )

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    kernel = plan.kernel
    MRk = mr(kernel)
    NRk = nr(kernel)

    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    Qk = axis_length(plan.kgroup)

    (Qm == 0 || Qn == 0) && return plan.Cstorage

    if Qk == 0 || iszero(alphaT)
        _scale_all_of_C!(plan, betaT, MRk, NRk, Qm, Qn)
        return plan.Cstorage
    end

    m_bufs = (ws.tw_m_buf_A, ws.tw_m_buf_C)
    n_bufs = (ws.tw_n_buf_B, ws.tw_n_buf_C)
    k_bufs = (ws.tw_k_buf_A, ws.tw_k_buf_B)
    kc_panel = plan.blocking.kc

    mfirst = 0
    while mfirst < Qm
        mcount = min(MRk, Qm - mfirst)
        (dM_A, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
        rowsA = _axis_of(dM_A, ws.tw_m_buf_A, 0)
        rowsC = _axis_of(dM_C, ws.tw_m_buf_C, 0)

        nfirst = 0
        while nfirst < Qn
            ncount = min(NRk, Qn - nfirst)
            (dN_B, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            colsB = _axis_of(dN_B, ws.tw_n_buf_B, 0)
            colsC = _axis_of(dN_C, ws.tw_n_buf_C, 0)

            kfirst = 0
            firstpanel = true
            while kfirst < Qk
                kcount = min(kc_panel, Qk - kfirst)
                (dK_A, dK_B) = block_descriptors!(k_bufs, plan.kgroup, kfirst, kcount)
                colsK_A = _axis_of(dK_A, ws.tw_k_buf_A, 0)
                rowsK_B = _axis_of(dK_B, ws.tw_k_buf_B, 0)

                # The third `_pack_sliver!` call site: the oracle must apply
                # the same transforms as `execute!` or it silently disagrees on
                # conjugated inputs. No `_sliver_panel` here -- these are whole
                # buffers sized by `packed_a_length`/`packed_b_length`, which
                # already count reals, so no `MRp`/`NRp` treatment is needed.
                _pack_sliver!(
                    pack_a!, ws.tw_packed_a, plan.Astorage, plan.Abase, rowsA, colsK_A,
                    kernel, plan.atransform
                )
                _pack_sliver!(
                    pack_b!, ws.tw_packed_b, plan.Bstorage, plan.Bbase, rowsK_B, colsB,
                    kernel, plan.btransform
                )

                beta_eff = firstpanel ? betaT : one(T)
                _execute_micro_tile!(
                    kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                    ws.tw_packed_a, ws.tw_packed_b, kcount, alphaT, beta_eff
                )

                firstpanel = false
                kfirst += kcount
            end

            nfirst += ncount
        end
        mfirst += mcount
    end

    return plan.Cstorage
end

# ----------------------------------------------------------------------------
# contract!
# ----------------------------------------------------------------------------

"""
    contract!(C::StridedView, alpha::Number,
              A::StridedView, indA::NTuple{NA,Int},
              B::StridedView, indB::NTuple{NB,Int},
              beta::Number,
              indC::NTuple{NC,Int}) where {NA,NB,NC}

Compute `C[indC] = alpha * sum_K A[indA] * B[indB] + beta * C[indC]`; see
`docs/decisions.md` for label semantics. Equivalent to
`execute!(plan_contract(C, A, indA, B, indB, indC), alpha, beta)` — use
those directly to reuse a plan across calls. Returns `C`.
"""
function contract!(
        C::StridedView, alpha::Number,
        A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        beta::Number,
        indC::NTuple{NC, Int}
    ) where {NA, NB, NC}
    plan = plan_contract(C, A, indA, B, indB, indC)
    execute!(plan, alpha, beta)
    return C
end
