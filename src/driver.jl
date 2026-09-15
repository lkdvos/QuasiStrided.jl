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

# Engine-wide default kernel (docs/decisions.md: "Amendment 2" for
# `SIMDKernel`, Phase G for the derived shape, Phase H for why NV stays 12).
#
# Shape from one detected capability, the vector register width:
#   W = vector_bytes/sizeof(T),  MR = 2W,  NR = NR_DEFAULT  =>  NV = 12.
# Reproduces the swept optimum for both dtypes on AVX-512 and reduces to the
# previous hardcoded (8,6,4) on AVX2. NV = 12 is also the only setting that
# fits a 16-register AVX2 machine and Julia 1.10's register allocation.
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

# For COMPLEX types the rule is overridden, and this is the one place in the
# package where a swept row beats the derived rule by enough to be worth the
# machine-specificity. Measured on ccqlin038 (Cascade Lake, `:avx512`, Julia
# 1.12.6), 21 reps, canary spread 0.4%, `benchmark/bench_complex_efficiency.jl`
# arm 2, ranked by geomean of per-shape time normalised to the best at that
# shape:
#
#   ComplexF64          ComplexF32
#   planar 24x3  1.055  planar 48x3  1.104   <- overridden to these
#   1m     16x6  1.116  1m     16x8  1.170
#   1m     12x8  1.179  1m     24x8  1.172
#   1m      8x8  1.224  1m     32x6  1.202
#   planar 16x6  1.452  planar 16x8  1.311
#   planar  8x8  1.502  planar 32x6  1.562   <- what the rule derives
#
# The derived `MR = 2W, NR = 6` shape is the *worst* planar configuration
# measured, by 38% (`ComplexF64`) and 41% (`ComplexF32`) against the best. That
# is not noise at a 0.4% canary spread, and it has a mechanism: Phase C
# measured the derived shape spilling 24-26 accumulator stores per K step while
# `24x3`/`48x3` is spill-free (docs/decisions.md, "Phase C integration
# findings" and its Phase D correction). The spill analysis predicted the
# ranking before the ranking was measured.
#
# Why an override rather than changing the rule: the rule is shared with the
# real path, where it is the measured optimum and must not move. Why a row here
# is acceptable when the real path deliberately has none: the real rule was
# within noise of its sweep's best, so a row would have encoded noise; this one
# corrects a 38-41% error. The cost is honest and stated -- these two rows are
# ccqlin038 measurements, and on any other microarchitecture the engine falls
# back to the derived rule via `_rule_applies_complex`, which is `:avx512`-only.
_shape_override(::Val{:avx512}, ::Type{ComplexF64}) = (24, 3, 8)
_shape_override(::Val{:avx512}, ::Type{ComplexF32}) = (48, 3, 16)

# The rule applies only to the ISAs it was validated on. `:neon` is detected
# but deliberately gets the legacy shape: there is no aarch64 measurement, and
# the rule would pick MR = 2W = 4 with 128-bit lanes, using 12 of 32 NEON
# registers -- narrower and smaller than the legacy (8,6,4), not obviously
# better. Derive where measured, fall back everywhere else.
_rule_applies(::Val{:avx512}) = true
_rule_applies(::Val{:avx2}) = true
_rule_applies(::Val) = false

# Complex counterpart, `:avx512` only. Same treatment, and for the same kind of
# reason, as the `:neon` fallback above: AVX2 has 16 ymm registers, and planar
# holds separate real and imaginary accumulator planes, so even (MV,NR) = (1,6)
# leaves exactly zero spare there (docs/decisions.md, "Cliff A"); the
# reference's AVX2 complex shapes are explicitly marked unmeasured. Derive
# where measured, fall back everywhere else.
_rule_applies_complex(::Val{:avx512}) = true
_rule_applies_complex(::Val) = false

function _derived_shape(profile::TargetProfile, ::Type{T}) where {T}
    vb = profile.vector_bytes
    key = Val(profile.isa)
    (_rule_applies(key) && vb > 0 && vb % sizeof(T) == 0) || return _legacy_shape(T)
    ovr = _shape_override(key, T)
    return ovr === nothing ? (2 * (vb ÷ sizeof(T)), NR_DEFAULT, vb ÷ sizeof(T)) : ovr
end

# A NEW METHOD, not an edit of the one above: "the real path is bit-identical"
# is then a `git diff` fact rather than an argument about whether
# `real(Float64) === Float64`.
#
# The one-line Phase G rule survives the complex extension unchanged; only the
# `sizeof` argument moves to the real type, because `W` is a count of real
# lanes. `MR = 2W, NR = NR_DEFAULT` with `W = vector_bytes / sizeof(real(T))`
# gives `(16, 6, 8)` for `ComplexF64` on AVX-512 -- a 16x6 complex tile, which
# is exactly the reference's measured planar menu head `(MV, NR) = (2, 6)`.
# `_shape_override` is consulted here symmetrically with the real path, but --
# unlike the real path -- it is NOT empty: Phase F measured the derived
# `(16, 6, 8)` / `(32, 6, 16)` shape to be the worst planar configuration by
# 38-41%, and the override carries the swept winner. See `_shape_override`.
# The rule, factored out of `_derived_shape` so that it can be tested
# independently of the override layered on top of it. `W` is a count of REAL
# lanes, which is the whole complex adaptation.
_complex_rule_shape(vb::Int, ::Type{T}) where {T <: Complex} =
    (2 * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

function _derived_shape(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    R = real(T)
    vb = profile.vector_bytes
    key = Val(profile.isa)
    (_rule_applies_complex(key) && vb > 0 && vb % sizeof(R) == 0) ||
        return _legacy_shape(T)
    ovr = _shape_override(key, T)
    return ovr === nothing ? _complex_rule_shape(vb, T) : ovr
end

# Closed set, so compiled SIMDKernel (and driver) specializations are bounded.
const KERNEL_SHAPES_F64 = ((8, 6, 4), (16, 6, 8))
const KERNEL_SHAPES_F32 = ((8, 6, 8), (32, 6, 16), (16, 6, 8))

# Complex menus, seeded from the reference's measured AVX-512 shapes
# (`crates/tensorcontract/src/kernel/x86.rs:182-190`) converted to this
# project's `(MR, NR, W)` with `MR` in LOGICAL complex rows and `W` in real
# lanes; at most three each, so the compiled specialization set stays bounded.
# The 1m menus look "unaligned" (MR = 12 with W = 8) only because the
# reference's MV counts REAL rows: 1m runs a real microkernel of `2MR` rows, so
# `2*12 = 24` is what must be a multiple of `W`.
# Ordered with the Phase F winner first, so the menu head and the shape the
# engine actually resolves to agree (see `_shape_override`). The set is
# unchanged -- only the order -- so no specialization is added or removed.
const KERNEL_SHAPES_C64_PLANAR = ((24, 3, 8), (16, 6, 8), (8, 8, 8))
const KERNEL_SHAPES_C64_ONEM = ((12, 8, 8), (16, 6, 8), (8, 8, 8))
const KERNEL_SHAPES_C32_PLANAR = ((48, 3, 16), (32, 6, 16), (16, 8, 16))
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
# from literal `Val`s; the last shape is the fallback. Generated because a
# plain loop would construct `Val(cand[1])` dynamically and widen to `Any`.
# Costs one dynamic dispatch per `plan_contract`, none per tile or K step:
# `_plan_contract` specializes, so `execute!` sees no abstract type.
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
# The complex counterpart of `_kernel_from_shape`, generated over
# `kernel_shapes(T, method)` for exactly the same reason: every branch must
# build a concrete kernel from literal `Val`s, because a plain loop would
# construct `Val(cand[1])` dynamically and widen to `Any`.
#
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

# The 1m arm, structurally identical to the planar one above and generated over
# 1m's OWN menu -- each method has its own measured shapes because each has its
# own packed A format and therefore its own register budget. Deliberately a
# second method rather than a shared generic over `M <: ComplexMethod`: keeping
# the planar arm byte-identical is a `git diff` fact rather than an argument.
#
# Reaching this arm requires `_default_complex_method` to return `OneMMethod()`,
# which it never does -- `PlanarMethod()` is the unconditional default and 1m is
# selectable ONLY by naming the kernel (docs/decisions.md, "Method ranking does
# not transfer between machines"). No auto-dispatch, no env var, no shape-driven
# rule. The arm exists so that `_complex_kernel_from_shape(shape, T,
# OneMMethod())` is total for callers -- benchmarks, and a future explicit
# request -- that name the method themselves.
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

# The complex register shapes are budgeted for a 32-register AVX-512 file: a
# planar kernel holds 2*MV*NR accumulators + 2*MV A vectors + 2 B broadcasts,
# which even at the smallest menu entry is over what AVX2's 16 ymm registers
# can hold. `_rule_applies_complex` already refuses to *derive* a shape off
# `:avx512`, but the legacy fallback would still hand one back, so gate kernel
# *construction* too rather than shipping a guaranteed-spilling default. An
# explicitly named `kernel =` still works everywhere -- this governs only what
# the engine picks on its own.
_complex_default_supported(::Val{:avx512}) = true
_complex_default_supported(::Val) = false

@noinline function _complex_unsupported_isa(::Type{T}, profile::TargetProfile) where {T}
    throw(
        ArgumentError(
            "QuasiStrided has no measured complex register shape for the detected " *
                "vector ISA :$(profile.isa) (only :avx512), so it will not choose a " *
                "complex kernel for $T on this machine: every candidate shape needs " *
                "more vector registers than this ISA provides, and a silently " *
                "spilling default would be worse than an error. Pass an explicit " *
                "`kernel = ...` to plan_contract if you want one anyway."
        )
    )
end

# The unconditional default; `OneMMethod` is selected only by naming the kernel
# (docs/decisions.md, "Method ranking does not transfer between machines").
_default_complex_method(::Type{<:Complex}) = PlanarMethod()

@noinline function _kernel_for(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    _complex_default_supported(Val(profile.isa)) || _complex_unsupported_isa(T, profile)
    return _complex_kernel_from_shape(
        _derived_shape(profile, T), T, _default_complex_method(T)
    )
end

# Extent-aware variant, used only when the caller did not name a kernel: a
# contraction whose M extent cannot fill one register tile pads every
# micro-tile away, so fall back to the legacy shape. Keys on padding waste (a
# countable quantity known at plan time), not on a cache estimate -- which is
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
    kernel = _kernel_for(target_profile(), T)
    (Qm > 0 && Qm < mr(kernel)) || return kernel
    legacy = _legacy_shape(T)
    return _complex_kernel_from_shape(legacy, T, _default_complex_method(T))
end

# LOAD-BEARING (docs/decisions.md, macro-blocking Phase A findings): `_axis_of`
# produces a Union{AffineAxis,ScatterAxis}, and each consumer below is a
# `where {R<:Axis, C<:Axis}` barrier method that Julia specializes per concrete
# (R,C), so no partially-applied (boxing) QSTile is ever built. Do not collapse
# these helpers into their call sites, and do not let a union cross any other
# boundary.
# Both arms are isbits, so this Union needs no heap box (see PtrScatterAxis).
@inline function _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    return d.regular ? AffineAxis(d.base, d.stride, d.count) :
        PtrScatterAxis(pointer(buffer, first + 1), d.count)
end

# `pack!` is pack_a! or pack_b! (a plain function, specialized on, never a
# closure); A and B differ only in which of rows/cols is the k axis, which
# the caller has already resolved.
#
# `transform` is the plan's per-operand elementwise transform (`identity` or
# `conj`), a singleton function value. It gets its OWN bound type parameter
# `TF`: leaving it unbound reintroduces the Phase 2b finding-5 allocation, for
# exactly the reason spelled out at src/kernel.jl:30-33. All three call sites
# -- `_execute_nest!`'s two and `execute_tilewise!`'s one -- must pass the
# matching operand's transform; missing the third would make the in-tree
# ORACLE silently wrong for conjugated inputs.
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
# `reg_tile` here is a count of REALS per logical K step --
# `packed_a_per_k(kernel)`/`packed_b_per_k(kernel)`, not `mr`/`nr`. The two
# coincide for every real kernel (pinned by a test in test/test_target.jl), so
# this is the identity on the real path; for a complex kernel the panel holds
# `reals_per_element` planes per element and only the packed count addresses it
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
# the tw_* (MR/NR-sized) buffers, since a beta-only pass needs no blocking.
# That is why those four buffers -- unlike the kc-sized tw_k_buf_*/tw_packed_*
# ones -- are allocated even under `oracle = false`: they are not oracle-only.
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
# These live here, in the engine, rather than in the TensorOperations adapter,
# because the invariant they protect is an engine invariant: the engine never
# goes through `StridedView` indexing at all (`_plan_contract` takes
# `parent`/`offset` and addresses the parent directly), so a view's `.op` is
# silently dropped on all three operands unless it is folded in here. A caller
# who wraps a complex array in a conjugated `StridedView` and calls
# `plan_contract`/`contract!` directly, with no adapter in sight, is exposed to
# exactly the same silent wrongness. `src/tensoroperations.jl` uses these.
#
# A TOTAL table with a throwing fallback, not TensorOperations' TBLIS
# extension's `A.op === conj` test: `StridedView(p, sz, st, off, adjoint)` is
# directly constructible, and `=== conj` would silently treat it as
# unconjugated -- precisely the silent-wrong-answer class this milestone exists
# to close. The fallback is `@noinline` and unreachable for every `op`
# `StridedViews` itself constructs, so it costs nothing.
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

# The two sources of conjugation are independent and compose with XOR: the
# flag conjugates the operand's data, and so does the view's `op`, so applying
# both is the identity. `false` unconditionally for a real element type --
# `StridedViews` defines `conj(::StridedView{<:Real}) = a`, so a real view's
# `op` can never conjugate anyway, and the real path therefore always gets
# `identity` and no new `execute!` specialization, even with `conjA = true`.
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

    # Elementwise transform applied to each source element as it is packed:
    # `identity` or `conj`, as singleton function VALUES with their own type
    # parameters. Not a `Bool` field and not a `Val{Bool}`: either would cross
    # `_pack_sliver!` as a `Union` or need mapping to a function at the pack
    # site, which is Phase 2b finding 5 and its ~80 B/call of dynamic dispatch
    # (docs/decisions.md, "Conjugation: semantics, and where each piece is
    # absorbed").
    atransform::TA
    btransform::TB

    # Every buffer both drivers use (docs/decisions.md, "Amendment 1").
    workspace::ContractWorkspace{T, VT}
end

"""
    plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                  B::StridedView, indB::NTuple{NB,Int},
                  indC::NTuple{NC,Int};
                  kernel = SIMDKernel(Val(8), Val(6), eltype(C)),
                  conjA = false, conjB = false,
                  mc = nothing, kc = nothing, nc = nothing,
                  workspace = nothing,
                  allocator = TensorOperations.DefaultAllocator(),
                  oracle = true) -> ContractPlan

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

    # Fold each flag with its view's `op`. The engine never indexes through a
    # `StridedView`, so `op` would otherwise be dropped on all three operands;
    # for C there is nowhere to absorb it, hence the rejection rather than a
    # transform (docs/decisions.md, "A conjugated output `C` is rejected this
    # milestone"). These are `Union{typeof(identity),typeof(conj)}` here and
    # die at the `_plan_contract` function barrier below, exactly as
    # `_default_kernel`'s Union already does.
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
# unreachable one disappear at compile time.
#
# Default path: a plain, GC-owned `ContractWorkspace{T,Vector{T}}`, reused via
# `reserve!` when one is handed in -- the zero-steady-state-allocation fast
# path (docs/decisions.md's workspace/allocator design-constraints section).
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
# invariant. The extra guard is what stops a pooled workspace of one precision
# serving a plan of another: the backend's pool is keyed by `eltype(C)` alone,
# so `T` matching is not by itself enough once the packed type is a separate
# notion. `R` is a type parameter and `realtype(kernel)` is a compile-time
# constant, so this folds away entirely.
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
    # Reals per sliver per LOGICAL K step, which is what addresses the packed
    # panels. `MRk`/`NRk` keep their existing meaning everywhere else in this
    # function -- sliver counts, block extents, `_classify_slivers!` -- and are
    # NOT interchangeable with these: one counts register-tile rows, the other
    # counts reals. For every real kernel `MRp === MRk` and `NRp === NRk`
    # (pinned in test/test_target.jl), so the substitution below is provably
    # the identity on the real path.
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

                # The third `_pack_sliver!` call site. The oracle must apply the
                # same transforms as `execute!`, or it silently disagrees on
                # conjugated inputs and the disagreement presents as an engine
                # bug. No `_sliver_panel` here: these are whole buffers, sized
                # by `packed_a_length`/`packed_b_length`, which already count
                # reals -- so this driver needs no `MRp`/`NRp` treatment.
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
