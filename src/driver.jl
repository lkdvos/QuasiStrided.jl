# contract!'s frozen signature/label semantics: see docs/decisions.md.
# Planning (label resolution, AxisGroup/buffer construction) is split from
# execution (the tiling loop) so a ContractPlan can be built once and reused:
#   plan_contract(...) -> ContractPlan; execute!(plan, alpha, beta); contract! = both.
#
# execute! implements a BLIS five-loop (NC/KC/MC) macro-blocking nest with
# packed-panel reuse (docs/decisions.md, "Macro-blocking milestone"). The
# pre-macro-blocking tile-by-tile driver is kept, unexported, as
# `execute_tilewise!` -- an independent correctness oracle for the nest below.

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

_default_kernel(::Type{T}) where {T} = ScalarKernel(Val(8), Val(6), T)

# ----------------------------------------------------------------------------
# Phase A binding requirement (docs/decisions.md): a Union{AffineAxis,
# ScatterAxis} value must never flow into a QSTile-producing call that is
# then passed on to further type-unstable code inside the hot loop. The
# fix is a function barrier: branch on `descriptor.regular` to build a
# concretely-typed axis, then immediately call a `where {R<:Axis, C<:Axis}`
# method with it. Julia specializes that method per concrete (R,C)
# combination that's actually invoked, so QSTile's own type parameters are
# always fully resolved *inside* the specialized method -- never left as a
# partially-applied UnionAll. Each helper below performs exactly one such
# hop and does no further type-unstable dispatch itself.
# ----------------------------------------------------------------------------

@inline function _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    return d.regular ? AffineAxis(d.base, d.stride, d.count) :
        ScatterAxis(view(buffer, (first + 1):(first + d.count)), d.count)
end

@inline function _pack_a_sliver!(
        packed::AbstractVector{T}, storage::S, base::Int,
        rows::R, cols::C, kernel
    ) where {T, S, R <: Axis, C <: Axis}
    pack_a!(packed, SourceTile(storage, base, rows, cols), kernel, identity)
    return nothing
end

@inline function _pack_b_sliver!(
        packed::AbstractVector{T}, storage::S, base::Int,
        rows::R, cols::C, kernel
    ) where {T, S, R <: Axis, C <: Axis}
    pack_b!(packed, SourceTile(storage, base, rows, cols), kernel, identity)
    return nothing
end

@inline function _execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
        kc_len::Int, alpha, beta
    ) where {T, S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    execute_tile!(kernel, destination, packed_a, packed_b, kc_len, alpha, beta)
    return nothing
end

@inline function _scale_micro_tile!(storage::S, base::Int, rows::R, cols::C, beta) where {S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    scale_tile!(destination, beta)
    return nothing
end

# Sliver `s`'s region within a shared packed-panel buffer, at the CURRENT
# block's actual depth `kc_len` (which may be < the buffer's max per-sliver
# capacity on a tail K block: the buffer is sized for `kc_eff`, the largest
# depth any block ever uses). Called identically from the packing step and
# the consuming (execute_tile!) step for a given (jc,pc,ic) iteration, so the
# two always agree on where sliver `s` lives.
@inline function _sliver_range(reg_tile::Int, kc_len::Int, s::Int)
    stride = reg_tile * kc_len
    lo = s * stride + 1
    return lo:(lo + stride - 1)
end

# Apply beta once to every element of C at MR x NR granularity, without
# reading A or B. Shared by execute!'s and execute_tilewise!'s Qk==0/alpha==0
# short-circuit; uses the tw_* (MR/NR-sized) buffers since no macro blocking
# is needed for a beta-only pass.
function _scale_all_of_C!(plan, betaT::T, MRk::Int, NRk::Int, Qm::Int, Qn::Int) where {T}
    m_bufs = (plan.tw_m_buf_A, plan.tw_m_buf_C)
    n_bufs = (plan.tw_n_buf_B, plan.tw_n_buf_C)
    mfirst = 0
    while mfirst < Qm
        mcount = min(MRk, Qm - mfirst)
        (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
        rowsC = _axis_of(dM_C, plan.tw_m_buf_C, 0)
        nfirst = 0
        while nfirst < Qn
            ncount = min(NRk, Qn - nfirst)
            (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            colsC = _axis_of(dN_C, plan.tw_n_buf_C, 0)
            _scale_micro_tile!(plan.Cstorage, plan.Cbase, rowsC, colsC, betaT)
            nfirst += ncount
        end
        mfirst += mcount
    end
    return nothing
end

"""
    ContractPlan

Reusable plan/workspace from [`plan_contract`](@ref): resolved M/N/K
`AxisGroup`s, kernel, operand storage/base, the effective [`Blocking`](@ref),
and every buffer [`execute!`](@ref)/[`execute_tilewise!`](@ref) need --
sized once here, never (re)allocated there.

Field layout (implementation detail, not part of the frozen interface):
`m_buf_*`/`n_buf_*`/`k_buf_*` are macro-block-sized offset buffers (one
`fill_offsets!` call per `jc`/`pc`/`ic` block, reused for every sliver inside
it); `m_desc_*`/`n_desc_*` are per-sliver `BlockDescriptor` tables classified
from those buffers via the 3-arg `describe_block` (one classification per
sliver, no buffer refill); `packed_a`/`packed_b` hold every A/B sliver of the
current macro panel contiguously (`cld(mc,MRk)`/`cld(nc,NRk)` slivers at
`kc`-eff depth). `execute_tilewise!` (the correctness oracle) uses its own,
separate `tw_*` buffers/panels, sized off `mr`/`nr`/`blocking.kc` only, so it
shares no mutable state with the macro-blocking buffers above.
"""
struct ContractPlan{T, Kern, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC}
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

    # Macro-block-sized offset buffers (refilled once per jc/pc/ic block).
    m_buf_A::Vector{Int}
    m_buf_C::Vector{Int}
    n_buf_B::Vector{Int}
    n_buf_C::Vector{Int}
    k_buf_A::Vector{Int}
    k_buf_B::Vector{Int}

    # Per-sliver descriptor tables (classified from the buffers above via the
    # 3-arg describe_block; reused across the pc/ic loops for a given jc/ic).
    m_desc_A::Vector{BlockDescriptor}
    m_desc_C::Vector{BlockDescriptor}
    n_desc_B::Vector{BlockDescriptor}
    n_desc_C::Vector{BlockDescriptor}

    # Packed macro-panel buffers: cld(mc,MRk)/cld(nc,NRk) slivers at kc-eff depth.
    packed_a::Vector{T}
    packed_b::Vector{T}

    # execute_tilewise!'s own small buffers (MR/NR/blocking.kc-sized).
    tw_m_buf_A::Vector{Int}
    tw_m_buf_C::Vector{Int}
    tw_n_buf_B::Vector{Int}
    tw_n_buf_C::Vector{Int}
    tw_k_buf_A::Vector{Int}
    tw_k_buf_B::Vector{Int}
    tw_packed_a::Vector{T}
    tw_packed_b::Vector{T}
end

"""
    plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                  B::StridedView, indB::NTuple{NB,Int},
                  indC::NTuple{NC,Int};
                  kernel = ScalarKernel(Val(8), Val(6), eltype(C)),
                  mc = nothing, kc = nothing, nc = nothing) -> ContractPlan

Planning phase of [`contract!`](@ref): resolves labels into M/N/K
`AxisGroup`s, validates matched axis lengths and matching eltypes, and
preallocates every buffer [`execute!`](@ref) needs. `mc`/`kc`/`nc` are the
macro-blocking factors (see [`Blocking`](@ref)); a `nothing` keyword uses the
corresponding field of `default_blocking(kernel)`. Each is validated `>= 1`
(`ArgumentError` otherwise, via `Blocking`'s own constructor), then rounded/
clamped into the *effective* blocking actually stored on the returned plan:
`mc`/`nc` round up to a whole `mr(kernel)`/`nr(kernel)` multiple and are then
capped at the contraction's own M/N extent (also rounded up to a register-
tile multiple); `kc` is capped at the contraction's own K extent. Throws
`ArgumentError`/`DimensionMismatch` on invalid input.
"""
function plan_contract(
        C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int};
        kernel = _default_kernel(eltype(C)),
        mc::Union{Int, Nothing} = nothing,
        kc::Union{Int, Nothing} = nothing,
        nc::Union{Int, Nothing} = nothing
    ) where {NA, NB, NC}
    T = eltype(C)
    eltype(A) === T ||
        throw(ArgumentError("eltype(A) = $(eltype(A)) does not match eltype(C) = $T"))
    eltype(B) === T ||
        throw(ArgumentError("eltype(B) = $(eltype(B)) does not match eltype(C) = $T"))
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match eltype(C) = $T"))

    defaults = default_blocking(kernel)
    # Blocking's own constructor validates all three >= 1.
    requested = Blocking(
        mc === nothing ? defaults.mc : mc,
        kc === nothing ? defaults.kc : kc,
        nc === nothing ? defaults.nc : nc
    )

    mlabels, nlabels, klabels = _classify_labels(indA, indB, indC)

    mgroup = _build_pair_group(mlabels, indA, A, indC, C)  # maps: (A, C)
    ngroup = _build_pair_group(nlabels, indB, B, indC, C)  # maps: (B, C)
    kgroup = _build_pair_group(klabels, indA, A, indB, B)  # maps: (A, B)

    MRk = mr(kernel)
    NRk = nr(kernel)
    Qm = axis_length(mgroup)
    Qn = axis_length(ngroup)
    Qk = axis_length(kgroup)

    mc_rounded = _roundup(requested.mc, MRk)
    nc_rounded = _roundup(requested.nc, NRk)

    # Qm/Qn/Qk == 0 short-circuits before any block loop ever reads these
    # values; the register-tile-sized floor here just keeps `Blocking`'s
    # own >=1 invariant and every buffer well-formed (never reached by a
    # loop bound in that case).
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

    m_slivers_max = cld(mc_eff, MRk)
    n_slivers_max = cld(nc_eff, NRk)

    m_buf_A = zeros(Int, mc_eff)
    m_buf_C = zeros(Int, mc_eff)
    n_buf_B = zeros(Int, nc_eff)
    n_buf_C = zeros(Int, nc_eff)
    k_buf_A = zeros(Int, kc_eff)
    k_buf_B = zeros(Int, kc_eff)

    m_desc_A = Vector{BlockDescriptor}(undef, m_slivers_max)
    m_desc_C = Vector{BlockDescriptor}(undef, m_slivers_max)
    n_desc_B = Vector{BlockDescriptor}(undef, n_slivers_max)
    n_desc_C = Vector{BlockDescriptor}(undef, n_slivers_max)

    packed_a = zeros(T, m_slivers_max * packed_a_length(kernel, kc_eff))
    packed_b = zeros(T, n_slivers_max * packed_b_length(kernel, kc_eff))

    tw_m_buf_A = zeros(Int, MRk)
    tw_m_buf_C = zeros(Int, MRk)
    tw_n_buf_B = zeros(Int, NRk)
    tw_n_buf_C = zeros(Int, NRk)
    tw_k_buf_A = zeros(Int, kc_eff)
    tw_k_buf_B = zeros(Int, kc_eff)
    tw_packed_a = zeros(T, packed_a_length(kernel, kc_eff))
    tw_packed_b = zeros(T, packed_b_length(kernel, kc_eff))

    return ContractPlan(
        kernel, mgroup, ngroup, kgroup, blocking,
        Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
        m_buf_A, m_buf_C, n_buf_B, n_buf_C, k_buf_A, k_buf_B,
        m_desc_A, m_desc_C, n_desc_B, n_desc_C,
        packed_a, packed_b,
        tw_m_buf_A, tw_m_buf_C, tw_n_buf_B, tw_n_buf_C, tw_k_buf_A, tw_k_buf_B,
        tw_packed_a, tw_packed_b
    )
end

"""
    execute!(plan::ContractPlan, alpha::Number, beta::Number)

Execution phase of [`contract!`](@ref): a BLIS five-loop macro-blocking nest
over `plan.blocking` (`nc`/loop 5, `kc`/loop 4, `mc`/loop 3), packing the
whole B panel once per `(jc,pc)` and the whole A panel once per `(jc,pc,ic)`,
then running `execute_tile!` over every micro-tile in that `(jc,pc,ic)` block
(loops 2/1). `beta` applies exactly once per output element (only on the
first K block, `pc == 0`; later K blocks accumulate with `beta = one(T)`).
Empty output is a no-op; empty K or `alpha == 0` applies `beta` once without
reading `A`/`B`. No buffer is (re)allocated here. Returns `plan.Cstorage`.

See [`execute_tilewise!`](@ref) for the independent tile-by-tile oracle this
is checked against.
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

    if Qk == 0 || iszero(alphaT)
        # Whole-contraction short-circuit: apply beta once to all of C,
        # never reading A or B (no packing at all).
        _scale_all_of_C!(plan, betaT, MRk, NRk, Qm, Qn)
        return plan.Cstorage
    end

    mc_eff = plan.blocking.mc
    kc_eff = plan.blocking.kc
    nc_eff = plan.blocking.nc

    # --- loop 5: jc over N in steps of nc_eff ---
    jc = 0
    while jc < Qn
        nblock = min(nc_eff, Qn - jc)
        n_slivers = cld(nblock, NRk)
        fill_offsets!((plan.n_buf_B, plan.n_buf_C), plan.ngroup, jc, nblock)
        for s in 0:(n_slivers - 1)
            sfirst = s * NRk
            scount = min(NRk, nblock - sfirst)
            plan.n_desc_B[s + 1] = describe_block(plan.n_buf_B, sfirst, scount)
            plan.n_desc_C[s + 1] = describe_block(plan.n_buf_C, sfirst, scount)
        end

        # --- loop 4: pc over K in steps of kc_eff ---
        pc = 0
        firstpanel = true
        while pc < Qk
            kblock = min(kc_eff, Qk - pc)
            fill_offsets!((plan.k_buf_A, plan.k_buf_B), plan.kgroup, pc, kblock)
            dK_A = describe_block(plan.k_buf_A, 0, kblock)
            dK_B = describe_block(plan.k_buf_B, 0, kblock)
            colsA_k = _axis_of(dK_A, plan.k_buf_A, 0)
            rowsB_k = _axis_of(dK_B, plan.k_buf_B, 0)

            beta_eff = firstpanel ? betaT : one(T)

            # Pack the whole B panel for this (jc, pc): every N-sliver.
            for s in 0:(n_slivers - 1)
                sfirst = s * NRk
                colsB = _axis_of(plan.n_desc_B[s + 1], plan.n_buf_B, sfirst)
                bview = view(plan.packed_b, _sliver_range(NRk, kblock, s))
                _pack_b_sliver!(bview, plan.Bstorage, plan.Bbase, rowsB_k, colsB, kernel)
            end

            # --- loop 3: ic over M in steps of mc_eff ---
            ic = 0
            while ic < Qm
                mblock = min(mc_eff, Qm - ic)
                m_slivers = cld(mblock, MRk)
                fill_offsets!((plan.m_buf_A, plan.m_buf_C), plan.mgroup, ic, mblock)
                for r in 0:(m_slivers - 1)
                    rfirst = r * MRk
                    rcount = min(MRk, mblock - rfirst)
                    plan.m_desc_A[r + 1] = describe_block(plan.m_buf_A, rfirst, rcount)
                    plan.m_desc_C[r + 1] = describe_block(plan.m_buf_C, rfirst, rcount)
                end

                # Pack the whole A panel for this (jc, pc, ic): every M-sliver.
                for r in 0:(m_slivers - 1)
                    rfirst = r * MRk
                    rowsA = _axis_of(plan.m_desc_A[r + 1], plan.m_buf_A, rfirst)
                    aview = view(plan.packed_a, _sliver_range(MRk, kblock, r))
                    _pack_a_sliver!(aview, plan.Astorage, plan.Abase, rowsA, colsA_k, kernel)
                end

                # --- loop 2: jr over N-slivers; loop 1: ir over M-slivers ---
                for s in 0:(n_slivers - 1)
                    sfirst = s * NRk
                    colsC = _axis_of(plan.n_desc_C[s + 1], plan.n_buf_C, sfirst)
                    bview = view(plan.packed_b, _sliver_range(NRk, kblock, s))
                    for r in 0:(m_slivers - 1)
                        rfirst = r * MRk
                        rowsC = _axis_of(plan.m_desc_C[r + 1], plan.m_buf_C, rfirst)
                        aview = view(plan.packed_a, _sliver_range(MRk, kblock, r))
                        _execute_micro_tile!(
                            kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                            aview, bview, kblock, alphaT, beta_eff
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
correctness oracle for [`execute!`](@ref)'s five-loop macro-blocking nest
(docs/decisions.md, "Macro-blocking milestone", item 5): tiles M/N in steps
of `mr(plan.kernel)`/`nr(plan.kernel)`, and for each output tile, K in one or
more `plan.blocking.kc`-sized panels (the same role the removed `kc_panel`
keyword played), packing then calling `execute_tile!` with `beta` on the
first panel and `one(T)` on later ones. Uses its own small `tw_*` buffers on
`plan` (sized off `mr`/`nr`/`plan.blocking.kc` only, never the macro `mc`/
`nc` blocks), so it shares no mutable state with `execute!`'s macro buffers.
No buffer is (re)allocated here.
"""
function execute_tilewise!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
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

    m_bufs = (plan.tw_m_buf_A, plan.tw_m_buf_C)
    n_bufs = (plan.tw_n_buf_B, plan.tw_n_buf_C)
    k_bufs = (plan.tw_k_buf_A, plan.tw_k_buf_B)
    kc_panel = plan.blocking.kc

    mfirst = 0
    while mfirst < Qm
        mcount = min(MRk, Qm - mfirst)
        (dM_A, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
        rowsA = _axis_of(dM_A, plan.tw_m_buf_A, 0)
        rowsC = _axis_of(dM_C, plan.tw_m_buf_C, 0)

        nfirst = 0
        while nfirst < Qn
            ncount = min(NRk, Qn - nfirst)
            (dN_B, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            colsB = _axis_of(dN_B, plan.tw_n_buf_B, 0)
            colsC = _axis_of(dN_C, plan.tw_n_buf_C, 0)

            kfirst = 0
            firstpanel = true
            while kfirst < Qk
                kcount = min(kc_panel, Qk - kfirst)
                (dK_A, dK_B) = block_descriptors!(k_bufs, plan.kgroup, kfirst, kcount)
                colsK_A = _axis_of(dK_A, plan.tw_k_buf_A, 0)
                rowsK_B = _axis_of(dK_B, plan.tw_k_buf_B, 0)

                _pack_a_sliver!(plan.tw_packed_a, plan.Astorage, plan.Abase, rowsA, colsK_A, kernel)
                _pack_b_sliver!(plan.tw_packed_b, plan.Bstorage, plan.Bbase, rowsK_B, colsB, kernel)

                beta_eff = firstpanel ? betaT : one(T)
                _execute_micro_tile!(
                    kernel, plan.Cstorage, plan.Cbase, rowsC, colsC,
                    plan.tw_packed_a, plan.tw_packed_b, kcount, alphaT, beta_eff
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
