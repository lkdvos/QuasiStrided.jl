# OWNER: driver implementer (Phase 3). See docs/decisions.md for the frozen
# contract! signature and Julia-Microkernel-Tile-Interface-Design.md section 10.
#
# Implements: contract! axis-list entry point, reusable planning/workspace
# construction, serial output tiling with multiple K panels.
#
# Frozen signature (do not change without a main-process decision-record update):
#
#   contract!(C::StridedView, alpha::Number,
#             A::StridedView, indA::NTuple{NA,Int},
#             B::StridedView, indB::NTuple{NB,Int},
#             beta::Number,
#             indC::NTuple{NC,Int}) where {NA,NB,NC}
#
# Label semantics: indA/indB/indC attach one Int label per axis, in axis order.
# A label present in both indA and indB but absent from indC is a contracted
# (K) axis. A label present in indC and in exactly one of indA/indB is a free
# axis (M if from A, N if from B). Repeated labels within a single one of
# indA/indB/indC (diagonals), and labels present in indC but absent from both
# indA and indB, are out of scope for this milestone and must raise
# ArgumentError. Every axis length for a shared label must match across the
# operands/tensors that share it.
#
# Design (Julia-Microkernel-Tile-Interface-Design.md section 2 and section
# 10): planning (label resolution into M/N/K AxisGroups, and reusable
# workspace/buffer sizing) is a separate step from execution (the serial
# output-tiling driver, multiple K panels, beta_effective handling), so setup
# cost is independently measurable and a plan/workspace can be built once and
# reused:
#
#   plan_contract(C, A, indA, B, indB, indC; kernel, kc_panel) -> ContractPlan
#   execute!(plan, alpha, beta) -> C's parent storage, mutated in place
#   contract!(...) = execute!(plan_contract(...), alpha, beta)
#
# `plan_contract` does all label resolution, AxisGroup construction, and
# buffer allocation (sized for the kernel's register-tile shape and the
# chosen K-panel size); `execute!` performs only the tiling loop against
# those preallocated buffers -- no allocation-sized decisions are made there.

# ----------------------------------------------------------------------------
# Label resolution (spec section 3 / section 9's "responsibilities outside
# AxisGroup")
# ----------------------------------------------------------------------------

# Classify every label appearing in indA ∪ indB ∪ indC into the M/N/K
# categories fixed by docs/decisions.md. Returns (mlabels, nlabels, klabels),
# each a Vector{Int} in a deterministic order (mlabels/klabels in indA
# appearance order, nlabels in indB appearance order).
#
# Every label falls into exactly one of the following (inA, inB, inC)
# combinations; anything not listed as M/N/K below is out of scope and raises
# ArgumentError:
#   (T,F,T) -> M (free, from A)         (F,T,T) -> N (free, from B)
#   (T,T,F) -> K (contracted)
#   (F,F,T) -> ArgumentError: labeled in C but absent from both A and B
#   (T,T,T) -> ArgumentError: present in all three (batch-like); unsupported
#              this milestone (deferred per the design doc's batch labels)
#   (T,F,F) -> ArgumentError: present only in A (dangling)
#   (F,T,F) -> ArgumentError: present only in B (dangling)
function _classify_labels(indA::NTuple{NA,Int}, indB::NTuple{NB,Int},
                           indC::NTuple{NC,Int}) where {NA,NB,NC}
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
            throw(ArgumentError(
                "label $lbl appears in indA, indB, and indC: labels present in all " *
                "three operands (batch-like) are out of scope for this milestone"))
        elseif inB && !inC
            push!(klabels, lbl)
        elseif !inB && inC
            push!(mlabels, lbl)
        else
            throw(ArgumentError(
                "label $lbl appears only in indA (not in indB or indC): not a valid " *
                "free (M) or contracted (K) label"))
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
            throw(ArgumentError(
                "label $lbl appears only in indB (not in indA or indC): not a valid " *
                "free (N) or contracted (K) label"))
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

# Build the two-map AxisGroup for one of the M/N/K categories: `labels` gives
# the shared logical enumeration order, `ind1`/`v1` and `ind2`/`v2` give the
# per-operand axis-position lookup and StridedView metadata for the two
# participating tensors (map order (v1, v2) matches the order consumed
# downstream: (A,C) for M, (B,C) for N, (A,B) for K). Every label is
# guaranteed present in both `ind1` and `ind2` by the classification above.
# Checks matched axis lengths, raising DimensionMismatch otherwise (spec
# section 9: "Checking that matched tensor labels have equal lengths").
function _build_pair_group(labels::Vector{Int},
                            ind1::NTuple, v1::StridedView,
                            ind2::NTuple, v2::StridedView)
    D = length(labels)
    pos1 = ntuple(d -> findfirst(==(labels[d]), ind1)::Int, D)
    pos2 = ntuple(d -> findfirst(==(labels[d]), ind2)::Int, D)
    lens = ntuple(D) do d
        l1 = size(v1, pos1[d])
        l2 = size(v2, pos2[d])
        l1 == l2 || throw(DimensionMismatch(
            "label $(labels[d]) has mismatched axis length: $l1 vs $l2"))
        l1
    end
    s1 = ntuple(d -> Base.strides(v1)[pos1[d]], D)
    s2 = ntuple(d -> Base.strides(v2)[pos2[d]], D)
    return AxisGroup(lens, (s1, s2))
end

# ----------------------------------------------------------------------------
# Plan / reusable workspace
# ----------------------------------------------------------------------------

_default_kernel(::Type{T}) where {T} = ScalarKernel(Val(8), Val(6), T)

"""
    ContractPlan

Reusable plan and workspace produced by [`plan_contract`](@ref): resolved
M/N/K `AxisGroup`s, the kernel to execute with, storage/base metadata
extracted from the three `StridedView`s, the chosen per-panel K depth
(`kc_panel`, already clamped to the contraction's total K extent), and every
buffer the tiling driver in [`execute!`](@ref) needs (per-axis offset
buffers, sized for one MR/NR/kc_panel-sized chunk, and the packed A/B
panels). All of it is sized once, at plan time, and reused unchanged across
however many `execute!` calls follow -- `execute!` itself performs no
allocation-sized decisions and no buffer (re)allocation.
"""
struct ContractPlan{T,Kern,GM<:AxisGroup,GN<:AxisGroup,GK<:AxisGroup,SA,SB,SC}
    kernel::Kern
    mgroup::GM
    ngroup::GN
    kgroup::GK
    kc_panel::Int
    Astorage::SA
    Abase::Int
    Bstorage::SB
    Bbase::Int
    Cstorage::SC
    Cbase::Int
    m_buf_A::Vector{Int}
    m_buf_C::Vector{Int}
    n_buf_B::Vector{Int}
    n_buf_C::Vector{Int}
    k_buf_A::Vector{Int}
    k_buf_B::Vector{Int}
    packed_a::Vector{T}
    packed_b::Vector{T}
end

"""
    plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                  B::StridedView, indB::NTuple{NB,Int},
                  indC::NTuple{NC,Int};
                  kernel = ScalarKernel(Val(8), Val(6), eltype(C)),
                  kc_panel::Int = typemax(Int)) -> ContractPlan

The planning phase of [`contract!`](@ref), separated out so its cost (label
resolution, `AxisGroup` construction, buffer allocation) is measurable
independently of execution and so the resulting [`ContractPlan`](@ref) can be
constructed once and reused across many [`execute!`](@ref) calls (e.g. with
different `alpha`/`beta`, or after mutating the same underlying arrays in
place).

Resolves `indA`/`indB`/`indC` labels into M/N/K `AxisGroup`s exactly per
`docs/decisions.md`'s frozen label semantics (see `_classify_labels`),
validates matched-label axis lengths and `eltype(A) === eltype(B) ===
eltype(C)`, and preallocates every buffer `execute!` needs: per-axis offset
buffers sized for one chunk of the kernel's register-tile shape
(`mr(kernel)`/`nr(kernel)`) and one K panel (`kc_panel`, clamped to the
contraction's total K extent `Q_K`), plus the packed A/B panel buffers sized
accordingly. `kc_panel` defaults to `typemax(Int)`, i.e. a single K panel
covering the whole contraction; pass a smaller value to force multiple K
panels (spec section 10).

Throws `ArgumentError` for diagonals, out-of-scope labels, or an eltype/
kernel-scalar-type mismatch; `DimensionMismatch` for a matched-label length
mismatch.
"""
function plan_contract(C::StridedView, A::StridedView, indA::NTuple{NA,Int},
                        B::StridedView, indB::NTuple{NB,Int},
                        indC::NTuple{NC,Int};
                        kernel = _default_kernel(eltype(C)),
                        kc_panel::Int = typemax(Int)) where {NA,NB,NC}
    T = eltype(C)
    eltype(A) === T ||
        throw(ArgumentError("eltype(A) = $(eltype(A)) does not match eltype(C) = $T"))
    eltype(B) === T ||
        throw(ArgumentError("eltype(B) = $(eltype(B)) does not match eltype(C) = $T"))
    scalartype(kernel) === T ||
        throw(ArgumentError("kernel scalar type $(scalartype(kernel)) does not match eltype(C) = $T"))
    kc_panel >= 1 ||
        throw(ArgumentError("kc_panel must be at least 1, got $kc_panel"))

    mlabels, nlabels, klabels = _classify_labels(indA, indB, indC)

    mgroup = _build_pair_group(mlabels, indA, A, indC, C)  # maps: (A, C)
    ngroup = _build_pair_group(nlabels, indB, B, indC, C)  # maps: (B, C)
    kgroup = _build_pair_group(klabels, indA, A, indB, B)  # maps: (A, B)

    MRk = mr(kernel)
    NRk = nr(kernel)
    Qk = axis_length(kgroup)
    kc_used = Qk == 0 ? 0 : min(kc_panel, Qk)

    Astorage = parent(A)
    Abase = offset(A)
    Bstorage = parent(B)
    Bbase = offset(B)
    Cstorage = parent(C)
    Cbase = offset(C)

    m_buf_A = zeros(Int, MRk)
    m_buf_C = zeros(Int, MRk)
    n_buf_B = zeros(Int, NRk)
    n_buf_C = zeros(Int, NRk)
    k_buf_A = zeros(Int, kc_used)
    k_buf_B = zeros(Int, kc_used)

    packed_a = zeros(T, packed_a_length(kernel, kc_used))
    packed_b = zeros(T, packed_b_length(kernel, kc_used))

    return ContractPlan(kernel, mgroup, ngroup, kgroup, kc_used,
                         Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
                         m_buf_A, m_buf_C, n_buf_B, n_buf_C, k_buf_A, k_buf_B,
                         packed_a, packed_b)
end

# ----------------------------------------------------------------------------
# Execution: serial output-tiling driver with multiple K panels
# ----------------------------------------------------------------------------

"""
    execute!(plan::ContractPlan, alpha::Number, beta::Number)

The execution phase of [`contract!`](@ref): a serial driver over output
tiles, iterating M and N in chunks of `mr(plan.kernel)`/`nr(plan.kernel)`
(the last chunk in each dimension may be a valid tail smaller than that), and
for each output tile, K in one or more contiguous panels of size
`plan.kc_panel` in increasing order (Julia-Microkernel-Tile-Interface-Design.md
section 10):

    for each output tile (M-chunk x N-chunk):
        for each K panel in increasing order:
            pack A panel, pack B panel
            execute_tile! with alpha and beta_effective

`beta_effective` is `beta` on the first K panel of each output tile and
`one(T)` on every subsequent panel, so each tile's contribution is
accumulated from zero across panels and `beta` is applied to the destination
exactly once per output element, not once per panel.

Short-circuits (checked before any packing or destination read, per spec
section 10):
- If the output is empty (`M` or `N` axis length zero), returns immediately:
  no packing, no beta scaling, nothing read or written.
- If `K`'s axis length is zero, or `alpha == zero(T)` (once converted),
  applies `beta` scaling once to every element of the output and returns,
  without ever reading `A` or `B`.

Reuses every buffer in `plan` -- no buffer is (re)allocated here. Returns
`plan.Cstorage` (the parent storage backing `C`, mutated in place).
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

    m_bufs = (plan.m_buf_A, plan.m_buf_C)
    n_bufs = (plan.n_buf_B, plan.n_buf_C)
    k_bufs = (plan.k_buf_A, plan.k_buf_B)

    if Qk == 0 || iszero(alphaT)
        # Whole-contraction short-circuit: apply beta once to all of C,
        # never reading A or B (no packing at all).
        mfirst = 0
        while mfirst < Qm
            mcount = min(MRk, Qm - mfirst)
            (_, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
            row_C = axis_from_descriptor(dM_C, plan.m_buf_C)
            nfirst = 0
            while nfirst < Qn
                ncount = min(NRk, Qn - nfirst)
                (_, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
                col_C = axis_from_descriptor(dN_C, plan.n_buf_C)
                destination = DestinationTile(plan.Cstorage, plan.Cbase, row_C, col_C)
                scale_tile!(destination, betaT)
                nfirst += ncount
            end
            mfirst += mcount
        end
        return plan.Cstorage
    end

    mfirst = 0
    while mfirst < Qm
        mcount = min(MRk, Qm - mfirst)
        (dM_A, dM_C) = block_descriptors!(m_bufs, plan.mgroup, mfirst, mcount)
        row_A = axis_from_descriptor(dM_A, plan.m_buf_A)
        row_C = axis_from_descriptor(dM_C, plan.m_buf_C)

        nfirst = 0
        while nfirst < Qn
            ncount = min(NRk, Qn - nfirst)
            (dN_B, dN_C) = block_descriptors!(n_bufs, plan.ngroup, nfirst, ncount)
            col_B = axis_from_descriptor(dN_B, plan.n_buf_B)
            col_C = axis_from_descriptor(dN_C, plan.n_buf_C)

            destination = DestinationTile(plan.Cstorage, plan.Cbase, row_C, col_C)

            kfirst = 0
            firstpanel = true
            while kfirst < Qk
                kcount = min(plan.kc_panel, Qk - kfirst)
                (dK_A, dK_B) = block_descriptors!(k_bufs, plan.kgroup, kfirst, kcount)
                row_K_A = axis_from_descriptor(dK_A, plan.k_buf_A)
                col_K_B = axis_from_descriptor(dK_B, plan.k_buf_B)

                source_A = SourceTile(plan.Astorage, plan.Abase, row_A, row_K_A)
                source_B = SourceTile(plan.Bstorage, plan.Bbase, col_K_B, col_B)

                pack_a!(plan.packed_a, source_A, kernel, identity)
                pack_b!(plan.packed_b, source_B, kernel, identity)

                beta_eff = firstpanel ? betaT : one(T)
                execute_tile!(kernel, destination, plan.packed_a, plan.packed_b,
                              kcount, alphaT, beta_eff)

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

Compute `C[indC] = alpha * sum_K A[indA] * B[indB] + beta * C[indC]`, where
one `Int` label is given per axis of `indA`/`indB`/`indC` (axis order), a
label in both `indA` and `indB` but absent from `indC` names a contracted
(K) axis, and a label in `indC` and in exactly one of `indA`/`indB` names a
free axis (M if from A, N if from B). Diagonals (a label repeated within one
tensor's own index tuple) and labels in `indC` absent from both `indA`/
`indB` raise `ArgumentError`; matched labels across tensors with differing
axis lengths raise `DimensionMismatch`.

Convenience wrapper equivalent to `execute!(plan_contract(C, A, indA, B,
indB, indC), alpha, beta)` -- see [`plan_contract`](@ref)/[`execute!`](@ref)
to separate planning from execution (e.g. to reuse a plan across several
calls, or to measure setup and steady-state execution cost independently).
Returns `C`.
"""
function contract!(C::StridedView, alpha::Number,
                    A::StridedView, indA::NTuple{NA,Int},
                    B::StridedView, indB::NTuple{NB,Int},
                    beta::Number,
                    indC::NTuple{NC,Int}) where {NA,NB,NC}
    plan = plan_contract(C, A, indA, B, indB, indC)
    execute!(plan, alpha, beta)
    return C
end
