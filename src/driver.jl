# contract!'s frozen signature/label semantics: see docs/decisions.md.
# Planning (label resolution, AxisGroup/buffer construction) is split from
# execution (the tiling loop) so a ContractPlan can be built once and reused:
#   plan_contract(...) -> ContractPlan; execute!(plan, alpha, beta); contract! = both.

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

"""
    ContractPlan

Reusable plan/workspace from [`plan_contract`](@ref): resolved M/N/K
`AxisGroup`s, kernel, operand storage/base, `kc_panel`, and every buffer
[`execute!`](@ref) needs — sized once here, never (re)allocated there.
"""
struct ContractPlan{T, Kern, GM <: AxisGroup, GN <: AxisGroup, GK <: AxisGroup, SA, SB, SC}
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

Planning phase of [`contract!`](@ref): resolves labels into M/N/K
`AxisGroup`s, validates matched axis lengths and matching eltypes, and
preallocates every buffer [`execute!`](@ref) needs. `kc_panel` defaults to
one panel covering the whole contraction; pass a smaller value to force
multiple K panels. Throws `ArgumentError`/`DimensionMismatch` on invalid input.
"""
function plan_contract(
        C::StridedView, A::StridedView, indA::NTuple{NA, Int},
        B::StridedView, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int};
        kernel = _default_kernel(eltype(C)),
        kc_panel::Int = typemax(Int)
    ) where {NA, NB, NC}
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

    return ContractPlan(
        kernel, mgroup, ngroup, kgroup, kc_used,
        Astorage, Abase, Bstorage, Bbase, Cstorage, Cbase,
        m_buf_A, m_buf_C, n_buf_B, n_buf_C, k_buf_A, k_buf_B,
        packed_a, packed_b
    )
end

"""
    execute!(plan::ContractPlan, alpha::Number, beta::Number)

Execution phase of [`contract!`](@ref): tiles M/N in chunks of
`mr(plan.kernel)`/`nr(plan.kernel)`, and for each output tile, K in one or
more `plan.kc_panel`-sized panels, packing then calling `execute_tile!` with
`beta` on the first panel and `one(T)` on later ones (so `beta` applies
exactly once per output element). Empty output is a no-op; empty K or
`alpha == 0` applies `beta` once without reading `A`/`B`. No buffer is
(re)allocated here. Returns `plan.Cstorage`.
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
                execute_tile!(
                    kernel, destination, plan.packed_a, plan.packed_b,
                    kcount, alphaT, beta_eff
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
