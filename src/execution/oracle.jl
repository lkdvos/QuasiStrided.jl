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
