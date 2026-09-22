# contract!'s frozen signature/label semantics: see docs/decisions.md.
# Planning (labels, AxisGroups, buffers) is split from execution so a
# ContractPlan can be built once and reused:
#   plan_contract(...) -> ContractPlan; execute!(plan, alpha, beta); contract! = both.
# execute! is a BLIS five-loop (NC/KC/MC) nest with packed-panel reuse; the
# pre-macro-blocking tile-by-tile driver is kept unexported as
# `execute_tilewise!`, an independent correctness oracle for it.
# The buffers themselves live in a separate, reusable `ContractWorkspace`
# (src/workspace.jl; docs/decisions.md, "Amendment 1").

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

Note the M/N orientation swap (docs/decisions.md, "Label-order milestone"):
after a swap, `Astorage`/`Abase`/`atransform` describe the ORIGINAL `B`
operand and `Bstorage`/`Bbase`/`btransform` describe the original `A`, so
`plan.Astorage === parent(A)` does not hold in general -- do not assume the
field name still tracks the user-facing argument it is named after.
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
every buffer [`execute!`](@ref) needs.

Label order and orientation (docs/decisions.md, "Label-order milestone"): the
labels inside the M composite (A's free labels) and the N composite (B's free
labels) are each stable-sorted by `abs(stride)` of the label's axis *in `C`*,
ascending, ties keeping the operand's own axis order -- so each composite is
enumerated with `C`'s fastest axis fastest, whatever A's or B's layout is. The
K composite keeps `indA` order. Then, if the sorted M list does *not* begin
with a unit-stride run of at least `mr(kernel)` elements in `C` while the sorted
N list does, the operand roles are swapped: `B` feeds M and `A` feeds N, and
the K maps, the storage/base fields and the conjugation transforms move with
them (so `plan.Astorage` may be `parent(B)`). The result is unchanged either
way; only the walk through `C` is. `mc`/`kc`/`nc` are the macro-blocking
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

    # C's layout, not A's/B's, decides the order within each composite.
    morder = _order_free_labels(mlabels, indC, C)
    norder = _order_free_labels(nlabels, indC, C)

    # Each composite's own leading unit-stride run length, derived ONCE here
    # (not per call site): both the swap decision and the F2 demotion below
    # consume these same two values, keyed on the composite (M or N), not on
    # which orientation ends up feeding the driver's own M role. Computed
    # only for real `T` (T5 review, S4): the swap decision below is already
    # `T <: Real`-gated, and `_demote_for_run`'s generic complex method
    # ignores `run`/`Qm` entirely, so a complex plan pays for neither
    # `_leading_unit_run` call -- unlike before item 3, when `_prefer_swap`'s
    # short-circuit already skipped both for complex; the placeholder `0`
    # keeps this genuinely a no-added-cost refactor for complex, not merely
    # "unused but computed".
    run_m = T <: Real ? _leading_unit_run(morder, indC, C) : 0
    run_n = T <: Real ? _leading_unit_run(norder, indC, C) : 0

    mgroup = _build_pair_group(morder, indA, A, indC, C)  # maps: (A, C)
    ngroup = _build_pair_group(norder, indB, B, indC, C)  # maps: (B, C)
    kgroup = _build_pair_group(klabels, indA, A, indB, B)  # maps: (A, B)

    Qm = axis_length(mgroup)
    Qn = axis_length(ngroup)
    Qk = axis_length(kgroup)

    # Resolved here, not in the signature default: the demotion needs Qm, and
    # Qm depends on the orientation, so both candidates are resolved.
    # Nothing between the eltype checks and here reads `kernel`.
    kernel_asis = kernel === nothing ? _default_kernel(T, Qm, Qn) : kernel
    kernel_swapped = kernel === nothing ? _default_kernel(T, Qn, Qm) : kernel

    # Complex kernels (`PlanarKernel`/`OneMKernel`) once always scatter-stored
    # unconditionally, and back then the swap measurably lost the as-is
    # orientation's N-side locality for no store-side gain (~2-4%, `ccsd_t_3`,
    # ComplexF64/32). `PlanarKernel` now has a vectorized store fast path
    # (`_store_tile_planar_vector!`, `src/kernels/planar.jl`), so there IS
    # potentially something for the swap to win on the complex path -- this
    # `T <: Real` guard is a deliberately deferred, unmeasured follow-up
    # (docs/proposals/complex-fast-paths.md Decision 3), not a settled case of
    # nothing to gain. Real kernels (`SIMDKernel` and, for this run-length
    # rule, `ScalarKernel` too) keep the swap. Uses the pre-computed
    # `run_m`/`run_n` (the core `_prefer_swap` method) rather than the
    # label-list wrapper, per the tensorcontract-rs-comparison milestone's
    # run-length dedup -- see `_prefer_swap`'s definition above.
    if T <: Real && _prefer_swap(run_m, run_n, mr(kernel_asis), mr(kernel_swapped))
        # B takes the M role and A the N role. Everything operand-bound moves
        # together: the groups (each already carries its own C map), the K
        # group's two maps, the storage/base pair `_plan_contract` reads off
        # its A/B arguments, and the packing transforms. The contraction is
        # unchanged: `*` commutes on `T` and `conj` is elementwise, so
        # `sum_k conj?(B[n,k]) * conj?(A[m,k])` is the same sum.
        kgroup_swapped = _build_pair_group(klabels, indB, B, indA, A)  # maps: (B, A)
        # F2 demotion (see `_demote_for_run`): only for an auto-selected
        # kernel, keyed on the CHOSEN (post-swap) M orientation, i.e. N's own
        # run against the kernel it would actually run.
        kernel_final = kernel === nothing ?
            _demote_for_run(T, kernel_swapped, run_n, Qn, Qk) :
            kernel_swapped
        return _plan_contract(
            C, B, A, indC, ngroup, mgroup, kgroup_swapped, Qn, Qm, Qk,
            kernel_final, btransform, atransform, mc, kc, nc, workspace, allocator, oracle
        )
    end
    kernel_final = kernel === nothing ?
        _demote_for_run(T, kernel_asis, run_m, Qm, Qk) :
        kernel_asis
    return _plan_contract(
        C, A, B, indC, mgroup, ngroup, kgroup, Qm, Qn, Qk,
        kernel_final, atransform, btransform, mc, kc, nc, workspace, allocator, oracle
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
