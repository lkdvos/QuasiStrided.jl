# Buffer workspace for the contraction driver, split out of `ContractPlan` per
# docs/decisions.md, "Amendment 1: `ContractWorkspace` and the `allocator`
# keyword". Frozen typing discipline from that milestone's workspace/allocator
# design-constraints section: `VT` is a `where`-bound parameter resolved at
# construction (never a `Union`- or `AbstractVector`-typed field), and the
# offset buffers stay concretely `Vector{Int}` -- acquired as non-temporaries,
# so only the packed panels genuinely route through the allocator.
#
# Frozen import convention: TensorOperations is always reached as `TO.<name>`.
import TensorOperations as TO

"""
    ContractWorkspace{T,VT<:AbstractVector{T}}

Every buffer [`execute!`](@ref) and `execute_tilewise!` need, sized once at
construction (or grown by [`reserve!`](@ref)) and never (re)allocated during
execution. Built by [`plan_contract`](@ref) and held by [`ContractPlan`](@ref);
pass an existing one back as `plan_contract(...; workspace = ws)` to reuse its
buffers across contractions of different shapes.

`VT` is the vector type of the two packed macro panels: `Vector{T}` on the
default, GC-owned, [`reserve!`](@ref)-able path; on an explicit-allocator path
it is whatever that allocator returns, and the workspace is then scoped to that
one call -- [`release!`](@ref) it and drop it, never pass it back as
`workspace = ws` (docs/decisions.md, "Verified allocator behavior").

The `tw_*` buffers belong to `execute_tilewise!`, the independent oracle, and
are allocated only under `oracle = true` -- except the four `MR`/`NR`-sized
ones, which the beta-only pass of *both* drivers uses.

Field layout is an implementation detail, not part of the frozen interface.
"""
struct ContractWorkspace{T, VT <: AbstractVector{T}}
    # Macro-block-sized offset buffers: one fill_offsets! per jc/pc/ic block,
    # reused by every sliver inside it.
    m_buf_A::Vector{Int}
    m_buf_C::Vector{Int}
    n_buf_B::Vector{Int}
    n_buf_C::Vector{Int}
    k_buf_A::Vector{Int}
    k_buf_B::Vector{Int}

    # Per-sliver descriptors classified from the buffers above (3-arg
    # describe_block), reused across the pc/ic loops of a given jc/ic.
    m_desc_A::Vector{BlockDescriptor}
    m_desc_C::Vector{BlockDescriptor}
    n_desc_B::Vector{BlockDescriptor}
    n_desc_C::Vector{BlockDescriptor}

    # Packed macro-panel buffers: cld(mc,MRk)/cld(nc,NRk) slivers at kc-eff
    # depth. The ONLY allocator-routed temporaries in this struct.
    packed_a::VT
    packed_b::VT

    # execute_tilewise!'s own small buffers (MR/NR/blocking.kc-sized).
    # Deliberately NOT shared with the macro buffers above: the oracle must
    # have no mutable state in common with the code it checks.
    tw_m_buf_A::Vector{Int}
    tw_m_buf_C::Vector{Int}
    tw_n_buf_B::Vector{Int}
    tw_n_buf_C::Vector{Int}
    tw_k_buf_A::Vector{Int}
    tw_k_buf_B::Vector{Int}
    tw_packed_a::VT
    tw_packed_b::VT
end

# Element counts every buffer needs for `kernel` at the *effective* `blocking`
# (the rounded/clamped one `plan_contract` stores on the plan). Shared by the
# constructors and by `reserve!` so the two can never disagree.
@inline function _workspace_sizes(kernel, blocking::Blocking)
    MRk = mr(kernel)
    NRk = nr(kernel)
    kc = blocking.kc
    pa = packed_a_length(kernel, kc)
    pb = packed_b_length(kernel, kc)
    m_slivers = cld(blocking.mc, MRk)
    n_slivers = cld(blocking.nc, NRk)
    return (
        mc = blocking.mc, nc = blocking.nc, kc = kc,
        mr = MRk, nr = NRk,
        m_slivers = m_slivers, n_slivers = n_slivers,
        packed_a = m_slivers * pa, packed_b = n_slivers * pb,
        tw_packed_a = pa, tw_packed_b = pb,
    )
end

# `Val(true)`: a genuine temporary, routed through `allocator` and never
# `resize!`d afterwards (docs/decisions.md, "Verified allocator behavior").
@inline function _alloc_temp(::Type{T}, n::Int, allocator) where {T}
    return TO.tensoralloc(Vector{T}, (n,), Val(true), allocator)
end

# `Val(false)`: a non-temporary, which every allocator serves as a plain,
# GC-owned `Vector{Int}` -- the frozen requirement for the offset buffers. The
# assertion makes a hypothetical violation fail loudly here rather than
# silently inside `fill_offsets!`.
@inline function _alloc_offsets(n::Int, allocator)
    return TO.tensoralloc(Vector{Int}, (n,), Val(false), allocator)::Vector{Int}
end

# Descriptor arrays are not tensor buffers (`BlockDescriptor` has no
# `scalartype`), so they never route through `tensoralloc`. `undef` is safe:
# `_classify_slivers!` writes entry `s+1` before the same iteration reads it.
@inline _alloc_descriptors(n::Int) = Vector{BlockDescriptor}(undef, n)

# The 18-field layout, written out exactly once for both constructors below:
# `ints` allocates a `Vector{Int}` of a given length, while the packed panels
# arrive already allocated because their type is what fixes `VT`.
@inline function _build_workspace(
        ::Type{T}, s, ntw::Int, ints::F,
        packed_a::VT, packed_b::VT, tw_packed_a::VT, tw_packed_b::VT
    ) where {T, F, VT <: AbstractVector{T}}
    return ContractWorkspace{T, VT}(
        ints(s.mc), ints(s.mc),
        ints(s.nc), ints(s.nc),
        ints(s.kc), ints(s.kc),
        _alloc_descriptors(s.m_slivers), _alloc_descriptors(s.m_slivers),
        _alloc_descriptors(s.n_slivers), _alloc_descriptors(s.n_slivers),
        packed_a, packed_b,
        ints(s.mr), ints(s.mr),
        ints(s.nr), ints(s.nr),
        ints(ntw), ints(ntw),
        tw_packed_a, tw_packed_b,
    )
end

"""
    ContractWorkspace(T, kernel, blocking::Blocking, oracle::Bool, allocator)
    ContractWorkspace(T, kernel, blocking::Blocking;
                      oracle = true, allocator = TensorOperations.DefaultAllocator())

Build a workspace for scalar type `T`, `kernel` and an already-effective
`blocking`. `oracle = false` skips the `execute_tilewise!`-only buffers
entirely (they are left empty), which is what the TensorOperations backend
path passes.

Under `DefaultAllocator` every buffer is an ordinary `Vector`, so the result is
a `ContractWorkspace{T,Vector{T}}` that [`reserve!`](@ref) may later grow.
Under any other allocator the packed panels are acquired once via
`TensorOperations.tensoralloc(..., Val(true), allocator)`, are never resized,
and must be handed back with [`release!`](@ref).

Buffers are `undef`-initialized, not zeroed: `_pack_panel!` (`src/packing.jl`)
writes every slot of a panel it is given, padding included, and the offset
buffers are fully rewritten by `fill_offsets!` before each block is read.
"""
function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking, oracle::Bool, ::TO.DefaultAllocator
    ) where {T}
    s = _workspace_sizes(kernel, blocking)
    return _build_workspace(
        T, s, oracle ? s.kc : 0, n -> Vector{Int}(undef, n),
        Vector{T}(undef, s.packed_a), Vector{T}(undef, s.packed_b),
        Vector{T}(undef, oracle ? s.tw_packed_a : 0),
        Vector{T}(undef, oracle ? s.tw_packed_b : 0),
    )
end

function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking, oracle::Bool, allocator
    ) where {T}
    s = _workspace_sizes(kernel, blocking)

    # Acquisition order matters for arena allocators: `release!` frees in the
    # exact reverse order.
    packed_a = _alloc_temp(T, s.packed_a, allocator)
    packed_b = _alloc_temp(T, s.packed_b, allocator)
    tw_packed_a = _alloc_temp(T, oracle ? s.tw_packed_a : 0, allocator)
    tw_packed_b = _alloc_temp(T, oracle ? s.tw_packed_b : 0, allocator)

    return _build_workspace(
        T, s, oracle ? s.kc : 0, n -> _alloc_offsets(n, allocator),
        packed_a, packed_b, tw_packed_a, tw_packed_b
    )
end

function ContractWorkspace(
        ::Type{T}, kernel, blocking::Blocking;
        oracle::Bool = true, allocator = TO.DefaultAllocator()
    ) where {T}
    return ContractWorkspace(T, kernel, blocking, oracle, allocator)
end

"""
    reserve!(ws::ContractWorkspace{T,Vector{T}}, kernel, blocking::Blocking,
             oracle::Bool) -> ws

Grow `ws` in place so every buffer is large enough for `kernel` at the
effective `blocking`, and return it. Grow-only: it never shrinks a buffer,
never reallocates one that is already large enough, and never hands back a
`view`; every consumer is length-tolerant and addresses only the live region
of the *current* block. `oracle = false` leaves the `execute_tilewise!`-only
buffers alone. Defined only for the default, GC-owned path -- an
allocator-provided temporary must never be `resize!`d (docs/decisions.md,
"Verified allocator behavior"), so that path sizes its workspace exactly once
at construction instead.
"""
function reserve!(
        ws::ContractWorkspace{T, Vector{T}}, kernel, blocking::Blocking, oracle::Bool
    ) where {T}
    s = _workspace_sizes(kernel, blocking)

    _grow!(ws.m_buf_A, s.mc)
    _grow!(ws.m_buf_C, s.mc)
    _grow!(ws.n_buf_B, s.nc)
    _grow!(ws.n_buf_C, s.nc)
    _grow!(ws.k_buf_A, s.kc)
    _grow!(ws.k_buf_B, s.kc)

    _grow!(ws.m_desc_A, s.m_slivers)
    _grow!(ws.m_desc_C, s.m_slivers)
    _grow!(ws.n_desc_B, s.n_slivers)
    _grow!(ws.n_desc_C, s.n_slivers)

    _grow!(ws.packed_a, s.packed_a)
    _grow!(ws.packed_b, s.packed_b)

    _grow!(ws.tw_m_buf_A, s.mr)
    _grow!(ws.tw_m_buf_C, s.mr)
    _grow!(ws.tw_n_buf_B, s.nr)
    _grow!(ws.tw_n_buf_C, s.nr)

    if oracle
        _grow!(ws.tw_k_buf_A, s.kc)
        _grow!(ws.tw_k_buf_B, s.kc)
        _grow!(ws.tw_packed_a, s.tw_packed_a)
        _grow!(ws.tw_packed_b, s.tw_packed_b)
    end

    return ws
end

# Grow-only resize. `resize!` upward leaves the new region undefined, which is
# the same `undef` discipline the constructors use.
@inline function _grow!(v::Vector, n::Int)
    length(v) < n && resize!(v, n)
    return v
end

"""
    release!(ws::ContractWorkspace, allocator) -> nothing

Hand `ws`'s allocator-routed temporaries (`packed_a`/`packed_b`, plus the
oracle's `tw_packed_*` when present) back to `allocator` via
`TensorOperations.tensorfree!`, in the exact reverse of acquisition order so
LIFO arenas unwind correctly. The offset and descriptor buffers are GC-owned
and untouched; `ws` must not be used afterwards. Calling this unconditionally
is correct for every allocator -- it returns memory for `ManualAllocator` and
is a no-op for the ones that unwind via `allocator_reset!`/`@no_escape`
instead (docs/decisions.md, "Verified allocator behavior").
"""
function release!(ws::ContractWorkspace, allocator)
    TO.tensorfree!(ws.tw_packed_b, allocator)
    TO.tensorfree!(ws.tw_packed_a, allocator)
    TO.tensorfree!(ws.packed_b, allocator)
    TO.tensorfree!(ws.packed_a, allocator)
    return nothing
end

# Whether `ws` carries `execute_tilewise!`'s buffers. `packed_a_length(kernel,
# kc) == mr(kernel) * kc >= 1` for every legal blocking (`kc >= 1`), so an
# empty `tw_packed_a` means -- and only means -- the workspace was built with
# `oracle = false`.
@inline _has_oracle(ws::ContractWorkspace) = !isempty(ws.tw_packed_a)
