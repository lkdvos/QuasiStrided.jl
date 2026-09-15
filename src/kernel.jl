# Concrete scalar reference kernel, plus everything every `DescriptorKernel`
# shares: descriptor forwarding, the axpby element helpers, the constructor
# checks the vector kernels run, and the validation prologues all four
# `execute_tile!`/`store_tile!` implementations run.

"""
    ScalarKernel(descriptor::KernelDescriptor{MR,NR,T})
    ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Scalar reference microkernel for register-tile shape `(MR, NR)` and scalar
type `T`. `zero_accumulator` returns an ordinary `Matrix{T}` of size
`(MR, NR)`, indexed `acc[i+1, j+1]` for zero-based `(i, j)`.
"""
struct ScalarKernel{MR, NR, T} <: DescriptorKernel{MR, NR, T}
    descriptor::KernelDescriptor{MR, NR, T}
end

ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    ScalarKernel(KernelDescriptor(Val(MR), Val(NR), T))

# Shared forwarding for any DescriptorKernel (ScalarKernel, SIMDKernel, ...).
mr(k::DescriptorKernel) = mr(k.descriptor)
nr(k::DescriptorKernel) = nr(k.descriptor)
scalartype(k::DescriptorKernel) = scalartype(k.descriptor)
packed_a_offset(k::DescriptorKernel, i::Int, p::Int) = packed_a_offset(k.descriptor, i, p)
packed_b_offset(k::DescriptorKernel, j::Int, p::Int) = packed_b_offset(k.descriptor, j, p)
packed_a_length(k::DescriptorKernel, kc::Int) = packed_a_length(k.descriptor, kc)
packed_b_length(k::DescriptorKernel, kc::Int) = packed_b_length(k.descriptor, kc)

# pack_a!/pack_b! dispatch on a bare KernelDescriptor; forward any wrapper.
#
# GUARDRAIL: every forwarded argument needs its OWN bound type parameter (`V`,
# `K`, `F`). Leaving one unbound here reintroduces Phase 2b finding 5's
# ~80 B/call of dynamic dispatch. `V` is unconstrained rather than
# `<: AbstractVector{T}` so that a `PackedPanel` (src/panel.jl) forwards too;
# `T` comes from `K` instead.
pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_a!(packed, source, kernel.descriptor, transform)
pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_b!(packed, source, kernel.descriptor, transform)

# Lane-width checks shared by SIMDKernel/PlanarKernel/OneMKernel. `name` only
# names the type in the message; construction-time only, never hot.
@inline function _check_lanewidth(name, W)
    W isa Int && W > 0 ||
        throw(ArgumentError("$name requires an Int vector width W > 0, got W = $W"))
    return nothing
end

@inline function _check_mr_multiple(name, MR::Int, W::Int)
    mod(MR, W) == 0 || throw(
        ArgumentError(
            "$name requires mr(kernel) = $MR to be a multiple of " *
                "the vector width W = $W"
        )
    )
    return nothing
end

"""
    zero_accumulator(kernel::ScalarKernel{MR,NR,T}) -> Matrix{T}

Return a logical `MR`-by-`NR` zero accumulator tile, `acc[i+1,j+1] == 0`
for every zero-based `(i,j)` in `0:MR-1 x 0:NR-1`.
"""
zero_accumulator(kernel::ScalarKernel{MR, NR, T}) where {MR, NR, T} = zeros(T, MR, NR)

"""
    accumulate(kernel::ScalarKernel, acc, packed_a, packed_b, kc::Int) -> acc

Extends `Base.accumulate` (avoids a name collision with the also-exported
`Base.accumulate` under `using QuasiStrided`), not type piracy. Updates
`acc[i,j] += sum_p Ap[i,p]*Bp[j,p]` in place over `kc` K-steps; `kc == 0`
returns `acc` unchanged without reading the packed buffers.
"""
function Base.accumulate(
        kernel::ScalarKernel{MR, NR, T}, acc::AbstractMatrix{T},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, PA, PB}
    kc == 0 && return acc
    kc > 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    @inbounds for p in 0:(kc - 1)
        for j in 0:(NR - 1)
            bj = panel_load(packed_b, packed_b_offset(kernel, j, p))
            for i in 0:(MR - 1)
                ai = panel_load(packed_a, packed_a_offset(kernel, i, p))
                acc[i + 1, j + 1] = muladd(ai, bj, acc[i + 1, j + 1])
            end
        end
    end
    return acc
end

"""
    scale_tile!(destination::QSTile{T}, beta::T) -> destination

Apply `C[i,j] = beta * C[i,j]` over the destination's valid rectangle.
`beta == 0` writes `zero(T)` without reading old `C`; `beta == 1` is a
no-op; an empty destination is a no-op in every branch.
"""
function scale_tile!(destination::QSTile, beta::T) where {T}
    m = nrows(destination)
    n = ncols(destination)
    (m == 0 || n == 0) && return destination
    if isone(beta)
        return destination
    elseif iszero(beta)
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            tile_store!(destination, i, j, zero(T))
        end
    else
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            tile_store!(destination, i, j, tile_load(destination, i, j) * beta)
        end
    end
    return destination
end

# `C = alpha*r + beta*C` at one element, in the two forms the kernels need.
# Ternaries, so `beta == 0` never reads the old value (contract pinned by the
# nonfinite-poisoning tests).
@inline _axpby_tile!(dest, i::Int, j::Int, alpha, r, beta) = tile_store!(
    dest, i, j,
    iszero(beta) ? alpha * r :
        isone(beta) ? muladd(alpha, r, tile_load(dest, i, j)) :
        muladd(alpha, r, beta * tile_load(dest, i, j))
)

@inline _axpby_at!(storage, idx::Int, alpha, r, beta) = @inbounds storage[idx] =
    iszero(beta) ? alpha * r :
    isone(beta) ? muladd(alpha, r, storage[idx]) :
    muladd(alpha, r, beta * storage[idx])

# ----------------------------------------------------------------------------
# Validation prologues shared by all four kernels' store_tile!/execute_tile!
# ----------------------------------------------------------------------------

# `store_tile!`'s alpha/beta preamble: an empty destination is a no-op, and
# `alpha == 0` degenerates to `scale_tile!`. Returns the `(m, n)` extent to
# store over, or `(0, 0)` when the store is already finished.
@inline function _store_prologue!(destination::QSTile, alpha, beta)
    m = nrows(destination)
    n = ncols(destination)
    if m == 0 || n == 0
        return (0, 0)
    elseif iszero(alpha)
        scale_tile!(destination, beta)
        return (0, 0)
    end
    return (m, n)
end

@noinline _throw_packed_short(which::Symbol, got::Int, need::Int, kc::Int) = throw(
    DimensionMismatch(
        "execute_tile!: packed_$which has length $got, " *
            "need at least packed_$(which)_length(kernel, kc=$kc) = $need"
    )
)

# `execute_tile!`'s validation sequence, in the order all four kernels share:
# destination extent vs. kernel shape, `kc >= 0`, the alpha/beta converts, the
# empty short-circuit, storage bounds BEFORE any `@inbounds` path (Phase 2b),
# the `kc == 0 || alpha == 0` beta-only branch, then both packed capacities.
# Returns `(run, alphaT, betaT)`; `run == false` means the call is finished and
# the caller returns `destination` untouched.
#
# GUARDRAIL: `@inline`, and one bound type parameter per argument, for the
# reason spelled out at `pack_a!` above. This is on the hot path.
@inline function _execute_tile_prologue!(
        kernel::K, destination::QSTile, packed_a::PA, packed_b::PB,
        kc::Int, alpha, beta
    ) where {MR, NR, T, K <: DescriptorKernel{MR, NR, T}, PA, PB}
    m = nrows(destination)
    n = ncols(destination)
    m <= MR || throw(ArgumentError("destination valid row extent $m exceeds mr(kernel) = $MR"))
    n <= NR || throw(ArgumentError("destination valid column extent $n exceeds nr(kernel) = $NR"))
    kc >= 0 || throw(ArgumentError("execute_tile! requires kc >= 0, got kc = $kc"))

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    (m == 0 || n == 0) && return (false, alphaT, betaT)

    checked_tile_storage_bounds(destination)

    if kc == 0 || iszero(alphaT)
        scale_tile!(destination, betaT)
        return (false, alphaT, betaT)
    end

    need_a = packed_a_length(kernel, kc)
    length(packed_a) >= need_a || _throw_packed_short(:a, length(packed_a), need_a, kc)
    need_b = packed_b_length(kernel, kc)
    length(packed_b) >= need_b || _throw_packed_short(:b, length(packed_b), need_b, kc)

    return (true, alphaT, betaT)
end

"""
    store_tile!(destination::QSTile{T}, acc, alpha::T, beta::T, kernel::ScalarKernel) -> destination

`C[i,j] = alpha*R[i,j] + beta*C[i,j]` over the valid rectangle only, with
BLAS-like shortcuts: `alpha == 0` never reads `acc`; `beta == 0` never reads
old `C`; `beta == 1` skips the multiplication. Padding lanes (`i>=m`/`j>=n`)
are never read, so nonfinite padding in `acc` cannot propagate.
"""
function store_tile!(
        destination::QSTile, acc::AbstractMatrix{T},
        alpha::T, beta::T, kernel::ScalarKernel
    ) where {T}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination
    @inbounds for j in 0:(n - 1), i in 0:(m - 1)
        _axpby_tile!(destination, i, j, alpha, acc[i + 1, j + 1], beta)
    end
    return destination
end

"""
    execute_tile!(kernel::ScalarKernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

Checked composition of `zero_accumulator`, `accumulate` and `store_tile!`
for a single K-panel call (multi-panel accumulation is the driver's job).
Validation order and short-circuits are `_execute_tile_prologue!`'s, shared
with every other kernel: `kc == 0` or `alpha == 0` scales by `beta` only,
without reading `packed_a`/`packed_b`.
"""
function execute_tile!(
        kernel::ScalarKernel{MR, NR, T}, destination::QSTile,
        packed_a::PA, packed_b::PB, kc::Int, alpha, beta
    ) where {MR, NR, T, PA, PB}
    run, alphaT, betaT =
        _execute_tile_prologue!(kernel, destination, packed_a, packed_b, kc, alpha, beta)
    run || return destination
    acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end
