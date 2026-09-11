# Concrete scalar reference kernel: wraps KernelDescriptor and adds
# zero_accumulator/accumulate/store_tile!/execute_tile! against a QSTile.

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

function ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    return ScalarKernel(KernelDescriptor(Val(MR), Val(NR), T))
end

# Shared forwarding for any DescriptorKernel (ScalarKernel, SIMDKernel, ...).
mr(k::DescriptorKernel) = mr(k.descriptor)
nr(k::DescriptorKernel) = nr(k.descriptor)
scalartype(k::DescriptorKernel) = scalartype(k.descriptor)
packed_a_offset(k::DescriptorKernel, i::Int, p::Int) = packed_a_offset(k.descriptor, i, p)
packed_b_offset(k::DescriptorKernel, j::Int, p::Int) = packed_b_offset(k.descriptor, j, p)
packed_a_length(k::DescriptorKernel, kc::Int) = packed_a_length(k.descriptor, kc)
packed_b_length(k::DescriptorKernel, kc::Int) = packed_b_length(k.descriptor, kc)

# pack_a!/pack_b! dispatch on a bare KernelDescriptor; forward any wrapper.
# Every forwarded argument needs its own bound type parameter (`V`, `K`, `F`):
# leaving one unbound here reintroduces the Phase 2b finding-5 allocation.
# `V` is unconstrained rather than `<: AbstractVector{T}` so that a
# `PackedPanel` (src/panel.jl) forwards too; `T` comes from `K` instead.
pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_a!(packed, source, kernel.descriptor, transform)
pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {V, MR, NR, T, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_b!(packed, source, kernel.descriptor, transform)

"""
    zero_accumulator(kernel::ScalarKernel{MR,NR,T}) -> Matrix{T}

Return a logical `MR`-by-`NR` zero accumulator tile, `acc[i+1,j+1] == 0`
for every zero-based `(i,j)` in `0:MR-1 x 0:NR-1`.
"""
function zero_accumulator(kernel::ScalarKernel{MR, NR, T}) where {MR, NR, T}
    return zeros(T, MR, NR)
end

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

# `C = alpha*r + beta*C` at one element, in the three forms the kernels need.
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
    m = nrows(destination)
    n = ncols(destination)
    (m == 0 || n == 0) && return destination

    if iszero(alpha)
        scale_tile!(destination, beta)
        return destination
    end

    @inbounds for j in 0:(n - 1), i in 0:(m - 1)
        _axpby_tile!(destination, i, j, alpha, acc[i + 1, j + 1], beta)
    end
    return destination
end

"""
    execute_tile!(kernel::ScalarKernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

Checked composition of `zero_accumulator`, `accumulate` and `store_tile!`
for a single K-panel call (multi-panel accumulation is the driver's job).
Validates destination extent vs. kernel shape, storage bounds, and packed
buffer sizes before any mutation. `kc == 0` or `alpha == 0` short-circuits
to `beta` scaling only, without reading `packed_a`/`packed_b`.
"""
function execute_tile!(
        kernel::ScalarKernel{MR, NR, T}, destination::QSTile,
        packed_a::PA, packed_b::PB, kc::Int, alpha, beta
    ) where {MR, NR, T, PA, PB}
    m = nrows(destination)
    n = ncols(destination)
    m <= MR || throw(ArgumentError("destination valid row extent $m exceeds mr(kernel) = $MR"))
    n <= NR || throw(ArgumentError("destination valid column extent $n exceeds nr(kernel) = $NR"))
    kc >= 0 || throw(ArgumentError("execute_tile! requires kc >= 0, got kc = $kc"))

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    (m == 0 || n == 0) && return destination

    checked_tile_storage_bounds(destination)  # Phase 2b: bounds before @inbounds path.

    if kc == 0 || iszero(alphaT)
        scale_tile!(destination, betaT)
        return destination
    end

    length(packed_a) >= packed_a_length(kernel, kc) ||
        throw(
        DimensionMismatch(
            "execute_tile!: packed_a has length $(length(packed_a)), " *
                "need at least packed_a_length(kernel, kc=$kc) = $(packed_a_length(kernel, kc))"
        )
    )
    length(packed_b) >= packed_b_length(kernel, kc) ||
        throw(
        DimensionMismatch(
            "execute_tile!: packed_b has length $(length(packed_b)), " *
                "need at least packed_b_length(kernel, kc=$kc) = $(packed_b_length(kernel, kc))"
        )
    )

    acc = zero_accumulator(kernel)
    acc = accumulate(kernel, acc, packed_a, packed_b, kc)
    store_tile!(destination, acc, alphaT, betaT, kernel)
    return destination
end
