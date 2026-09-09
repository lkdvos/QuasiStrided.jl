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
# `packed` and `kernel` (via `K`) each carry their own free type parameter,
# alongside `transform`'s `F` — a forwarding parameter left unbound here is
# exactly the Phase 2b finding-5 recurrence site (docs/decisions.md).
pack_a!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {T, V <: AbstractVector{T}, MR, NR, K <: DescriptorKernel{MR, NR, T}, F} =
    pack_a!(packed, source, kernel.descriptor, transform)
pack_b!(
    packed::V, source::QSTile, kernel::K, transform::F
) where {T, V <: AbstractVector{T}, MR, NR, K <: DescriptorKernel{MR, NR, T}, F} =
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
        packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
        kc::Int
    ) where {MR, NR, T}
    kc == 0 && return acc
    kc > 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    @inbounds for p in 0:(kc - 1)
        for j in 0:(NR - 1)
            bj = packed_b[packed_b_offset(kernel, j, p) + 1]
            for i in 0:(MR - 1)
                ai = packed_a[packed_a_offset(kernel, i, p) + 1]
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

    if iszero(beta)
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            tile_store!(destination, i, j, alpha * acc[i + 1, j + 1])
        end
    elseif isone(beta)
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            c = tile_load(destination, i, j)
            tile_store!(destination, i, j, muladd(alpha, acc[i + 1, j + 1], c))
        end
    else
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            c = tile_load(destination, i, j)
            tile_store!(destination, i, j, muladd(alpha, acc[i + 1, j + 1], beta * c))
        end
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
        packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
        kc::Int, alpha, beta
    ) where {MR, NR, T}
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
