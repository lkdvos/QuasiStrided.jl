# Scalar reference microkernel.

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
