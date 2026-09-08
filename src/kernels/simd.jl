# Explicit-SIMD execute_tile! candidate (SIMD.jl's Vec{N,T}), matching
# ScalarKernel's packed-format contract and API. Accumulator is an immutable
# tuple of Vec{W,T} (not a heap array) for register residency; the K-step
# body is a @generated, closure-free function so `acc = _accumulate_step(...)`
# is a plain reassignment that never boxes.

using SIMD: Vec, vload, vstore

"""
    SIMDKernel{MR,NR,T,W}(descriptor::KernelDescriptor{MR,NR,T})
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Explicit-SIMD microkernel using `SIMD.Vec{W,T}` lanes. Wraps a
`KernelDescriptor` and adds vector width `W`; `MR` must be a multiple of `W`
(checked at construction). `NR` need not be — B contributes one scalar per
column per K step. The 3-argument form defaults `W` via
[`_default_lanewidth`](@ref).
"""
struct SIMDKernel{MR,NR,T,W} <: DescriptorKernel{MR,NR,T}
    descriptor::KernelDescriptor{MR,NR,T}

    function SIMDKernel{MR,NR,T,W}(descriptor::KernelDescriptor{MR,NR,T}) where {MR,NR,T,W}
        W isa Int && W > 0 ||
            throw(ArgumentError("SIMDKernel requires an Int vector width W > 0, got W = $W"))
        mod(MR, W) == 0 ||
            throw(ArgumentError("SIMDKernel requires mr(kernel) = $MR to be a multiple of " *
                                 "the vector width W = $W"))
        return new{MR,NR,T,W}(descriptor)
    end
end

"""
    _default_lanewidth(::Type{T}) -> Int

Default `SIMD.Vec` lane count for `T` (one 256-bit register's worth): 4 for
`Float64`, 8 for `Float32`. Not hardware-detected or tuned.
"""
_default_lanewidth(::Type{Float64}) = 4
_default_lanewidth(::Type{Float32}) = 8

function SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR,NR,T,W}
    return SIMDKernel{MR,NR,T,W}(KernelDescriptor(Val(MR), Val(NR), T))
end
function SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR,NR,T}
    return SIMDKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(T)))
end

"""
    lanewidth(kernel::SIMDKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`.
"""
lanewidth(::SIMDKernel{MR,NR,T,W}) where {MR,NR,T,W} = W

"""
    avecs_per_column(kernel::SIMDKernel) -> Int

`mr(kernel) ÷ lanewidth(kernel)`: the number of full-width A vectors (and
thus accumulator vectors) per output column.
"""
avecs_per_column(::SIMDKernel{MR,NR,T,W}) where {MR,NR,T,W} = MR ÷ W

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::SIMDKernel{MR,NR,T,W}) -> NTuple{NV,SIMD.Vec{W,T}}

Return a logical `MR`-by-`NR` zero accumulator, represented as an immutable
tuple of `NV = (MR÷W)*NR` zero `Vec{W,T}` values (design doc section 8: "an
ordinary heap array of accumulators is not the intended fast path"). Entry
`(v, j)` (`v` the row-vector index in `0:MR÷W-1`, `j` the output column in
`0:NR-1`) lives at 1-based tuple position `v + (MR÷W)*j + 1`; physical row
`i` of that vector is lane `i - v*W + 1` (1-based `SIMD.Vec` indexing).
"""
function zero_accumulator(kernel::SIMDKernel{MR,NR,T,W}) where {MR,NR,T,W}
    NVECA = MR ÷ W
    z = zero(Vec{W,T})
    return ntuple(_ -> z, Val(NVECA * NR))
end

# Fully unrolled, closure-free K-step body: one vector load per A row-vector,
# one scalar load per B column, NVECA*NR FMAs, generated as straight-line code.
@generated function _accumulate_step(kernel::SIMDKernel{MR,NR,T,W}, acc::NTuple{NV,Vec{W,T}},
                                      packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
                                      p::Int) where {MR,NR,T,W,NV}
    NVECA = MR ÷ W
    NVECA * NR == NV ||
        throw(ArgumentError("_accumulate_step: accumulator length $NV does not match " *
                             "(mr÷W)*nr = $(NVECA * NR) for MR=$MR, NR=$NR, W=$W"))

    avars = [Symbol(:a, v) for v in 0:(NVECA - 1)]
    bvars = [Symbol(:b, j) for j in 0:(NR - 1)]

    load_a = [:( $(avars[v + 1]) = vload(Vec{$W,$T}, packed_a, packed_a_offset(kernel, $(v * W), p) + 1) )
              for v in 0:(NVECA - 1)]
    load_b = [:( $(bvars[j + 1]) = packed_b[packed_b_offset(kernel, $j, p) + 1] )
              for j in 0:(NR - 1)]

    acc_exprs = Vector{Any}(undef, NV)
    for j in 0:(NR - 1), v in 0:(NVECA - 1)
        idx = v + NVECA * j + 1
        acc_exprs[idx] = :( muladd($(avars[v + 1]), $(bvars[j + 1]), acc[$idx]) )
    end

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(load_a...)
            $(load_b...)
            return $(Expr(:tuple, acc_exprs...))
        end
    end
end

"""
    accumulate(kernel::SIMDKernel, acc, packed_a, packed_b, kc::Int) -> acc

SIMD counterpart of `ScalarKernel`'s `accumulate`; same contract (`kc == 0`
is a no-op), but **not bitwise identical** (FMA grouping/order differ —
compare with a tolerance, never `==`). `packed_a` must support `SIMD.vload`
(a `Vector{T}` or unit-range `view`); `packed_b` is read by scalar `getindex`.
"""
function Base.accumulate(kernel::SIMDKernel{MR,NR,T,W}, acc::NTuple{NV,Vec{W,T}},
                          packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
                          kc::Int) where {MR,NR,T,W,NV}
    kc == 0 && return acc
    kc > 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    @inbounds for p in 0:(kc - 1)
        acc = _accumulate_step(kernel, acc, packed_a, packed_b, p)
    end
    return acc
end

# scale_tile! is reused as-is from src/kernel.jl (no kernel argument).

_unit_stride_rows(ax::AffineAxis) = ax.stride == 1
_unit_stride_rows(::ScatterAxis) = false

@inline function _acc_lane(acc::NTuple{NV,Vec{W,T}}, v::Int, j::Int, lane1::Int,
                            ::Val{NVECA}) where {NV,W,T,NVECA}
    return acc[v + NVECA * j + 1][lane1]
end

"""
    store_tile!(destination::QSTile, acc::NTuple{NV,SIMD.Vec{W,T}}, alpha::T, beta::T, kernel::SIMDKernel) -> destination

SIMD counterpart of `ScalarKernel`'s `store_tile!`; same contract and
alpha/beta shortcuts. Fast path: unit-stride `AffineAxis` rows into a plain
`Vector{T}` get vector load/store for whole `W`-row blocks (scalar tail for
the remainder, never over-reading past the valid rectangle). Otherwise
falls back to the scalar path, one lane at a time. Empty destination is a
no-op.
"""
function store_tile!(destination::QSTile, acc::NTuple{NV,Vec{W,T}},
                      alpha::T, beta::T, kernel::SIMDKernel{MR,NR,T,W}) where {MR,NR,T,W,NV}
    m = nrows(destination)
    n = ncols(destination)
    (m == 0 || n == 0) && return destination

    if iszero(alpha)
        scale_tile!(destination, beta)
        return destination
    end

    NVECA = MR ÷ W
    nfull = m ÷ W  # whole W-row blocks entirely inside [0, m)

    if _unit_stride_rows(destination.rows) && destination.storage isa Vector{T}
        storage = destination.storage
        rows = destination.rows
        cols = destination.cols
        rowbase0 = destination.base + rows.base  # zero-based address at (i=0, j=0)'s row contribution
        @inbounds for j in 0:(n - 1)
            colbase = rowbase0 + axis_offset(cols, j)  # zero-based address of (i=0, j)
            for v in 0:(nfull - 1)
                idx = colbase + v * W + 1  # one-based storage index of row v*W
                rvec = acc[v + NVECA * j + 1]
                if iszero(beta)
                    outvec = alpha * rvec
                elseif isone(beta)
                    outvec = muladd(alpha, rvec, vload(Vec{W,T}, storage, idx))
                else
                    outvec = muladd(alpha, rvec, beta * vload(Vec{W,T}, storage, idx))
                end
                vstore(outvec, storage, idx)
            end
            for i in (nfull * W):(m - 1)
                idx = colbase + i + 1
                r = _acc_lane(acc, i ÷ W, j, (i % W) + 1, Val(NVECA))
                if iszero(beta)
                    storage[idx] = alpha * r
                elseif isone(beta)
                    storage[idx] = muladd(alpha, r, storage[idx])
                else
                    storage[idx] = muladd(alpha, r, beta * storage[idx])
                end
            end
        end
        return destination
    end

    @inbounds for j in 0:(n - 1), i in 0:(m - 1)
        r = _acc_lane(acc, i ÷ W, j, (i % W) + 1, Val(NVECA))
        if iszero(beta)
            tile_store!(destination, i, j, alpha * r)
        elseif isone(beta)
            tile_store!(destination, i, j, muladd(alpha, r, tile_load(destination, i, j)))
        else
            tile_store!(destination, i, j, muladd(alpha, r, beta * tile_load(destination, i, j)))
        end
    end
    return destination
end

"""
    execute_tile!(kernel::SIMDKernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

SIMD counterpart of `ScalarKernel`'s `execute_tile!`; same validation order
and short-circuits. Numerically matches `ScalarKernel` only to within a
tolerance (FMA grouping differs), never bitwise.
"""
function execute_tile!(kernel::SIMDKernel{MR,NR,T,W}, destination::QSTile,
                        packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
                        kc::Int, alpha, beta) where {MR,NR,T,W}
    m = nrows(destination)
    n = ncols(destination)
    m <= MR || throw(ArgumentError("destination valid row extent $m exceeds mr(kernel) = $MR"))
    n <= NR || throw(ArgumentError("destination valid column extent $n exceeds nr(kernel) = $NR"))
    kc >= 0 || throw(ArgumentError("execute_tile! requires kc >= 0, got kc = $kc"))

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    (m == 0 || n == 0) && return destination

    checked_tile_storage_bounds(destination)  # bounds before any unchecked path

    if kc == 0 || iszero(alphaT)
        scale_tile!(destination, betaT)
        return destination
    end

    length(packed_a) >= packed_a_length(kernel, kc) ||
        throw(DimensionMismatch("execute_tile!: packed_a has length $(length(packed_a)), " *
                                 "need at least packed_a_length(kernel, kc=$kc) = $(packed_a_length(kernel, kc))"))
    length(packed_b) >= packed_b_length(kernel, kc) ||
        throw(DimensionMismatch("execute_tile!: packed_b has length $(length(packed_b)), " *
                                 "need at least packed_b_length(kernel, kc=$kc) = $(packed_b_length(kernel, kc))"))

    acc = zero_accumulator(kernel)
    acc = accumulate(kernel, acc, packed_a, packed_b, kc)
    store_tile!(destination, acc, alphaT, betaT, kernel)
    return destination
end
