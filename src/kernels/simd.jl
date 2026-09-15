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
struct SIMDKernel{MR, NR, T, W} <: DescriptorKernel{MR, NR, T}
    descriptor::KernelDescriptor{MR, NR, T}

    function SIMDKernel{MR, NR, T, W}(descriptor::KernelDescriptor{MR, NR, T}) where {MR, NR, T, W}
        _check_lanewidth("SIMDKernel", W)
        _check_mr_multiple("SIMDKernel", MR, W)
        return new{MR, NR, T, W}(descriptor)
    end
end

"""
    _default_lanewidth(::Type{T}) -> Int

Default `SIMD.Vec` lane count for `T` (one 256-bit register's worth): 4 for
`Float64`, 8 for `Float32`. Not hardware-detected or tuned.
"""
_default_lanewidth(::Type{Float64}) = 4
_default_lanewidth(::Type{Float32}) = 8

SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W} =
    SIMDKernel{MR, NR, T, W}(KernelDescriptor(Val(MR), Val(NR), T))
SIMDKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} =
    SIMDKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(T)))

"""
    lanewidth(kernel::SIMDKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`.
"""
lanewidth(::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::SIMDKernel) -> Int

`mr(kernel) ÷ lanewidth(kernel)`: the number of full-width A vectors (and
thus accumulator vectors) per output column.
"""
avecs_per_column(::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W} = MR ÷ W

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
function zero_accumulator(kernel::SIMDKernel{MR, NR, T, W}) where {MR, NR, T, W}
    z = zero(Vec{W, T})
    return ntuple(_ -> z, Val((MR ÷ W) * NR))
end

# Fully unrolled, closure-free K-step body: one vector load per A row-vector,
# one scalar load per B column, NVECA*NR FMAs, generated as straight-line code.
@generated function _accumulate_step(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, p::Int
    ) where {MR, NR, T, W, NV, PA, PB}
    NVECA = MR ÷ W
    NVECA * NR == NV ||
        throw(
        ArgumentError(
            "_accumulate_step: accumulator length $NV does not match " *
                "(mr÷W)*nr = $(NVECA * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )

    avars = [Symbol(:a, v) for v in 0:(NVECA - 1)]
    bvars = [Symbol(:b, j) for j in 0:(NR - 1)]

    load_a = [
        :($(avars[v + 1]) = panel_vload(Vec{$W, $T}, packed_a, packed_a_offset(kernel, $(v * W), p)))
            for v in 0:(NVECA - 1)
    ]
    load_b = [
        :($(bvars[j + 1]) = panel_load(packed_b, packed_b_offset(kernel, $j, p)))
            for j in 0:(NR - 1)
    ]

    acc_exprs = Vector{Any}(undef, NV)
    for j in 0:(NR - 1), v in 0:(NVECA - 1)
        idx = v + NVECA * j + 1
        acc_exprs[idx] = :(muladd($(avars[v + 1]), $(bvars[j + 1]), acc[$idx]))
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
compare with a tolerance, never `==`). `packed_a`/`packed_b` may be a
[`PackedPanel`](@ref) — what the driver passes, and the only form that keeps
a large accumulator register-resident — or any contiguous `AbstractVector{T}`
such as a `Vector` or unit-range `view`.
"""
function Base.accumulate(
        kernel::SIMDKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, T}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, NV, PA, PB}
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
_unit_stride_rows(::PtrScatterAxis) = false

@inline function _acc_lane(
        acc::NTuple{NV, Vec{W, T}}, v::Int, j::Int, lane1::Int,
        ::Val{NVECA}
    ) where {NV, W, T, NVECA}
    return acc[v + NVECA * j + 1][lane1]
end

# Scalar/scattered store path.
#
# GUARDRAIL (Cliff B): every `acc[...]` here must be a *literal* tuple index,
# which is why this is `@generated` and unrolled over `(v, j)` rather than a
# plain `for j, i` loop. Indexing an `NTuple` dynamically forces the whole
# tuple to memory, and above NV = 16 the compiler heap-allocates it: measured
# 24576 B per `execute!` on the 3-index scattered fixture at
# (MR,NR,W) = (32,6,8), against 0 B at (16,6,8) (docs/decisions.md, Phase H).
# Scattered destinations are this engine's reason to exist, so that silently
# capped the usable register tile on exactly the workload that matters. Only
# the lane index inside a single `Vec` may be a runtime value.
@generated function _store_tile_scattered!(
        destination::QSTile, acc::NTuple{NV, Vec{W, T}},
        alpha::T, beta::T, kernel::SIMDKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, NV}
    NVECA = MR ÷ W
    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(NVECA - 1)
        idx = v + NVECA * j + 1
        push!(
            blocks, quote
                if $j < n
                    vec = acc[$idx]
                    for lane in 1:$W
                        i = $(v * W) + lane - 1
                        i < m || break
                        _axpby_tile!(destination, i, $j, alpha, vec[lane], beta)
                    end
                end
            end
        )
    end
    return quote
        @inbounds begin
            $(blocks...)
        end
        return destination
    end
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
function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, T}},
        alpha::T, beta::T, kernel::SIMDKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, NV}
    m, n = _store_prologue!(destination, alpha, beta)
    (m == 0 || n == 0) && return destination

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
                vstore(
                    iszero(beta) ? alpha * rvec :
                        isone(beta) ? muladd(alpha, rvec, vload(Vec{W, T}, storage, idx)) :
                        muladd(alpha, rvec, beta * vload(Vec{W, T}, storage, idx)),
                    storage, idx
                )
            end
            for i in (nfull * W):(m - 1)
                _axpby_at!(
                    storage, colbase + i + 1, alpha,
                    _acc_lane(acc, i ÷ W, j, (i % W) + 1, Val(NVECA)), beta
                )
            end
        end
        return destination
    end

    return _store_tile_scattered!(destination, acc, alpha, beta, kernel, m, n)
end

"""
    execute_tile!(kernel::SIMDKernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

SIMD counterpart of `ScalarKernel`'s `execute_tile!`; same validation order
and short-circuits (both are `_execute_tile_prologue!`'s). Numerically matches
`ScalarKernel` only to within a tolerance (FMA grouping differs), never
bitwise.
"""
function execute_tile!(
        kernel::SIMDKernel{MR, NR, T, W}, destination::QSTile,
        packed_a::PA, packed_b::PB, kc::Int, alpha, beta
    ) where {MR, NR, T, W, PA, PB}
    run, alphaT, betaT =
        _execute_tile_prologue!(kernel, destination, packed_a, packed_b, kc, alpha, beta)
    run || return destination
    acc = accumulate(kernel, zero_accumulator(kernel), packed_a, packed_b, kc)
    return store_tile!(destination, acc, alphaT, betaT, kernel)
end
