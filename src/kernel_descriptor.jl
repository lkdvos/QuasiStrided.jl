# OWNER: main process. Frozen Phase 2 coupling contract between the packing
# implementer (src/tiles.jl, src/packing.jl) and the scalar/SIMD kernel
# implementers (src/kernel.jl, src/kernels/simd.jl). Do not redefine
# KernelDescriptor or the packed-offset formulas anywhere else; extend this
# file (via the main process) if the contract needs to grow.
#
# See Julia-Microkernel-Tile-Interface-Design.md sections 5 and 7.

"""
    KernelDescriptor{MR,NR,T}

A microkernel's register-tile shape and scalar type. `MR`/`NR` are `Int`
type parameters (the logical output register-tile dimensions); `T` is the
scalar type used uniformly for source, packed values, accumulation, and
destination in one invocation (`Float32` or `Float64` for this milestone).

`KernelDescriptor` carries no runtime state — it exists purely to dispatch
packing/arithmetic to the right packed-offset formulas and to name `T`. A
concrete kernel (scalar or SIMD) is a *different* type that provides the
actual `zero_accumulator`/`accumulate`/`store_tile!`/`execute_tile!` methods;
`KernelDescriptor` is the shared, frozen shape/format description every such
kernel embeds or references.
"""
struct KernelDescriptor{MR,NR,T}
    function KernelDescriptor{MR,NR,T}() where {MR,NR,T}
        MR isa Int && NR isa Int ||
            throw(ArgumentError("KernelDescriptor requires Int type parameters MR, NR"))
        MR > 0 || throw(ArgumentError("KernelDescriptor requires MR > 0, got MR = $MR"))
        NR > 0 || throw(ArgumentError("KernelDescriptor requires NR > 0, got NR = $NR"))
        T === Float32 || T === Float64 ||
            throw(ArgumentError("KernelDescriptor requires T ∈ (Float32, Float64) for this milestone, got $T"))
        return new{MR,NR,T}()
    end
end

KernelDescriptor(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR,NR,T} = KernelDescriptor{MR,NR,T}()

mr(::KernelDescriptor{MR}) where {MR} = MR
nr(::KernelDescriptor{MR,NR}) where {MR,NR} = NR
scalartype(::KernelDescriptor{MR,NR,T}) where {MR,NR,T} = T

"""
    packed_a_offset(kernel::KernelDescriptor, i::Int, p::Int) -> Int

Zero-based packed offset of A's logical entry `(i, p)` (row `i` in
`0:mr(kernel)-1`, K step `p`) in a physical panel of shape `(MR, kc)`:
`i + MR*p`. Consecutive K steps consume consecutive rows of A.
"""
packed_a_offset(kernel::KernelDescriptor{MR}, i::Int, p::Int) where {MR} = i + MR * p

"""
    packed_b_offset(kernel::KernelDescriptor, j::Int, p::Int) -> Int

Zero-based packed offset of B's logical entry `(j, p)` (column `j` in
`0:nr(kernel)-1`, K step `p`) in a physical panel of shape `(NR, kc)`:
`j + NR*p`. This is deliberately *not* a column-major `(kc, NR)` layout.
"""
packed_b_offset(kernel::KernelDescriptor{MR,NR}, j::Int, p::Int) where {MR,NR} = j + NR * p

"""
    packed_a_length(kernel, kc::Int) -> Int
    packed_b_length(kernel, kc::Int) -> Int

Minimum packed-buffer capacity (in elements) needed to hold a full A or B
panel of K-depth `kc` for this kernel shape. `kc == 0` is valid and yields 0;
callers must still ensure any *reused* buffer has at least this many valid
elements for `kc > 0`. Uses `Int` arithmetic only; `MR`, `NR`, `kc` are all
expected to be small enough in this milestone that `MR*kc`/`NR*kc` do not
need overflow checking beyond ordinary `Int` bounds (no checked-arithmetic
requirement is imposed here, unlike the indexing layer's offset bounds).
"""
packed_a_length(kernel::KernelDescriptor{MR}, kc::Int) where {MR} = MR * kc
packed_b_length(kernel::KernelDescriptor{MR,NR}, kc::Int) where {MR,NR} = NR * kc
