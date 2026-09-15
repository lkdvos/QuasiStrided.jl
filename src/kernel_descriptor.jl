# Frozen coupling contract between packing (tiles.jl/packing.jl) and the
# kernels (kernel.jl/kernels/simd.jl); do not redefine elsewhere.

# Register-tile validation shared by `KernelDescriptor` and
# `ComplexKernelDescriptor`. `name` only names the type in the message;
# construction-time only, never hot.
@inline function _check_reg_tile(name, MR, NR)
    MR isa Int && NR isa Int ||
        throw(ArgumentError("$name requires Int type parameters MR, NR"))
    MR > 0 || throw(ArgumentError("$name requires MR > 0, got MR = $MR"))
    NR > 0 || throw(ArgumentError("$name requires NR > 0, got NR = $NR"))
    return nothing
end

"""
    KernelDescriptor{MR,NR,T}

A microkernel's register-tile shape (`MR`x`NR`, `Int` type params) and
scalar type `T` (`Float32`/`Float64`). Stateless: exists to dispatch
packed-offset formulas and name `T`; concrete kernels wrap or reference one.
"""
struct KernelDescriptor{MR, NR, T}
    function KernelDescriptor{MR, NR, T}() where {MR, NR, T}
        _check_reg_tile("KernelDescriptor", MR, NR)
        T === Float32 || T === Float64 ||
            throw(ArgumentError("KernelDescriptor requires T ∈ (Float32, Float64) for this milestone, got $T"))
        return new{MR, NR, T}()
    end
end

KernelDescriptor(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T} = KernelDescriptor{MR, NR, T}()

mr(::KernelDescriptor{MR}) where {MR} = MR
nr(::KernelDescriptor{MR, NR}) where {MR, NR} = NR
scalartype(::KernelDescriptor{MR, NR, T}) where {MR, NR, T} = T

"""
    packed_a_offset(kernel::KernelDescriptor, i::Int, p::Int) -> Int

Zero-based packed offset of A's entry `(row i, K-step p)` in an `(MR, kc)`
panel: `i + MR*p`.
"""
packed_a_offset(kernel::KernelDescriptor{MR}, i::Int, p::Int) where {MR} = i + MR * p

"""
    packed_b_offset(kernel::KernelDescriptor, j::Int, p::Int) -> Int

Zero-based packed offset of B's entry `(col j, K-step p)` in an `(NR, kc)`
panel: `j + NR*p` (not column-major).
"""
packed_b_offset(kernel::KernelDescriptor{MR, NR}, j::Int, p::Int) where {MR, NR} = j + NR * p

# Shared supertype for kernels wrapping a KernelDescriptor as `.descriptor`;
# lets mr/nr/scalartype/packed_*/pack_a!/pack_b! forward once for all of them.
abstract type DescriptorKernel{MR, NR, T} end

"""
    packed_a_length(kernel, kc::Int) -> Int
    packed_b_length(kernel, kc::Int) -> Int

Minimum packed-buffer capacity (elements) for a full A/B panel at K-depth
`kc` (`kc == 0` yields 0).
"""
packed_a_length(kernel::KernelDescriptor{MR}, kc::Int) where {MR} = MR * kc
packed_b_length(kernel::KernelDescriptor{MR, NR}, kc::Int) where {MR, NR} = NR * kc
