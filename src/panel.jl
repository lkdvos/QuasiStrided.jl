# Packed micro-panels are handed to the kernel as borrowed pointers, not
# `view`s: a `SubArray`'s address arithmetic evicts the accumulator above
# NV = 16 and costs ~4x (docs/decisions.md, Phase H). `SIMD.vload` already
# accepts a `Ptr{T}`, so this needs no dependency.

using SIMD: Vec, vload, vstore

"""
    PackedPanel{T}(ptr::Ptr{T}, len::Int)

`len` borrowed elements of `T` at `ptr`: one packed micro-panel. Does not keep
the memory alive — callers must `GC.@preserve` the owning buffer (`execute!`
does, once around the loop nest). `len` exists only for capacity checks; this
is not a bounds-checked array.
"""
struct PackedPanel{T}
    ptr::Ptr{T}
    len::Int
end

Base.length(panel::PackedPanel) = panel.len
Base.eltype(::PackedPanel{T}) where {T} = T
Base.eltype(::Type{PackedPanel{T}}) where {T} = T

"""
    packed_panel(buffer, first1::Int, len::Int) -> PackedPanel

Borrow `len` elements of contiguous `buffer` from one-based index `first1`.
"""
@inline packed_panel(buffer::AbstractVector{T}, first1::Int, len::Int) where {T} =
    PackedPanel{T}(pointer(buffer, first1), len)

# Panel element access. Offsets are ZERO-based, matching
# `packed_a_offset`/`packed_b_offset`. The `AbstractVector` methods keep every
# other caller (and the tests) working with `Vector`s and `view`s.
@inline panel_vload(::Type{Vec{W, T}}, p::PackedPanel{T}, o::Int) where {W, T} =
    vload(Vec{W, T}, p.ptr + sizeof(T) * o)
@inline panel_vload(::Type{Vec{W, T}}, v::AbstractVector{T}, o::Int) where {W, T} =
    vload(Vec{W, T}, v, o + 1)

@inline panel_load(p::PackedPanel{T}, o::Int) where {T} = unsafe_load(p.ptr + sizeof(T) * o)
@inline panel_load(v::AbstractVector, o::Int) = @inbounds v[o + 1]

@inline panel_store!(p::PackedPanel{T}, o::Int, x::T) where {T} =
    unsafe_store!(p.ptr + sizeof(T) * o, x)
@inline panel_store!(v::AbstractVector{T}, o::Int, x::T) where {T} = @inbounds v[o + 1] = x
