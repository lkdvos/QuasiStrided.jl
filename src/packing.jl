# Packs against the frozen physical formats in kernel_descriptor.jl
# (A: i + MR*p, B: j + NR*p; do not redefine).

# Explicit runtime check (not dispatch) so a mismatch raises ArgumentError.
@inline function _check_packed_eltype(packed::Vector{T1}, kernel::KernelDescriptor{MR, NR, T2}) where {T1, MR, NR, T2}
    T1 === T2 ||
        throw(ArgumentError("packed buffer eltype $T1 does not match kernel scalar type $T2"))
    return nothing
end

# Shared inner loop for pack_a!/pack_b!; `load`/`packed_offset` close over the
# operand-specific index mapping. `kc == 0` is handled by the caller.
@inline function _pack_panel!(
        packed::Vector{T}, physical_dim::Int, kc::Int, valid::Int,
        transform::F, load::L, packed_offset::P
    ) where {T, F, L, P}
    @inbounds for p in 0:(kc - 1)
        for i in 0:(physical_dim - 1)
            v = i < valid ? convert(T, transform(load(i, p)))::T : zero(T)
            packed[packed_offset(i, p) + 1] = v
        end
    end
    return packed
end

"""
    pack_a!(packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T}, transform) -> packed

Pack an A source tile into `packed` (reused buffer) at
`packed_a_offset(kernel, i, p) == i + mr(kernel)*p`. `source` has
`0 <= m <= mr(kernel)` rows and `kc = ncols(source)` columns; `packed` needs
`length >= packed_a_length(kernel, kc)`. Row `i < m` writes
`convert(T, transform(A[i,p]))`; padding rows (`i >= m`) write `zero(T)`
without reading `source` or calling `transform`. `kc == 0` is a no-op. All
validation happens before any write. Never allocates.
"""
function pack_a!(
        packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {T, MR, NR, T2, F}
    _check_packed_eltype(packed, kernel)

    m = nrows(source)
    kc = ncols(source)

    (0 <= m <= MR) ||
        throw(ArgumentError("pack_a!: source row count m=$m must satisfy 0 <= m <= mr(kernel)=$MR"))
    kc >= 0 || throw(ArgumentError("pack_a!: source column count (kc) must be nonnegative, got $kc"))

    needed = packed_a_length(kernel, kc)
    length(packed) >= needed ||
        throw(
        DimensionMismatch(
            "pack_a!: packed buffer has length $(length(packed)), " *
                "need at least packed_a_length(kernel, kc=$kc) = $needed"
        )
    )

    kc == 0 && return packed

    checked_tile_storage_bounds(source)  # Phase 2b: bounds before @inbounds loop.

    load = (i, p) -> tile_load(source, i, p)
    packed_offset = (i, p) -> packed_a_offset(kernel, i, p)
    _pack_panel!(packed, MR, kc, m, transform, load, packed_offset)
    return packed
end

"""
    pack_b!(packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T}, transform) -> packed

Pack a B source tile into `packed` at `packed_b_offset(kernel, j, p) == j +
nr(kernel)*p` (not column-major). `source` has `kc = nrows(source)` rows and
`0 <= n <= nr(kernel)` columns; `packed` needs `length >=
packed_b_length(kernel, kc)`. Column `j < n` writes `convert(T,
transform(B[p,j]))`; padding columns write `zero(T)` without reading
`source`. Same validation/allocation contract as [`pack_a!`](@ref).
"""
function pack_b!(
        packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR, NR, T2},
        transform::F
    ) where {T, MR, NR, T2, F}
    _check_packed_eltype(packed, kernel)

    kc = nrows(source)
    n = ncols(source)

    kc >= 0 || throw(ArgumentError("pack_b!: source row count (kc) must be nonnegative, got $kc"))
    (0 <= n <= NR) ||
        throw(ArgumentError("pack_b!: source column count n=$n must satisfy 0 <= n <= nr(kernel)=$NR"))

    needed = packed_b_length(kernel, kc)
    length(packed) >= needed ||
        throw(
        DimensionMismatch(
            "pack_b!: packed buffer has length $(length(packed)), " *
                "need at least packed_b_length(kernel, kc=$kc) = $needed"
        )
    )

    kc == 0 && return packed

    checked_tile_storage_bounds(source)  # Phase 2b: bounds before @inbounds loop.

    load = (j, p) -> tile_load(source, p, j)  # source.rows=K, source.cols=N
    packed_offset = (j, p) -> packed_b_offset(kernel, j, p)
    _pack_panel!(packed, NR, kc, n, transform, load, packed_offset)
    return packed
end
