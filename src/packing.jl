# OWNER: packing implementer (Phase 2). See docs/decisions.md and
# Julia-Microkernel-Tile-Interface-Design.md sections 5-6.
#
# Implements: pack_a!, pack_b!, against the frozen packed physical formats
# from src/kernel_descriptor.jl (A: i + MR*p, B: j + NR*p; do not redefine).

# ----------------------------------------------------------------------------
# Shared copy machinery
# ----------------------------------------------------------------------------

# Validate that `packed`'s element type matches the kernel's declared scalar
# type before any write. Kept as an explicit runtime check (rather than
# forcing it through dispatch with a single shared type parameter) so a
# mismatch raises a clear ArgumentError instead of a bare MethodError.
@inline function _check_packed_eltype(packed::Vector{T1}, kernel::KernelDescriptor{MR,NR,T2}) where {T1,MR,NR,T2}
    T1 === T2 ||
        throw(ArgumentError("packed buffer eltype $T1 does not match kernel scalar type $T2"))
    return nothing
end

# Shared inner loop for both pack_a! and pack_b!. `physical_dim` is MR (for A)
# or NR (for B); `valid` is the source's valid M or N count (m or n);
# `load` and `packed_offset` close over the operand-specific index mapping
# and packed-offset formula (kernel_descriptor.jl's packed_a_offset /
# packed_b_offset). `kc` is the logical K depth; kc == 0 is handled by the
# caller before this is ever invoked (no reads, no writes).
#
# Loop order (p outer, i/j inner) matches the packed formats' physical
# layout exactly: for fixed p, incrementing the inner index by one advances
# the packed offset by exactly one, so writes to `packed` are sequential.
@inline function _pack_panel!(packed::Vector{T}, physical_dim::Int, kc::Int, valid::Int,
                               transform::F, load::L, packed_offset::P) where {T,F,L,P}
    @inbounds for p in 0:(kc-1)
        for i in 0:(physical_dim-1)
            v = i < valid ? convert(T, transform(load(i, p)))::T : zero(T)
            packed[packed_offset(i, p)+1] = v
        end
    end
    return packed
end

# ----------------------------------------------------------------------------
# pack_a!
# ----------------------------------------------------------------------------

"""
    pack_a!(packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T}, transform) -> packed

Pack an A source tile into `packed`, a caller-supplied, reused buffer, using
the physical format frozen in `kernel_descriptor.jl`: logical entry `(i,p)`
(row `i` in `0:mr(kernel)-1`, K step `p`) lands at zero-based offset
`packed_a_offset(kernel, i, p) == i + mr(kernel)*p`.

`source` must have `0 <= m <= mr(kernel)` rows (`m = nrows(source)`) and
`kc = ncols(source)` columns, where `kc` is the logical K depth for this
call (there is no separate `kc` argument: it is read from `source`'s shape).
`packed` must have `length(packed) >= packed_a_length(kernel, kc)`.

For every K step `p` in `0:kc-1` and every physical row `i` in
`0:mr(kernel)-1`:
- if `i < m`: writes `convert(T, transform(A[i,p]))`, reading `A[i,p]` from
  `source` and calling `transform` exactly once;
- otherwise (a padding row): writes `zero(T)` directly, **without** reading
  `source` or calling `transform`.

`kc == 0` writes nothing and reads nothing. Buffer entries at positions
beyond `packed_a_length(kernel, kc)` (the declared physical panel extent)
are left untouched. All shape/capacity/eltype validation happens before any
write (`ArgumentError`/`DimensionMismatch` on failure, buffer never
partially mutated by a rejected call). `transform` must be a pure,
elementwise callable; exceptions it raises are not rolled back.

Never allocates: `packed` is written in place and returned.
"""
function pack_a!(packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T2},
                  transform::F) where {T,MR,NR,T2,F}
    _check_packed_eltype(packed, kernel)

    m = nrows(source)
    kc = ncols(source)

    (0 <= m <= MR) ||
        throw(ArgumentError("pack_a!: source row count m=$m must satisfy 0 <= m <= mr(kernel)=$MR"))
    kc >= 0 || throw(ArgumentError("pack_a!: source column count (kc) must be nonnegative, got $kc"))

    needed = packed_a_length(kernel, kc)
    length(packed) >= needed ||
        throw(DimensionMismatch("pack_a!: packed buffer has length $(length(packed)), " *
                                 "need at least packed_a_length(kernel, kc=$kc) = $needed"))

    kc == 0 && return packed

    # Fable review (Phase 2b) follow-up: validate reachable storage bounds
    # before entering the unchecked @inbounds load loop (spec section 6).
    checked_tile_storage_bounds(source)

    load = (i, p) -> tile_load(source, i, p)
    packed_offset = (i, p) -> packed_a_offset(kernel, i, p)
    _pack_panel!(packed, MR, kc, m, transform, load, packed_offset)
    return packed
end

# ----------------------------------------------------------------------------
# pack_b!
# ----------------------------------------------------------------------------

"""
    pack_b!(packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T}, transform) -> packed

Pack a B source tile into `packed`, a caller-supplied, reused buffer, using
the physical format frozen in `kernel_descriptor.jl`: logical entry `(j,p)`
(column `j` in `0:nr(kernel)-1`, K step `p`) lands at zero-based offset
`packed_b_offset(kernel, j, p) == j + nr(kernel)*p`. This is deliberately
*not* a column-major `(kc, nr(kernel))` layout.

`source` must have `kc = nrows(source)` rows (the logical K depth for this
call, read from `source`'s shape — there is no separate `kc` argument) and
`0 <= n <= nr(kernel)` columns (`n = ncols(source)`). `packed` must have
`length(packed) >= packed_b_length(kernel, kc)`.

For every K step `p` in `0:kc-1` and every physical column `j` in
`0:nr(kernel)-1`:
- if `j < n`: writes `convert(T, transform(B[p,j]))`, reading `B[p,j]` from
  `source` and calling `transform` exactly once;
- otherwise (a padding column): writes `zero(T)` directly, **without**
  reading `source` or calling `transform`.

`kc == 0` writes nothing and reads nothing. Buffer entries at positions
beyond `packed_b_length(kernel, kc)` are left untouched. All
shape/capacity/eltype validation happens before any write, and `transform`
must be pure and elementwise, exactly as in [`pack_a!`](@ref).

Never allocates: `packed` is written in place and returned.
"""
function pack_b!(packed::Vector{T}, source::QSTile, kernel::KernelDescriptor{MR,NR,T2},
                  transform::F) where {T,MR,NR,T2,F}
    _check_packed_eltype(packed, kernel)

    kc = nrows(source)
    n = ncols(source)

    kc >= 0 || throw(ArgumentError("pack_b!: source row count (kc) must be nonnegative, got $kc"))
    (0 <= n <= NR) ||
        throw(ArgumentError("pack_b!: source column count n=$n must satisfy 0 <= n <= nr(kernel)=$NR"))

    needed = packed_b_length(kernel, kc)
    length(packed) >= needed ||
        throw(DimensionMismatch("pack_b!: packed buffer has length $(length(packed)), " *
                                 "need at least packed_b_length(kernel, kc=$kc) = $needed"))

    kc == 0 && return packed

    # Fable review (Phase 2b) follow-up: validate reachable storage bounds
    # before entering the unchecked @inbounds load loop (spec section 6).
    checked_tile_storage_bounds(source)

    # B[p,j] lives at tile row p (the K axis), tile column j (the N axis):
    # source.rows is the K interval, source.cols is the N interval.
    load = (j, p) -> tile_load(source, p, j)
    packed_offset = (j, p) -> packed_b_offset(kernel, j, p)
    _pack_panel!(packed, NR, kc, n, transform, load, packed_offset)
    return packed
end
