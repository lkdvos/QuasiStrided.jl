# OWNER: scalar implementer (Phase 2). See docs/decisions.md and
# Julia-Microkernel-Tile-Interface-Design.md sections 7-9.
#
# Implements: a concrete scalar kernel type, zero_accumulator, accumulate,
# store_tile!, execute_tile! (scalar reference), plus scale_tile! and a
# minimal self-contained destination representation.
#
# --- Relationship to the frozen KernelDescriptor contract (src/kernel_descriptor.jl) ---
#
# `KernelDescriptor{MR,NR,T}` (main-process-owned, not edited here) carries
# only the register-tile shape/scalar type and the packed-offset formulas. It
# has no accumulator representation and no execute_tile! method of its own.
# `ScalarKernel` below wraps one `KernelDescriptor` and adds exactly that:
# the scalar reference's accumulator representation (an ordinary
# `Matrix{T}`) and the zero_accumulator/accumulate/store_tile!/execute_tile!
# methods. `mr`, `nr`, `scalartype`, `packed_a_offset`, `packed_b_offset`,
# `packed_a_length`, `packed_b_length` are forwarded so downstream code can
# treat `ScalarKernel` uniformly with any other concrete kernel type (e.g. a
# future SIMD kernel) without caring which one it holds.
#
# --- Destination representation ---
#
# Section 4 of the design doc specifies `DestinationTile(storage, base, rows,
# cols)` with per-axis addressing that is either affine (`base + t*stride`)
# or scattered (`offsets[t+1]`), and section 3 fixes the combined address as
# `base + row_offset(i) + col_offset(j)` (zero-based logical coordinates,
# one-based Julia storage access). `src/tiles.jl` (owned by a different,
# concurrently-running worker) is expected to eventually provide a concrete
# `DestinationTile`/`AffineAxis`/`ScatterAxis` implementing that contract.
# This file does not depend on that implementation landing: it defines its
# own minimal destination type, `ScalarDestination`, generalizing "affine or
# scattered" row/col addressing to a plain `Int -> Int` closure per axis
# (zero-based local coordinate t -> zero-based address contribution). This
# is a strict generalization of both AffineAxis (`t -> base + t*stride`) and
# ScatterAxis (`t -> offsets[t+1]`), so any admissible `DestinationTile` can
# be adapted to a `ScalarDestination` (or vice versa) by wrapping its row/col
# accessors as closures — the main process can reconcile the two names at
# integration time; the addressing *semantics* below match the spec exactly.
# Deliberately does not reuse the names `AffineAxis`/`ScatterAxis`/
# `DestinationTile`, which belong to `src/tiles.jl`.

"""
    ScalarKernel(descriptor::KernelDescriptor{MR,NR,T})
    ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Scalar reference microkernel for register-tile shape `(MR, NR)` and scalar
type `T`. Wraps a `KernelDescriptor` (the frozen shape/packed-format
contract) and adds the scalar accumulator representation: `zero_accumulator`
returns an ordinary `Matrix{T}` of size `(MR, NR)`, indexed
`acc[i+1, j+1]` for zero-based logical coordinates `(i, j)`.

An ordinary heap-allocated `Matrix` is adequate for a scalar reference
kernel (unlike a SIMD candidate, where heap-resident accumulators would
defeat register residency; see design doc section 8). This type carries no
other runtime state.
"""
struct ScalarKernel{MR,NR,T}
    descriptor::KernelDescriptor{MR,NR,T}
end

function ScalarKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR,NR,T}
    return ScalarKernel(KernelDescriptor(Val(MR), Val(NR), T))
end

mr(k::ScalarKernel) = mr(k.descriptor)
nr(k::ScalarKernel) = nr(k.descriptor)
scalartype(k::ScalarKernel) = scalartype(k.descriptor)
packed_a_offset(k::ScalarKernel, i::Int, p::Int) = packed_a_offset(k.descriptor, i, p)
packed_b_offset(k::ScalarKernel, j::Int, p::Int) = packed_b_offset(k.descriptor, j, p)
packed_a_length(k::ScalarKernel, kc::Int) = packed_a_length(k.descriptor, kc)
packed_b_length(k::ScalarKernel, kc::Int) = packed_b_length(k.descriptor, kc)

# Integration (main process): pack_a!/pack_b! (src/packing.jl) dispatch on a
# bare KernelDescriptor; forward ScalarKernel (and, by the same pattern, any
# future concrete kernel wrapping a KernelDescriptor as `.descriptor`) to it,
# so the same `kernel` object can be passed to packing and to
# zero_accumulator/accumulate/store_tile!/execute_tile! uniformly.
pack_a!(packed::Vector{T}, source::QSTile, kernel::ScalarKernel{MR,NR,T}, transform::F) where {T,MR,NR,F} =
    pack_a!(packed, source, kernel.descriptor, transform)
pack_b!(packed::Vector{T}, source::QSTile, kernel::ScalarKernel{MR,NR,T}, transform::F) where {T,MR,NR,F} =
    pack_b!(packed, source, kernel.descriptor, transform)

"""
    ScalarDestination(storage, base, rowaddr, coladdr, m, n)

Minimal destination-tile representation used by this file's `store_tile!`/
`execute_tile!`. `storage` is a `Vector{T}`-backed parent buffer; `base` is
the zero-based address (in `storage`) of the tile's logical origin;
`rowaddr`/`coladdr` are `Int -> Int` closures mapping a zero-based local row
(resp. column) coordinate `t` in `0:m-1` (resp. `0:n-1`) to a zero-based
address contribution, so the combined zero-based logical-to-physical address
for valid coordinates `(i, j)` is

    base + rowaddr(i) + coladdr(j)

and the one-based `storage` index is that address plus one (per design doc
section 3). `m`/`n` are the tile's *valid* logical extents (`0 <= m <= mr`,
`0 <= n <= nr` for the kernel it is used with) — not physical/register-tile
capacities. Only coordinates `i in 0:m-1`, `j in 0:n-1` are ever addressed or
mutated by the functions below; nothing outside that rectangle is read or
written, so canary values placed elsewhere in `storage` are untouched.

Two convenience closure constructors cover the spec's two addressing modes:

  * `affine_axis(base, stride)` returns `t -> base + t * stride` (unit,
    nonunit, and negative strides all fall out of this one form).
  * `scatter_axis(offsets)` returns `t -> offsets[t + 1]` for an explicit
    zero-based offset list.

Rows and columns are addressed independently: a `ScalarDestination` may
freely mix an `affine_axis` row closure with a `scatter_axis` column
closure, or vice versa.
"""
struct ScalarDestination{T,RF,CF}
    storage::Vector{T}
    base::Int
    rowaddr::RF
    coladdr::CF
    m::Int
    n::Int
end

affine_axis(base::Int, stride::Int) = (t::Int -> base + t * stride)
scatter_axis(offsets::AbstractVector{<:Integer}) = (t::Int -> Int(offsets[t + 1]))

@inline function _address(destination::ScalarDestination, i::Int, j::Int)
    return destination.base + destination.rowaddr(i) + destination.coladdr(j)
end

"""
    zero_accumulator(kernel::ScalarKernel{MR,NR,T}) -> Matrix{T}

Return a logical `MR`-by-`NR` zero accumulator tile, `acc[i+1,j+1] == 0`
for every zero-based `(i,j)` in `0:MR-1 x 0:NR-1`.
"""
function zero_accumulator(kernel::ScalarKernel{MR,NR,T}) where {MR,NR,T}
    return zeros(T, MR, NR)
end

"""
    accumulate(kernel::ScalarKernel, acc, packed_a, packed_b, kc::Int) -> acc

Note: this method extends `Base.accumulate` (dispatching on our own
`ScalarKernel` first argument, not type piracy) rather than introducing a
second, colliding top-level binding named `accumulate` — `Base.accumulate`
(cumulative reduction over arrays) is already exported by `Base`, and
`src/QuasiStrided.jl` also `export`s `accumulate`; two distinct global
functions of the same exported name are ambiguous at any call site that
`using`s both modules, which is exactly what happens under
`using QuasiStrided` (every module implicitly `using`s `Base`). Extending
`Base.accumulate` instead avoids that ambiguity: there is only ever one
generic function named `accumulate`, with this method added to it.

Update `acc` in place with `kc` K-steps of the packed product, in increasing
`p` order:

    R[i,j] = muladd(Ap[i,p], Bp[j,p], R[i,j])

for every `i in 0:mr(kernel)-1`, `j in 0:nr(kernel)-1`, `p in 0:kc-1`.
Supports a nonzero incoming `acc`. For `kc == 0` returns `acc` unchanged
without reading `packed_a`/`packed_b`. Never reads or writes any destination
tensor. Returns the (mutated, for this scalar representation) `acc`.
"""
function Base.accumulate(kernel::ScalarKernel{MR,NR,T}, acc::AbstractMatrix{T},
                          packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
                          kc::Int) where {MR,NR,T}
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
    scale_tile!(destination::ScalarDestination{T}, beta::T) -> destination

Apply `C[i,j] = beta * C[i,j]` over the destination's valid `m`-by-`n`
rectangle only. `beta == 0` (including signed zero) writes `zero(T)` to
every valid coordinate *without reading* the prior destination value —
there is textually no load of `destination.storage` in that branch, only a
store. `beta == 1` is a no-op (nothing read or written). Any other `beta`
reads then multiplies. An empty destination (`m == 0` or `n == 0`) is a
no-op in every branch.
"""
function scale_tile!(destination::ScalarDestination{T}, beta::T) where {T}
    m = destination.m
    n = destination.n
    (m == 0 || n == 0) && return destination
    if isone(beta)
        # No-op: beta=1 performs no destination reads or writes.
        return destination
    elseif iszero(beta)
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            addr = _address(destination, i, j)
            destination.storage[addr + 1] = zero(T)  # store only, no read of old C
        end
    else
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            addr = _address(destination, i, j)
            destination.storage[addr + 1] *= beta
        end
    end
    return destination
end

"""
    store_tile!(destination::ScalarDestination{T}, acc, alpha::T, beta::T, kernel::ScalarKernel) -> destination

For every valid destination coordinate `(i,j)` in `0:m-1 x 0:n-1` (`m`, `n`
the destination's valid extents, `m <= mr(kernel)`, `n <= nr(kernel)`),
compute

    C[i,j] = alpha * R[i,j] + beta * C[i,j]

with the BLAS-like shortcuts fixed by design doc section 9:

  * `alpha == 0`: ignore `acc` entirely (no read of it, and no evaluation of
    `alpha * R[i,j]` — this is what keeps a nonfinite `R[i,j]` from ever
    being read when alpha is zero) and perform only `beta` scaling via
    `scale_tile!`.
  * `alpha == 0 && beta == 0`: writes `zero(T)` without reading `acc` or the
    old destination value (falls out of the two rules above composing).
  * `alpha == 0 && beta == 1`: no destination reads or writes at all (falls
    out of `scale_tile!`'s `beta == 1` no-op branch).
  * `beta == 0`: never reads old `C` (see `scale_tile!`/the `beta==0` branch
    below — textually only a store, no load, of `destination.storage`).
  * `beta == 1`: may skip the multiplication (uses `muladd(alpha, R, C)`
    directly rather than `alpha*R + 1*C`).

Only the valid `m`-by-`n` rectangle is touched; accumulator lanes at
`i >= m` or `j >= n` (kernel padding lanes) are never read here, so
nonfinite padding-lane values in `acc` cannot propagate into the stored
output. Storage outside the valid rectangle (canaries) is never addressed.
"""
function store_tile!(destination::ScalarDestination{T}, acc::AbstractMatrix{T},
                      alpha::T, beta::T, kernel::ScalarKernel) where {T}
    m = destination.m
    n = destination.n
    (m == 0 || n == 0) && return destination

    if iszero(alpha)
        # alpha=0 MUST ignore the accumulator: no read of `acc`, and no
        # `alpha * R[i,j]` evaluation, so a nonfinite R[i,j] cannot surface.
        scale_tile!(destination, beta)
        return destination
    end

    if iszero(beta)
        # beta=0 MUST avoid reading old C: pure store, no load, below.
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            addr = _address(destination, i, j)
            destination.storage[addr + 1] = alpha * acc[i + 1, j + 1]
        end
    elseif isone(beta)
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            addr = _address(destination, i, j)
            c = destination.storage[addr + 1]
            destination.storage[addr + 1] = muladd(alpha, acc[i + 1, j + 1], c)
        end
    else
        @inbounds for j in 0:(n - 1), i in 0:(m - 1)
            addr = _address(destination, i, j)
            c = destination.storage[addr + 1]
            destination.storage[addr + 1] = muladd(alpha, acc[i + 1, j + 1], beta * c)
        end
    end
    return destination
end

"""
    execute_tile!(kernel::ScalarKernel, destination::ScalarDestination, packed_a, packed_b, kc::Int, alpha, beta) -> destination

Checked composition of `zero_accumulator`, `accumulate` and `store_tile!`
for a single K-panel call (multi-panel accumulation across several calls,
carrying `beta_effective` across panels, is the driver's responsibility —
see design doc section 10 — and is not implemented here). Validates
`destination`'s valid extents against the kernel's register-tile shape
before any mutation, and converts `alpha`/`beta` to `T` at this checked
boundary (never silently narrowing/widening beyond that one explicit
`convert`).

If the destination is empty, returns immediately. If `kc == 0` or
`alpha == 0` (once converted to `T`), applies only `beta` scaling — via
`scale_tile!` through `store_tile!`'s own `alpha==0` branch — and returns
without ever reading `packed_a`/`packed_b` (matching design doc section
10's short-circuit guidance: no accumulation, no read of input panels).
"""
function execute_tile!(kernel::ScalarKernel{MR,NR,T}, destination::ScalarDestination{T},
                        packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
                        kc::Int, alpha, beta) where {MR,NR,T}
    destination.m <= MR ||
        throw(ArgumentError("destination valid row extent $(destination.m) exceeds mr(kernel) = $MR"))
    destination.n <= NR ||
        throw(ArgumentError("destination valid column extent $(destination.n) exceeds nr(kernel) = $NR"))
    kc >= 0 || throw(ArgumentError("execute_tile! requires kc >= 0, got kc = $kc"))

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    (destination.m == 0 || destination.n == 0) && return destination

    if kc == 0 || iszero(alphaT)
        # Short-circuit: no accumulation, no read of packed_a/packed_b.
        scale_tile!(destination, betaT)
        return destination
    end

    # Fable review (Phase 2b) follow-up: reject undersized packed buffers
    # before the unchecked @inbounds reads in `accumulate` (spec section 7).
    # (This method's `destination` is the scalar worker's own test-only
    # `ScalarDestination`, so no storage-bounds check is added here — see
    # `checked_tile_storage_bounds` on the `QSTile` overload below, which is
    # the path Phase 3 uses.)
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

# ----------------------------------------------------------------------------
# Integration (main process, Phase 2 gate): the same store_tile!/execute_tile!
# contract, directly on a real tiles.jl `DestinationTile` (a `QSTile`).
#
# The scalar-kernel worker above wrote `ScalarDestination` independently and
# in parallel with the packing worker's `QSTile`/`DestinationTile`, per its
# task instructions (it was told not to block on tiles.jl landing). Per
# docs/decisions.md, the main process reconciles the two here rather than
# asking either worker to redo work: `QSTile` already exposes exactly the
# primitives (`nrows`, `ncols`, `tile_load`, `tile_store!`) that
# `ScalarDestination`'s `m`, `n`, and closure-based addressing were a stand-in
# for, so these methods mirror the logic above one-for-one, substituting
# those primitives for `destination.m`/`destination.n`/`_address`. This is
# the path Phase 3's driver is expected to use (`DestinationTile` built via
# `axis_from_descriptor`), not `ScalarDestination`, which remains only as the
# scalar kernel's own test scaffolding.
# ----------------------------------------------------------------------------

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

function store_tile!(destination::QSTile, acc::AbstractMatrix{T},
                      alpha::T, beta::T, kernel::ScalarKernel) where {T}
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

function execute_tile!(kernel::ScalarKernel{MR,NR,T}, destination::QSTile,
                        packed_a::AbstractVector{T}, packed_b::AbstractVector{T},
                        kc::Int, alpha, beta) where {MR,NR,T}
    m = nrows(destination)
    n = ncols(destination)
    m <= MR || throw(ArgumentError("destination valid row extent $m exceeds mr(kernel) = $MR"))
    n <= NR || throw(ArgumentError("destination valid column extent $n exceeds nr(kernel) = $NR"))
    kc >= 0 || throw(ArgumentError("execute_tile! requires kc >= 0, got kc = $kc"))

    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    (m == 0 || n == 0) && return destination

    # Fable review (Phase 2b) follow-up: validate reachable destination
    # storage bounds, and that the supplied packed buffers are at least as
    # large as this kc demands, before entering any unchecked @inbounds path
    # below (spec section 3: "in bounds ... before entering unchecked hot
    # paths"; section 7: "reject incompatible buffers ... before kernel
    # execution").
    checked_tile_storage_bounds(destination)

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
