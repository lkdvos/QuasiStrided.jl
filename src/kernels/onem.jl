# 1m (Van Zee's induced method): the *real* microkernel, run over `2*kc` real
# K steps against "1e"-packed A and "1r"-packed B.
#
# There is deliberately NO FMA loop in this file. `accumulate` is
#
#     accumulate(kernel.inner, acc, packed_a, packed_b, 2 * kc)
#
# i.e. exactly `real::<MV,NR>(2*kc, a, b, ab)` in the reference project
# (`crates/tensorcontract/src/kernel/simd.rs:170-177`). That is the whole point:
# one kernel body written once and parameterised, so that a planar-vs-1m
# measurement compares two *methods* rather than two hand-tunings (their
# D17/D24). If a future edit finds itself writing an inner loop here, the
# comparison has already been invalidated.
#
# WHY IT WORKS. `OneEFormat` A at logical K step `p` occupies `4*MR` reals laid
# out as two consecutive real K steps of `2*MR`:
#
#     reals   0 .. 2MR-1 :  re_0, im_0, re_1, im_1, ...
#     reals 2MR .. 4MR-1 : -im_0, re_0, -im_1, re_1, ...
#
# and `PlanarFormat` B at that step occupies `2*NR` reals as two real K steps of
# `NR` (`re_0..re_{NR-1}`, then `im_0..im_{NR-1}`). A real `SIMDKernel{2MR,NR}`
# addresses A at `i' + 2MR*p'` and B at `j + NR*p'`, which walks both buffers
# linearly -- so it reads exactly those blocks, with real step `p' = 2p` the
# first and `p' = 2p+1` the second. The real product it computes is therefore
#
#     Ar[2t,   2p] =  re(A[t,p])   Ar[2t,   2p+1] = -im(A[t,p])
#     Ar[2t+1, 2p] =  im(A[t,p])   Ar[2t+1, 2p+1] =  re(A[t,p])
#     Br[j,    2p] =  re(B[p,j])   Br[j,    2p+1] =  im(B[p,j])
#
# whose row `2t` sums `re*re - im*im` (the real part) and whose row `2t+1` sums
# `im*re + re*im` (the imaginary part). Hence the accumulator's real row `2i` is
# the real part and real row `2i+1` the imaginary part of complex row `i`.
#
# `accumulate` takes the LOGICAL (complex) `kc`; the `2*kc` doubling is confined
# to this file's own delegation and never appears in a length, an offset or a
# driver loop bound (their D16).
#
# Cliff B (Julia heap-allocating a dynamically indexed `NTuple` above NV = 16,
# 24576 B per `execute!` measured in Phase H) applies here exactly as it does to
# planar: the store below is `@generated` with literal tuple indices including
# the lane tail, and the real path's runtime-indexed `_acc_lane` helper is not
# used. `accumulate` inherits the real kernel's already-`@generated` body.

using SIMD: Vec

"""
    OneMKernel{MR,NR,T,W,KI}(descriptor, inner)
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W})
    OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T})

Van Zee's induced 1m microkernel, implementing [`OneMMethod`](@ref): a real
`SIMDKernel` of shape `2MR x NR` over `SIMD.Vec{W,real(T)}` lanes, driven over
`2*kc` real K steps by [`OneEFormat`](@ref) A and [`PlanarFormat`](@ref) B.

`MR`/`NR` are the **logical** (complex) register-tile extents and `T` is the
**storage** element type (`ComplexF32`/`ComplexF64`); `W` is the lane count of
the *real* type. Note the shape bookkeeping: `mr(kernel) == MR` while
`mr(kernel.inner) == 2MR`, and it is `2MR` -- not `MR` -- that must be a
multiple of `W`. That is why the shipped menu contains `MR = 12` at `W = 8`:
`2*12 = 24` is the multiple. `W` must additionally be **even**, which is what
makes the two halves of a complex row adjacent lanes of one `Vec` (see
[`store_tile!`](@ref)).

The field is named `descriptor`, so `mr`/`nr`/`scalartype`/`packed_a_length`/
`packed_b_length`/`realtype`/`packed_a_per_k`/`packed_b_per_k`/`a_format`/
`b_format` all forward through the existing `DescriptorKernel` methods in
src/kernel.jl and src/complex_format.jl.

`KI` exists only because Julia cannot compute a field type from type
parameters: the freeze spells the field `inner::SIMDKernel{2MR,NR,real(T),W}`,
which is not a legal struct field type (`2MR` and `real(T)` are computations on
`TypeVar`s). `KI` carries that computed type instead, and the inner constructor
pins it to exactly `SIMDKernel{2MR,NR,real(T),W}`, so nothing else can be
stored there. `OneMKernel{MR,NR,T,W}` remains a usable (partially applied)
spelling for `isa` and for dispatch.

The 3-argument form defaults `W` via `_default_lanewidth(real(T))`.
"""
struct OneMKernel{MR, NR, T, W, KI <: SIMDKernel} <: DescriptorKernel{MR, NR, T}
    descriptor::ComplexKernelDescriptor{MR, NR, T, OneEFormat, PlanarFormat}
    inner::KI

    function OneMKernel{MR, NR, T, W, KI}(
            descriptor::ComplexKernelDescriptor{MR, NR, T, OneEFormat, PlanarFormat},
            inner::KI
        ) where {MR, NR, T, W, KI <: SIMDKernel}
        W isa Int && W > 0 ||
            throw(ArgumentError("OneMKernel requires an Int vector width W > 0, got W = $W"))
        # Even `W` is load-bearing, not cosmetic: the accumulator is read back
        # as complex row `i` from lanes `2u+1`/`2u+2` of one vector, which is
        # only sound when vectors start at even real rows.
        iseven(W) ||
            throw(
            ArgumentError(
                "OneMKernel requires an even vector width W (the two halves of a " *
                    "complex row must be adjacent lanes of one Vec), got W = $W"
            )
        )
        mod(2 * MR, W) == 0 ||
            throw(
            ArgumentError(
                "OneMKernel requires 2*mr(kernel) = $(2 * MR) to be a multiple of " *
                    "the vector width W = $W (1m runs a REAL microkernel of 2*MR " *
                    "rows, so it is 2*MR and not MR that must divide by W)"
            )
        )
        KI === SIMDKernel{2 * MR, NR, real(T), W} ||
            throw(
            ArgumentError(
                "OneMKernel's inner kernel must be SIMDKernel{$(2 * MR),$NR," *
                    "$(real(T)),$W}, got $KI"
            )
        )
        return new{MR, NR, T, W, KI}(descriptor, inner)
    end
end

function OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}, ::Val{W}) where {MR, NR, T, W}
    T <: Complex ||
        throw(ArgumentError("OneMKernel requires a complex element type, got $T"))
    descriptor = ComplexKernelDescriptor(Val(MR), Val(NR), T, OneEFormat(), PlanarFormat())
    inner = SIMDKernel(Val(2 * MR), Val(NR), real(T), Val(W))
    return OneMKernel{MR, NR, T, W, typeof(inner)}(descriptor, inner)
end
function OneMKernel(::Val{MR}, ::Val{NR}, ::Type{T}) where {MR, NR, T}
    T <: Complex ||
        throw(ArgumentError("OneMKernel requires a complex element type, got $T"))
    return OneMKernel(Val(MR), Val(NR), T, Val(_default_lanewidth(real(T))))
end

complex_method(::OneMKernel) = OneMMethod()

"""
    lanewidth(kernel::OneMKernel) -> Int

The kernel's `SIMD.Vec` lane width `W`, counted in **reals**. Always even.
"""
lanewidth(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = W

"""
    avecs_per_column(kernel::OneMKernel) -> Int

`2*mr(kernel) ÷ lanewidth(kernel)`: A vectors (and accumulator vectors) per
output column. Counted in the inner **real** kernel's rows, so it is `2MR÷W`
and not `MR÷W`; each vector covers `W÷2` complex rows.
"""
avecs_per_column(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} = (2 * MR) ÷ W

# Plane-offset forwarding, as on `PlanarKernel`. The microkernel itself does
# not use these (it addresses through the inner real kernel's single-plane
# `packed_a_offset`/`packed_b_offset`); they exist so that the packed layout is
# reachable from the kernel for tests and for symmetry with planar.
# `packed_a_offset(k::OneMKernel, i, p)` deliberately has no method: the
# single-plane accessor is meaningless on a complex descriptor, and the
# `MethodError` is the intended outcome.
@inline packed_a_plane_offset(k::OneMKernel, plane::Int, i::Int, p::Int) =
    packed_a_plane_offset(k.descriptor, plane, i, p)
@inline packed_b_plane_offset(k::OneMKernel, plane::Int, j::Int, p::Int) =
    packed_b_plane_offset(k.descriptor, plane, j, p)

"""
    onem_register_pressure(kernel::OneMKernel) -> Int

Vector registers live at the bottom of the inner real kernel's K loop:

    MV*NR accumulators + MV A vectors + 1 B broadcast

with `MV = 2*mr(kernel) ÷ lanewidth(kernel)`. This is the **real** kernel's
budget, unchanged -- 1m holds one accumulator plane over a doubled real row
count where planar holds two planes over `MR` rows, so at equal `(MV, NR)`
1m needs a little over half of planar's registers. **Cliff A**: it must be
`<=` the architectural register count (`target_profile().nregisters`; 32 zmm
under AVX-512).

**Measured**, because Phase C found the freeze's register arithmetic optimistic
once already and a spill count is not something to assume. `@code_native` on
the inner loop of `accumulate` (Julia 1.12.6, ccqlin038, cascadelake
`:avx512`, 32 zmm), per **real** K step. `reloads` counts stack memory operands
used as sources, folded FMA operands (`vfmadd213pd zmm, zmm, [rbp-N]`)
included; `stores` counts stack memory operands written.

Measured on `kernel.inner`: `accumulate(::OneMKernel, ...)` does not inline
into its caller, so the code that actually runs the K loop *is* the real
path's `accumulate(::SIMDKernel{2MR,NR,real(T),W}, ...)`, byte for byte. That
is the reuse being confirmed rather than a limitation of the measurement.

| shape (MR,NR,W) | MV | pressure | stores | reloads | FMAs |
| --- | --- | --- | --- | --- | --- |
| 1m (12,8,8) CF64 / (24,8,16) CF32 | 3 | 28 | 0 | 0 | 24 |
| 1m (16,6,8) CF64 / (32,6,16) CF32 | 4 | 29 | 0 | 0 | 24 |
| 1m ( 8,8,8) CF64 / (16,8,16) CF32 | 2 | 19 | 0 | 0 | 16 |
| 1m ( 8,4,4) CF64 (small fixture)  | 4 | 21 | 0 | 0 | 16 |

Every shipped 1m shape is spill-free, at exactly `MV*NR` FMAs per real K step
(so `2*MV*NR` per *logical* complex K step). The same instrument reproduces
Phase C's real controls exactly -- real `(32,6,8)` clean at pressure 29, real
`(48,6,8)` at 12 stores / 12 reloads at pressure 43 -- and on planar it reports
`(16,6,8)`/`(32,6,16)` at 24 stores / 6 reloads and `(24,3,8)`/`(48,3,16)`
clean, close to but not identical with the table in
`planar_register_pressure`. It does *not* reproduce that table's planar
`(8,8,8)`/`(16,8,16)` row: this instrument measures 15 stores and 8 reloads
there, at pressure 20, while `(24,3,8)` at pressure 26 is clean. Recorded here
as a disagreement to be settled in Phase F, not worked around; it does not
affect 1m, whose rows are zero throughout.

**This is not a throughput or ranking claim.** Spill counts are not timings;
ranking planar against 1m is Phase F's, on measured throughput.
"""
onem_register_pressure(::OneMKernel{MR, NR, T, W}) where {MR, NR, T, W} =
    ((2 * MR) ÷ W) * NR + ((2 * MR) ÷ W) + 1

# ----------------------------------------------------------------------------
# zero_accumulator, accumulate -- both are the real kernel's, verbatim
# ----------------------------------------------------------------------------

"""
    zero_accumulator(kernel::OneMKernel{MR,NR,T,W}) -> NTuple{NV,SIMD.Vec{W,real(T)}}

The **inner real kernel's** zero accumulator: `NV = (2MR÷W)*NR` zero
`Vec{W,real(T)}` values, entry `(v, j)` at 1-based tuple position
`v + (2MR÷W)*j + 1`, exactly as `zero_accumulator(::SIMDKernel)` defines it.

Interpreted as a real `2MR x NR` tile, real row `2i` holds the real part and
real row `2i+1` the imaginary part of complex row `i`; see
[`store_tile!`](@ref) for how that is read back.
"""
zero_accumulator(kernel::OneMKernel) = zero_accumulator(kernel.inner)

"""
    accumulate(kernel::OneMKernel, acc, packed_a, packed_b, kc::Int) -> acc

The induced method itself: the real microkernel over `2*kc` real K steps.
`kc` is the **logical** (complex) K depth; `kc == 0` returns `acc` unchanged
without reading the packed buffers. `packed_a` must be in
[`OneEFormat`](@ref) and `packed_b` in [`PlanarFormat`](@ref).

There is no arithmetic here and there must never be: the body is
`accumulate(kernel.inner, acc, packed_a, packed_b, 2 * kc)`, the same
`_accumulate_step` the real path runs. Not bitwise identical to a scalar
complex dot product (FMA grouping differs) — compare with a tolerance, never
`==`. No tolerance widening relative to the real path at the same precision is
needed or expected.
"""
function Base.accumulate(
        kernel::OneMKernel{MR, NR, T, W}, acc::NTuple{NV, Vec{W, R}},
        packed_a::PA, packed_b::PB, kc::Int
    ) where {MR, NR, T, W, R, NV, PA, PB}
    # Checked here rather than left to the inner kernel so the message reports
    # the LOGICAL kc the caller passed, not the doubled real one.
    kc >= 0 || throw(ArgumentError("accumulate requires kc >= 0, got kc = $kc"))
    return accumulate(kernel.inner, acc, packed_a, packed_b, 2 * kc)
end

# ----------------------------------------------------------------------------
# store_tile! -- the one genuinely new piece
# ----------------------------------------------------------------------------

# The `OneM` tile reader. The accumulator is the inner real kernel's, holding a
# real `2MR x NR` tile; complex row `i` is real rows `2i` (re) and `2i+1` (im).
#
# ADJACENCY. Accumulator vector `v` covers real rows `v*W .. v*W+W-1`. `W` is
# even (enforced at construction) and `2MR` is a multiple of `W`, so `v*W` is
# even and `MR = MV*(W÷2)` exactly -- the complex rows partition cleanly across
# vectors, `W÷2` per vector, with none split across a vector boundary.
# Therefore the two halves of complex row `i = v*(W÷2) + u` are lanes `2u+1`
# and `2u+2` (1-based `SIMD.Vec` indexing) of vector `v`, always adjacent, in
# one vector. This is the cheapest of the recombinations: no cross-vector
# shuffle, no second load.
#
# `@generated` with literal tuple indices (Cliff B), unrolled over `(v, j)`
# exactly as `_store_tile_scattered!` (src/kernels/simd.jl) and
# `_store_tile_planar!` (src/kernels/planar.jl); only the lane index inside a
# single `Vec` is a runtime value, which stays on the stack.
#
# Scattered/scalar path only, delegating to the existing generic
# `_axpby_tile!` (src/kernel.jl) for the alpha/beta shortcuts. The unit-stride
# interleaved-store fast path is the same deliberately deferred,
# measurement-gated follow-on it is for planar, and is NOT built here.
@generated function _store_tile_onem!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::OneMKernel{MR, NR, T, W},
        m::Int, n::Int
    ) where {MR, NR, T, W, R, NV}
    R === real(T) ||
        throw(
        ArgumentError(
            "_store_tile_onem!: accumulator lane type $R does not match " *
                "real($T) = $(real(T))"
        )
    )
    iseven(W) ||
        throw(ArgumentError("_store_tile_onem!: the lane-pair reader requires an even W, got $W"))
    MV = (2 * MR) ÷ W
    MV * NR == NV ||
        throw(
        ArgumentError(
            "_store_tile_onem!: accumulator length $NV does not match " *
                "(2*mr÷W)*nr = $(MV * NR) for MR=$MR, NR=$NR, W=$W"
        )
    )
    HW = W ÷ 2  # complex rows per accumulator vector

    blocks = Any[]
    for j in 0:(NR - 1), v in 0:(MV - 1)
        idx = v + MV * j + 1
        push!(
            blocks, quote
                if $j < n
                    vec = acc[$idx]
                    for u in 0:$(HW - 1)
                        i = $(v * HW) + u
                        i < m || break
                        _axpby_tile!(
                            destination, i, $j, alpha,
                            Complex(vec[2 * u + 1], vec[2 * u + 2]), beta
                        )
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
    store_tile!(destination::QSTile, acc::NTuple{NV,SIMD.Vec{W,real(T)}}, alpha::T, beta::T, kernel::OneMKernel) -> destination

1m counterpart of `SIMDKernel`'s and `PlanarKernel`'s `store_tile!`, with the
**same contract**: `C[i,j] = alpha*R[i,j] + beta*C[i,j]` over the valid
rectangle only; `alpha == 0` never reads `acc`; `beta == 0` never reads old
`C`; `beta == 1` skips the multiplication; padding lanes (`i >= m`, `j >= n`)
are never read, so nonfinite padding in `acc` cannot propagate; an empty
destination is a no-op in every branch.

The accumulator is the *inner real* kernel's, read as a real `2MR x NR` tile in
which real row `2i` is the real part and real row `2i+1` the imaginary part of
complex row `i`. Because `W` is even and accumulator vectors therefore start at
even real rows, those two halves are always **adjacent lanes of the same
`Vec`**: complex row `i = v*(W÷2) + u` is `Complex(acc[v + MV*j + 1][2u+1],
acc[v + MV*j + 1][2u+2])` with `MV = 2MR÷W`. No shuffle, no second load.

Ships the scattered/scalar path only, delegating to the existing generic
`_axpby_tile!` (src/kernel.jl).
"""
function store_tile!(
        destination::QSTile, acc::NTuple{NV, Vec{W, R}},
        alpha::T, beta::T, kernel::OneMKernel{MR, NR, T, W}
    ) where {MR, NR, T, W, R, NV}
    m = nrows(destination)
    n = ncols(destination)
    (m == 0 || n == 0) && return destination

    if iszero(alpha)
        scale_tile!(destination, beta)
        return destination
    end

    return _store_tile_onem!(destination, acc, alpha, beta, kernel, m, n)
end

"""
    execute_tile!(kernel::OneMKernel, destination::QSTile, packed_a, packed_b, kc::Int, alpha, beta) -> destination

1m counterpart of `SIMDKernel`'s `execute_tile!`; same validation order and
short-circuits. `kc` is the **logical** (complex) K depth, and the
buffer-length checks go through `packed_a_length`/`packed_b_length`, which take
a logical `kc` and return a count of **reals** (`4*MR*kc` and `2*NR*kc`). The
`2*kc` real-step doubling lives inside `accumulate` and is not visible here.
"""
function execute_tile!(
        kernel::OneMKernel{MR, NR, T, W}, destination::QSTile,
        packed_a::PA, packed_b::PB, kc::Int, alpha, beta
    ) where {MR, NR, T, W, PA, PB}
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
