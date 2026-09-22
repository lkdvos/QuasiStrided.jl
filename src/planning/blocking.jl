# Cache-blocking factors for the driver's five-loop nest. `Blocking` is a
# plain, always-valid (>=1 per field) user-facing value; `plan_contract`
# rounds/clamps it into the *effective* blocking stored on `ContractPlan`.

"""
    Blocking(mc::Int, kc::Int, nc::Int)

Cache-blocking factors for the macro-blocking driver: `mc` (M block extent,
loop 3), `kc` (K block/panel depth, loop 4), `nc` (N block extent, loop 5).
All three fields must be `>= 1`; the constructor throws `ArgumentError`
otherwise.
"""
struct Blocking
    mc::Int
    kc::Int
    nc::Int

    function Blocking(mc::Int, kc::Int, nc::Int)
        mc >= 1 || throw(ArgumentError("Blocking requires mc >= 1, got mc = $mc"))
        kc >= 1 || throw(ArgumentError("Blocking requires kc >= 1, got kc = $kc"))
        nc >= 1 || throw(ArgumentError("Blocking requires nc >= 1, got nc = $nc"))
        return new(mc, kc, nc)
    end
end

"""
    default_blocking(kernel) -> Blocking

Cache-blocking factors keyed on the detected vector ISA
([`target_profile`](@ref)) and `scalartype(kernel)`. Measured constants, not a
cache model: `plan_contract` rounds `mc`/`nc` to `mr`/`nr` multiples and clamps
them to the contraction's extents.

Cache-geometry *derivation* is still not used although `src/target.jl` now
detects the geometry: re-measuring the 36-point grid found it spans only
9%/11% best-to-worst, so a model's upside is a few percent against the tens of
percent docs/decisions.md records such models losing. [`cache_topology`](@ref)
is exposed for reporting only.
"""
default_blocking(kernel) =
    default_blocking(Val(target_profile().isa), scalartype(kernel), complex_method(kernel))

# Same packed BYTE budget for every method, rather than the same element count.
# The reference's D13 records that deriving complex blocking from `sizeof(T)`
# "would hand 1m double the L2 footprint and quietly rig the comparison",
# because 1m's packed A carries four reals per complex element where planar
# carries two. Anchoring to the MEASURED real row and dividing by the packed
# reals per element keeps this project's "measured constants, not a cache
# model" policy intact while making the head-to-head fair by construction: 1m's
# `mc` comes out at exactly half planar's, DERIVED, never tabulated.
@inline _scale_blocking(base::Blocking, m::ComplexMethod) =
    Blocking(max(1, base.mc ÷ a_reals(m)), base.kc, max(1, base.nc ÷ b_reals(m)))

# The real path, verbatim: `RealMethod` has `a_reals == b_reals == 1`, but it
# forwards to the two-argument form rather than scaling by one, so every real
# lookup reaches exactly the same method it always did.
default_blocking(v::Val, ::Type{T}, ::RealMethod) where {T} = default_blocking(v, T)

# `T` is deliberately unconstrained here rather than `T <: Complex`: with the
# bound, this method and the `RealMethod` one above are mutually ambiguous at
# `(Val, Type{<:Complex}, RealMethod)` -- neither is more specific. Unbounded,
# the `RealMethod` method is strictly more specific and wins, and `real(T) === T`
# makes this arm total for a real `T` anyway.
default_blocking(v::Val, ::Type{T}, m::ComplexMethod) where {T} =
    _scale_blocking(default_blocking(v, real(T)), m)

# What shipped before hardware detection existed; every ISA without a measured
# row falls back here, so an unrecognized CPU is bit-identical to the old
# behavior. Measured 2026-09-08 at the (8,6) shape (Phase E).
_legacy_blocking(::Type{Float64}) = Blocking(64, 128, 768)
_legacy_blocking(::Type{Float32}) = Blocking(96, 384, 1152)
# Complex legacy rows are derived through the same formula rather than
# hand-tabulated, so a direct call degrades gracefully instead of throwing a
# `MethodError`. `PlanarMethod` is the default here for the same reason it is
# the default kernel method.
_legacy_blocking(::Type{T}, m::ComplexMethod = PlanarMethod()) where {T <: Complex} =
    _scale_blocking(_legacy_blocking(real(T)), m)
default_blocking(::Val, ::Type{T}) where {T} = _legacy_blocking(T)

# AVX-512 at the derived shape (Phase G). `kc` was swept jointly with the
# register shape, since MR*kc*sizeof(T) is the A-micropanel L1 footprint;
# `mc`/`nc` then re-validated on the full grid, landing within ~1.6% of its
# best at ~2.1x the old packed footprint (1.75 vs 0.81 MiB Float64, 3.66 vs
# 1.83 MiB Float32). The grid is a plateau, so these are not sharp optima.
default_blocking(::Val{:avx512}, ::Type{Float64}) = Blocking(128, 256, 768)
default_blocking(::Val{:avx512}, ::Type{Float32}) = Blocking(96, 768, 1152)

# Unchanged pre-detection behavior for callers passing a scalar type.
default_blocking(::Type{Float64}) = _legacy_blocking(Float64)
default_blocking(::Type{Float32}) = _legacy_blocking(Float32)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
