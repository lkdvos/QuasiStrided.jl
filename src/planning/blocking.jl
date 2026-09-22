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
([`target_profile`](@ref)), `scalartype(kernel)` and the kernel's method.
Measured constants, not a cache model: the measured grid spans only 9%/11%
best-to-worst, so a model's upside is a few percent against the tens of percent
docs/decisions.md records such models losing. [`cache_topology`](@ref) is
exposed for reporting only. `plan_contract` rounds `mc`/`nc` to `mr`/`nr`
multiples and clamps them to the contraction's extents.
"""
default_blocking(kernel) =
    default_blocking(Val(target_profile().isa), scalartype(kernel), complex_method(kernel))

# Complex blocking is the measured real row divided by the packed reals per
# element of each operand, so every method gets the same packed BYTE budget
# rather than the same element count. Deriving it from `sizeof(T)` instead
# would hand 1m (four reals per packed A element, planar two) double the L2
# footprint and rig any planar-vs-1m comparison; this way 1m's `mc` comes out
# at exactly half planar's.
@inline _scale_blocking(base::Blocking, m::ComplexMethod) =
    Blocking(max(1, base.mc ÷ a_reals(m)), base.kc, max(1, base.nc ÷ b_reals(m)))

# The real rows, unscaled.
default_blocking(v::Val, ::Type{T}, ::RealMethod) where {T} = default_blocking(v, T)

# `T` is deliberately unconstrained rather than `T <: Complex`: with the bound,
# this method and the `RealMethod` one above are mutually ambiguous at
# `(Val, Type{<:Complex}, RealMethod)`. Unbounded, the `RealMethod` method is
# strictly more specific and wins.
default_blocking(v::Val, ::Type{T}, m::ComplexMethod) where {T} =
    _scale_blocking(default_blocking(v, real(T)), m)

# Every ISA without a measured row uses the fallback, measured at the (8,6)
# fallback shape. Complex rows are derived through the same formula, with
# `PlanarMethod` as the default method.
_fallback_blocking(::Type{Float64}) = Blocking(64, 128, 768)
_fallback_blocking(::Type{Float32}) = Blocking(96, 384, 1152)
_fallback_blocking(::Type{T}, m::ComplexMethod = PlanarMethod()) where {T <: Complex} =
    _scale_blocking(_fallback_blocking(real(T)), m)
default_blocking(::Val, ::Type{T}) where {T} = _fallback_blocking(T)

# AVX-512 at the derived shape. `kc` was swept jointly with the register shape,
# since MR*kc*sizeof(T) is the A-micropanel L1 footprint; `mc`/`nc` then sit on
# a plateau of the full grid, so these are not sharp optima.
default_blocking(::Val{:avx512}, ::Type{Float64}) = Blocking(128, 256, 768)
default_blocking(::Val{:avx512}, ::Type{Float32}) = Blocking(96, 768, 1152)

# For a bare scalar type: the fallback row, independent of the host.
default_blocking(::Type{Float64}) = _fallback_blocking(Float64)
default_blocking(::Type{Float32}) = _fallback_blocking(Float32)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
