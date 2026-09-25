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
AVX-512 uses measured constants; AVX2 uses an analytical model of the
detected cache geometry (`_modelled_blocking`); every other ISA, and AVX2
with undetected caches, uses the fallback constants. Measured grids are wide
plateaus (1.7-17% best-to-worst over `bench_driver.jl`'s 36 points, 4-24%
over 157) whose one cliff is small `kc`, so none of these is a sharp optimum.
`plan_contract` rounds `mc`/`nc` to `mr`/`nr` multiples and clamps them to
the contraction's extents.
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

# --- analytical real row, from the detected cache geometry -------------------
#
# Byte accounting of `_execute_nest!` (src/execution/execute.jl): per (jc, pc)
# the packed B panel is `nc*kc` elements, per (jc, pc, ic) the packed A block is
# `mc*kc`, and loop 2 (jr) is outside loop 1 (ir), so one `NR x kc` B sliver is
# reused by every A sliver of the block while those stream past it. Hence
#
#     NR*kc*S  <= L1/2              the reused B sliver, half of L1
#     mc*kc*S  <= L2core/2          the A block, reused by every B sliver
#     nc*kc*S  <= L2core + L3core   the B panel, reused by every A block
#
# with `S = sizeof(T)`, rounded down to MR/NR multiples. Shared levels are
# divided by the cores sharing them (`sharing ÷ l1d.sharing`, L1d being private
# to one core's SMT siblings): on a shared node a single-threaded contraction
# cannot count on another core's slice. With no L3, `L3core` is 0.
#
# The `nc` bound is deliberately the whole per-core capacity (non-inclusive L3,
# as on every machine measured), not half the L3: an oversized B panel only
# re-streams B at `S/(2*mc)` bytes per flop, while an undersized one repacks A
# once per `jc` block. Measured (bench_blocking_model.jl, 2026-09-25, jobs
# 7107932-4 on Rome/Genoa/Ice Lake-SP): `nc <= 192` costs 3-6% on every
# machine, `nc` in 768..6144 is within ~2% of the best even where the B panel
# is 8x the core's L3 share; the half-L3 bound gave Ice Lake `nc = 192`.
#
# Real rows only; complex rows go through `_scale_blocking` as for every other
# row. `nothing` when L1d or L2 is undetected.
function _modelled_blocking(profile::TargetProfile, ::Type{T}, MR::Int, NR::Int) where {T}
    l1, l2, l3 = profile.l1d, profile.l2, profile.l3
    (l1.bytes > 0 && l2.bytes > 0) || return nothing
    smt = max(1, l1.sharing)
    core_share(c) = c.bytes ÷ max(1, c.sharing ÷ smt)
    l2core = core_share(l2)
    l3core = l3.bytes > 0 ? core_share(l3) : 0
    S = sizeof(T)
    kc = max(1, (l1.bytes ÷ 2) ÷ (NR * S))
    mc = max(MR, ((l2core ÷ 2) ÷ (kc * S)) ÷ MR * MR)
    nc = max(NR, ((l2core + l3core) ÷ (kc * S)) ÷ NR * NR)
    return Blocking(mc, kc, nc)
end

# At the real kernel shape the engine derives for `T` on `profile`.
function _modelled_blocking(profile::TargetProfile, ::Type{T}) where {T <: Real}
    MR, NR, _ = _derived_shape(profile, T)
    return _modelled_blocking(profile, T, MR, NR)
end

# Where the model replaces the fallback: only where it has been measured
# against it (the same gate as `_rule_applies`). bench_blocking_model.jl,
# 2026-09-25, geomean over its 9 shapes of time normalized per shape by the
# best of 157 grid points (1.000 = best everywhere), model / fallback / AVX-512 row:
#
#   Rome (znver2, AVX2), job 7108316   Float64 1.028 / 1.070 / 1.033
#                                      Float32 1.007 / 1.014 / 1.008
#   Genoa (znver4), job 7108317        Float64 1.009 / 1.029 / 1.013
#   Ice Lake-SP, job 7108318           Float64 1.029 / 1.048 / 1.016
#   Cascade Lake (ccqlin038, local)    Float64 1.102 / 1.090 / 1.071
#
# (Float32 within 1% of the row everywhere; complex rows, scaled as below,
# within 0.6% of the row and ahead of the fallback everywhere.) So AVX2 takes the model; AVX-512 keeps its
# row, which the model ties on AMD but trails by 1-3% on Intel. NEON and
# `:unknown` have no measurement of either and keep the fallback.
_model_applies(::Val) = false
_model_applies(::Val{:avx2}) = true

default_blocking(v::Val, ::Type{T}) where {T <: Union{Float32, Float64}} =
    _model_applies(v) ?
    something(_modelled_blocking(target_profile(), T), _fallback_blocking(T)) :
    _fallback_blocking(T)

# For a bare scalar type: the fallback row, independent of the host.
default_blocking(::Type{Float64}) = _fallback_blocking(Float64)
default_blocking(::Type{Float32}) = _fallback_blocking(Float32)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
