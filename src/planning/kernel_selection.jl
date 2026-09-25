# Kernel selection: which microkernel `plan_contract` builds when the caller
# does not name one.
#
# Everything is keyed on (element type `T`, method), where the method is
# `RealMethod()` for a real `T` and `PlanarMethod()` for a complex one
# (`_default_method`); `OneMMethod()` is never chosen here, only by naming the
# kernel. Per method there is
#
#   * a closed menu of `(MR, NR, W)` shapes (`kernel_shapes`), so the set of
#     compiled kernel and driver specializations stays bounded;
#   * a kernel type (`_kernel_type`), built from a menu shape by
#     `_kernel_from_shape`;
#   * a shape resolved from the detected hardware (`_derived_shape`): an
#     explicit override row, else the `MR = 2W` rule where it has been
#     validated, else a conservative fitted shape (`_fitted_shape`);
#   * two plan-time demotions: to the fitted shape when M cannot fill one
#     register tile (`_default_kernel`), and to a menu shape that keeps every
#     register sliver unit-stride in C (`_demote_for_run`).

# ----------------------------------------------------------------------------
# Methods, menus and kernel types
# ----------------------------------------------------------------------------

# The method the engine uses for `T`; `OneMMethod` is selected only by naming
# the kernel, since method ranking does not transfer between machines.
_default_method(::Type{<:Real}) = RealMethod()
_default_method(::Type{<:Complex}) = PlanarMethod()

const NR_DEFAULT = 6

# The conservative shape: a `(8, 6)` register tile in logical rows, with the
# lane width of one 256-bit register of `real(T)`. It is the default wherever
# no rule applies, and in every real menu.
_fallback_shape(::Type{T}) where {T} = (8, NR_DEFAULT, _default_lanewidth(real(T)))

# Menus. Complex `MR` counts logical (complex) rows and `W` real lanes. The 1m
# menus look "unaligned" (MR = 12 at W = 8) only because 1m runs a real
# microkernel of `2MR` rows, so it is `2MR` that must be a multiple of `W`.
# Each planar menu starts with the shape `_derived_shape` resolves to on
# `:avx512`; each real menu contains `_fallback_shape` and the AVX-512/AVX2
# rule shapes, and each 1m menu contains its AVX-512 rule shape.
#
# The last three entries of each planar menu are an `MV = 1` tile at each lane
# width the package compiles, so that `_fitted_shape` finds a fitting entry for
# any (lane count, register budget >= 16) pair. They are unmeasured and are not
# claimed to be good, only to fit.
const KERNEL_SHAPES_F64 = ((8, 6, 4), (16, 6, 8))
const KERNEL_SHAPES_F32 = ((8, 6, 8), (32, 6, 16), (16, 6, 8))
const KERNEL_SHAPES_C64_PLANAR = (
    (24, 3, 8), (16, 6, 8), (8, 8, 8), (4, 5, 4), (4, 6, 2), (2, 6, 2),
)
const KERNEL_SHAPES_C64_ONEM = ((12, 8, 8), (16, 6, 8), (8, 8, 8))
const KERNEL_SHAPES_C32_PLANAR = (
    (48, 3, 16), (32, 6, 16), (16, 8, 16), (8, 5, 8), (8, 6, 4), (4, 6, 4),
)
const KERNEL_SHAPES_C32_ONEM = ((24, 8, 16), (32, 6, 16), (16, 8, 16))

"""
    kernel_shapes(T, method::ComplexMethod = _default_method(T)) -> NTuple{<:Any,NTuple{3,Int}}

The closed menu of `(MR, NR, W)` register shapes the engine may build for
element type `T` under `method`. Each method has its own menu because each has
its own packed A format and therefore its own register budget.
"""
kernel_shapes(::Type{T}) where {T} = kernel_shapes(T, _default_method(T))
kernel_shapes(::Type{Float64}, ::RealMethod) = KERNEL_SHAPES_F64
kernel_shapes(::Type{Float32}, ::RealMethod) = KERNEL_SHAPES_F32
kernel_shapes(::Type{ComplexF64}, ::PlanarMethod) = KERNEL_SHAPES_C64_PLANAR
kernel_shapes(::Type{ComplexF64}, ::OneMMethod) = KERNEL_SHAPES_C64_ONEM
kernel_shapes(::Type{ComplexF32}, ::PlanarMethod) = KERNEL_SHAPES_C32_PLANAR
kernel_shapes(::Type{ComplexF32}, ::OneMMethod) = KERNEL_SHAPES_C32_ONEM

# The kernel type implementing `method` for `T`, or `nothing` when there is none
# (exactly the pairs `kernel_shapes` has a menu for).
_kernel_type(::RealMethod, ::Type{<:Union{Float32, Float64}}) = SIMDKernel
_kernel_type(::PlanarMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = PlanarKernel
_kernel_type(::OneMMethod, ::Type{<:Union{ComplexF32, ComplexF64}}) = OneMKernel
_kernel_type(::Any, ::Type) = nothing

"""
    _kernel_from_shape(shape, T, method = _default_method(T)) -> kernel

Build the `method` kernel for `T` at `shape`, which must be in
`kernel_shapes(T, method)`. Unrolled over the menu so that every branch builds
a concrete kernel from literal `Val`s (a plain loop would construct
`Val(shape[1])` dynamically and widen to `Any`); this costs one dynamic
dispatch per `plan_contract`, none per tile or K step. A shape outside the
menu, or a method with no kernel for `T`, throws rather than silently building
a different kernel than was asked for.
"""
_kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T} =
    _kernel_from_shape(shape, T, _default_method(T))

@generated function _kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method::M
    ) where {T, M}
    K = _kernel_type(M.instance, T)
    K === nothing && return :(_throw_no_kernel(shape, T, method))
    ex = :(_throw_shape_not_in_menu(shape, T, method))
    for (MR, NR, W) in reverse(kernel_shapes(T, M.instance))
        ex = :(shape === ($MR, $NR, $W) ? $K(Val($MR), Val($NR), T, Val($W)) : $ex)
    end
    return ex
end

@noinline _throw_no_kernel(shape, ::Type{T}, method) where {T} = throw(
    ArgumentError(
        "no microkernel is available for $T at shape $shape under $(method). " *
            "Pass an explicit `kernel = ...` to plan_contract to use a kernel this " *
            "engine does not pick itself."
    )
)

@noinline _throw_shape_not_in_menu(shape, ::Type{T}, method) where {T} = throw(
    ArgumentError(
        "shape $shape is not in the $(method) menu for $T, " *
            "$(kernel_shapes(T, method)); only menu shapes are compiled"
    )
)

# ----------------------------------------------------------------------------
# Shape resolution from the detected hardware
# ----------------------------------------------------------------------------

# Explicit per-ISA shapes, consulted first. Planar only. Empty for the real
# types: there the `MR = 2W` rule below is already the measured optimum, so a
# row would only pin the package to one machine's noise. 1m has none: it is
# never selected automatically.
_shape_override(key::Val, ::Type{T}) where {T} = _shape_override(key, T, _default_method(T))
_shape_override(::Val, ::Type, ::ComplexMethod) = nothing

# AVX-512, measured: the derived `MR = 2W, NR = 6` shape is the worst planar
# configuration (by 38-41%), and `24x3`/`48x3` win outright.
_shape_override(::Val{:avx512}, ::Type{ComplexF64}, ::PlanarMethod) = (24, 3, 8)
_shape_override(::Val{:avx512}, ::Type{ComplexF32}, ::PlanarMethod) = (48, 3, 16)

# NEON, measured on an Apple M3 Max by the sibling `tensorcontract-rs` project
# (planar winner `(MV, NR) = (2, 6)` for both precisions). The register-budget
# fit selects the same shapes; the rows pin them so a later menu edit cannot
# move them silently.
_shape_override(::Val{:neon}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 6, 2)
_shape_override(::Val{:neon}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 6, 4)

# AVX2, measured: `benchmark/bench_complex_efficiency.jl` arm 2 on a Rome
# (znver2) node (2026-09-24, job 7102205) ranks every menu shape for both
# planar and 1m; `(4, 5, 4)` wins ComplexF64 outright (next best, 1m `8x8/W8`,
# is 1.397x slower) and `(8, 5, 8)` wins ComplexF32 outright (next best, 1m
# `16x8/W16`, is 1.476x slower) -- confirming the register-budget reasoning
# these rows originally shipped with (at `MV = 1`, `NR = 6` costs all 16 of
# AVX2's vector registers (`2*6 + 2 + 2`), leaving none for address
# arithmetic, while `NR = 5` costs 14). No larger menu shape closes the gap to
# StridedBLAS on this ISA -- that gap is a throughput ceiling, not a
# shape-selection miss.
_shape_override(::Val{:avx2}, ::Type{ComplexF64}, ::PlanarMethod) = (4, 5, 4)
_shape_override(::Val{:avx2}, ::Type{ComplexF32}, ::PlanarMethod) = (8, 5, 8)

# Where the `MR = 2W` rule is validated. Real: AVX-512 and AVX2 (on NEON it
# would pick MR = 4 on 128-bit lanes, not obviously better than the fallback).
# Complex: AVX-512 only -- planar holds separate real and imaginary
# accumulator planes, so on AVX2's 16 registers even `(MV, NR) = (1, 6)`
# leaves nothing spare (Cliff A, src/microkernels/planar.jl).
_rule_applies(::Val{:avx512}, ::RealMethod) = true
_rule_applies(::Val{:avx512}, ::Union{PlanarMethod, OneMMethod}) = true
_rule_applies(::Val{:avx2}, ::RealMethod) = true
_rule_applies(::Val, ::ComplexMethod) = false

# `W` is one vector register's worth of REAL lanes, `MR = 2W`, `NR = NR_DEFAULT`.
_rule_shape(vb::Int, ::Type{T}) where {T} =
    (2 * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

"""
    _derived_shape(profile::TargetProfile, T, method = _default_method(T)) -> (MR, NR, W)

The register shape for `T` under `method` on `profile`, in precedence order:
an explicit `_shape_override` row, then `_rule_shape` where `_rule_applies`
and the rule's shape is in the menu, then `_fitted_shape`. Always a member of
`kernel_shapes(T, method)`.
"""
_derived_shape(profile::TargetProfile, ::Type{T}) where {T} =
    _derived_shape(profile, T, _default_method(T))

function _derived_shape(profile::TargetProfile, ::Type{T}, method) where {T}
    key = Val(profile.isa)
    ovr = _shape_override(key, T, method)
    ovr === nothing || return ovr
    vb = profile.vector_bytes
    if _rule_applies(key, method) && vb > 0 && vb % sizeof(real(T)) == 0
        shape = _rule_shape(vb, T)
        shape in kernel_shapes(T, method) && return shape
    end
    return _fitted_shape(profile, T, method)
end

"""
    _planar_pressure(MR, NR, W) -> Int

Vector registers a planar kernel holds live per K step: `2*MV*NR`
accumulators (two planes) + `2*MV` A vectors (two planes) + 2 B broadcasts,
with `MV = MR ÷ W`.

A *necessary* condition only, not a predictor: spilling is not monotone in
this number (planar `(24,3,8)` at 26 is clean while `(8,8,8)` at 20 spills),
so it is used to *exclude* shapes that cannot possibly fit, never to rank the
ones that can.
"""
_planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

"""
    _fitted_shape(profile::TargetProfile, T, method) -> (MR, NR, W)

The shape used where no rule applies, and the target of extent demotion: the
smallest thing guaranteed to be constructible on this host.

Real: `_fallback_shape(T)`. Planar: the largest *menu* shape that fits the
host -- selected from the menu so that membership holds by construction --
under two necessary constraints: `W <= hardware lanes` (a `Vec{8,Float64}` on
128-bit NEON is emulated across four registers) and `_planar_pressure <=
nregisters` (16 when the register count is unknown, the conservative x86
baseline). Not `_fallback_shape`, whose pressure of 30 is over AVX2's 16. The
fitted shapes are unmeasured off `:avx512`; they are not claimed to be good,
only to run without spilling by the budget's own reckoning.
"""
_fitted_shape(::TargetProfile, ::Type{T}, ::RealMethod) where {T} = _fallback_shape(T)

# 1m's menus hold AVX-512 shapes only; its head is spill-free at every lane
# width it compiles, so it is the conservative choice.
_fitted_shape(::TargetProfile, ::Type{T}, method::OneMMethod) where {T} =
    first(kernel_shapes(T, method))

function _fitted_shape(profile::TargetProfile, ::Type{T}, method::PlanarMethod) where {T}
    R = real(T)
    vb = profile.vector_bytes
    lanes = (vb > 0 && vb % sizeof(R) == 0) ? vb ÷ sizeof(R) : _default_lanewidth(R)
    budget = profile.nregisters > 0 ? profile.nregisters : 16
    best = nothing
    for shape in kernel_shapes(T, method)
        MR, NR, W = shape
        (W <= lanes && MR % W == 0) || continue
        _planar_pressure(MR, NR, W) <= budget || continue
        # Largest logical tile wins; ties by the wider vector.
        if best === nothing || (MR * NR, W) > (best[1] * best[2], best[3])
            best = shape
        end
    end
    # Unreachable for any budget >= 16 (see the menu comment); a correct but
    # slow kernel beats throwing here.
    return best === nothing ? last(kernel_shapes(T, method)) : best
end

# ----------------------------------------------------------------------------
# The engine's default kernel, and plan-time demotion
# ----------------------------------------------------------------------------

@noinline _kernel_for(profile::TargetProfile, ::Type{T}) where {T} =
    _kernel_from_shape(_derived_shape(profile, T), T, _default_method(T))

_default_kernel(::Type{T}) where {T} = _kernel_for(target_profile(), T)

# Extent-aware variant, used only when the caller did not name a kernel: a
# contraction whose M extent cannot fill one register tile pads every
# micro-tile away, so demote to the fitted shape. Keys on padding waste, a
# countable quantity known at plan time, not on a cache estimate.
@noinline function _default_kernel(::Type{T}, Qm::Int, Qn::Int) where {T}
    profile = target_profile()
    kernel = _kernel_for(profile, T)
    (Qm > 0 && Qm < mr(kernel)) || return kernel
    method = _default_method(T)
    return _kernel_from_shape(_fitted_shape(profile, T, method), T, method)
end

# Run-length-aware demotion, applied to whichever operand the M/N swap decision
# chose to feed M. The vectorized store needs EVERY register sliver unit-stride
# in C; given a leading unit-stride run of
# length `run` in C, that holds iff `Qm == run || run % mr(kernel) == 0` (not
# the weaker `mr <= run`: at run=20, mr=16 only 40% of slivers are
# contiguous). When the kernel fails this, demote to the LARGEST menu shape
# whose `mr` divides `run` (at run=16 for Float32, `(16,6,8)` beats
# `(8,6,8)`), which reuses an already-compiled specialization.
#
# The demotion wins by fixing the store, but it also changes the packing and
# microkernel cost, which grows with the contracted extent `Qk`; past
# `_RUN_DEMOTE_KMAX_*` that cost dominates and demoting loses, so the guard
# declines to demote there.
#
# Real element types only. Complex kernels now have a vectorized store too, so
# extending this is a deliberately deferred, unmeasured follow-up. The menu
# search below already goes through the kernel's own method, but lifting the
# guard would also need `plan_contract` to compute `run_m`/`run_n` for complex
# `T` (it passes `0` today) and a measured complex `Qk` cutoff.
#
# Order matters for cost: the cheap `Qm == run` / `run % mr == 0`
# short-circuits run before the O(Qm/mr) `_unbroken_fraction` scan, which is
# additionally skipped while `_RUN_DEMOTE_BROKEN_ENOUGH` is inert (`1.0`). Whenever a
# short-circuit holds the fraction is `1.0` anyway, so the order changes no
# decision.
function _demote_for_run(::Type{T}, kernel, run::Int, Qm::Int, Qk::Int) where {T}
    T <: Real || return kernel
    kmax = T === Float64 ? _RUN_DEMOTE_KMAX_F64 : _RUN_DEMOTE_KMAX_F32
    Qk > kmax && return kernel
    Qm == run && return kernel
    run % mr(kernel) == 0 && return kernel
    _RUN_DEMOTE_BROKEN_ENOUGH < 1.0 && _unbroken_fraction(Qm, run, mr(kernel)) > _RUN_DEMOTE_BROKEN_ENOUGH &&
        return kernel
    method = complex_method(kernel)
    best = nothing
    for shape in kernel_shapes(T, method)
        m = shape[1]
        if run % m == 0 && (best === nothing || m > best[1])
            best = shape
        end
    end
    return best === nothing ? kernel : _kernel_from_shape(best, T, method)
end

# Deepest `Qk` at which run-length demotion still wins: it crosses over from a
# win to a loss between Qk=32 and 64 for Float64 and between 64 and 128 for
# Float32.
const _RUN_DEMOTE_KMAX_F64 = 32
const _RUN_DEMOTE_KMAX_F32 = 64

# Demotion is also skipped when more than this fraction of the register
# slivers already lie inside one run. Inert at `1.0`: the evidence conflicts
# between Float64 and Float32 and one shared threshold cannot satisfy both, so
# the `Qk` cutoff above is the only active guard. Known residual: on
# less-broken shapes demotion still fires, and can lose, for `Qk <= kmax(T)`.
const _RUN_DEMOTE_BROKEN_ENOUGH = 1.0

# Fraction of the `cld(Qm, mr)` register slivers `[s*mr, min((s+1)*mr, Qm))`
# that lie ENTIRELY inside one run of length `run`, i.e. `lo ÷ run == hi ÷
# run` for the sliver's first and last valid index. Exactly `1.0` iff
# `Qm == run || run % mr == 0`, the predicate `_demote_for_run` tests first.
# Precondition, not checked: `1 <= run <= Qm` (`_leading_unit_run` never
# returns more than `Qm`, and callers never pass `run == 0`).
function _unbroken_fraction(Qm::Int, run::Int, mr::Int)::Float64
    nslivers = cld(Qm, mr)
    nslivers == 0 && return 1.0
    whole = 0
    for s in 0:(nslivers - 1)
        lo = s * mr
        hi = min((s + 1) * mr, Qm) - 1
        whole += lo ÷ run == hi ÷ run
    end
    return whole / nslivers
end
