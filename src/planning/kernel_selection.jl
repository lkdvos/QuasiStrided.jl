# Kernel selection: register-tile shape rules, the closed shape menus, kernel
# construction from a shape, and plan-time demotion.

# Fraction of the `cld(Qm, mr)` register slivers `[s*mr, min((s+1)*mr,
# Qm))` (`s` in `0:cld(Qm,mr)-1`) that lie ENTIRELY inside one run of length
# `run` -- i.e. `lo ÷ run == hi ÷ run`, `hi` the sliver's last valid index.
# By construction this is exactly `1.0` iff `Qm == run || run % mr == 0` (the
# predicate `_demote_for_run` has always used) -- checked over a randomized
# grid in `test/test_driver.jl`, not merely asserted. Feeds the
# `F2_BROKEN_ENOUGH` guard below. Precondition (T5 review, N4): `1 <= run <=
# Qm`, which every caller satisfies (`_leading_unit_run` never returns more
# than `Qm`, and `_demote_for_run`'s callers never pass `run == 0`); NOT
# checked here, so a hypothetical future caller passing `run > Qm` would get
# a wrong (too high) fraction silently rather than an error.
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

# Run-length-aware kernel-shape demotion (docs/decisions.md, "F2"). Applied
# DOWNSTREAM of `_default_kernel`/the M-N swap decision, on whichever
# orientation was actually chosen to feed M: `_store_tile_vector!` needs EVERY
# register sliver unit-stride in C, and given a leading unit-stride run of
# length `run`, that holds iff `Qm == run || run % mr(kernel) == 0` -- not the
# weaker `mr <= run` (verified by direct counterexample sweep,
# `benchmark/probes/probe_ccsd_t_stall_f2rule.jl`: e.g. run=20, mr=16 satisfies
# `mr <= run` but only 40% of slivers are actually contiguous). When the
# shipped default kernel's `mr` fails this predicate, every sliver falls to
# the slow scattered store; demoting to the LARGEST menu shape whose `mr`
# satisfies it (not the smallest -- measured: at run=16 for Float32, `(16,6,8)`
# beats `(8,6,8)`) reuses an already-compiled specialization from
# `kernel_shapes(T)`. Real dtypes only, currently: this was written back when
# complex kernels scatter-stored unconditionally, making the predicate moot
# for them. As of the planar vectorized store fast path
# (`_store_tile_planar_vector!`, `src/kernels/planar.jl`), that is no longer
# true, and this `T <: Real` bound is a DELIBERATELY DEFERRED, UNMEASURED
# follow-up (docs/proposals/complex-fast-paths.md Decision 3: ship the store
# fast path first, measure it, extend this guard as a separate later change),
# not a settled "moot" case -- the generic fallback method below is currently
# a no-op for complex, but need not stay one.
#
# LANDMINE for whoever picks this up: naively widening the bound to `T <:
# Number`/dropping `<: Real` here is NOT sufficient by itself. The loop below
# calls the single-argument `kernel_shapes(T)`, whose generic fallback for
# complex types returns only `_legacy_shape(T)` -- register pressure 30,
# explicitly warned against elsewhere in this file (~line 631-633, over
# AVX2's 16 ymm). The real complex kernel menus live behind the two-argument
# `kernel_shapes(T, method::ComplexMethod)` and `_complex_kernel_from_shape`,
# a different construction path entirely (see `_default_kernel(::Type{T})
# where {T<:Complex}` above). A correct fix must route the demotion search
# through `kernel_shapes(T, method)`/`_complex_kernel_from_shape`, not just
# relax this type bound. **Also applies to the K-depth guard immediately
# below**: it too is real-dtype-only today, for the same reason.
#
# K-depth guard (`Qk`, the contracted extent): a 28-point sweep
# (2026-09-22, docs/decisions.md, "tensorcontract-rs comparison") found F2
# fires unconditionally regardless of `Qk` and regresses once `Qk` grows
# past `F2_DEMOTE_KMAX_F64`/`F2_DEMOTE_KMAX_F32` -- the demotion's win comes
# from fixing the store path, but the packing/microkernel cost it also
# perturbs grows with `Qk` and eventually dominates. Declining to demote
# above that cutoff (or, once `F2_BROKEN_ENOUGH` moves below `1.0` and stops
# being inert) keeps the original shallow-K win (`ccsd_t_1`, `Qk=16`) while
# dropping the deep-K loss.
#
# Order matters for cost, not just correctness (T5 review, S1): the cheap
# `Qm == run` / `run % mr == 0` short-circuits run BEFORE the O(Qm/mr)
# `_unbroken_fraction` loop, and that loop is additionally gated on
# `F2_BROKEN_ENOUGH < 1.0` so it (and the fraction it would compute) is
# unreachable dead code while the guard is inert -- not merely "always
# false once reached". Equivalent to evaluating it first: whenever either
# short-circuit holds, `_unbroken_fraction` is provably `1.0` there too (by
# its own definition), so `> F2_BROKEN_ENOUGH` (`>= 1.0`) was always going
# to be false regardless of evaluation order; reordering only removes a
# wasted O(Qm/mr) scan on every real auto-selected plan with `Qk <= kmax`,
# it changes no decision.
function _demote_for_run(::Type{T}, kernel, run::Int, Qm::Int, Qk::Int) where {T <: Real}
    kmax = T === Float64 ? F2_DEMOTE_KMAX_F64 : F2_DEMOTE_KMAX_F32
    Qk > kmax && return kernel
    Qm == run && return kernel
    run % mr(kernel) == 0 && return kernel
    F2_BROKEN_ENOUGH < 1.0 && _unbroken_fraction(Qm, run, mr(kernel)) > F2_BROKEN_ENOUGH && return kernel
    best = nothing
    for shape in kernel_shapes(T)
        m = shape[1]
        if run % m == 0 && (best === nothing || m > best[1])
            best = shape
        end
    end
    return best === nothing ? kernel : _kernel_from_shape(best, T)
end
_demote_for_run(::Type{T}, kernel, run::Int, Qm::Int, Qk::Int) where {T} = kernel

# Engine-wide default kernel: shape from ONE detected capability, the vector
# register width -- `W = vector_bytes/sizeof(T)`, `MR = 2W`, `NR = NR_DEFAULT`,
# so `NV = 12`. Reproduces the swept optimum for both dtypes on AVX-512 and
# reduces to the previous hardcoded (8,6,4) on AVX2; NV = 12 is also the only
# setting that fits a 16-register AVX2 machine and Julia 1.10's register
# allocation (docs/decisions.md, Amendment 2 and Phases G/H).
const NR_DEFAULT = 6

_legacy_shape(::Type{T}) where {T} = (8, 6, _default_lanewidth(T))

# Complex counterpart: same `(8, 6)` register tile in LOGICAL (complex) rows,
# with the lane width taken from the real type, since every packed buffer and
# every `SIMD.Vec` below the kernel boundary is made of `real(T)`.
_legacy_shape(::Type{T}) where {T <: Complex} = (8, 6, _default_lanewidth(real(T)))

# Deliberately empty for the REAL types: after Phase H the rule above is
# already the measured optimum (1st of 24 for Float32, 2nd of 33 by 0.01% for
# Float64), so a row here would only pin this package to one machine's noise.
_shape_override(::Val, ::Type) = nothing

# For COMPLEX types the rule IS overridden, and these two rows are the only
# swept constants in the package. The derived `MR = 2W, NR = 6` shape measured
# as the *worst* planar configuration -- 38% (`ComplexF64`) / 41%
# (`ComplexF32`) off the best -- and `24x3`/`48x3` won outright. Ranking table,
# canary spread and the spill mechanism that predicted it: docs/decisions.md,
# "The register shape: the derived rule was wrong for complex by 38-41%".
#
# These are ccqlin038 measurements and are reached only on `:avx512`
# (`_rule_applies_complex`), so no other microarchitecture is handed them.
_shape_override(::Val{:avx512}, ::Type{ComplexF64}) = (24, 3, 8)
_shape_override(::Val{:avx512}, ::Type{ComplexF32}) = (48, 3, 16)

# NEON: **measured**, on an Apple M3 Max, by the sibline `tensorcontract-rs`
# project (`crates/tensorcontract/src/kernel/aarch64.rs`, `cfg_neon_f64` /
# `cfg_neon_f32`, three arms at `kc = 384`). Its planar winner is
# `(MV, NR) = (2, 6)` for both precisions, which converts to these two rows --
# `MR = MV * lanes`, lanes being 2 for `ComplexF64` and 4 for `ComplexF32` on
# 128-bit vectors.
#
# The register-budget fit already selected exactly these shapes, so this is a
# pin rather than a change. Pinned anyway, because reaching a measured optimum
# by coincidence is fragile: a later menu edit could move it silently, and
# nothing would notice.
_shape_override(::Val{:neon}, ::Type{ComplexF64}) = (4, 6, 2)
_shape_override(::Val{:neon}, ::Type{ComplexF32}) = (8, 6, 4)

# AVX2: **modelled, not measured** -- by anyone. Adopted from the same sibling
# project (`cfg_avx2_f64`, explicitly labelled "provisional and unmeasured"),
# whose planar choice is `(MV, NR) = (1, 5)`.
#
# It is adopted over what the budget fit picks -- `NR = 6` -- for one reason:
# `NR = 6` at `MV = 1` costs `2*6 + 2 + 2 = 16` registers out of AVX2's 16,
# leaving LLVM nothing for address arithmetic or loop counters, so it will
# spill something. `NR = 5` costs 14 and leaves two. The sibling's table
# records the same figure as `live 14`. The headroom argument is sound
# independently of whether 5 is the exact optimum, which is the part nobody
# has measured.
#
# This cannot be measured on `ccqlin038`: it has 32 registers, so forcing an
# AVX2 *shape* there would not exercise the 16-register constraint that
# motivates the row. It needs AVX2-only hardware, and is cheap to revisit --
# `benchmark/bench_complex_efficiency.jl` arm 2 is the sweep.
_shape_override(::Val{:avx2}, ::Type{ComplexF64}) = (4, 5, 4)
_shape_override(::Val{:avx2}, ::Type{ComplexF32}) = (8, 5, 8)

# The rule applies only to the ISAs it was validated on. `:neon` is detected
# but deliberately gets the legacy shape: no aarch64 measurement exists, and
# the rule would pick MR = 2W = 4 on 128-bit lanes -- narrower and smaller than
# the legacy (8,6,4), not obviously better. Derive where measured, fall back
# everywhere else.
_rule_applies(::Val{:avx512}) = true
_rule_applies(::Val{:avx2}) = true
_rule_applies(::Val) = false

# Complex counterpart, `:avx512` only, for the same kind of reason as the
# `:neon` fallback above: AVX2 has 16 ymm and planar holds separate real and
# imaginary accumulator planes, so even (MV,NR) = (1,6) leaves zero spare
# there (docs/decisions.md, "Cliff A").
_rule_applies_complex(::Val{:avx512}) = true
_rule_applies_complex(::Val) = false

function _derived_shape(profile::TargetProfile, ::Type{T}) where {T}
    vb = profile.vector_bytes
    key = Val(profile.isa)
    (_rule_applies(key) && vb > 0 && vb % sizeof(T) == 0) || return _legacy_shape(T)
    ovr = _shape_override(key, T)
    return ovr === nothing ? (2 * (vb ÷ sizeof(T)), NR_DEFAULT, vb ÷ sizeof(T)) : ovr
end

# A NEW METHOD, not an edit of the one above, so that "the real path is
# bit-identical" is a `git diff` fact rather than an argument about whether
# `real(Float64) === Float64`.
#
# The Phase G rule survives unchanged; only the `sizeof` argument moves to the
# real type, because `W` is a count of REAL lanes -- the whole complex
# adaptation. Factored out of `_derived_shape` so it can be tested
# independently of the override layered on top of it (see `_shape_override`,
# which unlike the real path is NOT empty here).
_complex_rule_shape(vb::Int, ::Type{T}) where {T <: Complex} =
    (2 * (vb ÷ sizeof(real(T))), NR_DEFAULT, vb ÷ sizeof(real(T)))

# Precedence, uniform across ISAs: an explicit override row, then the derived
# rule where it is validated, then a register-budget fit. The override is
# consulted FIRST and for every ISA -- an earlier revision checked
# `_rule_applies_complex` before it, which meant an AVX2 or NEON row could
# never be reached, so the non-AVX-512 rows below would have been dead code.
function _derived_shape(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    key = Val(profile.isa)
    ovr = _shape_override(key, T)
    ovr === nothing || return ovr
    R = real(T)
    vb = profile.vector_bytes
    (_rule_applies_complex(key) && vb > 0 && vb % sizeof(R) == 0) ||
        return _complex_fitted_shape(profile, T)
    return _complex_rule_shape(vb, T)
end

# Closed set, so compiled SIMDKernel (and driver) specializations are bounded.
const KERNEL_SHAPES_F64 = ((8, 6, 4), (16, 6, 8))
const KERNEL_SHAPES_F32 = ((8, 6, 8), (32, 6, 16), (16, 6, 8))

# F2 (run-length demotion, `_demote_for_run` above) regresses large-K cases;
# measured via a 28-point sweep, 2026-09-22 (docs/decisions.md,
# "tensorcontract-rs comparison"): demotion crosses over from a win to a loss
# between Qk=32 and Qk=64 for Float64 (kmax=32) and between Qk=64 and Qk=128
# for Float32 (kmax=64).
const F2_DEMOTE_KMAX_F64 = 32
const F2_DEMOTE_KMAX_F32 = 64
# The "broken-enough" fraction guard tested in the same sweep gave
# conflicting evidence between dtypes (Float64 supported adopting 0.75;
# Float32 contradicted it, with demotion winning at small Qk even at a
# less-broken 0.8 default fraction) -- kept as ONE constant per the
# planning contract's instruction not to split by dtype, set inert (1.0,
# i.e. never blocks demotion on its own) pending better evidence. This
# means the Qk cutoff above is the only active guard; known residual: on
# less-broken shapes, demotion still fires (and measurably loses, up to
# 37.4% over the pointwise-optimal choice at some Qk) between Qk=1 and
# kmax(T) -- recorded, not fixed, see docs/decisions.md.
const F2_BROKEN_ENOUGH = 1.0

# Complex menus, seeded from the reference's measured AVX-512 shapes, with `MR`
# in LOGICAL complex rows and `W` in real lanes; at most three each, so the
# compiled specialization set stays bounded. The 1m menus look "unaligned"
# (MR = 12 at W = 8) only because 1m runs a real microkernel of `2MR` rows, so
# it is `2*12 = 24` that must be a multiple of `W`. Ordered with the Phase F
# winner first, so the menu head and the shape the engine resolves to agree
# (see `_shape_override`); the SET is unchanged, only the order, so no
# specialization is added or removed (pinned by a test).
#
# The last three entries of each PLANAR menu exist for
# `_complex_fitted_shape` to select off `:avx512`: an `MV = 1` tile at each
# lane width the package compiles, so that for any (lane count, register
# budget >= 16) pair at least one entry fits. They are here
# because `_complex_kernel_from_shape` is `@generated` over this menu and falls
# through to the LAST entry on no match: a resolver that returned a shape
# absent from the menu would silently get a different kernel than it asked for,
# which is worse than either a spill or an error. A test pins that every ISA's
# fitted shape is present. They are unmeasured and are not claimed to be good,
# only to fit.
const KERNEL_SHAPES_C64_PLANAR = (
    (24, 3, 8), (16, 6, 8), (8, 8, 8), (4, 5, 4), (4, 6, 2), (2, 6, 2),
)
const KERNEL_SHAPES_C64_ONEM = ((12, 8, 8), (16, 6, 8), (8, 8, 8))
const KERNEL_SHAPES_C32_PLANAR = (
    (48, 3, 16), (32, 6, 16), (16, 8, 16), (8, 5, 8), (8, 6, 4), (4, 6, 4),
)
const KERNEL_SHAPES_C32_ONEM = ((24, 8, 16), (32, 6, 16), (16, 8, 16))

kernel_shapes(::Type{Float64}) = KERNEL_SHAPES_F64
kernel_shapes(::Type{Float32}) = KERNEL_SHAPES_F32
kernel_shapes(::Type{T}) where {T} = (_legacy_shape(T),)

"""
    kernel_shapes(T, method::ComplexMethod) -> NTuple{<:Any,NTuple{3,Int}}

Method-aware menu. Each complex method has its own measured menu because each
has its own packed A format and therefore its own register budget; `RealMethod`
forwards to the one-argument form so the real path is reached by exactly the
code it always was.
"""
kernel_shapes(::Type{T}, ::RealMethod) where {T} = kernel_shapes(T)
kernel_shapes(::Type{ComplexF64}, ::PlanarMethod) = KERNEL_SHAPES_C64_PLANAR
kernel_shapes(::Type{ComplexF64}, ::OneMMethod) = KERNEL_SHAPES_C64_ONEM
kernel_shapes(::Type{ComplexF32}, ::PlanarMethod) = KERNEL_SHAPES_C32_PLANAR
kernel_shapes(::Type{ComplexF32}, ::OneMMethod) = KERNEL_SHAPES_C32_ONEM
# Any other element type / method pair: the legacy shape, never a guess.
kernel_shapes(::Type{T}, ::ComplexMethod) where {T} = (_legacy_shape(T),)

# Unrolled over `kernel_shapes(T)` so every branch builds a concrete kernel
# from literal `Val`s (the last shape is the fallback); a plain loop would
# construct `Val(cand[1])` dynamically and widen to `Any`. Costs one dynamic
# dispatch per `plan_contract`, none per tile or K step -- `_plan_contract`
# specializes, so `execute!` sees no abstract type.
@generated function _kernel_from_shape(shape::Tuple{Int, Int, Int}, ::Type{T}) where {T}
    shapes = kernel_shapes(T)
    ex = :(SIMDKernel(Val($(shapes[end][1])), Val($(shapes[end][2])), T, Val($(shapes[end][3]))))
    for (MR, NR, W) in reverse(shapes[1:(end - 1)])
        ex = :(
            shape === ($MR, $NR, $W) ? SIMDKernel(Val($MR), Val($NR), T, Val($W)) : $ex
        )
    end
    return ex
end

@noinline _kernel_for(profile::TargetProfile, ::Type{T}) where {T} =
    _kernel_from_shape(_derived_shape(profile, T), T)

_default_kernel(::Type{T}) where {T} = _kernel_for(target_profile(), T)

# ----------------------------------------------------------------------------
# Complex kernel construction
# ----------------------------------------------------------------------------
# The complex counterpart of `_kernel_from_shape`, generated for the same
# reason: every branch must build a concrete kernel from literal `Val`s.
# An unimplemented method's arm throws rather than silently falling back to
# planar, since a silent method substitution would make a planar-vs-1m
# measurement meaningless.
@generated function _complex_kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method::PlanarMethod
    ) where {T}
    shapes = kernel_shapes(T, PlanarMethod())
    ex = :(PlanarKernel(Val($(shapes[end][1])), Val($(shapes[end][2])), T, Val($(shapes[end][3]))))
    for (MR, NR, W) in reverse(shapes[1:(end - 1)])
        ex = :(
            shape === ($MR, $NR, $W) ? PlanarKernel(Val($MR), Val($NR), T, Val($W)) : $ex
        )
    end
    return ex
end

# The 1m arm, over 1m's OWN menu (each method has its own packed A format and
# therefore its own register budget). Deliberately a second method rather than
# a shared generic over `M <: ComplexMethod`, so the planar arm stays
# byte-identical.
#
# Nothing in the engine reaches this arm: `_default_complex_method` always
# returns `PlanarMethod()` and 1m is selectable ONLY by naming the kernel (no
# auto-dispatch, no env var, no shape rule -- docs/decisions.md, "Method
# ranking does not transfer between machines"). It exists so the function is
# total for callers that name the method themselves.
@generated function _complex_kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method::OneMMethod
    ) where {T}
    shapes = kernel_shapes(T, OneMMethod())
    ex = :(OneMKernel(Val($(shapes[end][1])), Val($(shapes[end][2])), T, Val($(shapes[end][3]))))
    for (MR, NR, W) in reverse(shapes[1:(end - 1)])
        ex = :(
            shape === ($MR, $NR, $W) ? OneMKernel(Val($MR), Val($NR), T, Val($W)) : $ex
        )
    end
    return ex
end

@noinline function _complex_kernel_from_shape(
        shape::Tuple{Int, Int, Int}, ::Type{T}, method
    ) where {T}
    throw(
        ArgumentError(
            "no microkernel is available for $T at shape $shape under $(method). " *
                "Only $(PlanarMethod()) and $(OneMMethod()) are implemented. Pass an " *
                "explicit `kernel = ...` to plan_contract to use a kernel this " *
                "engine does not pick itself."
        )
    )
end

"""
    _planar_pressure(MR, NR, W) -> Int

Vector registers a planar kernel holds live per K step: `2*MV*NR`
accumulators (two planes) + `2*MV` A vectors (two planes) + 2 B broadcasts,
with `MV = MR ÷ W`.

A *necessary* condition only, not a predictor. Phase D measured that spilling
is not monotone in this number -- planar `(24,3,8)` at 26 is clean while
`(8,8,8)` at 20 spills -- so it is used below to *exclude* shapes that cannot
possibly fit, never to rank the ones that can.
"""
_planar_pressure(MR::Int, NR::Int, W::Int) = 2 * (MR ÷ W) * NR + 2 * (MR ÷ W) + 2

# Largest MENU shape that this host can actually run, for use off `:avx512`.
#
# AMENDS Phase C's decision to refuse a complex default off `:avx512`
# entirely. That was wrong, and CI said so: every runner this package has is
# AVX2 (Linux) or NEON (macOS), so refusing made complex support unavailable
# through `@tensor` on every machine the project tests on, and on most machines
# anyone would run it on. The priority was backwards -- a slow-but-correct
# kernel beats no complex support at all, and "an error beats a guaranteed
# spill" is only defensible when the user has an alternative, which they did
# not.
#
# It selects **from the menu** rather than computing a shape freely, so
# "whatever this returns is in the menu" holds by construction rather than by
# having enumerated the right hardware. That matters because
# `_complex_kernel_from_shape` is `@generated` over the menu and falls through
# to its LAST entry on no match: a freely computed shape absent from the menu
# would silently build a different kernel than was asked for. An earlier
# revision did compute freely, and the test that sweeps synthetic
# `(vector_bytes, nregisters)` pairs caught exactly that.
#
# Two constraints, both necessary:
#
#   * `W <= hardware lanes` -- a `Vec{8,Float64}` on 128-bit NEON is emulated
#     across four registers, so a shape whose pressure "fits" on paper would
#     not fit at all.
#   * `_planar_pressure <= nregisters` -- see that function; a necessary
#     condition, never used here to rank.
#
# `nregisters == 0` (unrecognised CPU) assumes 16, the conservative x86
# baseline, matching the conservatism `_legacy_shape` already applies.
#
# The selected shapes are UNMEASURED off `:avx512` and are not claimed to be
# good, only to run without spilling by the budget's own reckoning.
function _complex_fitted_shape(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    R = real(T)
    vb = profile.vector_bytes
    lanes = (vb > 0 && vb % sizeof(R) == 0) ? vb ÷ sizeof(R) : _default_lanewidth(R)
    budget = profile.nregisters > 0 ? profile.nregisters : 16
    best = nothing
    for shape in kernel_shapes(T, PlanarMethod())
        MR, NR, W = shape
        (W <= lanes && MR % W == 0) || continue
        _planar_pressure(MR, NR, W) <= budget || continue
        # Largest logical tile wins; ties by the wider vector.
        if best === nothing || (MR * NR, W) > (best[1] * best[2], best[3])
            best = shape
        end
    end
    # Unreachable for any budget >= 16 at any lane width the package compiles:
    # each menu carries an `MV = 1` entry per width, whose pressure is
    # `2*NR + 4 = 16` at `NR = 6`. Kept as a total fallback rather than an
    # assertion because returning a correct-but-slow kernel is always better
    # than throwing here -- this is the code path that CI proved must not
    # refuse.
    return best === nothing ? last(kernel_shapes(T, PlanarMethod())) : best
end

# The unconditional default; `OneMMethod` is selected only by naming the kernel
# (docs/decisions.md, "Method ranking does not transfer between machines").
_default_complex_method(::Type{<:Complex}) = PlanarMethod()

@noinline function _kernel_for(profile::TargetProfile, ::Type{T}) where {T <: Complex}
    return _complex_kernel_from_shape(
        _derived_shape(profile, T), T, _default_complex_method(T)
    )
end

# Extent-aware variant, used only when the caller did not name a kernel: a
# contraction whose M extent cannot fill one register tile pads every
# micro-tile away, so fall back to the legacy shape. Keys on padding waste -- a
# countable quantity known at plan time -- not on a cache estimate, which is
# what distinguishes it from the refuted depth-adaptive MC.
@noinline function _default_kernel(::Type{T}, Qm::Int, Qn::Int) where {T}
    kernel = _kernel_for(target_profile(), T)
    (Qm > 0 && Qm < mr(kernel)) || return kernel
    legacy = _legacy_shape(T)
    return SIMDKernel(Val(legacy[1]), Val(legacy[2]), T, Val(legacy[3]))
end

# Structurally identical to the real method above, so the demotion rule has one
# definition in two places rather than two rules; it routes through the same
# seam.
@noinline function _default_kernel(::Type{T}, Qm::Int, Qn::Int) where {T <: Complex}
    profile = target_profile()
    kernel = _kernel_for(profile, T)
    (Qm > 0 && Qm < mr(kernel)) || return kernel
    # Demote to the budget-fitted shape, NOT to `_legacy_shape`: the latter is
    # `(8, 6, W)` at pressure 30, which is over AVX2's 16 ymm. The fitted shape
    # is the smallest thing guaranteed to be constructible on this host.
    small = _complex_fitted_shape(profile, T)
    return _complex_kernel_from_shape(small, T, _default_complex_method(T))
end
