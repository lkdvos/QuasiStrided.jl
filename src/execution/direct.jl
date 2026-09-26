"""
    execute_direct!(plan::ContractPlan, alpha::Number, beta::Number)

Unexported. The "skip packing" (BLIS "SUP", skinny-unpacked-style) direct path
for small contractions: computes the same result as [`execute!`](@ref) --
`C = alpha * sum_K atransform(A) * btransform(B) + beta * C` -- as a plain
scalar triple loop that reads `A`/`B`/`C` straight from `plan.Astorage`/
`plan.Bstorage`/`plan.Cstorage`. No packing, no register microkernel, no cache
blocking: `plan.kernel`, `plan.blocking` and `plan.workspace` are never
touched, so it works on any plan (including one built with `oracle = false`)
and has no notion of `mr`/`nr` register tiles. For shapes this small the
O(MK + NK) packing cost of the blocked path is not repaid by register-tile
reuse, which is the whole point of skipping it.

Deliberately opt-in: nothing in [`plan_contract`](@ref) or [`contract!`](@ref)
dispatches here; call it explicitly on a plan from `plan_contract`. Its
result agrees with [`execute!`](@ref) and [`execute_tilewise!`](@ref) up to
floating-point reassociation (the K accumulation order differs).

Semantics match `execute!`: empty output is a no-op; empty K or `alpha == 0`
applies `beta` once to every element of `C` without reading `A`/`B`; `beta`
applies exactly once per output element; `beta == 0` never reads old `C` and
`beta == 1` adds without scaling, the same BLAS-like shortcuts as the
microkernels' stores. `atransform`/`btransform` (`identity` or `conj`) are
applied to each whole loaded element, as `pack_a!`/`pack_b!` do.

**Not allocation-free**, unlike `execute!`/`execute_tilewise!`: each call
allocates six `Vector{Int}` offset maps (lengths `Qm`, `Qm`, `Qn`, `Qn`,
`Qk`, `Qk`). Uses ordinary bounds-checked indexing throughout. Returns
`plan.Cstorage`.

**Measured, 2026-09-25, jobs 7109278 (ccq rome/znver2, AVX2) and 7109279 (ccq
icelake-server, AVX-512); see `benchmark/bench_skip_packing.jl` and its
`benchmark/submit_skip_packing.sh`**: this path is NEVER faster than
`execute!` at any swept shape, on either ISA. Best case is near-parity at
1x1x1/2x2x2 (0.81-0.99x); by 64^3 it is 25-40x SLOWER, and it does not improve
on the small-K family (`Qk in (1, 8, 63)` at M,N up to 256) that motivated this
work either (0.02-0.18x there). The scalar loop's own throughput is pinned
near 1-1.25 GFLOP/s regardless of shape -- it never vectorizes -- while the
packed path already reaches 30-90 GFLOP/s and costs as little as ~150ns at
1x1x1. So the packing-avoidance premise this path was built to test is FALSE
as measured: at these sizes packing's O(MK+NK) cost is not what the packed
path is paying for, and skipping it while staying scalar throws away far more
(vectorization) than it saves. **Do not wire this into `plan_contract`'s
automatic dispatch or any `contract!` opt-in kwarg** -- there is no shape
where it measurably wins. It remains useful only as a third independent
correctness check alongside `execute_tilewise!`, both exercised in
`test/execution/test_direct.jl`. A real BLIS-SUP-style win would need a
still-vectorized small-shape kernel (e.g. reading contiguous/lightly-strided
operands straight into SIMD registers without packing), which is a materially
larger, unmeasured follow-up, not a variant of this function.
"""
function execute_direct!(plan::ContractPlan{T}, alpha::Number, beta::Number) where {T}
    alphaT = convert(T, alpha)
    betaT = convert(T, beta)

    Qm = axis_length(plan.mgroup)
    Qn = axis_length(plan.ngroup)
    Qk = axis_length(plan.kgroup)

    # Empty output: no-op, nothing read or written at all.
    (Qm == 0 || Qn == 0) && return plan.Cstorage

    # Map order matches `_execute_nest!`: mgroup is (A, C), ngroup is (B, C),
    # kgroup is (A, B).
    aOffM = Vector{Int}(undef, Qm)
    cOffM = Vector{Int}(undef, Qm)
    fill_offsets!((aOffM, cOffM), plan.mgroup, 0, Qm)
    bOffN = Vector{Int}(undef, Qn)
    cOffN = Vector{Int}(undef, Qn)
    fill_offsets!((bOffN, cOffN), plan.ngroup, 0, Qn)

    C = plan.Cstorage
    Cbase = plan.Cbase

    # Nothing to contract: beta-only pass, A and B never read.
    if Qk == 0 || iszero(alphaT)
        _direct_scale_C!(C, Cbase, cOffM, cOffN, betaT)
        return C
    end

    aOffK = Vector{Int}(undef, Qk)
    bOffK = Vector{Int}(undef, Qk)
    fill_offsets!((aOffK, bOffK), plan.kgroup, 0, Qk)

    return _direct_nest!(
        C, Cbase, plan.Astorage, plan.Abase, plan.Bstorage, plan.Bbase,
        aOffM, cOffM, bOffN, cOffN, aOffK, bOffK,
        plan.atransform, plan.btransform, alphaT, betaT
    )
end

# Function barrier: every argument is concretely typed here, so the triple
# loop specializes on the storage types and on the transform singletons.
function _direct_nest!(
        C, Cbase::Int, A, Abase::Int, B, Bbase::Int,
        aOffM::Vector{Int}, cOffM::Vector{Int},
        bOffN::Vector{Int}, cOffN::Vector{Int},
        aOffK::Vector{Int}, bOffK::Vector{Int},
        atransform::FA, btransform::FB, alphaT::T, betaT::T
    ) where {FA, FB, T}
    Qm = length(aOffM)
    Qn = length(bOffN)
    Qk = length(aOffK)
    for n in 1:Qn
        bn = Bbase + bOffN[n]
        cn = Cbase + cOffN[n]
        for m in 1:Qm
            am = Abase + aOffM[m]
            acc = zero(T)
            for k in 1:Qk
                a = atransform(A[am + aOffK[k] + 1])
                b = btransform(B[bn + bOffK[k] + 1])
                acc += a * b
            end
            idx = cn + cOffM[m] + 1
            C[idx] = if iszero(betaT)
                alphaT * acc
            elseif isone(betaT)
                alphaT * acc + C[idx]
            else
                alphaT * acc + betaT * C[idx]
            end
        end
    end
    return C
end

# Element-by-element `C *= beta` with `scale_tile!`'s shortcuts: `beta == 1`
# touches nothing, `beta == 0` writes zeros without reading old `C`.
function _direct_scale_C!(C, Cbase::Int, cOffM::Vector{Int}, cOffN::Vector{Int}, betaT::T) where {T}
    isone(betaT) && return C
    for cn in cOffN, cm in cOffM
        idx = Cbase + cm + cn + 1
        C[idx] = iszero(betaT) ? zero(T) : C[idx] * betaT
    end
    return C
end
