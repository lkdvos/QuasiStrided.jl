# Fixtures and bindings shared across test files. Every test file is included
# into the same scope by runtests.jl, so this file is included first.

using StridedViews: StridedView, offset

# plan_contract/execute!/ContractPlan aren't exported (only contract! is);
# execute_tilewise! is never exported at all (it's an internal oracle).
# `test/runtests.jl` deliberately leaves these four out of its own
# name-restoring `using QuasiStrided: ...` block, because a `const` may not
# shadow an imported binding; the workspace API is reached as
# `QuasiStrided.<name>` for the same reason.
const plan_contract = QuasiStrided.plan_contract
const execute! = QuasiStrided.execute!
const ContractPlan = QuasiStrided.ContractPlan
const execute_tilewise! = QuasiStrided.execute_tilewise!

# TO names are always qualified: a bare `using TensorOperations` collides with
# QuasiStrided's `scalartype`.
import TensorOperations as TO

# Plan for the dense matmul C[m,n] = sum_k A[m,k]*B[k,n], the shape most
# testsets below use; every `plan_contract` keyword is forwarded verbatim, so
# an omitted one takes plan_contract's own default. (execution/test_macro_blocking.jl has
# its own copy: both files are included into the same scope, so the names must
# differ.)
function _mm_plan(Cmat, Amat, Bmat; kwargs...)
    return plan_contract(
        StridedView(Cmat), StridedView(Amat), (1, 2), StridedView(Bmat), (2, 3), (1, 3);
        kwargs...
    )
end

# Steady-state allocation of one execute!/execute_tilewise! call on a reused
# plan: warm up (compile) first, then measure.
function _steady_allocs!(run!, plan, Cmat)
    run!(plan, 1.0, 0.0)
    fill!(Cmat, 0.0)
    return @allocated run!(plan, 1.0, 0.0)
end

# Worked fixture: A[a,k,b] (3,5,2), B[k,n] (5,4), C[a,n,b] (3,4,2),
# C[a,n,b] = sum_k A[a,k,b]*B[k,n]; labels a=1,k=2,b=3,n=4.

function _worked_fixture()
    A = reshape(collect(1.0:30.0), 3, 5, 2)
    B = reshape(collect(1.0:20.0), 5, 4)
    Cref = zeros(Float64, 3, 4, 2)
    for b in 1:2, n in 1:4, a in 1:3
        Cref[a, n, b] = sum(A[a, k, b] * B[k, n] for k in 1:5)
    end
    return A, B, Cref
end

const _INDA = (1, 2, 3)
const _INDB = (2, 4)
const _INDC = (1, 4, 3)


using QuasiStrided: TargetProfile, CacheLevel

const VALID_ISAS = (:avx512, :avx2, :neon, :unknown)
# `nregisters` defaults to 32; the complex shape-fitting tests pass the real
# per-ISA count.
synthetic(isakey, vb; nregisters::Int = 32) = TargetProfile(
    isakey, Sys.ARCH, "synthetic", vb, nregisters,
    CacheLevel(), CacheLevel(), CacheLevel()
)

# Permuted A / negative-stride B / sliced-with-offset C, the fixture shape
# benchmark/harness.jl uses.
function scattered_fixture(::Type{T}, a_n = 32, k_n = 32, b_n = 8, n_n = 32) where {T}
    A2 = randn(MersenneTwister(11), T, a_n, k_n)
    Aperm = permutedims(StridedView(vec(A2), (a_n, k_n, b_n), (1, a_n, 0), 0), (2, 3, 1))
    Bneg = StridedView(randn(MersenneTwister(12), T, k_n * n_n), (k_n, n_n), (-1, k_n), k_n - 1)
    Cbig = zeros(T, a_n + 2, n_n + 3, b_n + 1)
    Cv = StridedView(view(Cbig, 2:(a_n + 1), 2:(n_n + 1), 1:b_n))
    return (Cv, Aperm, (2, 3, 1), Bneg, (2, 4), (1, 4, 3))
end
