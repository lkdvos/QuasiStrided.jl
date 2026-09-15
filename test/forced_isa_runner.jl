# Run the whole test suite as if the host had a different vector ISA, by
# overriding the process-wide profile that `_init_target!` populates.
#
# This exists because CI burned two rounds on one class of bug that a local
# run cannot see: a test that asserts something only true on the machine it
# was written on. `docs/decisions.md`'s Phase G recorded the pattern ("making a
# constant hardware-derived silently converts every test that asserted its old
# value into a platform-dependent test"); Amendment 5 records it recurring for
# a *capability*. This script is how it gets caught in seconds instead.
#
#   QS_FAKE_ISA=avx2 QS_FAKE_VB=32 QS_FAKE_NREG=16 \
#       julia --project=. test/forced_isa_runner.jl
#
# Useful triples: avx512/64/32, avx2/32/16, neon/16/32, unknown/0/0.
#
# Expected residue: `test_target.jl`'s "runs on this host without throwing"
# compares a fresh `_detect_target()` against the stored profile and therefore
# fails by construction here -- that one failure is this harness, not the
# package. Run from the test environment (`Pkg.test`-style) if you want the
# Bumper-dependent allocator testsets too.
using QuasiStrided
const _QS = QuasiStrided

_QS._TARGET[] = _QS.TargetProfile(
    Symbol(get(ENV, "QS_FAKE_ISA", "avx512")), Sys.ARCH, "forced-isa-runner",
    parse(Int, get(ENV, "QS_FAKE_VB", "64")),
    parse(Int, get(ENV, "QS_FAKE_NREG", "32")),
    _QS.CacheLevel(), _QS.CacheLevel(), _QS.CacheLevel()
)

let p = _QS.target_profile()
    println(
        "forced target: isa=", p.isa, " vector_bytes=", p.vector_bytes,
        " nregisters=", p.nregisters
    )
end

include(joinpath(@__DIR__, "runtests.jl"))
