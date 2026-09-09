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

Default cache-blocking factors for `kernel`, dispatched on
`scalartype(kernel)`. Hardcoded measured constants, not a cache model
(docs/decisions.md, "Block-size policy"). These are the *requested* `mc`/`nc`
-- `plan_contract` rounds them to `mr`/`nr` multiples and clamps them to the
contraction's actual extents.
"""
default_blocking(kernel) = default_blocking(scalartype(kernel))

# Measured 2026-09-08 on ONE machine class (Cascade Lake, Xeon Gold 6244,
# ccqlin038): benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/
# PROVENANCE.txt, docs/decisions.md -> Phase E.
default_blocking(::Type{Float64}) = Blocking(64, 128, 768)
default_blocking(::Type{Float32}) = Blocking(96, 384, 1152)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
