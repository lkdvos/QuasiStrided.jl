# Cache-blocking factors for the macro-blocking driver's five-loop nest
# (docs/decisions.md, "Macro-blocking milestone" -> frozen interface #4).
# `Blocking` itself is a plain, always-valid (>=1 per field) user-facing
# value; `plan_contract` is what rounds/clamps it into the *effective*
# blocking actually stored on `ContractPlan`.

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
`scalartype(kernel)`. Hardcoded, single-machine constants
(docs/decisions.md, "Block-size policy"), not derived from any cache model
-- measured by the Phase E benchmark sweep on the reference machine
(Cascade Lake; see `benchmark/results/.../PROVENANCE.txt` and
`docs/decisions.md`'s "Macro-blocking milestone" -> Phase E for the grid,
chosen values, and runner-up spread). Values are the *requested* `mc`/`nc`
(not yet rounded to `mr(kernel)`/`nr(kernel)` multiples -- `plan_contract`
does that, and also clamps everything to the contraction's actual extents).
"""
default_blocking(kernel) = default_blocking(scalartype(kernel))

# measured 2026-09-08, Cascade Lake (Xeon Gold 6244, ccqlin038; see
# benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/PROVENANCE.txt
# and docs/decisions.md, "Macro-blocking milestone" -> Phase E).
default_blocking(::Type{Float64}) = Blocking(64, 128, 768)
# measured 2026-09-08, Cascade Lake (Xeon Gold 6244, ccqlin038; see
# benchmark/results/ccqlin038.flatironinstitute.org-2026-09-08/PROVENANCE.txt
# and docs/decisions.md, "Macro-blocking milestone" -> Phase E).
default_blocking(::Type{Float32}) = Blocking(96, 384, 1152)

# Smallest multiple of `n` that is >= `x` (`x >= 0`, `n >= 1`).
@inline _roundup(x::Int, n::Int) = cld(x, n) * n
