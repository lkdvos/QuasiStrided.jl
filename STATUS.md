# QuasiStrided.jl status

Durable status record for the main orchestration process. Update at phase
boundaries or before context compaction. History belongs here only as far as
"what phase are we in"; do not let this rot into a changelog.

## Integrated revision

Local git repo initialized at
`/mnt/home/ldevos/Projects/QuasiStrided.jl` (no remote; not published,
registered, or pushed, per handoff §2/§8).

## Environment (scouted 2026-09-08)

- Julia 1.12.6 on PATH.
- `StridedViews` v0.5.1/0.5.2 and `Strided` v2.6.3/2.6.4 resolvable locally;
  local checkouts also exist at `~/Projects/StridedViews.jl`,
  `~/Projects/Strided.jl/main`.
- `SIMD.jl` v3.7.1 installed locally (chosen SIMD dependency).
- CPU: AVX-512 (f/dq/cd/bw/vl) + AVX2 + FMA capable, Cascade Lake-class.

## Phase status

- [x] Phase 0 (bootstrap/freeze): package skeleton created, `contract!`
      signature frozen, dependencies chosen. See `docs/decisions.md`.
- [ ] Phase 1 (indexing + oracle): not started.
- [ ] Phase 2 (tiles/packing/scalar): not started.
- [ ] Phase 2b (Fable review): not started. `fable_review_used: false`.
- [ ] Phase 3 (SIMD + serial driver): not started.
- [ ] Phase 4 (review, measurement, handoff): not started.

## Active owners

None yet; Phase 1 workers about to be launched.

## Next task

Launch Phase 1: indexing implementer (Sonnet High) and oracle/test implementer
(Sonnet Medium) in parallel, against the frozen `docs/decisions.md` contract
and `Julia-Tensor-Indexing-Agent-Spec.md`.
