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
- [x] Phase 1 (indexing + oracle): complete and gated 2026-09-08.
      `src/axis_group.jl` implements AxisGroup/offsets/fill_offsets!/
      BlockDescriptor/normalize_group per spec. Independent oracle +
      property tests + Strided integration tests + test-only packer/
      contraction consumer all pass: `Pkg.test()` → 11714/11714,
      `18.5s`. One integration fix by main process: added `Random` to
      `Project.toml`'s test targets (needed by the property tests).
      `benchmark/bench_axis_group.jl` exists and reports zero steady-state
      allocations for `fill_offsets!`/`block_descriptors!` on ccqlin038.
- [ ] Phase 2 (tiles/packing/scalar): not started.
- [ ] Phase 2b (Fable review): not started. `fable_review_used: false`.
- [ ] Phase 3 (SIMD + serial driver): not started.
- [ ] Phase 4 (review, measurement, handoff): not started.

## Active owners

None currently active; about to launch Phase 2.

## Next task

Launch Phase 2: packing implementer (Sonnet Medium, owns `src/tiles.jl`,
`src/packing.jl`, `test/test_packing.jl`) and scalar implementer (Sonnet
Medium, owns `src/kernel.jl`, `test/test_kernel.jl`), in parallel, against
`Julia-Microkernel-Tile-Interface-Design.md` sections 4-9 and the frozen
packed-offset formulas (A: `i + MR*p`, B: `j + NR*p`). Main process to freeze
exact packed offset formulas / kernel descriptor shape in `docs/decisions.md`
before launching, per handoff §6 Phase 2 gate instructions.
