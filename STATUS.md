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
- [x] Phase 2 contract frozen: `src/kernel_descriptor.jl` (`KernelDescriptor{MR,NR,T}`,
      `packed_a_offset`/`packed_b_offset` = `i+MR*p`/`j+NR*p`,
      `packed_a_length`/`packed_b_length`), main-process-owned, tested in
      `test/test_kernel_descriptor.jl`.
- [x] Phase 2 (tiles/packing/scalar): complete and gated 2026-09-08.
      `src/tiles.jl` (AffineAxis/ScatterAxis/QSTile=SourceTile=DestinationTile),
      `src/packing.jl` (pack_a!/pack_b!), `src/kernel.jl` (ScalarKernel,
      zero_accumulator/accumulate/store_tile!/execute_tile!). Two integration
      reconciliations by main process (destination-tile type drift,
      kernel/packing coupling — see docs/decisions.md "Phase 2 integration
      notes"), plus a new `test/test_phase2_integration.jl` end-to-end
      fixture. `Pkg.test()`: 12318/12318, 25.4s.
- [ ] Phase 2b (Fable review): not started. `fable_review_used: false`.
- [ ] Phase 3 (SIMD + serial driver): not started.
- [ ] Phase 4 (review, measurement, handoff): not started.

## Active owners

None currently active; about to launch the Phase 2b Fable review.

## Next task

Launch the single Fable High review (fable_review_used will become true) per
handoff §6 Phase 2b: give it both specs, docs/decisions.md, the integrated
source paths (src/axis_group.jl, src/kernel_descriptor.jl, src/tiles.jl,
src/packing.jl, src/kernel.jl), test results, and the
test/test_phase2_integration.jl fixture. Triage findings; Sonnet owners (or
main process for small fixes) implement, main process verifies, before
starting Phase 3 (SIMD + serial driver).
