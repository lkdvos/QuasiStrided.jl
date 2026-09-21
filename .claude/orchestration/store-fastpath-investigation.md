# Store fast-path investigation milestone — COMPLETE

PR: https://github.com/lkdvos/QuasiStrided.jl/pull/6 (branch `store-fastpath-investigation`, base `main`)

## Outcome
- Cause A (store fast-path guard unsatisfiable for Memory{T} on Julia >=1.11)
  confirmed and fixed. Full suite 34856/34856 passing.
- The ccsd_t_*_dim16 regression (the original motivation) is NOT fixed by
  this — confirmed both by measurement and by mechanism (zero unit-stride
  destination rows in all 4 cases x 2 dtypes, at dim=16).
- Unplanned finding, flagged not acted on: a label-order control shows
  3.3x-20x speedup on the same regression cases at dim=16. This is a
  candidate NEW milestone requiring a user decision — not started.
- One gated independent review (T8) ran; no blocking findings; should-fix
  items addressed (T9).

## Open follow-ups (not started, need a decision)
1. Label-ordering lever (Arm 3 finding) — potentially the biggest win
   available for the ccsd_t_* case class, 3-20x. Needs scoping: touches
   `_classify_labels` in `src/driver.jl`, currently pinned by an existing
   test. Product-level decision, not mechanical.
2. Repointing `TensorOperationsBenchmarks` dependency — blocked until
   upstream PR #303 merges (still open as of 2026-09-15).
3. Stale `_acc_lane` comments in `src/kernels/planar.jl`/`onem.jl`/
   `test/test_quality.jl` (cosmetic, low priority, out of this PR's scope).
4. Remaining upstream benchmark-suite categories (:permute, :trace,
   :mixed_precision, :mps, :ctmrg, :trg) — not run.

## Both PRs open, unmerged
- #5 (upstream-bench): the original benchmark-suite comparison + triage.
- #6 (store-fastpath-investigation): this milestone.
Expect a trivial append-conflict in docs/decisions.md/STATUS.md if both merge.
