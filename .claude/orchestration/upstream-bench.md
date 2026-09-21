# Upstream benchmark-suite comparison (preparatory milestone)

## Goal / acceptance criteria
See full execution contract (revision 1, orch-planner, 2026-09-15) archived below.
Deliverable: a working three-way (StridedNative/StridedBLAS/QuasiStrided-via-composite)
comparison driven by TensorOperations.jl PR #303's upstream `TensorOperationsBenchmarks`
suite (`:pairwise` + `:tccg` categories, small/moderate dims), plus a first-look profiling
triage on a handful of very different cases. No engine change; preparatory only.

## User decisions (binding)
1. Dependency: pin to `QuantumKitHub/TensorOperations.jl#benchmark` (SHA
   `528dd85d8bf886c734a207732a7cb591a3691dd3`) via `benchmark/Project.toml` `[sources]`.
   Repointing after PR merge is an explicit follow-up, out of scope now.
2. Provider: plain `ArrayProvider{T}(; backend=...)`. First task builds a benchmark-only
   composite backend (`tensorcontract!` -> `QuasiStridedBackend()`, `tensoradd!`/
   `tensortrace!` -> `StridedNative()`) so the full suite (incl. :permute/:trace) *could*
   run against one provider — lives under `benchmark/`, never `src/`, does not change
   `QuasiStridedBackend`'s frozen hard-reject/no-fallback invariant.
3. Case set this pass: `:pairwise` + `:tccg` only, small-to-moderate dims (not full sweep).
4. Profiling: broad first-look pass on ~4 very different cases (small pairwise, worst
   tccg loss, closest/win case, one larger pairwise) to see if causes are the same before
   deciding on a wider sweep.

## Repo / branch
Worktree `/mnt/home/ldevos/Projects/QuasiStrided.jl/main`, main HEAD `71c1536`.
Work branch: `upstream-bench` (created from `71c1536`).

## Constraints
- No `src/`, `test/`, or existing `benchmark/*.jl` edits; root `Project.toml` untouched
  (one narrow exception, see Trigger R1 in the contract).
- Single-machine measurement discipline: BLAS threads=1, >=15 reps median, canary at
  start/middle/end, noise floor = max(10%, canary spread).
- Julia 1.10 compat on the package itself untouched; `benchmark/Project.toml` may need 1.11+.
- Resource policy: balanced, <=3 active workers (peak 2).

## Task graph (from orch-planner, revision 1)
T0 (orch-scout) -> {T1, T2} (orch-builder, parallel) -> {T3, T4} (orch-builder, parallel)
-> T5 (orch-specialist) -> T7 (orch-builder, docs) -> T6 (orch-reviewer) -> T8 (orch-builder,
fix-ups) -> commit + PR.

- T0: upstream API + dependency fact-finding (read-only).
- T1: `benchmark/Project.toml` (+ optional `setup_env.jl` fallback).
- T2: `benchmark/composite_backend.jl` + `benchmark/check_composite_backend.jl`.
- T3: `benchmark/bench_to_suite.jl` + one full measurement run on ccqlin038-class machine.
- T4: `benchmark/profile_to_suite.jl` (bucketed flamegraph/text profiling tool).
- T5: profiling run + triage report (same-cause-or-not verdict across cases).
- T6: independent review (orch-reviewer, one gated pass) of composite semantics,
  measurement discipline, doc-number traceability, case-selection/skip logic.
- T7: `docs/decisions.md` new section + `STATUS.md` short entry, all numbers traced to
  artefacts on disk.
- T8: address T6 findings, re-verify, commit, open PR.

Full task contracts (edit scope, verification, deliverables per task): see the planner's
report in session transcript (2026-09-15) — reproduce into this file if resuming in a
fresh session and the transcript is unavailable.

## Replanning triggers
R1: PR touches TO core / compat conflict. R2: QS result mismatch vs StridedBLAS on a
tccg/pairwise case (correctness finding, out of edit-scope — stop and report).
R3: systematic QS rejections beyond isolated cases. R4: canary spread >15% twice.
R5: suite has no clean per-case executor. R6: profiling shows dominant cost outside
QuasiStrided. R7: any task believes a `src/` edit is necessary (always stop).

## Status — COMPLETE
- [x] Design discussion with user (2026-09-15)
- [x] Planning (orch-planner revision 1, 2026-09-15)
- [x] T0 (upstream API fact-finding)
- [x] T1 (benchmark/Project.toml), T2 (composite backend)
- [x] T3 (three-way benchmark script + run: 118 cases, 0 mismatches/rejections),
      T4 (profiling tool)
- [x] T5 (profiling triage: 2 separable causes found for ccsd_t_* regression,
      both unverified/read-only)
- [x] T7 (docs/decisions.md + STATUS.md sections)
- [x] T6 (independent review: 1 blocking + several should-fix findings)
- [x] T8 (fix-ups applied, Pkg.test() 34654/34654 passing, PR opened)

PR: https://github.com/lkdvos/QuasiStrided.jl/pull/5 (branch `upstream-bench`)

## Follow-ups for a future milestone (not started)
- Repoint `TensorOperationsBenchmarks` dependency once PR #303 merges/releases.
- Investigate/fix the `store_tile!` `Vector{T}` vs `Memory{T}` guard (Cause A).
- Any engine change for the `ccsd_t_*` six-index-output regression (Cause B).
- Remaining upstream categories (:permute, :trace, :mixed_precision, :mps,
  :ctmrg, :trg) — not run this pass.
