# Label-ordering (Cause B) milestone

Branch `label-order`, base `store-fastpath-investigation` at
`e2b5e5c1a0a0af1afdd73b1fcd873450c2f35c4c` (BASE SHA for all A/B
comparisons). Coordinator: this session (Sonnet, /orchestrate).

## Goal

Decide, on measurement, whether/how to capture the 3.3x-20x "label-ordering
lever" (Arm 3 finding, `docs/decisions.md` "Store fast-path investigation:
Phase A" on `store-fastpath-investigation`) for the `ccsd_t_*_dim16` case
class inside the engine (not just the diagnostic script), and if warranted
implement it with full regression verification, independent review, and a
dated `docs/decisions.md` addendum.

Full execution contract from orch-planner (fable/high) is reproduced below
verbatim. Task graph: T0 (coordinator, done) -> {T1 || T2} -> G1 (gate) ->
T3 -> T4 -> T5 -> T6 (or T6-alt if G1 fails).

## Status

| Task | Status | Worker | Notes |
| --- | --- | --- | --- |
| T0 | done | coordinator | upstream-bench WIP committed (`0287b52`); branched `label-order` from `store-fastpath-investigation` (`e2b5e5c`); baseline `Pkg.test()` 34856/34856 clean after fixing a stray depot gap (`Requires`) |
| T1 | done | orch-scout + coordinator backfill | see findings above; R6 does not fire; Q1 sharpened (S1b likely load-bearing) |
| T2 | done | orch-fable | see findings + G1 verdict below |
| G1 | **PASS (S1) / AMBIGUOUS (S1b)** | coordinator | see below -- escalating S1b to user per plan's own "ambiguous" escape valve |
| G1 | pending | coordinator | |
| T3 | done | orch-fable | commit `e7c4787`; 35165/35165 passing; real Arm-1 speedups 24-70x (dim16) / 4-17x (dim8); swap guard correctly excludes F32 regression cases; complex/conj correctness tests added; Runic clean. See findings below. |
| (housekeeping) | done | coordinator | committed T2's `bench_ccsd_t_store.jl` diff (`f06fde0`), left uncommitted by T3 on purpose |
| T4 | done | orch-builder | AC4/AC5 both PASS; see findings below (rate-limit interruption mid-task, resumed cleanly, no rework needed) |
| T5 | done | orch-reviewer | **verdict: no blocking correctness defect; safe to merge (after T6's doc/traceability follow-ups)**. Sharp catch (F1, "should-measure before merge") acted on immediately by coordinator -- see below |
| (fix) | done | coordinator | commit `a650af7`: restricted swap to `T <: Real` per T5's F1; fixed own first-pass overreach (`isa SIMDKernel` wrongly excluded `ScalarKernel`) via test failures; 35168/35168 clean |
| T6 | **done -- MILESTONE CLOSED** | orch-builder | commit `e292205`. `docs/decisions.md` "Label-order milestone" section (resolves dangling forward-refs), STATUS.md fix, both discharge notes, `ContractPlan` docstring sentence, `bench_ccsd_t_store.jl` staleness comment. Independently re-verified numbers against fresh on-disk artifacts, corrected 3 of my own estimates (see below). 35168/35168 unchanged. |

## T2 results and G1 gate outcome

T2 extended `benchmark/bench_ccsd_t_store.jl` (uncommitted, +239/-84) with
Arms 4M/4N/4both (independent |C-stride| sort of M/N, lazy `permutedims`
views -- zero data copy) and Arm 5 (4both + orientation swap so C's
stride-1 label always lands in M). Full 16-point sweep (4 cases x
dims{8,16} x {F64,F32}), 9 arms/backends per point. **0 correctness
mismatches; all direct-API arms bitwise `==` to Arm 2.** Canary spread
0.36%; identical-plan cross-checks <=2%.

**G1-pass (S1, independent sort): PASS, decisively.** Arm 4both vs Arm
1-QuasiStrided at dim=16 (the gate's own criterion, >=2.0x on >=6/8, never
worse): all 8/8 points, ratios 14.04x-66.73x. Dim-8 clause (4both must not
regress vs Arm1 where Arm1 median >100us): no violations, 4both is 4-7x
*faster* than Arm1 at every dim-8 point too. **No pre-authorized size guard
needed.** Recommendation: implement S1 in `src/driver.jl` -- this alone is
the milestone's primary deliverable.

**G1-swap (S1b, orientation swap): AMBIGUOUS -- escalating per the plan's
own "stop and present to the user" clause, not deciding unilaterally.**
Arm 5 vs Arm 4both (ratio, >1 = swap better), dim=16: ccsd_t_2 F64 1.48x /
**F32 0.81x (regression)**; ccsd_t_3 F64 2.61x / F32 2.03x; ccsd_t_4 F64
1.49x / **F32 0.82x (regression)**. Mechanism (T2 diagnosed via M-sliver
unit-stride counts): the store fast path needs an MR-wide (16 F64, 32 F32
on this host) unit-stride run. Swapping puts `a` (extent 16) first in M --
for F64 (MR=16) that's a perfect fit (100% unit-stride slivers); for F32
(MR=32) a 32-wide sliver spans `a` plus the next label, which is NOT
unit-stride unless that next label is C-adjacent (true for ccsd_t_3's
`(a,b)`, stride 1+16, false for t2/t4) -- so the swap gains nothing on M
*and* throws away the N-side locality win 4both already had, net ~1.2x
regression. T2's own proposal: guard the swap on "leading contiguous run of
the candidate M composite in C >= mr(kernel)", not the naive
`_prefer_swap` (min-stride comparison) sketched in the original interface
-- **this refined guard is a proposal, not yet measured**.

Full data, per-arm sliver diagnostics, and reproduction command: see T2's
report (delivered via SubagentHandback, not filed separately) and
`benchmark/results/ccqlin038.flatironinstitute.org-2026-09-19/{bench_ccsd_t_store.csv,summary_ccsd_t_store.txt,PROVENANCE_ccsd_t_store.txt}`.

**User decision (2026-09-19): option (c).** Ship S1 unconditionally; T3
implements S1b behind the refined MR-aware guard (leading contiguous run
of the candidate M composite in C must be >= mr(kernel)), verified by T3's
own tests/benchmarks before merge -- not shipped speculatively unverified.

## T3 results (implementation)

Commit `e7c4787` (`src/driver.jl` +97/-6; `test/test_driver.jl` +333/309
new assertions; `docs/decisions.md` +15 dated correction). Full suite
**35165/35165** (baseline 34856 + 309 new), 0 fail, 0 error, no pre-existing
test's expected values changed. Runic clean on both changed files.

**Real Arm-1 (adapter path, `QuasiStridedBackend()`) before -> after**,
median seconds, dim=16:

| case | dtype | before | after | speedup | vs StridedBLAS after |
| --- | --- | --- | --- | --- | --- |
| ccsd_t_1 | F64 | 1.924 | 0.0273 | 70.6x | 7.3x faster |
| ccsd_t_2 | F64 | 1.959 | 0.0362 | 54.2x | 4.1x faster |
| ccsd_t_3 | F64 | 1.974 | 0.0478 | 41.3x | 4.9x faster |
| ccsd_t_4 | F64 | 1.945 | 0.0357 | 54.5x | 4.2x faster |
| ccsd_t_1 | F32 | 1.779 | 0.0342 | 52.0x | 2.5x faster |
| ccsd_t_2 | F32 | 1.194 | 0.0395 | 30.3x | 1.9x faster |
| ccsd_t_3 | F32 | 1.206 | 0.0219 | 55.0x | 4.8x faster |
| ccsd_t_4 | F32 | 0.988 | 0.0402 | 24.6x | 1.8x faster |

(dim=8: 3.8x-17.5x speedup, sub-ms absolute times.) **The original
regression this whole milestone was opened to fix (6.6-14.2x slower than
StridedBLAS) is now QuasiStrided winning by 1.8x-7.3x on every one of the
8 dim=16 points.**

**Swap guard verified correct** (matches T2's measured-safe set exactly):
fires for ccsd_t_2/3/4 at Float64, ccsd_t_3 only at Float32 (correctly
excludes the two Float32 regression cases, ccsd_t_2/4), ccsd_t_1 never
(already fine as-is). Verified both by the guard's own logic
(`_leading_unit_run` >= `mr(kernel)`) and by dedicated pinning tests
asserting the exact swap/no-swap decision per case/dtype/kernel.

**Complex/conjugation**: swap IS reachable for complex dtypes (e.g.
ccsd_t_3 at ComplexF64/ComplexF32). Correctness argument given (each
operand keeps its own `atransform`/`btransform` under the swap; complex
multiplication commutes) PLUS a dedicated test exercising the swap with
`conjA=conjB=true` and an XOR-folded `conj`-op view, checked against an
independent loop reference for both `execute!` and `execute_tilewise!`.

**Two known-unmeasured corners** (correctness not in question, only
whether the guard's chosen orientation is actually the fastest one):
(1) cases where the default kernel's small-`Qm` demotion differs between
the as-is and swapped orientations (the guard compares `mr` of whichever
kernel would actually run on each side, but no case in the ccsd_t_* set
exercises this); (2) complex-dtype swap performance specifically (only
correctness was measured for complex, not speed). Neither blocks this
milestone; both are reasonable T5/follow-up notes, not defects.

**Docs correction landed**: `docs/decisions.md`'s false "pinned by an
existing test" claim now has a dated 2026-09-19 addendum pointing at the
real pinning tests. Forward-reference note: `src/driver.jl`'s new docstring
cites `docs/decisions.md, "Label-order milestone"` -- that section doesn't
exist yet, T6 must either use that exact heading or fix the reference.

## T4 results (broad regression)

BASE (`e2b5e5c`, `git archive` extraction, fresh offline manifest) vs FIX
(`f06fde0`, live worktree). Confirmed genuinely different `src/driver.jl`
(BASE lacks `_order_free_labels`/`_prefer_swap`) and different resolved
`QuasiStrided` source paths.

**`bench_to_suite.jl`** (ABBA BASE-1/FIX-1/FIX-2/BASE-2,
`--sources synthetic,tccg --dtypes Float64,ComplexF64 --contract-sizes
8,15,32 --tccg-sizes 8,16 --reps 21`): 0 mismatches/rejections all 4 runs.
Canary spreads 7.0-12.7% (noise floor = 12.7%, elevated by other users'
processes on the shared login node, not this task's own concurrency).
**Overall QuasiStrided geomean(fix/base) = 0.742** (25.8% net faster) --
synthetic essentially flat (1.093 F64 / 1.067 C64, within noise), tccg
strongly improved (0.527 F64 / 0.653 C64). **AC4: PASS** -- no case with
base median > 100us slowed beyond noise. One honest caveat: 18
microsecond-scale cases (3.8-35us absolute) show a reproducible (both ABBA
rounds independently) 1.13-2.48x slowdown -- plausibly a small fixed cost
from the new sort/guard step at scales where it's not amortized by any real
work. Flagged for the close-out addendum as a known minor effect, not a
blocker (dwarfed by the 0.742 overall geomean and outside AC4's own
100us-floor criterion).

**`bench_real_path_guard.jl`** (ABBA BASE-A/FIX-A/FIX-B/BASE-B): canary
spread 5.0-11.5%. Per-shape geomean(fix/base) over 18 (dtype,shape)
combos = **0.934** (net faster). **AC5: PASS**, and better than
"no systematic shift" -- two shapes (`scattered_a64k64b16n64`,
`smallN_256x256x12`/F32) show a genuine, round-reproducible improvement
(0.6-0.74x); all 16 other shapes within +-12% (noise).

**No replanning trigger fired** (R3/R4 both clear). Worktree left clean
(only `.claude/` untracked) -- confirmed by T4 itself. Full artifacts in
T4's scratchpad (session-local, not committed -- paths in its report).

Note: T4 was interrupted once mid-task by a session rate limit between
FIX-1 and FIX-2; resumed cleanly from its own saved scratchpad state, no
rework needed. Also corrected a background-brief inaccuracy: `benchmark/
Manifest.toml` is NOT tracked on `upstream-bench` either (gitignored
project-wide) -- T4 resolved fresh manifests offline in both trees instead
of copying one, which worked without issue.

## T5 results (independent review) + coordinator's immediate follow-up fix

**Verdict: no blocking correctness defect in `src/`.** T5 (no shell access,
pure code reading) independently confirmed: (1) `_build_pair_group`/
`fill_offsets!` really do enumerate first-label-fastest, and
`_order_free_labels`'s stability is real and load-bearing
(`sortperm(...; alg=Base.Sort.DEFAULT_STABLE)` -- `sortperm`'s *default* is
UNSTABLE, so the explicit `alg=` was necessary, not redundant); (2) traced
`plan_contract`'s full body and every consumer (`_execute_nest!`,
`execute_tilewise!`) and confirmed the swap moves every operand-bound field
(storage/base/transform/groups) as one consistent set, role-keyed not
tensor-keyed -- **and found the asymmetric-conj test case
(`conjA != conjB`) the brief worried might be missing was already present**
(`test/test_driver.jl`, now the "swap threads conj/transforms correctly"
testset); (3) hand-counted the new test assertions (+309/+310, matches);
(4) confirmed `_order_free_labels`/`_leading_unit_run`/`_prefer_swap` are
all concretely-typed, no `Any`/`Union` leakage.

**Should-fix items (T6's job, not blocking):** dangling
`docs/decisions.md, "Label-order milestone"` cross-reference in the public
`plan_contract` docstring (section doesn't exist yet -- T6 must use that
exact heading); `STATUS.md` still repeats the refuted "pinned by an
existing test" claim (only `docs/decisions.md` got the correction); two
other `docs/decisions.md` spots ("orientation swap ... out of scope",
"still deferred") now contradicted by the landed code, need dated notes;
this milestone's own numbers (35168, speedup ratios, geomeans) exist only
in this untracked `.claude/` file, not in anything git-tracked -- AC1 gap,
T6 must land them in `docs/decisions.md` with real CSV/PROVENANCE
references (T4's artifacts are in ITS scratchpad, session-local -- T6
should copy the essential ones into a committed, gitignored-results-dir
convention or inline the key numbers with enough provenance to be
re-derivable); AC5 close-out text should say "no regression detectable at
the guard's own stated ~5-6% resolution" rather than asserting the two
best-shape improvements as fully proven (they're consistent with, not
provably beyond, the instrument's stated resolution).

**Should-measure-before-merge item, acted on immediately (not deferred to
T6):** T5's sharpest finding (F1) -- the swap's whole justification
(`_vector_store_eligible`) only exists for `SIMDKernel` (real dtypes);
`PlanarKernel`/`OneMKernel` (complex) always scatter-store, so the swap
could be firing for complex with no possible benefit, unmeasured. The
coordinator measured this directly (monkey-patched `_prefer_swap` to
force-disable, A/B compared): **swap was in fact a small (~2-4%, single
quick run, not full ABBA) regression for ComplexF64/ComplexF32 `ccsd_t_3`**
-- directionally confirms F1. Fixed in commit `a650af7`: guard changed to
`T <: Real` (a first attempt narrowly gated on `kernel_asis isa SIMDKernel`
broke the existing pinning test's `ScalarKernel` case -- `ScalarKernel` is
also real-dtype and still benefits from the same run-length rule; caught
immediately by `Pkg.test()`, not shipped). Also required fixing one test's
own `atransform` expectation (had the conj-XOR direction backwards for the
now-unswapped complex case) and repurposing the one test that specifically
required the complex swap to fire into a "confirms it does NOT fire" +
correctness check, plus a new real-dtype-kernel test to keep direct
(not just code-reading) coverage of "transforms travel with the operands
under a swap" now that production complex contractions never reach that
branch. **Final: 35168/35168 passing**, swap behavior directly re-verified
(`Astorage===parent(A)` for both complex dtypes at ccsd_t_3/dim16 = no
swap; `Astorage===parent(B)` for both real dtypes = swap fires, as before).

**Not yet acted on (T6/follow-up, not blocking):** item 4's TTFX/
specialization-count hypothesis for T4's 18 microsecond-scale regressions
(plausible explanation, unmeasured); F2 (the `run>=mr` guard is necessary
but not sufficient for 100% unit-stride slivers when `mr` doesn't divide
the run -- a heuristic-quality note, not a defect); F3 (`ContractPlan`
docstring should mention `Astorage` may be `parent(B)`); F4
(`bench_ccsd_t_store.jl`'s Arm 3/4/5 comments now describe pre-fix
semantics, since the engine sorts internally now -- worth one clarifying
line if that script is reused).

## T6 results (close-out) -- MILESTONE COMPLETE

Final commit `e292205` on `label-order` (4 commits total on top of base
`e2b5e5c`: `e7c4787`, `f06fde0`, `a650af7`, `e292205`). 35168/35168 passing
throughout, unaffected by the doc-only close-out commit.

**Numbers T6 independently re-derived and corrected** (own estimates in
this coordinator's dispatch were slightly off, now corrected on-disk):
- QuasiStrided beats StridedBLAS by **1.96x-6.91x** at dim=16 (not the
  dispatch's guessed "1.2x-6.9x" -- floor is higher). Also newly noted: two
  dim=8 Float32 cells are still slightly *slower* than StridedBLAS
  (0.81x/0.90x) -- small-shape overhead, not this milestone's target, but
  now on record rather than glossed over.
- Broad sweep geomean: **0.7418** (matches the "0.742" figure).
- Microsecond-scale caveat cases: **18 cases, ratio 1.153x-2.483x, absolute
  4.1-38.5us** (dispatch said 1.13-2.48x / 3.8-35us -- close, now precise).
  Independently reconfirmed zero real (both-rounds-reproducible) regressions
  above the 100us floor.
- `bench_real_path_guard.jl` geomean 0.9335 (matches "~0.934"), framed per
  T5's request as "no regression detectable at the guard's own ~5-6%
  resolution" rather than a proven per-shape win.

Not independently re-run from scratch (explicitly reported as such, not
silently presented as verified): the broader sweep's canary/noise-floor
figures (12.7%) come from re-analyzing the still-on-disk CSVs, not from
re-executing the hour-long ABBA benchmarks again -- reasonable given they
were already run once this session with full methodology.

**Milestone status: COMPLETE.** All 6 planned tasks (T0-T6) done, gate G1
resolved (S1 shipped unconditionally, S1b shipped behind a measured,
corrected guard), one post-review fix applied and verified, full
documentation trail landed. Not done (deliberately, out of scope): pushing,
opening a PR, or merging -- awaiting user direction (plan's Q4: stack this
PR on the still-unmerged `store-fastpath-investigation`, or wait).

## Baseline test count

**34856/34856 passing** (confirmed on `ccqlin038`, Julia 1.13.0, 2026-09-19).
First two attempts errored on Aqua's "Persistent tasks" check (`Unable to
locate Requires, a dependency of PackageExtensionCompat`) -- a stray depot
gap on this shared machine, unrelated to JULIA_PKG_OFFLINE and unrelated to
any code here. Fixed with `julia -e 'import Pkg; Pkg.add("Requires")'`
(global v1.13 environment, not this project's Project.toml/Manifest.toml).
Third attempt: clean, 34856/34856, 4m40s.

## T1 (discovery) -- done, key findings

- A1 CONFIRMED: `_vector_store_eligible`/`DenseVector` guard present
  (`src/kernels/simd.jl:163-181`); `benchmark/bench_ccsd_t_store.jl` present
  with Arms 1-3.
- A2 CONFIRMED: `ccqlin038.flatironinstitute.org`, `cascadelake`. (Kernel
  shape probe MR/NR figures from the planner's brief, not re-verified here;
  low risk, matches prior milestone's own record.)
- A3 CONFIRMED: no `benchmark/Project.toml` on this branch; scripts run
  under `--project=.`. `git diff HEAD upstream-bench -- benchmark/harness.jl`
  is EMPTY -- harness.jl is byte-identical between the two branches, so T4
  can borrow `bench_to_suite.jl` + its env from `upstream-bench` freely
  without a harness reconciliation step.
- A4 CONFIRMED: `JULIA_PKG_OFFLINE=true julia --project=. -e 'using
  QuasiStrided, TensorOperations'` -> OK. Runic NOT available in the default
  julia env (errors on `using Runic`) -- T3's Runic check may need
  `Pkg.add("Runic")` first, or may need to be skipped/deferred; flag to T3.
- **Disputed claim REFUTED**: `docs/decisions.md`'s "pinned by an existing
  test" claim for `_classify_labels`'s label order has NO supporting test.
  `test/test_driver.jl:222` is an error-path test only;
  `test/test_driver.jl:945-949` reads `plan.mgroup`/`ngroup` from an existing
  plan but asserts nothing about their order/contents. No `@test` anywhere
  pins the M/N composite order for a normal case. **R6 does NOT fire** --
  proceed without user escalation on this point, but T3/T6 should correct
  the docs claim when writing the close-out addendum.
- **Q1 answered (data table, dim=16, all four cases, verified via
  `TensorOperations.contract_indices` + `QuasiStrided._qs_labels` +
  `strides(zeros(Float64, dims...))`):**

  | case | C's stride-1 label | owner | M (unsorted, A-order) w/ C-strides | N (unsorted, B-order) w/ C-strides | M sorted | N sorted |
  | --- | --- | --- | --- | --- | --- | --- |
  | ccsd_t_1 | a | **A** | i=4096, j=65536, a=1 | k=1048576, b=16, c=256 | a,i,j | b,c,k |
  | ccsd_t_2 | a | **B** | i=4096, j=65536, b=16 | k=1048576, a=1, c=256 | b,i,j | a,c,k |
  | ccsd_t_3 | a | **B** | i=4096, j=65536, c=256 | k=1048576, a=1, b=16 | c,i,j | a,b,k |
  | ccsd_t_4 | a | **B** | i=4096, k=1048576, b=16 | j=65536, a=1, c=256 | b,i,k | a,c,j |

  **Important sharpening of the planner's Q1 hypothesis, not just a
  confirmation**: `a` is on the M-side operand (A) ONLY for `ccsd_t_1` --
  the case with the biggest Arm-3 gain (20.0x/13.6x). For the other three
  (4.0-4.2x gains), `a` is on B (the N-side operand). The vectorized store
  fast-path guard (`_vector_store_eligible`) checks **rows** specifically
  (M-side), not columns -- so a naive "sort M and sort N independently by
  |C-stride|" fix (S1 alone) would put `a` first *within* N for cases 2-4,
  giving a locality win but likely NOT engaging the row-vectorized fast path
  at all, since `a` never becomes part of M. **This means S1b (the
  orientation swap) is very likely load-bearing for 3 of the 4 regression
  cases, not a rare edge case** -- T2 must treat Arm 5 (swap) as a primary
  measurement, not a fallback, and G1-swap's criteria should be weighted
  accordingly when reading T2's results.
- Q3 (blast radius): not exhaustively enumerated (T1 ran out of turns before
  this; coordinator did not backfill it either, lower priority than Q1).
  Qualitative estimate stands: any case with <=1 free label per operand side
  (the large majority of existing benchmarks/tests -- plain matmuls, most
  tensor-network contractions) is completely unaffected by any ordering
  change (order is moot with one element). Only shapes with >=2 free labels
  per side (the `ccsd_t_*`/`ccsd_*`-style multi-open-index chemistry
  equations, and a few synthetic "GEMM-like"/"shared bond" shapes from the
  upstream-bench suite) are in scope to change at all. T3's own new pinning
  test covers the multi-index case directly; full enumeration deferred
  unless T5 review flags it as needed.

Coordinator note: T1 (orch-scout) hit its 20-turn limit twice and needed a
nudge plus direct coordinator backfill (A2/A4 offline+Runic checks, the
harness.jl diff, and the full Q1 data table were done directly by the
coordinator rather than the scout) to produce a complete deliverable --
default orch-scout effort was borderline for this task's actual scope
(a Julia snippet + several git commands). Noting for future task sizing.

## Decision boundaries / replanning triggers

See full contract below (Section 5) -- not repeated here, this file is the
durable copy of record.

---

## Full contract (orch-planner output, verbatim)

Planner: read-only inspection of the `upstream-bench` checkout. Facts
verified directly are marked EVIDENCE; everything else is ASSUMPTION or
INFERENCE, to be confirmed by T1.

### 1. Goal, acceptance criteria, assumptions, non-goals

**Acceptance criteria:**
- AC1. Dated additive `docs/decisions.md` addendum, every number traceable
  to CSV+PROVENANCE on disk. STATUS.md pointer.
- AC2 (if fix ships). `Pkg.test()` on Julia 1.12.6: 0 failed/errored, count
  >= 34853 + new tests. Runic check passes.
- AC3 (if fix ships). Four `ccsd_t_*_dim16` cases x {F64,F32} through the TO
  adapter: geomean speedup >= 3.0x vs pre-fix, every point >= 1.5x, post-fix
  Arm 1 within 15% of Arm 4/5 lazy-permute prototype on every point.
  Correctness: isapprox vs StridedBLAS at rtol 1e-10 (F64) / 1e-5 (F32).
- AC4 (if fix ships). ABBA regression: canary spread <= 15%; no case with
  base median > 100us slows by more than max(10%, canary spread); QS
  geomean fix/base <= 1.0 within noise.
- AC5 (if fix ships). `bench_real_path_guard.jl` ABBA: no systematic
  one-sided shift.
- AC6. One `orch-reviewer` pass, no unresolved blocking finding.
- AC7. Untouched: `QuasiStridedBackend` eligibility/hard-reject/fallback;
  `_classify_labels` rejection paths; `_execute_nest!` five-loop nest;
  `src/target.jl`; blocking constants; kernel shape menus.

**Assumptions (T1 confirms):**
- A1. `store-fastpath-investigation` HEAD has `f467b45` (Cause A fix) +
  `benchmark/bench_ccsd_t_store.jl` (Arms 1-3).
- A2. Machine is ccqlin038-class (Cascade Lake AVX-512, MR=16/NR=6 F64,
  MR=32/NR=6 F32).
- A3. `bench_to_suite.jl`+`benchmark/Project.toml` exist only on
  `upstream-bench`, not here.
- A4. Benchmark env resolves offline (`JULIA_PKG_OFFLINE=true`).

**Non-goals:** K-label ordering; cache-size cost model; changing
`default_blocking`/kernel menus; Strided-style temp copy of C; threading;
merging `upstream-bench`; adapter eligibility; TBLIS parity; MR-from-C's-run
-length (follow-up candidate only).

### 2. Design decisions

**Mechanism (EVIDENCE):** `_classify_labels` (`src/driver.jl:19-88`) returns
`mlabels` in `indA` appearance order, `nlabels` in `indB` order.
`plan_contract` (`:753-757`) passes these straight to `_build_pair_group`
(`:92-113`), which builds `AxisGroup(lens, (sA, sC))` in list order --
`fill_offsets!` (`src/axis_group.jl:136-199`) enumerates first-label-fastest.
So the store order into C is dictated by A's/B's incidental axis order, not
C's. Permuting the label list permutes A's and C's enumeration identically
-- **correctness preserved by construction**. Engine is symmetric in A/B
(swapping "which operand is M" is a relabeling). No test currently pins the
M/N label order (only an error-path test at `test/test_driver.jl:222`) --
the docs' "pinned by an existing test" claim is unsupported on the current
tree; T1 re-checks on the branch.

**Worked example (ccsd_t_1, dim 16):** M=(i,j,a) unsorted, C-strides
(4096,65536,1); sorted by |C-stride| -> M=(a,i,j), strides (1,4096,65536).

**Precision facts:** F64 MR=16 at dim16 -> sorted M's first run (length 16)
is exactly one MR-sliver, unit-stride, fast path engages. F32 MR=32 -> every
sliver spans two runs -> scattered path but with 64-byte-contiguous runs --
F32 gains expected smaller than F64 (not a failure). Per-element arithmetic
is order-independent and A/B-role-independent (IEEE commutative) -- outputs
should be bitwise identical across Arms 2/4/5 for real dtypes (bonus check,
not a gate).

**Candidate strategies:**
- **S1 (primary):** stable-sort `mlabels`/`nlabels` by `abs(strides(C)[pos])`
  ascending; ties keep A/B appearance order (single-label groups and
  already-favorable cases unchanged bit-for-bit).
- **S1b (conditional on G1-swap):** orientation swap (make B the M-role
  operand) when C's smallest-stride free label lives in B's open set.
  Highest risk: conj-flag handling under the swap.
- **S2 (permute-copy small operand):** dominated by S1 standalone; only a
  hybrid add-on if T2 shows physical contiguity (Arm 3) beats lazy reorder
  (Arm 4) by more than noise (trigger R1).
- **S3 (do nothing):** deliverable if G1 fails.

**Interfaces (T3 binding):**
```julia
# src/driver.jl, planning time only, between _classify_labels and
# _build_pair_group. Pure, returns new Vector{Int}.
_order_free_labels(labels::Vector{Int}, indC::NTuple{NC,Int}, C::StridedView) -> Vector{Int}
#   stable sort by abs(Base.strides(C)[findfirst(==(lbl), indC)]) ascending.

# Only if G1 selects S1b:
_prefer_swap(mlabels, nlabels, indC, C) -> Bool
#   true iff both non-empty and min |stride_C| over nlabels < min over mlabels.
```
`_classify_labels` itself unchanged. `plan_contract` docstring updated to
state the ordering rule.

**Zero-src prototype (measurement gate):** `permutedims` on a `StridedView`
is a lazy view -- passing `permutedims(Av, perm)` with `indA[perm]` to
`plan_contract` makes the engine see exactly S1's strides without touching
`src/`. Arm 3 (physical copy) vs Arm 4 (lazy) separates "loop order" from
"A contiguity".

**Open questions:** Q1 which operand carries C's stride-1 label per case
(T1); Q2 does sorted order ever lose at dim 8 (T2); Q3 how many suite cases
have an unsorted group at all -- bounds blast radius (T1); Q4 (user, PR
time only) stack on `store-fastpath-investigation` or wait for merge --
default: stacked.

### 3. Task graph

Order: T0 -> {T1 || T2} -> G1 -> T3 -> T4 -> T5 -> T6 (or T6-alt). Max
concurrency 2. **Never run two benchmarks, or a benchmark and Pkg.test,
concurrently** (single-core discipline).

**T1 (orch-scout, default):** confirm A1-A4; settle "pinned by a test"
claim via `git grep`; tabulate per ccsd_t_* case: indA/indB/indC, M/N label
lists with C strides at dim16, which operand carries C's stride-1 label,
L1-vs-MR per dtype; count suite cases with unsorted M/N groups (blast
radius). Read-only. Deliverable: table + A1-A4 confirmed/refuted + Q1/Q3
answered.

**T2 (orch-fable, standard):** extend `bench_ccsd_t_store.jl` (or new
`bench_label_order.jl`) with Arm 4-M/4-N/4-both (lazy permute) and Arm 5
(4-both + role swap). Keep Arms 1-3. Four ccsd_t_* x dims{8,16} x
{F64,F32}. isapprox vs StridedBLAS + bitwise `==` check across Arms 2/4/5
(bonus). CSV+PROVENANCE under `benchmark/results/<host>-<date>/label_order/`.
Edit scope: benchmark script + results dir only, no src/test.

**G1 (coordinator gate):** dim-16 points (8 = 4 cases x 2 dtypes).
- G1-pass (implement S1): Arm 4-both >= 2.0x on >= 6/8 points, never worse
  than noise; all correct. Also check dim-8: if Arm 4-both slower than Arm 1
  by more than noise where Arm-1 median > 100us, T3 must add the
  pre-authorized size guard (Section 5).
- G1-swap (also S1b): Arm 5 beats Arm 4-both by more than noise on every
  point where C's fastest label is in B; never worse elsewhere.
- G1-fail: Arm 4-both < 1.5x on >= 4/8 points -> T6-alt + trigger R1.
- Ambiguous: stop, present table to user, do not pick.

**T3 (orch-fable, high):** implement `_order_free_labels` (+ `_prefer_swap`
if G1-swap) in `src/driver.jl` only. New tests: ordering-helper unit test;
pinning test on a ccsd_t-shaped fixture (`plan.mgroup.strides[2]`
nondecreasing abs); correctness vs reference (both dtypes, alpha!=1,
beta!=0, permuted/sliced C); execute! vs execute_tilewise! agreement; if
S1b, full conj x op cross-product on a swap-triggering ComplexF64 fixture.
`Pkg.test()` + Runic. Edit scope: `src/driver.jl`,
`test/test_driver.jl`/`test/test_macro_driver.jl` (additive only).

**T4 (orch-builder, standard):** re-run T2's script on fix tree (AC3); ABBA
broad regression pulling `bench_to_suite.jl`+env from `upstream-bench` as
uncommitted files into both BASE and fix trees (AC4); `bench_real_path_guard.jl`
ABBA (AC5). Must restore `benchmark/harness.jl` and leave `git status` clean
before handing back.

**T5 (orch-reviewer, high):** review `_build_pair_group`/`fill_offsets!`
semantics correctness; conj/op handling under swap (silent-wrong-answer
risk); stale test/doc assumptions; allocation/type stability; number
traceability; AC4 methodology.

**T6 (orch-builder, standard):** address T5 should-fix items; write dated
`docs/decisions.md` addendum; STATUS.md entry; update this file; final
`Pkg.test()`; commit; open PR against `store-fastpath-investigation` (or
ask user per Q4).

**T6-alt:** same without src changes -- addendum stating measured arms, why
no engine change warranted, R1 recommendation.

### 4. Integration order / conflicts

Serial except T1||T2. `src/driver.jl`+tests: T3 only. Benchmark script: T2
owns, T4 runs (may add uncommitted flags). `docs/decisions.md`/`STATUS.md`:
T6 only. `benchmark/harness.jl`: T4 may temporarily overwrite, must restore.
Expect trivial append-conflict in decisions.md/STATUS.md when PRs #5/#6/this
one all eventually merge -- do not pre-resolve. One benchmark/test process
at a time on the machine.

### 5. Decision boundaries / replanning triggers

**May decide without replanning:** reps in [11,21]; new-file vs extend for
T2; branch/file names; uncommitted --flags on pulled benchmark copies;
smaller F32 gains (expected); deferring S1b per G1-swap; fallback regression
set if A4 refuted; PR base per Q4 default.

**May not decide:** any src/ edit before G1-pass; edits outside
`src/driver.jl`; blocking constants/kernel menus/`_classify_labels` error
semantics/adapter eligibility; cache-size cost model; merging
`upstream-bench`; committing pulled benchmark tooling to `label-order`;
skipping T5.

**Pre-authorized size guard** (only if dim-8 clause of G1-pass fires):
reorder only when result differs from derived order AND
`sizeof(T)*prod(size(C)) >= sizeof(T)*prod(size(operand))` for the operand
whose group is reordered (byte-count comparison, not a cache model). T4
measures at both dims.

**Replanning triggers:**
- R1. Arm 3 beats Arm 4-both by more than noise on >=3/8 dim-16 points ->
  S2 hybrid needs planning.
- R2. Any correctness mismatch in Arms 4/5, or bitwise inequality between
  Arms 2/4/5 (real dtypes) accompanied by isapprox failure -> stop before T3.
- R3. AC4 violated after size guard, or case slower by more than noise with
  base median > 1ms.
- R4. Canary spread > 15% on two consecutive runs -> pause measurements.
- R5. Post-fix Arm 1 doesn't match Arm 4/5 within 15% -> orch-specialist
  investigation.
- R6. T1 finds a test/frozen doc statement pinning the old order.
- R7. Pkg.test regressions outside new tests, or allocation assertion fails.
- R8. A3/A4 refuted with no viable broad regression check left.

## Key file references

- `src/driver.jl` (`_classify_labels` 19-88, `_build_pair_group` 92-113,
  `plan_contract` 716-770, `_execute_nest!` 945-1046)
- `src/axis_group.jl` (`fill_offsets!` 136-199, `describe_block` 227-251)
- `src/kernels/simd.jl` (`store_tile!` 207-245)
- `src/kernel.jl` (`_axpby_tile!`/`_axpby_at!` 126-136)
- `src/tensoroperations.jl` (`_qs_labels` 123-130)
- `docs/decisions.md` (Cause B ~3709-3720; frozen label semantics 8-29;
  orientation-swap out-of-scope note ~1395) -- on `store-fastpath-investigation`
- `benchmark/harness.jl`, `benchmark/bench_real_path_guard.jl`,
  `benchmark/bench_ccsd_t_store.jl`
- `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-15/summary_to_suite.txt`
  (baseline ccsd_t rows 235-254)
