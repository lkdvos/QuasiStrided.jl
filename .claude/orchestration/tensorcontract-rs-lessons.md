# tensorcontract-rs comparison: packing unification and store-path generalization

**Status: planner contract received 2026-09-22 (orch-planner, fable/high);
execution starting.** Coordinator: this session (Sonnet, `/orchestrate`).
Base `f7fa490`. The planner's read-only citation check (Section 0 of its
report, below) **corrected several claims in the paragraphs immediately
below** -- notably that item 2's tensorcontract-rs analog is a guarded
demotion rule (two thresholds), not a writeback-path rewrite, and that item 3
has only ONE run-length derivation (`_leading_unit_run`), not three. The
corrected design is authoritative; treat this file's original prose below the
status table as superseded where the two disagree (they are kept, unedited,
for provenance).

## Task status

| Task | Status | Worker | Notes |
| --- | --- | --- | --- |
| T0 | done | coordinator | HEAD `f7fa490` confirmed, tree clean except this file. `uptime`: load avg 1.80/1.76/2.41 (32-core box, quiet-ish). Julia 1.13.0. tensorcontract-rs HEAD `8cda75e` confirmed (matches planner). **Baseline `Pkg.test()`: 54203/54203 passing, 4m44.5s** (higher than decisions.md's 52679 at per-call-floor close -- later small commits added tests; not investigated further, not needed). **Correction to AC2**: Runic is NOT currently clean -- `julia -e 'using Runic; exit(Runic.main(["--check","--diff","src","test","benchmark"]))'` (Runic v1.10.0, installed into the global v1.13 env, was absent) exits 1 with pre-existing diffs in `test/packing/test_pack_real.jl` and `test/planning/test_per_call_overhead.jl` only (both outside every item's edit scope). AC2 is reinterpreted as "no NEW Runic violations beyond these two pre-existing ones" for this milestone; fixing them is optional opportunistic cleanup, not required, and out of scope unless the user asks. |
| T1 | done | orch-scout + coordinator backfill | orch-scout hit its 20-turn limit twice (same failure mode as the label-order milestone's T1 -- noted there too); completed items 3/4/5 itself, coordinator backfilled 1/2 directly. See findings below. |
| T2a | done | orch-builder | orch-builder hit its 40-turn limit once, nudged, delivered full report. `benchmark/bench_pack_loops.jl` (new). Results: `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/{pack_loops.csv,pack_loops_PROVENANCE.txt,profiles/}` (gitignored, on disk only). See findings below. |
| G1 | **PASS, decisively** | coordinator | max packing share 62.8% (`smallN_256x256x12_c32`, named-bucket-only) >> 10% threshold; also independently satisfied via ns/real ratio: complex Planar-B-full 0.4583 vs real-B-full 0.2904 = **1.58x** >= 1.3x. Only `shallowK_256x24x256` (~4.4-5.4%) is below the packing-share bar, but the ratio arm passes regardless. **-> T3a authorized.** |
| T2b | done | orch-builder | clean full run, no turn-limit issue this time. `benchmark/probes/probe_f2_kdepth.jl` (new); results `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/{f2_kdepth.csv,f2_kdepth_PROVENANCE.txt}`. 28/28 points + 2 extra reps, numeric agreement and m-independence both confirmed at every checkpoint, no `@warn` fired. |
| G2 | **PASS with a recorded residual (U2 fires)** | coordinator | `kmax(Float64)=32`, `kmax(Float32)=64`, `F2_BROKEN_ENOUGH=1.0` (inert). Q1/Q2/Q3 all pass; Q4 evidence conflicts between dtypes (contract's own anticipated "wins only at F32" case) -> inert per contract's explicit instruction. **U2 fires**: residual envelope gap up to 37.4% on the "less-broken" a=40/a=80 rows exceeds the 15% record-and-ask threshold -- recorded here, to be raised with the user at T6, not replanned now (contract pre-authorized exactly this disposition). AC5(a)'s "every point" will show 4/28 measured violations (all on a=40/a=80, none on the fully-broken a=8/a=16 rows, which hit ratio=1.000 everywhere) -- see full reasoning below. **-> T3b authorized with these constants.** |
| T3a | **done** | orch-builder x2 + coordinator finish | Commits `69f8e4f` (implementation) + `d27286e` (measurement tail). All of AC4a-d PASSED (AC4b narrowly, 0.9718 geomean; a mild unexplained-but-inside-threshold one-sidedness on real-dtype shapes noted honestly in decisions.md despite AC4a proving the real path IR is bit-identical). `docs/decisions.md` "tensorcontract-rs comparison (2026-09-22)" section opened with the "Item 1" subsection complete. **Unplanned finding flagged for T6/user**: `bench_complex_efficiency.jl` reads roughly half the complex-milestone's historical geomean on two canary-violating runs -- not chased, out of scope, recorded in decisions.md. |
| T3b | **done** | orch-builder (delivery got stuck; coordinator finished the verification tail directly from its diff) | Commits `5efa37b` + `96c976e`. Guard implemented exactly as designed (`_unbroken_fraction`, `F2_DEMOTE_KMAX_F64=32`, `F2_DEMOTE_KMAX_F32=64`, `F2_BROKEN_ENOUGH=1.0` inert). All AC5a-e confirmed by direct re-measurement/re-inspection (not trusted blindly from the subagent's unreachable report): fully-broken rows track optimum within noise; less-broken residual present as predicted (up to ~39%, U2, unchanged); ccsd_t_1 shallow-K win preserved by construction; full suite 64443/64443; Runic/forced-ISA clean (both documented residues only, no new ones). One benchmark-script (not driver) labeling bug found and resolved by direct check, documented in decisions.md, not fixed (out of scope). **Process note**: this subagent got stuck repeating "finished" notifications with no deliverable content across 6+ resume attempts; stopped via `TaskStop` and the work was completed directly by re-reading its `git diff` and independently re-running every verification step. |
| G3 | **PASS** | coordinator | T3b merged -- item 3 authorized. |
| T3c | **done** | coordinator (implemented directly, given the pattern of subagent stalls on this task) | Commits `65ae65a` + `2730a5a`. Corrected two things found wrong in the original comparison/contract: (1) there was only ONE run-length derivation, not three -- fixed to compute-once/reuse; (2) the proposed `affine_ramp` relation test was WRONG as specified (compared against the two-map `mgroup`, which also constrains operand A -- a strictly stronger condition); shipped the corrected single-map-`AxisGroup` relation instead, verified over 2000 randomized draws (two genuine edge-case mismatches found and excluded before it passed clean). Pure dead-computation removal, no decision-semantics change. AC6: allocation byte-identical on all 4 fixtures; one single-shot ~60% timing swing on `smallMN_16x256x16` traced to noise (vanished under repeated median measurement) rather than accepted at face value. Full suite 66447/66447, Runic/forced-ISA clean. |
| T5 | **done** | orch-reviewer | **No blocking findings.** 5 should-fix (S1-S5) + several nits, all addressed in commit `2171456`: S1 (a real, previously-unmeasured added cost from the "inert" fraction guard -- fixed by reordering/gating), S4 (complex plans were paying for `_leading_unit_run` calls the old code never made -- fixed by gating on `T<:Real`), S5 (missing complex tail-sliver allocation coverage -- added), S2/S3 (decisions.md numbers didn't match the current on-disk artefact after an overwrite, and a flag paragraph misattributed a pre-existing complex-efficiency drop to this milestone -- both corrected), N1/N4/N7/N8/N9 (comment/doc corrections + boundary tests). Full suite 66451/66451 after fixes; Runic/forced-ISA clean. Reviewer also independently re-derived and confirmed the milestone's headline numbers (test-count arithmetic, AC5a ratios, ABBA geomean, profile shares) against on-disk artefacts. |
| T6 | **done -- MILESTONE CLOSED** | coordinator | Commit `f312988`. `STATUS.md`'s stale "Next task" section fixed (per-call-floor was already shipped, no entry existed; evidence-gate status corrected to "partially done, smoke-tested"), complex-element-type milestone header fixed from "open" to "complete", a full per-call-floor milestone entry added (backfilling a gap T1 found), and this milestone's own close-out entry added with the three flagged-not-fixed items for the user surfaced explicitly. |

Full contract from `orch-planner` is reproduced verbatim at the bottom of
this file, under "Full contract (orch-planner output, verbatim)".

## T1 findings (residual fact check)

1. **CONFIRMED** (coordinator, in T0): tensorcontract-rs HEAD is `8cda75e`.
2. **CONFIRMED** (coordinator backfill; orch-scout ran out of turns before
   executing Julia). Fixture `C[a,b,c,i,j,k]=A[i,j,m,a]*B[m,k,b,c]`,
   `i=j=k=b=c=6`, `Qn=216` fixed. `_demote_for_run`'s current signature
   (no `Qk` parameter yet) is at `src/planning/kernel_selection.jl` (real) / `:282`
   (complex no-op); call sites at `:1054` (swap decision), `:1062-1066`,
   `:1074`. **F2 fires unconditionally at every sweep point, independent of
   `m` (=Qk) and independent of the "broken-enough" fraction** -- `auto`'s
   kernel `!== _default_kernel(T,Qm,Qn)` (i.e. demoted) at all 8 of
   `{(F64,8),(F64,40)} x {m=8,m=512}` and `{(F32,16),(F32,80)} x {m=8,m=512}`.
   Auto/default pairs: F64 `(8,6,4)` vs `(16,6,8)`; F32 `(16,6,8)` vs
   `(32,6,16)`. This confirms the sweep design is sound -- T2b's `a=40`/`a=80`
   points genuinely test "does unconditional demotion help or hurt at a
   less-broken shape", not a tautology.
3. **CONFIRMED, matches contract** (orch-scout): label order is preserved,
   not reordered, in `_build_pair_group` (`src/layout/pair_group.jl`).
4. **CONFIRMED, matches contract** (orch-scout): `:mps`/`:ctmrg`/`:trg` wired
   (`docs/decisions.md:5324`) and even smoke-tested with real numbers on
   record (`:5382`, `:5389-5451` -- QuasiStrided 1.03-1.25x slower than
   StridedBLAS across the three categories at the smoke-test's sizes). **This
   is new information not in the original brief or the planner's contract**:
   the evidence gate for the dispatch-tiers proposal (`docs/proposals/
   dispatch-tiers.md` section 5.1) has *some* real numbers already, from a
   smoke test, not the full Slurm run (`benchmark/submit_evidence_gate.sh`
   exists, confirmed not yet submitted, per `STATUS.md:422`). Out of scope for
   this milestone (U1 is informational only); flagged for T6/STATUS.md
   accuracy and for the user's own awareness, not actioned here.
5. **CONFIRMED, matches contract** (orch-scout): `PlanarKernel`/`OneMKernel`
   are `DescriptorKernel`s whose `pack_a!`/`pack_b!` forward to their
   descriptor (`src/microkernels/interface.jl`), reaching `src/packing/pack.jl`. T2a's
   microbenchmark plan (through `plan.kernel`) is valid.

No Section-0 contradiction found (the one REPLAN trigger tied to T1 --
"finds a Section-0 fact wrong in a contract-changing way" -- does not fire).
Proceeding to T2a/T2b.

## T2a findings (item-1 packing evidence) and G1 verdict

Profiling share (named-buckets-only, `other` bucket was 0.05-0.32% in all 10
cases -- no idle-thread artifact this run): `smallN_256x256x12` **62.8%
(c32) / 48.7% (c64)**, `smallMN_16x256x16` ~30-31%, `smallM_12x256x256`
~28-29%, `plain_64` 18.2% (c64) / 26.9% (c32); only `shallowK_256x24x256`
stays low (4.4-5.4%).

Microbenchmark (ns per real emitted, 21 reps, canary spread 4.05%, host had
some background load: `uptime` load avg 1.65/2.63/2.78): complex Planar-B
full-sliver **0.4583** vs real-B-full-sliver (Float64, direct loop, not the
contiguous fast path) **0.2904** -- **1.58x**. Planar-A full-sliver is closer
(0.383 vs 0.345, 1.11x) -- the branch cost shows up more on B (scalar loads
already) than A.

Code inspection: `_pack_panel_complex!`'s `t < valid` conditional lowers to a
genuine LLVM branch (`icmp` + `br`), not a select/masked store -- confirmed
directly for `ComplexF32`; inferred (not independently re-checked) for
`ComplexF64` from identical source structure, flagged honestly by the worker.

**G1: PASS, decisively, on both independent arms of the gate rule** (packing
share arm: 62.8% >> 10%; ns/real ratio arm: 1.58x >= 1.3x on Planar-B).
**-> T3a (item-1 unification, variant 1-A) is authorized.**

## T2b findings (item-2 F2 K-depth sweep) and G2 verdict

Full 28-point sweep (2 dtypes x {fully-broken, less-broken} x 7 `m` values)
plus 2 extra reps of the flagged point, clean run (canary 6.8%, no
consistency-check failures). Kernel shapes: `auto` matched `demoted` at
**every single point measured** -- F2 fires unconditionally today, confirmed
at scale, not just at the two T1 spot-checks.

**Q1 (reproduces?): PASS.** `Float64, a=8, m=512`: `r` = 1.340, 1.366, 1.359
across 3 independent reps (>= 1.10 threshold). Direction confirmed; the
worker could not locate an on-disk artifact for the original "18-25%"
figure to reconcile the exact number (no
`benchmark/probes/probe_ccsd_t_stall_f2rule.jl` on disk) -- the regression's
*existence* is reproduced, its *exact prior magnitude* is not cross-checked
against a saved source. Not a REPLAN trigger (the rule only requires `r`
direction/magnitude at this one point, which it has).

**Q2 (crossover): PASS for both fully-broken fixtures.**
- `Float64, a=8`: `r<0.95` at `m=8` (0.554), `r>1.05` at `m=512` (1.340).
- `Float32, a=16`: `r<0.95` at `m=8` (0.341), `r>1.05` at `m=512` (1.376).
`kmax(Float64) = 32`: last `m` with `r<=0.95` (0.868) and every smaller `m`
also `<=0.95`; first losing `m=64` (1.098). `kmax(Float32) = 64`: last
passing `m=64` (0.800), first losing `m=128` (1.039). Both boundaries are one
sweep-step wide, so the contract's power-of-two-midpoint option applied --
both midpoints (45.25, 90.5) are exact log-space ties between the adjacent
powers of two, so the primary (unambiguous) rule value is kept in both cases
rather than invoking the tiebreak.

**Q3 (sanity): PASS.** `kmax(Float32) = 64 >= 16`.

**Q4 (broken-enough guard): ambiguous, resolved per the contract's own
anticipated disposition.** `Float64, a=40` (default fraction ~0.8): `r > 1.0`
at every single `m` (1.021 to 1.532) -- supports adopting `0.75`.
`Float32, a=80` (same fraction): `r < 1.0` at `m=8` (0.864) and `m=16`
(0.985) -- demotion *wins* at small `m`, contradicting the same threshold.
This is exactly the contract's pre-named "wins only at F32" case, whose
instruction was explicit: keep one constant, do not split by dtype -> **adopt
`F2_BROKEN_ENOUGH = 1.0` (inert)**. The K-depth cutoff (`kmax`) is therefore
the only active guard; the fraction guard exists in the interface (so a
future pass can revisit it with more evidence) but never fires.

**No REPLAN trigger fired**: demotion does win somewhere in both dtypes
(`m=8`: 0.554 / 0.341), a genuine crossover exists in both, `kmax(Float32)`
clears its floor.

**U2 fires: recorded, not acted on.** With `kmax`+inert-fraction as the only
guard, re-deriving `t_rule(m) = t_demoted` if `m<=kmax(T)` else `t_default`
against the actual 28-point table shows the rule tracks `min(default,
demoted)` **exactly (ratio 1.000) on every point of both fully-broken rows**
(`a=8`, `a=16` -- the case this item exists to fix) but **not** on the two
less-broken rows: `Float64, a=40` at `m=16` (ratio 1.148) and `m=32` (1.374);
`Float32, a=80` at `m=32` (1.107) and `m=64` (1.292). Worst case **37.4%**
over the point-wise-optimal choice -- above the 15% U2 threshold. This is a
direct consequence of Q4's forced choice (one constant, inert, chosen because
the two dtypes disagree) and was explicitly anticipated by the contract's own
U2/Option-B framing, which says to record and ask the user at close, not
build Option B or invent a second per-dtype constant now (both are outside
the coordinator's decision authority per Section 6). **Practical effect on
AC5(a)** ("`auto` <= 1.05x min at every point"): PASSES on all 14 fully-broken
points (ratio exactly 1.000); **4 of the 14 less-broken points measurably
violate it** (1.148, 1.374, 1.107, 1.292) -- T3b's decisions.md subsection
must report AC5(a) this way (split by fixture class, not as a single
pass/fail), and T6 must raise U2 with the user rather than let it read as a
silent partial failure.

**-> T3b is authorized** with `kmax(Float64)=32`, `kmax(Float32)=64`,
`F2_BROKEN_ENOUGH=1.0` (documented as inert with the reason above), and the
AC5(a) reporting caveat carried into T3b's brief.

---

**Original brief below (superseded where the contract above disagrees).**
Written 2026-09-22 against `main` @ `f7fa490` (after the per-call-floor merge). This
file exists to be handed to `/orchestrate` (which invokes `orch-planner`) --
it deliberately has not done the read-only fact-finding pass a real milestone
open requires (see T1 below), and every file:line citation in it is
subagent-derived from a single read-through, not independently re-verified.

## Origin

A general-purpose subagent read this package against its Rust sibling
project, `tensorcontract-rs` (`/mnt/home/ldevos/Projects/tensorcontract-rs/main`,
`8cda75e`), which implements the same algorithm (Matthews' block-scatter-matrix
contraction inside a BLIS five-loop nest) and was already the acknowledged
design source for this package's complex-method work (STATUS.md, "Complex
element-type milestone"). The subagent produced a 5-item prioritized list.
**Two of those five are moot as of this brief**, discovered only by checking
git history after the fact -- record this so nobody re-opens either:

1. **"Allocation-light `_classify_labels`/`_order_free_labels`" -- already
   shipped.** This is exactly what the per-call-floor milestone did
   (`7e0e28b`, review fix `553cfd7`, merged `f7fa490`; narrative in
   `docs/decisions.md`, "Per-call floor: cheaper planning, once-per-block
   bounds validation, closed-form affine blocks"). `plan_contract` went from
   4.1-6.4 us / 5.0-7.0 KB to 1.0-2.3 us / 1.8-2.6 KB per call, measured.
   **No further action.** `STATUS.md`'s "Next task" section still describes
   this as open work -- it is stale and needs a housekeeping fix (see
   close-out, below), not a re-run.
2. **"Opt-in cache-blocking model fed by `target.jl`'s `cache_topology()`" --
   already tried and explicitly rejected, before this comparison ever ran.**
   `src/planning/blocking.jl`'s `default_blocking` docstring: "re-measuring the
   36-point grid found it spans only 9%/11% best-to-worst" -- the model's own
   upside is a few percent. tensorcontract-rs shipping its own analytical
   model off-by-default (`kernel/cache.rs`) for essentially the same reason
   (measured constants win on measured hardware; the model exists for
   *portability* to unmeasured hardware) is **confirming** evidence for the
   existing rejection, not a new case to reopen it. **Do not build this.**

What's left is three items, all genuinely open, none previously measured and
rejected:

## Scope (3 items)

1. **[simplify] Unify real and complex packing into one format-dispatching
   routine.** `src/packing/pack.jl` still keeps `_pack_panel!` (real) and
   `_pack_panel_complex!` (complex) as two structurally separate loops. This
   was deliberate during the complex-element-type migration (to keep the real
   path's diff byte-identical while that landed) and the per-call-floor
   milestone's own "Deferred" list confirms it is *still* untouched and
   unrelated to that pass ("Complex packing's conditional load
   (`_pack_panel_complex!`) -- still untouched"). tensorcontract-rs's
   `pack.rs` (`pack_panel` + a format-dispatching `emit`, one traversal for
   Real/Planar/OneE/ThreeM alike) is evidence this generalizes cleanly without
   a per-format loop. The migration-era reason for the split has served its
   purpose; unifying now would cut real duplicated bounds-check/full-tail-split
   logic. Needs a measured before/after (this project never ships an
   unmeasured refactor) and a check that it doesn't regress the now-fast real
   path.

2. **[optimize] Generalize F2's shape-menu store demotion into a
   regularity-aware writeback path.** F2 (`_demote_for_run`, shipped on `main`
   since `a0b337a`) picks the largest kernel-shape menu entry whose `mr`
   divides `C`'s run length, to keep the vectorized store path reachable. Its
   own review recorded a real, documented, **unfixed** limitation: no
   `Qk`-aware cost model, so on a large-K case where it fires it is 18-25%
   *slower* than not demoting (`docs/decisions.md`, "Kernel-stalled/
   store-dominated fix", "Post-review fixes" subsection). tensorcontract-rs's
   `writeback.rs` decouples "is this block regular" from "which kernel shape
   to run" -- it drives the store fast path off the same block-scatter
   regularity test packing already uses, rather than picking a different,
   smaller microkernel. That is a structurally different fix for the same
   problem F2 patches around, and it's worth scoping as an alternative or a
   replacement, not a second heuristic stacked on F2 (the risk F2's own review
   already flagged).

3. **[simplify] Consolidate the three independent "run length in C"
   derivations.** `_leading_unit_run`, `_prefer_swap`, and `_demote_for_run`
   (all `src/planning/kernel_selection.jl`) each re-derive a version of the same quantity for a
   different purpose (orientation swap, store-path eligibility, shape
   demotion). tensorcontract-rs's `scatter::run_structure` answers "is this a
   concatenation of equal-length maximal runs, and what's the run
   length/stride" once, and both its orientation and demotion-equivalent logic
   consume that single answer. Note that the per-call-floor milestone's new
   `affine_ramp`/`_ramp_descriptor` (`src/layout/axis_group.jl`) already built
   adjacent closed-form-block machinery for a different purpose (bounds-check
   hoisting) -- check whether it's directly reusable here before writing a
   fourth primitive. This item is naturally sequenced *after* item 2, since a
   writeback rewrite may itself absorb or reshape the run-length logic it
   would otherwise be consolidating.

## Non-goals

3m/Karatsuba complex method; the batched-contraction API (`batch.rs` has no
QuasiStrided equivalent and nothing in this package's measured workload is
batch-count-bound); threading; a cache-blocking model (rejected, see above);
anything in `_classify_labels`, `_build_pair_group`, `_order_free_labels`, or
the bounds-check hoist (per-call-floor's shipped code -- frozen unless a
regression is found); `QuasiStridedBackend`'s hard-reject invariant;
`src/hardware/target.jl`'s register-shape derivation; `default_blocking`'s measured
constants.

## Suggested task shape (for `orch-planner` to expand, not gospel)

- **T0 (coordinator):** branch off `main` @ `f7fa490`; confirm baseline
  `Pkg.test()` green and record the count (per-call-floor's close reported
  52679 passed); fold in the `STATUS.md` staleness fix (item 1's "Next task"
  section still describes shipped work as open) as a housekeeping task, not a
  gate.
- **T1 (orch-scout):** re-verify every citation above against current `main`
  HEAD and current `tensorcontract-rs/main` HEAD (both were read once, by a
  subagent, and are not independently confirmed). Specifically: is
  `_pack_panel_complex!` still a real fraction of runtime worth unifying, or
  has something since changed its cost share; is F2's K-blind regression still
  reproducible on current `main`; read `pack.rs`/`writeback.rs`/`scatter.rs`
  in full (the subagent skimmed some sections) and confirm the mechanisms
  described above are accurately characterized, not paraphrased loosely.
- **T2 (gate, coordinator):** per this project's own convention (see
  `docs/proposals/dispatch-tiers.md`'s gate pattern), decide per item whether
  it clears a measurement bar *before* any `src/` edit. Do not implement any
  of the three on the strength of this brief alone.
- **T3+ (orch-fable / orch-builder, per item):** implement whichever items
  clear T2's gate. Each independently measured (ABBA guard, allocation
  assertions on both real and complex fixtures, Runic, full suite), each
  behind its own dated `docs/decisions.md` section, following this project's
  standing rule of no unmeasured refactor.
- **T-review (orch-reviewer):** one gated pass, scoped like prior milestones'
  (correctness of any changed conjugation/bounds semantics, allocation,
  forced-ISA portability).
- **T-close:** `STATUS.md`/`docs/decisions.md` close-out, including the
  per-call-floor staleness fix noted in T0.

## How to launch

Run this brief through the `orchestrate` skill (`/orchestrate`), pointing it
at this file. `orch-planner` should treat T1 as a hard prerequisite to
producing acceptance criteria and a task graph -- the citations here are a
starting point for its own read-only fact-finding pass, not verified facts.

## Reference: file locations

- QuasiStrided.jl: `src/packing/pack.jl` (`_pack_panel!`/`_pack_panel_complex!`),
  `src/planning/labels.jl` (`_leading_unit_run`/`_prefer_swap`/`_demote_for_run`),
  `src/layout/axis_group.jl` (`affine_ramp`/`_ramp_descriptor`),
  `src/microkernels/simd.jl` (`_store_tile_vector!`/`store_tile!`),
  `docs/decisions.md` ("Kernel-stalled/store-dominated fix", "Per-call
  floor..."), `docs/proposals/dispatch-tiers.md` (gate-pattern precedent).
- tensorcontract-rs (`main` worktree): `crates/tensorcontract/src/pack.rs`
  (`pack_panel`/`emit`), `crates/tensorcontract/src/writeback.rs`,
  `crates/tensorcontract/src/scatter.rs` (`run_structure`), `docs/decisions.md`
  and `docs/design.md` in that repo for rationale.

---

## Full contract (orch-planner output, verbatim)

# Execution contract: tensorcontract-rs comparison milestone (QuasiStrided.jl)

Working tree: `/mnt/home/ldevos/Projects/QuasiStrided.jl/compare` (branch `compare`, HEAD `f7fa490` per coordinator; T0 verifies). Rust sibling: `/mnt/home/ldevos/Projects/tensorcontract-rs/main`. Brief: `/mnt/home/ldevos/Projects/QuasiStrided.jl/compare/.claude/orchestration/tensorcontract-rs-lessons.md`.

## 0. Citation verification of the brief (done here, read-only) -- corrections that change the contracts

I read every cited file in both repos. Confirmed-or-corrected table (all EVIDENCE unless marked):

| Brief claim | Status | Where / what is actually true |
|---|---|---|
| `_pack_panel!` (real) and `_pack_panel_complex!` are two structurally separate loops | CONFIRMED | `src/packing/pack.jl` (real: full/tail split, no conditional load) vs `:401-417` (complex: `if t < valid` per element inside the inner loop -- the exact anti-pattern the packing-speed fix removed from the real loop; header comment `:310-319` explains the deliberate split) |
| tensorcontract-rs `pack.rs` `pack_panel`+`emit` is a format-dispatching pattern QS lacks | PARTLY WRONG | QS already has the `emit` half: `_pack_emit!`/`_pack_emit_zero!` dispatch on `PlanarFormat`/`OneEFormat` at `src/packing/pack.jl`. `RealFormat` exists (`src/packing/format.jl`, `a_format(::KernelDescriptor) = RealFormat()` at `:204`) but has NO emit method. What is missing is (a) a `RealFormat` emit and (b) ONE loop with the full/tail split. `pack.rs:54-107,125-157` confirmed as described (one traversal, `emit` match on format, `bs`-regular fast path). |
| Complex packing is "still a real fraction of runtime" | UNVERIFIED -> now partly answered | Local (gitignored, `.gitignore:6`) artefact `benchmark/results/ccqlin038.flatironinstitute.org-2026-09-22/profiles/buckets_summary.txt`: `ccsd_t_1_dim16_c64` packing 1.42%, store 30.28%, microkernel 17.30% (both backends show exactly 50% "other" -- the idle-thread artefact `profile_to_suite.jl:24-31` warns about, so named-bucket shares are ~2x these). Complex efficiency (`complex_efficiency.csv`, same dir, HEAD clean): ComplexF64 geomean 0.96, ComplexF32 0.75; ComplexF32 `shallowK` 0.50, `smallN` 0.66, `smallM` 0.74, `smallMN` 0.74 -- the small shapes are where a packing share could hide; UNMEASURED. |
| F2 has a documented unfixed K-blind 18-25% regression | CONFIRMED | `docs/decisions.md:5255-5293` (S1). Fixture: `C[a,b,c,i,j,k] = A[i,j,m,a]*B[m,k,b,c]`, Float64, `a=8`, `i=j=k=b=c=6`, `m=512`; auto (8,6,4) 1.735e-3/1.750e-3 s vs forced `SIMDKernel(Val(16),Val(6),Float64,Val(8))` 1.415e-3/1.402e-3 s. Author's own recommendation: "gate the demotion on `Qk` being small relative to `Qm*Qn`, or on a direct cost estimate". Whether it reproduces on `f7fa490` (after per-call-floor) is the G2 measurement. |
| `writeback.rs` is "a structurally different fix ... rather than picking a different, smaller microkernel" | WRONG, and this reverses item 2's design | `writeback.rs:172-183` dispatches per tile on the block-scatter stride (`IRREGULAR` -> gather, `1` -> unit, else strided). QS already has this: `store_tile!` at `src/microkernels/simd.jl` dispatches on `_vector_store_eligible` (`:180-181`), whose row axis is `AffineAxis` iff the sliver is regular (`_axis_of`, `src/execution/macrokernel.jl`). tensorcontract-rs's actual fix for straddling blocks is `Plan::preferred_row_block` (`plan.rs:933-950`): an MR-MENU DEMOTION -- i.e. F2 -- with two guards: `SHALLOW_K = 32` (`stats.k > 32` -> never demote) and `BROKEN_ENOUGH = 0.75` (default's `unbroken_fraction` above 0.75 -> never demote), picking the first menu shape with fraction >= 1.0. Its docs (`plan.rs:895-911`) say the unguarded rule is a net LOSS (0.936 in f32) -- exactly F2's S1 finding. `run_structure` (`scatter.rs:179-204`) and `unbroken_fraction` (`:219-232`) feed it. The transferable lesson is the two guards, not a store-path rewrite. |
| Three independent "run length in C" derivations | WRONG | There is ONE derivation, `_leading_unit_run` (`src/planning/labels.jl`). `_prefer_swap` (`:247-253`) calls it twice; the F2 call sites (`:1065-1067`, `:1073-1075`) call it once more for the chosen orientation. `_demote_for_run` (`:270-282`) consumes `run`, it does not derive it. Redundancy = one extra call per plan (2 values, 3 calls). |
| `affine_ramp`/`_ramp_descriptor` may be directly reusable | NO | `affine_ramp(g)` (`src/layout/axis_group.jl`) answers "is EVERY map of the group a single ramp over the whole domain" (Bool + steps); `_leading_unit_run` answers "length of the leading stride-+1 run of ONE map (C)". Different question. The checkable relation is `run_m == Qm  <=>  affine_ramp(mgroup)[1] && steps[C-map] == 1`. Also: QS enumerates composites mixed-radix first-label-fastest, so the C-offset sequence is a concatenation of EQUAL-length maximal runs of length `run` -- `_leading_unit_run` already IS `run_structure` for this engine, so `unbroken_fraction(Qm, run, mr)` is closed-form here. |
| Menus | -- | `KERNEL_SHAPES_F64 = ((8,6,4),(16,6,8))`, `KERNEL_SHAPES_F32 = ((8,6,8),(32,6,16),(16,6,8))` (`src/planning/labels.jl`). Every real demotion halves `W`. |
| Tests that pin the touched functions | -- | `test/planning/test_plan_contract.jl` binds `_leading_unit_run`/`_prefer_swap` by name and asserts on the 5/6-arg `_prefer_swap` signature; `:1205-1260` F2 testset (ISA-independent since `553cfd7`-era fix, derives from `kernel_shapes(T)`). `test/packing/test_pack_complex.jl` (12 testsets, `:26-502`) exercises ONLY `pack_a!`/`pack_b!` on a `ComplexKernelDescriptor` -- it never references `_pack_panel_complex!`/`_pack_emit!`, so the loop can be restructured freely. `test/packing/test_pack_real.jl` binds `_copies_unchanged`, `_pack_a_contiguous_eligible`, `_pack_a_contiguous!` (must keep names/signatures); `_pack_panel!` appears only in comments. `test/planning/test_per_call_overhead.jl` pins `plan_contract` allocation `<= 3000` B. |
| STATUS.md staleness | CONFIRMED, plus more | `STATUS.md:397-450` "Next task" still lists per-call floor as pending ("See docs/decisions.md ... once it lands"); header `:476` says "Complex element-type milestone -- open" while `:598` says "Milestone closed"; no per-call-floor section exists in STATUS.md at all (header list `:8-776`). |
| decisions.md sections | -- | F2: `:5019-5323`; per-call floor: `:5466-5797` (Deferred complex packing `:5715-5717`); `bench_to_suite` mps/ctmrg/trg wiring `:5324-5464`. |
| Verification tooling | -- | Runic via CI `.github/workflows/FormatCheck.yml` (runic-action); CI matrix `lts` + `1`. ABBA guard `benchmark/bench_real_path_guard.jl` (21 reps, `GUARD_SHAPES = MAIN+SMALL` = 8 shapes x 2 dtypes = 16 rows... prior passes report "18 shapes" incl. scattered; `QS_GUARD_LABEL` env for the archived tree, `:49-64`). Profiler `benchmark/profile_to_suite.jl` has complex case ids `<shape>_c64`/`_c32` for MAIN+SMALL (`:161-182`), `--tag`. Complex headline `benchmark/bench_complex_efficiency.jl` (21 reps). Fixture sweeps `benchmark/bench_ccsd_t_store.jl --dims 8,16 --dtypes Float32,Float64`. |

Residual facts needing command execution -> T0/T1 (ASSUMPTION until run): HEAD/clean tree; baseline test count (52679 reported at per-call-floor close, `docs/decisions.md:5686`); tensorcontract-rs HEAD `8cda75e`; Julia version (artefacts say 1.13.0); Runic locally available; machine load; whether the `:mps/:ctmrg/:trg` Slurm evidence-gate run (`benchmark/submit_evidence_gate.sh`, commit `17eefbf`) has results anywhere.

## 1. Goal, acceptance criteria, assumptions, non-goals

**Goal.** For each of the three open items, run its own measurement gate; ship an implementation only where the gate passes, each with before/after measurement, tests, Runic, full suite, and a dated `docs/decisions.md` subsection; otherwise ship a documented decision. Fix STATUS.md staleness. Nothing merges on the strength of the comparison alone.

**Acceptance criteria (measurable).**
- AC1: `julia --project=. -e 'using Pkg; Pkg.test()'` green at close; count >= T0 baseline + new tests; `QS_FAKE_ISA=avx2 QS_FAKE_VB=32 QS_FAKE_NREG=16` and `QS_FAKE_ISA=unknown QS_FAKE_VB=0 QS_FAKE_NREG=0` runs of `test/forced_isa_runner.jl` show only the residues its header documents.
- AC2: Runic check clean on `src/`, `test/`, `benchmark/`.
- AC3: `docs/decisions.md` gains one dated section "tensorcontract-rs comparison (2026-09-22)" with a subsection per item stating gate inputs (numbers + artefact paths), decision, and -- if shipped -- before/after with ratio direction spelled out. Every number re-derived from an on-disk artefact (Phase G lesson, `STATUS.md:586-588`).
- AC4 (item 1, if shipped): (a) `Base.code_llvm` of the real `_pack_a!` and `_pack_b!` at the driver's argument types is identical to base modulo SSA naming/metadata ids (primary criterion); (b) two-tree ABBA guard (>= 21 reps, >= 2 runs per tree, A-B-B-A) geomean new/base in [0.97, 1.03] and no one-sided shift (not >= 14 of 16-18 shapes moving the same direction by > 2%); (c) complex packing microbenchmark ns-per-real-emitted improves >= 1.2x on at least the A planar full-sliver and B planar cases, and `bench_complex_efficiency.jl` geomeans not below base by more than the run's canary spread; (d) existing complex zero-allocation testset (`test/packing/test_pack_complex.jl`) and layout/conj/padding pins pass unmodified.
- AC5 (item 2, if shipped): (a) on the K-sweep fixtures (Section 3, T2b), post-change `auto` time <= 1.05 x min(forced-default, forced-demoted) at EVERY point, both dtypes; (b) regression fixture (F64, a=8, m=512): `plan.kernel === _default_kernel(Float64, Qm, Qn)` (no demotion) and time within 5% of forced-default; (c) `bench_ccsd_t_store.jl --dims 8,16 --dtypes Float32` still shows ccsd_t_1 demoting and its QuasiStrided arm within 5% of the pre-change value (the 2.17x/1.92x wins preserved); (d) F2 testset + new tests derive every expectation from `_default_kernel`/`kernel_shapes(T)`/the shipped constant, never a literal `mr`; (e) ABBA guard: no one-sided shift (F2 never fires on plain GEMM, so this is a lost-specialization check only).
- AC6 (item 3, if shipped): swap and demotion decisions unchanged on every existing test (test_driver.jl label-order and F2 testsets pass without semantic edits); `test/planning/test_per_call_overhead.jl` 3000 B ceiling holds; `plan_contract` time on the four per-call-floor fixtures (`plain_64`, `smallMN_16x256x16`, `ao2mo_2_dim16`, `ccsd_t_1` dim 4) within noise of base (report before/after; no acceptance on speed, only on non-regression).
- AC7: STATUS.md: "Next task" rewritten (per-call floor complete with its numbers; evidence-gate status stated honestly), complex header `:476` reads "complete", a short per-call-floor entry added, this milestone recorded.

**Assumptions.** EVIDENCE: everything in Section 0's table. ASSUMPTION (T0/T1 verify): HEAD `f7fa490`, tree clean except the brief; baseline 52679; `ccqlin038` is the measurement host and can be made quiet enough (canary spread <= ~10%); Julia 1.13.x local; Runic installable in a scratch env; `Manifest.toml` in the working tree is valid for an archived base tree (Manifest is gitignored, so `git archive` of base has none -- copy the working tree's).

**Non-goals (frozen).** As in the brief: 3m/Karatsuba, batched API, threading, cache-blocking model, `_classify_labels`/`_build_pair_group`/`_order_free_labels`/bounds-check hoist, `QuasiStridedBackend` hard-reject, `src/hardware/target.jl` register-shape derivation, `default_blocking` constants. ADDED here: `_default_kernel` and the real/complex menus; `_prefer_swap`'s decision semantics (S2 joint optimisation stays deferred); the `_axis_of` Union (GUARDRAIL `src/execution/macrokernel.jl`); `src/packing/format.jl`; complex kernels' store paths (`planar.jl:341`, `onem.jl:227` scatter unconditionally -- out of scope); Option B below (piecewise-affine vector store) -- record only.

## 2. Design decisions and interfaces

### Item 1 -- unified packing loop (variant 1-A), fallback 1-B, no-op 1-C
- 1-A interface (all in `src/packing/pack.jl`):
  `_pack_panel!(packed::V, ::Type{T}, format::FMT, ::Val{PD}, kc::Int, valid::Int, transform::F, load::L, plane_offset::P) where {V,T,FMT<:PackFormat,PD,F,L,P}` -- body = today's real full/tail split (`:123-138`) with `panel_store!(packed, packed_offset(i,p), ...)` replaced by `_pack_emit!(packed, format, plane_offset, t, p, z)` and `zero(T)` stores by `_pack_emit_zero!(packed, format, plane_offset, t, p, real(T))`. New methods `_pack_emit!(packed, ::RealFormat, plane_offset, t, p, z) = panel_store!(packed, plane_offset(0,t,p), z)` and the matching `_pack_emit_zero!`. Real callers (`:257-259`, `:304-306`) pass `RealFormat()` and `plane_offset = (plane,i,p) -> packed_a_offset(kernel,i,p)` (resp. `packed_b_offset`). Complex callers (`:462-464`, `:504-506`) pass `Val(MR)`/`Val(NR)` instead of `Int physical_dim`. Delete `_pack_panel_complex!`; rewrite the header comment `:310-319` (its rationale is being retired on purpose -- say so, cite this decisions.md section). `_pack_a_contiguous_eligible`/`_pack_a_contiguous!`/`_copies_unchanged` untouched (test-bound).
  Rationale: at `RealFormat` the emit inlines to exactly the current store, so the real IR should be identical (verify, AC4a); the complex loop gains the full/tail split that measured 1.3-2x on the real loop (`docs/decisions.md` "Packing speed"). Net source delta is modest (~-15 lines); the win is the complex loop, not the line count -- state this honestly in the record.
- 1-B (fallback if 1-A's real IR is not identical after one repair attempt): leave `_pack_panel!` byte-identical; give `_pack_panel_complex!` the full/tail split only.
- 1-C: no code; record the gate numbers.
- Conventions: `transform` applies to the complex element BEFORE splitting (`:324-329`); padding writes literal zeros without reading/transforming (`:377-380`, `-0.0` hazard); `T` is the storage type, buffer holds `real(T)`.

### Item 2 -- K-aware and regularity-aware F2 (variant 2-A); Option B deferred
- 2-A interface (`src/planning/kernel_selection.jl`): `_demote_for_run(::Type{T}, kernel, run::Int, Qm::Int, Qk::Int) where {T<:Real}`; generic method unchanged (no-op). New `_unbroken_fraction(Qm::Int, run::Int, mr::Int)::Float64` = fraction of the `cld(Qm, mr)` register slivers `[s*mr, min((s+1)*mr, Qm))` lying inside one run, i.e. `lo ÷ run == hi ÷ run`; by construction `== 1.0 <=> (Qm == run || run % mr == 0)` (the existing predicate -- add a test asserting this equivalence over a random grid). New constants next to `KERNEL_SHAPES_*`: `F2_DEMOTE_KMAX_F64::Int`, `F2_DEMOTE_KMAX_F32::Int` (values from G2), `F2_BROKEN_ENOUGH::Float64` (0.75 adopted from tensorcontract-rs if G2 Q4 validates its direction, else 1.0 = inert), each with a comment naming the measurement (house style of `default_blocking`). Rule: return `kernel` unchanged if `Qk > kmax(T)` or `_unbroken_fraction(Qm, run, mr(kernel)) > F2_BROKEN_ENOUGH`; otherwise today's search (largest menu `mr` with `run % m == 0`). Call sites `:1065-1067`, `:1073-1075` pass `Qk`.
  Rationale: this is precisely tensorcontract-rs's guarded row-block rule (`plan.rs:933-950`), calibrated on THIS machine rather than transplanting `32`/`0.75`. It removes the S1 regression class without stacking a new heuristic: it narrows the existing one.
- Option B (piecewise-affine vector store for straddling slivers, keeps the wide kernel): NOT built. Record the envelope gap from the sweep (best single threshold vs per-point best arm) as its measured upper bound; it is the per-call-floor "piecewise-affine slivers" deferred idea (`decisions.md:5707-5714`) and would widen the `_axis_of` Union (GUARDRAIL) and add a `@generated` store path -- a separate proposal if the gap ever exceeds ~10% on a measured-population case (none exists today: all F2-firing measured cases have `Qk = 8/16`).

### Item 3 -- one derivation, explicit consumers (conditional on 2-A)
- Interface (`src/planning/plan.jl`): in `plan_contract`, after `morder`/`norder` (`:1031-1032`): `run_m = _leading_unit_run(morder, indC, C)`, `run_n = _leading_unit_run(norder, indC, C)`. New core `_prefer_swap(run_m::Int, run_n::Int, mr_asis::Int, mr_swapped::Int = mr_asis) = run_m < mr_asis && run_n >= mr_swapped`; keep the existing `(morder, norder, indC, C, mr...)` method as a one-line wrapper so `test/planning/test_plan_contract.jl` stands. Demotion call sites use `run_n`/`run_m`. `_unbroken_fraction` and `_leading_unit_run` sit together under one comment stating the run-structure fact (Section 0, `affine_ramp` row). Add test: over random label/stride fixtures, `(_leading_unit_run(morder, indC, C) == Qm) == (affine_ramp(mgroup)[1] && affine_ramp(mgroup)[2][2] == 1)` -- ties the two primitives without merging them (verify the C map is map index 2 of `mgroup`, `:1034`).
- Do NOT rebuild on `AxisGroup`; do NOT change decision semantics.

### Unresolved (non-blocking) questions
- U1: whether the Slurm evidence-gate run has results (affects STATUS.md wording only).
- U2: if G2's envelope gap for Option B exceeds 15% at some K, the user may wish to reopen Option B -- record and ask at close, do not build.

## 3. Task graph

```
T0 (coord) --> T1 (scout) ----------------------------.
T0 --> T2a (item-1 evidence) --> G1 --> [T3a] ---------+--> T5 (review) --> T6 (close)
T0 --> T2b (item-2 evidence) --> G2 --> [T3b] --> G3 --> [T3c] --'
```
Measurements never overlap on the machine (coordinator holds a "benchmark lock"): T2a and T2b run sequentially; T1 may overlap with either. One implementation worker at a time.

**T0 -- baseline (coordinator).** `git status`/`git log -1` (expect `f7fa490`, clean + untracked brief); `uptime`; `julia --version`; `julia --project=. -e 'using Pkg; Pkg.test()'` -> record count; Runic check command (use the one prior milestones used if recorded in decisions.md, else a scratch env with Runic: `julia --project=<scratch> -e 'using Runic; exit(Runic.main(["--check","--diff","src","test","benchmark"]))'`); check `julia +lts` availability (optional AC1 extension). Create `.claude/orchestration/tensorcontract-rs-lessons.md` status table (append; do not rewrite the brief). Completion: numbers recorded.

**T1 -- residual fact check (orch-scout, haiku/default; read-only).** Confirm: tensorcontract-rs HEAD (`git -C /mnt/home/ldevos/Projects/tensorcontract-rs/main log -1`); `kernel_shapes(Float64/Float32)` and `_default_kernel(T, Qm, Qn)` on this host for the sweep fixtures (Qm = a*36, Qn = 216) -- report the default and the F2-demoted kernel for a in {8,40} (F64) and {16,80} (F32) by calling `plan_contract` on the fixture and printing `plan.kernel` with and without a named kernel; `_build_pair_group` dimension order == label order (for the item-3 test); presence of results for the `:mps/:ctmrg/:trg` run (grep `docs/decisions.md`, list `benchmark/results/`); confirm `pack_a!`/`pack_b!` forwarding methods from `PlanarKernel`/`OneMKernel` to their descriptor (so T2a benchmarks through `plan.kernel`). Deliverable: short report; any discrepancy with Section 0 flagged. Rationale: bounded, factual, command-requiring.

**T2a -- item-1 gate evidence (orch-builder, sonnet/medium).** Deps: T0. Edit scope: new `benchmark/bench_pack_loops.jl` only (no `src/`, no `test/`). Inputs: `src/packing/pack.jl`, `benchmark/harness.jl` (`median_time_s`, canaries, `print_env_header`), `benchmark/profile_to_suite.jl`. Work: (i) `julia -t 1 --project=benchmark benchmark/profile_to_suite.jl shallowK_256x24x256_c32 smallN_256x256x12_c32 smallM_12x256x256_c32 smallMN_16x256x16_c32 plain_64_c32 shallowK_256x24x256_c64 smallN_256x256x12_c64 smallM_12x256x256_c64 smallMN_16x256x16_c64 plain_64_c64 --tag tcrs-pre` -> report packing share per case as a fraction of NAMED buckets (exclude "other"); (ii) microbenchmark, 21 reps, adjacent: ns per element AND ns per real emitted for `pack_a!`/`pack_b!` through `plan.kernel` of a default `ComplexF64`/`ComplexF32` 256^3 plan (planar A/B), an explicitly named `OneMKernel` A (1e), and the real fallback loop (`Float64`/`Float32`, A with stride-2 rows so `_pack_a_contiguous_eligible` is false, and B), full sliver and one tail sliver each, `kc = 256`; (iii) `Base.code_llvm` of `_pack_panel_complex!` at the planar-A types: state whether the inner loop is a per-element branch or a select. Deliverable: CSV + summary under `benchmark/results/<host>-<date>/`, and the P1/P2/P3 numbers in the handback. Completion: all three inputs reported with canary spread and `uptime`.

**G1 (coordinator).** PASS -> T3a (1-A) iff P2 shows the complex loop >= 1.3x slower per real emitted than the real fallback loop on at least the planar A full-sliver or B case, OR P1 max packing share >= 10%. Else 1-C (record in T6). No replanning needed either way.

**T2b -- item-2 gate evidence (orch-builder, sonnet/medium).** Deps: T0, T1 (kernel identities). Edit scope: new `benchmark/probes/probe_f2_kdepth.jl` only. Fixture family: `C[a,b,c,i,j,k] = A[i,j,m,a] * B[m,k,b,c]`, `i=j=k=b=c=6`, `a in {8, 40}` for Float64 and `{16, 80}` for Float32, `m in {8,16,32,64,128,256,512}`; three arms per point via `plan_contract`: `auto` (no kernel named), `forced-default` (`_default_kernel(T, Qm, Qn)` named -- naming bypasses F2), `forced-demoted` (the menu shape F2 would pick, named); assert `auto`'s kernel `===` one of the two and report which; 15+ reps median, arms adjacent per point, canary + `uptime` recorded; verify all three arms agree numerically (`isapprox`) once per point. Report `r = t_demoted / t_default` per point and `t_auto`. Deliverable: CSV + summary + numbers in handback. Completion: 2 dtypes x 2 a x 7 m x 3 arms, plus the `a=8, m=512` F64 point reproduced twice.

**G2 (coordinator).** Read `r(T, a, m)`.
- Q1 (reproduces?): F64 a=8 m=512: `r >= 1.10` -> proceed; `r < 1.05` -> item 2 closes as "not reproducible on f7fa490" (record; T6 also amends the F2 S1 note); `1.05 <= r < 1.10` -> re-run once more; if still ambiguous treat as not reproducible.
- Q2 (crossover): for the fully-broken fixtures (F64 a=8, F32 a=16): PASS iff `r < 0.95` at the smallest m and `r > 1.05` at the largest m. Threshold `kmax(T)` = largest m with `r <= 0.95` at that m and every smaller m; if the last winning and first losing m differ by exactly one step, coordinator may instead take the geometric midpoint rounded to a power of two.
- Q3 (sanity): `kmax(Float32) >= 16` required (ccsd_t_1 dim16 F32 must still demote). Violation -> REPLAN.
- Q4 (broken-enough guard): F64 a=40 / F32 a=80 (default fraction 0.8): if `r > 1.0` at every m -> adopt `F2_BROKEN_ENOUGH = 0.75`; if demotion wins at small m -> set 1.0 (inert) and record. Anything else (e.g. wins only at F32) -> adopt 0.75 only for the dtype(s) where it holds via a per-dtype constant? NO -- keep one constant; choose inert and record.
- Demotion never wins anywhere (r > 1 at m=8 for a=8) -> contradicts F2's measured 2.17x -> REPLAN.
- Also compute envelope gap for the record: max over points of `min(t_default, t_demoted) / t_rule(kmax)`.

**T3a -- item 1 implementation + verification + measurement (orch-builder, sonnet/medium; escalate to orch-specialist opus/high if AC4a fails after one repair attempt).** Deps: G1 PASS. Edit scope: `src/packing/pack.jl`; `test/packing/test_pack_complex.jl`, `test/packing/test_pack_real.jl` (additive only); `docs/decisions.md` (create the milestone section + item-1 subsection); `benchmark/bench_pack_loops.jl` re-run only. Forbidden: `src/packing/format.jl`, `src/packing/format.jl`, kernels, driver. Steps: (1) BEFORE editing, dump `code_llvm` (with `debuginfo=:none`) of `_pack_a!` and `_pack_b!` for `(PackedPanel{T}, SourceTile{Memory{T},AffineAxis,AffineAxis}, KernelDescriptor{16,6,Float64}, typeof(identity), Val{false})` and the Float32 `{32,6}` twin, both `Val(true)`/`Val(false)`, to the scratchpad; (2) implement 1-A per Section 2; (3) re-dump and diff (modulo SSA names) -> AC4a; if not identical, one repair attempt, else fall back to 1-B and say so; (4) tests: add a testset asserting real fallback `pack_a!`/`pack_b!` on full and tail slivers equal the direct-indexing oracles at both dtypes (may already be covered at `test_packing.jl:853+` -- extend, do not duplicate); complex: ensure full-sliver AND tail-sliver AND `kc == 1` cases for Planar A, Planar B, 1e A with `conj` are asserted (extend `:313`/`:146` if any cell is missing); (5) `Pkg.test()`, Runic, forced-ISA x2; (6) measurement: commit; `git archive f7fa490 | tar -x -C <scratch>/base`, copy `Manifest.toml` into it; ABBA guard A-B-B-A (>= 2 runs per tree, `QS_GUARD_LABEL=base` for the archive), compute per-shape median and geomean; `bench_complex_efficiency.jl` on both trees adjacent; re-run `bench_pack_loops.jl`; `profile_to_suite.jl` same 10 cases `--tag tcrs-post`; (7) decisions.md subsection with numbers + artefact paths; commit. Completion: AC4 all sub-items, or 1-B with AC4b-d and an explicit statement of why 1-A was abandoned.

**T3b -- item 2-A implementation + verification + measurement (orch-builder, sonnet/medium).** Deps: G2 PASS (with `kmax(T)`, `F2_BROKEN_ENOUGH` decided). Edit scope: `src/planning/kernel_selection.jl` ONLY at `_demote_for_run` + its comment (`:255-282`), the two call sites (`:1065-1067`, `:1073-1075`), new `_unbroken_fraction`, new constants near `:402-403`; `test/planning/test_plan_contract.jl` (F2 testset `:1205-1260` + new); `benchmark/probes/probe_f2_kdepth.jl` re-run; `docs/decisions.md`. Forbidden: `_prefer_swap`, `_leading_unit_run`, `_default_kernel`, menus, `_plan_contract`, the nest, kernels. Tests (rules from the F2 post-review, `decisions.md:5210-5242`): expectations computed from `_default_kernel`/`kernel_shapes(T)`/the constants; assert (a) `_unbroken_fraction == 1.0 <=> Qm == run || run % mr == 0` over a random grid; (b) regression fixture at `m = 2*kmax(Float64)` does NOT demote (`plan.kernel === default`), at `m = 8` DOES (if a menu shape divides run); (c) existing ccsd_t_1 dim16 fixture behaviour preserved on every ISA (the testset already computes this); (d) a fixture with default fraction 0.8 does not demote iff `F2_BROKEN_ENOUGH < 0.8`. Verification: `Pkg.test()`, Runic, forced-ISA x2, `probe_f2_kdepth.jl` re-run (AC5a), `bench_ccsd_t_store.jl --dims 8,16 --dtypes Float32,Float64` pre (on base archive) and post (AC5c), ABBA guard (AC5e). decisions.md subsection: gate table, constants' provenance, envelope gap (Option B upper bound), what tensorcontract-rs's rule is and how it differs. Commit.

**G3 (coordinator).** PASS iff T3b merged. Else item 3 = record only ("one derivation, three call sites; hoisting one call is not worth an ABBA").

**T3c -- item 3 (orch-builder, sonnet/medium; may be the same session as T3b, separate commit).** Deps: G3. Edit scope: `src/planning/labels.jl` (`_prefer_swap`, `plan_contract` `:1031-1075` region, comments at `:208-253`); `test/planning/test_plan_contract.jl` additive; `docs/decisions.md`. Steps: per Section 2; measure `plan_contract` time + `@allocated` on the four per-call-floor fixtures before/after (21 reps, adjacent) -> AC6; `Pkg.test()`, Runic, forced-ISA. Record that `affine_ramp` was evaluated and rejected as the primitive (with the relation test as evidence).

**T5 -- independent review (orch-reviewer, opus/high).** Deps: all shipped T3x. Read-only. Scope: (1) item 1: IR-identity claim substantiated by the dumps; conj-before-split and literal-zero padding preserved for Planar/1e; no new `Union`/allocation (`test_packing_complex.jl:448` + a scattered complex fixture); (2) item 2: every test expectation ISA-independent; constants' provenance matches the CSV on disk; `_unbroken_fraction` equivalence proof; call-site symmetry (swapped branch uses `run_n`, `Qn`, `Qk`); no change to `_prefer_swap` semantics; (3) item 3: wrapper preserves every existing assertion; C-map index correct in the relation test; (4) all decisions.md numbers re-derived from artefacts, ratio directions correct (the C0 CSV `ratio_note` lesson). Deliverable: findings ranked blocking/should-fix/nit with file:line. Fixes go back to the implementing profile; a blocking SEMANTIC finding (wrong conj/padding/decision change) -> REPLAN.

**T6 -- close-out (orch-builder, sonnet/medium).** Deps: T5 resolved. Edit scope: `STATUS.md`, `docs/decisions.md` (complete the milestone section incl. any 1-C/2-closed/3-record subsections and the F2 S1 amendment if Q1 failed), `.claude/orchestration/tensorcontract-rs-lessons.md` (status table + "corrections to this brief" pointer to Section 0's findings). AC7 + AC3. Every number checked against an artefact path.

## 4. Integration order, conflicts, review points
Order: T3a -> T3b -> T3c, each its own commit on `compare`, each measured against a `git archive` of the commit before it (attribution). `docs/decisions.md` touched by T3a/T3b/T3c/T6 -- append-only, sequential, no conflict. `src/execution/execute.jl` touched by T3b then T3c -- sequential. `src/packing/pack.jl` only T3a. Benchmarks: one at a time on the host; record `uptime` and canary spread in every artefact; >= 15 reps, 21 for the guard; treat < 5-6% as noise (`STATUS.md:452-457`). Independent review: single T5 pass after all items (plus the coordinator's own artefact re-read at each gate). Merge to `main` only after T6.

## 5. Risks
- R1: touching the tuned real packing loop -- mitigated by IR identity (AC4a) and 1-B fallback.
- R2: sweep noise inventing a crossover -- mitigated by adjacent arms, 15+ reps, the a=8 m=512 double reproduction, and Q3's ccsd_t consistency check.
- R3: threshold overfit to one synthetic family -- accepted and recorded; AC5c pins the real measured wins; the constant is documented as ccqlin038-calibrated like `default_blocking`.
- R4: ISA-literal tests (F2 history) -- explicit derivation rules + forced-ISA runs + reviewer item.
- R5: 50% "other" profiler artefact inflating/deflating shares -- report shares among named buckets.
- R6: machine not quiet -- rerun; after 3 failed attempts (canary > 15%) ask the user (Slurm submission is theirs).

## 6. Decision boundaries and replanning triggers
Coordinator MAY decide without me: gate outcomes by the numeric rules above; 1-A vs 1-B fallback; `kmax(T)` per the stated rule (incl. the power-of-two midpoint option); `F2_BROKEN_ENOUGH in {0.75, 1.0}`; wrapper vs test rewrite for `_prefer_swap` (prefer wrapper); rep counts above minimums; re-running noisy measurements; running T3c in the T3b session; STATUS.md wording; escalating T3a to orch-specialist.
Coordinator MAY NOT: alter the sweep's fixture family/arms in a way that changes gate semantics; add calibrated constants beyond `kmax(F64)`, `kmax(F32)`, `F2_BROKEN_ENOUGH`; build Option B; touch the frozen list or `_prefer_swap` semantics; skip ABBA for any `src/` change; accept a literal-`mr` test; write decisions.md numbers not backed by an artefact.
REPLAN triggers: G2 Q3 violated (`kmax(F32) < 16`); demotion never wins anywhere; no crossover in either dtype while Q1 reproduces; T1 finds a Section-0 fact wrong in a contract-changing way (e.g. complex tests do pin the loop, or `_build_pair_group` reorders labels); T3a fails AC4a AND 1-B also shows a one-sided ABBA shift; reviewer blocking semantic finding; user wants Option B (U2).
Minimal user decision needed now: none. At close: U1/U2 as information, not blockers.
