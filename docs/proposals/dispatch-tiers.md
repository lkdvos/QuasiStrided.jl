# Proposal: Octavian-style dispatch tiers (`dontpack` / `maybeinline`)

**Status: design review and sign-off gate. Nothing in `src/` changes on the
authority of this document.** It answers one question STATUS.md has carried as
its "highest-value next item" since Phase H: should this engine grow
Octavian.jl-style dispatch tiers that skip packing for shapes where packing
does not pay off? It was written 2026-09-21 against `main` at `ebfbeb3`, with
the `ccsd-t-stall` branch at `1da0ef7` read alongside for Finding 1. Every
number below is either re-derived here from a file in the tree or cited to a
`docs/decisions.md` section by name; where a figure could not be checked it
says so.

Frozen and untouched by anything proposed here, per the standing list in
`.claude/orchestration/profiling-followups.md`: `QuasiStridedBackend`'s
hard-reject invariant (`src/integrations/tensoroperations.jl`, `:304-357`),
`src/hardware/target.jl`'s register-shape derivation, and the beta-applied-once /
conjugation numerical semantics (`src/execution/execute.jl`, `:821-836`,
`:651-688`). Section 6 proposes nothing that modifies any of them, and says so
at each point where a reader might expect it to.

## 1. Recommendation, up front

**Do not build the Octavian-style tier system now -- neither `dontpack` nor
`maybeinline` -- and do not build a narrower "skip packing" variant either
until one cheap evidence gate has been run.** The evidence has moved since
STATUS.md's framing, and it moved against this item:

1. The one real, large, reproducible loss class this project has ever
   measured against `StridedBLAS` (`ccsd_t_*_dim16`, 6.6-14.2x slower) had
   the *highest* packing reuse in the entire 118-case sweep and a directly
   measured packing cost of 0.02-0.03% of its runtime. Its causes were a
   label-order/store-path defect and a non-inlined accumulator, both now
   fixed (Section 3.1). Packing amortization had nothing to do with it.
2. The packing *code* was the problem where packing was large, and that has
   already been fixed on `main` (Section 3.2): packing throughput roughly
   doubled, and on the genuine multi-index cases profiled the packing share
   is now 1-3% / 10.1% / 11.2% / 7.0%. The upper bound on what any
   "don't pack" tier could still buy on those cases is that share.
3. Translating Octavian's own triggers onto this project's measured cases
   (Section 4, Table 4.1): among the 17 cases with a recoverable
   QuasiStrided-vs-`StridedBLAS` ratio, the trigger fires on 12. Nine of
   those are cases QuasiStrided already *wins* by 2-4.7x with packing at
   1-10% of runtime; the other three are the residual *losses*, and in every
   one of them packing is demonstrably not the bottleneck (a `K=1` outer
   product at a fixed ~14 us per-call floor; two `K=8` store-bound cases).
   There is no case on record where the trigger fires, QuasiStrided loses,
   and packing is what it loses on.
4. What the tiers *would* fix is the plain-matmul benchmark gap on
   small-`N` shapes such as `smallN_256x256x12` (34.6 GFLOP/s post-fix vs
   Octavian's 87.2). Even a perfect no-pack path caps that shape at roughly
   51 GFLOP/s by the post-fix profile's own arithmetic (Section 4.3); the
   other half of the remaining gap is per-call overhead, which is the same
   lever the residual losses point at.

What should happen instead, in order (Section 5): (a) run the upstream
`:mps`/`:ctmrg`/`:trg` categories, which are the tensor-network workloads
this package claims as its target and which have never been measured here --
the tooling exists and the run is cheap; (b) attack the per-call floor
(`plan_contract` allocation and per-tile validation), which is the lever the
residual losses actually indict; (c) only if (a) shows the skewed-small-`N`
pattern occurring *and losing* in the target workload, build the narrow,
real-dtype-only "A-direct" tier sketched in Section 6, behind the kernel
microbenchmark gate in Section 6.2 that Phase H's own measurements say is
necessary.

That is a recommendation *against* the item, and it is this document's
primary result. The conditional design in Section 6 exists so that, if the
user overrides the recommendation or the gate passes, the work does not start
from zero.

## 2. What the tiers are, exactly

Read from Octavian.jl 0.3.29's source in the local depot
(`~/.julia/packages/Octavian/4f4xi/src/matmul.jl`), so this section describes
what Octavian does rather than what STATUS.md remembers it doing. Its
single-threaded dispatch (`_matmul_serial!`, lines 320-360) is:

```
if maybeinline(M, N, T, is_column_major(A))        # compile-time only
    inlineloopmul!(...)                             # fully unrolled, no packing
elseif (nᵣ ≥ N) || dontpack(pA, M, K, Mc, Kc, T)
    loopmul!(...)                                   # @turbo triple loop, no packing
else
    matmul_st_pack_dispatcher!(...)                 # pack A only, or pack A and B
end
```

- **`maybeinline`** (lines 210-229) is defined only for `StaticInt` `M` and
  `N` -- sizes known at compile time -- and returns `false` for anything
  else (`maybeinline(::Any, ::Any, ::Any, ::Any) = false`, line 210). The
  static test is `sizeof(T)*M*N < 176*mᵣ*nᵣ` for a column-major `A`. **This
  tier has no analog in QuasiStrided at all**: `plan_contract` receives
  runtime extents from `StridedView`s (`src/planning/plan.jl`), and the
  TensorOperations adapter never sees a static size. The runtime version of
  the idea -- "a problem so small the fixed per-call cost dominates" -- is a
  different lever, discussed in Section 5.2.
- **`dontpack`** (lines 130-150) skips packing `A` when `A` is
  contiguous along `M` **and either** `M ≤ (MᵣW_mul_factor + 5)·W` -- 9
  vector widths on AVX-512 (`MᵣW_mul_factor(::True) = 4`,
  `global_constants.jl:17`), i.e. `M ≤ 72` for `Float64`, `M ≤ 144` for
  `Float32` -- **or** `A`'s column byte-stride is a multiple of the vector
  register size, `M·K ≤ Mc·Kc` (the whole of `A` fits one cache block), and
  `A`'s base pointer is 64-byte aligned. Note that neither arm mentions `N`:
  the "reuse ratio" framing in STATUS.md is the separate `nᵣ ≥ N` test on
  the same line, which fires when the entire `N` extent fits one register
  tile column count.
- **`loopmul!`** (`macrokernels.jl:124-133`) is not "the microkernel over
  unpacked data". It is a LoopVectorization `@turbo` triple loop over the
  whole `M×N×K` problem, for which LoopVectorization chooses its own loop
  order, unrolling and register tiling. This project rejected
  LoopVectorization as a dependency (decisions.md, "Panel addressing
  milestone: Phase H", "Dependency note"), so this tier cannot be
  transplanted; a QuasiStrided equivalent would have to be the existing
  `SIMDKernel` reading a strided, unpacked panel -- Section 6.2 examines
  whether that is even fast.
- **Pack A only vs. pack A and B** (`matmul_st_pack_dispatcher!`, lines
  362-396): `B` is left unpacked when `Kc·Nc ≥ K·N` (for a `K`-contiguous
  `B`) or when its first byte stride is at most 1600.
- Octavian's `Mc`/`Kc`/`Nc` come from a cache-size model with per-arch
  fitted constants (`block_sizes.jl:4-33`, `global_constants.jl:23-60`).
  This project's `default_blocking` deliberately uses measured constants and
  not a cache model (`src/planning/blocking.jl`, decisions.md "Why cache
  geometry is still not used"), so `dontpack`'s `M·K ≤ Mc·Kc` arm would be
  read here against `mc·kc = 128·256 = 32768` (`Float64`) and `96·768 =
  73728` (`Float32`) on AVX-512 (`src/planning/blocking.jl`).

Octavian's block sizes were **not** evaluated on this host for this document
(no local environment has Octavian as a dependency; the Phase H comparison
ran in a since-removed scratch environment). Where Section 4 says "the
Octavian trigger fires", it means the translation above with this project's
own `mc·kc`, not Octavian's.

## 3. Evidence

### 3.1 Finding 1: the `ccsd_t_*_dim16` loss class was never a packing problem

The four six-index `ccsd_t_*` contractions at `dim=16` were the one
substantive loss in the upstream `TensorOperationsBenchmarks` comparison:
QS/BLAS time ratios of {10.11, 13.98, 9.68, 14.24} (`Float64`) and {11.51,
10.75, 6.60, 9.89} (`Float32`), also 2.0-4.4x slower than plain
`StridedNative` (decisions.md, "T3 measurement results"). The T5 profiling
triage of `ccsd_t_1_dim16` put the store path at 75.30% / 88.36%
(`Float64`/`Float32`) of QuasiStrided's own time, the FMA loop at 0.40% /
0.24%, and **packing at 0.02% / 0.03%** (decisions.md, "T5 profiling triage
results"). Every subsequent step confirmed that attribution:

- The label-order milestone (`_order_free_labels`, `_prefer_swap`,
  `src/planning/labels.jl`, `:864-876`) turned the class into a **1.96x-6.91x
  win** over `StridedBLAS` at `dim=16` across all eight case/dtype cells
  (decisions.md, "Label-order milestone", table under "The four regression
  cases"). At `dim=8`, two `Float32` cells (`ccsd_t_2`, `ccsd_t_4`) remain
  0.90x / 0.81x, i.e. QuasiStrided 1.11x / 1.23x *slower*, "near
  `StridedBLAS`'s own floor".
- The profiling pass then found the remaining `ccsd_t_1_dim16` behaviour was
  "kernel-stalled" (`Float64`) and store-dominated at ~75% (`Float32`)
  (decisions.md, "Profiling pass", T3 table). Track A's mechanism analysis
  (`ccsd-t-stall` branch, decisions.md there, "Kernel-stalled/store-dominated
  fix") confirmed two causes with LLVM/native dumps and structural sliver
  dumps:
  - **F1**: `Base.accumulate` for `SIMDKernel` (`src/microkernels/simd.jl`)
    was not inlined into `execute_tile!`, so the 768-byte accumulator went
    through a real `memset` and a stack round-trip per micro-tile. One
    `@inline`: **+8.9%** on `ccsd_t_1` `dim=16` `Float64`, +3.4% (no
    regression) on 512^3 `kc=256`.
  - **F2**: the vectorized store `_store_tile_vector!` needs every register
    sliver unit-stride in `C`, which holds iff `Qm == run || run % mr == 0`
    -- not `mr ≤ run` (refuted by counterexample sweep,
    `benchmark/probes/probe_ccsd_t_stall_f2rule.jl`). The `Float32` default
    `(32,6,16)` against a run of 16 sent every store down the scattered
    path. `_demote_for_run` (`ccsd-t-stall`, `src/planning/kernel_selection.jl`, called
    at `:904-906` and `:912-914` after the swap decision) picks the largest
    menu shape whose `mr` divides the run: **2.17x** (`dim=16` `Float32`),
    **1.92x** (`dim=8` `Float32`), store share 75% to 21.73%.
  - The review of F2 recorded a genuine **known limitation, not fixed**:
    F2 has no `K`-aware cost model, and on a shape where it fires with a
    large contracted extent (`m = 512`) it is **18-25% slower** than not
    demoting, because every demotion in the real menus also halves the SIMD
    lane width (`(16,6,8) -> (8,6,4)`). This matters for Section 6.6: any
    new shape heuristic added on top of F2 stacks a second uncalibrated
    heuristic on the same planning path.

Two bookkeeping facts for the coordinator: `_demote_for_run` is **not on
`main`** at `ebfbeb3` (it is on `ccsd-t-stall` at `1da0ef7`), and that branch
was cut from `f318eb9`, so it does **not** contain the packing fix `8dd01dd`
(verified with `git merge-base --is-ancestor`). The two touch disjoint files
(`src/planning/kernel_selection.jl` + `src/microkernels/simd.jl` vs. `src/packing/pack.jl`), so a rebase is
expected to be clean, but the combined tree has not been measured.

### 3.2 Finding 2: the packing code was slow, and that is already fixed

Merged to `main` as `8dd01dd` (decisions.md, "Packing speed: vectorizing the
real packing loop as it exists today"). The mechanism was a per-element
conditional load (`v = i < valid ? load(i,p) : zero(T)`) that LLVM cannot
if-convert without a masked load, so `_pack_panel!` never vectorized:
0.74-0.94 ns/element for `A` at `Float64` against 0.26-0.28 for `copyto!` of
the same bytes. The fix, all in `src/packing/pack.jl`: a full/tail split of
`_pack_panel!` (`:102-123`) and a gated contiguous fast path
`_pack_a_contiguous!` (`:162-175`, gate `_pack_a_contiguous_eligible`
`:141-146`: `PackedPanel` destination, `DenseVector` source, `identity`/real
`conj` transform, `m == MR`, unit-stride rows). Measured:

| what | before -> after |
|---|---|
| `A`, `Float64`, `MR=16`, 256x256 | 48.7 -> 18.5 us (2.6x; `copyto!` 17.7 us) |
| `A`, `Float32`, `MR=32`, 256x256 | 36.6 -> 5.3 us (7.0x) |
| `B`, `Float64`, `NR=6`, 256x12 / 225x225 | 1.5x / 1.4x |
| `smallN_256x256x12` | 16.0 -> **34.6** GFLOP/s; packing share 62.1% -> **32.8%** |
| `scattered_64` | 39.3 -> 51.6; packing 25.7% -> **7.0%** |
| `ao2mo_2_dim16` | 11.7 -> 14.5; packing 34.2% -> **10.1%** |
| `dim15_2_2_2` | 46.0 -> 49.4; packing 19.5% -> **11.2%** |

Two things in that record bear directly on this proposal and must both be
stated. First, the fix's own author wrote that after it "~90% of `smallN`'s
remaining packing samples sit inside `SIMD.vload`/`vstore` -- i.e. plain
data movement at ~35 GB/s against `copyto!`'s ~60 GB/s -- so the loop is no
longer the problem; what is left is memory traffic that only 'don't pack A
for small N' removes." That is a correct statement about *that shape*, and it
is the strongest argument in the record for the A-direct tier of Section 6.
Second, the record's "not done" list names three residual costs that are not
packing at all: per-sliver validation (`checked_tile_storage_bounds` +
`_check_pack_a/b`) at 7.4% of `ao2mo_2_dim16`, `driver_loop` bookkeeping at
28.7% of the same case, and the complex packing loop (untouched, still has
the conditional load).

**Staleness note on the extended profiling grid.** The T4 full-grid profiles
(decisions.md, "T4: extended the grid...", the 35-row table) were taken on
the `profile-grid` worktree at `275b6f5`, which does **not** contain
`8dd01dd` (`profile-grid/src/packing.jl` has no `_pack_a_contiguous!`;
`git merge-base --is-ancestor` confirms). Every packing share in that table
is therefore pre-fix. The finding "packing dominance generalizes across all
small shapes at all four dtypes" is a true statement about the old packing
loop; the post-fix shares on the real dtypes should be expected to be
roughly half on `A`-heavy shapes and ~30% lower on `B`-heavy ones, and are
unmeasured. Any decision that leans on the T4 magnitudes should first re-run
those rows on `main`.

### 3.3 Finding 3: the workload-representativeness table, re-derived here

`benchmark/results/dontpack-C0-feature-table.csv` (118 rows: every upstream
`:pairwise`/`:tccg` case at both real dtypes) tabulates, per case, `Qm`,
`Qn`, `Qk`, the resolved `mr`/`nr`, the total M- and N-sliver counts, and
`kc_eff`, joined to every QuasiStrided-vs-`StridedBLAS` ratio recoverable
from `docs/decisions.md` (the raw sweep CSVs are gitignored and gone). Read
directly from that file for this document:

- **17 rows carry a ratio** (the C0 write-up in
  `.claude/orchestration/profiling-followups.md` says 18; 17 is what is on
  disk). This is a real sample-size limitation and is why Section 4's
  conclusions are stated as specific counter-evidence, not a population
  claim.
- **The loss class had the highest reuse in the sweep.** `ccsd_t_*_dim16`:
  `Qm = Qn = 4096`, `Qk = 16`, 256 (`Float64`) / 128 (`Float32`) M-slivers
  and 683 N-slivers -- an M x N sliver product of 174,848, above
  `dim63_2_1_2`'s 164,838 and far above every `intensli_*` row. Each packed
  `A` sliver is consumed by 683 micro-tiles. Packing cost 0.02-0.03% of
  runtime (Section 3.1). This is the opposite of the shape a reuse-ratio
  trigger targets.
- **The genuinely low-reuse cases are the best wins.** Every row with a
  single M-sliver (`m_slivers = 1`: `Qm ≤ mr`) that has a ratio is a win:
  `ccsd_3_dim16` 0.372, `ao2mo_2_dim16` 0.315, `ao2mo_3_dim16` 0.323 --
  QuasiStrided 2.7-3.2x *faster*. The T5 triage explains why: `StridedBLAS`
  spends 63-76% of its own time on `permute/copy` for these shapes; the win
  is "avoided the temp/permute", not a faster GEMM.
- **The residual losses correlate with tiny total work, not with reuse.**
  `dim63_1_0_1` (`Float32`, `Qk = 1`, a 63x63 outer product): 14.3 us vs
  0.50 us, **28.5x** slower, inside T3's observed "flat per-call floor of
  roughly 5-25 us" (decisions.md, "T3 measurement results"). `ccsd_t_2/4_dim8`
  (`Float32`): `Qm = Qn = 512`, `Qk = 8`, 1.11x / 1.23x slower -- 16 flops
  per output element, so the store of 262,144 elements per call dominates by
  construction. **Data-quality note**: the CSV's `ratio_note` column labels
  these two rows "marginal WIN" with ratios 0.9 / 0.81; decisions.md's
  label-order table defines those figures as `StridedBLAS` time over
  QuasiStrided time, so they are marginal *losses* (QS/BLAS 1.11 / 1.23).
  The coordinator's summary has the direction right; the CSV note does not
  and should be corrected.
- **Of the 38 profiled cases, four are not plain-matmul-shaped.** Reading
  `benchmark/profile_to_suite.jl`'s `CASES`/`DIRECT_CASES`: `ccsd_t_1_dim16`
  (and its `_f32` twin), `ao2mo_2_dim16`, `dim15_2_2_2` (rank-4 operands
  with 2+2+2 labels; its composites fold to contiguous), and `scattered_64`
  (plain 64^3 extents but permuted `A`, negative-stride `B`, sliced `C` --
  the fixture this engine exists for). Everything else is a square or
  skewed GEMM with a suggestive name. Post-fix packing shares on those four:
  **1-3% / 10.1% / 11.2% / 7.0%**. The 60-80%+ packing shares that motivated
  "packing is the whole gap" all come from plain-matmul rows.

### 3.4 What the evidence does not cover

- **The target workload itself has not been measured.** The upstream suite's
  `:mps`, `:ctmrg` and `:trg` categories -- the tensor-network workloads --
  have been listed as "still out of scope" in every milestone since the
  upstream comparison was set up (decisions.md, "Follow-ups, explicitly out
  of scope"; STATUS.md, three separate sections). `bench_to_suite.jl
  --categories` already supports them. Whether tensor networks produce
  `N ≤ 12` GEMM-shaped contractions in volume, and whether QuasiStrided
  loses on them, is exactly the question this proposal turns on, and it is
  answerable for the cost of one quiet-machine run.
- Only 17 of 118 cases have a ratio; the rest of the population's
  win/loss status is a geomean (QS/BLAS 1.77 on `:tccg`, 4.29-6.84 on
  `:pairwise`, decisions.md "T3 measurement results") whose per-case
  composition is not recoverable.
- Single machine (`ccqlin038`, Cascade Lake), and the profiling grid ran on
  Julia 1.13.0 while every earlier reference ran on 1.12.6 (decisions.md,
  "Profiling pass", verdict-rule paragraph). No cross-machine claim is made
  anywhere in this document either.

## 4. Value analysis: applying the triggers to what was measured

### 4.1 Where the Octavian-style trigger fires, among cases with a known outcome

Trigger, translated per Section 2: fire if `Qm ≤ 9W` (72 `Float64` / 144
`Float32`) **or** `Qm·Qk ≤ mc·kc` (32768 / 73728). Alignment and
contiguity arms are assumed satisfiable (they are for the benchmark's fresh
`Array`s). `Float64` unless marked.

| case | `Qm` | `Qn` | `Qk` | M-slivers x N-slivers | fires? | QS/BLAS | packing share (post-fix where measured) | verdict |
|---|---|---|---|---|---|---|---|---|
| `dim63_1_0_1` f32 | 63 | 63 | 1 | 2 x 11 | yes (`Qm ≤ 144`) | **28.5** loss | not profiled; 63+63 elements packed per call | per-call floor, not packing |
| `ccsd_t_2_dim8` f32 | 512 | 512 | 8 | 16 x 86 | yes (4096 ≤ 73728) | **1.11** loss | not profiled; 4096+4096 elements packed, 262k stored | store-bound by construction |
| `ccsd_t_4_dim8` f32 | 512 | 512 | 8 | 16 x 86 | yes | **1.23** loss | as above | as above |
| `ccsd_3_dim16` | 16 | 256 | 16 | 1 x 43 | yes (`Qm ≤ 72`) | 0.372 win | not profiled (sibling `ao2mo_2`: 10.1%) | already 2.7x faster |
| `ccsd_6_dim16` | 256 | 256 | 16 | 16 x 43 | yes (4096) | 0.509 win | not profiled | already 2.0x faster |
| `ccsd_8_dim16` | 256 | 256 | 256 | 16 x 43 | **no** (65536) | 0.508 win | not profiled | -- |
| `ccsd_t_1..4_dim16` | 4096 | 4096 | 16 | 256 x 683 | **no** (65536 > 32768) | 0.14-0.25 win | **1-3%** | -- |
| `ccsd_t_1..4_dim16` f32 | 4096 | 4096 | 16 | 128 x 683 | yes (65536 ≤ 73728) | 0.21-0.51 win | **~1%** | fires on a case with 1% packing |
| `ao2mo_2_dim16` | 16 | 4096 | 16 | 1 x 683 | yes (`Qm ≤ 72`) | 0.315 win | **10.1%** | upside ≤ 10% on a 3.2x win |
| `ao2mo_3_dim16` | 16 | 4096 | 16 | 1 x 683 | yes | 0.323 win | as `ao2mo_2` | as above |
| `intensli_1_dim16` | 256 | 16 | 16 | 16 x 3 | yes (4096) | 0.45 win | not profiled | already 2.2x faster |

Fires on 12 of 17. **Zero rows where the trigger fires, QuasiStrided loses,
and packing is the measured or structurally plausible bottleneck.** On the
`Float32` `ccsd_t_dim16` rows it fires on cases whose packing is ~1% of
runtime and whose `A`-side M composite is not even unit-stride in `A`
(`C[a,b,c,i,j,k] = A[i,j,m,a]·B[m,k,b,c]`: `a` has stride 16^3 in `A`), so
a layout-gated implementation would refuse them -- but a reuse-only trigger
in the STATUS.md sense would not.

The alternative "reuse ratio" trigger STATUS.md actually describes --
`cld(Qn, nr) ≤ 2`, i.e. `N ≤ 12` at `nr = 6` -- fires on **none** of the 17
rows with a ratio (the one `intensli` row with a ratio, `intensli_1_dim16`,
has `Qn = 16`, three N-slivers; the `Qn = 8` `dim8` rows that would fire
have no recoverable ratio).

### 4.2 Upper bound on the genuine multi-index cases

If a tier removed packing entirely, at zero cost, wherever it applied, the
achievable speedup on each non-plain profiled case is bounded by
`1/(1 - packing share)`: `ccsd_t_1_dim16` **≤ 1.03x**, `ao2mo_2_dim16`
**≤ 1.11x**, `dim15_2_2_2` **≤ 1.13x**, `scattered_64` **≤ 1.08x**. On `dim15_2_2_2` (`Float64`, the
profiled dtype) the Octavian-style trigger does not fire (`Qm = 225 > 72`,
`Qm·Qk = 50625 > 32768`); on `scattered_64` it would (`64·64 = 4096`), but
its `A` is permuted, so its M-slivers are not unit-stride and any
layout-gated implementation refuses it. These bounds sit inside the ~10%
band this project's own measurement-hygiene note treats as noise on this
machine (STATUS.md, "Measurement hygiene").

### 4.3 What it would buy on the shape that motivated it

`smallN_256x256x12`, `Float64`, post-fix: 34.6 GFLOP/s, packing 32.8%,
microkernel 40.3%. `Qn = 12` is two N-slivers; `Qk = 256` is one K-block;
`mc = 128` gives two M-blocks; so **every element of `A` is packed exactly
once per call** -- this is not repeated packing, it is a single pass over
`A` costing about as much as the arithmetic because `N = 12` gives only 24
flops per `A` element. With packing removed and every other bucket held
fixed, throughput would be `34.6 / (1 - 0.328) ≈ 51.5` GFLOP/s. The
microkernel's in-situ rate is already `34.6 / 0.403 ≈ 86` GFLOP/s -- at the
isolated reference -- so the remaining 1.7x to Octavian's 87.2 is
`driver_loop` + `planning` + `store` (~27% combined), i.e. the per-call
floor again. The tier would close roughly half of the remaining gap on the
one shape it is built for, and Section 6.2 explains why even that half is
not guaranteed.

`smallM_12x256x256` (pre-fix packing 37-45%) is the mirror image: `Qm = 12`
demotes the kernel to `(8,6,4)`, giving two M-slivers, and the cost is
packing `B` (65536 elements, used twice). An `A`-direct tier does nothing
there; it would need a `B`-direct tier, whose kernel-side story is
different (scalar loads, Section 6.2). `smallMN_16x256x16` (pre-fix packing
~42%, planning ~22-27%) packs 8192 elements per call in total; at ~40 us per
call it is a floor case.

### 4.4 Bottom line of the value analysis

The item was proposed when two things were believed: that the engine's gap
was "entirely packing plus per-call cost" (STATUS.md, "Next task"), and that
small/skewed shapes are the tensor-network common case. The first is now
known to have been mostly a packing-*code* defect on plain-matmul shapes,
fixed, plus a store-path defect on the real loss class, fixed. The second is
unmeasured (Section 3.4). What remains attributable to *whether* to pack, on
shapes in the measured population, is at most a ~1.5x on `N ≤ 12`
GEMM-shaped calls and ≤ 1.1x on every genuine multi-index case profiled.

## 5. Recommended course

### 5.1 The evidence gate (do this first; ~one quiet-machine session)

1. `julia --project=benchmark benchmark/bench_to_suite.jl --categories mps,ctmrg,trg`
   (both dtypes, 21 reps, quiet machine, canary spread recorded), then
   `plot_bench_to_suite.jl` and a feature-table pass identical to C0's over
   the new cases (`Qm`, `Qn`, `Qk`, slivers, `kc_eff`, plus
   `_leading_unit_run` of `A`'s M composite in `A`, which is the
   layout-eligibility quantity Section 6.1 needs).
2. Re-run the T4 small-shape rows on `main` (post-`8dd01dd`) so the record's
   packing shares are current: `profile_to_suite.jl smallN_256x256x12
   smallM_12x256x256 smallMN_16x256x16 plain_64 ao2mo_2_dim16 dim15_2_2_2
   --tag post-d1`, both real dtypes.
3. Profile the three worst `:mps`/`:ctmrg`/`:trg` losses (if any) with the
   same tool.

**Gate**: the A-direct tier of Section 6 is worth building only if the new
categories contain cases that (a) lose to `StridedBLAS` by more than the
~10% noise band, (b) profile as packing-bound after the fix (packing share
above, say, 25%), and (c) satisfy Section 6.1's layout eligibility. If the
losses are instead floor-bound, Section 5.2 is the lever; if they are
store-bound, F2's known limitation (Section 3.1) is.

### 5.2 The per-call floor (higher expected value on the measured evidence)

Not a "maybeinline" tier -- a cheaper plan and a cheaper per-tile prologue,
which benefit every call and matter at the microsecond scale where every
residual loss lives:

- `plan_contract` allocates ~3.8-4.4 KB per call (README, decisions.md
  "T10/T11", complex milestone close) and costs ~4.4-6.7 us (STATUS.md,
  "What works, measured"): `_classify_labels` builds three `Set`s and three
  `Vector{Int}`s (`src/planning/labels.jl`), `_order_free_labels` allocates a
  `sortperm` (`:131-132`), `_build_pair_group` and the swap path build
  another `kgroup`. Labels are `NTuple`s of static length; all of this can be
  tuple arithmetic. The T4 grid puts `planning` at 17-27% on `plain_64` and
  `smallMN_16x256x16`, and the label-order milestone's own open item
  hypothesises the double `_default_kernel` resolution (`:855-856`) behind
  its 18 microsecond-scale regressions (1.15-2.48x at 4-38 us).
- Per-sliver validation (`_check_pack_a/b` + `checked_tile_storage_bounds`,
  `src/packing/pack.jl`, `src/layout/tiles.jl`) at 7.4% of
  `ao2mo_2_dim16`, and `_execute_tile_prologue!`'s per-tile
  `checked_tile_storage_bounds` (`src/microkernels/interface.jl`). Hoisting to
  once-per-macro-block changes `pack_a!`'s documented "all validation before
  any write" contract (`src/packing/pack.jl`) -- the D1 record flagged
  this as "a separate decision", and it is one of the decisions requested in
  Section 7.
- `driver_loop` at 28.7% of `ao2mo_2_dim16`: `fill_offsets!` +
  `describe_block` + `_classify_slivers!` per block, for composites that are
  frequently rank-1 affine after `normalize_group` and could be described
  analytically without materialising an offset buffer.

None of these touch the frozen list. Each is independently measurable with
the existing profiler and ABBA guard.

### 5.3 Housekeeping surfaced by this review

- Rebase and merge `ccsd-t-stall` (F1/F2) onto `main`; re-run the ABBA guard
  on the combined tree, since F2 + the packing fix have not been measured
  together. Decide whether F2 ships with its `K`-blind limitation documented
  or gains a `Qk`-relative guard first (Track A's call; noted here because
  Section 6 would stack on it).
- Correct the two `ratio_note` cells in
  `benchmark/results/dontpack-C0-feature-table.csv` (Section 3.3).

## 6. Conditional design: a narrow, real-only "A-direct" tier

Built **only** if Section 5.1's gate passes or the user overrides Section 1.
This is the smallest thing that captures the `smallN`-class win, and it is
deliberately not a tier *system*.

### 6.1 Definition and trigger

**A-direct**: for a real element type, when `A`'s M-slivers are already in
the layout the microkernel consumes and `A` is reused by few micro-tiles,
skip `pack_a!` and hand the kernel a borrowed pointer into `A` itself, with a
runtime K-stride. `B` is packed exactly as today. Everything downstream of
the accumulator -- `store_tile!`, `beta_eff`, the scattered/vector store
choice -- is untouched.

Decided once per plan, at plan time, on quantities `plan_contract` already
has (`src/planning/plan.jl`):

| condition | source | why |
|---|---|---|
| `T <: Real` | `eltype(C)` | complex kernels consume planar/1e formats that do not exist unpacked (Section 6.5) |
| `kernel === nothing` (auto-selected) | keyword | mirrors F2's "only for an auto-selected kernel" guard; a named kernel is a request for the reference path |
| `atransform === identity` | `:835` | always true for real `T` (`_qs_isconj`, `:687-688`); stated so the complex increment cannot forget it |
| `cld(Qn, nr) ≤ N_SLIVERS_MAX` (proposed `2`) | `Qn`, `nr(kernel)` | the reuse trigger. Each `A` sliver is consumed `cld(Qn,nr)` times per K-block; below ~2-3 the packed copy is read fewer times than it cost to write. Deliberately *not* Octavian's `Qm·Qk ≤ mc·kc` arm, which Table 4.1 shows firing on 1%-packing wins |
| `run_A == Qm || run_A % mr == 0`, with `run_A = _leading_unit_run(morder, indA, A)` | reuse of `:142-156` on `A`'s strides (currently only applied to `C`) | exactly F2's predicate, on the source side: every M-sliver of `A` is a unit-stride run of `mr` rows, so a `Vec{W}` load per row-vector is legal without padding |
| `Qm % mr == 0` (increment 1 only) | `Qm`, `mr` | a tail sliver has `m < mr` valid rows; the kernel loads all `mr`, and unpacked memory has no zero padding to read -- over-reading past the tail is a memory-safety bug, not a rounding one. Increment 2 may pack only the tail sliver (per-sliver decision, Section 6.3) |
| `A`'s K composite is rank-1 affine in `A` | `normalize_group` over `kgroup`'s `A` map alone (a one-map variant of `src/layout/axis_group.jl`) | the kernel needs one runtime `lda` per plan; a multi-label K that does not fold has no single stride. Re-checked at run time from `dK_A.regular` (`:1090`) as a belt-and-braces fallback to packing |
| `parent(A) isa DenseVector{T}` | `Astorage` | same storage class `_pack_a_contiguous_eligible` and `_vector_store_eligible` already require (`src/packing/pack_contiguous.jl`, `src/microkernels/simd.jl`) |

Every condition is a plan-time scalar comparison; `_classify_slivers!`
(`:609-621`) is *not* needed for the decision because the eligibility is
provable from the `AxisGroup` strides before any block is filled -- the
per-sliver descriptors it produces are only needed for the increment-2
per-sliver refinement.

### 6.2 Is an unpacked operand even a legal kernel input? (the prerequisite gate)

The kernel reads `A` through `panel_vload(Vec{W,T}, packed_a,
packed_a_offset(kernel, v*W, p))` (`src/microkernels/simd.jl`), where
`packed_a_offset(kernel, i, p) = i + MR*p` is a frozen format
(`src/packing/format.jl`, `:43`). `panel_vload` on a `PackedPanel` is
`vload(Vec{W,T}, p.ptr + sizeof(T)*o)` (`src/packing/panel.jl`) -- it accepts
any address. So a `StridedAPanel{T}(ptr::Ptr{T}, lda::Int)` with an offset
rule `i + lda*p` is representable **without redefining the descriptor's
formula**: the generated body's `packed_a_offset(kernel, $(v*W), p)` becomes
`_a_offset(packed_a, kernel, $(v*W), p)`, with the `PackedPanel`/
`AbstractVector` method forwarding to `packed_a_offset` (byte-identical
behaviour) and the new panel type supplying `i + lda*p`. `B`'s loads are
scalar (`panel_load`, `:100-103`) and would be unaffected in increment 1.

**The risk that makes this a gate, not a step.** Phase H measured two
attempts to plumb a runtime quantity into this exact generated body and both
collapsed: "a runtime `Int` offset plumbed through the generated step, 27.0"
GFLOP/s against 96.6 for the raw pointer at `(16,14,8)` (decisions.md,
"Panel addressing milestone: Phase H", rejected-approaches paragraph). The
mechanism was not fully diagnosed there; the recorded lesson is "only a
genuine raw pointer works". A runtime `lda` is one loop-invariant register
plus one multiply-add per K-step in the address computation -- normal for a
C BLIS kernel, but Julia's register allocator has already surprised this
project once at this exact seam. Separately, an unpacked column-major `A`
with `lda` a multiple of 4 KiB (e.g. `M = 512` at `Float64`) is the classic
4K-aliasing / L1-set-conflict case that packing exists to avoid: `MR = 16`
rows x `kc = 256` K-steps at 2 KiB stride touch 256 lines mapping to two of
L1's 64 sets. With a reuse of 2 this may not matter; with reuse 3+ it
plausibly does, which is one more reason `N_SLIVERS_MAX` should be small.

**Gate G6.2 (benchmark-only, no `src/` change, ~half a day):** a scratch
copy of `_accumulate_step` with an `lda` parameter, timed in isolation as
`bench_store_path.jl` times the kernel today, at the shipped shapes for both
real dtypes, `kc ∈ {16, 256}`, `lda ∈ {MR (packed-equivalent), 256, 512,
4096/sizeof(T) (4K-aliased)}`, `PackedPanel` as the control. Pass criterion:
the `lda = MR` arm within 5% of the control (proves the runtime stride
itself is free), and the realistic `lda` arms no worse than the packing
cost they would displace (`~18.5 us` per 256x256 `Float64` block, Section
3.2). If the `lda = MR` arm shows Phase H's collapse, the tier is dead
without touching `src/` and this document's Section 1 stands unqualified.

### 6.3 Code paths, if G6.2 passes

- `src/packing/panel.jl`: `struct StridedAPanel{T}; ptr::Ptr{T}; lda::Int; rows::Int; kc::Int; end`
  (isbits), a `strided_a_panel(storage, base0, lda, rows, kc)` constructor,
  `panel_vload` method (identical to `PackedPanel`'s), and `Base.length`
  returning `rows + lda*(kc-1)` so `_execute_tile_prologue!`'s capacity check
  (`src/microkernels/interface.jl`) still means "the last address the kernel will
  touch is inside the span the caller validated" -- or, cleaner, a
  `_check_a_capacity(kernel, panel, kc)` method pair so the packed check is
  untouched and the strided one asserts `panel.kc >= kc`.
- `src/microkernels/simd.jl`: `_a_offset(packed_a, kernel, i, p)` as
  described in 6.2. One new function, one changed expression in the
  generated body, zero change for existing callers.
- `src/planning/plan.jl`: `ContractPlan` gains a type parameter or `Val` field
  `ADirect` (a `Bool` field would work too, but the nest's branch should be
  a compile-time constant so the packed path is byte-identical when it is
  off -- the project's established pattern for "the real path is unchanged
  as a `git diff` fact"). `plan_contract` computes the 6.1 predicate after
  the swap and F2 steps (Section 6.6). In `_execute_nest!`, the loop-3 body
  (`:1113-1128`) becomes: fill/classify as today; `if ADirect` skip the pack
  loop; in loop 1 (`:1135-1143`) construct `apanel` as
  `strided_a_panel(plan.Astorage, plan.Abase + ws.m_buf_A[rfirst+1] +
  ws.k_buf_A[1], dK_A.stride, MRk, kblock)` after one
  `checked_tile_storage_bounds(plan.Abase, rowsA, colsA_k,
  length(plan.Astorage))` per sliver (O(1) for affine axes, `src/layout/tiles.jl`)
  -- the same guarantee `pack_a!` gives today, and exactly the argument
  `_pack_a_contiguous!`'s safety note makes (`src/packing/pack_contiguous.jl`). Two
  concretely-typed `_execute_micro_tile!` call sites under the compile-time
  branch, never a `Union`-typed panel argument (GUARDRAIL at `:536-543`).
- `src/execution/workspace.jl`: `packed_a` still allocated (a plan may be re-planned
  into the same pooled workspace with a different verdict; `reserve!` is
  grow-only, `:249-282`). No change needed; noted so nobody "optimises" it
  away and breaks the pool invariant.
- `execute_tilewise!` (`:1175-1259`): **unchanged and always packs**, so it
  remains an independent oracle for the direct path.
- Increment 2 (later, separate decision): per-sliver eligibility from
  `ws.m_desc_A[r+1]` (`regular && stride == 1 && count == MRk`) so a tail
  sliver packs while full ones go direct; this reintroduces a per-sliver
  panel-type branch and doubles reachable `execute_tile!` specialisations
  inside the nest, which is why it is not increment 1.

### 6.4 Numerical correctness argument

- **beta applied once**: untouched. `beta_eff = firstpanel ? betaT : one(T)`
  (`:1095`) and the K-block loop structure are not modified; the tier
  changes where the kernel *reads* `A`, not when or how it *writes* `C`.
  The existing "beta applied exactly once across multiple M/N/K blocks" and
  "beta=0 with NaN-filled C" testsets (`test/execution/test_macro_blocking.jl`)
  apply verbatim to an A-direct plan.
- **Values**: the accumulator receives `A[m, k]` from the same address the
  packer would have read (`base + rows.base + i + col_offset(p)`, packing.jl
  `:150-156`), in the same K order, so the FMA sequence is identical to the
  packed path's for a full sliver -- agreement should be **bitwise** with
  `execute!`-packed on the same plan, not merely within tolerance. That is
  a stronger test than the suite's usual `atol`, and worth asserting.
- **Padding**: never read, because increment 1 requires `Qm % mr == 0` and
  the run predicate, so every A row the kernel loads is a valid element.
  `store_tile!` still guards `i < m`, `j < n`.
- **Conjugation**: real path only; `atransform` is `identity` by
  construction (`_qs_isconj`, `:687-688`). Nothing about `conjA`/`conjB`/
  `op` folding changes. The complex path is excluded (6.5), so no
  `transform` is ever applied in-kernel.
- **Bounds**: one `checked_tile_storage_bounds` per direct sliver, before
  any `@inbounds`/pointer read -- the Phase 2b review rule (`src/layout/tiles.jl`).
- **Adapter invariant**: `QuasiStridedBackend` neither knows nor needs to
  know about the tier; eligibility never causes a rejection or a fallback,
  only a different internal path. The hard-reject contract at
  `src/integrations/tensoroperations.jl` is not touched.

### 6.5 Complex: excluded from increment 1, and structurally

`PlanarKernel` reads `A` as two real planes (`re` at `t`, `im` at `vr + t`,
`src/packing/pack.jl`) and `OneMKernel` reads the 1e 2x2 real block
(`:273-297`); complex data in memory is interleaved `(re, im)`. An unpacked
complex `A` would need a de-interleaving vector load path -- a new kernel
body, not a new panel type -- and would have to apply `conj` in-kernel,
where it is currently a pack-time transform. Both complex kernels also
scatter-store unconditionally, so the swap and F2 already skip them
(`:864`, `_demote_for_run`'s generic method). Real-only first, and "complex
later" should be read as "complex requires a kernel change and its own
proposal".

### 6.6 Interactions with what already ships

- **`_demote_for_run` (F2)**: composes, does not conflict. F2 changes `mr`
  from `C`'s run; A-direct checks `A`'s run against the *final* `mr`, so it
  must run **after** F2 (order: `_default_kernel` -> swap -> F2 -> A-direct).
  A smaller `mr` makes both `Qm % mr == 0` and `run_A % mr == 0` more likely
  to hold, so F2 firing widens A-direct's eligibility. The concern is the
  one F2's review recorded: two shape heuristics without cost models on one
  planning path. `N_SLIVERS_MAX = 2` is chosen small enough that the
  no-pack decision is close to unconditionally right (the packed copy would
  be read at most twice), which is the only defensible setting without a
  cost model.
- **`_prefer_swap`**: after a swap, "`A`" is the original `B`
  (`ContractPlan` docstring, `:704-708`); the predicate is evaluated on
  whichever operand ended up in the A role, using its own `ind`/strides.
  The pinning test at `test/planning/test_plan_contract.jl` reads `mr(plan.kernel)`
  post-plan (`:1200`) against a pre-demotion swap decision; A-direct does not
  change `mr`, so it cannot disturb that assertion further than F2 already
  might.
- **`_pack_a_contiguous_eligible`**: A-direct's layout conditions are a
  strict subset of that gate's, so every A-direct-eligible sliver is one the
  vectorised packer already handles at ~`copyto!` speed. That is the honest
  statement of the tier's marginal value: it removes a fast memcpy, not a
  slow loop.
- **`_default_kernel`'s `Qm < mr` demotion** (`:515-520`) and its `===` pin
  (`test/planning/test_plan_contract.jl`, `test/planning/test_kernel_selection.jl`): untouched;
  A-direct is downstream of kernel resolution and never constructs a kernel.
- **Allocator path** (`src/integrations/tensoroperations.jl`): the workspace is
  still acquired and released identically; `packed_a` is merely not written.
- **Threading (deferred)**: `StridedAPanel` borrows a pointer into user
  storage, so the `GC.@preserve` scope (`:1047`) must also cover
  `plan.Astorage` -- today it preserves only `ws`. This is a real addition,
  not a detail: without it a direct panel is a dangling pointer the moment
  the GC moves nothing but decides to.

### 6.7 Risk and regression surface

- **Kernel collapse** (Phase H precedent) -- gated by G6.2.
- **Cache-set conflicts** for unlucky `lda` -- mitigated by the small reuse
  threshold; measured by G6.2's `lda` arms and by including `plain_512`
  (`lda = 4 KiB`) in the ABBA guard even though the trigger will not fire on
  it (a regression there would mean the branch is not compile-time).
- **Allocation**: the new panel is isbits; the compile-time branch adds no
  Union; the added `GC.@preserve` does not allocate. Must be asserted on the
  scattered fixture (`test/planning/test_kernel_selection.jl`) and on an eligible
  fixture, both dtypes, and on Julia 1.10 LTS where the `NTuple` accumulator
  is already fragile (STATUS.md, "Published").
- **Specialisation count**: one extra `_execute_nest!` specialisation per
  `(T, kernel)` when the tier is on; measurable via the microsecond-scale
  regression methodology the label-order milestone used.
- **Forced-ISA correctness**: eligibility depends on `mr`, which differs by
  ISA; every new test must derive its expectation from `kernel_shapes(T)`
  and `_default_kernel`, never a literal, and run under
  `test/forced_isa_runner.jl` with `avx2` and `unknown` -- the F2 review's
  blocking finding was exactly this mistake.
- **Not a correctness risk but a record risk**: the frozen packed format
  comment in `src/packing/format.jl` says "do not redefine". The
  design does not redefine it; the new offset rule lives on the new panel
  type. The comment should gain one sentence saying so, or a future reader
  will think it was violated.

### 6.8 Verification plan

1. **G6.2** first (Section 6.2). No `src/` change until it passes.
2. **Oracle tests** (`test/execution/test_macro_blocking.jl` style): randomized
   agreement vs. dense matmul and vs. `execute_tilewise!` on eligible
   shapes (`Qn ∈ {1, 6, 12}`, `Qm ∈ {mr, 2mr, 16mr}`, `Qk ∈ {1, 16, 256,
   1000}` to cross `kc`), alpha/beta in {0, 1, -0.5/2.5}, NaN-poisoned `C`
   with beta = 0, poisoned `packed_a` buffer (it must never be read on the
   direct path -- the staleness test at `:237` inverted). Bitwise equality
   against the packed `execute!` on identical plans (6.4).
3. **Gate-firing tests**, per the D3 review's S1 lesson: assert the plan
   *is* A-direct on an eligible fixture, and is *not* on each single
   violated condition (tail `Qm`, `run_A` failing the predicate, permuted
   `A` with non-unit rows, `PtrScatterAxis` rows, complex `T`, named kernel,
   `Qn` one sliver over threshold, non-foldable K composite).
4. **Zero allocation** through `execute!` on an A-direct plan (both dtypes),
   plus the existing scattered assertion, plus 1.10 LTS in CI.
5. **ABBA guard**: `benchmark/bench_real_path_guard.jl` two-tree, ≥ 21 reps,
   quiet machine (canary pair recorded), all 18 shapes + the three
   `SMALL_SHAPES`; a systematic one-sided shift anywhere the tier does not
   fire is a lost specialisation and blocks the merge.
6. **Re-profile**: `profile_to_suite.jl smallN_256x256x12 smallM_12x256x256
   smallMN_16x256x16 ao2mo_2_dim16 dim15_2_2_2 ccsd_t_1_dim16 --tag
   tiers-post`, both dtypes, compared against the post-`8dd01dd` baseline
   from Section 5.1 step 2 -- packing share on `smallN` should drop from
   ~33% to near zero and nothing else should move.
7. Forced-ISA runs (6.7). Full suite on 1.12/1.13 and 1.10.

### 6.9 Why there is no "maybeinline" and no tier 3 here

`maybeinline` needs static sizes this engine never has (Section 2). The
runtime analog -- a scalar triple loop for tiny problems -- would be slower
than the existing kernel at anything above one register tile, and the
measured tiny-problem losses are floor-bound, not kernel-bound (Section
3.3); Section 5.2 is the correct response. "Pack A only vs. A and B" is
already this engine's only mode; a `B`-direct tier (`smallM`-class shapes)
is left as a separate later decision because `B`'s kernel loads are scalar
and its cache story differs, and because nothing in the measured population
loses on it.

## 7. Decisions requested from the user

This is a sign-off gate. **No `src/` code should be written on the basis of
this document without an explicit go-ahead on the items below.**

1. **Accept or override the primary recommendation** (Section 1): do not
   build Octavian-style dispatch tiers now; treat STATUS.md's "highest-value
   next item" framing as superseded by Findings 1-3.
2. **Authorize the evidence gate** (Section 5.1): one quiet-machine run of
   the upstream `:mps`/`:ctmrg`/`:trg` categories at both dtypes, the
   feature-table pass over them, and a post-fix re-profile of the T4
   small-shape rows. Benchmark tooling only; no `src/`.
3. **Authorize the per-call-floor track** (Section 5.2), and specifically
   whether hoisting `checked_tile_storage_bounds` to once per macro block
   -- which changes `pack_a!`/`pack_b!`'s documented "all validation before
   any write" contract -- is acceptable, or whether only the
   contract-preserving items (allocation-free `plan_contract`, analytic
   rank-1 descriptors) are in scope.
4. **If (1) is overridden or the gate in (2) passes**: authorize G6.2 (the
   kernel microbenchmark, benchmark-only) as a hard precondition; confirm
   the tier's scope as real-dtype-only, A-direct-only, `Qm % mr == 0`,
   `N_SLIVERS_MAX = 2`; confirm the ordering after F2; confirm that
   `execute_tilewise!` stays packed as the oracle; and confirm the added
   `GC.@preserve plan.Astorage`.
5. **Housekeeping** (Section 5.3): rebase/merge `ccsd-t-stall` onto `main`
   and re-run the ABBA guard on the combined tree; decide whether F2 needs a
   `Qk`-relative guard before or after merge; correct the two mislabelled
   `ratio_note` cells in `benchmark/results/dontpack-C0-feature-table.csv`.
6. **Record**: whether STATUS.md's "Next task" section should be rewritten to
   reflect this review, or left as the historical framing with the pointer
   paragraph this document added. (Only the pointer was added; the original
   paragraphs are untouched.)

## Appendix A. Feature-table arithmetic used in Section 4

Sliver counts are `cld(Qm, mr)` and `cld(Qn, nr)` at the resolved shape
(`Float64` `(16,6,8)`, `Float32` `(32,6,16)` on AVX-512; `_default_kernel`
demotes to `(8,6,4)`/`(8,6,8)` when `Qm < mr`, `src/planning/kernel_selection.jl`).
`kc_eff = min(kc, Qk)` with `kc = 256` (`Float64`) / `768` (`Float32`). The
Octavian trigger's `mc·kc` uses `default_blocking(::Val{:avx512}, T)`
(`src/planning/blocking.jl`). Packing shares are from decisions.md's "Packing
speed" table (post-fix) or the "Profiling pass" T3 table (pre-fix, marked).
The `smallN` per-call decomposition: `Qn = 12 -> 2` N-slivers, `Qk = 256 ->
1` K-block, `Qm = 256 / mc = 128 -> 2` M-blocks, so `A` is packed once in
total and each `A` sliver is consumed by 2 micro-tiles; 24 flops per `A`
element loaded.

## Appendix B. Files and lines this document relies on (`main` @ `ebfbeb3`)

- `src/planning/labels.jl`: `_classify_labels` 19-88; `_order_free_labels` 127-133;
  `_leading_unit_run` 142-156; `_prefer_swap` 174-180; menus 300-301;
  `_kernel_from_shape` 356-365; `_default_kernel(T,Qm,Qn)` 515-520;
  `_axis_of` 544-547; `_pack_sliver!` 560-566; `_execute_micro_tile!`
  568-575; `_sliver_panel` 602-605; `_classify_slivers!` 609-621;
  `ContractPlan` 710-735; `plan_contract` 801-881 (swap 864-876);
  `_plan_contract` 887-937; `execute!` 1016-1053; `_execute_nest!`
  1056-1157 (B pack 1098-1106, A pack 1120-1128, micro-tiles 1131-1144,
  `beta_eff` 1095); `execute_tilewise!` 1175-1259.
- `ccsd-t-stall` @ `1da0ef7`, `src/planning/kernel_selection.jl`: `_demote_for_run` 197-209;
  call sites 904-906, 912-914.
- `src/packing/pack.jl`: `_check_pack_a/b` 45-83; `_pack_panel!` 102-123;
  `_copies_unchanged` 130-132; `_pack_a_contiguous_eligible` 141-146;
  `_pack_a_contiguous!` 162-175; `pack_a!` 188-205; `pack_b!` 218-230;
  complex packing 232-392.
- `src/packing/panel.jl`: `PackedPanel` 17-20; `panel_vload` 37-40; `panel_load` 42-43.
- `src/microkernels/simd.jl`: `_accumulate_step` 80-119 (A loads 96-99, B loads
  100-103); `accumulate` 131-141; `_unit_stride_rows` 145-147;
  `_vector_store_eligible` 180-181; `_store_tile_vector!` 244-301;
  `store_tile!` 314-326; `execute_tile!` 336-345.
- `src/microkernels/interface.jl`: `_execute_tile_prologue!` 173-201.
- `src/packing/format.jl`: frozen formats 1-2, 43, 51, 64-65.
- `src/execution/workspace.jl`: `ContractWorkspace` 42-105; `reserve!` 249-282.
- `src/layout/tiles.jl`: `PtrScatterAxis` 56-65; `QSTile` 128-133;
  `checked_tile_storage_bounds` 263-272.
- `src/layout/axis_group.jl`: `fill_offsets!` 136-199; `BlockDescriptor` 211-216;
  `describe_block` 227-251; `normalize_group` 277-327.
- `src/planning/blocking.jl`: measured AVX-512 rows 85-86; "not a cache model" 33-38.
- `src/integrations/tensoroperations.jl`: `_qs_prepare` 248-276; `tensorcontract!`
  304-357; hard-reject note 358-366.
- Tests: `test/planning/test_plan_contract.jl` (kernel `===` pin), `:1140-1203` (swap
  pinning; `mr(plan.kernel)` read at 1200); `test/planning/test_kernel_selection.jl`
  (scattered zero-alloc), `:189-195` (Qm demotion pin);
  `test/execution/test_macro_blocking.jl`, `:237`, `:274` (beta-once, staleness,
  tilewise oracle); `test/forced_isa_runner.jl`.
- Benchmarks: `benchmark/results/dontpack-C0-feature-table.csv`;
  `benchmark/profile_to_suite.jl`, `benchmark/profile_buckets.jl`;
  `benchmark/bench_real_path_guard.jl`; `benchmark/bench_store_path.jl`;
  `benchmark/bench_to_suite.jl`; `benchmark/probes/probe_ccsd_t_stall_f2rule.jl`.
- `docs/decisions.md` sections: "Panel addressing milestone: Phase H"
  (rejected approaches; "Where the remaining gap actually is");
  "T3 measurement results"; "T5 profiling triage results"; "Label-order
  milestone"; "Profiling pass: where does QuasiStrided spend its time?";
  "Packing speed: vectorizing the real packing loop as it exists today";
  "T4: extended the grid..."; and, on `ccsd-t-stall`, "Kernel-stalled/
  store-dominated fix: run-length-aware demotion (F2) and inlined
  `accumulate` (F1)".
- Octavian.jl 0.3.29 (`~/.julia/packages/Octavian/4f4xi/src/`):
  `matmul.jl` 130-150 (`dontpack`), 210-229 (`maybeinline`), 320-360
  (`_matmul_serial!`), 362-396 (`matmul_st_pack_dispatcher!`);
  `macrokernels.jl` 124-133 (`loopmul!`); `global_constants.jl` 17-19;
  `block_sizes.jl` 4-33.
