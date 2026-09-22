# Proposal: vectorized fast paths for the planar complex store and complex packing

**Status: design review and sign-off gate. Nothing in `src/` changes on the
authority of this document.** It answers the question `docs/decisions.md`'s
"ComplexF64 `:tccg` slowdown" section (2026-09-22) left open: is it worth
building the "deliberately deferred, measurement-gated follow-on" that
`src/microkernels/planar.jl`'s own docstring names -- a vectorized unit-stride store
for the planar complex kernel -- and the analogous fast path for complex
packing? It was written 2026-09-22 against `main` at `f7fa490`. Every number
below is either re-derived here from a file in the tree or cited to a
`docs/decisions.md` section by name.

Frozen and untouched by anything proposed here: `QuasiStridedBackend`'s
hard-reject invariant (`src/integrations/tensoroperations.jl`), the frozen packed
format formula in `src/packing/format.jl` (`i + MR*p`) and its documented
generalization in `src/microkernels/interface.jl` (`p * per_k + plane * reg_tile + i`,
:245-256), the beta-applied-once contract (`src/microkernels/interface.jl`'s `_store_prologue!`
/ `_axpby_tile!`), and `_execute_nest!`'s loop structure (`src/execution/execute.jl`). No
part of this proposal touches any of them; where a design choice could be
mistaken for doing so, this document says so explicitly.

## 1. Recommendation, up front

**Build both, but not as one unit and not in the order their impact alone
would suggest.** Reading the actual code changes the picture from what the
investigation's summary implied:

1. The store fast path (Section 3) is the bigger single-case win --
   `ccsd_t_1_dim16`'s store share falls from 16.1% to 60.8% under ComplexF64
   (`docs/decisions.md`, cited in full in Section 2) -- but it is **also the
   harder and riskier of the two**, for a reason the investigation's summary
   did not surface: the store's job is not a same-format copy, it is an
   **interleave of two SIMD planes into one `Complex{T}` value, combined with
   a genuine complex multiply-add** (`alpha`/`beta` are `ComplexF64`/
   `ComplexF32` scalars, so `alpha*r + beta*C_old` mixes the real and
   imaginary planes together -- it is not "scale each plane independently").
   That is new arithmetic this kernel has never vectorized before, on a data
   layout (`Complex{T}`'s native interleaved binary layout) this package has
   never read or written through a raw `Ptr{real(T)}` before either.
2. The packing fast path (Section 4) is the smaller win by the profiled
   evidence (`ccsd_6_dim16`'s packing share moves 31.0% to 40.4%, a shallower
   shift than the store side's) but is **structurally simpler**: packing's
   `transform` is `identity` or `conj`, and `conj` on a complex number
   negates only the imaginary plane -- there is no cross-plane arithmetic, no
   `alpha`/`beta`, and no need to read the destination first. It is a
   deinterleave-and-copy, the mirror image of the store's interleave, minus
   the part that makes the store hard.
3. Recommended order: **packing first**, as the lower-risk change that
   exercises the new interleave/deinterleave machinery (Section 5's shared
   primitives) on the easier of the two problems, then the store fast path
   once that machinery is proven correct and fast in the tree. This is the
   reverse of "biggest win first" and is deliberately so -- Section 6 explains
   why derisking in this order is worth the smaller first win.
4. Both are real-`T`-complex-storage engineering, not a research question the
   way the dispatch-tiers proposal's kernel-collapse risk was (`docs/
   proposals/dispatch-tiers.md`, Section 6.2's Phase H precedent): the
   arithmetic is fully specified below, the primitives exist in this
   project's SIMD dependency already (`SIMD.shufflevector`, checked in
   Section 6), and the eligibility gate is a narrowing of predicates this
   package already has (Section 3.2, 4.2). The genuine open risk is Section
   6's register-pressure and `_prefer_swap` interaction, not "can this be
   built at all".
5. **Scope was expanded to four pieces after this section was first written**
   (Decision 2, Section 8, pulling `OneEFormat`/`OneMMethod` in). Items 1-4
   above remain accurate for the original `PlanarFormat` pair; Sections 3.4,
   4.4 and 4.5 work out the other two pieces (`OneEFormat`'s A-pack,
   `OneMKernel`'s store), and **Section 6.5 gives the full, updated four-piece
   build order** -- pack both formats first, `PlanarFormat` store second,
   `OneMKernel` store last, for reasons items 1-4 above do not cover
   (`OneMKernel`'s store needs a technique novel to this codebase, not an
   extension of an established one, and has no default-path value -- Section
   3.4).

## 2. Motivation, re-derived

`docs/decisions.md`'s "ComplexF64 `:tccg` slowdown" section (2026-09-22) is
this proposal's trigger; quoted in full because every later section reasons
from it:

> Job 7087420's evidence-gate run
> (`benchmark/results/worker6160-2026-09-22/bench_to_suite.csv`) showed
> QuasiStrided's median ComplexF64/Float64 GFLOP/s ratio on `:tccg` sitting at
> **0.29** across all 48 (dtype, case) pairs, against StridedBLAS's **0.68**
> on the identical cases. A truly efficient complex kernel doing ~4x the real
> arithmetic of its real counterpart ... would land near 0.25 by construction;
> QuasiStrided sits right on that naive floor while StridedBLAS's ZGEMM
> extracts real efficiency beyond it.

and the two bucket-share findings:

| case | bucket | Float64 | ComplexF64 |
|---|---|---|---|
| `ccsd_t_1_dim16` | microkernel | 79.3% | 34.6% |
| `ccsd_t_1_dim16` | store | 16.1% | **60.8%** |
| `ccsd_6_dim16` | microkernel | 54.3% | 44.6% |
| `ccsd_6_dim16` | packing | 31.0% | **40.4%** |

**A second, independent, already-existing measurement corroborates the
direction and gives it this project's own standard vocabulary.**
`benchmark/bench_complex_efficiency.jl` computes the *complex efficiency
ratio* -- one engine's complex throughput divided by its own real throughput
at the same shape, complex charged the textbook 8 flops/MAC so `1.0` means
"complex is treated exactly as well as real", and the reference project
(`docs/decisions.md`) measures 1.42-1.47. Run today on `main` at `f7fa490`
(`ccqlin038`, `MAIN_SHAPES` + `SMALL_SHAPES`, both dtypes, 21 reps):

```
geomean complex efficiency: ComplexF64 0.938   ComplexF32 0.690
```

with the script's own warning firing on the `ComplexF32` figure ("< 0.9 ...
indicates a structural overhead specific to complex"). **This is a different
population from the `:tccg` measurement above** -- plain square/skewed
matmul shapes from `benchmark/harness.jl`, not upstream's genuine multi-index
tensor-network-style contractions -- and the two must not be conflated as one
number. What they agree on is the direction: `ComplexF32` already shows a
flagged structural overhead on the simplest shapes this engine has, and
`:tccg`'s real multi-index contractions show a much larger gap (0.29) on
`ComplexF64` specifically, which `bench_complex_efficiency.jl`'s plain-matmul
population does not by itself predict (its `ComplexF64` geomean, 0.938, looks
fine) -- i.e. **the plain-matmul efficiency benchmark would not, on its own,
have caught this**, because the store/packing cost that dominates on
multi-index shapes is amortized away on large square matmuls. Both facts are
true and both belong in the record: the structural overhead exists even on
the simplest shapes (packing/store fixed costs matter more per flop when
`kc`/tile counts are large and dense), and it is far worse on the genuinely
skewed shapes this package exists for.

## 3. Current state, Part 1: the planar store path

### 3.1 What `store_tile!` does today for a `PlanarKernel`

`src/microkernels/planar.jl`, `_store_tile_planar!`: for each output
column `j < n` and each register-tile row-vector `v`, the two accumulator
planes for that `(v,j)` are read (`revec = acc[idx]`, `imvec =
acc[NV+idx]`), and **every lane is stored one at a time**:

```julia
for lane in 1:W
    i = v*W + lane - 1
    i < m || break
    _axpby_tile!(destination, i, j, alpha, Complex(revec[lane], imvec[lane]), beta)
end
```

`_axpby_tile!` (`src/microkernels/interface.jl`) is generic in the value type and
already `@inline`, and its `alpha`/`beta` branch (`iszero(beta)`/`isone(beta)`
/ general) is resolved once per call by `_store_prologue!`
(`src/microkernels/interface.jl`), not per lane -- so **there is no cheap hoist left
inside the existing scalar loop**; the cost is the `MR*NR` count of scalar
`tile_store!` writes (`src/layout/tiles.jl`, one bounds-free array write and
one `axis_offset` computation per element) against the real path's `MV*NR`
count of `W`-wide vector writes. This matches the file's own docstring:
"Ships the scattered/scalar path only ... The unit-stride plane-to-interleave
fast path is a deliberately deferred, measurement-gated follow-on and is NOT
built here" (`:279-280`).

There is **no eligibility check at all** on the planar store path -- unlike
the real path, which branches on `_vector_store_eligible`
(`src/microkernels/simd.jl`) between `_store_tile_vector!` and
`_store_tile_scattered!`. Every planar store, regardless of destination
layout, takes the scalar path.

### 3.2 What the real path's fast store actually requires (the template)

`_store_tile_vector!` (`src/microkernels/simd.jl`) is gated by
`_vector_store_eligible` (`:180-181`):

```julia
_unit_stride_rows(tile.rows) && tile.storage isa DenseVector{T}
```

i.e. `AffineAxis` rows with `stride == 1`, into `Vector`/`Memory`-backed
storage. When eligible, for each full `W`-row block it does exactly one
`vload`/`vstore` pair:

```julia
vstore(
    iszero(beta) ? alpha*vec :
    isone(beta)  ? muladd(alpha, vec, vload(Vec{W,T}, storage, at)) :
                   muladd(alpha, vec, beta*vload(Vec{W,T}, storage, at)),
    storage, at
)
```

with a scalar tail for any row-vector that straddles `m`. This is the
shape Section 3.3 below generalizes -- but for real `T`, `alpha*vec` is a
plain real scalar-vector multiply and `beta*vload(...)` likewise: no
cross-lane mixing, no interleave. That is exactly the part complex adds.

### 3.3 What a vectorized planar store needs, concretely

**The destination's binary layout.** `destination.storage` for a complex
`QSTile` is a `DenseVector{Complex{T}}` (`src/layout/tiles.jl`, `tile_load`/
`tile_store!` at `:185-208` index it as an ordinary `AbstractVector`).
`Complex{T}` is an `isbits` struct of two `T` fields with no padding, so `n`
contiguous `Complex{T}` values occupy `2n` contiguous `T`s in memory as
`[re_0, im_0, re_1, im_1, ..., re_{n-1}, im_{n-1}]` -- the "1r"/interleaved
layout BLIS itself targets, and Julia's own default binary representation
for the type, not something this package has to construct. Nothing in the
codebase currently reads or writes through this reinterpretation --
`src/packing/pack.jl` states as a design note that "the source is never
`reinterpret`ed, because a `QSTile` addresses arbitrary strided (possibly
scattered) storage for which that would be unsound" -- which is exactly why
the fast path below needs its own eligibility gate at least as strict as
`_vector_store_eligible`'s: `reinterpret`ing a `Ptr{Complex{T}}` as
`Ptr{real(T)}` is a sound bitcast only for genuinely dense, contiguous
storage, never for a `ScatterAxis`/`PtrScatterAxis` row or a `SubArray`.

**The arithmetic.** For a full `W`-row block at column `j`, the kernel
already has `revec`/`imvec` (the two accumulator planes, `Vec{W,real(T)}`
each). The store must produce, per lane, `Complex{T}`'s real and imaginary
parts of `alpha*r + beta*C_old`, where `alpha`, `beta` are **complex**
scalars (`T`, not `real(T)`) and `r = Complex(revec[lane], imvec[lane])`.
Writing `alpha = ar + ai*im`, `beta = br + bi*im`, `C_old = or + oi*im`:

```
new_re = ar*revec - ai*imvec + br*or - bi*oi
new_im = ai*revec + ar*imvec + bi*or + br*oi
```

-- the same four-real-FMA structure `_accumulate_step_planar`
(`src/microkernels/planar.jl`) already uses for `A*B`, just applied
once per output element instead of `kc` times per K step, and with the
**guardrail that file already states** for FMA grouping (its own comment at
`:147-151`: `c - ai*bi` is REJECTED because Julia's `muladd` chain does not
fuse it correctly without `contract`; the real part must be built as
`muladd(-ai, bi, muladd(ar, br, c))`). This is not a new risk to discover --
it is the existing guardrail, applied at a second call site, and the same
discipline (`@code_native` check per shipped shape, per the file's own
verification note at `:152-156`) should be repeated here rather than trusted
by inspection.

Old `C` (`or`, `oi`) is only needed when `beta != 0`: at `beta == 0` the
`br*or - bi*oi` / `bi*or + br*oi` terms are never computed and `C_old` is
never read (preserving the existing "beta == 0 never reads old C"
contract, `src/microkernels/interface.jl`'s ternary and the NaN-poisoning test at
`test/microkernels/test_planar_kernel.jl` that pins it for the *scalar* path
today -- the fast path must pass the identical test).

**The interleave/deinterleave.** Reading old `C` (`beta != 0`) means loading
`2W` contiguous reals from the destination and splitting them into `oi`/`or`
planes; writing the result means combining `new_re`/`new_im` planes back
into `2W` interleaved reals. `SIMD.jl` v3 (this package's pinned major
version, `Project.toml`) provides exactly the primitive:

```julia
shufflevector(x::Vec{N,T}, y::Vec{N,T}, ::Val{I}) -> Vec{length(I),T}
```

(`~/.julia/packages/SIMD/UiGbs/src/simdvec.jl:507-509`, a compile-time index
tuple `I`, so a deinterleave/interleave shuffle is a single LLVM
`shufflevector` instruction, not a runtime gather). Interleave:
`shufflevector(new_re, new_im, Val(ntuple(k -> isodd(k) ? (k-1)÷2 : W+(k-1)÷2, 2W)))`
-- expressed here as the general odd/even pattern; the exact `Val` tuple
should be written out at each shipped `W` and pinned by
`@code_native`/`@code_llvm`, the same discipline `_accumulate_step_planar`'s
own header comment already asks for its FMA count. Deinterleave (for reading
old `C`) is the inverse gather. Both are compile-time-index shuffles at every
concrete `(MR,NR,T,W)` specialization, so this composes with the existing
`@generated`-over-literal-indices discipline every store/accumulate function
in this file already follows (the Cliff B guardrail, `:16-22`); no
`@generated` boundary is crossed that does not already exist for a different
reason.

### 3.4 `OneMKernel`'s store path: same contract, a genuinely different (interleaved, not split-plane) technique -- not "apply 3.3 twice"

**This section exists because Decision 2 (Section 8) pulled `OneEFormat`/
`OneMMethod` into scope after this document's first draft, which had scoped
them out (the original Section 4.4, now superseded below).** Reading
`src/microkernels/onem.jl` in full changes the picture from "the same store problem
on a doubled layout" to a structurally different one, in both directions --
easier in one respect, harder and novel in another.

**Current state.** `_store_tile_onem!` (`src/microkernels/onem.jl`) is the
1m counterpart of `_store_tile_planar!`, same scalar `_axpby_tile!` loop, same
"deliberately deferred" framing in its own header comment (`:227-229`). But
the accumulator it reads from is not two split planes -- it is **the real
`SIMDKernel`'s own accumulator**, reused verbatim (`accumulate(kernel::OneMKernel,
...)` literally calls `accumulate(kernel.inner, ...)`, `:200-208`, "There is no
arithmetic here and there must never be"). That accumulator holds a real
`2MR x NR` tile in which complex row `i`'s real and imaginary parts are real
rows `2i` and `2i+1` -- and because the constructor enforces `W` even and
`2MR` a multiple of `W` (`:74-88`, the "ADJACENCY" comment at `:217-223`),
those two real rows always land in **adjacent lanes of the same accumulator
`Vec`**: complex row `i = v*(W/2) + u` is `Complex(acc[v+MV*j+1][2u+1],
acc[v+MV*j+1][2u+2])`. No cross-`Vec` plane split, ever.

**Why this makes the data-movement half of the problem *easier* than
Planar's.** `Complex{T}`'s native memory layout is `[re_0,im_0,re_1,im_1,...]`
(Section 3.3). An accumulator vector `vec` for `HW = W/2` complex rows already
holds exactly `[re,im,re,im,...]` in that same adjacent-pair order (`:260-268`'s
`vec[2u+1], vec[2u+2]` reader is reading a layout that is *already*
interleaved, not two things that need interleaving). So:

- **Writing a full block needs no interleave shuffle at all** for the
  low-level data movement -- if the arithmetic result is computed as an
  interleaved `Vec{W,real(T)}` (see below), one `vstore` at the destination's
  raw pointer is a direct copy, byte order already matching.
- **Reading old `C` for `beta != 0` needs no deinterleave either** -- a plain
  `vload(Vec{W,real(T)}, ...)` from the destination's raw pointer already
  produces `[or,oi,or,oi,...]` in the same adjacent-pair order the arithmetic
  needs, with zero shuffle cost. Planar's store (Section 3.3) needs a
  deinterleave here; 1m's does not.

**Why the arithmetic half is *not* "apply 3.3's four-real-FMA recipe" and is
the genuinely novel, higher-risk part.** Section 3.3's four-real-FMA grouping
computes `new_re`/`new_im` as two **separate** `Vec{W,real(T)}` values (one
plane each), then interleaves them together as a last step. There is no
"separate plane" to compute here -- the value that must come out of the
arithmetic is already the single interleaved `Vec{W,real(T)}`
`[new_re_0,new_im_0,new_re_1,new_im_1,...]`, computed from an interleaved
input `vec` and complex scalars `alpha`,`beta`. The standard SIMD technique
for "multiply an interleaved-complex vector by a scalar complex" is:

```
swapped = shufflevector(vec, Val(pair-swap))   # [im,re,im,re,...]
result  = signpat .* (aR_bcast .* vec) + signpat2 .* (aI_bcast .* swapped)
```

(broadcast `aR`/`aI` = `real(alpha)`/`imag(alpha)`, a fixed alternating
`+1/-1` sign-pattern constant vector to get `aR*re - aI*im` at even lanes and
`aR*im + aI*re` at odd lanes) -- the "shuffle + alternating add/subtract"
family of techniques that maps directly onto hardware `fmaddsub`/`fmsubadd`
instructions where available. **This is exactly the family of technique this
codebase's own planar file explicitly rejects for its `accumulate` step**:
`src/microkernels/planar.jl` states the design choice in its very first lines
-- planar's data is kept in split planes specifically "so the data is already
in the right lanes and the body is four real FMAs ... -- no shuffles, no
`fmaddsub`, no duplicated lanes" -- and `PlanarMethod`'s own docstring
(`src/microkernels/interface.jl`) repeats the same phrase almost verbatim as
the method's defining property. Both citations are about planar's hot
*accumulate* loop, not a store epilogue, and neither gives the underlying
numerical reason shuffle/`fmaddsub` was rejected there (that reasoning is not
re-derived in this addition) -- so it is not automatically true that the same
objection applies to a once-per-tile store. But the same general technique
being singled out for rejection, twice, in the file whose whole design this
proposal otherwise leans on, is at minimum a signal that this project has not
treated "shuffle + fmaddsub-style complex arithmetic" as a safe default
elsewhere. **A from-scratch correctness derivation -- including explicitly
answering whether planar's rejection reason applies here or not -- is
required before writing any code**, not an assumption of correctness by
analogy to Section 3.3's already-used technique.

**Net honest comparison to Planar's store (Section 3.3):** fewer shuffles
needed for data movement (0 vs. 1-2), but a novel arithmetic technique with no
established precedent *and* a documented local precedent for treating that
family of technique with suspicion. This nets out, in this document's
judgment, to **comparable or somewhat higher risk than Planar's store**, not
lower -- "fewer shuffles" is not the same as "easier", and should not be read
as license to treat this as the smaller task. See Section 6.5 for the
resulting build-order recommendation.

**Value proposition, a genuinely separate question from difficulty.**
`OneMMethod` is "selected only by naming the kernel" (`src/execution/execute.jl`)
and no case in `bench_to_suite.jl`, `profile_to_suite.jl`, or any default
`tensorcontract!` call path ever constructs a `OneMKernel` without a caller
explicitly asking for one by name. Unlike Planar's store fast path (which
speeds up every complex contraction using the default kernel, i.e. everything
this document's Section 2 motivation is about), this fast path's only current
consumers are `benchmark/bench_complex_efficiency.jl`'s "arm 2"
(`:120-160ish`, exercises both methods across all shipped shapes at plain-matmul
shapes) and anyone doing method-comparison research. It cannot move the
`:tccg` numbers in Section 2 at all, because `:tccg` never names a kernel.
This is not an argument against building it -- Decision 2 already authorized
it -- but it is the honest reason this document recommends it be built last
(Section 6.5) and re-confirms Decision 2 rather than silently treating it as
equally urgent to the default-path work.

## 4. Current state, Part 2: complex packing

### 4.1 What `_pack_panel_complex!` does today

`src/packing/pack.jl`, called from the `ComplexKernelDescriptor` overloads
of `_pack_a!`/`_pack_b!` (`:454-466`, `:496-508`): for each logical K step and
each row/column index, one complex element is loaded via the fully generic
`tile_load` (`src/layout/tiles.jl`), `transform`ed (`identity` or `conj`,
never per-real-half -- the file's own contract comment at `:321-329`), then
split into the packed format by `_pack_emit!` (`:343-349` for
`PlanarFormat`, one `panel_store!` per plane, i.e. **two scalar stores per
complex element**). Explicitly documented as "a parallel loop rather than a
generalisation of `_pack_panel!`" (`:313-319`) so the real path's later
restructuring (the full/tail split and `_pack_a_contiguous!` fast path,
`docs/decisions.md` "Packing speed") left this loop untouched -- it has never
had a fast path of any kind, for either operand, under either complex
method.

### 4.2 What the real path's fast pack requires (the template)

`_pack_a_contiguous!` (`src/packing/pack_contiguous.jl`), gated by
`_pack_a_contiguous_eligible` (`:158-163`):

```julia
packed isa PackedPanel{T} && source.storage isa DenseVector{T} &&
    _copies_unchanged(transform, T) && m == MR && _unit_stride_rows(source.rows)
```

When eligible, packing one A sliver is `kc` iterations of one `vload`/
`vstore` pair of width `MR`:

```julia
v = vload(Vec{MR,T}, sp + sizeof(T)*(rowbase + axis_offset(cols, p)))
vstore(v, dp + sizeof(T)*(MR*p))
```

-- a straight vectorized copy, because `_copies_unchanged(identity, T)` and
`_copies_unchanged(conj, T)` are both `true` for real `T` (`conj` is the
identity on reals, `:147-149`).

### 4.3 What a vectorized complex pack needs, concretely

**This is genuinely simpler than the store side, for one structural reason:
`conj` on a complex number touches only the imaginary plane.**
`_copies_unchanged` is `false` for `conj` on a complex `T`
(`_copies_unchanged(::Any, ::Type) = false` catches it, `:149`), so the
existing real-path predicate already correctly excludes complex `conj` from
being treated as a copy -- but a complex fast path does not need
`_copies_unchanged` to be `true` for `conj` the way the real path does,
because `conj`'s effect on a *split* plane pair is exactly "negate the
imaginary `Vec`, leave the real `Vec` alone": no cross-plane arithmetic, no
scalar broadcast, no `alpha`/`beta` at all (packing never applies them --
only `transform`).

For `PlanarFormat` (the default method's format for both operands, and the
only format this proposal scopes -- Section 4.4), packing one full `MR`-row
(or `NR`-column) sliver at one K step, from unit-stride dense
`Complex{T}` source, eligible under a gate mirroring `_pack_a_contiguous_eligible`
but checking `source.storage isa DenseVector{Complex{T}}`, `m == MR`,
`transform ∈ {identity, conj}` (both handled, not excluded), and
`_unit_stride_rows(source.rows)`:

1. One `vload(Vec{2*MR,real(T)}, ...)` of the `MR` contiguous source
   `Complex{T}` values, reinterpreted as `2*MR` interleaved reals (the same
   bitcast Section 3.3 justifies, same eligibility strictness required).
2. One deinterleave shuffle into `revec::Vec{MR,real(T)}`,
   `imvec::Vec{MR,real(T)}` (even/odd lanes).
3. If `transform === conj`: `imvec = -imvec`. (Zero cost relative to
   `identity` beyond one negate, and it is a plane-only operation --
   contrast with the store side, where every operation mixes planes.)
4. Two `vstore`s: `revec` at `packed_a_plane_offset(kernel, 0, 0, p) ==
   p*per_k + i` for `i = 0` (i.e. the base of the real plane's `MR`-wide
   contiguous region for this K step) and `imvec` at plane `1`'s
   equivalent offset (`src/microkernels/interface.jl`: `p*packed_a_per_k(d)
   + plane*MR + i`, both `MR`-contiguous in `i` for fixed `p`, `plane`) --
   i.e. **the destination is already laid out as two separate `MR`-wide
   contiguous regions per K step**, so no destination-side interleave is
   needed at all; only the *source* needs deinterleaving. This is the
   concrete sense in which packing is the easier half of the round trip:
   the store fast path needs an interleave on the way out AND (when
   `beta != 0`) a deinterleave on the way in; packing needs only one
   deinterleave, ever, and never a destination-side shuffle.

No `alpha`/`beta`, no old-value read, no complex multiply. `NR`-side (`B`)
packing is the same shape with `NR` replacing `MR`.

### 4.4 `OneEFormat`'s A-pack: a design, superseding this section's original exclusion rationale

**This section originally scoped `OneEFormat` out** (Decision 2, Section 8,
overrode that after this document's first draft). On rereading
`_pack_emit!` for `OneEFormat` (`src/packing/pack.jl`) closely rather than
characterizing it from a distance, the actual layout turns out to be **at
least as tractable as `PlanarFormat`'s, and for one of its two halves,
literally trivial** -- the opposite of "fundamentally different, doesn't
reduce the same way."

**What `_pack_emit!` for `OneEFormat` actually writes**, for one source
complex element `z = re + im*i` at physical row/column index `t`, logical K
step `p`, register-tile width `vr` (`MR` for the A operand, the only operand
that uses this format -- Section 4.5):

```
plane_offset(0, 2t,   p) = re     plane_offset(0, 2t+1, p) = im
plane_offset(2, 2t,   p) = -im    plane_offset(2, 2t+1, p) = re
```

Both "planes" are `2*vr`-real regions, `2vr` apart (`plane_offset(0,...)` and
`plane_offset(2,...)`, `p*4vr + plane*vr + index` at `plane ∈ {0,2}`). Written
out for a full `vr`-row sliver at one `p` (`t = 0..vr-1`), **plane 0 is
literally**:

```
[re_0, im_0, re_1, im_1, ..., re_{vr-1}, im_{vr-1}]
```

**-- which is `Complex{T}`'s own native interleaved binary layout, byte for
byte, for `identity` transform.** Under `_unit_stride_rows`/`DenseVector{T}`
eligibility (the same gate Section 3.3/4.3 already need), packing plane 0 for
`identity` is not "deinterleave then store" -- it needs **no shuffle at all**,
the same `vload(Vec{2vr,real(T)}) -> vstore` pure-copy case the real path's
`_pack_a_contiguous!` already has for `RealFormat` (Section 4.2). This is
*simpler* than `PlanarFormat`'s pack (Section 4.3), which always needs a
deinterleave shuffle regardless of transform.

Plane 2 (`[-im_0, re_0, -im_1, re_1, ...]`) is the same source vector with (a)
each adjacent pair swapped and (b) the now-first-of-pair lane negated -- one
`shufflevector` with a fixed compile-time pair-swap index pattern (the same
primitive Section 3.3 already needs, applied to the *load* side instead of a
store epilogue), composed with one multiply against a fixed alternating
`[-1,+1,-1,+1,...]` constant vector. Two fixed, compile-time, `W`-independent*
primitives (*modulo needing the pattern at each shipped `vr`, exactly as
Section 6.2 already requires for every other shuffle in this document) -- no
runtime branch, no cross-element data dependency.

**`conj` does not break this, it just swaps which plane is free.** Applying
`transform = conj` before splitting negates `im` first, so the two planes
become `[re,-im,re,-im,...]` (plane 0, now needs the pair-swap-free
alternating-negate the identity case's plane 2 needed) and `[im,re,im,re,...]`
(plane 2, now the pair-swap *without* a negate that plane 0's identity case
had). **Both `transform` values reduce to the same two primitives
(pair-swap-or-not, negate-or-not), just swapped between the planes** -- a
compile-time `Val{transform}` dispatch picks which primitive goes with which
plane, with no new technique needed for `conj` beyond what `identity` already
requires. This is a strictly smaller design surface than Planar's pack
(Section 4.3), which needs a per-transform branch on the *same* plane
(negate `imvec` or not) rather than a swap between planes, but is
comparable in total primitive count.

**Net honest assessment: `OneEFormat`'s A-pack is not harder than
`PlanarFormat`'s pack, and for the `identity`-transform, `beta`-free (packing
never applies `beta`) case, it is the simplest fast path in this entire
document** -- a literal `vload`/`vstore` copy with no arithmetic, on par with
`RealFormat`'s existing trivial case. The one genuine added cost, inherent to
1m's design and not to the fast path, is that it writes **twice** the
destination bandwidth of `PlanarFormat` (two `2vr`-real regions instead of one
`2vr`-real region) -- unavoidable, since `OneEFormat` has twice `PlanarFormat`'s
packed footprint by construction (`reals_per_element(OneEFormat) = 4` vs. `2`,
`src/packing/format.jl`), not a fast-path inefficiency.

### 4.5 `OneMMethod`'s B operand needs no new design: it is already `PlanarFormat`, verbatim

`OneMMethod`'s descriptor is `ComplexKernelDescriptor{MR,NR,T,OneEFormat,
PlanarFormat}` (`src/microkernels/onem.jl`, `b_format(OneMMethod()'s descriptor)
== PlanarFormat()`) -- **bit-identical** to `PlanarMethod`'s own B-panel
format (`PlanarFormat`'s own docstring already says the planar/1m panels "are
bit-identical, not merely similar"). `_pack_b!`'s dispatch (Section 5) keys
off the *format* type parameter, not the *method*, so once Section 4.3's
`PlanarFormat` pack fast path ships, `OneMMethod`'s B-side packing is already
fast with **zero additional code** -- it is the same function, the same
eligibility gate, called with the same format value. This is worth stating
explicitly rather than leaving implicit, because it means the actual new
surface area for `OneEFormat`/`OneMMethod` packing is Section 4.4's A-side
design alone, not "packing for a second method" -- roughly half the work
Section 4.4's own framing might otherwise suggest.

## 5. Shared primitives and where they slot into existing dispatch

Both parts need the same two building blocks, which should be written once
(a natural place: a new `src/kernels/planar_simd.jl` or a clearly-marked
section of `planar.jl`, not duplicated):

- **`_complex_vector_eligible(tile::QSTile, ::Type{T})`**: literally
  `_vector_store_eligible(tile, real(T))`'s *shape* but checked against
  `DenseVector{T}` for the complex `T`, i.e. `_unit_stride_rows(tile.rows)
  && tile.storage isa DenseVector{T}`. One predicate, reused by both the
  store gate (Section 3, on the destination) and the pack gate (Section 4,
  on the source) -- the two eligibility conditions are the same condition
  applied to different tiles, and should be visibly the same function, not
  two copies that could drift.
- **Interleave/deinterleave via `SIMD.shufflevector`**, as derived in 3.3/4.3.

**`OneEFormat`/`OneMMethod` need two DIFFERENT primitives, not the above
two** -- this is worth stating plainly rather than forcing a false shared
abstraction across all four fast paths:

- **Pair-swap-and-signed-negate** (Section 4.4): a fixed compile-time
  `shufflevector` pair-swap pattern plus a fixed alternating `[-1,+1,...]`
  constant-vector multiply, picked by a compile-time `Val{transform}` between
  the two operand planes. Shares the *mechanism* (`shufflevector` with a
  literal index tuple) with Section 3.3/4.3's interleave/deinterleave, but is
  a genuinely different pattern (pair-swap, not full deinterleave) applied for
  a different reason (format conversion, not plane separation) -- do not
  attempt to express it as a call to the same helper function as the
  Planar-side interleave; write it as its own named primitive
  (`_oneE_pack_shuffle`-shaped) even if it lives in the same file.
- **Interleaved-complex-by-scalar multiply** (Section 3.4): the
  shuffle+alternating-sign-vector `fmaddsub`-style technique Section 3.4
  derives and flags as needing its own from-scratch correctness check. This
  is not a variant of the eligibility predicate or the interleave shuffle --
  it is new arithmetic, and per Section 3.4 must not be assumed correct by
  analogy to `_accumulate_step_planar`'s four-real-FMA grouping.

`_complex_vector_eligible` above still applies unchanged as the eligibility
gate for `OneMKernel`'s store (same destination-tile shape question,
independent of which arithmetic technique fires once eligible) and for
`OneEFormat`'s pack (same source-tile shape question). Only the *arithmetic
and shuffle pattern* differ per format/method, not the *eligibility test*.

**Where each slots into existing dispatch (no `_execute_nest!` change
needed for either):**

- Packing: exactly the branch `_pack_a!`/`_pack_b!` for
  `ComplexKernelDescriptor` already has one arm of
  (`src/packing/pack.jl`, `:496-508`) -- add an eligibility check before
  the `_pack_panel_complex!` fallback, identical in shape to the real
  path's `_pack_a!` (`:244-261`, `if _pack_a_contiguous_eligible(...) ...
  else _pack_panel!(...)`). `_execute_nest!` calls `pack_a!`/`pack_b!`
  through the same generic reference regardless of kernel type
  (`src/execution/execute.jl`, `pack!` variable), so **no driver change is
  needed** -- this is a leaf-level change entirely inside
  `packing.jl`.
- Store: exactly the branch `store_tile!(destination, acc, alpha, beta,
  kernel::PlanarKernel)` (`src/microkernels/planar.jl`) should gain,
  mirroring the real path's `store_tile!` (`src/microkernels/simd.jl`):
  `if _complex_vector_eligible(destination, T) ... else
  _store_tile_planar!(...) end`. `_execute_micro_tile!`/
  `unsafe_execute_micro_tile!` (`src/execution/macrokernel.jl` and its sibling)
  dispatch to `execute_tile!` generically per kernel type already, so
  **this is also a leaf-level change entirely inside `planar.jl`** -- with
  one exception, Section 6.1.
- `OneEFormat` A-pack: same branch point as Planar's pack above -- the
  `ComplexKernelDescriptor{MR,NR,T,OneEFormat,B}` arm of `_pack_a!`
  (`src/packing/pack.jl`) gains its own eligibility check ahead of the
  `_pack_panel_complex!` fallback, dispatching on `a_format(kernel)` being
  `OneEFormat` rather than `PlanarFormat`. No new call site anywhere else --
  `_pack_a!`'s existing dispatch on the descriptor's format parameter already
  routes correctly; this is Section 4.4's design slotting into machinery
  Section 5 already describes for Planar.
- `OneMKernel`'s store: exactly the branch `store_tile!(destination, acc,
  alpha, beta, kernel::OneMKernel)` (`src/microkernels/onem.jl`) gains an
  eligibility check ahead of `_store_tile_onem!`, mirroring both of the above.
  Same "no driver change needed" property: `execute_tile!` for `OneMKernel`
  (`:320-329`) already calls the generic `store_tile!` name.

## 6. Interactions and risks

### 6.1 The store fast path reopens a question `_prefer_swap` currently closes for complex, and this is the real design risk

`src/planning/labels.jl`:

```julia
# Complex kernels (`PlanarKernel`/`OneMKernel`) always scatter-store --
# `_vector_store_eligible` only exists on the real path -- so there is
# nothing for the swap to win there, and it measurably loses the as-is
# orientation's N-side locality instead (~2-4%, `ccsd_t_3`, ComplexF64/32).
# Real kernels (`SIMDKernel` and, for this run-length rule, `ScalarKernel`
# too) keep the swap.
if T <: Real && _prefer_swap(...)
```

This is the single most important finding of this document's research pass,
and the investigation that recommended this proposal did not surface it,
because it is not visible from profiling -- only from reading why
`_prefer_swap` is gated the way it is. **The moment a complex kernel *can*
fast-store, "there is nothing for the swap to win there" stops being true.**
`_prefer_swap`'s real-path logic exists to trade `C`'s N-side memory
locality against `mr(kernel)`-driven M-side vectorized-store eligibility;
once `PlanarKernel` has an M-side vectorized store too, the same tradeoff
this rule already encodes for real `T` becomes live for complex `T`, and the
`T <: Real &&` guard becomes a real, silent missed-optimization rather than
a "there's nothing to gain here" no-op. Concretely: after this proposal's
store fast path ships, some complex contractions that keep the as-is
orientation today (because swapping cannot help a scatter-store) may in fact
be better served by swapping, if that gives the fast-store predicate a
unit-stride M composite it does not have as-is.

**This must be an explicit decision (Section 8, item 4), not something the
implementation quietly resolves either way.** Two honest options: (a) ship
the store fast path with the `T <: Real` guard on `_prefer_swap` left
exactly as is -- correct and safe, strictly additive, but leaves exactly the
kind of unmeasured gap this proposal exists to close; or (b) extend
`_prefer_swap`'s gate to complex kernels once the fast-store predicate
exists, which requires re-deriving `_prefer_swap`'s cost model for the
complex case (its current thresholds, `src/planning/labels.jl`, were tuned
against real vectorized-store eligibility and register-tile shapes; complex
tiles have a different `mr`/`W` relationship at every shipped shape, Section
6.2) and re-measuring the `~2-4%` `ccsd_t_3` regression this document quotes
to see whether it inverts. Option (b) is a second, separable proposal-sized
piece of work, not an increment of this one; this document's Section 8 asks
only whether to authorize it as an explicit follow-up once (a) has shipped
and been measured, mirroring how `dispatch-tiers.md` staged its own
conditional Section 6 behind a gate rather than building it inline.

### 6.2 Register pressure and shape coverage across the complex menus

The store fast path's `W`-wide vector operations must exist at every shipped
complex `(MR, NR, W)` combination, and unlike the real path (two menus,
`KERNEL_SHAPES_F64`/`_F32`, `src/planning/kernel_selection.jl`) there are two complex
menus (`KERNEL_SHAPES_C64_PLANAR`, `KERNEL_SHAPES_C32_PLANAR`,
`:424-430`) plus per-ISA overrides (`:313-347`) and a `_legacy_shape`
fallback (`:292-297`, `(8,6,W)` on unmeasured ISAs). Concretely, on
`ccqlin038` (`:avx512`) the shipped default is `(24,3,8)` for `ComplexF64`
(`MV = 3`) and `(48,3,16)` for `ComplexF32` (`MV = 3`); on `:neon`,
`(4,6,2)`/`(8,6,4)` (`MV = 2`); on `:avx2`, `(4,5,4)`/`(8,5,8)` (`MV = 1`).
Every one of these must be exercised under `test/forced_isa_runner.jl`
(`avx512`/`avx2`/`neon`/`unknown`, same discipline the per-call-floor and
Track A fixes used this session), because the store fast path's interleave
shuffle pattern is a function of `W` and must be derived from
`kernel_shapes(T, PlanarMethod())`/`lanewidth(kernel)` at each
specialization, never hardcoded to the AVX-512 shape this document was
written against -- exactly the F2 review's blocking finding from
`dispatch-tiers.md` (Section 3.1 there), applied here to a new file.

Register pressure itself is not expected to change: the fast path replaces
a scalar loop over already-live `revec`/`imvec` with a vector operation over
the same two values, plus (when `beta != 0`) one additional loaded
`Vec{2W,real(T)}` per full block, live only briefly before its deinterleave
-- no new accumulator-resident state, so `planar_register_pressure`'s
existing formula (`src/microkernels/planar.jl`) and the Cliff A analysis
built on it are unaffected. This should be confirmed, not assumed, by the
same `@code_native` spot-check the file already asks for its FMA count.

### 6.3 `GC.@preserve` and pointer scope

`_pack_a_contiguous!`'s pointer arithmetic is already wrapped in
`GC.@preserve storage begin ... end` (`src/packing/pack_contiguous.jl`); a
`reinterpret`-based store/pack fast path reading/writing through raw
`Ptr{real(T)}` arithmetic on `destination.storage`/`source.storage` needs
the identical discipline, scoped to whichever buffer the fast path
constructs a raw pointer into. This is a known, already-solved pattern in
this codebase (same section, same file), not a new risk -- it is listed here
only so an implementation does not skip it by treating the store side as
"just like the real path's `_store_tile_vector!`", which uses `SIMD.jl`'s
array-based `vload`/`vstore(storage, at)` overloads (bounds-implicit,
`storage`-relative) rather than raw pointers, and therefore needs no
`GC.@preserve` of its own -- the complex fast path's `reinterpret` step
forces a move to the raw-pointer form, and with it the obligation the real
packer already discharges correctly.

### 6.4 Numerical correctness

- **`beta` applied once, per element**: unchanged in effect -- the fast
  path computes the same `alpha*r + beta*C_old` exactly once per output
  element, just batched across `W` lanes; `_store_prologue!`'s
  `alpha == 0`/short-circuit and the existing beta-zero/-one branch survive
  as a per-call (not per-lane) decision exactly as today.
- **Values**: not expected to be bitwise identical to the scalar path,
  because the two implementations group the four real multiplies of the
  complex product differently (the FMA-chain guardrail in Section 3.3 fixes
  *a* correct grouping, and there is no requirement it matches
  `_axpby_tile!`'s scalar grouping) -- compare with a tolerance, the same
  discipline `Base.accumulate`'s own docstring already states for planar
  ("Not bitwise identical to a scalar complex dot product ... compare with a
  tolerance, never `==`", `src/microkernels/planar.jl`).
- **Padding**: `store_tile!`'s existing `m`/`n` guards (`_store_prologue!`)
  must still gate which rows/columns the fast path is even offered for --
  the fast path's own "full block" test (`(v+1)*W <= m`, mirroring
  `_store_tile_vector!`'s literal condition) keeps a straddling block on the
  scalar tail, exactly as the real path does, so no padding lane is ever
  read through the vectorized branch.
- **Conjugation**: the store side never applies `conj` (that is
  `atransform`/`transform`, a *packing*-time concept, `src/execution/execute.jl`); the
  packing side's `conj` handling is Section 4.3's point 3, a plane-only
  negate, and must be verified against the padding contract (`:377-380`
  padding is a literal zero, never `-0.0`, on the fast path exactly as
  `_pack_emit_zero!` already guarantees on the scalar one).
- **Adapter invariant**: `QuasiStridedBackend` neither knows nor needs to
  know about either fast path; eligibility never causes a rejection or a
  fallback to a different backend, only a different internal path. The
  hard-reject contract at `src/integrations/tensoroperations.jl` is untouched.

### 6.5 Build order across all four pieces, and why `OneMKernel`'s store goes last

Sections 3.4/4.4/4.5 change the relative-difficulty picture Section 1
originally drew for two pieces into a four-piece picture:

1. **`PlanarFormat` pack (Section 4.3)** -- easiest of the two default-path
   pieces, established technique (deinterleave + optional plane negate).
2. **`OneEFormat` A-pack (Section 4.4)** -- comparable to, arguably simpler
   than, (1) for its `identity`-transform case (a pure copy); B-side needs
   nothing new (Section 4.5). Recommended immediately alongside (1), since it
   reuses (1)'s eligibility-gate machinery and shuffle mechanism, and is
   isolated packing-only work with no interaction with Section 6.1's
   `_prefer_swap` question (packing has no orientation to swap).
3. **`PlanarFormat` store (Section 3.3)** -- the bigger default-path win,
   using an established local technique (`_accumulate_step_planar`'s
   four-real-FMA grouping, extended to a new call site) but carrying the real
   design risk this document has: the `_prefer_swap` interaction (Section
   6.1).
4. **`OneMKernel` store (Section 3.4)** -- build **last**, after (3) has
   shipped and been measured, for three independent reasons, any one of which
   would already justify the ordering: (a) it needs a technique novel to this
   codebase and flagged with suspicion elsewhere in it (Section 3.4's
   `fmaddsub`-family finding); (b) its practical value today is narrower than
   any other piece -- it cannot move a single default-path number, only
   `bench_complex_efficiency.jl`'s arm-2 method comparison (Section 3.4's
   value-proposition paragraph); (c) building (3) first gives (4) a working,
   tested reference implementation of "a vectorized complex store with a
   correct `alpha`/`beta`/padding contract" to compare against during its own
   from-scratch arithmetic derivation, lowering (4)'s risk for free.

This reorders Decision 1's original packing-then-store framing into
**pack (both formats) first, `PlanarFormat` store second, `OneMKernel` store
last** -- an extension of, not a reversal of, the already-accepted Decision 1,
and is called out as its own confirmation point in Section 8 rather than
silently assumed.

## 7. Verification plan

1. **Oracle/contract tests**, mirroring the existing `test/
   test_planar_kernel.jl`/`test/packing/test_pack_complex.jl` structure (both
   already test the scalar paths this proposal adds a second path
   alongside, at `:263-295` and `:367-433` for the alpha/beta contract and
   NaN-poisoning, respectively): re-run every existing `store_tile!`/
   `pack_a!`/`pack_b!` contract test (alpha/beta shortcuts, `kc == 0`,
   padded-lane isolation, scattered/negative-stride agreement) on both the
   scalar and the new fast path, for every complex dtype and every shipped
   `(MR,NR,W)`. The scattered/negative-stride cases exist specifically to
   assert the *scalar* path still fires correctly when the fast path's
   eligibility gate excludes them -- these tests must not merely keep
   passing, they must be extended to assert *which* path fired (mirroring
   `dispatch-tiers.md` Section 6.8's "gate-firing tests" lesson from the D3
   review: assert eligible on a fixture that should fast-path, assert
   ineligible on each single violated condition -- permuted rows, scattered
   storage, a tail block, `beta` in each of its three regimes).
2. **Interleave/deinterleave shuffle correctness**: a direct unit test
   comparing `shufflevector`-based interleave/deinterleave against a
   reference scalar loop, at every shipped `W` (2, 4, 8, 16), independent
   of the kernel -- this isolates "is the shuffle pattern right" from "is
   the kernel plumbing right", so a failure in one is not misdiagnosed as
   the other.
3. **FMA-grouping spot check**: `@code_native`/`@code_llvm` on the fast
   store's arithmetic at every shipped shape, confirming the guardrail
   grouping from Section 3.3 compiles to the fused form (no stray `vmulsd`+
   `vsubsd` pair where a `vfnmadd`/`vfmadd` was intended) -- the same
   discipline `_accumulate_step_planar`'s header comment already requires
   of itself (`src/microkernels/planar.jl`).
4. **Zero allocation**: `@allocated` through `execute!` on an eligible
   complex plan, both dtypes, mirroring `test/microkernels/test_planar_kernel.jl`'s
   existing Cliff B check -- the new code path must not reintroduce a
   heap-allocated accumulator or a boxed closure.
5. **Forced-ISA runs**: `test/forced_isa_runner.jl` under `avx512`, `avx2`,
   `neon`, `unknown`, per Section 6.2 -- every derived expectation (`mr`,
   `W`, which shuffle pattern) must come from `kernel_shapes`/`lanewidth`
   at runtime, never a literal.
6. **`bench_complex_efficiency.jl` re-run**: the headline metric from
   Section 2, before/after each part ships, on `ccqlin038` -- expect the
   `ComplexF32` geomean (currently 0.690, below the script's own 0.9
   warning threshold) to move, and record whether it clears the threshold.
   This is the project's own standing instrument for exactly this
   question and should be the acceptance number quoted in the eventual
   `docs/decisions.md` entry, not a one-off case's GFLOP/s.
7. **Re-profile the two triggering cases**: `julia --project=benchmark
   benchmark/profile_to_suite.jl ccsd_t_1_dim16_c64 ccsd_6_dim16_c64
   --tag fast-paths-post` (both cases already added to `profile_to_suite.jl`
   by the investigation this proposal follows up on) -- expect
   `ccsd_t_1_dim16_c64`'s store share to fall from 60.8% toward its Float64
   sibling's 16.1% after Part 1, and `ccsd_6_dim16_c64`'s packing share to
   fall from 40.4% toward its Float64 sibling's 31.0% after Part 2. A
   shift that does not land near those reference points is a finding to
   report, not a target to force.
8. **No ABBA guard exists for the complex path today**
   (`benchmark/bench_real_path_guard.jl` is real-`T`-only by name and by
   its own header). Either extend it to a complex two-tree comparison or
   accept `bench_complex_efficiency.jl`'s before/after geomean (item 6) as
   the regression signal for this change -- this is one of the decisions
   requested in Section 8, since building a genuine ABBA guard is itself
   nontrivial extra scope.
9. Full test suite (`Pkg.test()`) and, per this project's standing
   discipline, Julia 1.10 LTS in CI (the `NTuple` accumulator's fragility
   there is a standing concern noted throughout `docs/decisions.md`).
10. **`OneEFormat`/`OneMMethod`'s own measurement path, since it is invisible
    to items 6-7 above.** Section 3.4's value-proposition paragraph already
    establishes that `OneMMethod` is never selected by `:tccg` or any default
    kernel, so neither `bench_complex_efficiency.jl`'s headline geomean (arm
    1, `time_default`, `PlanarMethod`-only) nor the `ccsd_*` re-profiling in
    item 7 will show ANY movement from `OneEFormat` pack or `OneMKernel`
    store work, by construction -- their absence is not evidence the change
    did nothing. The actual signal is `bench_complex_efficiency.jl`'s **arm
    2** (`arm_methods`, already exercises every shipped `(MR,NR,W)` for both
    `PlanarMethod()` and `OneMMethod()` at `MAIN_SHAPES`/`SMALL_SHAPES`,
    `benchmark/bench_complex_efficiency.jl:120-160`ish) -- read its per-shape
    `OneMMethod` GFLOP/s rows before/after, not a single geomean (arm 2 does
    not currently compute one across methods/shapes the way arm 1 does across
    dtypes). Decision 4 (accepting `bench_complex_efficiency.jl` as the
    verification budget) already covers this arm; this item exists so the
    *absence* of movement in arm 1 during `OneEFormat`/`OneM` work is not
    mistaken for a failed fix.

## 8. Decisions requested from the user

This is a sign-off gate. **No `src/` code should be written on the basis of
this document without an explicit go-ahead on the items below.**

1. **Authorize the work at all**, and in the order Section 1 recommends:
   complex packing (Section 4) first, planar store (Section 3) second --
   not the reverse, even though the store path is the larger measured win,
   because packing exercises the shared interleave/deinterleave primitives
   (Section 5) on the strictly easier problem first. If you would rather
   take the bigger win first and accept the higher risk up front, say so
   explicitly; it is a defensible choice, just not this document's default.
2. **Scope confirmation**: `PlanarFormat` only (Section 4.4), i.e. the
   default `PlanarMethod()` complex kernels -- `OneEFormat`/`OneMMethod`
   gets no fast path from this work. Confirm this is acceptable, or ask for
   `OneEFormat` to be scoped in now (which Section 4.4 argues is
   substantially harder and would be its own design pass).
3. **The `_prefer_swap` question (Section 6.1)**: ship the store fast path
   with `_prefer_swap`'s `T <: Real` guard left exactly as is (strictly
   additive, but leaves an unmeasured gap for complex orientations that
   could now benefit from swapping), or authorize extending the swap
   decision to complex kernels as an explicit, separate follow-up once the
   store fast path has shipped and been measured. Do not authorize doing
   both in one change.
4. **Verification budget (Section 7, item 8)**: build a genuine
   complex-path ABBA guard (real extra scope, mirroring
   `bench_real_path_guard.jl`), or accept `bench_complex_efficiency.jl`'s
   before/after geomean as the project's regression signal for this
   specific change. 
5. **ISA coverage for the first pass**: all four shipped complex menus
   (`avx512`/`avx2`/`neon`/`unknown`-legacy) from the start, or `avx512`
   first (the only ISA with a *measured*, as opposed to modelled or
   pinned-by-fit, complex shape -- Section 6.2) with the others gated
   behind forced-ISA test coverage before being declared supported rather
   than merely "does not crash".
6. **Record**: whether `docs/decisions.md`'s "ComplexF64 `:tccg` slowdown"
   entry should gain a forward pointer to this document (mirroring how
   `dispatch-tiers.md` was cross-referenced from STATUS.md), and whether
   `STATUS.md` should list this as the current highest-value complex-path
   item.

### Decisions recorded (2026-09-22)

1. **Order: packing first, store second.** Accepted as recommended.
2. **Scope: `OneEFormat`/`OneMMethod` is IN scope, not deferred.** This
   overrides Section 4.4's default recommendation -- that section only
   argued why `OneEFormat` was *excluded*, it does not yet contain a design
   for including it (the doubled-footprint, two-K-steps-per-source-element
   layout in `_pack_emit!` for `OneEFormat`, `:365-375`, does not reduce to
   the same "contiguous run in, deinterleave, two contiguous runs out" shape
   `PlanarFormat` gets). Before any `OneEFormat` implementation work starts,
   this document needs an actual `OneEFormat` design added (mirroring
   Sections 3/4's treatment of `PlanarFormat`), not just the exclusion
   rationale it currently has -- treat that as a required precursor task,
   not something to improvise during implementation.
3. **`_prefer_swap` (Section 6.1): authorized as a follow-up, not bundled.**
   Per the item's own "do not authorize doing both in one change" framing:
   ship the store fast path first, measure it, and only then extend
   `_prefer_swap`'s guard to complex kernels as a separate, subsequent
   change with its own before/after measurement.
4. **Verification budget: `bench_complex_efficiency.jl`'s before/after
   geomean accepted as the regression signal.** No new dedicated
   complex-path ABBA guard will be built for this work.
5. **ISA coverage: `avx512` first.** The only ISA with a *measured* (not
   modelled or pinned-by-fit) complex shape (Section 6.2). `avx2`/`neon`/
   `unknown`-legacy remain gated behind forced-ISA test coverage before
   being declared supported, per Section 7 -- not shipped silently as a
   side effect of a generic-looking eligibility predicate.

## Appendix. Files and lines this document relies on (`main` @ `f7fa490`)

- `src/microkernels/planar.jl`: file-header "no shuffles, no `fmaddsub`" guardrail
  1-5; `PlanarKernel` 45-55; plane-offset forwarding 90-93;
  `planar_register_pressure` 96-118; `zero_accumulator` 124-140;
  `_accumulate_step_planar` (FMA-grouping guardrail) 142-238;
  `Base.accumulate` 253-263; `_store_tile_planar!` 269-329 (deferred-fast-path
  docstring 279-280); `store_tile!` 331-352; `execute_tile!` 354-372.
- `src/microkernels/onem.jl`: `OneMKernel` struct and even-`W`/`2*mr` constructor
  guardrails 60-98; `complex_method`/`lanewidth`/`avecs_per_column` 113-129;
  `onem_register_pressure` 144-166; `zero_accumulator`/`accumulate` (verbatim
  delegation to the real kernel) 172-208; `_store_tile_onem!`
  (adjacency comment, deferred-fast-path header) 214-279; `store_tile!`
  281-308; `execute_tile!` 310-329.
- `src/microkernels/simd.jl`: `_unit_stride_rows` 145-147; `_vector_store_eligible`
  162-181; `_store_tile_scattered!` 194-222; `_store_tile_vector!` 224-301;
  `store_tile!` 303-326.
- `src/packing/pack.jl`: `_pack_panel!` 106-140; `_copies_unchanged` 147-149;
  `_pack_a_contiguous_eligible` 151-163; `_pack_a_contiguous!` 165-192; real
  `_pack_a!`/`_pack_b!` 244-261, 296-308; complex packing section header
  310-334; `_pack_emit!` (`PlanarFormat`) 342-349, (`OneEFormat`) 351-375;
  `_pack_emit_zero!` 377-397; `_pack_panel_complex!` 399-417; complex
  `_pack_a!`/`_pack_b!` 454-466, 496-508.
- `src/packing/format.jl`: `PackFormat`/`RealFormat`/`PlanarFormat`/
  `OneEFormat` 18-68 (`reals_per_element` 66-68); `ComplexMethod`/`RealMethod`/
  `PlanarMethod` ("no shuffles, no `fmaddsub`" docstring, 99-105)/`OneMMethod`
  70-131 (`a_reals`/`b_reals` 126-131); `ComplexKernelDescriptor` 146-232;
  `packed_a_plane_offset`/`packed_b_plane_offset` 244-261.
- `src/microkernels/interface.jl`: `_axpby_tile!`/`_axpby_at!` 134-147; `_store_prologue!`
  156-166.
- `src/layout/tiles.jl`: `QSTile` 120-133; `tile_offset`/`tile_load`/`tile_store!`
  158-208.
- `src/packing/panel.jl`: `PackedPanel` 9-20; `panel_vload`/`panel_load`/
  `panel_store!` 34-47.
- `src/planning/labels.jl`: `_prefer_swap` gate and complex-scatter-store note
  1048-1054; `_demote_for_run` real-only gate 267-282; complex shape
  overrides 292-347; `_derived_shape` (complex) 385-399; complex kernel
  menus 405-431 (`KERNEL_SHAPES_C64_ONEM`/`_C32_ONEM` 427-431);
  `_default_complex_method` 602-604; `_kernel_for`/`_default_kernel` (complex)
  606-636; `OneMMethod` named-kernel construction 506-524.
- Tests: `test/microkernels/test_planar_kernel.jl` (alpha/beta contract 263-295,
  padded-lane isolation 367-433, Cliff B zero-alloc 434-483);
  `test/packing/test_pack_complex.jl`; `test/forced_isa_runner.jl`.
- Benchmarks: `benchmark/bench_complex_efficiency.jl` (arm 1/`time_default`,
  `PlanarMethod`-only headline geomean, re-run in Section 2, lines ~46-53;
  arm 2/`arm_methods`, exercises both `PlanarMethod`/`OneMMethod` across all
  shipped shapes at plain-matmul shapes, lines ~120-160, the verification
  path for Section 3.4/4.4/4.5's work per Section 7 item 10);
  `benchmark/profile_to_suite.jl`/`benchmark/profile_buckets.jl`;
  `benchmark/bench_real_path_guard.jl` (real-`T`-only today, Section 7 item 8).
- `docs/decisions.md`: "ComplexF64 `:tccg` slowdown: root-caused to the
  planar store path, no fix shipped (2026-09-22)" (this document's
  trigger, quoted in full in Section 2); "Method ranking does not transfer
  between machines"; "Cliff A bites at the shipped shape"; "The planar
  microkernel: accumulator, body, and two independent cliffs"; "Complex
  element-type milestone".
- `docs/proposals/dispatch-tiers.md`: structural template for this document
  and the direct source of the "gate-firing tests" and staged-conditional-
  design conventions reused in Sections 6.1 and 7.
- `SIMD.jl` v3 (`~/.julia/packages/SIMD/UiGbs/src/`): `shufflevector`
  `simdvec.jl:504-509`.
