# Label planning: classify every label into M/N/K, order the free labels
# by their stride in C, and decide the M/N operand orientation.

# Membership test against a statically-sized label tuple. Replaces the three
# `Set`s `_classify_labels` used to build: the label tuples have a
# compile-time-known LENGTH (the `NA`/`NB`/`NC` parameters, one specialization
# per arity), so this unrolls into a chain of integer compares and allocates
# nothing, where each `Set` cost a `Dict`'s slot/key arrays. Measured on
# ccqlin038 / Julia 1.13: `_classify_labels` 1232 -> 224 B and 0.73 -> 0.25 us
# on a 2-label plain GEMM (docs/decisions.md, "Per-call floor"). The tuples are
# `allunique` by the checks at the top of `_classify_labels`, so a linear scan
# is also the whole of the membership question.
@inline _label_in(lbl::Int, t::NTuple{N, Int}) where {N} = any(==(lbl), t)

# Classify every label in indA ∪ indB ∪ indC into M/N/K. Returns
# (mlabels, nlabels, klabels) in indA/indB appearance order. Per (inA,inB,inC):
#   (T,F,T)->M  (F,T,T)->N  (T,T,F)->K  everything else -> ArgumentError
# (labeled in C only, present in all three, or dangling in just A or B).
function _classify_labels(
        indA::NTuple{NA, Int}, indB::NTuple{NB, Int},
        indC::NTuple{NC, Int}
    ) where {NA, NB, NC}
    allunique(indA) ||
        throw(ArgumentError("indA has a repeated label (diagonal), not supported: $indA"))
    allunique(indB) ||
        throw(ArgumentError("indB has a repeated label (diagonal), not supported: $indB"))
    allunique(indC) ||
        throw(ArgumentError("indC has a repeated label (diagonal), not supported: $indC"))

    # Sized once to their worst case and trimmed at the end, rather than grown
    # by `push!`: `NA`/`NB` are compile-time bounds on the M+K and N counts.
    mlabels = Vector{Int}(undef, NA)
    klabels = Vector{Int}(undef, NA)
    nm = 0
    nk = 0
    for lbl in indA
        inB = _label_in(lbl, indB)
        inC = _label_in(lbl, indC)
        if inB && inC
            throw(
                ArgumentError(
                    "label $lbl appears in indA, indB, and indC: labels present in all " *
                        "three operands (batch-like) are out of scope for this milestone"
                )
            )
        elseif inB && !inC
            nk += 1
            @inbounds klabels[nk] = lbl
        elseif !inB && inC
            nm += 1
            @inbounds mlabels[nm] = lbl
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indA (not in indB or indC): not a valid " *
                        "free (M) or contracted (K) label"
                )
            )
        end
    end

    nlabels = Vector{Int}(undef, NB)
    nn = 0
    for lbl in indB
        inA = _label_in(lbl, indA)
        inC = _label_in(lbl, indC)
        if inA && inC
            continue  # already rejected while scanning indA, above.
        elseif inA && !inC
            continue  # already classified as K, above.
        elseif !inA && inC
            nn += 1
            @inbounds nlabels[nn] = lbl
        else
            throw(
                ArgumentError(
                    "label $lbl appears only in indB (not in indA or indC): not a valid " *
                        "free (N) or contracted (K) label"
                )
            )
        end
    end

    for lbl in indC
        inA = _label_in(lbl, indA)
        inB = _label_in(lbl, indB)
        (inA || inB) ||
            throw(ArgumentError("label $lbl appears in indC but not in indA or indB"))
    end

    resize!(mlabels, nm)
    resize!(nlabels, nn)
    resize!(klabels, nk)
    return mlabels, nlabels, klabels
end

@noinline _throw_label_length(lbl::Int, l1::Int, l2::Int) = throw(
    DimensionMismatch("label $lbl has mismatched axis length: $l1 vs $l2")
)

# ----------------------------------------------------------------------------
# Free-label order and M/N orientation (docs/decisions.md, "Label-order
# milestone"). `_classify_labels` lists free labels in A's/B's own axis order,
# which is incidental to C: `fill_offsets!` enumerates a composite with its
# FIRST label fastest, so that order fixes the store loop's walk through C.
# Both helpers below are pure planning-time functions of (labels, indC, C).
# ----------------------------------------------------------------------------

# Stable sort of `labels` by `abs(stride)` of each label's axis in C,
# ascending; ties keep input order, so a single label or an already-sorted list
# comes back unchanged. Every label must occur in `indC` (the M/N lists from
# `_classify_labels` do by construction; K labels never come here).
# Insertion sort rather than `sortperm` + permuted copy: the old body
# allocated the key vector, the permutation and the result (three `Vector`s
# where one is needed), and these lists have at most `ndims(C)` entries, so an
# O(n^2) sort with n <= 6 is not a cost. Strict `>` in the shift test keeps it
# STABLE, which is the contract (ties keep input order) that
# `alg = DEFAULT_STABLE` supplied before. A fresh vector is still returned:
# sorting `labels` in place would mutate `_classify_labels`'s output, which
# callers (and test/test_driver.jl's label-order pinning) read afterwards.
function _order_free_labels(
        labels::Vector{Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {NC}
    st = Base.strides(C)
    key(l::Int) = abs(st[findfirst(==(l), indC)::Int])
    out = copy(labels)
    @inbounds for i in 2:length(out)
        x = out[i]
        kx = key(x)
        j = i - 1
        while j >= 1 && key(out[j]) > kx
            out[j + 1] = out[j]
            j -= 1
        end
        out[j + 1] = x
    end
    return out
end

# Element count of the leading unit-stride run when `labels` (already ordered
# by `_order_free_labels`) is enumerated first-label-fastest into C: the first
# non-singleton label must have C-stride exactly +1 (`_unit_stride_rows` is
# `stride == 1`, a descending run does not qualify), and each following label
# extends the run only if its stride equals the run so far. Singleton axes are
# skipped (their coordinate never advances, whatever their stride says).
# Returns 1 when no run starts, 0 if an empty axis is met first.
function _leading_unit_run(
        labels::Vector{Int}, indC::NTuple{NC, Int}, C::StridedView
    ) where {NC}
    st = Base.strides(C)
    run = 1
    for l in labels
        p = findfirst(==(l), indC)::Int
        L = size(C, p)
        L == 1 && continue
        L == 0 && return 0
        st[p] == run || break
        run *= L
    end
    return run
end

# Whether to swap the operand roles (B feeds M, A feeds N), given the two
# composites' OWN leading unit-stride run lengths (`_leading_unit_run` above
# -- the one and only place that quantity is computed; `plan_contract` derives
# `run_m`/`run_n` once and reuses them here and at both `_demote_for_run` call
# sites, rather than recomputing per call site as an earlier revision did).
# The vectorized store (`_vector_store_eligible`) needs a register sliver --
# `mr(kernel)` consecutive M coordinates -- to be unit-stride in C, so a
# leading run shorter than `mr` buys nothing (measured: swapping onto a
# 16-wide run under a 32-wide kernel is a ~1.2x REGRESSION). Swap only when the
# as-is orientation misses that bar and the swapped one clears it. The two `mr`
# arguments are the widths of the kernel each orientation would actually run
# (they differ only when the default kernel's small-Qm demotion applies to one
# side).
#
# Callers must additionally restrict this to real dtypes -- measured directly
# (`ccsd_t_3`, dim=16, both complex dtypes): the swap is a ~2-4% regression
# there (loses the as-is orientation's N-side locality for no store-side
# gain), back when `PlanarKernel`/`OneMKernel` (complex) shipped only a
# scattered/scalar store. As of the planar vectorized store fast path
# (`_store_tile_planar_vector!`, `src/kernels/planar.jl`), that measurement is
# STALE: there is now a vector store for the swap to potentially win on the
# complex path too. The `T <: Real` guard below is a DELIBERATELY DEFERRED,
# UNMEASURED follow-up, not a settled "moot" case -- per
# docs/proposals/complex-fast-paths.md Decision 3, extending `_prefer_swap` to
# complex was explicitly scoped out of this change to ship the store fast path
# first and measure it, with any extension here to be a separate later change
# with its own before/after measurement. See the `T <: Real` guard at the
# call site.
function _prefer_swap(run_m::Int, run_n::Int, mr_asis::Int, mr_swapped::Int = mr_asis)
    return run_m < mr_asis && run_n >= mr_swapped
end

# Label-list form, kept for its existing external test coverage and for any
# caller that has `morder`/`norder` but not their run lengths in hand; derives
# the same two run lengths `plan_contract` itself now derives once and passes
# to the method above directly.
function _prefer_swap(
        morder::Vector{Int}, norder::Vector{Int}, indC::NTuple{NC, Int}, C::StridedView,
        mr_asis::Int, mr_swapped::Int = mr_asis
    ) where {NC}
    return _prefer_swap(
        _leading_unit_run(morder, indC, C), _leading_unit_run(norder, indC, C),
        mr_asis, mr_swapped
    )
end
