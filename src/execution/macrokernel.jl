# Macro-kernel helpers for the five-loop nest: function barriers over the
# tile axis types, packed-sliver addressing, and sliver classification.

# GUARDRAIL, load-bearing: `_axis_of` returns a `Union{AffineAxis,PtrScatterAxis}`, and each
# consumer below is a `where {R<:Axis, C<:Axis}` barrier method that Julia
# specializes per concrete (R,C), so no partially-applied -- heap-boxed --
# `QSTile` is ever built. **Do not** collapse these helpers into their call
# sites, and do not let a union cross any other boundary.
# Both arms are isbits, so this Union itself needs no heap box (see
# PtrScatterAxis).
@inline function _axis_of(d::BlockDescriptor, buffer::Vector{Int}, first::Int)
    return d.regular ? AffineAxis(d.base, d.stride, d.count) :
        PtrScatterAxis(pointer(buffer, first + 1), d.count)
end

# `pack!` is pack_a!/pack_b! -- or their `unsafe_pack_a!`/`unsafe_pack_b!`
# siblings (src/packing/pack.jl) -- as a plain function, specialized on, never a
# closure; A and B differ only in which of rows/cols is the k axis, which the
# caller has already resolved. `transform` is the plan's per-operand
# `identity`/`conj` singleton.
#
# This helper is only as safe as the `pack!` it is handed: with an `unsafe_*`
# packer it performs no storage-bounds check, which is why the call sites spell
# that name out rather than hiding it behind a flag.
#
# GUARDRAIL: every argument here has its OWN bound type parameter, `transform`
# included. Leaving `TF` unbound costs a dynamic dispatch (~80 B/call), for the
# reason spelled out at `pack_a!` in src/microkernels/interface.jl.
# And all THREE call sites -- `_execute_nest!`'s two and `execute_tilewise!`'s
# one -- must pass the matching operand's transform: missing the third makes
# the in-tree ORACLE silently wrong for conjugated inputs.
@inline function _pack_sliver!(
        pack!::PF, packed::PK, storage::S, base::Int,
        rows::R, cols::C, kernel, transform::TF
    ) where {PF, PK, S, R <: Axis, C <: Axis, TF}
    pack!(packed, SourceTile(storage, base, rows, cols), kernel, transform)
    return nothing
end

@inline function _execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, kc_len::Int, alpha, beta
    ) where {PA, PB, S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    execute_tile!(kernel, destination, packed_a, packed_b, kc_len, alpha, beta)
    return nothing
end

# Same guardrail barrier as `_execute_micro_tile!`, over `unsafe_execute_tile!`
# (src/microkernels/interface.jl) instead of `execute_tile!`: the destination's storage-bounds
# check has already been made ONCE for the whole (ic, jc) macro block this tile
# belongs to. `unsafe_` is in the name at every call site precisely because the
# precondition lives at the caller.
@inline function unsafe_execute_micro_tile!(
        kernel, storage::S, base::Int, rows::R, cols::C,
        packed_a::PA, packed_b::PB, kc_len::Int, alpha, beta
    ) where {PA, PB, S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    # Off by default; folds to nothing (sites `:ctile`/`:ctile_w`, below).
    _ctile_prefetch!(storage, base, rows, cols)
    unsafe_execute_tile!(kernel, destination, packed_a, packed_b, kc_len, alpha, beta)
    return nothing
end

# C micro-tile prefetch (sites `:ctile` -> `prefetcht0`, `:ctile_w` ->
# `prefetchw`; src/hardware/prefetch.jl; EXPERIMENTAL, off by default) -- the
# classic BLIS C prefetch. Issued for the CURRENT tile immediately before its
# microkernel call, i.e. before the K loop: the K loop (`kc_len` rank-1
# updates, hundreds of cycles at the shipped `kc`) is the latency the lines
# have to arrive in before the store phase reads/writes them. The next tile is
# not prefetched: its C axes are only built at the next iteration, and the
# current tile's K loop already hides the latency for the current one.
#
# Every line of the tile, following C's actual layout: when one axis is a
# gap-free affine run (`_dense_lanes`, src/packing/pack.jl -- column-major C's
# rows, or a row-major C's columns), one line range per index of the other
# axis; otherwise (both axes long-stride or block-scattered, offsets from the
# scatter tables) one prefetch per element. With both sites off the body folds
# away (checked in test/execution/test_prefetch.jl).
@inline function _ctile_prefetch!(storage::S, base::Int, rows::R, cols::C) where {S, R <: Axis, C <: Axis}
    RD = _prefetch_distance(Val(:ctile))
    WD = _prefetch_distance(Val(:ctile_w))
    (RD > 0 || WD > 0) || return nothing
    storage isa DenseArray || return nothing
    GC.@preserve storage begin
        WD > 0 && _prefetch_tile_lines!(storage, base, rows, cols, Val(1))
        RD > 0 && _prefetch_tile_lines!(storage, base, rows, cols, Val(0))
    end
    return nothing
end

@inline function _prefetch_tile_lines!(
        storage::S, base::Int, rows::R, cols::C, rw::Val
    ) where {S, R <: Axis, C <: Axis}
    p0 = pointer(storage)
    E = sizeof(eltype(storage))
    m = axis_length(rows)
    n = axis_length(cols)
    (m == 0 || n == 0) && return nothing
    if _dense_lanes(rows, E)
        for j in 0:(n - 1)
            a, b = _lane_bytes(p0, E, base + axis_offset(cols, j), rows, m)
            _prefetch_lines!(a, b, rw)
        end
    elseif _dense_lanes(cols, E)
        for i in 0:(m - 1)
            a, b = _lane_bytes(p0, E, base + axis_offset(rows, i), cols, n)
            _prefetch_lines!(a, b, rw)
        end
    else
        for j in 0:(n - 1), i in 0:(m - 1)
            prefetch(p0 + E * (base + axis_offset(rows, i) + axis_offset(cols, j)), rw, Val(3))
        end
    end
    return nothing
end

@inline function _scale_micro_tile!(
        storage::S, base::Int, rows::R, cols::C, beta
    ) where {S, R <: Axis, C <: Axis}
    destination = DestinationTile(storage, base, rows, cols)
    scale_tile!(destination, beta)
    return nothing
end

# Sliver `s` of a shared packed panel, as a borrowed pointer, at the CURRENT
# block's depth `kc_len` (< the buffer's per-sliver capacity on a tail K block,
# since the buffer is sized for kc_eff). Used by both the packing and the
# consuming step of a (jc,pc,ic) iteration, so the two cannot disagree.
#
# GUARDRAIL: `reg_tile` here is a count of REALS per logical K step --
# `packed_a_per_k`/`packed_b_per_k`, NOT `mr`/`nr`. They coincide for every
# real kernel (pinned in test/planning/test_kernel_selection.jl), so this is the identity on the
# real path; for a complex kernel only the packed count addresses the panel
# correctly.
@inline function _sliver_panel(buffer, reg_tile::Int, kc_len::Int, s::Int)
    stride = reg_tile * kc_len
    return packed_panel(buffer, s * stride + 1, stride)
end

# Panel-ahead software prefetch (site `:macro`, src/hardware/prefetch.jl;
# EXPERIMENTAL, off by default) -- the classic BLIS `a_next`/`b_next` hint.
# Called right before the microkernel consumes A micropanel `r` against B
# micropanel `s`, it prefetches the first `distance` cache lines of the A
# micropanel the NEXT call will read (`r + 1`, wrapping to `0` after the last,
# which is where the next `s` restarts), and on the last `r` also the head of
# B micropanel `s + 1`. The current B micropanel needs nothing: every `r`
# re-reads it, so it is already hot. Heads only, because the microkernel then
# walks each micropanel sequentially and the hardware stream prefetcher takes
# over from there; a whole micropanel would be hundreds of prefetches per
# tile.
#
# `packed_a`/`packed_b` are the workspace's packed buffers; their pointers are
# valid under `execute!`'s `GC.@preserve ws`. Every prefetched address lies inside the
# current block's packed panels (`r' < m_slivers`, `s' < n_slivers`); a line
# past a short micropanel's end is still inside the buffer or, at worst, a
# harmless hint -- a prefetch never faults.
#
# With the site off, `distance` is the literal `0` and the whole body folds
# away (checked in test/hardware/test_prefetch.jl).
const _PREFETCH_LINE_BYTES = 64

@inline function _macro_prefetch!(
        packed_a::VA, packed_b::VB, a_stride::Int, b_stride::Int,
        r::Int, s::Int, m_slivers::Int, n_slivers::Int
    ) where {VA, VB}
    L = _prefetch_distance(Val(:macro))
    L > 0 || return nothing
    # Only a dense buffer has a plain `pointer`; any other workspace vector type
    # simply gets no prefetch. Folds at compile time either way.
    (packed_a isa DenseVector && packed_b isa DenseVector) || return nothing
    R = eltype(packed_a)
    pa = pointer(packed_a)
    pb = pointer(packed_b)
    rnext = r + 1 < m_slivers ? r + 1 : 0
    a_head = pa + sizeof(R) * (rnext * a_stride)
    for l in 0:(L - 1)
        prefetch(a_head + _PREFETCH_LINE_BYTES * l)
    end
    if r + 1 == m_slivers && s + 1 < n_slivers
        b_head = pb + sizeof(R) * ((s + 1) * b_stride)
        for l in 0:(L - 1)
            prefetch(b_head + _PREFETCH_LINE_BYTES * l)
        end
    end
    return nothing
end

# Classify each register sliver of a just-filled macro block. Shared by the
# N side (jc: B/C) and the M side (ic: A/C), which are structurally identical.
#
# Also returns the two maps' BLOCK offset ranges, `((lo1, hi1), (lo2, hi2))`,
# accumulated from the sliver descriptors as they are produced rather than in a
# second pass -- `O(1)` per regular sliver, and for an irregular one exactly
# the scan a per-sliver `checked_tile_storage_bounds` would do.
# Because the slivers partition `buf[1:blocklen]`, this union IS the range of
# the whole block, which is what `_execute_nest!`'s hoisted
# `checked_span_bounds` calls need.
@inline function _classify_slivers!(
        desc1::Vector{BlockDescriptor}, desc2::Vector{BlockDescriptor},
        buf1::Vector{Int}, buf2::Vector{Int},
        blocklen::Int, reg_tile::Int, nslivers::Int
    )
    lo1 = typemax(Int); hi1 = typemin(Int)
    lo2 = typemax(Int); hi2 = typemin(Int)
    for s in 0:(nslivers - 1)
        sfirst = s * reg_tile
        scount = min(reg_tile, blocklen - sfirst)
        d1 = describe_block(buf1, sfirst, scount)
        d2 = describe_block(buf2, sfirst, scount)
        desc1[s + 1] = d1
        desc2[s + 1] = d2
        (l1, h1) = descriptor_offset_range(d1, buf1, sfirst)
        if h1 >= l1
            lo1 = min(lo1, l1); hi1 = max(hi1, h1)
        end
        (l2, h2) = descriptor_offset_range(d2, buf2, sfirst)
        if h2 >= l2
            lo2 = min(lo2, l2); hi2 = max(hi2, h2)
        end
    end
    # `hi < lo` is `checked_span_bounds`'s "empty, always passes" convention,
    # which is what these initial values mean when no sliver contributed.
    return ((lo1, hi1), (lo2, hi2))
end

# ----------------------------------------------------------------------------
# Closed-form block description for an affine-ramp composite. When
# `affine_ramp(g)` holds, logical
# coordinate `q` maps to offset `q * step[p]` for every map `p`, so a block's
# whole sliver structure follows from arithmetic and neither the offset buffer
# nor `describe_block`'s scan is needed. `fill_offsets!` + `_classify_slivers!`
# stay as the fallback for every composite that is not provably a ramp.
# ----------------------------------------------------------------------------

# Exactly what `describe_block` classifies a materialized ramp interval as,
# INCLUDING its `stride == 0` convention for a count-1 block (which is
# behaviourally irrelevant -- an `AffineAxis` of count 1 never multiplies by
# its stride -- but keeping it identical means the descriptors these two paths
# produce are `==`, not merely equivalent, which a test can assert).
@inline _ramp_descriptor(step::Int, first::Int, count::Int) =
    count == 0 ? BlockDescriptor(0, 0, 0, true) :
    count == 1 ? BlockDescriptor(first * step, 0, 1, true) :
    BlockDescriptor(first * step, step, count, true)

# Offset range of `[first, first+count)` under a ramp, in `axis_offset_range`'s
# `(lo, hi)` / `(0, -1)`-if-empty convention. `first * step` and
# `(first+count-1) * step` are offsets of coordinates inside the group's
# domain, so they are covered by `AxisGroup`'s construction-time excursion
# validation and cannot overflow.
@inline _ramp_offset_range(step::Int, first::Int, count::Int) =
    count == 0 ? (0, -1) : minmax(first * step, (first + count - 1) * step)

# `_classify_slivers!`'s closed-form twin: same descriptors, same returned
# block ranges, no buffer touched.
@inline function _ramp_slivers!(
        desc1::Vector{BlockDescriptor}, desc2::Vector{BlockDescriptor},
        step1::Int, step2::Int, first::Int,
        blocklen::Int, reg_tile::Int, nslivers::Int
    )
    for s in 0:(nslivers - 1)
        sfirst = s * reg_tile
        scount = min(reg_tile, blocklen - sfirst)
        q0 = first + sfirst
        desc1[s + 1] = _ramp_descriptor(step1, q0, scount)
        desc2[s + 1] = _ramp_descriptor(step2, q0, scount)
    end
    return (
        _ramp_offset_range(step1, first, blocklen),
        _ramp_offset_range(step2, first, blocklen),
    )
end
