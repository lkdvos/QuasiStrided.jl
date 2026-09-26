# Software prefetch (src/hardware/prefetch.jl): the intrinsic wrapper, the
# per-site switches, and the two properties the three EXPERIMENTAL insertion
# points must have -- a disabled site emits no prefetch instruction at all, and
# no setting of any site changes a single bit of any result.
#
# Every testset that switches a site restores all of them in a `finally`, so a
# failure here cannot leak a prefetch setting into later test files.

using InteractiveUtils: code_native
using QuasiStrided: prefetch, set_prefetch!, prefetch_distance, PREFETCH_SITES,
    PackedPanel, PtrScatterAxis, unsafe_pack_a!, unsafe_pack_b!

function _pf_all_off!()
    for site in PREFETCH_SITES
        set_prefetch!(site, 0)
    end
    return nothing
end

# Number of prefetch instructions in the native code of `f` at `types`, or
# `nothing` on an architecture whose mnemonic this does not know.
function _pf_count(f, types)
    rx = Sys.ARCH === :x86_64 ? r"^\s*prefetch" :
        Sys.ARCH === :aarch64 ? r"^\s*prfm" : nothing
    rx === nothing && return nothing
    s = sprint(io -> code_native(io, f, types; debuginfo = :none))
    return count(l -> occursin(rx, l), split(s, '\n'))
end

function _pf_count_rx(f, types, rx)
    s = sprint(io -> code_native(io, f, types; debuginfo = :none))
    return count(l -> occursin(rx, l), split(s, '\n'))
end

@testset "prefetch: intrinsic wrapper" begin
    x = rand(64)
    GC.@preserve x begin
        p = pointer(x)
        for rw in (0, 1), loc in 0:3
            @test prefetch(p, Val(rw), Val(loc)) === nothing
        end
        @test prefetch(p) === nothing
        # A hint never faults, whatever the address.
        @test prefetch(Ptr{Float64}(0)) === nothing
        @test_throws ArgumentError prefetch(p, Val(2), Val(3))
        @test_throws ArgumentError prefetch(p, Val(0), Val(4))
    end
    if Sys.ARCH === :x86_64
        f(p) = prefetch(p, Val(0), Val(3))
        g(p) = prefetch(p, Val(1), Val(3))
        h(p) = prefetch(p, Val(0), Val(0))
        asm(fn) = sprint(io -> code_native(io, fn, (Ptr{Float64},); debuginfo = :none))
        @test occursin("prefetcht0", asm(f))
        @test occursin("prefetchw", asm(g))
        @test occursin("prefetchnta", asm(h))
    end
end

@testset "prefetch: site switches" begin
    try
        _pf_all_off!()
        for site in PREFETCH_SITES
            @test prefetch_distance(site) == 0
        end
        @test set_prefetch!(:pack_b, 8) == 0
        @test prefetch_distance(:pack_b) == 8
        @test set_prefetch!(:pack_b, 8) == 8   # no-op redefinition skipped
        @test set_prefetch!(:pack_b, 0) == 8
        @test_throws ArgumentError set_prefetch!(:nonsense, 1)
        @test_throws ArgumentError set_prefetch!(:pack_a, -1)
    finally
        _pf_all_off!()
    end
end

# The two gather packers at the types `_execute_nest!` calls them with (a
# `PackedPanel` destination, `PtrScatterAxis`/`AffineAxis` source axes), and
# the whole nest. A's rows are stride-3 so its contiguous fast path is not
# eligible and the gather loop is what runs.
const _PF_KERNEL = SIMDKernel(Val(8), Val(6), Float64)
const _PF_DESC = typeof(_PF_KERNEL.descriptor)
const _PF_BTYPES = Tuple{
    PackedPanel{Float64}, QuasiStrided.QSTile{Vector{Float64}, PtrScatterAxis, AffineAxis},
    _PF_DESC, typeof(identity),
}
const _PF_ATYPES = Tuple{
    PackedPanel{Float64}, QuasiStrided.QSTile{Vector{Float64}, AffineAxis, AffineAxis},
    _PF_DESC, typeof(identity),
}

function _pf_nest_types()
    A = randn(40, 40); B = randn(40, 40); C = zeros(40, 40)
    plan = _mm_plan(C, A, B)
    execute!(plan, 1.0, 0.0)
    I = Int
    return Tuple{
        typeof(plan), typeof(plan.workspace), typeof(plan.kernel),
        I, I, I, I, I, I, I, I, Float64, Float64,
    }
end

@testset "prefetch: disabled sites emit no prefetch instruction" begin
    nest = _pf_nest_types()
    counts() = (
        _pf_count(unsafe_pack_a!, _PF_ATYPES),
        _pf_count(unsafe_pack_b!, _PF_BTYPES),
        _pf_count(QuasiStrided._execute_nest!, nest),
    )
    try
        _pf_all_off!()
        c = counts()
        if c[1] === nothing
            @test_skip "no prefetch mnemonic known for $(Sys.ARCH)"
        else
            @test c == (0, 0, 0)
            # Each site switched on alone reaches exactly its own code...
            set_prefetch!(:pack_a, 16)
            ca = counts()
            @test ca[1] > 0 && ca[2] == 0 && ca[3] == 0
            _pf_all_off!(); set_prefetch!(:pack_b, 16)
            cb = counts()
            @test cb[1] == 0 && cb[2] > 0 && cb[3] == 0
            _pf_all_off!(); set_prefetch!(:macro, 4)
            cm = counts()
            @test cm[1] == 0 && cm[2] == 0 && cm[3] > 0
            _pf_all_off!(); set_prefetch!(:pack_a_line, 16)
            cal = counts()
            @test cal[1] > 0 && cal[2] == 0 && cal[3] == 0
            _pf_all_off!(); set_prefetch!(:pack_b_line, 16)
            cbl = counts()
            @test cbl[1] == 0 && cbl[2] > 0 && cbl[3] == 0
            _pf_all_off!(); set_prefetch!(:ctile, 1)
            cc = counts()
            @test cc[1] == 0 && cc[2] == 0 && cc[3] > 0
            _pf_all_off!(); set_prefetch!(:ctile_w, 1)
            cw = counts()
            @test cw[1] == 0 && cw[2] == 0 && cw[3] > 0
            if Sys.ARCH === :x86_64
                # write intent really is `prefetchw`, and only under :ctile_w
                nestw = _pf_count_rx(QuasiStrided._execute_nest!, nest, r"^\s*prefetchw")
                @test nestw == cw[3]
                _pf_all_off!(); set_prefetch!(:ctile, 1)
                @test _pf_count_rx(QuasiStrided._execute_nest!, nest, r"^\s*prefetchw") == 0
            end
            # ...and switching it back off really removes it (the redefinition
            # invalidated the code that had inlined the old constant).
            _pf_all_off!()
            @test counts() == (0, 0, 0)
        end
    finally
        _pf_all_off!()
    end
end

# Fixtures covering every gather path the sites touch: real and complex, tail
# slivers in every dimension, scattered/negative-stride operands, an A whose
# M axis is not unit-stride (A's contiguous fast path ineligible), and a small
# blocking so the macro nest crosses many M/N/K blocks and the panel-ahead
# prefetch takes its wrap and next-B branches.
function _pf_cases()
    cases = Tuple{String, Function}[]
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        push!(cases, ("plain $T 67x53x71", () -> begin
            rng = MersenneTwister(21)
            A = randn(rng, T, 67, 53); B = randn(rng, T, 53, 71); C = zeros(T, 67, 71)
            execute!(_mm_plan(C, A, B), one(T), zero(T))
            C
        end))
        push!(cases, ("plain $T small blocking", () -> begin
            rng = MersenneTwister(22)
            A = randn(rng, T, 45, 37); B = randn(rng, T, 37, 29); C = randn(rng, T, 45, 29)
            execute!(_mm_plan(C, A, B; mc = 16, kc = 8, nc = 12), one(T), T(0.5))
            C
        end))
        push!(cases, ("scattered $T", () -> begin
            Cv, Av, iA, Bv, iB, iC = scattered_fixture(T)
            execute!(plan_contract(Cv, Av, iA, Bv, iB, iC), one(T), zero(T))
            copy(parent(Cv))
        end))
        push!(cases, ("transposed-A $T", () -> begin
            rng = MersenneTwister(23)
            At = randn(rng, T, 41, 50)          # K x M storage
            Av = permutedims(StridedView(At), (2, 1))  # M stride = 41
            B = randn(rng, T, 41, 33); C = zeros(T, 50, 33)
            plan = plan_contract(
                StridedView(C), Av, (1, 2), StridedView(B), (2, 3), (1, 3);
                mc = 24, kc = 16, nc = 18
            )
            execute!(plan, one(T), zero(T))
            C
        end))
        # C micro-tiles in every layout `_prefetch_tile_lines!` distinguishes:
        # row-major C (the dense run is along N) and a C whose M and N groups
        # are multi-axis composites (scatter-table offsets, element-wise).
        push!(cases, ("row-major C $T", () -> begin
            rng = MersenneTwister(25)
            A = randn(rng, T, 37, 29); B = randn(rng, T, 29, 43)
            Ct = zeros(T, 43, 37)
            Cv = permutedims(StridedView(Ct), (2, 1))
            execute!(
                plan_contract(Cv, StridedView(A), (1, 2), StridedView(B), (2, 3), (1, 3)),
                one(T), zero(T)
            )
            Ct
        end))
        push!(cases, ("interleaved C $T", () -> begin
            rng = MersenneTwister(26)
            d = 5
            Av = StridedView(randn(rng, T, d, d, d, d))   # (a1,k1,a2,k2)
            Bv = StridedView(randn(rng, T, d, d, d, d))   # (k1,n1,k2,n2)
            C = zeros(T, d, d, d, d)                      # (a1,n1,a2,n2)
            execute!(
                plan_contract(StridedView(C), Av, (1, 3, 2, 4), Bv, (3, 5, 4, 6), (1, 5, 2, 6)),
                one(T), zero(T)
            )
            C
        end))
    end
    return cases
end

@testset "prefetch: results bit-identical under every site setting" begin
    configs = (
        "pack_a" => ((:pack_a, 16),),
        "pack_b" => ((:pack_b, 16),),
        "macro" => ((:macro, 4),),
        "pack_b dist 1" => ((:pack_b, 1),),
        "all on" => ((:pack_a, 8), (:pack_b, 8), (:macro, 2)),
        "pack_a_line" => ((:pack_a_line, 16),),
        "pack_b_line" => ((:pack_b_line, 16),),
        "pack_b_line dist 1" => ((:pack_b_line, 1),),
        "ctile" => ((:ctile, 1),),
        "ctile_w" => ((:ctile_w, 1),),
        "every site" => Tuple((site, 3) for site in PREFETCH_SITES),
    )
    cases = _pf_cases()
    try
        _pf_all_off!()
        baseline = [f() for (_, f) in cases]
        for (cname, settings) in configs
            _pf_all_off!()
            for (site, d) in settings
                set_prefetch!(site, d)
            end
            @testset "$cname" begin
                for ((name, f), ref) in zip(cases, baseline)
                    @testset "$name" begin
                        @test isequal(f(), ref)
                    end
                end
            end
        end
    finally
        _pf_all_off!()
    end
end

@testset "prefetch: gather packers with a tail sliver and a ScatterAxis" begin
    kern = _PF_KERNEL
    rng = MersenneTwister(24)
    storage = randn(rng, 4000)
    koffs = collect(0:13:(13 * 39))                 # 40 K steps, scattered
    function run()
        pb = zeros(nr(kern) * 40)
        pa = zeros(mr(kern) * 40)
        src_b = SourceTile(storage, 3, ScatterAxis(koffs, 40), AffineAxis(0, 600, 4))  # n = 4 < NR
        src_a = SourceTile(storage, 7, AffineAxis(0, 2, 5), ScatterAxis(koffs, 40))    # m = 5 < MR
        pack_b!(pb, src_b, kern, identity)
        pack_a!(pa, src_a, kern, identity)
        return pb, pa
    end
    try
        _pf_all_off!()
        ref = run()
        set_prefetch!(:pack_a, 3); set_prefetch!(:pack_b, 3)
        @test run() == ref
        set_prefetch!(:pack_a, 100); set_prefetch!(:pack_b, 100)  # distance > kc
        @test run() == ref
        _pf_all_off!()
        for d in (1, 3, 100)
            set_prefetch!(:pack_a_line, d); set_prefetch!(:pack_b_line, d)
            @test run() == ref
        end
    finally
        _pf_all_off!()
    end
end

# Every mode of the line-granular hook (`_line_mode`, src/packing/pack.jl):
# dense lanes over a long-stride or short-stride affine K axis (1), long-stride
# lanes over a unit-stride K axis (2), a K stride that does not divide a line
# (3), and a scattered K axis under both lane kinds. Read-only sources may
# overlap, so the short-stride cases need no special storage.
@testset "prefetch: line-granular gather, every mode, bit-identical" begin
    kern = _PF_KERNEL
    rng = MersenneTwister(27)
    storage = randn(rng, 20_000)
    koffs = collect(0:13:(13 * 39))
    layouts = (
        "dense lanes, K stride 50" => (AffineAxis(0, 50, 40), AffineAxis(0, 1, 6)),
        "dense lanes, K stride 2" => (AffineAxis(0, 2, 40), AffineAxis(0, 1, 6)),
        "long lanes, K unit" => (AffineAxis(0, 1, 40), AffineAxis(0, 600, 6)),
        "long lanes, K stride 3" => (AffineAxis(0, 3, 40), AffineAxis(0, 600, 6)),
        "long lanes, K negative" => (AffineAxis(39, -1, 40), AffineAxis(0, 600, 6)),
        "dense lanes, K scattered" => (ScatterAxis(koffs, 40), AffineAxis(0, 1, 6)),
        "long lanes, K scattered" => (ScatterAxis(koffs, 40), AffineAxis(0, 600, 5)),
        "negative dense lanes" => (AffineAxis(0, 40, 40), AffineAxis(5, -1, 6)),
    )
    function run(kaxis, laxis)
        pb = zeros(nr(kern) * 40)
        pa = zeros(mr(kern) * 40)
        pack_b!(pb, SourceTile(storage, 11, kaxis, laxis), kern, identity)
        # A: the same axes transposed (lanes on rows), padded to MR by a tail
        pack_a!(pa, SourceTile(storage, 11, laxis, kaxis), kern, identity)
        return pb, pa
    end
    try
        for (name, (kaxis, laxis)) in layouts
            _pf_all_off!()
            ref = run(kaxis, laxis)
            for d in (1, 4, 16)
                set_prefetch!(:pack_a_line, d); set_prefetch!(:pack_b_line, d)
                @test run(kaxis, laxis) == ref
            end
        end
    finally
        _pf_all_off!()
    end
end

@testset "prefetch: execute! stays allocation-free with every site on" begin
    try
        for settings in (
                ((:pack_a, 8), (:pack_b, 8), (:macro, 2), (:ctile_w, 1)),
                ((:pack_a_line, 8), (:pack_b_line, 8), (:ctile, 1)),
            )
            _pf_all_off!()
            for (site, d) in settings
                set_prefetch!(site, d)
            end
            Cv, Av, iA, Bv, iB, iC = scattered_fixture(Float64)
            plan = plan_contract(Cv, Av, iA, Bv, iB, iC)
            @test _steady_allocs!(execute!, plan, parent(Cv)) == 0
            A = randn(70, 50); B = randn(50, 60); C = zeros(70, 60)
            @test _steady_allocs!(execute!, _mm_plan(C, A, B), C) == 0
        end
    finally
        _pf_all_off!()
    end
end
