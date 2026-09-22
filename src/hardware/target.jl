# Runtime hardware detection, no external dependency (docs/decisions.md,
# Phase G). Two rules are load-bearing: detection happens once per *process*
# in `__init__`, never at precompile time (a .ji cached on one node class of a
# shared depot must not carry another node's features), and every failure path
# resolves to `:unknown`, which maps back to the constants this package
# shipped before detection existed.

"""
    CacheLevel(bytes, ways, line, sharing)

One detected cache level; any field may be `0` for "not detected". `sharing`
is the number of logical CPUs sharing this level -- *detected*, never assumed,
since assuming L2 is private is what the recorded A57 refutation was about.
"""
struct CacheLevel
    bytes::Int
    ways::Int
    line::Int
    sharing::Int
end
CacheLevel() = CacheLevel(0, 0, 0, 0)

"""
    TargetProfile

What was detected about the host CPU. `isa` is the closed dispatch key:
`:avx512`, `:avx2`, `:neon` or `:unknown`.
"""
struct TargetProfile
    isa::Symbol
    arch::Symbol
    cpu_name::String
    vector_bytes::Int
    nregisters::Int
    l1d::CacheLevel
    l2::CacheLevel
    l3::CacheLevel
end

unknown_target() = TargetProfile(
    :unknown, Sys.ARCH, "", 0, 0, CacheLevel(), CacheLevel(), CacheLevel()
)

# LLVM micro-architecture name -> ISA key. A *recognition* table, not a tuning
# table: it answers only "which vector ISA", a discrete capability question.
# Unlisted names fall through to the CPUID probe, then to `:unknown`.
const _UARCH_ISA = Dict{String, Symbol}(
    n => :avx512 for n in (
            "skylake-avx512", "cascadelake", "cooperlake", "cannonlake",
            "icelake-client", "icelake-server", "tigerlake", "rocketlake",
            "sapphirerapids", "emeraldrapids", "graniterapids", "knl", "knm",
            "znver4", "znver5",
        )
)
for n in (
        "haswell", "broadwell", "skylake", "alderlake", "raptorlake",
        "meteorlake", "sierraforest", "grandridge", "tremont", "goldmont",
        "goldmont-plus", "znver1", "znver2", "znver3", "bdver4",
    )
    _UARCH_ISA[n] = :avx2
end

# `Base.BinaryPlatforms.CPUID` is an undocumented Base submodule, so every use
# is wrapped: if it moves, detection degrades rather than failing to load.
# This is the path that makes an *unlisted* CPU still get the right shape.
function _isa_from_cpuid()
    return try
        C = Base.BinaryPlatforms.CPUID
        C.test_cpu_feature(C.JL_X86_avx512f) ? :avx512 :
            C.test_cpu_feature(C.JL_X86_avx2) ? :avx2 : :unknown
    catch
        :unknown
    end
end

function _detect_isa()
    if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
        key = get(_UARCH_ISA, Sys.CPU_NAME, :miss)
        return key === :miss ? _isa_from_cpuid() : key
    elseif Sys.ARCH === :aarch64
        # 128-bit NEON is baseline. SVE is deliberately not detected: its
        # vector length is runtime-variable, which a fixed SIMDKernel shape
        # cannot express, so SVE machines get the NEON shape, not a wrong one.
        return :neon
    end
    return :unknown
end

# Register *count* is set by the ISA, not the lane width in use: (16,6,4) has
# 24 live 256-bit accumulators and does not spill on AVX-512, because AVX512VL
# supplies 32 ymm registers (measured 2026-09-11, ccqlin038).
_isa_vector_bytes(::Val{K}) where {K} = K === :avx512 ? 64 : K === :avx2 ? 32 : K === :neon ? 16 : 0
_isa_nregisters(::Val{K}) where {K} = K === :avx512 ? 32 : K === :avx2 ? 16 : K === :neon ? 32 : 0

# --- cache topology: read on demand, never on a load or first-call path -----

# "32K" / "1024K" / "25344K" as Linux sysfs writes them.
function _parse_size(s::AbstractString)
    s = strip(s)
    isempty(s) && return 0
    mult = get(Dict('K' => 1024, 'M' => 1024^2, 'G' => 1024^3), uppercase(s[end]), 1)
    mult == 1 || (s = s[1:(end - 1)])
    return something(tryparse(Int, strip(s)), 0) * mult
end

# "0,16" -> 2; "0-7,16-23" -> 16; "" -> 0.
function _count_cpu_list(s::AbstractString)
    n = 0
    for part in split(strip(s), ',')
        isempty(part) && continue
        lohi = split(part, '-')
        lo = tryparse(Int, lohi[1])
        lo === nothing && continue
        hi = length(lohi) == 2 ? tryparse(Int, lohi[2]) : lo
        hi === nothing || (n += max(0, hi - lo + 1))
    end
    return n
end

_read_or(path, default = "") = try
    isfile(path) ? chomp(read(path, String)) : default
catch
    default
end
_int_or(path) = something(tryparse(Int, _read_or(path)), 0)

function _cache_topology_linux()
    base = "/sys/devices/system/cpu/cpu0/cache"
    isdir(base) || return nothing
    levels = Dict{Symbol, CacheLevel}()
    for entry in readdir(base)
        startswith(entry, "index") || continue
        d = joinpath(base, entry)
        level = tryparse(Int, _read_or(joinpath(d, "level")))
        level === nothing && continue
        kind = _read_or(joinpath(d, "type"))
        key = level == 1 ? (kind == "Data" ? :l1d : :skip) :
            level == 2 ? :l2 : level == 3 ? :l3 : :skip
        key === :skip && continue
        levels[key] = CacheLevel(
            _parse_size(_read_or(joinpath(d, "size"))),
            _int_or(joinpath(d, "ways_of_associativity")),
            _int_or(joinpath(d, "coherency_line_size")),
            _count_cpu_list(_read_or(joinpath(d, "shared_cpu_list"))),
        )
    end
    return (
        l1d = get(levels, :l1d, CacheLevel()), l2 = get(levels, :l2, CacheLevel()),
        l3 = get(levels, :l3, CacheLevel()),
    )
end

# macOS exposes no associativity, but does expose `hw.perflevel0.cpusperl2` --
# how many cores share one L2, which is the A57 datum on Apple Silicon.
_sysctl_int(name) = try
    something(tryparse(Int, chomp(read(`sysctl -n $name`, String))), 0)
catch
    0
end

function _cache_topology_darwin()
    line = _sysctl_int("hw.cachelinesize")
    pick(a, b) = (v = _sysctl_int(a); v == 0 ? _sysctl_int(b) : v)
    return (
        l1d = CacheLevel(pick("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"), 0, line, 1),
        l2 = CacheLevel(
            pick("hw.perflevel0.l2cachesize", "hw.l2cachesize"), 0, line,
            _sysctl_int("hw.perflevel0.cpusperl2")
        ),
        l3 = CacheLevel(
            _sysctl_int("hw.l3cachesize"), 0, line,
            _sysctl_int("hw.perflevel0.logicalcpu")
        ),
    )
end

"""
    cache_topology() -> NamedTuple or nothing

Detected cache hierarchy as `(; l1d, l2, l3)` of [`CacheLevel`](@ref), or
`nothing`. Reads Linux sysfs or macOS `sysctl` on demand; any field may be `0`.
Exposed for reporting -- nothing that picks a number at runtime consults it.
"""
function cache_topology()
    return try
        Sys.islinux() ? _cache_topology_linux() :
            Sys.isapple() ? _cache_topology_darwin() : nothing
    catch
        nothing
    end
end

function _detect_target()
    key = _detect_isa()
    topo = cache_topology()
    l1d, l2, l3 = topo === nothing ?
        (CacheLevel(), CacheLevel(), CacheLevel()) : (topo.l1d, topo.l2, topo.l3)
    return TargetProfile(
        key, Sys.ARCH, Sys.CPU_NAME,
        _isa_vector_bytes(Val(key)), _isa_nregisters(Val(key)), l1d, l2, l3
    )
end

const _TARGET = Ref{TargetProfile}(unknown_target())

"""
    target_profile() -> TargetProfile

The [`TargetProfile`](@ref) detected for this process, or the all-`:unknown`
profile if detection failed.
"""
target_profile() = _TARGET[]

_init_target!() = (
    _TARGET[] = try
        _detect_target()
    catch
        unknown_target()
    end; nothing
)

# Whether the vectorized complex fast paths (the deinterleaving packer in
# src/packing/pack_contiguous.jl and the planar store in
# src/microkernels/planar.jl) apply on this host: its vector register must be
# as wide as AVX-512's, which is what makes a 2*PD-real load/deinterleave/store
# cheaper than 2*PD scalar stores. Keyed on vector width, not ISA name;
# AVX2/NEON/unknown take the scalar loops. One shared predicate, so the two
# fast paths cannot drift apart.
@inline _complex_fastpath_isa_eligible(profile::TargetProfile) =
    profile.vector_bytes == _isa_vector_bytes(Val(:avx512))
@inline _complex_fastpath_isa_eligible() = _complex_fastpath_isa_eligible(target_profile())
