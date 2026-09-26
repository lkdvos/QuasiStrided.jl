# Instruction-selection probe for src/microkernels/fmaddsub.jl, next to the
# planar and 1m kernels at the same shapes: counts, in `@code_native` of
# `accumulate` with the driver's own `PackedPanel` argument types, the fused
# ops, shuffles, broadcasts and stack spill traffic.
#
#   julia --project=. benchmark/probes/fmaddsub_codegen.jl              # host ISA
#   julia -C znver2 --project=. benchmark/probes/fmaddsub_codegen.jl    # AVX2 codegen
#
# `-C znver2` changes LLVM's target only; the counts are a codegen property and
# need no Rome hardware (running the code would need it). Pass
# `--dump` to also print the loop body of each kernel.

using QuasiStrided
using QuasiStrided: PlanarKernel, OneMKernel, FMAddSubKernel, PackedPanel, zero_accumulator
using InteractiveUtils: code_native
using Printf

const DUMP = "--dump" in ARGS

count_re(re, s) = count(re, s)

# Stack traffic: any vector move to/from [rsp/rbp + ...].
const SPILL_ST = r"vmov\w*\s+[xyz]mmword ptr \[r[sb]p[^\]]*\],"
const SPILL_LD = r"vmov\w*\s+[xyz]mm\d+,\s*[xyz]mmword ptr \[r[sb]p"

# The K loop: from the label of the (last) backward conditional branch to that
# branch. Counts are taken over this body only -- the entry/exit copies of the
# accumulator tuple (sret, the `kc == 0` memcpy path) are not per-K-step cost.
function hot_loop(asm::AbstractString)
    lines = split(asm, '\n')
    labels = Dict{String, Int}()
    best = nothing
    for (n, l) in enumerate(lines)
        m = match(r"^(\.LBB\w+):", l)
        m === nothing || (labels[m[1]] = n)
        b = match(r"^\s+j(?:ne|nz|b|l|g|a|e)\w*\s+(\.LBB\w+)", l)
        if b !== nothing && haskey(labels, b[1])
            best = (labels[b[1]], n)   # backward branch: a loop
        end
    end
    best === nothing && return asm
    return join(lines[best[1]:best[2]], '\n')
end

function probe(kernel)
    T = QuasiStrided.scalartype(kernel)
    R = real(T)
    acc = zero_accumulator(kernel)
    io = IOBuffer()
    code_native(
        io, Base.accumulate,
        (typeof(kernel), typeof(acc), PackedPanel{R}, PackedPanel{R}, Int);
        debuginfo = :none, syntax = :intel
    )
    full = String(take!(io))
    asm = hot_loop(full)
    return (
        fmaddsub = count_re(r"vfmaddsub\d+p", asm),
        fmsubadd = count_re(r"vfmsubadd\d+p", asm),
        fma = count_re(r"vfn?madd\d+p", asm),       # plain/negated FMA (not addsub)
        fms = count_re(r"vfn?msub\d+p", asm),
        mul = count_re(r"vmulp", asm),
        addsub = count_re(r"v(add|sub)p[sd]", asm),
        blend = count_re(r"vblend", asm),
        shuf = count_re(r"v(shufp|permilp|perm[0-9a-z]*p|unpck)", asm),
        bcast = count_re(r"vbroadcasts", asm),
        spill_st = count_re(SPILL_ST, asm),
        spill_ld = count_re(SPILL_LD, asm),
        asm = asm, full = full,
    )
end

kernels(::Type{ComplexF64}) = [
    ("planar", PlanarKernel(Val(4), Val(5), ComplexF64, Val(4))),
    ("planar", PlanarKernel(Val(24), Val(3), ComplexF64, Val(8))),
    ("1m", OneMKernel(Val(4), Val(6), ComplexF64, Val(4))),
    ("1m", OneMKernel(Val(12), Val(8), ComplexF64, Val(8))),
    [("fmaddsub", FMAddSubKernel(Val(MR), Val(NR), ComplexF64, Val(W)))
        for (MR, NR, W) in QuasiStrided.KERNEL_SHAPES_C64_FMADDSUB]...,
]
kernels(::Type{ComplexF32}) = [
    ("planar", PlanarKernel(Val(8), Val(5), ComplexF32, Val(8))),
    ("planar", PlanarKernel(Val(48), Val(3), ComplexF32, Val(16))),
    ("1m", OneMKernel(Val(8), Val(6), ComplexF32, Val(8))),
    ("1m", OneMKernel(Val(24), Val(8), ComplexF32, Val(16))),
    [("fmaddsub", FMAddSubKernel(Val(MR), Val(NR), ComplexF32, Val(W)))
        for (MR, NR, W) in QuasiStrided.KERNEL_SHAPES_C32_FMADDSUB]...,
]

println("cpu = ", Sys.CPU_NAME, "   julia = ", VERSION,
    "   (LLVM target: ", Base.JLOptions().cpu_target == C_NULL ? "native" :
    unsafe_string(Base.JLOptions().cpu_target), ")")
@printf("%-10s %-9s %-10s %8s %8s %5s %5s %4s %7s %5s %5s %5s %8s %8s\n",
    "T", "method", "shape", "fmaddsub", "fmsubadd", "fma", "fms", "mul", "add/sub",
    "shuf", "blend", "bcast", "spill_st", "spill_ld")
for T in (ComplexF64, ComplexF32)
    for (name, k) in kernels(T)
        r = probe(k)
        shape = "$(QuasiStrided.mr(k))x$(QuasiStrided.nr(k))/W$(QuasiStrided.lanewidth(k))"
        @printf("%-10s %-9s %-10s %8d %8d %5d %5d %4d %7d %5d %5d %5d %8d %8d\n",
            T, name, shape, r.fmaddsub, r.fmsubadd, r.fma, r.fms, r.mul, r.addsub,
            r.shuf, r.blend, r.bcast, r.spill_st, r.spill_ld)
        if DUMP
            println(r.asm)
        end
    end
end
