# Hardware counters for the software-prefetch experiment: for one
# representative shape per prefetch site, baseline vs site-on under
# `perf stat`, so a ratio from bench_prefetch.jl can be read against what the
# memory system and the load ports actually did.
#
#   julia --project=. benchmark/perf_prefetch.jl                # every case
#   julia --project=. benchmark/perf_prefetch.jl --cases pack_b,macro --seconds 1.5
#   julia --project=. benchmark/perf_prefetch.jl --smoke        # tiny, checks plumbing
#
# One separate julia process per (case, variant) -- the only way to give each
# its own counters, and it also means each variant is compiled fresh with its
# site setting, never switched in-process. Inside the worker: build the
# fixture, switch the site, warm up (compile), time one call, then run the
# measured loop of about `--seconds` (default 1.5) of calls.
#
# Only the measured loop is counted when this `perf` supports
# `--control fifo:` with `--delay=-1` (perf >= 5.x): the worker enables the
# counters right before the loop and disables them right after. Otherwise the
# fallback is SUBTRACTION: each (case, variant) runs twice, identical except
# that one skips the loop, and the loop's counts are the difference. That is
# noisier (process startup and the fixture build are counted and cancelled),
# and the output says which mode ran.
#
# Events: the generic perf names first (cycles, instructions, L1/LLC loads and
# misses), then vendor events where the PMU has them -- Intel's
# `sw_prefetch_access.*` (which counts the prefetch instructions actually
# executed, i.e. checks each site's per-call prefetch count), L2 misses and
# load-port dispatch; AMD Zen's software-prefetch dispatch, load dispatch and
# fill-source events. Each candidate is probed with a trivial `perf stat`
# first and dropped if unsupported, so one missing counter never costs the
# run. More events than hardware counters means perf multiplexes; the
# `pct_running` column shows how much of the loop each event was live for.
#
# perf may not be permitted at all on a compute node (`perf_event_paranoid`,
# no `perf` binary): then this prints why and exits 0, having measured
# nothing.
#
# Everything here is a per-machine measurement; nothing is wired into any
# default.

include(joinpath(@__DIR__, "harness.jl"))

using QuasiStrided: set_prefetch!, PREFETCH_SITES

# ---------------------------------------------------------------------------
# Cases: (name, dtype, fixture builder, variants). `variant` is `base` or a
# `site` / `site@distance` as in bench_prefetch.jl.
# ---------------------------------------------------------------------------
const CASES = [
    # pack_b's worst round-1 control regression (icelake 1.45x).
    (
        name = "pack_b", T = Float32,
        build = T -> build_plain(T, ShapeSpec("smallM_12x256x256", 12, 256, 256), Random.MersenneTwister(1)),
        variants = ["base", "pack_b", "pack_b_line"],
    ),
    # A through its gather loop (M stride -2), L3-sized.
    (
        name = "pack_a", T = Float64,
        build = T -> build_scattered(T, Random.MersenneTwister(1), :negstride_big),
        variants = ["base", "pack_a", "pack_a_line"],
    ),
    # Compute-bound square: the macro-kernel and C-tile sites.
    (
        name = "macro_ctile", T = Float64,
        build = T -> build_plain(T, ShapeSpec("512^3", 512, 512, 512), Random.MersenneTwister(1)),
        variants = ["base", "macro", "ctile", "ctile_w"],
    ),
    # DRAM-resident B (256 MB), reversed: packing streams from memory.
    (
        name = "large", T = Float64,
        build = T -> build_large(T, Random.MersenneTwister(1), :smallM_negB),
        variants = ["base", "pack_b", "pack_b_line", "ctile_w"],
    ),
    # The most irregular gathers the engine can express.
    (
        name = "irregular", T = Float64,
        build = T -> build_irregular(T, Random.MersenneTwister(1), :hypercube_AB),
        variants = ["base", "pack_a", "pack_a_line", "pack_b_line"],
    ),
]

const SMOKE_CASES = [
    (
        name = "smoke", T = Float64,
        build = T -> build_plain(T, ShapeSpec("64^3", 64, 64, 64), Random.MersenneTwister(1)),
        variants = ["base", "pack_b_line", "ctile_w"],
    ),
]

const CANDIDATE_EVENTS = [
    # generic
    "cycles", "instructions", "L1-dcache-loads", "L1-dcache-load-misses",
    "LLC-loads", "LLC-load-misses", "cache-references", "cache-misses",
    # Intel (Ice Lake / Cascade Lake names)
    "sw_prefetch_access.any", "sw_prefetch_access.t0", "sw_prefetch_access.prefetchw",
    "l2_rqsts.references", "l2_rqsts.miss", "mem_load_retired.l2_miss",
    "mem_load_retired.l3_miss", "cycle_activity.stalls_l3_miss",
    "uops_dispatched.port_2_3", "uops_dispatched_port.port_2", "uops_dispatched_port.port_3",
    # AMD Zen2 / Zen4
    "ls_pref_instr_disp", "ls_dispatch.ld_dispatch", "ls_dispatch.store_dispatch",
    "l2_cache_req_stat.ls_rd_blk_c", "ls_dmnd_fills_from_sys.mem_io_local",
    "ls_any_fills_from_sys.dram_io_all", "ls_sw_pf_dc_fills.mem_io_local",
    "ls_hw_pf_dc_fills.mem_io_local",
]

default_distance(site::Symbol) = site === :macro ? 4 : site in (:ctile, :ctile_w) ? 1 : 16

function variant_settings(name::AbstractString)
    name == "base" && return ()
    if occursin('@', name)
        site, d = split(name, '@'; limit = 2)
        return ((Symbol(site), parse(Int, d)),)
    end
    return ((Symbol(name), default_distance(Symbol(name))),)
end

# ---------------------------------------------------------------------------
# Worker: one (case, variant) in its own process.
# ---------------------------------------------------------------------------
function worker()
    case = argval("case")
    variant = argval("variant")
    seconds = parse(Float64, argopt("seconds", "1.5"))
    loop = !hasflag("no-loop")
    ctl = argval("ctl")
    ack = argval("ack")
    c = only(filter(x -> x.name == case, vcat(CASES, SMOKE_CASES)))
    T = c.T
    fx = c.build(T)
    for (site, d) in variant_settings(variant)
        set_prefetch!(site, d)
    end
    return Base.invokelatest() do
        plan = plan_contract(fx.Cv, fx.Av, fx.indA, fx.Bv, fx.indB, fx.indC)
        execute!(plan, one(T), zero(T))   # compile + warm
        t1 = @elapsed execute!(plan, one(T), zero(T))
        iters = max(1, round(Int, seconds / t1))
        if !loop
            println("WORKER iters=0 seconds=0.0 t1=$t1")
            return
        end
        ctlio = ctl === nothing ? nothing : open(ctl, "w")
        ackio = ack === nothing ? nothing : open(ack, "r")
        if ctlio !== nothing
            write(ctlio, "enable\n"); flush(ctlio); readline(ackio)
        end
        t0 = time_ns()
        for _ in 1:iters
            execute!(plan, one(T), zero(T))
        end
        dt = (time_ns() - t0) / 1.0e9
        if ctlio !== nothing
            write(ctlio, "disable\n"); flush(ctlio); readline(ackio)
            close(ctlio); close(ackio)
        end
        println("WORKER iters=$iters seconds=$dt t1=$t1")
    end
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
function try_read(cmd)
    try
        return strip(read(cmd, String))
    catch
        return nothing
    end
end

# `perf stat` writes its counts to stderr; `-o` sends them to a file instead.
function perf_probe(ev)
    f = tempname()
    try
        run(pipeline(`perf stat -x, -o $f -e $ev -- true`; stdout = devnull, stderr = devnull))
        return read(f, String)
    catch
        return nothing
    finally
        rm(f; force = true)
    end
end

function supported_events()
    ok = String[]
    for ev in CANDIDATE_EVENTS
        out = perf_probe(ev)
        out === nothing && continue
        (occursin("<not supported>", out) || occursin("event syntax error", out) ||
            occursin("Cannot find", out) || !occursin(ev, out)) && continue
        push!(ok, ev)
    end
    return ok
end

function has_control()
    help = try_read(pipeline(`perf stat --help`; stderr = devnull))
    help === nothing && return false
    return occursin("--control", help)
end

# Parse `perf stat -x,` output: value,unit,event,run_time,pct_running,...
function parse_perf(path)
    res = Dict{String, Tuple{Float64, Float64}}()
    isfile(path) || return res
    for line in eachline(path)
        (isempty(line) || startswith(line, "#")) && continue
        f = split(line, ',')
        length(f) >= 5 || continue
        v = tryparse(Float64, f[1])
        v === nothing && continue
        pct = something(tryparse(Float64, f[5]), NaN)
        res[String(f[3])] = (v, pct)
    end
    return res
end

function worker_cmd(c, variant, seconds; loop = true, ctl = nothing, ack = nothing)
    args = [
        "--worker", "--case", c.name, "--variant", variant, "--seconds", string(seconds),
    ]
    loop || push!(args, "--no-loop")
    ctl === nothing || append!(args, ["--ctl", ctl, "--ack", ack])
    return `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) --startup-file=no $(@__FILE__) $args`
end

function parse_worker(out)
    m = match(r"WORKER iters=(\d+) seconds=([0-9.eE+-]+)", out)
    m === nothing && error("worker produced no result:\n$out")
    return parse(Int, m[1]), parse(Float64, m[2])
end

function run_one(c, variant, events, seconds, control, tmp)
    evs = join(events, ",")
    out1 = joinpath(tmp, "perf_$(c.name)_$(variant).csv")
    if control
        ctl = joinpath(tmp, "ctl.fifo"); ack = joinpath(tmp, "ack.fifo")
        for f in (ctl, ack)
            ispath(f) && rm(f)
            run(`mkfifo $f`)
        end
        cmd = `perf stat -x, -o $out1 --delay=-1 --control fifo:$ctl,$ack -e $evs -- $(worker_cmd(c, variant, seconds; ctl = ctl, ack = ack))`
        iters, secs = parse_worker(read(cmd, String))
        return iters, secs, parse_perf(out1)
    else
        out0 = joinpath(tmp, "perf_$(c.name)_$(variant)_noloop.csv")
        iters, secs = parse_worker(read(`perf stat -x, -o $out1 -e $evs -- $(worker_cmd(c, variant, seconds))`, String))
        read(`perf stat -x, -o $out0 -e $evs -- $(worker_cmd(c, variant, seconds; loop = false))`, String)
        full = parse_perf(out1); none = parse_perf(out0)
        diff = Dict(k => (v[1] - get(none, k, (0.0, NaN))[1], v[2]) for (k, v) in full)
        return iters, secs, diff
    end
end

function driver()
    smoke = hasflag("smoke")
    seconds = parse(Float64, argopt("seconds", smoke ? "0.2" : "1.5"))
    names = argval("cases")
    cases = smoke ? SMOKE_CASES : CASES
    names === nothing || (cases = filter(c -> c.name in split(names, ','), cases))

    print_env_header(stdout, "perf_prefetch.jl")
    paranoid = try_read(`cat /proc/sys/kernel/perf_event_paranoid`)
    perfpath = try_read(`which perf`)
    perfver = perfpath === nothing ? nothing : try_read(`perf version`)
    println("perf = ", something(perfpath, "NOT FOUND"), "  (", something(perfver, "?"), ")")
    println("perf_event_paranoid = ", something(paranoid, "unreadable"))
    println("node = ", gethostname(), "  slurm job = ", get(ENV, "SLURM_JOB_ID", "none"))
    if perfpath === nothing
        println("no perf binary on this node: nothing measured.")
        return
    end
    probe = perf_probe("instructions")
    if probe === nothing || !occursin("instructions", probe) || occursin("<not supported>", probe)
        println("perf cannot count even `instructions` here (paranoid = $paranoid?): nothing measured.")
        println("probe output: ", probe)
        return
    end
    events = supported_events()
    control = has_control() && !hasflag("subtract")  # `--subtract` forces the fallback
    # Subtraction cancels a process startup of several seconds, so the loop
    # has to dominate it: four times the requested length.
    control || (seconds *= 4)
    println("counting mode = ", control ? "control fifo (measured loop only)" : "SUBTRACTION (full run minus no-loop run; noisier; loop length x4)")
    println("events (", length(events), ") = ", join(events, ","))

    outdir = results_dir() * (haskey(ENV, "SLURM_JOB_ID") ? "-job" * ENV["SLURM_JOB_ID"] : "")
    mkpath(outdir)
    tmp = mktempdir()
    csvpath = joinpath(outdir, "perf_prefetch.csv")
    csv = open(csvpath, "w")
    println(csv, "case,dtype,variant,event,value,per_call,pct_running,iters,seconds,mode")
    for c in cases
        println("\n== case $(c.name) ($(c.T)) ==")
        base = nothing
        for v in c.variants
            iters, secs, res = run_one(c, v, events, seconds, control, tmp)
            percall = Dict(k => x[1] / iters for (k, x) in res)
            v == "base" && (base = percall)
            for ev in events
                haskey(res, ev) || continue
                println(csv, "$(c.name),$(c.T),$v,$ev,$(res[ev][1]),$(percall[ev]),$(res[ev][2]),$iters,$secs,",
                    control ? "control" : "subtract")
            end
            @printf("  %-12s %6d calls, %.3f s  (%.1f us/call)\n", v, iters, secs, 1.0e6secs / iters)
            for ev in events
                haskey(percall, ev) || continue
                r = base === nothing || !haskey(base, ev) || base[ev] == 0 ? NaN : percall[ev] / base[ev]
                @printf("      %-36s %14.1f /call  x%.3f vs base  (%.0f%% live)\n", ev, percall[ev], r, res[ev][2])
            end
            flush(csv)
        end
    end
    close(csv)
    open(joinpath(outdir, "perf_prefetch_PROVENANCE.txt"), "w") do io
        print_env_header(io, "perf_prefetch.jl")
        println(io, "git commit = ", git_commit())
        println(io, "perf = ", perfpath, " ", perfver, "  paranoid = ", paranoid)
        println(io, "mode = ", control ? "control" : "subtract", "  seconds = ", seconds)
        println(io, "events = ", join(events, ","))
    end
    println("\nwrote ", csvpath)
    return
end

hasflag("worker") ? worker() : driver()
