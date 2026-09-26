#!/bin/bash
# Slurm job for round 2 of the software-prefetch spike,
# benchmark/bench_prefetch.jl:
#
#   1. main: every single prefetch site -- the round-2 ones (ctile, ctile_w,
#      pack_b_line, pack_a_line) next to the round-1 ones (pack_b, pack_a,
#      macro) as in-job controls -- over the CONTROL + SCATTERED families, all
#      four dtypes;
#   2. big: the same variants over the LARGE (DRAM-resident, 128-256 MB
#      operands) and IRREGULAR (many short axes, pseudo-random large strides)
#      families, Float64 only, with fewer reps on the slow shapes (the script's
#      `--budget`);
#   3. perf: benchmark/perf_prefetch.jl, `perf stat` counters per site on one
#      representative shape each (also runnable alone as
#      benchmark/submit_perf_prefetch.sh).
#
# Single core, single node -- no internal parallelism, so plain sbatch (not
# disBatch). One job per node class:
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_prefetch2.sh
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_prefetch2.sh
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_prefetch2.sh
#
# Submit from the repo root. See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
# Check the `cpu =` / `target =` header, the canary spread and each shape's
# base spread before trusting any ratio. Locally (Cascade Lake workstation)
# the three arms take about 8, 9 and 3.5 minutes, so the hour is a wide margin.
#SBATCH --job-name=qs-prefetch2
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repo root.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# Same Julia resolution as submit_blocking_model.sh: on Rocky 8 nodes
# (`--reservation=rocky8`) the module system's own binaries need a newer glibc
# than the node has, so `module load` fails and there is no `julia`; then fall
# back to $QS_JULIA or a juliaup-installed 1.12 on the shared home, since the
# official Julia builds need only glibc 2.17.
module load modules/2.5-beta1 || true
module load julia/1.12.6 || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac  # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
echo "julia = $JULIA"

export JULIA_NUM_THREADS=1
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()'

VARIANTS=pack_b,pack_a,pack_b_line,pack_a_line,macro,ctile,ctile_w

"$JULIA" --project=. benchmark/bench_prefetch.jl --tag round2_main \
    --family control,scattered --variants "$VARIANTS"

"$JULIA" --project=. benchmark/bench_prefetch.jl --tag round2_big \
    --family large,irregular --dtypes Float64 --variants "$VARIANTS"

# Last, and allowed to fail without failing the job: perf may be missing or
# not permitted on the node (the script reports which and measures nothing).
echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unreadable)"
"$JULIA" --project=. benchmark/perf_prefetch.jl --seconds 1.5 || echo "perf arm failed (exit $?)"

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
