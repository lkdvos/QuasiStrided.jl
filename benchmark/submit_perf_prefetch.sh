#!/bin/bash
# Slurm job for benchmark/perf_prefetch.jl: `perf stat` hardware counters
# (cycles, instructions, L1/L2/LLC misses, software-prefetch and load-port
# counts where the PMU has them) for baseline vs each prefetch site, one
# representative shape per site, one julia process per (shape, variant). The
# script records `perf_event_paranoid` and the perf version, drops any
# counter this node does not support, and measures nothing (exit 0) if perf
# is unavailable or not permitted -- check the head of the output first.
# Single core, single node; plain sbatch. One job per node class:
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_perf_prefetch.sh
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_perf_prefetch.sh
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_perf_prefetch.sh
#
# benchmark/submit_prefetch2.sh already runs this as its last arm; this
# script is for re-running the counters alone (about 3.5 minutes locally).
#
# Submit from the repo root. See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-perf-prefetch
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unreadable)"
echo "perf = $(command -v perf || echo NOT FOUND)"

# perf_prefetch.jl spawns its workers with the same julia binary it runs
# under (Base.julia_cmd()), so $JULIA carries through.
"$JULIA" --project=. benchmark/perf_prefetch.jl --seconds 1.5

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
