#!/bin/bash
# Slurm job for benchmark/bench_prefetch.jl: baseline vs each EXPERIMENTAL
# software-prefetch site (pack_b, pack_a, macro, and all three together) over
# the plain CONTROL shapes and the SCATTERED gather shapes, plus a distance
# probe for the two packing sites. Single core, single node -- the script has
# no internal parallelism, so plain sbatch is the right tool (not disBatch).
# Pick the microarchitecture with --constraint on the command line, one job
# per node class:
#
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_prefetch.sh   # Ice Lake-SP, AVX-512
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_prefetch.sh   # Zen4, AVX-512
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_prefetch.sh   # Zen2, AVX2
#
# Submit from the repo root. See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
# Check the job's `cpu =` / `target =` header and the canary / base spread
# before trusting any ratio. Both arms together take ~10 min locally on a
# Cascade Lake workstation, so the hour below is a wide margin.
#SBATCH --job-name=qs-prefetch
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
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

# Main arm: every site at its default distance, and all together.
"$JULIA" --project=. benchmark/bench_prefetch.jl --tag main \
    --variants pack_b,pack_a,macro,all

# Distance probe for the two packing sites (K steps) and the macro head length
# (cache lines), around the defaults of 16 and 4. Real types only, to bound
# walltime; same per-job results dir, its own `--tag`.
"$JULIA" --project=. benchmark/bench_prefetch.jl --tag distance \
    --dtypes Float64,Float32 \
    --variants pack_b@4,pack_b@64,pack_a@4,pack_a@64,macro@1,macro@16

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
