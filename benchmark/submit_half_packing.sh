#!/bin/bash
# Slurm job for benchmark/bench_half_packing.jl: the packed-vs-half-packed
# (in-place A) sweep at M = mr(kernel) exactly. Single core, single node --
# no internal parallelism, so plain sbatch is the right tool (not disBatch).
# Pick the microarchitecture with --constraint, one job per node class, since
# mr(kernel) itself is ISA-dependent (AVX2 vs AVX-512 pick different default
# register shapes) and any win/loss ratio may be too:
#
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_half_packing.sh  # Zen2, AVX2
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_half_packing.sh  # Ice Lake-SP, AVX-512
#   sbatch --reservation=rocky8 --constraint=genoa   benchmark/submit_half_packing.sh  # Zen4, AVX-512
#
# Submit from the repo root. See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
# Check the job's `cpu =` header and the printed kernel= shape per row before
# trusting a conclusion.
#SBATCH --job-name=qs-half-packing
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# See submit_blocking_model.sh / submit_complex_efficiency.sh / submit_skip_packing.sh
# for why this resolution block: on Rocky 8 nodes (--reservation=rocky8) the
# module system's own binaries need a newer glibc than the node has, so
# `module load` fails and there is no `julia`; fall back to $QS_JULIA or a
# juliaup-installed 1.12 on the shared home (official Julia builds need only
# glibc 2.17).
module load modules/2.5-beta1 || true
module load julia/1.12.6 || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac  # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
echo "julia = $JULIA"

export JULIA_NUM_THREADS=1
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()'

"$JULIA" --project=. benchmark/bench_half_packing.jl

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true

echo "=== cpu check (trust the run above only if this matches the requested --constraint) ==="
grep -m1 "cpu = " "benchmark/results/"*"/half_packing_PROVENANCE.txt" || true
