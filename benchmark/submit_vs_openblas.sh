#!/bin/bash
# Slurm job: OpenBLAS ZGEMM/CGEMM (and DGEMM/SGEMM) reference next to
# QuasiStrided's default path and best planar / 1m / fmaddsub menu shape, same
# process, same node, same shapes, back to back per shape
# (benchmark/bench_vs_openblas.jl). Single core: harness.jl pins
# BLAS.set_num_threads(1) and JULIA_NUM_THREADS=1.
#
# One job per node class; submit from the repo root:
#
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_vs_openblas.sh  # Zen2 -> OpenBLAS core ZEN
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_vs_openblas.sh  # Ice Lake-SP -> OpenBLAS core SKYLAKEX
#
# Plain sbatch, not disBatch: one process, no internal parallelism. See
# https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-vs-openblas
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repo root.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# Julia resolution copied from submit_blocking_model.sh (analytical branch):
# on Rocky 8 nodes (`--reservation=rocky8`) the module system's binaries need a
# newer glibc than the node has, so `module load` yields no julia; fall back to
# $QS_JULIA or a juliaup-installed 1.12 on the shared home (official Julia
# builds need only glibc 2.17).
module load modules/2.5-beta1 2>/dev/null || true
module load julia/1.12.6 2>/dev/null || true
MODJULIA=$(command -v julia 2>/dev/null || true)
case "$MODJULIA" in /mnt/sw/*) ;; *) MODJULIA="" ;; esac  # only a module-provided one
JULIA=${QS_JULIA:-${MODJULIA:-$(ls -d "$HOME"/.julia/juliaup/julia-1.12*/bin/julia 2>/dev/null | tail -1)}}
[ -x "${JULIA:-}" ] || { echo "no julia found; set QS_JULIA" >&2; exit 127; }
echo "julia = $JULIA"

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QS_RESULTS_DIR="benchmark/results/vs-openblas-${SLURM_JOB_ID}"
mkdir -p "$QS_RESULTS_DIR"
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()'

"$JULIA" --project=. benchmark/bench_vs_openblas.jl | tee "$QS_RESULTS_DIR/stdout.txt"

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name" || true
grep -E "^(cpu|BLAS) = |canary spread" "$QS_RESULTS_DIR/vs_openblas_PROVENANCE.txt" || true
