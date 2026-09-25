#!/bin/bash
# Slurm job for benchmark/bench_blocking_model.jl: the (mc, kc, nc) plateau
# sweep and the analytical blocking model against the shipped constants.
# Single core, single node -- the script has no internal parallelism, so plain
# sbatch is the right tool (not disBatch). Pick the microarchitecture with
# --constraint on the command line, one job per node class:
#
#   sbatch --constraint=rome    benchmark/submit_blocking_model.sh   # Zen2, AVX2
#   sbatch --constraint=genoa   benchmark/submit_blocking_model.sh   # Zen4, AVX-512
#   sbatch --constraint=icelake benchmark/submit_blocking_model.sh   # Ice Lake-SP, AVX-512
#
# Submit from the repo root. See https://wiki.flatironinstitute.org/SCC/Software/Slurm.
# Check the job's `cpu =` / `target =` header before trusting a conclusion.
#SBATCH --job-name=qs-blocking-model
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm runs a spooled copy of this script; SLURM_SUBMIT_DIR is the repo root.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# See submit_complex_efficiency.sh for why these two modules.
module load modules/2.5-beta1
module load julia/1.12.6

export JULIA_NUM_THREADS=1
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. benchmark/bench_blocking_model.jl --scalar

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name\|cache" || true
