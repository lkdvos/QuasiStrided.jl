#!/bin/bash
# Slurm job for a full-size benchmark/bench_to_suite.jl run (StridedNative vs
# StridedBLAS vs QuasiStridedBackend over the upstream TensorOperationsBenchmarks
# :pairwise/:tccg cases). Single core, single node -- this script has no
# internal parallelism, so plain sbatch is the right tool (not disBatch).
#
# Submit from the repo root:
#   sbatch benchmark/submit_bench_to_suite.sh
#
# Adjust --partition/--time/--mem below for your cluster (Rusty/Popeye) and
# account as needed; see https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#SBATCH --job-name=qs-bench-to-suite
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

module load julia 2>/dev/null || true

julia --project=benchmark benchmark/bench_to_suite.jl \
    --categories pairwise \
    --dtypes Float64,Float32 \
    --pairwise-sizes 8,15,32,63,96,128,200,256 \
    --reps 21

julia --project=benchmark benchmark/plot_bench_to_suite.jl
