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
# Slurm copies this script into a spool directory before running it, so
# `${BASH_SOURCE[0]}`/`dirname "$0"` point there, not into the repo. Use
# SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# juliaup's own toolchain lives on this workstation's LOCAL disk (/home, not
# /mnt/home -- see this org's CLAUDE.md filesystem table), so it isn't visible
# on a compute node at all, and the site's bare `module load julia` resolves
# to julia/1.11.2, whose depot (this repo's Manifest, precompiled for 1.12)
# fails to precompile under it. `module spider julia/1.12.6` shows it needs a
# newer `modules` meta-module loaded first to become visible.
module load modules/2.5-beta1
module load julia/1.12.6

julia --project=benchmark benchmark/bench_to_suite.jl \
    --categories pairwise \
    --dtypes Float64,Float32 \
    --pairwise-sizes 8,15,32,63,96,128,200,256 \
    --reps 21

julia --project=benchmark benchmark/plot_bench_to_suite.jl
