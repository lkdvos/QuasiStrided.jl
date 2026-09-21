#!/bin/bash
# Slurm job for the "evidence gate" full run of benchmark/bench_to_suite.jl
# (docs/proposals/dispatch-tiers.md, section 5.1): StridedNative vs
# StridedBLAS vs QuasiStridedBackend over the upstream TensorOperationsBenchmarks
# :pairwise/:tccg cases plus the newly-wired :mps/:ctmrg/:trg tensor-network
# categories, at full reps. Single core, single node -- this script has no
# internal parallelism, so plain sbatch is the right tool (not disBatch).
#
# Submit from the repo root:
#   sbatch benchmark/submit_evidence_gate.sh
#
# Adjust --partition/--time/--mem below for your cluster (Rusty/Popeye) and
# account as needed; see https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#
# NOTE on --trg-chis: deliberately capped at 48 here, narrower than upstream
# TensorOperationsBenchmarks' own default sweep (which goes to 96). A smoke
# test found StridedNative (not QuasiStrided) has a severe, size-growing
# slowdown on :trg specifically -- 42x slower than StridedBLAS at chi=32, and
# a single un-warmed call did not finish within 100s at chi=48. Raising this
# past 48 risks the StridedNative arm alone blowing the walltime budget below;
# see docs/decisions.md's ":mps/:ctmrg/:trg wiring" section for the numbers.
# If you want chi>48 data, either drop StridedNative from that category first
# (a script change, not made here) or give it its own generous time budget.
#SBATCH --job-name=qs-evidence-gate
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

julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

julia --project=benchmark benchmark/bench_to_suite.jl \
    --categories pairwise,tccg,mps,ctmrg,trg \
    --dtypes Float64,Float32 \
    --pairwise-sizes 15,63,128 \
    --tccg-sizes 8,16 \
    --mps-bonddims 32,48,64,100,128,256,300,512 \
    --ctmrg-chis 16,24,32,48,64,100 \
    --trg-chis 16,24,32,48 \
    --reps 21

julia --project=benchmark benchmark/plot_bench_to_suite.jl
