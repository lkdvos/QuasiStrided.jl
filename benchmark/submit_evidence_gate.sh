#!/bin/bash
# Slurm job for the "evidence gate" full run of benchmark/bench_to_suite.jl:
# StridedBLAS vs
# QuasiStridedBackend over the upstream TensorOperationsBenchmarks
# :pairwise/:tccg cases plus the :mps/:ctmrg/:trg tensor-network
# categories, at full reps. Single core, single node -- this script has no
# internal parallelism, so plain sbatch is the right tool (not disBatch).
#
# Submit from the repo root:
#   sbatch benchmark/submit_evidence_gate.sh
#
# Adjust --partition/--time/--mem below for your cluster (Rusty/Popeye) and
# account as needed; see https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#
# --trg-chis 16..48: capped at 48, narrower than upstream
# TensorOperationsBenchmarks' own default sweep (which goes to 96).
# StridedNative has a severe, size-growing slowdown on :trg (42x slower than
# StridedBLAS at chi=32; a single un-warmed call does not finish within 100s
# at chi=48), which is why it is not in bench_to_suite.jl's BACKENDS. chi>48
# has not been validated for StridedBLAS/QuasiStrided; raise the cap if you
# want that data.
#
# --reps 41: some StridedBLAS case timings (~21 of 170) show a suspicious,
# size-independent ~11.8-12.0ms median floor on the compute node (~500x the
# standalone per-call cost for the same tensors). More reps make a transient
# stall less likely to be the median, but the floor persists at 41 reps on
# largely the same cases, so it is a majority-of-calls effect, not a rare
# transient. Root cause unknown (it reproduces only inside the Slurm
# compute-node allocation, not standalone on the login node); a real fix
# belongs in bench_to_suite.jl or the job's environment (e.g. pinning
# OPENBLAS_NUM_THREADS=1), not more reps.
#
# --dtypes Float64,ComplexF64: plot_bench_to_suite.jl plots only these, so
# Float32 timings would be wasted walltime.
#
# --time 04:00:00: a margin for the ComplexF64 cases at this full size sweep,
# whose walltime is not otherwise validated (a smoke test at --reps 3 with one
# size per category found no correctness mismatches or backend errors across
# all 5 categories).
#SBATCH --job-name=qs-evidence-gate
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=benchmark/results/slurm-%j.out

set -euo pipefail
# Slurm copies this script into a spool directory before running it, so
# `${BASH_SOURCE[0]}`/`dirname "$0"` point there, not into the repo. Use
# SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- run this script via sbatch, not directly}"

# juliaup's own toolchain lives on this workstation's LOCAL disk (/home, not
# the shared /mnt/home), so it isn't visible
# on a compute node at all, and the site's bare `module load julia` resolves
# to julia/1.11.2, whose depot (this repo's Manifest, precompiled for 1.12)
# fails to precompile under it. `module spider julia/1.12.6` shows it needs a
# newer `modules` meta-module loaded first to become visible.
module load modules/2.5-beta1
module load julia/1.12.6

julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

julia --project=benchmark benchmark/bench_to_suite.jl \
    --categories pairwise,tccg,mps,ctmrg,trg \
    --dtypes Float64,ComplexF64 \
    --pairwise-sizes 15,63,128 \
    --tccg-sizes 8,16 \
    --mps-bonddims 32,48,64,100,128,256,300,512 \
    --ctmrg-chis 16,24,32,48,64,100 \
    --trg-chis 16,24,32,48 \
    --reps 41

julia --project=benchmark benchmark/plot_bench_to_suite.jl
