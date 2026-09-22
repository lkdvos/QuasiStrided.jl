#!/bin/bash
# Slurm job for the "evidence gate" full run of benchmark/bench_to_suite.jl
# (docs/proposals/dispatch-tiers.md, section 5.1): StridedBLAS vs
# QuasiStridedBackend over the upstream TensorOperationsBenchmarks
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
# test found StridedNative (not QuasiStrided) had a severe, size-growing
# slowdown on :trg specifically -- 42x slower than StridedBLAS at chi=32, and
# a single un-warmed call did not finish within 100s at chi=48; see
# docs/decisions.md's ":mps/:ctmrg/:trg wiring" section for the numbers.
# StridedNative was dropped from bench_to_suite.jl's BACKENDS entirely
# (2026-09-22, see that file's header) since it was already excluded from
# every plot, so this cap is no longer protecting against a walltime blowup
# -- it's just left in place because nobody has re-validated chi>48 for
# StridedBLAS/QuasiStrided specifically. Raise it if you want that data.
#
# --reps bumped 21 -> 41 (2026-09-21): the first evidence-gate run
# (job 7085230) showed ~27 case timings with a StridedBLAS median stuck at a
# suspicious, size-independent ~11.8-12.0ms floor (reproduced standalone as
# ~500x too slow for the same tensors -- not a real per-call cost, some kind
# of one-off stall on the compute node). A plain median over more reps makes
# a transient stall less likely to still be the median value; --time bumped
# 2h -> 3h to match. The rerun (job 7085608, --reps 41) STILL showed the same
# floor on largely the same cases (down from 27 to 21 of 170), so more reps
# alone does not fix it -- it's a majority-of-calls effect on those specific
# cases, not a rare transient. Root cause not yet found (not reproducible
# standalone on the login node, only inside the Slurm compute-node
# allocation); a real fix belongs in bench_to_suite.jl or the job's
# environment (e.g. pinning OPENBLAS_NUM_THREADS=1), not just more reps.
#
# --dtypes Float32 -> ComplexF64 (2026-09-22): plot_bench_to_suite.jl only
# plots Float64/ComplexF64 now (Float32 out of scope), so collecting Float32
# timings here was wasted walltime; --time bumped 3h -> 4h since ComplexF64
# cases haven't been timed at this full size sweep before (a small smoke
# test at --reps 3 with one size per category found no correctness
# mismatches or backend errors across all 5 categories).
#
# StridedNative dropped from BACKENDS entirely (2026-09-22, same day): job
# 7087143 (this --dtypes change, StridedNative still present) ran past 1.5h
# with zero visible progress past package precompilation -- almost certainly
# StridedNative hitting its known :ctmrg/:trg slowdown (see the --trg-chis
# note above), now likely worse under ComplexF64's larger footprint, for
# numbers plot_bench_to_suite.jl was already discarding. Cancelled and fixed
# at the source (bench_to_suite.jl's BACKENDS) rather than worked around
# here; --time left at 4h as a margin since ComplexF64 timing at this size
# sweep is still otherwise unvalidated.
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
    --dtypes Float64,ComplexF64 \
    --pairwise-sizes 15,63,128 \
    --tccg-sizes 8,16 \
    --mps-bonddims 32,48,64,100,128,256,300,512 \
    --ctmrg-chis 16,24,32,48,64,100 \
    --trg-chis 16,24,32,48 \
    --reps 41

julia --project=benchmark benchmark/plot_bench_to_suite.jl
