#!/bin/bash
# Slurm job for benchmark/profile_to_suite.jl: profiles QuasiStridedBackend()
# vs StridedBLAS() (bucketed into adapter/planning/packing/microkernel/store/
# blas/etc.) on a hand-picked case list, not the full CASES/DIRECT_CASES grid
# (that grid is 43 cases x 2 backends, most of them either already-competitive
# or overhead-dominated at microsecond scale -- not worth the wall-clock).
# The list below was chosen from job 7101019's bench_to_suite.csv (evidence
# gate run, 2026-09-24): the worst real (non-noise) QS/BLAS regression
# (dim63_2_1_2, added to profile_to_suite.jl's CASES for this), plus the
# existing ccsd_t_1_dim16/ccsd_6_dim16 cases as QS-wins contrast. Single
# core, single node -- this script has no internal parallelism, so plain
# sbatch is the right tool (not disBatch).
#
# Submit from the repo root:
#   sbatch benchmark/submit_profile_to_suite.sh
#
# Adjust --partition/--time/--mem below for your cluster (Rusty/Popeye) and
# account as needed; see https://wiki.flatironinstitute.org/SCC/Software/Slurm.
#
# --tag $SLURM_JOB_ID: profile_to_suite.jl overwrites each case's output
# files on every run (fixed filenames under results_dir()); tagging by job id
# keeps this run's artefacts from clobbering a previous one's.
#SBATCH --job-name=qs-profile-to-suite
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

julia --project=benchmark benchmark/profile_to_suite.jl --tag "${SLURM_JOB_ID:-manual}" \
    ccsd_t_1_dim16 ccsd_t_1_dim16_c64 \
    ccsd_6_dim16 ccsd_6_dim16_c64 \
    dim15_2_2_2 \
    dim63_2_1_2 dim63_2_1_2_c64
