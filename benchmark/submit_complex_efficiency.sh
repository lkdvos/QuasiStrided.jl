#!/bin/bash
# Slurm job for benchmark/bench_complex_efficiency.jl: the complex-efficiency
# ratio (arm 1) and the planar-vs-1m x register-shape sweep (arm 2). Both arms
# are single-machine measurements by design (see the file's header comments);
# this submission's point is specifically to run arm 2 on AVX2 hardware, which
# src/planning/kernel_selection.jl:144-149 calls out as unmeasured -- the
# AVX2 ComplexF64/ComplexF32 `_shape_override` rows are "modelled rather than
# measured". Single core, single node -- this script has no internal
# parallelism, so plain sbatch is the right tool (not disBatch).
#
# Submit from the repo root:
#   sbatch benchmark/submit_complex_efficiency.sh
#
# Adjust --partition/--time/--mem below for your cluster (Rusty/Popeye) and
# account as needed; see https://wiki.flatironinstitute.org/SCC/Software/Slurm.
# NOTE: this only measures whatever ISA the allocated node actually has --
# check the job's PROVENANCE/env header for `cpu =` before trusting an AVX2
# conclusion from it; resubmit if Slurm lands it on an AVX-512 node instead.
#SBATCH --job-name=qs-complex-efficiency
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

julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. benchmark/bench_complex_efficiency.jl

echo "=== cpu check (trust the run above only if this says an AVX2 uarch) ==="
grep -m1 "cpu = " "benchmark/results/"*"/complex_efficiency_PROVENANCE.txt" || true
