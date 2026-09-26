#!/bin/bash
# Slurm job: FMAddSubKernel (src/microkernels/fmaddsub.jl) against PlanarKernel
# and OneMKernel, on real hardware. Two steps, both single core:
#
#   1. benchmark/probes/fmaddsub_codegen.jl at the NODE's native target --
#      confirms `vfmaddsub` selection and spill counts where the timing runs.
#   2. benchmark/bench_complex_efficiency.jl arm 2 only (QS_COMPLEX_ARMS=2):
#      planar x 1m x fmaddsub over every menu shape, with the harness's canary
#      (spread must be < 10% before any ranking is believed).
#
# One job per node class, microarchitecture picked on the command line; submit
# from the repo root:
#
#   sbatch --reservation=rocky8 --constraint=rome    benchmark/submit_fmaddsub.sh  # Zen2, AVX2
#   sbatch --reservation=rocky8 --constraint=icelake benchmark/submit_fmaddsub.sh  # Ice Lake-SP, AVX-512
#
# Plain sbatch, not disBatch: one process, no internal parallelism. See
# https://wiki.flatironinstitute.org/SCC/Software/Slurm.
# Trust a result only if the `cpu =` line matches the constraint (checked at
# the end: rome -> znver2, icelake -> icelake-server).
#SBATCH --job-name=qs-fmaddsub
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
export QS_RESULTS_DIR="benchmark/results/fmaddsub-${SLURM_JOB_ID}"
mkdir -p "$QS_RESULTS_DIR"
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()'

echo "=== step 1: instruction selection on this node ==="
"$JULIA" --project=. benchmark/probes/fmaddsub_codegen.jl | tee "$QS_RESULTS_DIR/codegen.txt"

echo "=== step 2: planar x 1m x fmaddsub sweep (arm 2) ==="
QS_COMPLEX_ARMS=2 "$JULIA" --project=. benchmark/bench_complex_efficiency.jl

echo "=== node check ==="
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST:-?}"
lscpu | grep -i "model name" || true
CPU=$(grep -m1 "^cpu = " "$QS_RESULTS_DIR/complex_efficiency_PROVENANCE.txt" | awk '{print $3}')
echo "cpu = $CPU"
case "${SLURM_JOB_CONSTRAINTS:-}" in
    *rome*)    want=znver2 ;;
    *icelake*) want=icelake-server ;;
    *)         want="" ;;
esac
if [ -n "$want" ] && [ "$CPU" != "$want" ]; then
    echo "WARNING: cpu = $CPU but constraint '${SLURM_JOB_CONSTRAINTS}' expects $want -- do not trust this run" >&2
fi
grep "canary spread" "$QS_RESULTS_DIR/complex_efficiency_PROVENANCE.txt" || true
