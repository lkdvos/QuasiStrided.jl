#!/usr/bin/env bash
# Reports which Julia versions/toolchains are available on this machine.
#
# Usage: bash benchmark/probes/probe_julia_versions.sh
# Writes benchmark/results/<hostname>-<date>/probes_julia_versions.txt
set -uo pipefail

OUTDIR="$(dirname "$0")/../results/$(hostname)-$(date +%Y-%m-%d)"
mkdir -p "$OUTDIR"
OUTPATH="$OUTDIR/probes_julia_versions.txt"

{
    echo "hostname = $(hostname)"
    echo "date = $(date -Iseconds)"
    echo
    echo "--- which -a julia ---"
    which -a julia 2>&1
    echo
    echo "--- julia --version ---"
    julia --version 2>&1
    echo
    echo "--- ls ~/.julia/juliaup ---"
    ls ~/.julia/juliaup 2>&1
    echo
    echo "--- module avail julia (head -20) ---"
    module avail julia 2>&1 | head -20
    echo
    echo "--- command -v juliaup ---"
    command -v juliaup 2>&1
    echo
    echo "--- juliaup list channels (if available) ---"
    juliaup list 2>&1 || echo "(juliaup not usable)"
} | tee "$OUTPATH"

echo
echo "Wrote $OUTPATH"
