#!/usr/bin/env bash
# Companion test: does smoothing help at L=0 too? Helps disentangle
# "lookahead-specific benefit" from "general non-causal smoothing benefit".
# Same chunk seed, same trajectory, same N — only varies L and smooth_window.

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

RUN_NUM=$1
if [ -z "$RUN_NUM" ]; then
    echo "Usage: $0 <run_number>"
    exit 1
fi

VENV=${VENV:-.venv/bin/python}
PROFILE=${ROBOT_PROFILE:-$HOME/.config/lerobot/robots/white_pred.json}
SAFE=${SAFE_SOURCE:-$HOME/.config/lerobot/robots/white.trajectory.json}
OUTDIR=outputs/chunk_cadence/safe_smooth_run${RUN_NUM}  # write into same dir

mkdir -p "$OUTDIR"

run_cell() {
    local lk="$1"
    local sw="$2"
    local trace_file="$OUTDIR/trace_N4_L${lk}_bias0.5_smooth${sw}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: L=${lk} smooth=${sw} ==="
    $VENV "$HERE/backtest.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --update-every 4 \
        --lookahead-ms "$lk" \
        --bias-seed 42 \
        --chunk-smooth-window "$sw" \
        --output-dir "$OUTDIR"
}

# Fill in the L=0 × smooth ∈ {3,5,7,9} grid (the L=0+smooth=0 baseline and
# all L=80 cells were already produced by run_smoothing_sweep.sh).
for sw in 3 5 7 9; do
    run_cell 0 "$sw"
done

echo "=== run${RUN_NUM} L=0 smoothing done ==="
