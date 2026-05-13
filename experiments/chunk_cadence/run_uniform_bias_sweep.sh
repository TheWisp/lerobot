#!/usr/bin/env bash
# Uniform-per-chunk bias model: every frame of chunk_i gets the same bias_i,
# so there's no in-chunk discontinuity — only chunk-to-chunk transitions.
# Tests whether the "lookahead+smoothing wins" finding from the overlap-only
# model generalizes, or is specific to that model's internal step artifact.
#
# Cells: L ∈ {0, 80} × smooth_window ∈ {0, 5, 9}. 6 cells per replicate.

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
OUTDIR=outputs/chunk_cadence/safe_uniform_run${RUN_NUM}

mkdir -p "$OUTDIR"

run_cell() {
    local lk="$1"
    local sw="$2"
    local trace_file="$OUTDIR/trace_N4_L${lk}_bias0.5_smooth${sw}_bmu_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: uniform L=${lk} smooth=${sw} ==="
    $VENV "$HERE/backtest.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --update-every 4 \
        --lookahead-ms "$lk" \
        --bias-seed 42 \
        --chunk-smooth-window "$sw" \
        --bias-model uniform \
        --output-dir "$OUTDIR"
}

for lk in 0 80; do
    for sw in 0 5 9; do
        run_cell "$lk" "$sw"
    done
done

echo "=== run${RUN_NUM} uniform-bias done ==="
