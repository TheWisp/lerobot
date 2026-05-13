#!/usr/bin/env bash
# Cylinder ring assembly ep 287 — both bias models × smoothing sweep.
# Tests whether the safe-trajectory finding (L=80 + smoothing dominates)
# holds on a more dynamic, real-task trajectory.
#
# Cells per replicate: 2 bias models × 2 lookahead × 3 smooth_window = 12

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
CYLINDER=${CYLINDER_SOURCE:-thewisp/cylinder_ring_assembly@287}
OUTDIR=outputs/chunk_cadence/cylinder_smooth_run${RUN_NUM}

mkdir -p "$OUTDIR"

run_cell() {
    local lk="$1"
    local sw="$2"
    local bm="$3"
    local bm_tag
    bm_tag=$(case "$bm" in overlap_only) echo o ;; uniform) echo u ;; esac)
    local trace_file="$OUTDIR/trace_N4_L${lk}_bias0.5_smooth${sw}_bm${bm_tag}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: cylinder bm=${bm} L=${lk} smooth=${sw} ==="
    $VENV "$HERE/backtest.py" \
        --source "$CYLINDER" \
        --robot-profile "$PROFILE" \
        --update-every 4 \
        --lookahead-ms "$lk" \
        --bias-seed 42 \
        --chunk-smooth-window "$sw" \
        --bias-model "$bm" \
        --output-dir "$OUTDIR"
}

for bm in overlap_only uniform; do
    for lk in 0 80; do
        for sw in 0 5 9; do
            run_cell "$lk" "$sw" "$bm"
        done
    done
done

echo "=== run${RUN_NUM} cylinder smoothing done ==="
