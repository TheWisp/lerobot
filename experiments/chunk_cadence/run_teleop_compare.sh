#!/usr/bin/env bash
# Compare chunk-based command jitter vs. teleop velocity-LSQ-extrapolation
# jitter on the same trajectory, same robot, same lookahead value.
#
# Cells per replicate:
#   1. teleop, L=0           (single-frame intent stream, no lookahead — pure trajectory replay)
#   2. teleop, L=80, noise=0 (LSQ extrapolation, clean intent — best-case teleop)
#   3. teleop, L=80, noise=0.1 (LSQ extrapolation, 0.1 deg leader-noise sim)
#   4. teleop, L=80, noise=0.5 (LSQ extrapolation, larger noise — bracketing)
#
# Chunked baselines come from earlier sweeps (safe_smooth_run*) so don't
# need re-running here.

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
OUTDIR=outputs/chunk_cadence/teleop_run${RUN_NUM}

mkdir -p "$OUTDIR"

run_cell() {
    local lk="$1"
    local noise="$2"
    local trace_file="$OUTDIR/trace_teleop_L${lk}_noise${noise}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: teleop L=${lk} noise=${noise} ==="
    $VENV "$HERE/backtest_teleop.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --lookahead-ms "$lk" \
        --intent-noise-deg "$noise" \
        --intent-noise-seed 42 \
        --output-dir "$OUTDIR"
}

run_cell 0  0.0
run_cell 80 0.0
run_cell 80 0.1
run_cell 80 0.5

echo "=== run${RUN_NUM} teleop done ==="
