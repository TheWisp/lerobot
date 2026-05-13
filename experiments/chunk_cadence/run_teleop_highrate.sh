#!/usr/bin/env bash
# Teleop at 200 Hz outer-loop rate — simulates SO107LeaderHighRate sample
# rate. Tests the hypothesis that the 30 Hz outer-loop teleop's elevated
# τ_motor (134 ms vs chunked's 100 ms) is a staircase-pattern artifact
# that disappears with interpolated high-rate intent.

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
    local fps="$3"
    local trace_file="$OUTDIR/trace_teleop_fps${fps}_L${lk}_noise${noise}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: teleop@${fps}Hz L=${lk} noise=${noise} ==="
    $VENV "$HERE/backtest_teleop.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --lookahead-ms "$lk" \
        --intent-noise-deg "$noise" \
        --intent-noise-seed 42 \
        --outer-fps "$fps" \
        --output-dir "$OUTDIR"
}

# 200 Hz teleop: same lookahead grid as the 30 Hz baseline. Need L=0 too
# so we can compare τ_motor.
run_cell 0  0.0  200
run_cell 80 0.0  200
run_cell 80 0.1  200

echo "=== run${RUN_NUM} 200Hz teleop done ==="
