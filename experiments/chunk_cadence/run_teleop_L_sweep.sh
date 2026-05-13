#!/usr/bin/env bash
# 200 Hz teleop, L sweep at the predicted-optimal point and beyond.
#
# Hypothesis: L=110 should bring residual state lag close to zero
# (matches τ_follower + τ_leader_sampling). L=130 deliberately exceeds
# the existing max_lookahead_ms cap to verify whether the cap is
# well-chosen (expected: residual stays near zero but jitter rises).

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
    local trace_file="$OUTDIR/trace_teleop_fps200_L${lk}_noise${noise}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: teleop@200Hz L=${lk} noise=${noise} ==="
    $VENV "$HERE/backtest_teleop.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --lookahead-ms "$lk" \
        --intent-noise-deg "$noise" \
        --intent-noise-seed 42 \
        --outer-fps 200 \
        --output-dir "$OUTDIR"
}

# L=110 — predicted optimal (matches existing cap). L=130 — past the cap,
# verifies whether the cap is positioned correctly. backtest_teleop.py
# auto-pins max_lookahead_ms = lookahead_ms so the cap doesn't bite our
# L=130 cell.
for lk in 110 130; do
    for noise in 0.0 0.1; do
        run_cell "$lk" "$noise"
    done
done

echo "=== run${RUN_NUM} L-sweep done ==="
