#!/usr/bin/env bash
# corrector_alpha sweep at 200 Hz teleop, L=110 (the sweet spot), noise=0.1
# (visible shake regime). Tests whether the predictor-corrector smoothing
# can claw back jitter at acceptable latency cost.
#
# alpha=1.0 = no blend, pure responsive (current default)
# alpha<1.0 = blend in action-history predictor (causal LP on output,
#             adds ~(1-α)/α control ticks of latency)

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
    local alpha="$1"
    local trace_file="$OUTDIR/trace_teleop_fps200_L110_noise0.1_alpha${alpha}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: teleop@200Hz L=110 noise=0.1 alpha=${alpha} ==="
    $VENV "$HERE/backtest_teleop.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --lookahead-ms 110 \
        --intent-noise-deg 0.1 \
        --intent-noise-seed 42 \
        --outer-fps 200 \
        --corrector-alpha "$alpha" \
        --output-dir "$OUTDIR"
}

for alpha in 1.0 0.7 0.5 0.3; do
    run_cell "$alpha"
done

echo "=== run${RUN_NUM} alpha-sweep done ==="
