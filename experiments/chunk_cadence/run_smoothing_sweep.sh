#!/usr/bin/env bash
# Smoothing sweep: hold update_every=4, L=80, safe trajectory, bias seed=42.
# Vary --chunk-smooth-window ∈ {0, 3, 5, 7, 9} to see how non-causal SG
# smoothing of the chunk frames affects jitter and latency.
#
# Usage: run_smoothing_sweep.sh <run_number>
# Saves to outputs/chunk_cadence/safe_smooth_run<N>/

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
OUTDIR=outputs/chunk_cadence/safe_smooth_run${RUN_NUM}

mkdir -p "$OUTDIR"

run_cell() {
    local sw="$1"
    local trace_file="$OUTDIR/trace_N4_L80_bias0.5_smooth${sw}_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: smooth_window=${sw} ==="
    $VENV "$HERE/backtest.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --update-every 4 \
        --lookahead-ms 80 \
        --bias-seed 42 \
        --chunk-smooth-window "$sw" \
        --output-dir "$OUTDIR"
}

# Include a baseline L=0 control once per replicate so we can confirm
# τ_motor stays unchanged across replicates (sanity check).
baseline_cell() {
    local trace_file="$OUTDIR/trace_N4_L0_bias0.5_smooth0_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: L=0 baseline ==="
    $VENV "$HERE/backtest.py" \
        --source "$SAFE" \
        --robot-profile "$PROFILE" \
        --update-every 4 \
        --lookahead-ms 0 \
        --bias-seed 42 \
        --chunk-smooth-window 0 \
        --output-dir "$OUTDIR"
}

baseline_cell
for sw in 0 3 5 7 9; do
    run_cell "$sw"
done

echo "=== run${RUN_NUM} done ==="
