#!/usr/bin/env bash
# Cadence sweep, one replicate. Usage: run_sweep.sh <run_number>
# Saves to outputs/chunk_cadence/{safe,cylinder}_run<N>/
#
# Override env vars to point at a different robot or trajectory source:
#   ROBOT_PROFILE   path to GUI robot profile JSON  (default: ~/.config/lerobot/robots/white_pred.json)
#   SAFE_SOURCE     path to safe-trajectory JSON OR a dataset spec "<repo_id>@<ep_idx>"
#   CYLINDER_SOURCE same shape
# Skips cells whose .npz already exists — idempotent / resumable.

set -e

# Resolve repo root from the script location so this works regardless of CWD.
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
CYLINDER=${CYLINDER_SOURCE:-thewisp/cylinder_ring_assembly@287}
OUTDIR_SAFE=outputs/chunk_cadence/safe_run${RUN_NUM}
OUTDIR_CYLINDER=outputs/chunk_cadence/cylinder_run${RUN_NUM}

mkdir -p "$OUTDIR_SAFE" "$OUTDIR_CYLINDER"

run_cell() {
    local source="$1"
    local outdir="$2"
    local n="$3"
    local lk="$4"
    local trace_file="$outdir/trace_N${n}_L${lk}_bias0.5_seed42.npz"
    if [ -f "$trace_file" ]; then
        echo "[skip] $trace_file"
        return
    fi
    echo "=== run${RUN_NUM}: source=$(basename "$source") N=$n L=$lk ==="
    $VENV "$HERE/backtest.py" \
        --source "$source" \
        --robot-profile "$PROFILE" \
        --update-every "$n" \
        --lookahead-ms "$lk" \
        --bias-seed 42 \
        --output-dir "$outdir"
}

for n in 2 4 8 16; do
    for lk in 0 80; do
        run_cell "$SAFE" "$OUTDIR_SAFE" "$n" "$lk"
    done
done

for n in 2 4 8 16; do
    for lk in 0 80; do
        run_cell "$CYLINDER" "$OUTDIR_CYLINDER" "$n" "$lk"
    done
done

echo "=== run${RUN_NUM} done ==="
