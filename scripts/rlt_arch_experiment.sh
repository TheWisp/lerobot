#!/usr/bin/env bash
# RLT token architecture experiment: 4-layer vs 2-layer baseline.
#
# Runs train_token.py with --encoder-layers 4 --decoder-layers 4 in the
# background, polls the output dir for new checkpoints, probes each one
# against the held-out data, compares against the 2-layer baseline curve,
# and early-stops if the 2k checkpoint isn't bending the curve downward.
#
# Safe to exit your terminal after launching — ``nohup``-detached. Read
# ``experiment_log.txt`` when you come back.
#
# Usage:
#   nohup bash scripts/rlt_arch_experiment.sh > /dev/null 2>&1 &
#   tail -f outputs/rlt_token_v3_4layer/experiment_log.txt

set -u

OUTPUT_DIR=${OUTPUT_DIR:-outputs/rlt_token_v3_4layer}
S1_CKPT=${S1_CKPT:-/home/feit/Documents/lerobot/outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model}
DATASET=${DATASET:-thewisp/cylinder_ring_assembly}
STEPS=${STEPS:-10000}
SAVE_FREQ=${SAVE_FREQ:-1000}
ENC_LAYERS=${ENC_LAYERS:-4}
DEC_LAYERS=${DEC_LAYERS:-4}
# 4-layer enc/dec at batch=64 OOMs on a 32GB 5090 (activations + grads
# ~28GB). Halve to batch=32 to fit. Different per-sample gradient variance
# than the 2-layer batch=64 baseline, but the relative trend of
# "does depth help?" is preserved for the early-stop decision.
BATCH_SIZE=${BATCH_SIZE:-32}
# Widened bottleneck (paper spec: 2048). When unset, we train at
# whatever S1's hidden_dim is (symmetric, fits in RLTConfig default).
# When set, the encoder gains input/output projections; memory scales
# with d^2, so pair with a smaller BATCH_SIZE (e.g. 8-16 for d=2048).
RL_TOKEN_DIM=${RL_TOKEN_DIM:-}
# PyTorch allocator: reduces fragmentation-OOMs on long training runs.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Reference curve for "previous best arch we're trying to beat".
# - If you set BASELINE=4L in the env, compares against the 4-layer d=768
#   run (when measuring the widened-bottleneck experiment).
# - If BASELINE=2L (or unset), compares against the original 2-layer run.
# Early-stop threshold is derived from whichever curve is active.
BASELINE=${BASELINE:-2L}
declare -A BASELINE_CURVE
case "$BASELINE" in
    2L)
        BASELINE_CURVE[1000]=50.9
        BASELINE_CURVE[2000]=48.0  # interpolated
        BASELINE_CURVE[3000]=45.0
        BASELINE_CURVE[5000]=41.4
        BASELINE_CURVE[7000]=39.1
        BASELINE_CURVE[10000]=36.4
        EARLY_STOP_DEFAULT=45.0   # beat 2L @ 3k by step 2000
        ;;
    4L)
        # From the 4-layer d=768 run (outputs/rlt_token_v3_4layer/).
        BASELINE_CURVE[1000]=50.6
        BASELINE_CURVE[2000]=44.1
        BASELINE_CURVE[3000]=40.3
        BASELINE_CURVE[4000]=37.1
        BASELINE_CURVE[5000]=35.3
        BASELINE_CURVE[6000]=32.7
        BASELINE_CURVE[7000]=30.8
        BASELINE_CURVE[9000]=29.4
        BASELINE_CURVE[10000]=29.8
        EARLY_STOP_DEFAULT=40.0   # beat 4L @ 3k by step 2000 (44->40 is a real lead)
        ;;
    *)
        echo "BASELINE must be 2L or 4L, got $BASELINE" >&2
        exit 2
        ;;
esac
EARLY_STOP_STEP=2000
EARLY_STOP_THRESHOLD=${EARLY_STOP_THRESHOLD:-$EARLY_STOP_DEFAULT}

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/experiment_log.txt"
TRAIN_LOG="$OUTPUT_DIR/training.stdout"

# Activate conda env — training requires it
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate lerobot 2>/dev/null || true

log()  { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

log "=== RLT arch experiment starting ==="
log "output_dir=$OUTPUT_DIR  enc_layers=$ENC_LAYERS  dec_layers=$DEC_LAYERS  rl_token_dim=${RL_TOKEN_DIM:-default}"
log "steps=$STEPS  save_freq=$SAVE_FREQ  batch_size=$BATCH_SIZE"
log "early-stop at step $EARLY_STOP_STEP if rel-err >= ${EARLY_STOP_THRESHOLD}%"

# Launch training, detach from this shell
train_args=(
    --s1-checkpoint "$S1_CKPT"
    --dataset-repo-id "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --steps "$STEPS"
    --save-freq "$SAVE_FREQ"
    --encoder-layers "$ENC_LAYERS"
    --decoder-layers "$DEC_LAYERS"
    --batch-size "$BATCH_SIZE"
)
if [ -n "$RL_TOKEN_DIM" ]; then
    train_args+=(--rl-token-dim "$RL_TOKEN_DIM")
fi
nohup python -u -m lerobot.policies.hvla.rlt.train_token \
    "${train_args[@]}" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
log "training launched, PID=$TRAIN_PID  (tail $TRAIN_LOG for stdout)"

# Startup check — if the process dies within 20s, startup failed
sleep 20
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "ERROR: training process exited within 20s. Last stdout:"
    tail -30 "$TRAIN_LOG" | tee -a "$LOG"
    exit 1
fi
log "training alive after 20s warm-up; entering monitor loop"

probed=""
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    for ckpt in "$OUTPUT_DIR"/checkpoint-*; do
        [ -d "$ckpt" ] || continue
        step=$(basename "$ckpt" | sed 's/checkpoint-//')
        # Already handled this step?
        [[ " $probed " == *" $step "* ]] && continue
        # Save may be in flight — require both state_dicts + manifest
        [ -f "$ckpt/encoder.pt" ] && [ -f "$ckpt/decoder.pt" ] && [ -f "$ckpt/config.json" ] || continue

        log "probing checkpoint-$step ..."
        probe_out=$(timeout 180 python scripts/rlt_token_probe.py \
            --s1-checkpoint "$S1_CKPT" \
            --rl-token-checkpoint "$ckpt" \
            --dataset "$DATASET" \
            --batches 10 --batch-size 4 2>&1 \
            | grep -E 'relative error|RLT arch' | head -3)
        rel_err=$(echo "$probe_out" | grep 'relative error' | grep -oE '[0-9]+\.[0-9]+%' | head -1)
        base="${BASELINE_CURVE[$step]-n/a}"
        if [ -n "$rel_err" ]; then
            log "  step=$step  4L=$rel_err   ${BASELINE}-baseline=${base}%"
        else
            log "  step=$step  probe FAILED:"
            echo "$probe_out" | sed 's/^/    /' | tee -a "$LOG"
        fi
        probed="$probed $step"

        # Early-stop evaluation at step 2000
        if [ "$step" = "$EARLY_STOP_STEP" ] && [ -n "$rel_err" ]; then
            rel_num=${rel_err%\%}
            if python3 -c "import sys; sys.exit(0 if float('$rel_num') >= $EARLY_STOP_THRESHOLD else 1)"; then
                log ""
                log "EARLY STOP: step $EARLY_STOP_STEP rel-err ($rel_err) >= threshold (${EARLY_STOP_THRESHOLD}%)."
                log "Depth alone isn't bending the curve. Killing training PID $TRAIN_PID."
                log "Suggested next experiments: bump rl_token_dim (768->2048), or joint S1 fine-tune."
                kill "$TRAIN_PID" 2>/dev/null
                break 2
            else
                log ""
                log "CONTINUE: step $EARLY_STOP_STEP rel-err ($rel_err) < threshold (${EARLY_STOP_THRESHOLD}%)."
                log "4-layer arch is helping. Letting training run to $STEPS steps."
            fi
        fi
    done
    sleep 30
done

log ""
log "training process done (PID $TRAIN_PID) — running final sweep for any checkpoints saved between polls"

# Final-sweep: when training exits normally it saves checkpoint-N at the
# last step just before exit. The monitor loop runs at 30s cadence, so
# that final checkpoint can slip through. Probe anything unprobed now
# before emitting the summary.
for ckpt in "$OUTPUT_DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    [[ " $probed " == *" $step "* ]] && continue
    [ -f "$ckpt/encoder.pt" ] && [ -f "$ckpt/decoder.pt" ] && [ -f "$ckpt/config.json" ] || continue
    log "probing checkpoint-$step (post-training final sweep) ..."
    probe_out=$(timeout 180 python scripts/rlt_token_probe.py \
        --s1-checkpoint "$S1_CKPT" \
        --rl-token-checkpoint "$ckpt" \
        --dataset "$DATASET" \
        --batches 10 --batch-size 4 2>&1 \
        | grep -E 'relative error|RLT arch' | head -3)
    rel_err=$(echo "$probe_out" | grep 'relative error' | grep -oE '[0-9]+\.[0-9]+%' | head -1)
    base="${BASELINE_CURVE[$step]-n/a}"
    if [ -n "$rel_err" ]; then
        log "  step=$step  4L=$rel_err   ${BASELINE}-baseline=${base}%"
    else
        log "  step=$step  probe FAILED"
    fi
done

log ""
log "=== experiment complete ==="
log "Final probe curve (from this experiment_log):"
grep -E 'step=[0-9]+ *4L=' "$LOG" | sed 's/^/  /' | tee -a "$LOG"
log "Check $TRAIN_LOG for training stdout (loss trajectory)."
