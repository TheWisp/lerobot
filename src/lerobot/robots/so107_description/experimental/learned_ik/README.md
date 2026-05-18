# SO-107 learned IK

Tiny MLP trained on existing `bi_so107_follower` episodes to produce
"human-natural" joint configurations for Cartesian teleop. Sidesteps the
redundancy / weird-pose issue that analytical IK (placo SLSQP, raw DLS) hits
because it implicitly learns the joint configurations that humans actually
chose when teleoperating the same arm.

## Concept

Each tick of teleop is essentially a question:

> Given that I'm currently at joints `q`, and I want my EE to move by `Δee`,
> what should my next joint config be?

Analytical IK answers this with math: minimum-norm `Δq` solving `J·Δq = Δee`.
Mathematically correct but doesn't know about _human-natural_ posture
preferences (elbow up vs. down, wrist neutral vs. twisted, etc.).

Learned IK trains a network on `(q_t, ee_{t+1} − ee_t) → q_{t+1} − q_t`
extracted from real teleop demonstrations. The network's output is the
joint delta a human-controlled leader arm produced for that EE move.

## Pipeline

```
Datasets (lerobot bi_so107_follower episodes)
        │
        │  dataset_extractor.py
        ▼
.npz of (joints_t, joints_t+1, ee_t, ee_t+1) pairs
        │
        │  train.py  (10→128→128→7 MLP, SmoothL1, AdamW, cosine LR)
        ▼
.pt checkpoint
        │
        │  kinematics_nn.py loads checkpoint
        ▼
So107NNKinematics (drop-in for So107Kinematics)
        │
        │  teleop_keyboard_nn.py uses it
        ▼
Real-arm Cartesian teleop
```

## Quick start

```bash
# 1a. Reserve the longest N episodes for eval (typical complete trajectories;
#     better OOD signal than random sampling).
.venv/bin/python -m lerobot.robots.so107_description.experimental.learned_ik.dataset_extractor \
    --longest-n 3 --out /tmp/so107_ik_eval.npz

# 1b. Extract everything else for training.
.venv/bin/python -m lerobot.robots.so107_description.experimental.learned_ik.dataset_extractor \
    --skip-longest-n 3 --out /tmp/so107_ik_train.npz

# 2. Train (a few minutes on CPU, seconds on GPU).
.venv/bin/python -m lerobot.robots.so107_description.experimental.learned_ik.train \
    --data /tmp/so107_ik_train.npz --out /tmp/so107_ik_model.pt

# 3. Eval on the longest-N reserved episodes (true held-out).
.venv/bin/python -m lerobot.robots.so107_description.experimental.learned_ik.eval \
    --model /tmp/so107_ik_model.pt \
    --eval-data /tmp/so107_ik_eval.npz \
    --train-data /tmp/so107_ik_train.npz

# 4. Teleop with it.
.venv/bin/python -m lerobot.robots.so107_description.teleop_keyboard_nn \
    --port /dev/ttyACM2 --id right_white \
    --model /tmp/so107_ik_model.pt
```

## Hybrid (NN + DLS refinement)

By default, `teleop_keyboard_nn` runs the NN then does 2 DLS Newton steps to
pin the EE position to the target exactly. The NN provides "good initial
guess" (human-natural posture); DLS provides "guaranteed EE accuracy."

Pass `--no-dls-refine` to disable refinement and see pure NN behavior. The
NN alone produces decent motion but its EE accuracy depends on training
data coverage at the queried point.

## Why this should work

- 200k+ training pairs from real teleop demos
- Same physical embodiment (bi_so107_follower) — same joint geometry,
  same offsets when humans operate the arm
- Tiny model fits in <1MB, inference is microseconds
- Cannot violate physics: NN can't make singular configs reachable, but
  can route around regions humans don't teleop into

## Limitations

- Out-of-distribution behavior is unpredictable. If you try to move the
  arm somewhere no human ever drove it, the NN may pick weird joints.
- Calibration sensitivity. The training data assumes the SAME bridge
  offsets as `RIGHT_ARM_MAP`. If a future re-calibration changes those,
  the model needs retraining or a calibration-invariant input encoding.
- The bridge is applied at FK time inside the extractor. If our bridge
  numbers are off, the EE labels in training data are off the same way —
  but training and inference are self-consistent, so it shouldn't hurt
  motion quality.
