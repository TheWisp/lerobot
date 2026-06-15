# Self-Play Embodiment — staged follow-ups

Recorded 2026-06-13. Ordered; each line is a decided change, not a maybe.

## NOW — base-model SR-vs-K sweep (no world model)

Goal: SR-vs-#demos curves to (a) find the demo-reduction ratio read **horizontally**, (b) see
whether either arm plateaus below ceiling, (c) pick the substrate whose curve sits in a measurable
mid-range (not floored, not saturated at K-min).

- Eval harness: **keep SYNC for the in-dist K-sweep.** Decision (2026-06-13): once OOD is dropped the
  K-sweep is TRAINING-bound (~25 min/cell train vs ~3 min/cell eval), so async (a measured 5× on eval:
  ~1500 vs ~290 env-steps/s at n_envs=32) saves only ~10-15% of total here and isn't worth its cost.
  Cost = a worker-side spawn-override wrapper: even in-dist eval uses the qpos override path for
  protocol parity, which breaks under subprocess envs; LeRobot async per-env reset-options threading
  is unverified. **Build + validate async (vs sync SR) when we next do OOD-heavy runs**, where eval
  dominates and the 5× is real. (n_envs=32 is the plateau; N>=48 OOMs — full mujoco+EGL ctx/worker.)
- **Drop OOD** (f=1 only) — data-efficiency is in-distribution; removes the x4 factor.
- **Success-terminate + cap 300 steps** (reward is max-over-episode; metric-neutral, ~1.6x).
- **Leave render at 480x640** — resolution is ~2% of step cost (measured 0.79 vs 0.73 ms), NOT a lever.
- Grid: **K in {5,10,20,40,80} x 2 seeds x n=96**, insert-rung curve, read horizontally.
- Arms: **ACT (current)** first; **SmolVLA-base** (lerobot/smolvla_base, ~450M, sub-1B) as the
  follow-on comparison. SmolVLA finetunes from a community-pretrained base (not from scratch) — note
  this when reading its curve.

## LATER — when we add ACTION + adjust the model (the affordance line)

- **Continuable self-play collector**: reset-free, recoverability-gated (objects stay on table /
  soft-reset just the escaped object), relaxed max length. Kills the episode-start-anchor confound
  structurally (current H=150 < 128-frame window -> anchors only at episode start) and yields
  compositional diversity. Rehearses the real-robot safety loop.
- **Log actions** in that collection -> v2 HF dataset, written at generation time (store-first).
- **Affordance predictor**: block = `[s_k | z_k | a_k]`, real action on **self-play blocks only** +
  action-dropout (masked on demo blocks to dodge the a-shortcut collapse). Forces z to encode the
  `a -> future` conditional map without the student ever seeing a (leakage-free; a is a cause within
  the prediction window, not an observation of the future). Self-play's random/decorrelated actions
  are what make this identifiable — demos would collapse it.
- **Counterfactual-sensitivity probe** added to the selftest gate: swapped-a must change the
  prediction a lot AND shuffle-z must still hurt. Only the intended solution passes both (action-free
  z fails the first; a-shortcut-collapsed z fails the second). Must pass before any stage-2 run.
- **Resolution/capacity**: finer spatial grid (14x14, or ResNet-34 / DINOv2-S backbone) +/- 6 encoder
  layers. Resolution is the measured limiter (few-pixel peg, 7x7 too coarse at contact), not depth.
  Keep spatial tokens end-to-end; never mean-pool image features for a policy/target.
- **Bring OOD back** for WM experiments — the WM's value shows up disproportionately OOD (E4 + paper).
- **Re-baseline** control+wmDS on whatever substrate is chosen — new encoder/model => E4/E5 numbers
  don't transfer.
- Optional capability fork: action-conditioning also yields a **steerable WM** (V-JEPA2-AC style) for
  MPC/planning — same training run, predictor kept instead of discarded.

## Backlog

- Grasp-capable play (can cheap/failed grasp attempts replace demo video's behavioral content?).
- True GPU-bound eval (MJX/Madrona batched sim) — only if eval becomes the dominant recurring cost.
- Real-world self-play (safety stack: torque/current/temp caps, walled arena, GUI Recover, watchdog).

## Summary/figures polish (later)

- **Draw uncertainty range on the vertical bars** in the summary PDF (reward-ladder grouped bars): show 95% binomial CI whiskers per bar (n per cell), so the reader sees the error range, not just point heights. (The E4 ladder fig has CIs; ensure the final summary's bars carry them consistently across all arms/rungs.)

## Optional follow-ups (LeRobot ACT WM graft, Jun 15)

- [ ] Wide-scene (OOD) representation gate: hold out WIDE self-play episodes for the val/copy + shuffle-z gate (current gate uses random in-distribution held-out from the union corpus). Passed in-distribution: val/copy 0.494, shuffle-z +35% @12k.
- [ ] 3-corpus stage-1 ablation (demos-only / demos+SP / SP-only) to reproduce prototype Result 1 incl. the "demos-only = dead (2.04, +1.3%)" negative control. Current run uses the union corpus only.
