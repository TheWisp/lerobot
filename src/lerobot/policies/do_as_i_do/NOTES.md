# do_as_i_do — PARKED 2026-06-21

An experimental prototype, **parked**. Two ideas were explored here; both are
shelved because the **SO-107 is too inaccurate** for the execution half (encoder

- backlash error, an imperfect URDF, and _pose-dependent gravity sag_ — so the
  FK-predicted end-effector pose is not where the gripper actually is).

This note is the pick-up-later map. The long-form reasoning + durable learnings
live in the auto-memory file `project_do_as_i_do_mvp.md` (search "do as i do").

## The two ideas

1. **Do as I Do** — teach the SO-107 a pick by _showing_ it with a bare hand on
   the top RealSense: reconstruct the hand demo, retarget to the gripper.
   _Verdict:_ for a single demo on a parallel gripper, this is **not easier than
   teleop**. Human-video's real value is data _scale_ (not our goal).

2. **Object-centric pipeline** (the pivot) — teleop one demo, store the EE
   trajectory **relative to the object** (`grasp_in_object = inv(O) @ ee`; replay
   `O' @ grasp_in_object`), then replay at any object pose to multiply teleop
   data N×. _Sound idea_, but execution needs one of two paths and the SO-107
   closes both:
   - **Open-loop FK** (target → IK → send): needs an accurate robot. We don't
     have one → killed by the inaccuracy above.
   - **Visual servoing** (observe gripper + object in the camera, servo): robust
     to the inaccuracy _and_ needs **no** FK↔camera calibration — but requires a
     tag on the gripper, a servo loop, and occlusion handling. Real engineering;
     not worth it for this arm right now.

## Files — what works vs. what's design-only

**Perception / reconstruction — validated, zero motor motion:**

- `capture.py` — standalone RealSense aligned RGB + metric depth + intrinsics.
- `calib.py` — pinhole (un)projection + the base↔camera extrinsic + SE(3) helpers.
  Reuses the shelved `experiment/click-target` math (the 12 mm-RMSE extrinsic at
  `~/.config/lerobot/click_target_extrinsics.json`).
- `reconstruct.py` — object grasp geometry from a SAM3 mask + metric depth, table
  plane via RANSAC, everything lifted camera→base. **The reusable core.**
- `hand.py` — MediaPipe hand → 3D wrist/grasp via metric depth (thumb↔index
  distance = grasp signal). Runs MediaPipe in an isolated venv subprocess.
- `overlay.py` — reprojection overlay; the zero-hardware correctness proof.

**Retargeting / control — math is correct, NEVER closed-loop on real hardware:**

- `hand_to_robot.py` — the entire hand→gripper relationship as ONE rigid
  calibration transform (`ee = T @ hand`). The robot's own IK + gripper handle
  the rest.
- `retarget.py` — minimal top-down grasp planner: gripper points straight down,
  one spin angle sets jaw direction, IK each of pregrasp→approach→close→lift.
  (Clean rewrite; an earlier `choose_grasp_orientation` sweep was discarded.)
- `skill.py` — `PickSkill`: the grasp pose **in the object frame**. The
  object-centric core — what makes one demo generalize.
- `controller.py` — `DoAsIDoController`: closed-loop phase machine that
  re-localizes the object every step and recomputes the target relative to the
  live pose. Design only; the loop was never closed on the real arm.

**Sim:**

- `mujoco_scene.py` — MuJoCo SO-107 pick-scene builder (supports loading a mesh
  as the object; collides via convex hull).

## Key learnings (why this is worth keeping)

- Object-centric record/replay **cancels the gripper geometry** — record AND
  replay the same FK frame and the IK-tip-to-grasp offset / jaw-side / convention
  all drop out. This is the elegant core, and it dissolved a lot of painful
  SO-107 gripper math.
- Rigid transforms **compose by matrix multiply, not addition** — camera→base
  needs the rotation (a tilted camera re-points the axes; 1° tilt @ 0.7 m ≈ 1.2 cm).
- An FK-based pipeline needs only **base↔camera** calibration: tag on the MOVING
  gripper, motion-based hand-eye (`cv2.calibrateHandEye`, eye-to-hand). The tag's
  unknown FK offset cancels because the solve uses relative _motions_. But robot
  inaccuracy degrades it — which is exactly why an inaccurate arm should close the
  loop on the camera instead.

## External, not in this repo

- **SAM 3D Objects on the RTX 5090** — isolated install at `~/.cache/sam3d`
  (`run_sam3d.py`, `anchor_scale.py`). See memory `reference_sam3d_blackwell_install`.
  Produced the metric cylinder mesh (caliper truth 101×19 mm; SAM3D hallucinated a
  hollow tube; naive silhouette diameter over-reads ~38% from depth flying pixels).
- **Throwaway drivers + demo data** were in `/tmp/do_as_i_do` (ephemeral, gone on
  reboot; not committed by design — single-use, host-hardcoded).
- **Visual evidence** shipped as small throwaway LeRobot datasets viewable in the GUI.

## To revive

A more accurate arm, OR willingness to build the visual-servo loop, OR a real push
to scale teleop data where the N× multiplier pays for the engineering. Start from
`reconstruct.py` (object pose in base frame) + `skill.py` (object-relative grasp);
add the camera-frame servo loop in place of `controller.py`'s FK/IK open-loop step.
