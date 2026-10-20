# Held-object attachment policy

The live trial reads all grasp-assumption and held-pose correction thresholds from one `attachment` object in its rig JSON. `AttachmentRuntimeLimits.from_config` validates that object once; the planner and pose updater receive the same resulting policy. These are trial parameters, not properties of the dowel or ring.

| Parameter                  | Meaning                                                                                                                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `motor_history_capacity`   | Number of measured joint samples retained to align a camera observation with the gripper at image capture time.                                                                                                     |
| `max_motor_sample_gap_s`   | Largest gap between joint samples that may bracket a camera image.                                                                                                                                                  |
| `max_unverified_age_s`     | Optional maximum time since the grasp assumption or last independent held observation. `null` keeps a rigid grasp assumption active until release or contradictory evidence; the replay must label it as predicted. |
| `max_observation_shift_m`  | Maximum translation change allowed when an independently observed held pose updates the gripper-to-object relation.                                                                                                 |
| `max_observation_turn_deg` | Matching rotation gate for that update.                                                                                                                                                                             |
| `max_pregrasp_pose_age_s`  | Maximum age of the object pose used to seed the grasp assumption.                                                                                                                                                   |

The current SO-107 trial sets `max_unverified_age_s` to `null`. The old 20 s setting stopped the 2026-10-20 19:12:46 local trial during its ring approach, before the insertion contact interval. Elapsed time alone supplied no evidence that the object had slipped. The 15 mm and 15° correction gates remain provisional configuration values: several postgrasp RGB-D proposals in that trial differed from the grasp seed by 22–34 mm and could not safely be treated as a confirmed correction. A later uncertainty-aware tracker should replace fixed gates with calibrated observation confidence. Until then, do not silently interpret the gripper prediction as a measured held pose.

The [example trial JSON](trial_config.example.json) also exposes the planner's IK sample and retiming budgets, evidence wait, and every causal group-tracking limit. The live launcher requires these values in its rig JSON so a trial cannot silently inherit a different internal default.

The GUI recording now stores `held_prediction` separately from the independent tracker `accepted` pose. Older recordings lack that field; their stationary tray outline is an old tracker state, not the pose the held-to-target planner used.
