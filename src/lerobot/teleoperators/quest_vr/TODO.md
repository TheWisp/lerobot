# Quest VR teleop — TODO

Follow-ups not blocking the initial landing. Captured here so they don't
get lost.

## Unimanual robots: which Quest controller drives the one arm

Quest is hardware-bimanual (one immersive session always streams both
controllers' poses), and we lean into that: a single config type,
`quest_vr`, emits `left_*` / `right_*` action keys and pairs naturally
with a bimanual robot whose `attach_teleop` splits the action dict
per-arm and installs IK on each half.

A single-arm robot (e.g. a unimanual SO-101 / SO-107 follower) wants the
opposite: drop one controller's keys and feed the other to its single
Cartesian-IK controller. The question is **which controller**, and
**where the choice lives**.

Sketches to consider when this actually has a consumer:

- **Config field on the single-arm follower's `attach_teleop`.** The
  follower decides which prefix to consume (`left_target_x` vs
  `right_target_x`), strips the prefix, and feeds the unprefixed dict
  into its `CartesianIKController`. Keeps the teleop unaware of arm
  count; keeps the mapping where the arm lives.
- **Config field on `QuestVRTeleopConfig`** (e.g. `controller_hand:
Literal["left", "right", "both"]`). The teleop filters its emitted
  keys to just the chosen half (unprefixing in the `"left" | "right"`
  case). Single-arm robots get a clean unprefixed dict. Bimanual robots
  use `"both"` (the current default behavior).
- **Wrapper teleop class.** Thin `UniHandQuestVRTeleop` that wraps the
  bimanual instance and forwards a deprefixed action. More code, but
  the bimanual path stays untouched.

No single-arm SO-\* follower currently has a Cartesian `attach_teleop`
branch, so this is genuinely deferred work — the decision is cleanest
when there's an actual single-arm consumer to fit it to. Until then,
the Quest VR + bimanual SO-107 path is the only configuration that
exists.

Related — the action-feature shape that a follower's `attach_teleop`
matches against currently checks for `left_target_x` AND `right_target_x`
([cartesian_ik.py: `is_so107_bimanual_cartesian_teleop`](cartesian_ik.py)).
A unimanual variant will need a sibling detector or a broader contract.
