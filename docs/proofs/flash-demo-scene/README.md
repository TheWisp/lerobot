# FLASH demo scene: insertion recording

The Run → FLASH → Demos viewer now has an optional **3D scene** view. It unprojects each recorded aligned RGB-D frame into a subdued, rotatable scene and places the saved partial surface patches at the poses in the selected prepared version. Object axes, saved extraction estimates, and gripper-backed predictions can be shown separately. The viewer does not change the prepared skill or rerun tracking.

This check used `insertion` episode 0 and prepared version `insertion-causal-parity-20261023` on the local SO-107 recording. That version saved a per-frame extraction trace; earlier versions without that trace show the missing source explicitly. The group-support point set and per-frame masks were **not** saved, so the group layer is a pose applied to the saved reference patch, not the original group points.

- [Frame 228: group estimate beside the prepared reference pose](group-frame-228.png). The scene is focused on the dowel; the measured reference patch is sharp against the recorded depth scene.
- [Frame 500: carried dowel with gripper-backed prediction](held-frame-500.png). The orange ghost is the gripper-backed estimate; it is not an independent observation.
- [Frame 540: after release](release-frame-540.png). The saved extractor reports the dowel as lost, so no dowel estimate is painted onto this frame.

The display keeps missing depth open rather than filling a back surface. In headless software-rendered Chromium, the 3D preview displayed about 40 of 53 requested frames over a 3-second playback sample; this is preview rendering throughput, not the robot-control loop rate. Browser navigation, camera/orbit, focus, source toggles and all three frames produced no page errors.
