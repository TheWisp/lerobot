# FLASH demo scene: insertion recording

The Run → FLASH → Demos viewer shows a soft recorded RGB frame behind the current object masks. Masked RGB pixels remain sharp where depth is missing. When the camera orbits, it shows only measured 3D surface; unseen geometry remains absent. The coarse full-scene depth mesh is optional. The saved skill pose patch, extraction pose, actual scene-group support points, and gripper-backed prediction remain separate layers.

The `insertion-causal-parity-20261023` and default `insertion-migrated-v10-preferred-hover` prepared versions originally saved poses but not masks or group point identities. Those can be regenerated from their ordinary RGB-D demo using the same SAM3, appearance, and causal scene-tracking pipeline as preparation:

```bash
python -m lerobot.policies.flash.sealed_demo_evidence DATASET_ROOT BUNDLE_JSON
```

Both versions now have all 713 camera frames rebuilt into local `scene-evidence` sidecars, without modifying either prepared skill. For the causal version, the first 591 frames covered by the original pose cache reproduced its evidence labels, estimator labels, and available pose matrices **exactly** for both objects. At frame 228 the rebuilt verified support groups contained 157 dowel-context points and 294 ring-context points. The dowel group is withdrawn during the attached phase, as the causal pipeline requires.

- [Frame 100: current observed surfaces](observed-frame-100.png), colored by the recorded camera instead of a single reference patch.
- [Frame 228: observed surfaces and actual group support points](group-frame-228.png).
- [Frame 500: carried object](held-frame-500.png), with the gripper-backed pose available separately.
- [Frame 540: after release](release-frame-540.png). The dowel's SAM mask and measured visible surface remain available even though the saved pose extractor reports **lost**. This exposes a pose-estimation failure, not loss of every object pixel.
- [Frame 540: 3D close-up](release-focus-frame-540.png). The flat RGB backdrop is hidden when the camera orbits, avoiding false duplicate geometry.

The rebuilt masks and groups are an inspection sidecar. They do not revise the stored policy pose or make an unmeasured back surface into geometry. Browser inspection of camera, focus, and group toggles produced no page errors.
