# FLASH object pose, in plain terms

FLASH currently uses a 3D rigid pose for each object. It means “where is the
object's **saved reference frame** in the robot base frame?” The reference is
created from one segmented RGB-D view of that object. Its origin is fixed when
the reference is saved; it is **not** a recalculated center of mass or the
center of whatever surface happens to be visible now. Every estimate and skill
sample carries that reference's version. A changed version cannot silently be
combined with an old gripper attachment.

```text
aligned RGB + depth → object mask → visible 3D surface
                                  ↓
                 match/fit to the tracked reference surface
                                  ↓
                  pose of the saved object frame
                       ↙                    ↘
            scene/point motion         gripper prediction
            across sparse views        during uncertain carry
                       ↘                    ↙
                    source-time pose decision
```

The direct tracker moves a fixed partial reference surface between images and
corrects it against the current mask and depth. The scene group tracks nearby
points to bridge views where direct matching is unavailable. During carry,
`T_gripper_object` predicts the object's pose from measured gripper motion. A
pre-grasp pose only **seeds** this relation; it does not prove that the object
stayed there while the fingers closed. The demo and live paths use the same
direct and scene tracking components.

When an independent held-object observation arrives, FLASH uses the gripper
pose measured at that image's capture time. A small change updates
`T_gripper_object` immediately. For a larger change, FLASH compares the visual
pose and gripper prediction against the **same image**, using the same fixed
object reference. The visual pose must explain at least as much masked surface
and no worse depth, with a strict improvement in one of them. It must then
produce a compatible gripper-to-object relation in a later image. Until that
second image, the attachment is conflicted for control. If the image cannot
distinguish the proposals but the gripper prediction still fits measured depth
within the configured tolerance, FLASH keeps that prediction and labels the
visual ambiguity. It does not call the ambiguous image a correction. If the
gripper prediction is unsupported or contradicted, the attachment remains
conflicted. The size of a small update, depth tolerance, and minimum measured
point count come from the trial's configuration.

The recorded-demo importer now runs this same correction rule at source time
when the demo annotation includes `held_attachment` and the aligned masks are
available. A corrected carried sample uses the latest accepted
`T_gripper_object`; frames between observations remain gripper predictions and
are labelled `vision_corrected_prediction` in the skill. Unresolved intervals
are labelled `conflicted_prediction` even though the last accepted attachment
is retained for replay. Ambiguous visual proposals with a depth-consistent
gripper prediction are labelled `ambiguous_visual_prediction`. A candidate bundle
still needs review before use. The older prepared versions are preserved.

**Limit:** a partial surface can fit the current image while leaving the
hidden end or a symmetric rotation undetermined. Mask/depth fit, two consistent
images, and a fixed origin do not prove the full pose or the contact point.
Those cases need an additional view, task-relevant contact evidence, or an
explicit human decision. The replay must keep visual and gripper proposals
available for debugging even though the ordinary view shows one selected pose.
