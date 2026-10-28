# FLASH: demo extraction and interaction editing

Status: proposed

**Teach through a timeline of keyframes and editable surface relationships.**
Review the extracted demonstration and test its adaptations in the same 3D scene.

## Principles

- **Timeline drives behavior.** Keyframes select participants and required relationships. Each interval preserves demonstrated motion/timing or generates a connection. No built-in grasp, push or placement stages.
- **People permit; geometry checks.** Similarity alone does not authorize adaptation.
- **Choices have consequences.** An adaptation persists through its relationship's lifetime. Later keyframes constrain it; they cannot silently reset physical contact.
- **Evidence stays visible.** Reliable vision corrects tracking. Supported gripper or group relationships predict through lost observations; pushing never implies rigid attachment.
- **One pipeline.** Demo extraction, Test and debugging share tracking and normal LeRobot datasets.

## Components

```mermaid
flowchart TD
    RGBD[Recorded or test RGB-D] --> Tracking[Shared tracking: surfaces and evidence]
    Tracking --> Extraction[Demo extraction]
    Editing[Timeline and relationship annotations] --> Extraction
    Extraction --> Demo[Prepared demo]
    Tracking --> Matching[Test: match the current keyframes]
    Demo --> Matching
    Matching --> Planning[Async planning and action queue]
    Planning --> Motor[30 Hz LeRobot control]
    Extraction -.-> View[Shared 3D view]
    Matching -.-> View
    Planning -.-> View
```

<a id="demo-extraction-from-the-timeline"></a>

## What extraction saves

| Data      | Contents                                                                                                               |
| --------- | ---------------------------------------------------------------------------------------------------------------------- |
| Objects   | Observed surfaces, masks, supported alignments, source frames and uncertainty.                                         |
| Keyframes | Participants, relative surface arrangement, required relationship and allowed adaptation.                              |
| Intervals | Demonstrated relative motion/actions/timing, or connecting goals with speed/constraints; rule lifetime and milestones. |

**Regenerate → inspect → apply a version.** Cache geometry; retain source data.
Predictions and unlinked views remain explicit. A complete mesh is unnecessary.

<a id="one-scene-view-for-demoedit-test-and-replay"></a>

## One view: Demo/Edit · Test · Replay

![Proposed editor using the actual insertion RGB-D scene](assets/surface_relations/scene_editor.png)

_Actual insertion RGB-D, frame 100. Proposed controls and hypothetical handles; not a policy result._

Reconstruct the objects, robot and surroundings. Orbit/pan/zoom follows the
URDF viewer; **Camera** restores the capture viewpoint. Keep playback below
the scene. Collapse the recording/object lists and contextual inspector for space.

- **Demo/Edit:** edit keyframes and rules directly on the scene and timeline.
- **Test:** show matched surfaces, active rules, progress and available plans.
- **Replay:** exactly Test with recorded snapshots. Future “Use as demo” preserves the recording.

## Editing a relationship

1. Scrub to a keyframe; select two participants and their contact regions.
2. Select a tool below. Drag its handles in 3D; the inspector mirrors the values.
3. Drag the rule's timeline span; scrub forward to inspect consequences.
4. Preview feasibility/unknowns, then save. Previewing never moves the robot.

![Turn and slide handles on measured ring and dowel geometry](assets/surface_relations/adaptation_tools.png)

| Tool                   | Gesture and meaning                                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Turn at a point**    | Pick a pivot; review the suggested surface-normal direction. Drag arc endpoints or allow a full turn.                                   |
| **Slide on a surface** | Paint a permitted ribbon. Drag a ghost contact along it; orientation follows compatible measured geometry. No cylinder axis is assumed. |

Both edit relationships between any participants; sliding does not imply grasping.

## Make the proposed change stand out

- **Scene:** soften/desaturate observed objects; preserve outlines, contact patches and obstacles. Adjust **Scene emphasis**; hold **Original** to compare. Missing geometry fades to a neutral background.
- **Allowed choices:** a sharp translucent amber arc or ribbon with draggable endpoints. This is permission, not a path.
- **Candidate:** sharp cyan object surface or tool point + approach arrow; a crossbar shows closing direction. Alternatives are faint ghosts.
- **Motion:** show a labelled path only when computed; mark unchecked/occluded portions. Permission, prediction and measured motion stay distinct. Display effects never change checking geometry.

**Layers** stays clear of **Expand** and playback. Hover to highlight, click to
pin. Visibility affects neither control nor recording. Geometry and overlays
share source timestamps; replay reproduces the same snapshot.
