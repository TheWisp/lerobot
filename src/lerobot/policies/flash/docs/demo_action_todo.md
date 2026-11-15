# FLASH demo action follow-ups

This is the branch's open action-system work after the 2026-11-15 UI-initiated
gamepad-on-ring run. The run dispatched 423 commands and completed 343 marked
keyframes, but the gamepad ended beside the ring. Completion of the command
queue is not evidence of task success. The [run record](../../../../../docs/proofs/flash-demo-editor/real-run-20261115/ui-run/README.md)
contains the final image and compressed plan and motor trace.

- [ ] **Return the robot to its saved rest position.** The action worker now
      connects, reads current joints, sends the demo path, and disconnects. It does
      not reset the arm before or after a run. Use the existing robot rest-position
      capability to make reset available in the demo workflow, including after a
      stopped or completed run. Make its state visible; do not call a disconnect a
      reset.
- [ ] **Get one verified physical placement.** The last UI run ended beside the
      ring. Inspect the gamepad, ring, gripper, and commanded tool path together and
      verify the object actually rests on the ring before calling the action path
      successful.
- [ ] **Stabilize the placement reference.** Two accepted ring poses 1.63 s
      apart moved the computed final tool target by 61.9 mm, although the fitted
      ring surface centre moved about 10 mm. Determine whether the ring's coordinate
      frame changed under symmetry or the object moved; test the selected live
      target against the recorded scene, including through occlusion.
- [ ] **Use the actual held-object relation.** The gamepad was carried, but its
      grasp relative to the gripper differed from the demonstration. Track or
      measure that relation, use robot pose as an independent source while the
      gamepad is hidden, and handle release or slip. Keep this separate from scene
      group tracking.
- [ ] **Inspect placement contact.** An earlier interrupted placement moved the
      ring. Align camera frames with robot readback to identify first contact and
      distinguish a bad pose target from a collision or grasp error.
- [ ] **Save and replay every applied run.** The editor's “Last replay” currently
      opens the teleoperated demonstration. The applied run was visible only while
      Live was open, and rolling camera history expired before Stop. Preserve raw
      and processed camera frames, tracker results, motor commands, readback, and
      timestamps in one replayable run.
- [ ] **Measure end-to-end speed against the demo.** Retained motor ticks were
      near 30 Hz and placement travel was roughly 1× over 0.33 s samples, but the
      run had a 2.67 s command gap. The pickup and placement keyframe ranges took
      longer than their marked-frame spans. Report both local end-effector speed and
      whole-action elapsed time, including approach, pauses, and readback lag.
- [ ] **Make progress and failure visible.** Show the current target, arm pose,
      active reference, and reason for waiting or stopping in the shared live view.
      Separate “all commands issued” from measured arrival and object placement.
      The no-motion test view should draw the simulated arm and tool target; its
      completion alone cannot establish grasp or contact success.
- [ ] **Review demo-to-live object identity after a successful run.** Decide how
      demo objects are initialized and bound to live objects, including unnamed
      objects selected by clicking, and show that binding for each keyframe.
- [ ] **Measure causal pose age.** Record capture, depth/mask completion, pose
      publication, planning, and command dispatch times so the action system uses
      and displays the actual age of its reference. Investigate actuator lag or
      blocked motion without restarting the command path from each joint readback.
