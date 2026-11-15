# UI robot run, 2026-11-15

The saved “Gamepad onto green ring” demo was started from FLASH → Demo editor → Live, after moving `white_left` to its recorded rest position in the Robot tab. The UI reached 343/343 keyframes and 423 motor commands. [The final camera frame](final.jpg) shows the gamepad beside the ring, so this was not a successful placement. The final gripper command was 95.0°, with readback 94.9°.

The command stream used the intended 30 Hz cadence: retained commanded ticks had a 33.34 ms median separation. The pickup's 131 keyframes took 4.333 s between first and last command, matching 130/30 s. The placement's 211 keyframes took 6.977 s, matching 210/30 s. The largest dispatch gap anywhere in the run was 2.667 s, so the whole run was not uninterrupted playback. The motor trace retains only its last 3,600 of 3,803 ticks; the dropped beginning prevents a complete physical-speed calculation for pickup.

For placement, the recorded tool path was 282 mm over 7 s. The requested live path was 299 mm and motor-readback path 338 mm over 6.98 s at 30 Hz. One-tick path length exaggerates readback motion because of joint quantization and small oscillations. Comparing 10-frame (0.33 s) displacements gives 255 mm recorded, 261 mm requested, and 257 mm measured: roughly 1× physical travel over this range, with local speed differences. The tool settled to 2.95 mm and 0.9° from the final command within 0.5 s; motor following was not the gross placement error.

The two accepted ring reference poses saved 1.63 s apart in `preflight.json` and `execution.json` differ by 22.7° in orientation. Applying each to the _same_ final ring surface moves its center about 10 mm, but applying each to the demonstration's final tool offset moves the placement target 61.9 mm. This is a concrete unstable reference-to-target conversion. The retained camera history starts after the robot motion, so it cannot prove whether this target shift or a changed grasp offset caused the final miss.

`Last replay` in Demo editor opens the teleoperated demonstration, not this application run. The editor showed this run only in Live while executing. The 60-second live-camera history rolled past the motion before Stop; the complete applied-run video is unavailable. The plan, motor readback trace, execution status, and final image are saved here for review.

The full plan and motor trace are stored as `plan.json.gz` and `motor_trace.json.gz` to keep the evidence compact; both contain ordinary JSON after decompression.
