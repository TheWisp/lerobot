# SAM3 Amodal 3D Tracking — quick test (v1: the green ring)

Overlays the FoundationPose-tracked 3D mesh of an object onto the live camera view,
including the parts hidden by occlusion. v1 is hardcoded to the **green ring** on the
**top RealSense**. Run on the rig box (needs the cached ring mesh + the SAM3D/FoundationPose envs).

1. Open the GUI at `http://<host>:8010` and **hard-refresh** (Ctrl/Cmd-Shift-R).
2. Start **teleop** with the bimanual robot (this streams the camera **+ depth**).
3. Debug model → select **"SAM3 — locked object tracking"**. In the **Cameras** filter,
   tick **only `top`** (uncheck front/left_wrist/right_wrist), then **load**.
   (Amodal only uses `top` — the depth camera — so there's no need to run the others.)
4. Add one object named **`green ring`**, and put the ring in the **top** camera's view.
   It's detected once, then locked + tracked geometrically (GPU stays flat).
5. Tick **"Amodal 3D tracking"** (the checkbox under the SAM3 tracker).
6. Wait ~15–20 s (the sidecar loads + registers once). The gray ring mesh + cyan
   outline appears on the top view and tracks the ring; **magenta = occluded geometry**
   the model fills in. Move the ring slowly to see it track.

**Note:** the tracker now rebuilds its session periodically, so GPU memory stays flat
indefinitely (no more OOM). Watch the VRAM label next to the FPS to confirm it holds steady.

**No overlay after ~20 s?** Most likely depth isn't reaching the stream (the one piece
not yet validated live) — ping Fei. Only the **top** camera (the one with depth) shows
the overlay. The ~15–20 s wait is the sidecar **loading its models** (one-time); after
that, registration is sub-second and tracking runs **~90 FPS** on the 5090, so the
overlay loop stays smooth (no freeze).
