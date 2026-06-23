# SAM3 Amodal 3D Tracking — quick test (v1: the green ring)

Overlays the FoundationPose-tracked 3D mesh of an object onto the live camera view,
including the parts hidden by occlusion. v1 is hardcoded to the **green ring** on the
**top RealSense**. Run on the rig box (needs the cached ring mesh + the SAM3D/FoundationPose envs).

1. Open the GUI at `http://<host>:8010` and **hard-refresh** (Ctrl/Cmd-Shift-R).
2. Start **teleop** with the bimanual robot (this streams the camera **+ depth**).
3. Debug model → select **"SAM3 video — tracked masks"**. In the **Cameras** filter,
   tick **only `top`** (uncheck front/left*wrist/right_wrist), then **load**.
   ⚠️ SAM3-video keeps a growing memory bank \_per camera* — running all 4 at once can
   exhaust the GPU (OOM). Amodal only uses `top` (the depth camera) anyway.
4. Add one concept named **`green ring`**, and put the ring in the **top** camera's view.
5. Tick **"Amodal 3D tracking"** (the checkbox under SAM3 video).
6. Wait ~15–20 s (the sidecar loads + registers once). The gray ring mesh + cyan
   outline appears on the top view and tracks the ring; **magenta = occluded geometry**
   the model fills in. Move the ring slowly to see it track.

**GPU out-of-memory?** You ran SAM3-video on all 4 cameras — set the Cameras filter to
`top` only (step 3). Each camera holds its own growing memory bank.

**No overlay after ~20 s?** Most likely depth isn't reaching the stream (the one piece
not yet validated live) — ping Fei. Only the **top** camera (the one with depth) shows
the overlay. The ~15–20 s wait is the sidecar **loading its models** (one-time); after
that, registration is sub-second and tracking runs **~90 FPS** on the 5090, so the
overlay loop stays smooth (no freeze).
