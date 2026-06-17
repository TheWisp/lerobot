# Customer demo artifacts

Curated walkthrough of the LeRobot GUI feature work, for showing to (Nebius) customers.

## Outputs

- **`walkthrough.html`** — interactive gallery (open in any browser). Sections: data editing,
  URDF + trajectory, latency, VR teleop, transfers, training, **Cloud GPUs on Nebius** (closer).
  VR videos play inline; images load from GitHub raw / YouTube (needs internet).
- **`highlight_reel.mp4`** — ~44 s, 1080p, silent, motion-only highlight reel ending on Nebius.
  Also published as a GitHub prerelease for easy download:
  <https://github.com/TheWisp/lerobot/releases/download/demo-reel-20260617/lerobot_highlight_reel.mp4>

## Rebuild the reel

```bash
bash fetch_assets.sh          # re-downloads the 2 VR clips into assets/ (yt-dlp + node)
/home/feit/miniforge3/envs/lerobot/bin/python build_reel.py   # -> highlight_reel.mp4
```

Retune by editing the segment list at the bottom of `build_reel.py` (durations, VR `ss=` windows,
order, captions). Resolution/quality: `W,H` + `-crf`. It's silent — add music/voiceover after.

## Reel sources

| Segment                 | Asset                           | Origin                                                |
| ----------------------- | ------------------------------- | ----------------------------------------------------- |
| VR — real hardware      | YouTube `KSwNev5JRIc`           | PR #18 (quest_vr)                                     |
| VR — virtual robot      | YouTube `C5pX30HpgeI`           | PR #18                                                |
| Live training dashboard | `assets/training_dashboard.mp4` | recorded live off PR #32 (ACT / pusht), not in any PR |
| HVLA RLT telemetry      | `assets/rlt.png`                | PR #35 (`charts-rlt-panel.png`)                       |
| Cartesian IK            | `assets/ik_circle.gif`          | PR #9 (`ik_so107_circle-30mm.gif`)                    |
| Record → replay         | `assets/record_replay.gif`      | PR #17 (`virtual_bi_so107_demo.gif`)                  |
| Nebius closer           | `assets/nebius.png`             | PR #33 (`35-ephemeral-dialog-connection-status.png`)  |

Committed assets are the small, non-trivially-reproducible ones. The VR clips are re-downloaded
(large, on YouTube). `training_dashboard.mp4` is a one-off live capture — keep it.

## Gallery sources

Every image/gif/video used (and the rest available across all closed PRs) is inventoried in
**`media_sources.md`** — PR number, caption, and URL. Two assets from that scan are dead (deleted
feature branches: an old `urdf_viz.png` and `pink_ik_tradeoff.png`); everything used here is live.
Most URLs are SHA-pinned (stable); a few point at still-present feature branches.

## Ideas for next iteration

- Add a voiceover or background track (reel is currently silent).
- Swap the VR window if a punchier moment is wanted (`ss=` in `build_reel.py`).
- Longer training segment / add the RLT _hover_ shot for the charts story.
- A real Nebius spawn→train→destroy screen-capture would beat the static dialog as the closer.
