#!/bin/bash
# Render the 6 IK shape trajectories (3 shapes x 2 arms) into GIFs:
# one headless Chrome screenshot per frame, then ffmpeg the stills.
set -u
cd /tmp/ikproof
mkdir -p gifframes
rm -f gifframes/*.png gifframes/*.gif
N=20
TRAJS="so101_circle-30mm so101_circle-60mm so101_square-50mm so107_circle-30mm so107_circle-60mm so107_square-50mm"
for traj in $TRAJS; do
  for f in $(seq 0 $((N - 1))); do
    K=$(( f * 255 / (N - 1) ))
    idx=$(printf "%03d" "$f")
    timeout 90 google-chrome --headless=new --hide-scrollbars --window-size=1100,840 \
      --virtual-time-budget=12000 --enable-unsafe-swiftshader \
      --screenshot="gifframes/${traj}_${idx}.png" \
      "http://localhost:8742/harness.html?traj=${traj}&frame=${K}" >/dev/null 2>&1
  done
  ffmpeg -y -framerate 14 -i "gifframes/${traj}_%03d.png" \
    -vf "scale=720:-1:flags=lanczos,palettegen=stats_mode=diff" \
    "gifframes/${traj}_pal.png" >/dev/null 2>&1
  ffmpeg -y -framerate 14 -i "gifframes/${traj}_%03d.png" -i "gifframes/${traj}_pal.png" \
    -lavfi "scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer" \
    "gifframes/${traj}.gif" >/dev/null 2>&1
  echo "${traj}: done"
done
echo "ALL DONE"
ls -la gifframes/*.gif
