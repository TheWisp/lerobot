#!/usr/bin/env bash
# Re-download the VR clips (the only reel assets not committed — they're large
# and live on YouTube). The small assets (gifs / pngs / training_dashboard.mp4)
# are committed under assets/. Requires yt-dlp + a JS runtime (node works).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-/home/feit/miniforge3/envs/lerobot/bin/python}"
"$PY" -m pip install -q yt-dlp 2>/dev/null || true

dl() { "$PY" -m yt_dlp --js-runtimes node -f "bv*[height<=1080][ext=mp4]" -o "assets/$1" "$2"; }
dl vr_real.mp4    "https://youtu.be/KSwNev5JRIc"   # PR #18 — Quest 3 -> real bimanual SO-107
dl vr_virtual.mp4 "https://youtu.be/C5pX30HpgeI"   # PR #18 — Quest 3 -> virtual robot
echo "done -> assets/vr_real.mp4 assets/vr_virtual.mp4"
