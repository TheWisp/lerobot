#!/bin/bash
# Restart the GUI server, locally or on a remote host, and wait until it serves.
#
# Two reasons this exists rather than being retyped each time:
#
#   * A running server keeps the code it started with. After syncing a change,
#     the server must be restarted or you are testing what you already had --
#     which has repeatedly produced "verified" fixes that were never running.
#   * RSS grows over a long session (measured: 2.2 GB at start, 9.2 GB after
#     ~24 h). A restart reclaims it.
#
# Usage:
#   scripts/gui/restart_gui.sh                 # restart on this machine
#   scripts/gui/restart_gui.sh <ssh-alias>     # restart on a remote host
#
# The remote form runs this same script over ssh, so the two paths cannot
# drift. Everything is discovered from the running process -- no hostnames,
# ports or checkout paths are baked in, so it works for any host and any
# checkout.
set -u

PORT="${LEROBOT_GUI_PORT:-8000}"

if [ $# -ge 1 ]; then
    host="$1"
    shift
    # Ship the script itself; the remote may not have this checkout.
    ssh "$host" "PORT_OVERRIDE=${PORT} bash -s" -- < "$0"
    exit $?
fi

[ -n "${PORT_OVERRIDE:-}" ] && PORT="$PORT_OVERRIDE"

old=$(pgrep -f 'lerobot-gui --host' | head -1)
if [ -z "${old:-}" ]; then
    echo "no GUI process found; nothing to restart" >&2
    exit 1
fi

# Relaunch it exactly as it was running: same working directory, same
# executable, same flags. Reconstructed from the live process so a server
# started with different options comes back with them.
dir=$(readlink -f "/proc/$old/cwd" 2>/dev/null) || dir=""
cmd=$(tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null)
log=$(readlink -f "/proc/$old/fd/1" 2>/dev/null)
case "$log" in
    /dev/*|"" ) log=/tmp/lerobot-gui.log ;;
esac
[ -n "$dir" ] || { echo "cannot read the working directory of pid $old" >&2; exit 1; }

echo "stopping GUI pid $old (rss $(ps -o rss= -p "$old" | tr -d ' ') KB)"
kill -TERM "$old"
for _ in $(seq 20); do
    kill -0 "$old" 2>/dev/null || break
    sleep 1
done
if kill -0 "$old" 2>/dev/null; then
    echo "did not exit on SIGTERM; sending SIGKILL"
    kill -KILL "$old"
    sleep 2
fi

cd "$dir" || exit 1
# shellcheck disable=SC2086 — the reconstructed command line is intentionally split
nohup $cmd >> "$log" 2>&1 &
new=$!
echo "started pid $new in $dir (log: $log)"

for _ in $(seq 40); do
    sleep 2
    if curl -s --max-time 5 -o /dev/null "http://127.0.0.1:${PORT}/api/process/jobs"; then
        echo "serving after $(ps -o etime= -p "$new" | tr -d ' ') | rss $(ps -o rss= -p "$new" | tr -d ' ') KB"
        exit 0
    fi
done
echo "did not come up within 80 s — check $log" >&2
exit 1
