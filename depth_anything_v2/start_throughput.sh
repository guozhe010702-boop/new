#!/bin/sh
set -eu
root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec python3 "$root/tools/control.py" start --project depth_anything_v2 --depth-profile camera-rect --workers 3 --mask 7 --capture-convert selected --inference-cpus big --save-every 0 "$@"
