#!/bin/sh
set -eu
root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec python3 "$root/tools/control.py" start --project yolo11_bytetrack "$@"
