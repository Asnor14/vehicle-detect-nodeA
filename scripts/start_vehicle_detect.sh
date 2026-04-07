#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/raspi2/vehicle_detect"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_PREFIX="[vehicle-detect]"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-45}"

CAMERAS=(
  "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0"
  "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0"
)

echo "$LOG_PREFIX Starting launcher"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "$LOG_PREFIX Missing virtualenv python at $VENV_PYTHON" >&2
  exit 1
fi

for camera in "${CAMERAS[@]}"; do
  waited=0
  while [[ ! -e "$camera" ]]; do
    if (( waited >= MAX_WAIT_SECONDS )); then
      echo "$LOG_PREFIX Camera did not appear in time: $camera" >&2
      exit 1
    fi
    echo "$LOG_PREFIX Waiting for camera: $camera"
    sleep 1
    ((waited += 1))
  done
done

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
exec "$VENV_PYTHON" main.py
