#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/raspi2/vehicle_detect"
SERVICE_SRC="$PROJECT_DIR/systemd/vehicle-detect.service"
SERVICE_DEST="/etc/systemd/system/vehicle-detect.service"

if [[ ! -f "$SERVICE_SRC" ]]; then
  echo "Service file not found: $SERVICE_SRC" >&2
  exit 1
fi

sudo install -m 644 "$SERVICE_SRC" "$SERVICE_DEST"
sudo systemctl daemon-reload
sudo systemctl enable vehicle-detect.service
sudo systemctl restart vehicle-detect.service

echo "vehicle-detect.service installed and started."
echo "Check status with: sudo systemctl status vehicle-detect.service"
echo "View logs with: journalctl -u vehicle-detect.service -f"
