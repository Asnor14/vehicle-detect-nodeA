# Vehicle Detect Node A

Raspberry Pi blind-spot vehicle detection using two USB webcams, YOLO11, OpenCV, and a Flask web dashboard.

## What This Project Does

- Uses two cameras:
  - `Approach` camera for early warning
  - `Blindspot` camera for occupancy confirmation
- Runs YOLO vehicle detection on the combined camera view
- Keeps the relay ON using two-stage logic:
  - approach detections reset the approach hold timer
  - blindspot detections keep the relay ON until the blindspot clears
- Serves a web dashboard with event history and millisecond timestamps
- Supports Raspberry Pi desktop autostart so the preview opens after login

## Raspberry Pi Install

### 1. Clone the repository

```bash
git clone https://github.com/Asnor14/vehicle-detect-nodeA.git
cd vehicle-detect-nodeA
```

### 2. Run the Pi installer

```bash
chmod +x install_on_rpi.sh
./install_on_rpi.sh
```

This creates `.venv`, installs the Pi-friendly Python packages, and keeps OpenCV on the system package.

### 3. Start manually

```bash
source .venv/bin/activate
python main.py
```

## Camera Mapping

The app is configured to use stable Raspberry Pi camera device paths instead of raw `/dev/video0` numbers:

- Approach camera:
  `/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0`
- Blindspot camera:
  `/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0`

Keep the cameras on the same USB ports for this mapping to stay correct.

## Web Dashboard

When the app is running, open:

```text
http://<raspberry-pi-ip>:5000
```

The dashboard shows:

- live approach/blindspot detection state
- relay source and hold timers
- current event and completed event history
- millisecond timestamps in cards, table, copy output, and print/PDF view

## Autostart

Desktop autostart is included for Raspberry Pi GUI login.

- Launcher:
  `scripts/start_vehicle_detect_gui.sh`
- Desktop entry template in repo:
  `autostart/vehicle-detect.desktop`

If you want the preview window to appear automatically after reboot, make sure Raspberry Pi desktop autologin is enabled.

To install the desktop autostart entry on another Pi:

```bash
mkdir -p ~/.config/autostart
cp autostart/vehicle-detect.desktop ~/.config/autostart/
```

There is also a background `systemd` service template in `systemd/vehicle-detect.service` plus helper scripts in `scripts/`, but the GUI autostart is the preferred mode when you want the OpenCV preview window.

## Main Files

- `main.py` - two-camera detection, relay logic, preview window
- `webapp.py` - Flask API and event logging
- `hotspot.py` - optional hotspot control, disabled by default
- `install_on_rpi.sh` - Raspberry Pi dependency setup
- `scripts/` - autostart helpers
- `systemd/` - optional background service unit

## Notes

- YOLO model weights (`*.pt`) are ignored in git and download automatically on first run.
- `.venv` is intentionally not committed.
- If you move either camera to a different USB port, update the camera paths in `main.py`.
