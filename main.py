"""
Vehicle Detection & Counter using YOLO11
=========================================
Optimized for Raspberry Pi 4B (Bookworm OS) with threaded camera capture
for smooth preview. All detected vehicle types (car, bus, truck, motorcycle,
bicycle, train) are labeled as "Vehicle".

Based on architecture from github.com/Asnor14/cpe4bVehicleDetection
"""

import os
import sys
import threading
import time
from contextlib import nullcontext

# Fix for OpenCV Qt/Wayland issues on RPi Bookworm
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"
if not os.environ.get("QT_QPA_FONTDIR") and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

# GPIO — only available on Raspberry Pi
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    GPIO = None
    HAS_GPIO = False

# Web App Integration
from webapp import (
    app as flask_app, 
    log_detection_event, 
    log_relay_signal,
    log_relay_timeout,
    set_app_state
)
from hotspot import enable_hotspot, disable_hotspot, hotspot_feature_enabled

# ============================================================================
# Configuration — tweak these for your setup
# ============================================================================

# Model: Use yolo11n (nano) on RPi for speed. Use yolo11s or yolo11l on desktop.
MODEL_PATH = 'yolo11n.pt'

# Webcam settings
CAMERA_CONFIGS = [
    {
        'name': 'Approach',
        'device_path': '/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0',
    },
    {
        'name': 'Blindspot',
        'device_path': '/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0',
    },
]
APPROACH_CAMERA_NAME = 'Approach'
BLINDSPOT_CAMERA_NAME = 'Blindspot'
FRAME_WIDTH = 640              # Camera capture width
FRAME_HEIGHT = 480             # Camera capture height
TARGET_FPS = 30                # Requested camera FPS

# YOLO inference settings
IMG_SIZE = 512                 # YOLO input image size (512 catches motorcycles better)
DETECT_CONF = 0.15             # Low confidence to catch motorcycles easier
FRAME_SKIP = 1                 # Process every Nth frame (1 = every frame)
MAX_DETECTIONS = 20            # Cap detections to reduce NMS overhead on Pi CPU
SHOW_INFERENCE_STATS = True    # Overlay detector latency/FPS to help tuning on Pi

# COCO class IDs for vehicles:
#   1=bicycle, 2=car, 3=motorcycle, 5=bus, 6=train, 7=truck
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

PREDICT_KWARGS = {
    'classes': VEHICLE_CLASSES,
    'imgsz': IMG_SIZE,
    'conf': DETECT_CONF,
    'device': 'cpu',
    'verbose': False,
    'max_det': MAX_DETECTIONS,
}

# Generic label — all vehicles shown as "Vehicle" instead of individual types
GENERIC_LABEL = 'Vehicle'

# Counting line (relative to frame height, 0.0=top, 1.0=bottom)
LINE_POSITION_RATIO = 0.65    # 65% down the frame
LINE_MARGIN_X = 50             # Pixels from left/right edge

# Output video
SAVE_OUTPUT = False            # Set True to record output (uses more CPU)
OUTPUT_PATH = 'output_webcam.mp4'

# GPIO Relay settings (Raspberry Pi only)
RELAY_ON_PIN = 17              # GPIO pin to trigger relay ON
RELAY_OFF_PIN = 27             # GPIO pin to trigger relay OFF
APPROACH_HOLD_SECONDS = 4.0    # Resettable hold time from the approach camera
BLINDSPOT_CLEAR_HOLD_SECONDS = 1.0  # Short delay after blindspot clears
RELAY_TRIGGER_PULSE_SECONDS = 0.2  # Duration of the trigger pulse

# Window
WINDOW_NAME = "Vehicle Detection & Counter (Press Q to quit)"

# ============================================================================
# Shared state between threads
# ============================================================================
state_lock = threading.Lock()
last_boxes = []                # List of (x1,y1,x2,y2,confidence)
latest_frame = None
latest_frame_idx = 0
latest_camera_layout = []
pending_inference = False
running = True
inference_error = None
inference_ms = 0.0
inference_fps = 0.0

# Relay state (shared between inference thread and main loop)
relay_lock = threading.Lock()
relay_is_on = False            # Current relay state
approach_hold_until = 0.0      # Resettable hold deadline from the approach camera
blindspot_clear_hold_until = 0.0  # Extra hold after blindspot becomes clear
relay_timer_remaining = 0.0    # Countdown value shown for the approach hold
detected_count_now = 0         # How many vehicles in current frame
approach_detected_now = False
blindspot_detected_now = False
relay_source_now = "None"

# Smart debouncing: tracks current detection event
current_event_id = None        # Current detection event ID (for 5-second debounce)


# ============================================================================
# CPU optimization for RPi
# ============================================================================
def configure_runtime():
    """Apply CPU-side optimizations for Raspberry Pi."""
    cv2.setUseOptimized(True)
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(max(1, min(2, os.cpu_count() or 1)))
    if torch is not None:
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
        thread_count = max(1, min(4, os.cpu_count() or 1))
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def warmup_model(model, frame_width, frame_height):
    """Run a single dry inference to reduce first-detection latency."""
    warmup_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    inference_context = torch.inference_mode if torch is not None else nullcontext
    start = time.perf_counter()
    with inference_context():
        model.predict(source=warmup_frame, **PREDICT_KWARGS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"[INFO] Warmup complete in {elapsed_ms:.0f} ms")


def open_camera(camera_config):
    """Open and configure a single camera feed."""
    camera_name = camera_config['name']
    camera_source = camera_config.get('device_path', camera_config.get('index'))

    print(f"[INFO] Opening {camera_name} camera ({camera_source})...")
    if sys.platform.startswith('linux'):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {camera_name} camera at {camera_source}.")

    # Request MJPG to reduce USB bandwidth and decode overhead.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or TARGET_FPS
    print(f"[INFO] {camera_name} camera: {actual_width}x{actual_height} @ {actual_fps}fps")

    return {
        'name': camera_name,
        'source': camera_source,
        'capture': cap,
        'width': actual_width,
        'height': actual_height,
        'fps': actual_fps,
    }


def annotate_camera_frame(frame, camera_name):
    """Add a camera label to an individual feed before composing it."""
    annotated = frame.copy()
    cv2.rectangle(annotated, (12, 12), (220, 48), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        camera_name,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def compose_camera_frames(camera_frames):
    """Resize feeds to a shared height and stitch them side-by-side."""
    target_height = min(frame.shape[0] for _, frame in camera_frames)
    prepared_frames = []
    layout = []
    x_offset = 0

    for camera_name, frame in camera_frames:
        if frame.shape[0] != target_height:
            scale = target_height / frame.shape[0]
            new_width = max(1, int(frame.shape[1] * scale))
            frame = cv2.resize(frame, (new_width, target_height))
        labeled_frame = annotate_camera_frame(frame, camera_name)
        prepared_frames.append(labeled_frame)
        layout.append({
            'name': camera_name,
            'x_start': x_offset,
            'x_end': x_offset + labeled_frame.shape[1],
            'width': labeled_frame.shape[1],
            'height': labeled_frame.shape[0],
        })
        x_offset += labeled_frame.shape[1]

    composite = cv2.hconcat(prepared_frames)
    return composite, layout


# ============================================================================
# GPIO Relay functions (Raspberry Pi only)
# ============================================================================
def set_relay_idle():
    """Set both relay pins to idle (HIGH = inactive for active-low relay)."""
    if not HAS_GPIO:
        return
    GPIO.output(RELAY_ON_PIN, GPIO.HIGH)
    GPIO.output(RELAY_OFF_PIN, GPIO.HIGH)


def initialize_gpio():
    """Set up GPIO pins for relay control. Returns True if successful."""
    if not HAS_GPIO:
        print("[INFO] RPi.GPIO not available — relay control disabled (normal on Windows).")
        return False
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_ON_PIN, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(RELAY_OFF_PIN, GPIO.OUT, initial=GPIO.HIGH)
    set_relay_idle()
    print(f"[INFO] GPIO initialized: ON=GPIO{RELAY_ON_PIN}, OFF=GPIO{RELAY_OFF_PIN}")
    return True


def pulse_relay_on():
    """Send a brief LOW pulse on the ON pin to activate the relay."""
    if not HAS_GPIO:
        return
    with relay_lock:
        GPIO.output(RELAY_OFF_PIN, GPIO.HIGH)
        GPIO.output(RELAY_ON_PIN, GPIO.LOW)
    time.sleep(RELAY_TRIGGER_PULSE_SECONDS)
    with relay_lock:
        GPIO.output(RELAY_ON_PIN, GPIO.HIGH)


def pulse_relay_off():
    """Send a brief LOW pulse on the OFF pin to deactivate the relay."""
    if not HAS_GPIO:
        return
    with relay_lock:
        GPIO.output(RELAY_ON_PIN, GPIO.HIGH)
        GPIO.output(RELAY_OFF_PIN, GPIO.LOW)
    time.sleep(RELAY_TRIGGER_PULSE_SECONDS)
    with relay_lock:
        GPIO.output(RELAY_OFF_PIN, GPIO.HIGH)


def update_relay_inputs(approach_detected, blindspot_detected):
    """Apply the agreed two-camera relay rules and return whether relay just turned on."""
    global relay_is_on, approach_hold_until, blindspot_clear_hold_until
    global approach_detected_now, blindspot_detected_now

    now = time.time()
    activated = False

    with relay_lock:
        approach_detected_now = approach_detected
        blindspot_detected_now = blindspot_detected

        if approach_detected:
            approach_hold_until = now + APPROACH_HOLD_SECONDS

        if blindspot_detected:
            blindspot_clear_hold_until = now + BLINDSPOT_CLEAR_HOLD_SECONDS

        approach_active = approach_hold_until > now
        blindspot_active = blindspot_detected or blindspot_clear_hold_until > now

        if (approach_active or blindspot_active) and not relay_is_on:
            relay_is_on = True
            activated = True

    if activated:
        # Pulse ON in a short thread to avoid blocking inference.
        threading.Thread(target=pulse_relay_on, daemon=True).start()

    return activated


def get_relay_snapshot():
    """Return the current relay state based on approach hold and blindspot occupancy."""
    global relay_is_on, relay_timer_remaining, relay_source_now

    now = time.time()
    turn_off = False

    with relay_lock:
        approach_remaining = max(0.0, approach_hold_until - now)
        blindspot_clear_remaining = max(0.0, blindspot_clear_hold_until - now)
        blindspot_active = blindspot_detected_now or blindspot_clear_remaining > 0.0
        approach_active = approach_remaining > 0.0

        relay_timer_remaining = approach_remaining

        if approach_active and blindspot_active:
            relay_source_now = "Both"
        elif blindspot_active:
            relay_source_now = "Blindspot"
        elif approach_active:
            relay_source_now = "Approach"
        else:
            relay_source_now = "None"

        if relay_is_on and not (approach_active or blindspot_active):
            relay_is_on = False
            turn_off = True

        snapshot = {
            'relay_on': relay_is_on,
            'approach_remaining': approach_remaining,
            'blindspot_clear_remaining': blindspot_clear_remaining,
            'approach_detected': approach_detected_now,
            'blindspot_detected': blindspot_detected_now,
            'relay_source': relay_source_now,
        }

    if turn_off:
        threading.Thread(target=pulse_relay_off, daemon=True).start()
        log_relay_timeout()

    return snapshot


def cleanup_gpio():
    """Release GPIO pins on shutdown."""
    global relay_is_on
    if not HAS_GPIO:
        return
    # Make sure relay is OFF before cleanup
    if relay_is_on:
        pulse_relay_off()
        relay_is_on = False
    with relay_lock:
        set_relay_idle()
        GPIO.cleanup((RELAY_ON_PIN, RELAY_OFF_PIN))
    print("[INFO] GPIO cleaned up.")


# ============================================================================
# Inference worker thread — runs YOLO on frames without blocking the camera
# ============================================================================
def inference_worker(model):
    """Run YOLO inference on the latest frame in a background thread.
    
    This is the key to smooth camera preview: the main thread never waits
    for YOLO to finish. Instead, inference runs in parallel and updates
    shared detection results.
    """
    global last_boxes, running, inference_error, detected_count_now
    global pending_inference, inference_ms, inference_fps

    while running:
        # Grab the latest raw frame. The main thread only copies frames when
        # there is no outstanding inference work, which keeps Pi CPU usage down.
        with state_lock:
            if pending_inference and latest_frame is not None:
                frame_idx = latest_frame_idx
                frame_for_infer = latest_frame
                layout_for_infer = list(latest_camera_layout)
                pending_inference = False
            else:
                frame_idx = latest_frame_idx
                frame_for_infer = None
                layout_for_infer = []

        if frame_for_infer is None:
            time.sleep(0.002)
            continue

        try:
            infer_start = time.perf_counter()
            inference_context = torch.inference_mode if torch is not None else nullcontext
            with inference_context():
                results = model.predict(source=frame_for_infer, **PREDICT_KWARGS)
            elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
        except Exception as exc:
            inference_error = str(exc)
            running = False
            break

        current_boxes = []
        camera_detect_counts = {
            APPROACH_CAMERA_NAME: 0,
            BLINDSPOT_CAMERA_NAME: 0,
        }
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu().tolist()

            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                camera_name = "Unknown"
                for camera_info in layout_for_infer:
                    if camera_info['x_start'] <= cx < camera_info['x_end']:
                        camera_name = camera_info['name']
                        break
                if camera_name in camera_detect_counts:
                    camera_detect_counts[camera_name] += 1
                current_boxes.append((camera_name, x1, y1, x2, y2, float(confidence)))

        # Smart Detection Event Logging (Web App Integration):
        #   Vehicle detected → log event (handles 5-second debouncing internally)
        #   Multiple vehicles within 5s → same event (no new row)
        #   After 5s timeout → new event on next detection
        global current_event_id
        approach_detected = camera_detect_counts[APPROACH_CAMERA_NAME] > 0
        blindspot_detected = camera_detect_counts[BLINDSPOT_CAMERA_NAME] > 0

        if current_boxes:
            # Log detection (intelligently handles debouncing)
            event_id = log_detection_event()
            current_event_id = event_id
            relay_activated = update_relay_inputs(approach_detected, blindspot_detected)
            if relay_activated:
                log_relay_signal(event_id)
        else:
            update_relay_inputs(False, False)
        
        # Update detected count and smoothed inference stats for UI display.
        with state_lock:
            detected_count_now = len(current_boxes)
            last_boxes = current_boxes
            if inference_ms == 0.0:
                inference_ms = elapsed_ms
            else:
                inference_ms = (inference_ms * 0.8) + (elapsed_ms * 0.2)
            inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0


# ============================================================================
# Main — camera capture + display on main thread
# ============================================================================
def main():
    global latest_frame, latest_frame_idx, latest_camera_layout, pending_inference, running

    configure_runtime()

    # Initialize GPIO (no-op on Windows)
    gpio_ready = initialize_gpio()

    # Start Flask Web App in background thread
    print("[INFO] Starting Flask web app on port 5000...")
    flask_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    time.sleep(2)  # Give Flask time to start
    print("[INFO] Web app running at http://<your-ip>:5000")

    if hotspot_feature_enabled():
        enable_hotspot(ssid="VehicleDetector", password="detection123")
    else:
        print("[INFO] Hotspot auto-enable is off. Web app stays on the current network connection.")
    
    # Update app state to indicate detection is ready
    set_app_state(
        running=True,
        detection_enabled=False,
        approach_detected=False,
        blindspot_detected=False,
        relay_source="None",
        approach_timer_remaining=0.0,
        blindspot_clear_remaining=0.0,
    )

    # Load model
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    try:
        model.fuse()
        print("[INFO] Model fused for faster CPU inference.")
    except Exception as exc:
        print(f"[WARN] Model fuse skipped: {exc}")
    print("[INFO] Model loaded successfully.")

    camera_streams = []
    try:
        for camera_config in CAMERA_CONFIGS:
            camera_streams.append(open_camera(camera_config))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        for stream in camera_streams:
            stream['capture'].release()
        sys.exit(1)

    initial_frames = []
    for stream in camera_streams:
        ret, frame = stream['capture'].read()
        if not ret:
            print(f"[ERROR] Failed to read initial frame from {stream['name']} camera.")
            for active_stream in camera_streams:
                active_stream['capture'].release()
            sys.exit(1)
        initial_frames.append((stream['name'], frame))

    composite_frame, camera_layout = compose_camera_frames(initial_frames)
    actual_width = composite_frame.shape[1]
    actual_height = composite_frame.shape[0]
    fps = min(stream['fps'] for stream in camera_streams) if camera_streams else TARGET_FPS

    warmup_model(model, actual_width, actual_height)

    # Video writer (optional)
    out = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (actual_width, actual_height))
        print(f"[INFO] Recording to: {OUTPUT_PATH}")

    # Vehicle counter state
    vehicle_count = 0
    crossed_ids = set()  # Track which detection zones have crossed the line

    # Start inference thread
    infer_thread = threading.Thread(target=inference_worker, args=(model,), daemon=True)
    infer_thread.start()

    frame_idx = 0
    prev_time = time.time()

    print("[INFO] Starting vehicle detection... Press 'Q' to quit.")

    try:
        while running:
            camera_frames = []
            missing_frame = False
            for stream in camera_streams:
                ret, frame = stream['capture'].read()
                if not ret:
                    print(f"[WARN] Failed to grab frame from {stream['name']} camera. Retrying...")
                    missing_frame = True
                    break
                camera_frames.append((stream['name'], frame))

            if missing_frame:
                continue

            frame, camera_layout = compose_camera_frames(camera_frames)

            frame_idx += 1

            # Queue only raw frames that will actually be inferred. This avoids
            # feeding the model UI overlays and keeps copy work close to the
            # real detector throughput instead of the camera FPS.
            with state_lock:
                if frame_idx % FRAME_SKIP == 0 and not pending_inference:
                    latest_frame = frame.copy()
                    latest_frame_idx = frame_idx
                    latest_camera_layout = list(camera_layout)
                    pending_inference = True
                boxes_snapshot = list(last_boxes)
                current_detected = detected_count_now
                last_infer_ms = inference_ms
                last_infer_fps = inference_fps

            # --- Draw counting lines and camera divider ---
            for idx, camera_info in enumerate(camera_layout):
                line_y = int(camera_info['height'] * LINE_POSITION_RATIO)
                line_x_start = camera_info['x_start'] + LINE_MARGIN_X
                line_x_end = camera_info['x_end'] - LINE_MARGIN_X
                cv2.line(frame, (line_x_start, line_y), (line_x_end, line_y), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"{camera_info['name']} Line",
                    (line_x_start, max(24, line_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if idx > 0:
                    divider_x = camera_info['x_start']
                    cv2.line(frame, (divider_x, 0), (divider_x, actual_height), (255, 255, 255), 2)

            # --- Draw detections ---
            for camera_name, x1, y1, x2, y2, confidence in boxes_snapshot:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                camera_line_y = int(actual_height * LINE_POSITION_RATIO)
                for camera_info in camera_layout:
                    if camera_info['name'] == camera_name:
                        camera_line_y = int(camera_info['height'] * LINE_POSITION_RATIO)
                        break

                # Draw bounding box with generic "Vehicle" label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{camera_name} {GENERIC_LABEL} {confidence:.2f}',
                            (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Count vehicles crossing each camera's line independently.
                box_id = (camera_name, x1 // 10, y1 // 10, x2 // 10, y2 // 10)
                if cy > camera_line_y and box_id not in crossed_ids:
                    crossed_ids.add(box_id)
                    vehicle_count += 1

            # --- Check relay state (approach prediction + blindspot override) ---
            relay_snapshot = get_relay_snapshot()
            is_relay_on = relay_snapshot['relay_on']
            remaining = relay_snapshot['approach_remaining']
            set_app_state(
                running=True,
                detection_enabled=True,
                approach_detected=relay_snapshot['approach_detected'],
                blindspot_detected=relay_snapshot['blindspot_detected'],
                relay_source=relay_snapshot['relay_source'],
                approach_timer_remaining=relay_snapshot['approach_remaining'],
                blindspot_clear_remaining=relay_snapshot['blindspot_clear_remaining'],
            )

            # --- Draw status panel (top-left labels) ---
            label_y = 30

            # Vehicle count
            cv2.putText(frame, f"Vehicles Counted: {vehicle_count}", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            label_y += 30

            # Detected right now
            cv2.putText(frame, f"Detected Now: {current_detected}", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            label_y += 30

            if SHOW_INFERENCE_STATS:
                cv2.putText(frame, f"Infer: {last_infer_ms:.0f} ms ({last_infer_fps:.1f} fps)",
                            (10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                label_y += 30

            # Relay status
            if is_relay_on:
                relay_status = "ON"
                relay_color = (0, 255, 0)     # Green
            else:
                relay_status = "OFF"
                relay_color = (0, 0, 255)     # Red
            cv2.putText(frame, f"Relay: {relay_status}", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, relay_color, 2)
            label_y += 30

            approach_text = "YES" if relay_snapshot['approach_detected'] else "NO"
            cv2.putText(frame, f"Approach Detect: {approach_text}", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            label_y += 30

            blindspot_text = "YES" if relay_snapshot['blindspot_detected'] else "NO"
            cv2.putText(frame, f"Blindspot Detect: {blindspot_text}", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            label_y += 30

            if relay_snapshot['approach_remaining'] > 0:
                cv2.putText(frame, f"Approach Hold: {relay_snapshot['approach_remaining']:.1f}s", (10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Approach Hold: --", (10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 2)
            label_y += 30

            if relay_snapshot['blindspot_clear_remaining'] > 0:
                cv2.putText(frame, f"Blindspot Hold: {relay_snapshot['blindspot_clear_remaining']:.1f}s", (10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Blindspot Hold: --", (10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 2)
            label_y += 30

            cv2.putText(frame, f"Relay Source: {relay_snapshot['relay_source']}", (10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            label_y += 30

            # --- Display FPS (top-right) ---
            curr_time = time.time()
            fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (actual_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Save frame
            if out is not None:
                out.write(frame)

            # Display
            cv2.imshow(WINDOW_NAME, frame)

            # Press 'Q' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or Esc
                print("[INFO] Quitting...")
                break

            # Check for inference errors
            if inference_error:
                print(f"[ERROR] Inference failed: {inference_error}")
                break

    finally:
        running = False
        
        # Update app state
        set_app_state(
            running=False,
            detection_enabled=False,
            approach_detected=False,
            blindspot_detected=False,
            relay_source="None",
            approach_timer_remaining=0.0,
            blindspot_clear_remaining=0.0,
        )

        # Wait for inference thread to finish
        if infer_thread is not None:
            infer_thread.join(timeout=2.0)

        # Release resources
        for stream in camera_streams:
            stream['capture'].release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        # Cleanup GPIO
        if gpio_ready:
            cleanup_gpio()
        
        # Disable WiFi hotspot (no-op on non-RPi)
        disable_hotspot()
        
        print("[INFO] Hotspot disabled and cleanup complete.")

    # Print summary
    print()
    print("=" * 50)
    print("  Vehicle Count Summary")
    print("=" * 50)
    print(f"  Total Vehicles Counted: {vehicle_count}")
    print("=" * 50)

    if SAVE_OUTPUT:
        print(f"\n  Output saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
