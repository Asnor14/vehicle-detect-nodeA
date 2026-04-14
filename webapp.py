"""
Flask Web App for Vehicle Detection with Smart Event Logging
=============================================================
Provides REST API for detection events and serves dashboard.
Implements 5-second debouncing logic for multi-vehicle events.
"""

import os
import json
import threading
import time
from datetime import datetime
from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from hotspot import (
    disable_hotspot,
    enable_hotspot,
    get_hotspot_status,
    hotspot_feature_enabled,
)

# ============================================================================
# Event Management System (5-second debouncing)
# ============================================================================

class DetectionEvent:
    """Represents a single detection event/session."""
    def __init__(self, event_id):
        self.event_id = event_id
        self.detection_time = None
        self.relay_signal_time = None
        self.relay_off_time = None
        self.vehicle_count = 0
        self.status = "pending"  # pending, active, completed
        self.camera_name = None  # Camera that triggered detection
        
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'detection_time': self.detection_time.isoformat() if self.detection_time else None,
            'relay_signal_time': self.relay_signal_time.isoformat() if self.relay_signal_time else None,
            'relay_off_time': self.relay_off_time.isoformat() if self.relay_off_time else None,
            'vehicle_count': self.vehicle_count,
            'status': self.status,
            'duration_seconds': self._calculate_duration(),
            'camera_name': self.camera_name,
            'signal_time_ms': self._calculate_signal_time_ms()
        }
    
    def _calculate_duration(self):
        """Calculate duration from relay signal to relay off."""
        if self.relay_signal_time and self.relay_off_time:
            delta = self.relay_off_time - self.relay_signal_time
            return round(delta.total_seconds(), 2)
        return None
    
    def _calculate_signal_time_ms(self):
        """Calculate time from detection to relay signal in milliseconds."""
        if self.detection_time and self.relay_signal_time:
            delta = self.relay_signal_time - self.detection_time
            return round(delta.total_seconds() * 1000, 1)
        return None


class EventLogger:
    """Manages detection events with 5-second debouncing logic."""
    
    def __init__(self, debounce_seconds=5.0):
        self.debounce_seconds = debounce_seconds
        self.events = []
        self.current_event = None
        self.event_counter = 0
        self.lock = threading.Lock()
        self.last_detection_time = None
        self.session_start_time = None
        self.detection_camera_context = None  # Camera that triggered detection
    
    def set_detection_camera(self, camera_name):
        """Set the camera that triggered the current detection."""
        self.detection_camera_context = camera_name
    
    def on_vehicle_detected(self):
        """Call when a vehicle is detected. Returns event_id."""
        with self.lock:
            now = datetime.now()
            
            # Start a new event only when there is no active event. While the
            # timer is still being reset by fresh detections, stay on the same
            # event id and keep extending the same session.
            if self.current_event is None:
                self.event_counter += 1
                self.current_event = DetectionEvent(self.event_counter)
                self.current_event.detection_time = now
                self.current_event.status = "pending"
                self.session_start_time = now
                self.current_event.vehicle_count = 1
                self.current_event.camera_name = self.detection_camera_context
                self.last_detection_time = now
                return self.current_event.event_id

            self.current_event.vehicle_count += 1
            self.last_detection_time = now
            return self.current_event.event_id
    
    def on_relay_signal(self, event_id):
        """Call when relay is activated for an event."""
        with self.lock:
            if self.current_event and self.current_event.event_id == event_id:
                self.current_event.relay_signal_time = datetime.now()
                self.current_event.status = "active"
    
    def on_relay_timeout(self):
        """Call when relay timer expires. Finalizes the current event."""
        with self.lock:
            if self.current_event:
                self.current_event.relay_off_time = datetime.now()
                self.current_event.status = "completed"
                self.events.append(self.current_event)
                self.current_event = None
                self.last_detection_time = None
                self.session_start_time = None
    
    def _finalize_current_event(self):
        """Finalize current event and save to history."""
        if self.current_event:
            if not self.current_event.relay_off_time:
                self.current_event.relay_off_time = datetime.now()
            self.current_event.status = "completed"
            self.events.append(self.current_event)
            self.current_event = None
            self.last_detection_time = None
            self.session_start_time = None
    
    def get_all_events(self):
        """Return all completed events as dictionaries."""
        with self.lock:
            return [e.to_dict() for e in self.events]
    
    def get_current_event(self):
        """Return current pending event if any."""
        with self.lock:
            if self.current_event:
                return self.current_event.to_dict()
            return None
    
    def check_timeout(self):
        """Check if current event should be finalized (helper for main loop)."""
        with self.lock:
            if self.current_event and self.last_detection_time:
                elapsed = (datetime.now() - self.last_detection_time).total_seconds()
                if elapsed >= self.debounce_seconds:
                    return True
        return False


# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global event logger (shared with main.py via this module)
event_logger = EventLogger(debounce_seconds=5.0)

# Application state
app_state = {
    'running': False,
    'detection_enabled': False,
    'relay_enabled': True,
    'approach_detected': False,
    'blindspot_detected': False,
    'relay_source': 'None',
    'approach_timer_remaining': 0.0,
    'blindspot_clear_remaining': 0.0,
}
state_lock = threading.Lock()


# ============================================================================
# Detection Trial Recorder
# ============================================================================

TRIAL_LABELS = ["trial_1", "trial_2", "trial_3"]
RESPONSE_SEQUENCE = [
    'detection_trial_1',
    'response_trial_1',
    'detection_trial_2',
    'response_trial_2',
    'detection_trial_3',
    'response_trial_3',
]
trial_lock = threading.Lock()
trial_recorder = {
    'rows': [
        {
            'number': number,
            'trial_1_ms': None,
            'trial_2_ms': None,
            'trial_3_ms': None,
        }
        for number in range(1, 11)
    ],
    'current_row': 1,
    'current_trial': 'trial_1',
}
response_trial_lock = threading.Lock()
response_trial_recorder = {
    'rows': [
        {
            'number': number,
            'detection_trial_1_ms': None,
            'detection_trial_2_ms': None,
            'detection_trial_3_ms': None,
            'response_trial_1_ms': None,
            'response_trial_2_ms': None,
            'response_trial_3_ms': None,
        }
        for number in range(1, 11)
    ],
    'current_row': 1,
    'current_slot': 'detection_trial_1',
}


def _snapshot_trial_recorder():
    """Build a serializable recorder snapshot. Caller manages locking."""
    rows = [row.copy() for row in trial_recorder['rows']]
    current_row = trial_recorder['current_row']
    current_trial = trial_recorder['current_trial']

    return {
        'rows': rows,
        'current_row': current_row,
        'current_trial': current_trial,
        'completed': current_row is None,
    }


def get_trial_recorder_state():
    """Return a snapshot of the current manual trial recorder state."""
    with trial_lock:
        return _snapshot_trial_recorder()


def _row_is_complete(row):
    """Return True when all three trial slots are filled."""
    return all(row[f'{trial}_ms'] is not None for trial in TRIAL_LABELS)


def _snapshot_response_trial_recorder():
    """Build a serializable response-time recorder snapshot."""
    rows = [row.copy() for row in response_trial_recorder['rows']]
    current_row = response_trial_recorder['current_row']
    current_slot = response_trial_recorder['current_slot']

    return {
        'rows': rows,
        'current_row': current_row,
        'current_slot': current_slot,
        'completed': current_row is None,
    }


def get_response_trial_recorder_state():
    """Return a snapshot of the response-time recorder state."""
    with response_trial_lock:
        return _snapshot_response_trial_recorder()


def _response_row_is_complete(row):
    """Return True when all detection and response trials are filled."""
    return all(row[f'{slot}_ms'] is not None for slot in RESPONSE_SEQUENCE)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return render_template('dashboard.html')


@app.route('/detection-time')
def detection_time():
    """Serve the manual detection-time recorder page."""
    return render_template('detection_time.html')


@app.route('/response-time')
def response_time():
    """Serve the paired detection and relay response time recorder page."""
    return render_template('response_time.html')


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start vehicle detection and enable hotspot."""
    with state_lock:
        app_state['running'] = True
        app_state['detection_enabled'] = True
    
    hotspot_ok = enable_hotspot() if hotspot_feature_enabled() else True
    hotspot = get_hotspot_status()
    message = 'Detection started on the current network.'
    if hotspot_feature_enabled():
        message = 'Detection started. Hotspot enabled if on RPi.'
    
    return jsonify({
        'status': 'started',
        'message': message,
        'hotspot_enabled': hotspot_ok,
        'hotspot': hotspot,
        'hotspot_configured': hotspot_feature_enabled(),
    }), 200


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop vehicle detection and disable hotspot."""
    with state_lock:
        app_state['running'] = False
        app_state['detection_enabled'] = False
    
    # Finalize any pending event
    event_logger.on_relay_timeout()
    
    hotspot_ok = disable_hotspot() if hotspot_feature_enabled() else True
    hotspot = get_hotspot_status()
    message = 'Detection stopped.'
    if hotspot_feature_enabled():
        message = 'Detection stopped. Hotspot disabled if on RPi.'
    
    return jsonify({
        'status': 'stopped',
        'message': message,
        'hotspot_disabled': hotspot_ok,
        'hotspot': hotspot,
        'hotspot_configured': hotspot_feature_enabled(),
    }), 200


@app.route('/api/events', methods=['GET'])
def get_events():
    """Get all completed detection events."""
    all_events = event_logger.get_all_events()
    current = event_logger.get_current_event()
    
    return jsonify({
        'events': all_events,
        'current_event': current,
        'total_events': len(all_events)
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current app status."""
    with state_lock:
        status = app_state.copy()
    
    current_event = event_logger.get_current_event()
    
    return jsonify({
        **status,
        'hotspot': get_hotspot_status(),
        'hotspot_configured': hotspot_feature_enabled(),
        'current_event': current_event,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/clear-events', methods=['POST'])
def clear_events():
    """Clear all stored events (for testing)."""
    with event_logger.lock:
        event_logger.events = []
        event_logger.current_event = None
    
    return jsonify({'status': 'cleared'}), 200


@app.route('/api/trial-recorder', methods=['GET'])
def get_trial_recorder():
    """Get the current state of the manual detection-time recorder."""
    return jsonify(get_trial_recorder_state()), 200


@app.route('/api/trial-recorder/record', methods=['POST'])
def record_trial_time():
    """Record a completed manual trial time in milliseconds."""
    payload = request.get_json(silent=True) or {}
    elapsed_ms = payload.get('elapsed_ms')

    if not isinstance(elapsed_ms, (int, float)) or elapsed_ms < 0:
        return jsonify({'error': 'elapsed_ms must be a non-negative number.'}), 400

    elapsed_ms = round(float(elapsed_ms), 3)

    with trial_lock:
        current_row = trial_recorder['current_row']
        current_trial = trial_recorder['current_trial']

        if current_row is None or current_trial is None:
            return jsonify({'error': 'All trial rows are already complete.'}), 400

        row = trial_recorder['rows'][current_row - 1]
        key = f'{current_trial}_ms'
        row[key] = elapsed_ms

        next_trial = None
        for trial in TRIAL_LABELS:
            if row[f'{trial}_ms'] is None:
                next_trial = trial
                break

        trial_recorder['current_trial'] = next_trial

        state = _snapshot_trial_recorder()

    return jsonify({
        'status': 'recorded',
        'recorded_row': current_row,
        'recorded_trial': current_trial,
        **state,
    }), 200


@app.route('/api/trial-recorder/proceed', methods=['POST'])
def proceed_trial_row():
    """Move the recorder to the next numbered row after three trials."""
    with trial_lock:
        current_row = trial_recorder['current_row']

        if current_row is None:
            return jsonify({'error': 'All trial rows are already complete.'}), 400

        row = trial_recorder['rows'][current_row - 1]
        if not _row_is_complete(row):
            return jsonify({'error': 'Complete Trial 1, Trial 2, and Trial 3 first.'}), 400

        if current_row >= len(trial_recorder['rows']):
            trial_recorder['current_row'] = None
            trial_recorder['current_trial'] = None
        else:
            trial_recorder['current_row'] = current_row + 1
            trial_recorder['current_trial'] = 'trial_1'

        state = _snapshot_trial_recorder()

    return jsonify({
        'status': 'proceeded',
        **state,
    }), 200


@app.route('/api/trial-recorder/reset', methods=['POST'])
def reset_trial_recorder():
    """Reset all manual detection-time recorder rows."""
    with trial_lock:
        for row in trial_recorder['rows']:
            row['trial_1_ms'] = None
            row['trial_2_ms'] = None
            row['trial_3_ms'] = None

        trial_recorder['current_row'] = 1
        trial_recorder['current_trial'] = 'trial_1'

    return jsonify({
        'status': 'reset',
        **get_trial_recorder_state(),
    }), 200


@app.route('/api/response-recorder', methods=['GET'])
def get_response_recorder():
    """Get the current state of the response-time recorder."""
    return jsonify(get_response_trial_recorder_state()), 200


@app.route('/api/response-recorder/record', methods=['POST'])
def record_response_trial_time():
    """Record a detection-time or relay-response-time trial in milliseconds."""
    payload = request.get_json(silent=True) or {}
    elapsed_ms = payload.get('elapsed_ms')

    if not isinstance(elapsed_ms, (int, float)) or elapsed_ms < 0:
        return jsonify({'error': 'elapsed_ms must be a non-negative number.'}), 400

    elapsed_ms = round(float(elapsed_ms), 3)

    with response_trial_lock:
        current_row = response_trial_recorder['current_row']
        current_slot = response_trial_recorder['current_slot']

        if current_row is None or current_slot is None:
            return jsonify({'error': 'All response-time rows are already complete.'}), 400

        row = response_trial_recorder['rows'][current_row - 1]
        row[f'{current_slot}_ms'] = elapsed_ms

        next_slot = None
        for slot in RESPONSE_SEQUENCE:
            if row[f'{slot}_ms'] is None:
                next_slot = slot
                break

        response_trial_recorder['current_slot'] = next_slot
        state = _snapshot_response_trial_recorder()

    return jsonify({
        'status': 'recorded',
        'recorded_row': current_row,
        'recorded_slot': current_slot,
        **state,
    }), 200


@app.route('/api/response-recorder/proceed', methods=['POST'])
def proceed_response_trial_row():
    """Move the response-time recorder to the next numbered row."""
    with response_trial_lock:
        current_row = response_trial_recorder['current_row']

        if current_row is None:
            return jsonify({'error': 'All response-time rows are already complete.'}), 400

        row = response_trial_recorder['rows'][current_row - 1]
        if not _response_row_is_complete(row):
            return jsonify({'error': 'Complete all detection and response trials first.'}), 400

        if current_row >= len(response_trial_recorder['rows']):
            response_trial_recorder['current_row'] = None
            response_trial_recorder['current_slot'] = None
        else:
            response_trial_recorder['current_row'] = current_row + 1
            response_trial_recorder['current_slot'] = 'detection_trial_1'

        state = _snapshot_response_trial_recorder()

    return jsonify({
        'status': 'proceeded',
        **state,
    }), 200


@app.route('/api/response-recorder/reset', methods=['POST'])
def reset_response_trial_recorder():
    """Reset all response-time rows."""
    with response_trial_lock:
        for row in response_trial_recorder['rows']:
            row['detection_trial_1_ms'] = None
            row['detection_trial_2_ms'] = None
            row['detection_trial_3_ms'] = None
            row['response_trial_1_ms'] = None
            row['response_trial_2_ms'] = None
            row['response_trial_3_ms'] = None

        response_trial_recorder['current_row'] = 1
        response_trial_recorder['current_slot'] = 'detection_trial_1'

    return jsonify({
        'status': 'reset',
        **get_response_trial_recorder_state(),
    }), 200


# ============================================================================
# Helper Functions (called from main.py)
# ============================================================================

def set_detection_camera(camera_name):
    """Set the camera that triggered the current detection (call before log_detection_event)."""
    event_logger.set_detection_camera(camera_name)


def log_detection_event():
    """Call from main.py inference_worker when vehicle detected.
    
    Returns the event_id (same for multiple detections within 5s).
    """
    return event_logger.on_vehicle_detected()


def log_relay_signal(event_id):
    """Call from main.py when relay is activated."""
    event_logger.on_relay_signal(event_id)


def log_relay_timeout():
    """Call from main.py when relay timer expires."""
    event_logger.on_relay_timeout()


def should_finalize_event():
    """Check if current event should be finalized (helper for timer check)."""
    return event_logger.check_timeout()


def get_app_state():
    """Get current application state."""
    with state_lock:
        return app_state.copy()


def set_app_state(running=None, detection_enabled=None, **extra_fields):
    """Set application state."""
    with state_lock:
        if running is not None:
            app_state['running'] = running
        if detection_enabled is not None:
            app_state['detection_enabled'] = detection_enabled
        app_state.update(extra_fields)


if __name__ == '__main__':
    # For testing only
    app.run(host='0.0.0.0', port=5000, debug=False)
