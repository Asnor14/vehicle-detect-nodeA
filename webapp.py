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
        
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'detection_time': self.detection_time.isoformat() if self.detection_time else None,
            'relay_signal_time': self.relay_signal_time.isoformat() if self.relay_signal_time else None,
            'relay_off_time': self.relay_off_time.isoformat() if self.relay_off_time else None,
            'vehicle_count': self.vehicle_count,
            'status': self.status,
            'duration_seconds': self._calculate_duration()
        }
    
    def _calculate_duration(self):
        """Calculate duration from relay signal to relay off."""
        if self.relay_signal_time and self.relay_off_time:
            delta = self.relay_off_time - self.relay_signal_time
            return round(delta.total_seconds(), 2)
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
# REST API Endpoints
# ============================================================================

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return render_template('dashboard.html')


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


# ============================================================================
# Helper Functions (called from main.py)
# ============================================================================

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
