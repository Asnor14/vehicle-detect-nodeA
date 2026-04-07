"""
WiFi Hotspot Control for Raspberry Pi
======================================
Manages WiFi hotspot enabling/disabling using nmcli.
Safe to use on Windows (no-op).
"""

import subprocess
import sys
import os

HOTSPOT_CONN_NAME = "VehicleDetectorHotspot"


def is_raspberry_pi():
    """Check if running on Raspberry Pi."""
    return sys.platform.startswith('linux') and os.path.exists('/sys/firmware/devicetree/base/model')


def hotspot_feature_enabled():
    """Return True when hotspot control is explicitly enabled."""
    value = os.environ.get("VEHICLE_ENABLE_HOTSPOT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def enable_hotspot(ssid="VehicleDetector", password="detection123"):
    """Enable WiFi hotspot on Raspberry Pi.
    
    Args:
        ssid: Network name
        password: WiFi password
    
    Returns:
        bool: True if successful or not on RPi
    """
    if not is_raspberry_pi():
        print("[INFO] Not on RPi - hotspot control skipped.")
        return True
    if not hotspot_feature_enabled():
        print("[INFO] Hotspot disabled by config (set VEHICLE_ENABLE_HOTSPOT=1 to enable).")
        return True
    
    try:
        # Check if nmcli is available
        subprocess.run(['which', 'nmcli'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARN] nmcli not found. Install NetworkManager or use hostapd.")
        return False
    
    try:
        # Create or reuse hotspot connection. NetworkManager 1.42 requires
        # the Wi-Fi band when a channel is specified.
        cmd = [
            'nmcli', 'device', 'wifi', 'hotspot',
            'ifname', 'wlan0',
            'ssid', ssid,
            'password', password,
            'con-name', HOTSPOT_CONN_NAME,
            'band', 'bg',
            'channel', '6'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[INFO] Hotspot enabled: {ssid}")
            return True
        else:
            print(f"[WARN] Hotspot enable failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Hotspot error: {e}")
        return False


def disable_hotspot():
    """Disable WiFi hotspot on Raspberry Pi.
    
    Returns:
        bool: True if successful or not on RPi
    """
    if not is_raspberry_pi():
        print("[INFO] Not on RPi - hotspot control skipped.")
        return True
    if not hotspot_feature_enabled():
        print("[INFO] Hotspot disabled by config; nothing to disable.")
        return True
    
    try:
        # Bring down the hotspot connection if it exists. `nmcli device wifi hotspot off`
        # is not supported on this NetworkManager version.
        cmd = ['nmcli', 'connection', 'down', HOTSPOT_CONN_NAME]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[INFO] Hotspot disabled")
            return True
        else:
            message = (result.stderr or result.stdout).strip()
            if "unknown connection" in message.lower() or "not an active connection" in message.lower():
                print("[INFO] Hotspot already inactive.")
                return True
            print(f"[WARN] Hotspot disable failed: {message}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Hotspot disable error: {e}")
        return False


def get_hotspot_status():
    """Get current hotspot status.
    
    Returns:
        dict: {'active': bool, 'ssid': str or None}
    """
    if not is_raspberry_pi() or not hotspot_feature_enabled():
        return {'active': False, 'ssid': None}
    
    try:
        result = subprocess.run(
            ['nmcli', '-t', '-f', 'NAME,TYPE,DEVICE', 'connection', 'show', '--active'],
            capture_output=True, text=True, check=False
        )

        for line in result.stdout.splitlines():
            parts = line.split(':')
            if len(parts) >= 3 and parts[0] == HOTSPOT_CONN_NAME and parts[1] == 'wifi':
                return {'active': True, 'ssid': HOTSPOT_CONN_NAME}

        return {'active': False, 'ssid': None}
        
    except Exception as e:
        print(f"[WARN] Could not check hotspot status: {e}")
        return {'active': False, 'ssid': None}


if __name__ == '__main__':
    # Test script
    print("Testing hotspot control...")
    print(f"Is RPi: {is_raspberry_pi()}")
    print(f"Status: {get_hotspot_status()}")
    
    if sys.argv[1:]:
        if sys.argv[1] == 'enable':
            enable_hotspot()
        elif sys.argv[1] == 'disable':
            disable_hotspot()
        elif sys.argv[1] == 'status':
            print(get_hotspot_status())
