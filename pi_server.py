import os
import time
import math
import shutil
import subprocess
import threading
import json
try:
    import serial
except ImportError:
    serial = None

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory

# Attempt importing rccar autonomous pipeline components
try:
    from rccar.calibration.homography_api import load_homography
    from rccar.capture.source import FrameSource
    from rccar.capture.live import LiveCameraSource
    from rccar.capture.file import VideoFileSource
    from rccar.serial_client.protocol import decode_command, SpeedTier
    from rccar.watchdog.watchdog import Watchdog
    from rccar.viz.overlay import draw_overlay
    from rccar.main import PipelineState, process_frame
    from rccar.segmentation.classify import AdaptiveClassifier
    from rccar.curb.confidence import CurbConfidenceTracker
    from rccar.decision.smoothing import MajorityVoteSmoother
    from rccar.decision.speed import load_thresholds
    RCCAR_AVAILABLE = True
    print("[RCCAR SUCCESS] rccar autonomous driving pipeline loaded.")
except ImportError as e:
    RCCAR_AVAILABLE = False
    print(f"[RCCAR INFO] rccar module not found ({e}). Running server with manual teleoperation and simulated auto capability.")

app = Flask(__name__)

SERIAL_PORTS = ['/dev/serial0', '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyAMA0', '/dev/ttyS0']
BAUD_RATE = 115200
ser = None
serial_lock = threading.Lock()


def send_serial(cmd_str):
    """Safely writes newline-terminated JSON commands to serial with mutex locking."""
    if ser and ser.is_open:
        try:
            with serial_lock:
                ser.write((cmd_str + '\n').encode('utf-8'))
                ser.flush()
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")


# Attempt connection across known serial interfaces
if serial is not None:
    for port in SERIAL_PORTS:
        try:
            s = serial.Serial(
                port=port,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            s.reset_input_buffer()
            s.reset_output_buffer()
            time.sleep(0.1)
            ser = s
            print(f"[SERIAL SUCCESS] Connected to {port}")

            # Initialize UGV02 firmware mode and continuous feedback
            send_serial('{"T":5,"main":2,"module":0}')
            time.sleep(0.05)
            send_serial('{"T":131,"cmd":1}')
            print("[SERIAL SUCCESS] UGV02 firmware mode and continuous feedback initialized.")
            break
        except Exception as e:
            continue

if ser is None:
    print(f"[SERIAL WARNING] Could not open any serial port in {SERIAL_PORTS}. Running in offline/simulation mode.")

# Velocity Control Mapping (T:13, X: linear m/s, Z: angular rad/s)
# Differential Steering Kinematics (v_L = X - Z*W/2, v_R = X + Z*W/2, Track Width W = 0.20m):
# - Forward Straight: Both wheels forward at 100%
# - Forward + Turn Left (W+A): Left turn arc (Z > 0)
# - Forward + Turn Right (W+D): Right turn arc (Z < 0)
# - Backward + Turn Left (S+A): Left reverse arc (Z < 0)
# - Backward + Turn Right (S+D): Right reverse arc (Z > 0)
# - Stationary Turn Left (A): Controlled in-place left turn (Z = +0.75 rad/s)
# - Stationary Turn Right (D): Controlled in-place right turn (Z = -0.75 rad/s)
COMMAND_TARGETS = {
    'low': {
        'b': (-0.70, 0.0),    # Forward Straight
        'f': (0.70, 0.0),     # Backward Straight
        'l': (0.0, 0.75),     # In-place Turn Left (A)
        'r': (0.0, -0.75),    # In-place Turn Right (D)
        'bl': (-0.60, 1.40),  # Forward + Left (W+A)
        'br': (-0.60, -1.40), # Forward + Right (W+D)
        'fl': (0.60, -1.40),  # Backward + Left (S+A)
        'fr': (0.60, 1.40),   # Backward + Right (S+D)
        's': (0.0, 0.0),      # Stop
    },
    'high': {
        'b': (-1.15, 0.0),    # Forward Straight
        'f': (1.15, 0.0),     # Backward Straight
        'l': (0.0, 1.10),     # In-place Turn Left (A)
        'r': (0.0, -1.10),    # In-place Turn Right (D)
        'bl': (-0.98, 2.30),  # Forward + Left (W+A)
        'br': (-0.98, -2.30), # Forward + Right (W+D)
        'fl': (0.98, -2.30),  # Backward + Left (S+A)
        'fr': (0.98, 2.30),   # Backward + Right (S+D)
        's': (0.0, 0.0),      # Stop
    }
}

current_speed_mode = 'low'
last_cmd_time = time.time()
current_cmd = 's'
drive_mode = 'manual'  # 'manual' or 'auto'
auto_running = False
auto_thread = None
auto_stop_event = threading.Event()

target_x = 0.0
target_z = 0.0
curr_x = 0.0
curr_z = 0.0
motion_lock = threading.Lock()

# Acceleration limits for continuous, smooth turning and transitions
ACCEL_X = 2.4   # m/s^2 linear acceleration ramp (~250ms ramp-up)
DECEL_X = 3.2   # m/s^2 linear braking ramp
ACCEL_Z = 3.8   # rad/s^2 angular acceleration ramp (soft entry into turns)
DECEL_Z = 4.8   # rad/s^2 angular deceleration ramp (clean exit from turns)

# Auto-mode obstacle-avoidance scan: when blocked, pivot in place by
# SCAN_STEP_DEG at a time (re-checking for a clear path after each step)
# up to a total of SCAN_MAX_DEG before giving up and stopping for good.
# SCAN_MAX_DEG is deliberately small: the car must stay next to the curb
# it's following, so the scan may only nudge it around a small obstacle,
# never turn far enough to point it out into the middle of the road.
# SCAN_ANGULAR_RATE_RAD_S is deliberately well under UGV02SerialAdapter's
# MAX_STEER_Z (1.40 rad/s) -- this pivot has no forward speed to help
# stabilize it, so a slower, more controllable rate is used.
SCAN_STEP_DEG = 10
SCAN_MAX_DEG = 30
SCAN_ANGULAR_RATE_RAD_S = 0.7
SCAN_STEP_DURATION_S = math.radians(SCAN_STEP_DEG) / SCAN_ANGULAR_RATE_RAD_S

telemetry_data = {
    'pos_x': 0.0,
    'pos_y': 0.0,
    'distance_home_m': 0.0,
    'heading_home_deg': 0.0,
    'speed_mode': 'low',
    'drive_mode': 'manual',
    'rccar_available': RCCAR_AVAILABLE,
    'nearest_obs_cm': None,
    'curb_side': None,
    'curb_offset_cm': None,
    'battery_pct': 100,
    'voltage': 12.6,
    'is_recording': False,
    'rec_file': '',
    'free_storage_gb': 0.0
}

filtered_voltage = None


class UGV02SerialAdapter:
    """
    Bridges rccar's SerialClient interface and ASCII wire protocol (S,<speed>,<steer>\n)
    into UGV02 differential drive JSON velocity commands ({"T":13, "X": linear, "Z": angular}).
    """
    MAX_STEER_Z = 1.40  # rad/s angular deflection at 100% steer
    SPEED_MAP = {
        0: 0.0,    # STOP
        1: -0.20,  # SLOW (m/s)
        2: -0.35,  # FULL (m/s)
    }

    def __init__(self):
        self._is_open = True

    def write(self, data: bytes) -> int:
        """Decodes wire command and sends UGV02 velocity JSON."""
        if not self._is_open:
            return 0
        try:
            line = data.decode('ascii', errors='ignore').strip()
            if not line:
                return len(data)
            # Ignore version header V,1
            if line.startswith('V,'):
                return len(data)

            speed_tier, steer = decode_command(line + '\n')
            self.send_command(speed_tier, steer)
            return len(data)
        except Exception as e:
            print(f"[SERIAL ADAPTER ERROR] {e}")
            return len(data)

    def send_command(self, speed_tier: int, steer: int) -> None:
        global target_x, target_z
        linear_x = self.SPEED_MAP.get(speed_tier, 0.0)
        # Negative steer is left -> maps to positive Z in UGV02 kinematics
        angular_z = -(float(steer) / 100.0) * self.MAX_STEER_Z if linear_x != 0.0 else 0.0

        with motion_lock:
            target_x = linear_x
            target_z = angular_z

        cmd_json = f'{{"T":13,"X":{linear_x:.3f},"Z":{angular_z:.3f}}}'
        send_serial(cmd_json)

    def pivot(self, angular_z: float) -> None:
        """Rotate in place at `angular_z` rad/s, holding linear velocity at
        zero. Used only by the auto-mode obstacle-avoidance scan -- normal
        driving never needs a stationary pivot, which is why
        `send_command()` above deliberately zeroes Z whenever X is zero.
        """
        global target_x, target_z
        with motion_lock:
            target_x = 0.0
            target_z = angular_z
        send_serial(f'{{"T":13,"X":0.000,"Z":{angular_z:.3f}}}')

    def close(self) -> None:
        self._is_open = False
        with motion_lock:
            global target_x, target_z
            target_x = 0.0
            target_z = 0.0
        send_serial('{"T":13,"X":0.000,"Z":0.000}')


class PerceptionStateStore:
    """Stores latest rccar perception metadata for live HUD overlay rendering."""
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            'road_mask': None,
            'curb_line': None,
            'curb_side': None,
            'curb_offset_cm': None,
            'obstacles': [],
            'nearest_distance_cm': None,
            'speed_tier': None,
            'steer': None,
            'fps': 0.0,
            'last_update': 0.0
        }

    def update(self, meta: dict):
        with self.lock:
            now = time.time()
            dt = max(0.001, now - self.data['last_update']) if self.data['last_update'] > 0 else 0.03
            meta['fps'] = round(1.0 / dt, 1)
            meta['last_update'] = now
            self.data.update(meta)

    def get(self) -> dict:
        with self.lock:
            return dict(self.data)

    def clear(self):
        with self.lock:
            self.data = {
                'road_mask': None,
                'curb_line': None,
                'curb_side': None,
                'curb_offset_cm': None,
                'obstacles': [],
                'nearest_distance_cm': None,
                'speed_tier': None,
                'steer': None,
                'fps': 0.0,
                'last_update': 0.0
            }


perception_store = PerceptionStateStore()


def voltage_to_battery_pct(voltage):
    """
    Converts 3S Li-ion battery voltage (3x 18650 in series) to accurate percentage.
    Full: 12.6V (4.2V/cell), Nominal: ~11.1V (3.7V/cell), Empty/Cutoff: 9.8V (3.27V/cell)
    """
    curve = [
        (12.60, 100),
        (12.45, 95),
        (12.30, 90),
        (12.15, 80),
        (11.95, 70),
        (11.75, 60),
        (11.55, 50),
        (11.35, 40),
        (11.15, 30),
        (10.90, 20),
        (10.60, 10),
        (10.20, 5),
        (9.80, 0),
    ]
    if voltage >= curve[0][0]:
        return 100
    if voltage <= curve[-1][0]:
        return 0
    for i in range(len(curve) - 1):
        v_high, p_high = curve[i]
        v_low, p_low = curve[i + 1]
        if v_low <= voltage <= v_high:
            ratio = (voltage - v_low) / (v_high - v_low)
            pct = p_low + ratio * (p_high - p_low)
            return max(0, min(100, int(round(pct))))
    return 50


def process_serial_line(line):
    """Parses incoming JSON feedback packets from ESP32 driver board."""
    global telemetry_data, filtered_voltage
    if not line:
        return
    try:
        start = line.find('{')
        end = line.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return
        json_str = line[start:end + 1]
        data = json.loads(json_str)

        raw_v = None
        for key in ['v', 'voltage', 'volt', 'V', 'bat_v', 'bat_voltage', 'batV', 'V_bat']:
            if key in data:
                try:
                    raw_v = float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

        if raw_v is not None:
            # Normalize units: millivolts (e.g. 12600), centivolts (e.g. 1260), or volts (12.60).
            # A 3S pack's real voltage is 7-15V, so centivolt readings land in 700-1500 -
            # well above 1000 - and must not be mistaken for millivolts (7000-15000).
            if raw_v > 1500:
                v = raw_v / 1000.0
            elif raw_v > 50:
                v = raw_v / 100.0
            else:
                v = raw_v

            if 7.0 <= v <= 15.0:
                if filtered_voltage is None:
                    filtered_voltage = v
                else:
                    filtered_voltage = 0.20 * v + 0.80 * filtered_voltage

                telemetry_data['voltage'] = round(filtered_voltage, 2)
                telemetry_data['battery_pct'] = voltage_to_battery_pct(filtered_voltage)

        for key in ['bat', 'battery', 'battery_pct', 'soc', 'pct']:
            if key in data and raw_v is None:
                try:
                    pct = int(round(float(data[key])))
                    telemetry_data['battery_pct'] = max(0, min(100, pct))
                except (ValueError, TypeError):
                    pass

    except Exception:
        pass


def serial_reader_thread():
    """Continuously reads incoming telemetry lines from serial."""
    buffer = ""
    while True:
        if ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    raw = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += raw
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        process_serial_line(line.strip())
                else:
                    time.sleep(0.02)
            except Exception:
                time.sleep(0.1)
        else:
            time.sleep(0.5)


threading.Thread(target=serial_reader_thread, daemon=True).start()


def serial_telemetry_poller():
    """Periodically requests chassis feedback (T:130) to guarantee continuous telemetry."""
    while True:
        if ser and ser.is_open:
            send_serial('{"T":130}')
        time.sleep(0.5)


threading.Thread(target=serial_telemetry_poller, daemon=True).start()


def motion_controller_thread():
    """
    Continuous 30Hz velocity control loop for manual teleoperation and autonomous ramp.
    Smoothly ramps linear (X) and angular (Z) velocities towards target values,
    preventing motor jerking, wheel slip, and harsh camera shake.
    Also acts as a failsafe watchdog if network commands cease during manual teleop.
    """
    global curr_x, curr_z, target_x, target_z, current_cmd
    last_loop_time = time.time()
    last_serial_send_time = 0.0
    was_stopped = True

    while True:
        now = time.time()
        dt = max(0.005, min(0.1, now - last_loop_time))
        last_loop_time = now

        # Only enforce manual watchdog timeout if in manual mode
        if drive_mode == 'manual':
            with motion_lock:
                # Failsafe Watchdog: Auto-stop if no command received for > 0.5s
                if now - last_cmd_time > 0.5 and current_cmd != 's':
                    current_cmd = 's'
                    target_x = 0.0
                    target_z = 0.0

                # Smoothly ramp linear velocity X
                if curr_x < target_x:
                    accel = ACCEL_X if target_x >= 0 else DECEL_X
                    curr_x = min(target_x, curr_x + accel * dt)
                elif curr_x > target_x:
                    accel = DECEL_X if target_x >= 0 else ACCEL_X
                    curr_x = max(target_x, curr_x - accel * dt)

                # Smoothly ramp angular velocity Z
                if curr_z < target_z:
                    accel = ACCEL_Z if target_z >= 0 else DECEL_Z
                    curr_z = min(target_z, curr_z + accel * dt)
                elif curr_z > target_z:
                    accel = DECEL_Z if target_z >= 0 else ACCEL_Z
                    curr_z = max(target_z, curr_z - accel * dt)

                # Clean zero snap
                if abs(curr_x) < 0.01 and target_x == 0.0:
                    curr_x = 0.0
                if abs(curr_z) < 0.01 and target_z == 0.0:
                    curr_z = 0.0

                is_stopped = (curr_x == 0.0 and curr_z == 0.0)

            # Transmit continuous velocity updates when in motion or transitioning to stop
            needs_send = (not is_stopped) or (not was_stopped) or (now - last_serial_send_time > 0.1)
            if needs_send:
                cmd_json = f'{{"T":13,"X":{curr_x:.3f},"Z":{curr_z:.3f}}}'
                send_serial(cmd_json)
                last_serial_send_time = now

            was_stopped = is_stopped

        time.sleep(0.03)  # ~33Hz loop rate


threading.Thread(target=motion_controller_thread, daemon=True).start()


def update_telemetry_loop():
    """Background worker updating system storage, power, and continuous dead-reckoning positioning."""
    global telemetry_data
    last_time = time.time()
    while True:
        now = time.time()
        dt = max(0.05, min(0.5, now - last_time))
        last_time = now

        try:
            _, _, free = shutil.disk_usage("/")
            telemetry_data['free_storage_gb'] = round(free / (1024 ** 3), 2)
        except Exception:
            pass
        telemetry_data['speed_mode'] = current_speed_mode
        telemetry_data['drive_mode'] = drive_mode
        telemetry_data['rccar_available'] = RCCAR_AVAILABLE

        with motion_lock:
            # Forward motion is -curr_x, lateral turn rate is -curr_z
            telemetry_data['pos_y'] += (-curr_x) * dt
            telemetry_data['pos_x'] += (-curr_z * 0.15) * dt

            x, y = telemetry_data['pos_x'], telemetry_data['pos_y']
            telemetry_data['distance_home_m'] = round(math.sqrt(x**2 + y**2), 2)
            heading_rad = math.atan2(-x, -y) if (x != 0 or y != 0) else 0.0
            telemetry_data['heading_home_deg'] = round(math.degrees(heading_rad) % 360, 1)

        time.sleep(0.2)


threading.Thread(target=update_telemetry_loop, daemon=True).start()


RECORD_DIR = os.path.expanduser('~/videos')
os.makedirs(RECORD_DIR, exist_ok=True)


class CameraGrabber:
    """
    Dedicated single-owner frame grabber thread.
    Reads frames from physical camera device into a synchronized memory buffer,
    allowing simultaneous non-blocking access by Flask streaming and rccar CV.

    Captures via an `ffmpeg` v4l2 subprocess (rawvideo/bgr24) rather than
    cv2.VideoCapture: the opencv-python wheel available on this platform
    (armv7l, no prebuilt V4L2/GStreamer backend) cannot open the USB camera
    at all -- VideoCapture.isOpened() always returns False, silently
    falling back to a blank frame forever. ffmpeg's own v4l2 demuxer (built
    with --enable-libudev/v4l2 support) reads the device fine, so we shell
    out to it instead and only use cv2 for downstream image processing.
    """
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    def __init__(self, device_paths=('/dev/video0', '/dev/video1', '/dev/video2')):
        self.proc = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.stopped = False
        self._frame_size = self.FRAME_WIDTH * self.FRAME_HEIGHT * 3

        for path in device_paths:
            if not os.path.exists(path):
                continue
            # A device just released by a previous (e.g. crashed, or just
            # restarted) instance can stay EBUSY for several seconds while
            # the kernel/USB webcam finishes tearing down the prior
            # session -- retry for up to ~15s before giving up on an
            # otherwise-valid device path.
            for attempt in range(15):
                proc = self._spawn_ffmpeg(path)
                first_frame = self._read_frame(proc.stdout)
                if first_frame is not None:
                    self.proc = proc
                    self.latest_frame = first_frame
                    print(f"[CAMERA SUCCESS] Connected to {path} via ffmpeg")
                    break
                proc.kill()
                proc.wait()
                time.sleep(1.0)
            if self.proc is not None:
                break

        if self.proc is None:
            print("[CAMERA WARNING] No physical camera opened. Using test pattern generator.")
            # Blank frame for simulation
            self.latest_frame = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _spawn_ffmpeg(self, device_path):
        return subprocess.Popen(
            [
                'ffmpeg',
                '-f', 'v4l2',
                '-input_format', 'mjpeg',
                '-video_size', f'{self.FRAME_WIDTH}x{self.FRAME_HEIGHT}',
                '-i', device_path,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_size,
        )

    def _read_frame(self, stdout):
        data = stdout.read(self._frame_size)
        if data is None or len(data) != self._frame_size:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(self.FRAME_HEIGHT, self.FRAME_WIDTH, 3)

    def _update_loop(self):
        while not self.stopped:
            if self.proc is not None:
                frame = self._read_frame(self.proc.stdout)
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.02)
            else:
                time.sleep(0.03)

    def read(self) -> np.ndarray:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)

    def has_camera(self) -> bool:
        """True if reading from a real device; False if on the blank fallback.

        Obstacle detection is blind on the blank fallback frame -- callers
        (auto/obstacle-avoidance mode) must not drive on that signal.
        """
        return self.proc is not None

    def stop(self):
        self.stopped = True
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()


camera_grabber = CameraGrabber()
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
video_writer = None
writer_lock = threading.Lock()


class GrabberFrameSource:
    """FrameSource adapter delivering frames from CameraGrabber to rccar."""
    def __init__(self, grabber: CameraGrabber):
        self.grabber = grabber
        self._active = True

    def read(self):
        if not self._active or self.grabber is None:
            return None
        frame = self.grabber.read()
        if frame is not None:
            # Resize to standard 320x240 for high performance perception
            if frame.shape[1] != 320 or frame.shape[0] != 240:
                return cv2.resize(frame, (320, 240))
        return frame

    def is_live(self):
        return self._active

    def release(self):
        self._active = False


def start_auto_pipeline():
    """Launches the rccar autonomous perception and navigation pipeline in a background thread."""
    global auto_thread, auto_running, drive_mode, auto_stop_event, telemetry_data
    if not RCCAR_AVAILABLE:
        print("[AUTO WARNING] rccar module not found on system. Auto mode cannot be started.")
        return False

    if not camera_grabber.has_camera():
        print("[AUTO WARNING] No live camera feed (blank test-pattern frame). "
              "Refusing to start auto mode: obstacle detection would be blind.")
        return False

    if auto_running:
        return True

    auto_stop_event.clear()
    perception_store.clear()
    auto_running = True
    drive_mode = 'auto'
    telemetry_data['drive_mode'] = 'auto'

    def _worker():
        global auto_running, drive_mode, telemetry_data
        print("[AUTO] Starting rccar autonomous roadside driving pipeline...")

        source = None
        adapter = None
        try:
            homography_path = "config/homography.yaml"
            alt_paths = [
                "/home/pi/config/homography.yaml",
                os.path.join(os.path.dirname(__file__), "config", "homography.yaml"),
                "homography.yaml"
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    homography_path = p
                    break

            homography = load_homography(homography_path)
            adapter = UGV02SerialAdapter()
            watchdog = Watchdog(adapter)
            source = GrabberFrameSource(camera_grabber)

            def _state_cb(meta):
                perception_store.update(meta)
                with motion_lock:
                    telemetry_data['nearest_obs_cm'] = round(meta['nearest_distance_cm'], 1) if meta.get('nearest_distance_cm') is not None else None
                    telemetry_data['curb_side'] = meta.get('curb_side')
                    telemetry_data['curb_offset_cm'] = round(meta['curb_offset_cm'], 1) if meta.get('curb_offset_cm') is not None else None
                    speed_tier = meta.get('speed_tier')
                    telemetry_data['obstacle_speed_tier'] = speed_tier.name if speed_tier is not None else None
                    telemetry_data['obstacle_steer'] = meta.get('steer')

            # rccar.main.run_pipeline() has no state_callback/stop_event hooks
            # (it just runs until the source is exhausted and returns a batch
            # list), so drive process_frame() directly to get per-frame
            # telemetry and a way to stop a live camera source on demand.
            #
            # load_thresholds() with no path defaults to a config/thresholds.yaml
            # resolved relative to the *rccar package's own* checkout, not this
            # repo -- silently ignoring our config/thresholds.yaml. Resolve our
            # own path explicitly here (mirrors the homography_path lookup
            # above) so the tuned values actually take effect.
            thresholds_path = "config/thresholds.yaml"
            thresholds_alt_paths = [
                "/home/pi/config/thresholds.yaml",
                os.path.join(os.path.dirname(__file__), "config", "thresholds.yaml"),
                "thresholds.yaml"
            ]
            for p in thresholds_alt_paths:
                if os.path.exists(p):
                    thresholds_path = p
                    break
            stop_distance_cm, slow_distance_cm = load_thresholds(thresholds_path)
            state = PipelineState(
                classifier=AdaptiveClassifier(),
                curb_tracker=CurbConfidenceTracker(),
                homography=homography,
                speed_smoother=MajorityVoteSmoother(),
                steer_smoother=MajorityVoteSmoother(),
                stop_distance_cm=stop_distance_cm,
                slow_distance_cm=slow_distance_cm,
            )

            # Positive angular_z is a left turn, negative is right (matches
            # UGV02SerialAdapter.send_command's steer convention: negative
            # steer/left -> positive Z).
            _TURN_LEFT = SCAN_ANGULAR_RATE_RAD_S
            _TURN_RIGHT = -SCAN_ANGULAR_RATE_RAD_S

            def _scan_for_clear_path(blocked_result):
                """Pivot in SCAN_STEP_DEG increments, re-checking for a
                clear path after each, up to SCAN_MAX_DEG total. Returns
                the process_frame() result that found a clear path, or
                None if still blocked after the full scan (caller should
                treat that as a confirmed, final STOP).

                Direction: turn away from a tracked curb (curb behind/
                beside the car is the one obstacle we know isn't drivable
                regardless of what the scan sees). If no curb is tracked,
                there's nothing to derive a direction from, so start left;
                if the very first step makes the obstacle reading worse
                (closer than it was before turning), that guess was wrong
                -- switch to right for the remaining steps.
                """
                curb_side = blocked_result.get("curb_side")
                if curb_side == "left":
                    direction = _TURN_RIGHT
                elif curb_side == "right":
                    direction = _TURN_LEFT
                else:
                    direction = _TURN_LEFT

                pre_scan_distance_cm = blocked_result.get("obstacle_distance_cm")
                direction_confirmed = curb_side in ("left", "right")

                degrees_turned = 0
                while degrees_turned < SCAN_MAX_DEG and not auto_stop_event.is_set():
                    try:
                        adapter.pivot(direction)
                        time.sleep(SCAN_STEP_DURATION_S)
                    finally:
                        adapter.pivot(0.0)
                    degrees_turned += SCAN_STEP_DEG

                    frame = source.read()
                    if frame is None:
                        continue
                    watchdog.on_frame_received()
                    recheck = process_frame(frame, state)
                    _state_cb({
                        'nearest_distance_cm': recheck.get('obstacle_distance_cm'),
                        'curb_offset_cm': recheck.get('current_offset_cm'),
                        'curb_side': recheck.get('curb_side'),
                        'speed_tier': recheck.get('speed'),
                        'steer': recheck.get('steer'),
                    })
                    if recheck["speed"] != SpeedTier.STOP:
                        return recheck

                    if not direction_confirmed:
                        # First step of an undirected (no-curb) scan: check
                        # whether guessing left made things worse.
                        new_distance_cm = recheck.get("obstacle_distance_cm")
                        if (
                            pre_scan_distance_cm is not None
                            and new_distance_cm is not None
                            and new_distance_cm < pre_scan_distance_cm
                        ):
                            direction = _TURN_RIGHT
                        direction_confirmed = True

                return None

            frame_count = 0
            while not auto_stop_event.is_set():
                frame = source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                watchdog.on_frame_received()
                result = process_frame(frame, state)

                if result["speed"] == SpeedTier.STOP:
                    # Hold fully stopped before pivoting -- don't start a
                    # scan turn while still carrying forward momentum.
                    watchdog.write_command(SpeedTier.STOP, 0)
                    cleared = _scan_for_clear_path(result)
                    if cleared is not None:
                        # Path is clear at the new heading: drive straight
                        # out of the turn rather than continuing to steer,
                        # which would just carve a circle.
                        watchdog.write_command(cleared["speed"], 0)
                    else:
                        watchdog.write_command(SpeedTier.STOP, 0)
                    continue

                watchdog.write_command(result["speed"], result["steer"])

                _state_cb({
                    'nearest_distance_cm': result.get('obstacle_distance_cm'),
                    'curb_offset_cm': result.get('current_offset_cm'),
                    'curb_side': result.get('curb_side'),
                    'speed_tier': result.get('speed'),
                    'steer': result.get('steer'),
                })

                frame_count += 1
                if frame_count % 10 == 0:
                    watchdog.check_frame_staleness()

        except Exception as e:
            import traceback
            print(f"[AUTO PIPELINE ERROR] {e}")
            traceback.print_exc()
        finally:
            if source is not None:
                try:
                    source.release()
                except Exception:
                    pass
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass
            auto_running = False
            drive_mode = 'manual'
            telemetry_data['drive_mode'] = 'manual'
            with motion_lock:
                global target_x, target_z
                target_x = 0.0
                target_z = 0.0
            send_serial('{"T":13,"X":0.0,"Z":0.0}')
            print("[AUTO] Autonomous driving pipeline exited. Reverted to manual mode.")

    auto_thread = threading.Thread(target=_worker, daemon=True)
    auto_thread.start()
    return True


def stop_auto_pipeline():
    """Safely stops the rccar autonomous driving pipeline."""
    global auto_running, drive_mode, auto_stop_event, telemetry_data
    auto_stop_event.set()
    was_running = auto_running
    auto_running = False
    drive_mode = 'manual'
    telemetry_data['drive_mode'] = 'manual'
    with motion_lock:
        global target_x, target_z
        target_x = 0.0
        target_z = 0.0
    send_serial('{"T":13,"X":0.0,"Z":0.0}')
    if was_running:
        print("[AUTO] Manual override: stopped rccar pipeline.")


def gen_frames():
    global video_writer
    while True:
        frame = camera_grabber.read()
        if frame is None:
            time.sleep(0.03)
            continue

        # Render perception overlay during auto mode
        if drive_mode == 'auto' and RCCAR_AVAILABLE:
            meta = perception_store.get()
            h_f, w_f = frame.shape[:2]
            scale_x = w_f / 320.0
            scale_y = h_f / 240.0

            scaled_curb = None
            if meta.get('curb_line'):
                x1, y1, x2, y2 = meta['curb_line']
                scaled_curb = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

            scaled_obs = []
            for ox, oy, ow, oh in (meta.get('obstacles') or []):
                scaled_obs.append((int(ox * scale_x), int(oy * scale_y), int(ow * scale_x), int(oh * scale_y)))

            scaled_mask = None
            if meta.get('road_mask') is not None:
                scaled_mask = cv2.resize(meta['road_mask'], (w_f, h_f), interpolation=cv2.INTER_NEAREST)

            frame = draw_overlay(
                frame,
                scaled_mask,
                scaled_curb,
                meta.get('speed_tier'),
                meta.get('steer'),
                meta.get('curb_side') or 'unknown',
            )

        with writer_lock:
            if telemetry_data['is_recording'] and video_writer is not None:
                video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control', methods=['POST'])
def control():
    global last_cmd_time, current_cmd, target_x, target_z
    data = request.json or {}
    cmd = data.get('command', 's')

    # Manual keypress overrides auto mode immediately
    if drive_mode == 'auto' and cmd != 's':
        stop_auto_pipeline()

    with motion_lock:
        last_cmd_time = time.time()
        current_cmd = cmd
        targets = COMMAND_TARGETS.get(current_speed_mode, COMMAND_TARGETS['low'])
        if cmd in targets:
            target_x, target_z = targets[cmd]
        else:
            target_x, target_z = 0.0, 0.0

    return jsonify({'status': 'ok', 'command': cmd, 'drive_mode': drive_mode})


@app.route('/mode', methods=['POST'])
def handle_mode():
    data = request.json or {}
    requested_mode = data.get('mode', 'manual')
    if requested_mode == 'auto':
        success = start_auto_pipeline()
        if not success:
            return jsonify({'status': 'error', 'message': 'rccar pipeline unavailable on server', 'mode': drive_mode}), 400
    else:
        stop_auto_pipeline()
    return jsonify({'status': 'ok', 'mode': drive_mode})


@app.route('/telemetry', methods=['GET'])
def telemetry():
    return jsonify(telemetry_data)


@app.route('/speed', methods=['POST'])
def set_speed():
    global current_speed_mode, target_x, target_z
    data = request.json or {}
    mode = data.get('mode', 'low')
    if mode in COMMAND_TARGETS:
        with motion_lock:
            current_speed_mode = mode
            targets = COMMAND_TARGETS[current_speed_mode]
            if current_cmd in targets:
                target_x, target_z = targets[current_cmd]
    return jsonify({'status': 'ok', 'speed_mode': current_speed_mode})


@app.route('/recording', methods=['POST'])
def handle_recording():
    global video_writer
    data = request.json or {}
    action = data.get('action', 'stop')

    with writer_lock:
        if action == 'start' and not telemetry_data['is_recording']:
            filename = f"rec_{int(time.time())}.avi"
            filepath = os.path.join(RECORD_DIR, filename)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
            telemetry_data['is_recording'] = True
            telemetry_data['rec_file'] = filename
        elif action == 'stop' and telemetry_data['is_recording']:
            telemetry_data['is_recording'] = False
            if video_writer:
                video_writer.release()
                video_writer = None

    return jsonify({
        'status': 'ok',
        'is_recording': telemetry_data['is_recording'],
        'rec_file': telemetry_data['rec_file']
    })


@app.route('/recordings/list', methods=['GET'])
def list_recordings():
    files = []
    for filename in os.listdir(RECORD_DIR):
        if filename.endswith('.avi') or filename.endswith('.mp4'):
            filepath = os.path.join(RECORD_DIR, filename)
            files.append({
                'filename': filename,
                'mtime': os.path.getmtime(filepath),
                'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
            })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify(files)


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(RECORD_DIR, filename, as_attachment=True)


@app.route('/reset_origin', methods=['POST'])
def reset_origin():
    with motion_lock:
        telemetry_data['pos_x'] = 0.0
        telemetry_data['pos_y'] = 0.0
        telemetry_data['distance_home_m'] = 0.0
        telemetry_data['heading_home_deg'] = 0.0
    return jsonify({'status': 'ok'})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    stop_auto_pipeline()
    with motion_lock:
        target_x, target_z = 0.0, 0.0
    send_serial('{"T":0}')
    def _shutdown():
        time.sleep(1)
        os.system('sudo shutdown -h now')
    threading.Thread(target=_shutdown).start()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
