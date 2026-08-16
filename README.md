# RC_Car_Movement

# UGV02 Teleoperation & Autonomous Movement System

## Project Overview & Goal
This project establishes a low-latency network control architecture and computer vision pipeline for an autonomous ground vehicle (UGV). It combines a high-framerate teleoperation dashboard with real-time video streaming, telemetry, and an integrated monocular autonomous roadside navigation system (`rccar`) for curb-following, corridor maintenance, and obstacle avoidance.

### Hardware Architecture
* **Robotic Base:** UGV02 Differential Drive Chassis connected via USB-to-Serial (`/dev/ttyUSB0` or `/dev/serial0` at 115200 baud).
* **Onboard Computer:** Raspberry Pi running Linux, managing hardware communication, telemetry generation, multi-consumer video capture, and the `rccar` perception pipeline.
* **Camera:** Single Front-Facing Monocular USB/Pi Camera module (2D RGB feed).
* **Control Station:** Remote laptop running a Pygame Graphical User Interface (HUD) over Wi-Fi.

---

## Architecture & Code Specifics

### 1. `pi_server.py` (Raspberry Pi Backend Server)
A multi-threaded Flask application running on the Raspberry Pi that acts as the hardware abstraction layer, REST API interface, and autonomous perception host.

* **Autonomous Roadside Navigation (`start_auto_pipeline` / `POST /mode`):** Integrates the `rccar` pipeline from `RCC_KrithikMohan`. Uses adaptive HSV color road segmentation, Hough line slope-filtered curb detection (`detect_curb_side`), confidence tracking (`CurbConfidenceTracker`), drivable corridor definition (`define_corridor`), obstacle blob extraction (`detect_obstacles`), ground-plane metric distance projection (`homography.yaml`), and 3-tier speed selection with proportional curb-offset steering.
* **Actuation Bridge (`UGV02SerialAdapter`):** Bridges `rccar` speed tiers (`0=STOP`, `1=SLOW`, `2=FULL`) and steering percentages (`-100..100`) to UGV02 differential drive JSON velocity payloads `{"T":13, "X": linear_m_s, "Z": angular_rad_s}` over the shared serial bus.
* **Thread-Safe Camera Grabber (`CameraGrabber`):** A single-owner background thread continuously pulls frames from the physical camera and serves in-memory clones to both the Flask MJPEG streamer (`/video_feed`) and the `rccar` perception pipeline without V4L2 lock contention or frame drops.
* **Perception HUD Overlay (`rccar.viz.overlay`):** During auto mode, renders live semi-transparent road masks, detected curb lines, obstacle bounding boxes, and ground-distance metrics directly onto `/video_feed`.
* **Hardware Watchdog Failsafe:** Monitors frame arrival intervals (`frame_timeout_ms=500`) and command frequency. Automatically zeros motor velocities on signal drops or stalls.
* **Telemetry & Dead Reckoning (`/telemetry`):** Tracks real-time position coordinates ($X, Y$), scalar distance from home, return heading angle, battery status, nearest obstacle distance, active curb side/offset, and available micro-SD storage.
* **Video Streaming & Recording (`/video_feed`, `/recording`, `/download`):** Streams continuous MJPEG video frames with local hardware MP4/AVI recording and asynchronous offloading.

### 2. `car_movement.py` (Laptop HUD & Controller)
The client-side interface running on the laptop, combining a Pygame HUD overlay with teleoperation controls and perception status feedback.

* **Pygame HUD & Teleoperation:** Displays the live camera feed overlaid with real-time telemetry metrics, speed toggles, recording buttons, active drive state indicators, and visual WASD key indicators.
* **Autonomous Mode Controls:**
  * **Toggle Button:** Interactive `AUTO: ON` / `AUTO: OFF` button on the top HUD toolbar.
  * **Hotkeys:** Press `M` to toggle between Manual and Autonomous modes; press `SPACE` for immediate emergency stop.
  * **Instant Override:** Pressing any manual key (`W`, `A`, `S`, `D`) or clicking directional buttons instantly halts auto mode and restores manual driver control.

---

## Configuration (`config/`)

All tunable parameters live in `config/*.yaml`:
* `config/homography.yaml`: 3x3 camera pixel $\rightarrow$ ground-plane cm transformation matrix.
* `config/curb.yaml`: Confidence window size, min confidence threshold, and horizontal angle rejection threshold.
* `config/steer.yaml`: Target lateral curb offset in cm, proportional gain ($P$), and max steer clamp.
* `config/thresholds.yaml`: `stop_distance_cm` (default 30cm) and `slow_distance_cm` (default 100cm).
* `config/watchdog.yaml`: Heartbeat staleness timeouts before emergency stop.
* `config/roi.yaml`: Near-field trapezoid pixel coordinates for road color sampling.
* `config/smoothing.yaml`: Majority-vote temporal smoothing window.

---

## Feature Progress & Status Matrix

| Feature | Status | Completion | Notes |
| :--- | :--- | :--- | :--- |
| **Flask REST Server & Serial Relay** | Operational | `[██████████] 100%` | Hardware serial communications and API endpoints stable. |
| **Pygame Control HUD & Teleoperation** | Operational | `[██████████] 100%` | Full WASD driving, GUI buttons, and instant override working. |
| **Video Streaming & Local MP4 Transfer** | Operational | `[██████████] 100%` | Thread-safe MJPEG streaming and asynchronous file offloading verified. |
| **Watchdog Failsafe & Dead-Reckoning** | Operational | `[██████████] 100%` | Automatic motor cut-off on signal drop and telemetry active. |
| **Outdoor Auto Navigation (Curb Follow & Obstacles)** | Operational | `[██████████] 100%` | Full `rccar` pipeline integrated: curb tracking, corridor maintenance, obstacle braking, and HUD overlays. |
| **Single-Camera Indoor Auto Navigation** | Deprecated | `[██████████] 100%` | Superseded by the unified `rccar` corridor vision and obstacle pipeline. |
