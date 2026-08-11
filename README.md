# RC_Car_Movement

# UGV02 Teleoperation & Autonomous Movement System

## Project Overview & Goal
This project establishes a low-latency network control architecture and computer vision pipeline for an autonomous ground vehicle (UGV). The primary goal is to build a full-featured teleoperation dashboard with real-time video streaming and telemetry, alongside exploring single-camera autonomous navigation algorithms for indoor room exploration and outdoor curb tracking.

### Hardware Architecture
* **Robotic Base:** UGV02 Differential Drive Chassis connected via USB-to-Serial (`/dev/ttyUSB0` at 115200 baud).
* **Onboard Computer:** Raspberry Pi running Linux, managing hardware communication, telemetry generation, and video streaming.
* **Camera:** Single Front-Facing Monocular USB/Pi Camera module (2D RGB feed).
* **Control Station:** Remote laptop running a Pygame Graphical User Interface (HUD) over Wi-Fi.

---

## Architecture & Code Specifics

### 1. `pi_server.py` (Raspberry Pi Backend Server)
A multi-threaded Flask application running on the Raspberry Pi that acts as the hardware abstraction layer and REST API interface.

* **Serial Motor Control (`/control`):** Relays directional drive commands (`f`, `b`, `l`, `r`, `s`) to the chassis as JSON motor payloads containing left (`L`) and right (`R`) wheel power values. Supports toggling between `low` and `high` speed profiles (`/speed`).
* **Hardware Watchdog Failsafe:** A background watchdog thread monitors incoming command timestamps. If network latency or dropped packets exceed `500ms`, the server automatically cuts motor power to prevent runaway collisions.
* **Telemetry & Dead Reckoning (`/telemetry`):** Tracks real-time position coordinates ($X, Y$), scalar distance from home, return heading angle, battery status, and available micro-SD storage.
* **Video Streaming & Recording (`/video_feed`, `/recording`, `/download`):** Streams continuous MJPEG video frames via OpenCV. Supports local hardware MP4 video recording on the Pi with asynchronous HTTP file offloading to the laptop control station.
* **System Utilities (`/reset_origin`, `/shutdown`):** Resets origin coordinates to zero or triggers a safe Linux shutdown command on the Pi via remote POST requests.

### 2. `car_movement.py` (Laptop HUD & Controller)
The client-side interface running on the laptop, combining a Pygame HUD overlay with openCV computer vision processing.

* **Pygame HUD & Teleoperation:** Displays the live camera feed overlaid with real-time telemetry metrics, speed toggles, recording buttons, active drive state indicators, and visual WASD key indicators. Manual key presses immediately override any active autonomous mode.
* **Asynchronous Networking:** Offloads command dispatches, telemetry polling, and video file downloads to background daemon threads to prevent UI lockup.
* **Indoor Vision & State Machine (`process_indoor_auto`):** Evaluates ground clearance using adaptive thresholding, Gaussian blurring, morphological closing, and contour area filtering. Executes a 4-state recovery state machine (`FORWARD`, `RECOVER_BACK`, `RECOVER_TURN`, `FORWARD_CLEAR`) alongside odometry stall detection if physical movement stops while motors are driven.

---

## Feature Progress & Status Matrix

| Feature | Status | Completion | Notes |
| :--- | :--- | :--- | :--- |
| **Flask REST Server & Serial Relay** | Operational | `[██████████] 100%` | Hardware serial communications and API endpoints stable. |
| **Pygame Control HUD & Teleoperation** | Operational | `[██████████] 100%` | Full WASD driving, GUI buttons, and instant override working. |
| **Video Streaming & Local MP4 Transfer** | Operational | `[██████████] 100%` | MJPEG streaming and asynchronous file offloading verified. |
| **Watchdog Failsafe & Dead-Reckoning** | Operational | `[██████████] 100%` | Automatic motor cut-off on signal drop and telemetry active. |
| **Single-Camera Indoor Auto Navigation** | Unreliable | `[██░░░░░░░░] 20%` | **Failed.** 2D monocular vision lacks true depth perception; lighting shifts and featureless walls cause loop behavior without distance sensors (Ultrasonic/ToF). |
| **Outdoor Auto Navigation (Curb Follow)** | Planned | `[░░░░░░░░░░] 0%` | Algorithm structured in code; physical field testing and curb contour tuning not yet started. |
