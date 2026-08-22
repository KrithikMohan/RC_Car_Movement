# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & commands

This repo depends on a sibling repo, `rccar` (package `RCC_KrithikMohan`), installed in editable mode — it must exist at `../RCC_KrithikMohan` relative to this directory for imports of `rccar.*` to resolve.

```bash
pip install -r requirements.txt   # installs ../RCC_KrithikMohan as editable dependency
pytest                             # run full test suite
pytest tests/test_serial_adapter.py -q          # single file
pytest tests/test_api_endpoints.py::test_mode_and_manual_override  # single test
python pi_server.py               # run the Pi-side Flask server (binds 0.0.0.0:5000)
python car_movement.py            # run the laptop Pygame HUD/teleop client
```

There is no lint/format tooling configured in this repo.

## Architecture

Two processes talk over HTTP/MJPEG: `pi_server.py` runs on the Raspberry Pi attached to the UGV02 chassis and camera; `car_movement.py` runs on a laptop as the operator HUD. Serial control of the chassis (`ugv_command.py` / `UGV02SerialAdapter`) and the `rccar` perception pipeline both run inside `pi_server.py`.

### `pi_server.py` (Pi backend — hardware + perception + REST API)

- **`CameraGrabber`** is the single owner of the physical camera (V4L2). A background thread continuously grabs frames and hands clones to consumers, avoiding lock contention between the Flask MJPEG stream and the `rccar` pipeline. `GrabberFrameSource` adapts it to the `rccar.capture.source.FrameSource` interface.
- **Autonomous pipeline** (`start_auto_pipeline` / `stop_auto_pipeline`, triggered via `POST /mode`): runs `rccar`'s `run_pipeline` (imported from `RCC_KrithikMohan`) in its own thread — adaptive HSV road segmentation, Hough-line curb detection with `CurbConfidenceTracker`, corridor definition, obstacle blob extraction, homography-based ground-distance projection, and 3-tier speed + proportional steering decisions. `rccar.viz.overlay` draws perception state onto `/video_feed` frames during auto mode.
- **`UGV02SerialAdapter`** bridges `rccar`'s `speed,steer` output (speed tiers `0=STOP/1=SLOW/2=FULL`, steer `-100..100`) into UGV02 JSON velocity commands `{"T":13,"X":<m/s>,"Z":<rad/s>}`, sent through the shared `send_serial` mutex-guarded serial write.
- **`PerceptionStateStore`** holds the latest perception metadata (curb side/offset, obstacle distance, etc.) behind a lock, shared between the pipeline thread and the `/telemetry` endpoint.
- **Watchdog**: `motion_controller_thread` / frame-timeout logic zeroes motor velocity if frames stop arriving (`frame_timeout_ms`) or commands go stale — tunable via `config/watchdog.yaml`.
- Threads to know about: `serial_reader_thread`, `serial_telemetry_poller`, `motion_controller_thread`, `update_telemetry_loop`, plus the `CameraGrabber` thread and (when auto mode is active) the `rccar` pipeline thread. All motor/serial writes funnel through the same serial port, so new features that touch the chassis must go through `send_serial` / `UGV02SerialAdapter`, not open the port directly.
- REST surface: `/video_feed` (MJPEG), `/control` (manual drive commands — immediately preempts auto mode), `/mode` (manual/auto toggle), `/telemetry`, `/speed`, `/recording`, `/recordings/list`, `/download/<filename>`, `/reset_origin`, `/shutdown`.

### `car_movement.py` (laptop HUD/teleop client)

Pygame client that renders the `/video_feed` stream plus telemetry overlay and sends `/control` / `/mode` / `/speed` requests. Manual keys (`W/A/S/D`) or HUD button clicks instantly override auto mode; `M` toggles auto/manual, `SPACE` is emergency stop.

### `config/*.yaml`

All perception/control tuning lives here, loaded by `pi_server.py` / `rccar` at startup — not hardcoded:
- `homography.yaml`: camera-pixel → ground-plane cm transform.
- `curb.yaml`: confidence window size/threshold, angle rejection.
- `steer.yaml`: target curb offset, proportional gain, max steer clamp.
- `thresholds.yaml`: `stop_distance_cm` / `slow_distance_cm`.
- `watchdog.yaml`: staleness timeouts before e-stop.
- `roi.yaml`: near-field trapezoid for road color sampling.
- `smoothing.yaml`: majority-vote temporal smoothing window.

### Tests

`tests/` exercises this repo's glue code against the real `rccar` package (not mocked out): `test_api_endpoints.py` drives the Flask app via `app.test_client()`, `test_serial_adapter.py` and `test_camera_grabber.py` unit-test the adapter/grabber in isolation, and `test_integration_autodrive.py` runs `rccar.main.run_pipeline` end-to-end against a synthetic `FrameSource` and asserts on the resulting serial commands. When changing the `rccar` protocol (speed/steer encoding) or the adapter, check both sides — this repo's adapter and `../RCC_KrithikMohan`'s pipeline output format must agree.
