#!/usr/bin/env python3
"""T27 — full-pipeline FPS validation on Pi 3B/3B+ hardware.

Measures sustained FPS with the COMPLETE wired pipeline (capture -> segmentation
-> curb detection -> obstacles -> decision -> serial -> watchdog), not just the
stub workload from T10. T10 validated the core CV primitives; this validates that
the full decision logic + serial writes don't blow the budget.

All the new work in T27 vs. T10:
- detect_curb_side (Hough-line filtering + angle checks)
- CurbConfidenceTracker (state machine over last N frames)
- estimate_curb_offset_cm (a *second* detect_lines call + homography mapping)
- define_corridor + detect_obstacles (blob detection inside the corridor)
- nearest_obstacle_real_distance (homography distance math)
- decide_speed_tier, compute_steer, temporal smoothing (decision logic)
- encode_command + serial writes (I/O)
- Watchdog.on_frame_received / check_frame_staleness

Any of these could pull FPS from ~6.9 (T10 result) toward or below the 5 FPS
floor. T27 is the MANDATORY re-check before declaring the pipeline acceptable.

MUST be run on actual Raspberry Pi 3B/3B+ hardware, not a dev laptop. Uses a
mocked serial client (no real MCU I/O backpressure) so timing measures only the
Pi's processing, not Arduino latency.

Usage
-----
    # Requires a fixture video file (T6 output):
    python3 scripts/perf_smoke_test_full.py --source-file tests/fixtures/videos/clean_road.mp4

    # Optionally test live camera if one is available:
    python3 scripts/perf_smoke_test_full.py --source-file tests/fixtures/videos/clean_road.mp4 --device-index 0

At least one of --source-file / --device-index must be given.
"""

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rccar.capture.file import VideoFileSource  # noqa: E402
from rccar.capture.live import LiveCameraSource  # noqa: E402
from rccar.capture.source import FrameSource  # noqa: E402
from rccar.calibration.homography_api import load_homography  # noqa: E402
from rccar.main import run_pipeline  # noqa: E402
from rccar.serial_client.client import SerialClient  # noqa: E402
from rccar.watchdog.watchdog import Watchdog  # noqa: E402

MIN_FRAMES = 100
MIN_ACCEPTABLE_FPS = 5.0
TARGET_FPS_RANGE = (5.0, 10.0)


@dataclass
class RunResult:
    label: str
    frame_count: int
    total_seconds: float
    cold_start_seconds: float
    steady_state_seconds: float

    @property
    def sustained_fps(self) -> float:
        if self.steady_state_seconds <= 0:
            return 0.0
        return (self.frame_count - 1) / self.steady_state_seconds


class FakeSerialClient:
    """Mock serial client for testing — records writes but doesn't touch hardware."""

    def __init__(self):
        self.writes = []
        self.is_open = True

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def close(self) -> None:
        self.is_open = False


def run_pipeline_timed(
    source: FrameSource,
    homography_path: str = "config/homography.yaml",
    min_frames: int = MIN_FRAMES,
) -> RunResult:
    """Run the full pipeline and measure FPS."""
    homography = load_homography(homography_path)
    serial_client = FakeSerialClient()
    watchdog = Watchdog(serial_client, frame_timeout_ms=500)

    frame_count = 0
    cold_start_seconds = 0.0
    steady_state_start: Optional[float] = None
    run_start = time.perf_counter()

    try:
        # run_pipeline returns a list of result dicts; we just time the loop
        results = run_pipeline(source, serial_client, homography, watchdog, max_frames=min_frames)
        frame_count = len(results)
    finally:
        source.release()
        serial_client.close()

    run_end = time.perf_counter()
    total_seconds = run_end - run_start

    # Rough approximation: assume first frame is ~cold-start, rest steady-state
    # (run_pipeline doesn't expose per-frame timings, so this is coarse)
    if frame_count > 0:
        cold_start_seconds = total_seconds / frame_count  # avg per-frame as rough proxy
        steady_state_start = run_start + cold_start_seconds
        steady_state_seconds = run_end - steady_state_start
    else:
        steady_state_seconds = total_seconds

    return RunResult(
        label="full-pipeline",
        frame_count=frame_count,
        total_seconds=total_seconds,
        cold_start_seconds=cold_start_seconds,
        steady_state_seconds=steady_state_seconds,
    )


def print_result(result: RunResult) -> None:
    print(f"\n--- {result.label} ---")
    print(f"frames processed:        {result.frame_count}")
    print(f"total duration:          {result.total_seconds:.2f} s")
    print(f"sustained FPS:           {result.sustained_fps:.2f}")
    if result.frame_count < MIN_FRAMES:
        print(
            f"WARNING: only {result.frame_count} frames were available "
            f"(< {MIN_FRAMES} requested) — FPS number is less reliable, "
            f"use a longer fixture clip."
        )
    verdict = "PASS" if result.sustained_fps >= MIN_ACCEPTABLE_FPS else "FAIL"
    print(f"verdict:                 {verdict} (gate: >= {MIN_ACCEPTABLE_FPS} FPS, target {TARGET_FPS_RANGE[0]}-{TARGET_FPS_RANGE[1]} FPS)")


def hardware_info() -> str:
    lines = [
        f"platform: {platform.platform()}",
        f"machine:  {platform.machine()}",
        f"python:   {platform.python_version()}",
    ]
    try:
        model = Path("/proc/device-tree/model")
        if model.exists():
            lines.append(f"pi model: {model.read_text().strip(chr(0))}")
    except OSError:
        pass
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-file", type=str, default=None, help="Path to a fixture video file.")
    parser.add_argument("--device-index", type=int, default=None, help="Live camera device index (e.g. 0).")
    parser.add_argument(
        "--homography",
        type=str,
        default="config/homography.yaml",
        help="Path to the homography config file (default: config/homography.yaml).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=MIN_FRAMES,
        help=f"Frames to sustain (default {MIN_FRAMES}).",
    )
    args = parser.parse_args()

    if args.source_file is None and args.device_index is None:
        parser.error("at least one of --source-file or --device-index is required")

    print("=== T27 full-pipeline FPS validation (complete pipeline, not stub) ===")
    print(hardware_info())

    results = []

    if args.source_file is not None:
        source = VideoFileSource(args.source_file)
        result = run_pipeline_timed(source, homography_path=args.homography, min_frames=args.min_frames)
        print_result(result)
        results.append(result)

    if args.device_index is not None:
        source = LiveCameraSource(args.device_index)
        result = run_pipeline_timed(source, homography_path=args.homography, min_frames=args.min_frames)
        print_result(result)
        results.append(result)

    overall_pass = all(r.sustained_fps >= MIN_ACCEPTABLE_FPS for r in results)

    print("\n=== Summary ===")
    for r in results:
        print(f"{r.label}: {r.sustained_fps:.2f} FPS ({'PASS' if r.sustained_fps >= MIN_ACCEPTABLE_FPS else 'FAIL'})")
    print(
        f"\nOverall: {'PASS' if overall_pass else 'FAIL'} "
        f"(target range {TARGET_FPS_RANGE[0]}-{TARGET_FPS_RANGE[1]} FPS, hard floor {MIN_ACCEPTABLE_FPS} FPS)"
    )
    print(
        "\nNext step: copy these numbers, the hardware_info() block above, and "
        "the verdict into docs/perf_results_pi3b_full_pipeline.md. If FAIL, document "
        "what was cut/optimized (e.g. drop redundant detect_lines, widen histogram-rebuild "
        "interval, tighten Hough ROI) and re-measure before declaring the pipeline acceptable."
    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
