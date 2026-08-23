#!/usr/bin/env python3
"""T10 — on-hardware FPS validation (early risk-mitigation gate).

Runs capture -> T8 segmentation (adaptive road/non-road classify) -> T9 Hough
curb stub -> a no-op decision step, in a loop, and measures sustained FPS.
This is deliberately the same stub-level pipeline T10 is scoped to test, not
the full T25 pipeline (that's T27's job, later).

MUST be run on the actual target hardware (Raspberry Pi 3B/3B+), not a dev
laptop -- ARM perf characteristics differ enough that a laptop number is not
a substitute. See docs/perf_results_pi3b.md for where to record the result.

Usage
-----
    # Against a fixture video file (required):
    python3 scripts/perf_smoke_test.py --source-file tests/fixtures/videos/clean_road.mp4

    # Also test the live camera, if one is attached to the test rig:
    python3 scripts/perf_smoke_test.py --source-file tests/fixtures/videos/clean_road.mp4 --device-index 0

    # Just live camera:
    python3 scripts/perf_smoke_test.py --device-index 0

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
from rccar.curb.hough_stub import detect_lines  # noqa: E402
from rccar.segmentation.classify import (  # noqa: E402
    AdaptiveClassifier,
    should_rebuild_model,
)

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
        # steady-state count excludes the first frame (cold start)
        return (self.frame_count - 1) / self.steady_state_seconds


def no_op_decision(mask, lines) -> None:
    """Placeholder decision step: touch the outputs so nothing is optimized
    away, but do no real speed/steer logic (that's T18-T20's job later)."""
    _ = mask.sum() if mask is not None else 0
    _ = len(lines)


def run_pipeline(source: FrameSource, label: str, min_frames: int = MIN_FRAMES) -> RunResult:
    classifier = AdaptiveClassifier(rebuild_interval_k=30)
    frame_count = 0
    cold_start_seconds = 0.0
    steady_state_start: Optional[float] = None
    run_start = time.perf_counter()

    while frame_count < min_frames:
        t0 = time.perf_counter()
        frame = source.read()
        if frame is None:
            break

        mask = classifier.process_frame(frame)
        lines = detect_lines(frame)
        no_op_decision(mask, lines)

        t1 = time.perf_counter()

        if frame_count == 0:
            cold_start_seconds = t1 - t0
            steady_state_start = t1
        frame_count += 1

    run_end = time.perf_counter()
    total_seconds = run_end - run_start
    steady_state_seconds = (
        run_end - steady_state_start if steady_state_start is not None else 0.0
    )

    return RunResult(
        label=label,
        frame_count=frame_count,
        total_seconds=total_seconds,
        cold_start_seconds=cold_start_seconds,
        steady_state_seconds=steady_state_seconds,
    )


def print_result(result: RunResult) -> None:
    print(f"\n--- {result.label} ---")
    print(f"frames processed:        {result.frame_count}")
    print(f"cold-start (1st frame):  {result.cold_start_seconds * 1000:.1f} ms")
    print(f"steady-state duration:   {result.steady_state_seconds:.2f} s")
    print(f"sustained FPS:           {result.sustained_fps:.2f}")
    if result.frame_count < MIN_FRAMES:
        print(
            f"WARNING: only {result.frame_count} frames were available "
            f"(< {MIN_FRAMES} requested) -- FPS number is less reliable, "
            f"use a longer fixture clip."
        )
    verdict = "PASS" if result.sustained_fps >= MIN_ACCEPTABLE_FPS else "FAIL"
    print(f"verdict:                 {verdict} (gate: >= {MIN_ACCEPTABLE_FPS} FPS)")


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-file", type=str, default=None, help="Path to a fixture video file.")
    parser.add_argument("--device-index", type=int, default=None, help="Live camera device index (e.g. 0).")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES, help=f"Frames to sustain (default {MIN_FRAMES}).")
    args = parser.parse_args()

    if args.source_file is None and args.device_index is None:
        parser.error("at least one of --source-file or --device-index is required")

    print("=== T10 on-hardware FPS validation ===")
    print(hardware_info())

    results = []

    if args.source_file is not None:
        source = VideoFileSource(args.source_file)
        try:
            result = run_pipeline(source, label=f"file: {args.source_file}", min_frames=args.min_frames)
        finally:
            source.release()
        print_result(result)
        results.append(result)

    if args.device_index is not None:
        source = LiveCameraSource(args.device_index)
        try:
            result = run_pipeline(source, label=f"live: device {args.device_index}", min_frames=args.min_frames)
        finally:
            source.release()
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
        "the verdict into docs/perf_results_pi3b.md. If FAIL, do not proceed to "
        "curb-side-detection/decision/serial/watchdog/viz work until a "
        "mitigation (frame skip, smaller ROI-only Hough, less frequent "
        "histogram rebuild) is applied and re-measured."
    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
