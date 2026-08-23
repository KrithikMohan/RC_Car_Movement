"""Main pipeline wiring (T25): capture -> segmentation/curb -> obstacles ->
decision -> serial -> watchdog, glued into one runnable loop with a CLI to
select a live camera or a recorded video file as the frame source.

Pipeline stages per frame (see :func:`process_frame`):

1. **Capture** -- a frame is pulled from a :class:`~rccar.capture.source.FrameSource`
   (either :class:`~rccar.capture.live.LiveCameraSource` or
   :class:`~rccar.capture.file.VideoFileSource`).
2. **Segmentation** -- :class:`~rccar.segmentation.classify.AdaptiveClassifier`
   turns the frame into a road/non-road mask.
3. **Curb detection** -- :func:`~rccar.curb.detect.detect_curb_side` finds the
   per-frame curb side/confidence, fed into a
   :class:`~rccar.curb.confidence.CurbConfidenceTracker` for temporal
   stability (tracking vs. fallback).
4. **Obstacle detection** -- the road mask and curb side define a drivable
   corridor (:func:`~rccar.obstacles.detect.define_corridor`), inside which
   non-road blobs are found (:func:`~rccar.obstacles.detect.detect_obstacles`)
   and mapped to a real-world nearest-obstacle distance
   (:func:`~rccar.obstacles.distance.nearest_obstacle_real_distance`).
5. **Decision** -- distance drives a speed tier
   (:func:`~rccar.decision.speed.decide_speed_tier`), curb side/offset drives
   a steer value (:func:`~rccar.decision.steer.compute_steer`), and both are
   smoothed over time via :class:`~rccar.decision.smoothing.MajorityVoteSmoother`.
6. **Serial + watchdog** -- the decision is encoded and sent over the serial
   link via :class:`~rccar.watchdog.watchdog.Watchdog`, which also guards
   against stale frames and serial write failures by forcing a STOP.

Curb lateral offset derivation
-------------------------------
No earlier task produced a dedicated "curb lateral offset in cm" function,
but :func:`~rccar.decision.steer.compute_steer` needs a ``current_offset_cm``
input. This module derives it (see :func:`estimate_curb_offset_cm`) as an
integration-level decision:

1. Re-run :func:`~rccar.curb.hough_stub.detect_lines` to get the raw Hough
   line segments (the same primitive :func:`~rccar.curb.detect.detect_curb_side`
   itself is built on).
2. Filter out near-horizontal lines and pick, among the candidates on the
   tracked curb's side of the frame, the *longest* one -- mirroring
   ``detect_curb_side``'s own "length as a confidence proxy" heuristic, so
   the line picked here is (in the common case) the same one that produced
   the tracked side.
3. Take that line's pixel midpoint ``(mid_x, mid_y)`` and map it through
   :func:`~rccar.calibration.homography_api.image_point_to_ground` to get a
   ground-plane ``(x, y)`` in cm.
4. Treat ``abs(x)`` -- the ground-plane point's distance from the car's
   centerline (x=0 in the calibrated ground frame) -- as ``current_offset_cm``.

If ``image_point_to_ground`` returns ``None`` (point above the horizon) or no
curb line is available on the tracked side, this falls back to
``current_offset_cm=None``, which ``compute_steer`` already handles (holds
the last direction straight rather than dividing by a missing measurement).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from rccar.calibration.homography_api import image_point_to_ground, load_homography
from rccar.capture.file import VideoFileSource
from rccar.capture.live import LiveCameraSource
from rccar.capture.source import FrameSource
from rccar.curb.confidence import CurbConfidenceTracker
from rccar.curb.detect import (
    HORIZONTAL_ANGLE_THRESHOLD_DEG,
    _is_near_horizontal,
    _line_length,
    _line_midpoint_x,
    detect_curb_side,
)
from rccar.curb.hough_stub import Line, detect_lines
from rccar.decision.smoothing import MajorityVoteSmoother
from rccar.decision.speed import decide_speed_tier, load_thresholds
from rccar.serial_client.protocol import SpeedTier
from rccar.decision.steer import compute_steer
from rccar.obstacles.detect import define_corridor, detect_obstacles
from rccar.obstacles.distance import nearest_obstacle_real_distance
from rccar.segmentation.classify import AdaptiveClassifier
from rccar.segmentation.roi import load_roi_points, load_roi_reference_frame_size
from rccar.serial_client.client import SerialClient, SerialClientError
from rccar.watchdog.watchdog import Watchdog

# How often (in frames) the main loop polls the watchdog's staleness check.
# on_frame_received() is called every frame regardless -- this just governs
# how often we proactively ask "has it actually been too long?" in between.
_STALENESS_CHECK_INTERVAL = 10

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SERIAL_CONFIG_PATH = _REPO_ROOT / "config" / "serial.yaml"

_DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
_DEFAULT_BAUD = 9600
_DEFAULT_SERIAL_TIMEOUT = 1.0
_DEFAULT_HOMOGRAPHY_PATH = "config/homography.yaml"


def _load_serial_defaults(config_path: Optional[Path] = None) -> Tuple[str, int, float]:
    """Best-effort load of (port, baud, timeout) from config/serial.yaml.

    Falls back to hardcoded defaults if the file is missing/malformed, so
    the CLI remains usable (with explicit --serial-port/--baud overrides)
    even outside a full checkout -- mirrors the pattern used throughout the
    rest of the codebase (e.g. rccar.curb.confidence._load_defaults).
    """
    path = config_path if config_path is not None else _SERIAL_CONFIG_PATH
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return (
            str(cfg.get("port", _DEFAULT_SERIAL_PORT)),
            int(cfg.get("baud", _DEFAULT_BAUD)),
            float(cfg.get("timeout", _DEFAULT_SERIAL_TIMEOUT)),
        )
    except (OSError, ValueError, yaml.YAMLError):
        return _DEFAULT_SERIAL_PORT, _DEFAULT_BAUD, _DEFAULT_SERIAL_TIMEOUT


def _line_midpoint(line: Line) -> Tuple[float, float]:
    x1, y1, x2, y2 = line
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def estimate_curb_offset_cm(
    frame: np.ndarray,
    curb_side: Optional[str],
    homography: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Derive ``(curb_x_px, current_offset_cm)`` for the tracked curb line.

    See the module docstring's "Curb lateral offset derivation" section for
    the full reasoning. Returns ``(None, None)`` if ``curb_side`` isn't
    ``"left"``/``"right"``, if no suitable Hough line survives filtering on
    that side, or (for the offset only -- ``curb_x_px`` may still be
    returned) if the chosen line's midpoint maps above the horizon.
    """
    if curb_side not in ("left", "right"):
        return None, None

    lines: List[Line] = detect_lines(frame)
    candidates = [
        line for line in lines if not _is_near_horizontal(line, HORIZONTAL_ANGLE_THRESHOLD_DEG)
    ]
    if not candidates:
        return None, None

    width = frame.shape[1]
    center_x = width / 2.0
    if curb_side == "left":
        side_candidates = [c for c in candidates if _line_midpoint_x(c) < center_x]
    else:
        side_candidates = [c for c in candidates if _line_midpoint_x(c) >= center_x]

    if not side_candidates:
        return None, None

    best_line = max(side_candidates, key=_line_length)
    mid_x, mid_y = _line_midpoint(best_line)

    ground = image_point_to_ground(homography, (mid_x, mid_y))
    if ground is None:
        return mid_x, None

    ground_x, _ground_y = ground
    return mid_x, abs(ground_x)


@dataclass
class PipelineState:
    """Mutable per-run pipeline state threaded through :func:`process_frame`.

    Bundles everything :func:`process_frame` needs beyond the current frame
    itself, so that function stays pure with respect to its explicit inputs
    (no hidden module-level globals) -- required for T5/T28's live-vs-file
    parity: given the same frames and the same starting state, process_frame
    must behave identically regardless of which FrameSource produced them.
    """

    classifier: AdaptiveClassifier
    curb_tracker: CurbConfidenceTracker
    homography: np.ndarray
    speed_smoother: MajorityVoteSmoother
    steer_smoother: Optional[MajorityVoteSmoother] = None
    stop_distance_cm: float = 30.0
    slow_distance_cm: float = 100.0
    frame_index: int = 0
    # Fraction of frame height (0-1), not a raw pixel row: roi.yaml's points
    # are authored against a fixed 320x240 reference frame, so a cached
    # pixel value wouldn't scale to whatever resolution frames actually
    # arrive at (e.g. the live pipeline runs at 320x240, but test/calibration
    # frames can be smaller).
    horizon_y_frac: Optional[float] = None
    max_obstacle_range_cm: Optional[float] = 60.96  # 2 ft; farther objects are ignored entirely
    # A raw STOP must repeat this many consecutive frames before it's
    # trusted (see the STOP-confirmation comment in process_frame below).
    stop_confirm_frames: int = 3
    consecutive_raw_stop_frames: int = 0
    # Curb-proximity speed cap: compute_steer only ever corrects the
    # heading, it never touches speed, so a car driving FULL speed a few cm
    # from a tracked curb has nothing braking it while the steer correction
    # (and the curb tracker, which can lose the line entirely at very close
    # range) catches up -- that gap is what drives it into the curb. These
    # mirror stop_distance_cm/slow_distance_cm but against curb offset
    # instead of obstacle distance, so proximity to the curb itself throttles
    # speed the same way an obstacle ahead does.
    curb_stop_offset_cm: float = 15.0
    curb_slow_offset_cm: float = 25.0


def process_frame(frame: np.ndarray, state: PipelineState) -> dict:
    """Run one frame through the full perception -> decision pipeline.

    Parameters
    ----------
    frame:
        A single BGR frame, as returned by ``FrameSource.read()``.
    state:
        A :class:`PipelineState` carrying all stateful pipeline objects
        (classifier, curb tracker, smoothers, homography, thresholds).
        Mutated in place (the stateful sub-objects it holds advance their
        own internal counters/histories) but no state is read from or
        written to anywhere outside this object -- keeping this function
        deterministic for identical ``(frame, state)`` inputs regardless of
        whether the frame came from a live camera or a video file.

    Returns
    -------
    dict with at least: ``speed`` (SpeedTier), ``steer`` (int),
    ``curb_state`` (str, "tracking"/"fallback"), and
    ``obstacle_distance_cm`` (Optional[float]). Also includes
    ``curb_side`` and ``current_offset_cm`` for observability/testing.
    """
    road_mask = state.classifier.process_frame(frame)

    raw_side, raw_confidence = detect_curb_side(frame)
    state.curb_tracker.update(raw_side, raw_confidence)
    curb_side = state.curb_tracker.current_side
    curb_state = state.curb_tracker.state

    curb_x, current_offset_cm = estimate_curb_offset_cm(frame, curb_side, state.homography)

    # No curb reference: use the same centered band as the tracked-curb
    # case. A full-frame-width corridor here falsely treats anything
    # visible at the frame's edges (walls, furniture well outside the car's
    # actual path) as a blocking obstacle, which pins speed at STOP
    # indoors, where curb_side is always None -- the car never moves.
    fallback_band_frac = 0.6
    corridor_mask = define_corridor(
        frame.shape, curb_side, curb_x=curb_x, fallback_band_frac=fallback_band_frac
    )
    obstacles = detect_obstacles(frame, road_mask, corridor_mask)
    # nearest_obstacle_real_distance's horizon_y guard is documented as
    # required here ("callers are assumed to have already filtered image
    # points to the region of interest ... before calling this function")
    # but was never wired up: without it, pixels above the ROI's near-field
    # sampling region -- which the road-color model was never calibrated
    # against, so they misclassify as non-road constantly (far floor,
    # walls, background) -- get reprojected through the ground homography
    # anyway. Near/above the vanishing line that reprojection is unstable
    # and produces deceptively close-looking distances, pinning speed at
    # STOP even with a clear path ahead. Guard at the ROI's own top edge,
    # the same near-field region the classifier is actually trained on.
    #
    # roi.yaml's points are authored against a fixed reference frame size,
    # not necessarily this frame's actual size, so the top edge is cached
    # as a fraction of frame height and rescaled per frame.
    if state.horizon_y_frac is None:
        ref_width, ref_height = load_roi_reference_frame_size()
        state.horizon_y_frac = min(y for _, y in load_roi_points()) / ref_height
    horizon_y = state.horizon_y_frac * frame.shape[0]
    obstacle_distance_cm = nearest_obstacle_real_distance(
        obstacles,
        state.homography,
        horizon_y=horizon_y,
        max_forward_cm=state.max_obstacle_range_cm,
    )

    obstacle_speed = decide_speed_tier(
        obstacle_distance_cm,
        stop_distance_cm=state.stop_distance_cm,
        slow_distance_cm=state.slow_distance_cm,
    )
    # Curb proximity also throttles speed, independent of obstacles: only
    # meaningful while actually tracking a curb (current_offset_cm is a
    # stale/None value otherwise, per estimate_curb_offset_cm).
    curb_speed = SpeedTier.FULL
    if curb_side is not None and current_offset_cm is not None:
        curb_speed = decide_speed_tier(
            current_offset_cm,
            stop_distance_cm=state.curb_stop_offset_cm,
            slow_distance_cm=state.curb_slow_offset_cm,
        )
    raw_speed = min(obstacle_speed, curb_speed)
    # STOP is safety-critical and must never wait on the majority-vote
    # smoother's temporal consensus (it needs 2-of-3 recent frames to
    # agree before a new value wins) -- that lag is fine for SLOW/FULL
    # jitter but can let the car coast into something it already detected.
    # Feed every raw tier into the smoother (so it keeps tracking recent
    # history for SLOW/FULL decisions), but let a raw STOP win immediately
    # once confirmed, regardless of what the smoothed vote says.
    #
    # "Confirmed" requires stop_confirm_frames consecutive raw STOPs, not
    # just one: a single-frame glare/reflection misclassification (bright,
    # briefly non-road-colored, gone again next frame) would otherwise
    # trip an immediate, unrecoverable STOP the same way a real obstacle
    # does. A real obstacle stays put across several frames; a lighting
    # artifact usually doesn't. This still reacts within a fraction of a
    # second (a few frames), just not on a single potentially-spurious one.
    smoothed_speed = state.speed_smoother.update(raw_speed)
    if raw_speed == SpeedTier.STOP:
        state.consecutive_raw_stop_frames += 1
    else:
        state.consecutive_raw_stop_frames = 0
    stop_confirmed = state.consecutive_raw_stop_frames >= state.stop_confirm_frames
    speed = SpeedTier.STOP if stop_confirmed else smoothed_speed

    raw_steer = compute_steer(curb_side, current_offset_cm)
    if state.steer_smoother is not None:
        steer = state.steer_smoother.update(raw_steer)
    else:
        steer = raw_steer

    state.frame_index += 1

    return {
        "speed": speed,
        "steer": steer,
        "curb_state": curb_state,
        "obstacle_distance_cm": obstacle_distance_cm,
        "curb_side": curb_side,
        "current_offset_cm": current_offset_cm,
    }


def run_pipeline(
    source: FrameSource,
    serial_client,
    homography: np.ndarray,
    watchdog: Watchdog,
    max_frames: Optional[int] = None,
) -> List[dict]:
    """Run the main capture -> process -> drive loop.

    Reads frames from ``source`` until it's exhausted (``read()`` returns
    ``None``) or ``max_frames`` frames have been processed (whichever comes
    first). For every frame: notifies ``watchdog`` a frame arrived, runs
    :func:`process_frame`, and sends the resulting decision over
    ``serial_client`` via ``watchdog.write_command`` (which itself forces a
    STOP if the write fails). Periodically polls
    ``watchdog.check_frame_staleness()``.

    Parameters
    ----------
    source:
        Frame source to pull frames from (live or file -- see
        :func:`build_frame_source`).
    serial_client:
        A ``SerialClient``-like object; not called directly here (all
        writes go through ``watchdog``), but accepted so the caller's
        serial connection is explicit and easy to find/mock in tests.
    homography:
        3x3 ground-plane homography matrix.
    watchdog:
        :class:`~rccar.watchdog.watchdog.Watchdog` wrapping ``serial_client``.
    max_frames:
        Optional cap on the number of frames to process. ``None`` (default)
        means "run until the source is exhausted" (or the process is
        interrupted, e.g. Ctrl+C, for a live source).

    Returns
    -------
    List of per-frame result dicts, one per processed frame (see
    :func:`process_frame`).
    """
    stop_distance_cm, slow_distance_cm = load_thresholds()
    state = PipelineState(
        classifier=AdaptiveClassifier(),
        curb_tracker=CurbConfidenceTracker(),
        homography=homography,
        speed_smoother=MajorityVoteSmoother(),
        steer_smoother=MajorityVoteSmoother(),
        stop_distance_cm=stop_distance_cm,
        slow_distance_cm=slow_distance_cm,
    )

    results: List[dict] = []
    frame_count = 0

    while max_frames is None or frame_count < max_frames:
        frame = source.read()
        if frame is None:
            break

        watchdog.on_frame_received()

        result = process_frame(frame, state)
        watchdog.write_command(result["speed"], result["steer"])
        results.append(result)

        frame_count += 1
        if frame_count % _STALENESS_CHECK_INTERVAL == 0:
            watchdog.check_frame_staleness()

    return results


def build_frame_source(args: argparse.Namespace) -> FrameSource:
    """Construct the FrameSource selected by the parsed CLI args.

    ``--source-file`` -> :class:`VideoFileSource`;
    ``--device-index`` -> :class:`LiveCameraSource`. ``parse_args`` enforces
    that exactly one of these is set, so no further validation is needed
    here.
    """
    if args.source_file is not None:
        return VideoFileSource(args.source_file)
    return LiveCameraSource(args.device_index)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the main pipeline entry point."""
    default_port, default_baud, default_timeout = _load_serial_defaults()

    parser = argparse.ArgumentParser(
        prog="rccar-main",
        description=(
            "RC-car obstacle-detection main pipeline: capture -> "
            "segmentation/curb -> obstacles -> decision -> serial -> watchdog."
        ),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-file",
        type=str,
        default=None,
        help="Path to a video file to use as the frame source.",
    )
    source_group.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Camera device index to use as a live frame source.",
    )

    parser.add_argument(
        "--serial-port",
        type=str,
        default=default_port,
        help=f"Serial port for the MCU link (default from config/serial.yaml: {default_port!r}).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=default_baud,
        help=f"Serial baud rate (default from config/serial.yaml: {default_baud}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=default_timeout,
        help=f"Serial write timeout in seconds (default from config/serial.yaml: {default_timeout}).",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default=_DEFAULT_HOMOGRAPHY_PATH,
        help=f"Path to the homography config YAML (default: {_DEFAULT_HOMOGRAPHY_PATH!r}).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process (for finite test/demo runs).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None, serial_client=None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument list to parse (defaults to ``sys.argv[1:]`` via argparse).
    serial_client:
        Optional pre-built serial client. If given, it is used as-is
        (useful for testing/dry-running main() without a real serial port
        or CLI-provided --serial-port/--baud); if omitted (the primary,
        real-hardware path), a real :class:`SerialClient` is constructed
        from the parsed ``--serial-port``/``--baud``/``--timeout``.

    Returns
    -------
    0 on success, 1 on error (e.g. missing homography config, bad frame
    source, or failure to open the serial port).
    """
    args = parse_args(argv)

    try:
        homography = load_homography(args.homography)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        source = build_frame_source(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    owns_serial_client = serial_client is None
    if owns_serial_client:
        try:
            serial_client = SerialClient(args.serial_port, args.baud, args.timeout)
        except SerialClientError as exc:
            print(f"error: {exc}", file=sys.stderr)
            source.release()
            return 1

    watchdog = Watchdog(serial_client)

    try:
        run_pipeline(source, serial_client, homography, watchdog, max_frames=args.max_frames)
    finally:
        source.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
