"""Integration test for the T25 main pipeline (capture -> segmentation/curb
-> obstacles -> decision -> serial -> watchdog), spanning the whole repo
rather than a single module -- hence its placement at the top level of
``tests/`` alongside the other cross-cutting suites.

Homography source
------------------
Uses the repo's real ``config/homography.yaml`` (loaded via
``rccar.calibration.homography_api.load_homography``) rather than
constructing a synthetic ``H_TRUE`` matrix like
``tests/calibration/test_homography_api.py`` does. It already exists, is a
valid 3x3 homography, and this test only needs *some* valid ground-plane
mapping to exercise the full pipeline end-to-end -- it doesn't assert on
specific real-world distances/offsets, so there's no need to hand-derive a
matrix with known image<->ground correspondences.

Serial client
-------------
A real serial port is never opened. A tiny ``FakeSerialClient`` fake (not a
``MagicMock``, for a stronger contract-check: its ``.write()`` validates it
was actually handed protocol-encoded bytes) stands in for
``rccar.serial_client.client.SerialClient`` and is passed directly to
``Watchdog``/``run_pipeline``.

Cases
-----
- happy: a short synthetic video (varying solid-color frames, same
  ``cv2.VideoWriter`` technique as ``tests/capture/test_source.py``) is run
  end-to-end through ``run_pipeline``. Confirms no crash and that every
  frame's result dict has non-None ``speed``/``steer``/``curb_state``, and
  that a valid protocol message was sent for every frame.
- edge: a synthetic clip of frames that are all the exact same flat color
  (no edges at all -> ``detect_curb_side`` returns ``("none", 0.0)`` on
  every frame) is run through the pipeline. Confirms steering falls back to
  0 on every frame (no curb to track), while speed is driven purely by
  obstacle distance (no obstacles on a blank frame -> distance is None ->
  speed stays FULL) -- i.e. losing the curb never forces a STOP by itself.
"""

import os

import cv2
import numpy as np
import pytest

from rccar.curb.confidence import CurbConfidenceTracker
from rccar.decision.smoothing import MajorityVoteSmoother
from rccar.main import PipelineState, process_frame, run_pipeline
from rccar.segmentation.classify import AdaptiveClassifier
from rccar.serial_client.protocol import SpeedTier, decode_command
from rccar.watchdog.watchdog import Watchdog
from rccar.calibration.homography_api import load_homography

_REPO_ROOT_HOMOGRAPHY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "homography.yaml"
)

FRAME_SIZE = (160, 120)  # width, height


class FakeSerialClient:
    """Records every write; never opens a real port.

    Mirrors ``SerialClient.write``'s contract (raise, don't return False)
    closely enough for ``Watchdog`` to use it, while letting the test
    inspect exactly what was sent.
    """

    def __init__(self):
        self.writes = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)


def _make_varying_video(path: str, num_frames: int) -> None:
    """Short synthetic clip with per-frame varying solid colors."""
    width, height = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"failed to open VideoWriter for {path}"
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=(i * 37) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _make_flat_video(path: str, num_frames: int, color: int = 120) -> None:
    """Synthetic clip where every frame is the exact same flat color --
    no edges at all, so detect_lines/detect_curb_side find nothing on every
    frame (guaranteed ("none", 0.0))."""
    width, height = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"failed to open VideoWriter for {path}"
    try:
        frame = np.full((height, width, 3), fill_value=color, dtype=np.uint8)
        for _ in range(num_frames):
            writer.write(frame)
    finally:
        writer.release()


def _build_watchdog_and_state(homography):
    fake_serial = FakeSerialClient()
    watchdog = Watchdog(fake_serial, frame_timeout_ms=10_000)
    return fake_serial, watchdog


def test_run_pipeline_happy_path_produces_valid_commands_for_every_frame(tmp_path):
    from rccar.capture.file import VideoFileSource

    num_frames = 8
    video_path = str(tmp_path / "varying.mp4")
    _make_varying_video(video_path, num_frames)

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    source = VideoFileSource(video_path)
    fake_serial, watchdog = _build_watchdog_and_state(homography)

    try:
        results = run_pipeline(source, fake_serial, homography, watchdog, max_frames=num_frames)
    finally:
        source.release()

    assert len(results) == num_frames

    for result in results:
        assert result["speed"] is not None
        assert isinstance(result["speed"], SpeedTier)
        assert result["steer"] is not None
        assert isinstance(result["steer"], int)
        assert result["curb_state"] in ("tracking", "fallback")

    # Every frame produced a serial write, and every write is a valid,
    # decodable protocol message.
    assert len(fake_serial.writes) == num_frames
    for raw in fake_serial.writes:
        speed, steer = decode_command(raw)
        assert speed in (SpeedTier.STOP, SpeedTier.SLOW, SpeedTier.FULL)
        assert -100 <= steer <= 100


def test_run_pipeline_no_curb_falls_back_to_straight_steer_without_forcing_stop(tmp_path):
    from rccar.capture.file import VideoFileSource

    num_frames = 6
    video_path = str(tmp_path / "flat.mp4")
    _make_flat_video(video_path, num_frames)

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    source = VideoFileSource(video_path)
    fake_serial, watchdog = _build_watchdog_and_state(homography)

    try:
        results = run_pipeline(source, fake_serial, homography, watchdog, max_frames=num_frames)
    finally:
        source.release()

    assert len(results) == num_frames

    for result in results:
        # No detectable curb on a flat frame -> curb tracker stays in
        # fallback and steering holds straight.
        assert result["curb_state"] == "fallback"
        assert result["curb_side"] is None
        assert result["steer"] == 0

        # No obstacles on a blank frame -> distance is None -> speed tier
        # is driven purely by obstacle distance, which defaults to FULL.
        # Crucially, it must NOT be forced to STOP just because the curb
        # is unavailable.
        assert result["obstacle_distance_cm"] is None
        assert result["speed"] == SpeedTier.FULL


def test_run_pipeline_no_curb_uses_centered_band_not_full_frame_corridor(tmp_path, monkeypatch):
    """Regression test: with no curb tracked (the normal indoor case),
    the obstacle corridor must stay a centered band, not widen to the
    full frame width. A full-frame corridor treats anything visible at
    the frame's edges (walls, furniture outside the car's actual path)
    as a blocking obstacle, which pins speed at STOP indoors and the car
    never moves."""
    import rccar.main as main_module
    from rccar.capture.file import VideoFileSource

    real_define_corridor = main_module.define_corridor
    seen_fracs = []

    def _spy_define_corridor(frame_shape, curb_side, curb_x=None, fallback_band_frac=0.6):
        seen_fracs.append(fallback_band_frac)
        return real_define_corridor(frame_shape, curb_side, curb_x=curb_x, fallback_band_frac=fallback_band_frac)

    monkeypatch.setattr(main_module, "define_corridor", _spy_define_corridor)

    num_frames = 6
    video_path = str(tmp_path / "flat.mp4")
    _make_flat_video(video_path, num_frames)

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    source = VideoFileSource(video_path)
    fake_serial, watchdog = _build_watchdog_and_state(homography)

    try:
        run_pipeline(source, fake_serial, homography, watchdog, max_frames=num_frames)
    finally:
        source.release()

    assert len(seen_fracs) == num_frames
    assert all(frac == 0.6 for frac in seen_fracs)


def test_process_frame_stops_when_dangerously_close_to_tracked_curb(monkeypatch):
    """Regression test: compute_steer only ever corrects heading, it never
    touches speed. Without a curb-proximity speed cap, a car with no
    obstacle in front (obstacle_distance_cm=None -> FULL) but only a few cm
    from a tracked curb would keep driving FULL speed while steering away
    -- exactly the gap that let a real unit crash into a curb it was
    correctly steering away from, because the steer correction couldn't
    act fast enough at full speed. Speed must be capped by curb proximity
    the same way it is by obstacle proximity."""
    import rccar.main as main_module

    monkeypatch.setattr(main_module, "detect_curb_side", lambda frame: ("left", 1.0))
    monkeypatch.setattr(
        main_module, "estimate_curb_offset_cm", lambda frame, curb_side, homography: (10.0, 3.2)
    )
    monkeypatch.setattr(main_module, "detect_obstacles", lambda frame, road_mask, corridor_mask: [])

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    state = PipelineState(
        classifier=AdaptiveClassifier(),
        curb_tracker=CurbConfidenceTracker(),
        homography=homography,
        speed_smoother=MajorityVoteSmoother(),
        steer_smoother=MajorityVoteSmoother(),
        stop_confirm_frames=1,
    )
    frame = np.zeros((*FRAME_SIZE[::-1], 3), dtype=np.uint8)

    result = process_frame(frame, state)

    assert result["obstacle_distance_cm"] is None
    assert result["current_offset_cm"] == 3.2
    assert result["speed"] == SpeedTier.STOP


def test_process_frame_is_deterministic_given_identical_state_inputs():
    """process_frame must behave identically for identical (frame, state)
    inputs, independent of any hidden global state -- the property T28's
    live-vs-file parity test relies on."""
    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    frame = np.full((120, 160, 3), fill_value=90, dtype=np.uint8)

    def _fresh_state():
        return PipelineState(
            classifier=AdaptiveClassifier(),
            curb_tracker=CurbConfidenceTracker(),
            homography=homography,
            speed_smoother=MajorityVoteSmoother(),
            steer_smoother=MajorityVoteSmoother(),
        )

    result_a = process_frame(frame, _fresh_state())
    result_b = process_frame(frame, _fresh_state())

    assert result_a["speed"] == result_b["speed"]
    assert result_a["steer"] == result_b["steer"]
    assert result_a["curb_state"] == result_b["curb_state"]
    assert result_a["obstacle_distance_cm"] == result_b["obstacle_distance_cm"]
