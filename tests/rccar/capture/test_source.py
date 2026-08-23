"""Tests for the FrameSource abstraction (T4).

Covers:
  - happy: VideoFileSource on a short synthetic clip yields N frames then None.
  - edge: LiveCameraSource with a bad device index fails cleanly (no hang).
  - error: VideoFileSource pointed at a nonexistent path raises at
    construction time, not deep in a read loop.
"""

import os

import cv2
import numpy as np
import pytest

from rccar.capture.file import VideoFileSource
from rccar.capture.live import LiveCameraSource
from rccar.capture.source import FrameSource


NUM_FRAMES = 5
FRAME_SIZE = (64, 48)  # width, height


def _make_synthetic_video(path: str, num_frames: int = NUM_FRAMES) -> None:
    """Write a tiny solid-color test video to `path` using cv2.VideoWriter."""
    width, height = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"failed to open VideoWriter for {path}"
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=(i * 40) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_file_source_yields_n_frames_then_none(tmp_path):
    """Happy path: a short synthetic clip yields exactly N frames, then None at EOF."""
    video_path = str(tmp_path / "synthetic.mp4")
    _make_synthetic_video(video_path, NUM_FRAMES)

    source = VideoFileSource(video_path)
    try:
        assert isinstance(source, FrameSource)
        assert source.is_live() is False

        frames = []
        for _ in range(NUM_FRAMES):
            frame = source.read()
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            frames.append(frame)

        assert len(frames) == NUM_FRAMES

        # End of stream: further reads should cleanly return None.
        assert source.read() is None
        assert source.read() is None
    finally:
        source.release()


def test_video_file_source_missing_path_raises_at_construction(tmp_path):
    """Error path: nonexistent file path raises FileNotFoundError immediately,
    at construction time -- not deep inside a read loop."""
    missing_path = str(tmp_path / "does_not_exist.mp4")
    assert not os.path.exists(missing_path)

    with pytest.raises(FileNotFoundError):
        VideoFileSource(missing_path)


def test_live_camera_source_bad_device_index_fails_cleanly():
    """Edge case: an obviously-invalid device index should fail fast (raise)
    rather than hang or silently succeed."""
    bad_index = 999

    with pytest.raises(RuntimeError):
        # Should return promptly (no hang) and raise a clear error rather
        # than yielding a source whose read() spins forever.
        LiveCameraSource(bad_index)
