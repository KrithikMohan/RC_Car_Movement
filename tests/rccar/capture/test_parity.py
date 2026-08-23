"""Code-path parity test (T5): the same processing function, run unchanged
over LiveCameraSource and VideoFileSource, must produce identical output.

Covers:
  - happy: process_frame() output sequence is identical for both source
    types when they're backed by the same underlying synthetic video.
  - edge: both source types hit early EOF (return None) at exactly the
    same frame index.
"""

from unittest.mock import patch

import cv2
import numpy as np

from rccar.capture.file import VideoFileSource
from rccar.capture.live import LiveCameraSource


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


def process_frame(frame: np.ndarray):
    """Trivial stub processing function -- deterministic summary of a frame.

    Not real CV logic; just enough to prove the exact same function runs
    unchanged over both source types with identical results.
    """
    return (frame.shape, round(float(frame.mean()), 6))


def _live_source_backed_by_file(video_path: str, fake_device_index: int = 7) -> LiveCameraSource:
    """Construct a LiveCameraSource whose underlying cv2.VideoCapture is
    actually pointed at `video_path`, regardless of the device index passed.
    Only cv2.VideoCapture as seen from rccar.capture.live is redirected."""
    real_video_capture = cv2.VideoCapture

    def fake_video_capture(_index, *args, **kwargs):
        return real_video_capture(video_path)

    with patch("rccar.capture.live.cv2.VideoCapture", side_effect=fake_video_capture):
        return LiveCameraSource(fake_device_index)


def test_process_frame_identical_across_file_and_live_sources(tmp_path):
    """Happy path: the same process_frame() function, run unchanged over
    VideoFileSource and a (monkeypatched) LiveCameraSource backed by the
    same synthetic video, yields an identical output sequence."""
    video_path = str(tmp_path / "synthetic.mp4")
    _make_synthetic_video(video_path, NUM_FRAMES)

    file_source = VideoFileSource(video_path)
    live_source = _live_source_backed_by_file(video_path)
    try:
        assert file_source.is_live() is False
        assert live_source.is_live() is True

        file_outputs = []
        frame = file_source.read()
        while frame is not None:
            file_outputs.append(process_frame(frame))
            frame = file_source.read()

        live_outputs = []
        frame = live_source.read()
        while frame is not None:
            live_outputs.append(process_frame(frame))
            frame = live_source.read()

        assert len(file_outputs) == NUM_FRAMES
        assert live_outputs == file_outputs
    finally:
        file_source.release()
        live_source.release()


def test_early_eof_matches_across_file_and_live_sources(tmp_path):
    """Edge case: both source types stop yielding frames (return None) at
    the same frame index for the same underlying synthetic video, and
    keep returning None cleanly on subsequent reads."""
    video_path = str(tmp_path / "synthetic_short.mp4")
    _make_synthetic_video(video_path, NUM_FRAMES)

    file_source = VideoFileSource(video_path)
    live_source = _live_source_backed_by_file(video_path)
    try:
        # Read one past the known frame count from both sources, tracking
        # the index at which each first returns None.
        file_eof_index = None
        live_eof_index = None
        for i in range(NUM_FRAMES + 2):
            file_frame = file_source.read()
            if file_frame is None and file_eof_index is None:
                file_eof_index = i
            live_frame = live_source.read()
            if live_frame is None and live_eof_index is None:
                live_eof_index = i

        assert file_eof_index == NUM_FRAMES
        assert live_eof_index == NUM_FRAMES
        assert file_eof_index == live_eof_index

        # Both should continue to cleanly return None past EOF.
        assert file_source.read() is None
        assert live_source.read() is None
    finally:
        file_source.release()
        live_source.release()
