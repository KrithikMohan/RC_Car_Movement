"""Tests for the T26 debug/viz overlay tool.

Covers:
  - happy: run_viz against a short synthetic video, writing to an output
    video file with show_window=False, confirms the output file exists,
    has non-zero size, and has the same frame count as the input.
  - edge: run_viz with show_window=False never calls cv2.imshow (headless
    mode -- no GUI/display required), confirmed by mocking cv2.imshow and
    asserting it is never invoked.
"""

import os
from unittest.mock import patch

import cv2
import numpy as np

from rccar.calibration.homography_api import load_homography
from rccar.capture.file import VideoFileSource
from rccar.viz.overlay import run_viz

NUM_FRAMES = 6
FRAME_SIZE = (160, 120)  # width, height

_REPO_ROOT_HOMOGRAPHY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "config",
    "homography.yaml",
)


def _make_synthetic_video(path: str, num_frames: int = NUM_FRAMES) -> None:
    """Write a short synthetic clip with per-frame varying solid colors,
    same technique as tests/capture/test_source.py and tests/test_main.py."""
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


def _count_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    try:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        return count
    finally:
        cap.release()


def test_run_viz_writes_output_video_with_same_frame_count(tmp_path):
    """Happy path: run_viz over a synthetic clip, writing to output_path
    with show_window=False, produces a non-empty video file with the same
    frame count as the input."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")
    _make_synthetic_video(input_path, NUM_FRAMES)

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    source = VideoFileSource(input_path)
    try:
        run_viz(
            source,
            output_path=output_path,
            show_window=False,
            homography=homography,
            max_frames=NUM_FRAMES,
        )
    finally:
        source.release()

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    output_frame_count = _count_frames(output_path)
    assert output_frame_count == NUM_FRAMES


def test_run_viz_show_window_false_never_calls_imshow(tmp_path):
    """Edge case: show_window=False must never call cv2.imshow, so run_viz
    completes cleanly in a headless/CI environment with no display and no
    GUI dependency -- confirmed by mocking cv2.imshow directly."""
    input_path = str(tmp_path / "input.mp4")
    _make_synthetic_video(input_path, NUM_FRAMES)

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)
    source = VideoFileSource(input_path)

    with patch("cv2.imshow") as mock_imshow:
        try:
            run_viz(
                source,
                output_path=None,
                show_window=False,
                homography=homography,
                max_frames=NUM_FRAMES,
            )
        finally:
            source.release()

        mock_imshow.assert_not_called()
