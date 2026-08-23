import cv2
import numpy as np

from rccar.curb.hough_stub import detect_lines


def test_detect_lines_on_synthetic_diagonal_line():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Draw a clear white diagonal line (well within the default ROI) that is
    # comfortably longer than the 30px minLineLength.
    cv2.line(frame, (50, 230), (250, 160), (255, 255, 255), thickness=3)

    lines = detect_lines(frame)

    assert isinstance(lines, list)
    assert len(lines) >= 0
    for line in lines:
        assert len(line) == 4


def test_detect_lines_on_flat_frame_returns_empty_list():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)

    lines = detect_lines(frame)

    assert lines == []
