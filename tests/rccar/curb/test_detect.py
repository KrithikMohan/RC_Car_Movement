import cv2
import numpy as np

from rccar.curb.detect import detect_curb_side


def test_detect_curb_side_left():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Diagonal/vertical-ish line on the left side of frame center (x < 160).
    cv2.line(frame, (60, 240), (100, 150), (255, 255, 255), thickness=3)

    side, confidence = detect_curb_side(frame)

    assert side == "left"
    assert 0.0 <= confidence <= 1.0


def test_detect_curb_side_right():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Diagonal/vertical-ish line on the right side of frame center (x >= 160),
    # kept within the default ROI trapezoid (which narrows toward the top)
    # and long enough to clear the Hough vote threshold in the stub.
    cv2.line(frame, (270, 240), (200, 160), (255, 255, 255), thickness=3)

    side, confidence = detect_curb_side(frame)

    assert side == "right"
    assert 0.0 <= confidence <= 1.0


def test_detect_curb_side_none_on_flat_frame():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)

    side, confidence = detect_curb_side(frame)

    assert side == "none"
    assert confidence == 0.0


def test_detect_curb_side_none_on_horizontal_noise_only():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Shallow, near-horizontal lines (well within the 20 degree threshold)
    # simulating shadows / road texture / horizon noise -- should be
    # filtered out entirely, leaving no valid candidates.
    cv2.line(frame, (40, 200), (280, 210), (255, 255, 255), thickness=3)
    cv2.line(frame, (40, 170), (280, 175), (255, 255, 255), thickness=3)

    side, confidence = detect_curb_side(frame)

    assert side == "none"
    assert confidence == 0.0
