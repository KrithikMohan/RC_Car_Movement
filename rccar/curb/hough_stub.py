"""Bare-bones Canny + Hough line detection stub for curb perception.

This is deliberately minimal: it exists to give a later on-hardware
performance test a representative CV workload (grayscale conversion, Canny
edge detection, ROI masking, probabilistic Hough transform) to measure. It
does not attempt any curb-side classification or line filtering — that is
left to a later task that will build real curb-side-detection logic on top
of this stub.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from rccar.segmentation.roi import build_roi_mask

Line = Tuple[int, int, int, int]


def detect_lines(
    frame: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> List[Line]:
    """Run Canny edge detection + probabilistic Hough transform on a frame.

    Parameters
    ----------
    frame:
        Image array of shape ``(H, W)`` or ``(H, W, C)``.
    roi_mask:
        Optional binary mask (from :func:`build_roi_mask`) restricting edge
        detection to a region of interest. If not given, a default ROI mask
        is built via ``build_roi_mask(frame.shape[:2])``.

    Returns
    -------
    List of ``(x1, y1, x2, y2)`` integer line segment tuples. Empty list if
    no lines are found (e.g. on a blank/solid-color frame).
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    edges = cv2.Canny(gray, 50, 150)

    if roi_mask is None:
        roi_mask = build_roi_mask(frame.shape[:2])

    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10,
    )

    if lines is None:
        return []

    # cv2.HoughLinesP's output shape varies by opencv-python version --
    # some return (N, 1, 4), others (N, 4). Normalize via reshape instead
    # of assuming either shape.
    return [tuple(int(v) for v in line) for line in np.asarray(lines).reshape(-1, 4)]
