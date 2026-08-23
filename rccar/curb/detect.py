"""Curb-line side auto-detection built on top of the raw Hough line stub.

Given the raw ``(x1, y1, x2, y2)`` line segments produced by
:func:`rccar.curb.hough_stub.detect_lines`, decide whether a curb line is
visible on the left or right side of the frame, or not at all.

Approach
--------
1. Run ``detect_lines`` to get raw Hough segments.
2. Filter out near-horizontal lines. The car drives roughly straight ahead,
   so a real curb line runs away from it and should have a fairly steep
   slope in the image, not be near-flat like shadows / road texture /
   horizon-ish noise. Lines within ``HORIZONTAL_ANGLE_THRESHOLD_DEG``
   (default 20 degrees) of horizontal (0 or 180 degrees) are rejected.
3. For each surviving candidate, compute its midpoint x-coordinate and
   compare it to the frame's horizontal center to classify it as left-of-
   center or right-of-center.
4. Pick the single longest surviving candidate as the "best" line (the stub
   does not expose Hough vote counts, so segment length is used as a proxy
   for line strength/confidence) and report its side.
5. If no candidates survive filtering, return ``("none", 0.0)`` rather than
   guessing a side.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from rccar.curb.hough_stub import Line, detect_lines

# Reject lines within this many degrees of horizontal (0 or 180 deg) as
# noise (shadows, road texture, horizon-ish lines) rather than a curb.
HORIZONTAL_ANGLE_THRESHOLD_DEG = 20.0


def _line_angle_deg(line: Line) -> float:
    """Return the line's angle in degrees, folded into [0, 180)."""
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle % 180.0


def _is_near_horizontal(line: Line, threshold_deg: float) -> bool:
    angle = _line_angle_deg(line)
    # Near-horizontal means close to 0 deg or close to 180 deg.
    return angle <= threshold_deg or angle >= (180.0 - threshold_deg)


def _line_length(line: Line) -> float:
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def _line_midpoint_x(line: Line) -> float:
    x1, _, x2, _ = line
    return (x1 + x2) / 2.0


def detect_curb_side(
    frame: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[str, float]:
    """Detect whether a curb line is visible on the left or right of frame.

    Parameters
    ----------
    frame:
        Image array of shape ``(H, W)`` or ``(H, W, C)``.
    roi_mask:
        Optional binary ROI mask, forwarded to
        :func:`rccar.curb.hough_stub.detect_lines`.

    Returns
    -------
    ``(side, confidence)`` where ``side`` is one of ``"left"``, ``"right"``,
    or ``"none"``, and ``confidence`` is a float in ``[0, 1]``.

    Confidence formula
    -------------------
    Let ``best`` be the length (in pixels) of the longest surviving
    candidate line, and ``runner_up`` be the length of the longest
    surviving candidate on the *opposite* side of center (0 if there is
    none). Confidence is::

        margin = (best - runner_up) / best        # in [0, 1]
        length_factor = min(best / 100.0, 1.0)     # saturate at 100px
        confidence = margin * length_factor

    This rewards a long, clearly-dominant line on one side and penalizes
    both short lines (low ``length_factor``) and cases where a comparably
    long competing line exists on the other side (low ``margin``). If no
    candidate lines survive the horizontal-angle filter, ``("none", 0.0)``
    is returned without guessing a side.
    """
    lines: List[Line] = detect_lines(frame, roi_mask)

    candidates = [
        line for line in lines if not _is_near_horizontal(line, HORIZONTAL_ANGLE_THRESHOLD_DEG)
    ]

    if not candidates:
        return "none", 0.0

    width = frame.shape[1]
    center_x = width / 2.0

    left_candidates = [c for c in candidates if _line_midpoint_x(c) < center_x]
    right_candidates = [c for c in candidates if _line_midpoint_x(c) >= center_x]

    best_left = max((_line_length(c) for c in left_candidates), default=0.0)
    best_right = max((_line_length(c) for c in right_candidates), default=0.0)

    if best_left == 0.0 and best_right == 0.0:
        return "none", 0.0

    if best_left >= best_right:
        side = "left"
        best = best_left
        runner_up = best_right
    else:
        side = "right"
        best = best_right
        runner_up = best_left

    margin = (best - runner_up) / best if best > 0 else 0.0
    length_factor = min(best / 100.0, 1.0)
    confidence = margin * length_factor

    return side, confidence
