"""Obstacle blob detection within the drivable corridor (T16).

Combines the road/non-road mask (T8, see :mod:`rccar.segmentation.classify`)
with the curb side (T11/T12, see :mod:`rccar.curb.detect` and
:mod:`rccar.curb.confidence`) to define a "drivable corridor" -- the region
of the frame the car should actually be checking for obstacles -- and then
finds non-road blobs inside that corridor, discarding anything too small to
be a real obstacle (noise/quantization speckle from the T8 classifier).

Corridor definition
--------------------
The car keeps to the side of a detected curb (away from the curb, not
straddling it):

* ``curb_side == "left"``: the curb is on the frame's left, so the
  drivable corridor is everything to the *right* of the curb line's
  x-position -- the half-plane ``x >= curb_x``.
* ``curb_side == "right"``: symmetric -- the corridor is ``x <= curb_x``.
* ``curb_side in (None, "none")`` (fallback mode, i.e.
  ``CurbConfidenceTracker.state == "fallback"``): there is no reliable
  curb to reference, so we fall back to a fixed lane-center band -- the
  centered ``fallback_band_frac``-width vertical strip of the frame
  (default: middle 60% of frame width). This mirrors driving down the
  middle of the lane when no curb reference is available. We chose a
  fixed centered band over reusing ``config/roi.yaml``'s trapezoid because
  the corridor here is a simple full-height x-range (matching the
  left/right curb cases above), whereas the ROI trapezoid is a
  near-field-only, narrowing shape meant for road-color sampling, not for
  defining "everything the car might hit ahead". Callers that want a
  ROI-based fallback can still build a mask from ``config/roi.yaml`` and
  intersect it with the returned corridor mask themselves.

``curb_x``
----------
``define_corridor`` accepts an optional ``curb_x``: the curb line's
approximate x-position in pixels (e.g. the midpoint x of the raw Hough
line segment picked by :func:`rccar.curb.detect.detect_curb_side`,
computed the same way that module computes ``_line_midpoint_x`` -- average
of the segment's two endpoint x-coordinates). If the caller has it, pass
it for an accurate split at the curb's actual position. If omitted (e.g.
only the categorical side is available, not the raw line), the split
falls back to the frame's horizontal center (``width / 2``), which is a
reasonable default since curbs are usually detected close to one side of
the frame.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


def define_corridor(
    frame_shape: tuple,
    curb_side: Optional[str],
    curb_x: Optional[float] = None,
    fallback_band_frac: float = 0.6,
) -> np.ndarray:
    """Return a binary mask of the drivable corridor for ``frame_shape``.

    Parameters
    ----------
    frame_shape:
        ``(H, W)`` or ``(H, W, C)`` -- only the first two dims are used.
    curb_side:
        ``"left"``, ``"right"``, ``None``, or ``"none"``. ``None``/``"none"``
        selects fallback (lane-center band) mode -- see module docstring.
    curb_x:
        Optional curb line x-position in pixels. Used only when
        ``curb_side`` is ``"left"`` or ``"right"``; ignored in fallback
        mode. Defaults to the frame's horizontal center if not given. See
        module docstring for how to derive this from a raw Hough line.
    fallback_band_frac:
        Width of the centered fallback band, as a fraction of frame width
        in ``(0, 1]``. Default 0.6 (middle 60% of frame width).

    Returns
    -------
    ``np.ndarray`` of shape ``(H, W)``, dtype ``uint8``, 255 inside the
    corridor and 0 outside.
    """
    height, width = frame_shape[0], frame_shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)

    if curb_side in ("left", "right"):
        split_x = curb_x if curb_x is not None else width / 2.0
        split_x = int(round(split_x))
        split_x = max(0, min(width, split_x))
        if curb_side == "left":
            # Curb on the left -> corridor is right of the curb line.
            mask[:, split_x:] = 255
        else:
            # Curb on the right -> corridor is left of the curb line.
            mask[:, :split_x] = 255
    else:
        # Fallback: centered vertical band, fallback_band_frac of width.
        if not (0.0 < fallback_band_frac <= 1.0):
            raise ValueError(
                f"fallback_band_frac must be in (0, 1], got {fallback_band_frac}"
            )
        band_width = width * fallback_band_frac
        center = width / 2.0
        lo = int(round(center - band_width / 2.0))
        hi = int(round(center + band_width / 2.0))
        lo = max(0, lo)
        hi = min(width, hi)
        mask[:, lo:hi] = 255

    return mask


@dataclass
class Obstacle:
    """A detected non-road blob inside the drivable corridor."""

    centroid: Tuple[float, float]
    area: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


def detect_obstacles(
    frame: np.ndarray,
    road_mask: np.ndarray,
    corridor_mask: np.ndarray,
    min_blob_area: int = 50,
) -> List[Obstacle]:
    """Find non-road blobs inside the drivable corridor.

    Parameters
    ----------
    frame:
        The source frame. Not directly used for pixel values (the blob
        detection works purely off ``road_mask``/``corridor_mask``); kept
        as a parameter for a stable call signature and so callers/future
        extensions (e.g. color-based blob refinement) have it on hand.
    road_mask:
        ``(H, W)`` uint8 mask from :func:`rccar.segmentation.classify.classify_frame`
        (255 = road, 0 = non-road).
    corridor_mask:
        ``(H, W)`` uint8 mask from :func:`define_corridor` (255 = inside
        drivable corridor).
    min_blob_area:
        Minimum connected-component pixel area for a blob to be reported
        as an obstacle; smaller blobs are treated as classifier noise and
        discarded.

    Returns
    -------
    List of :class:`Obstacle`, one per surviving blob. Empty if none
    survive (including the case where there are no non-road pixels in the
    corridor at all).
    """
    if road_mask.shape[:2] != corridor_mask.shape[:2]:
        raise ValueError(
            f"road_mask shape {road_mask.shape} and corridor_mask shape "
            f"{corridor_mask.shape} must match"
        )

    non_road_mask = cv2.bitwise_not(road_mask)
    candidate_mask = cv2.bitwise_and(non_road_mask, corridor_mask)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )

    obstacles: List[Obstacle] = []
    # Label 0 is always the background component; skip it.
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_blob_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        obstacles.append(
            Obstacle(
                centroid=(float(cx), float(cy)),
                area=area,
                bbox=(x, y, w, h),
            )
        )

    return obstacles
