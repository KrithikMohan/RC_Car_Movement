"""Trapezoid region-of-interest (ROI) geometry for near-field road sampling.

The ROI is a fixed, config-driven trapezoid: wide at the bottom of the frame
(closest to the car) and narrower toward a horizon-ish line partway up the
frame. It is defined in ``config/roi.yaml`` as pixel coordinates for a
reference 320x240 frame.

This module is responsible for:

* Loading the trapezoid points (from the YAML config, or accepting explicit
  points passed in directly).
* Building a binary mask for a given frame shape via ``cv2.fillPoly``,
  clipping the polygon to the frame bounds so an ROI that partially exceeds
  the frame (e.g. a car nose/hood visible in the crop, or a mismatched
  frame size) never crashes.
* Extracting the pixel values that fall inside the ROI from a frame.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# config/roi.yaml lives at <project-root>/config/roi.yaml; this file lives at
# <project-root>/src/rccar/segmentation/roi.py.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "roi.yaml"

Point = Tuple[int, int]


def load_roi_points(config_path: Optional[Path] = None) -> List[Point]:
    """Load the trapezoid ROI points from a YAML config file.

    Parameters
    ----------
    config_path:
        Path to a YAML file with a ``points`` key (list of ``[x, y]`` pairs).
        Defaults to ``config/roi.yaml`` at the project root.

    Returns
    -------
    List of ``(x, y)`` integer pixel-coordinate tuples.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    points = [(int(x), int(y)) for x, y in data["points"]]
    return points


def load_roi_reference_frame_size(config_path: Optional[Path] = None) -> Tuple[int, int]:
    """Load the ``(frame_width, frame_height)`` the ROI's points were authored
    against, so callers can scale those points to a differently-sized frame.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return int(data["frame_width"]), int(data["frame_height"])


def _clip_points(points: Sequence[Point], width: int, height: int) -> List[Point]:
    """Clip ROI points into ``[0, width-1] x [0, height-1]`` bounds.

    This is what lets a configured ROI that partially (or wholly) exceeds
    the actual frame dimensions still produce a valid, non-crashing mask:
    any point outside the frame is pulled back to the nearest edge rather
    than raising or silently producing garbage indices.
    """
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    return [
        (int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))) for x, y in points
    ]


def build_roi_mask(
    frame_shape: Tuple[int, ...],
    points: Optional[Sequence[Point]] = None,
    config_path: Optional[Path] = None,
) -> np.ndarray:
    """Build a binary ROI mask matching ``frame_shape``.

    Parameters
    ----------
    frame_shape:
        Shape of the target frame, e.g. ``frame.shape`` (``(H, W)`` or
        ``(H, W, C)``).
    points:
        Explicit ``(x, y)`` trapezoid points to use instead of loading from
        config.
    config_path:
        Optional override for the ROI config file (used only when
        ``points`` is not given).

    Returns
    -------
    ``np.ndarray`` of shape ``(H, W)``, dtype ``uint8``, with 255 inside the
    (frame-clipped) trapezoid and 0 outside.
    """
    height, width = frame_shape[0], frame_shape[1]

    if points is None:
        points = load_roi_points(config_path)

    clipped = _clip_points(points, width, height)

    mask = np.zeros((height, width), dtype=np.uint8)
    if width <= 0 or height <= 0:
        return mask

    polygon = np.array([clipped], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return mask


def extract_roi_pixels(
    frame: np.ndarray,
    points: Optional[Sequence[Point]] = None,
    config_path: Optional[Path] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the pixel values of ``frame`` that fall inside the ROI.

    Parameters
    ----------
    frame:
        Image array of shape ``(H, W)`` or ``(H, W, C)``.
    points, config_path:
        See :func:`build_roi_mask`; ignored if ``mask`` is given.
    mask:
        Precomputed mask (from :func:`build_roi_mask`) to reuse instead of
        rebuilding it.

    Returns
    -------
    Array of shape ``(N, C)`` (or ``(N,)`` for single-channel frames)
    containing the selected pixels. May be empty (``N == 0``) if the ROI
    does not overlap the frame at all, but is always a valid array.
    """
    if mask is None:
        mask = build_roi_mask(frame.shape, points=points, config_path=config_path)

    return frame[mask.astype(bool)]
