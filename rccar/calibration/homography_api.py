"""High-level pixel-to-ground-plane distance API (T15).

This module wraps the low-level homography math in
``rccar.calibration.homography`` with the conveniences needed by
obstacle-detection code:

- loading a homography matrix from the YAML config format described
  in ``config/homography.yaml.example``,
- mapping a single image point to ground-plane cm with a "horizon
  guard" that rejects points that can't plausibly be on the ground
  plane (e.g. sky/background above the camera's horizon line), and
- reducing a list of obstacle centroids to the single nearest
  real-world distance.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import numpy as np
import yaml

from rccar.calibration.homography import apply_homography


def load_homography(config_path: str) -> np.ndarray:
    """Load a homography matrix from a YAML config file.

    The file must match the format produced by
    ``scripts/calibrate_camera.py`` (see
    ``config/homography.yaml.example``): a top-level ``homography`` key
    holding a 3x3 nested list.

    Args:
        config_path: path to a homography YAML file (e.g.
            ``config/homography.yaml``).

    Returns:
        3x3 numpy array (dtype float64).

    Raises:
        FileNotFoundError: if `config_path` does not exist. This is
            expected until a physical camera calibration has been run
            and its output saved to ``config/homography.yaml`` -- the
            error message says so explicitly so callers don't mistake
            it for a bug.
        ValueError: if the file exists but doesn't contain a valid
            3x3 ``homography`` matrix.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Homography config not found at '{config_path}'. This file is "
            f"produced by running the camera calibration procedure (see "
            f"scripts/calibrate_camera.py and config/homography.yaml.example "
            f"for the expected format) -- it does not exist until that "
            f"physical calibration has been performed."
        )

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "homography" not in data:
        raise ValueError(
            f"'{config_path}' does not contain a top-level 'homography' key. "
            f"Expected format matches config/homography.yaml.example."
        )

    matrix = np.array(data["homography"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"'homography' in '{config_path}' must be a 3x3 matrix, got "
            f"shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"'homography' in '{config_path}' contains non-finite values."
        )

    return matrix


def image_point_to_ground(
    homography: np.ndarray,
    image_point: tuple[float, float],
    horizon_y: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    """Map a single image pixel to ground-plane (x, y) cm, with a
    horizon guard.

    Points above the horizon/vanishing line don't lie on the ground
    plane (they're sky or distant background) and would otherwise
    reproject through the homography to nonsense coordinates (huge,
    negative, or wildly unstable distances near the vanishing line).

    Args:
        homography: 3x3 matrix as returned by `compute_homography` or
            `load_homography`.
        image_point: (px, py) pixel coordinates.
        horizon_y: the image y-coordinate of the horizon/vanishing
            line. Points with py < horizon_y (i.e. above the horizon
            in image space, since image y grows downward) are
            considered off the ground plane and this function returns
            None. If None (default), the check is skipped entirely --
            callers are assumed to have already filtered image points
            to the region of interest (e.g. below a configured ROI top
            edge) before calling this function.

    Returns:
        (x, y) ground-plane coordinates in cm, or None if the point is
        above the supplied horizon.
    """
    px, py = image_point
    if horizon_y is not None and py < horizon_y:
        return None
    return apply_homography(homography, image_point)


def nearest_obstacle_distance(
    homography: np.ndarray,
    obstacle_image_points: Sequence[tuple[float, float]],
    horizon_y: Optional[float] = None,
    max_forward_cm: Optional[float] = None,
) -> Optional[float]:
    """Return the distance (in cm) to the nearest obstacle.

    Each obstacle centroid pixel is mapped to ground-plane cm via
    `image_point_to_ground` (applying the horizon guard), and the
    Euclidean distance from the origin -- i.e. sqrt(x^2 + y^2) in the
    homography's ground-plane coordinate system, where (0, 0) is
    wherever the calibration defined the camera/origin reference point
    -- is computed for each valid mapping. The minimum such distance
    is returned.

    Euclidean distance (rather than forward-Y-only) is used so that
    obstacles off to the side are still accounted for; if your
    calibration's world frame puts pure "forward distance" entirely in
    Y with X centered on the camera, Euclidean and forward-Y distance
    will be very close for anything roughly ahead of the car.

    Args:
        homography: 3x3 matrix.
        obstacle_image_points: list of (px, py) obstacle centroid
            pixel coordinates.
        horizon_y: forwarded to `image_point_to_ground`; points above
            this y are ignored.
        max_forward_cm: if given, obstacles farther than this in the
            forward (ground-plane Y) direction are not considered
            obstacles at all, regardless of lateral (X) offset -- this
            bounds detection range straight ahead without penalizing
            things that are simply off to the side.

    Returns:
        Minimum ground-plane distance in cm across all valid (below-
        horizon, within max_forward_cm) obstacle points, or None if the
        input list is empty or no point qualifies.
    """
    best: Optional[float] = None
    for point in obstacle_image_points:
        ground = image_point_to_ground(homography, point, horizon_y=horizon_y)
        if ground is None:
            continue
        x, y = ground
        if max_forward_cm is not None and y > max_forward_cm:
            continue
        distance = math.hypot(x, y)
        if best is None or distance < best:
            best = distance
    return best
