"""Core homography math for camera calibration (T13).

Given >= 4 correspondences between image pixel coordinates and
ground-plane world coordinates (in cm), compute the 3x3 homography
matrix that maps image pixels -> ground-plane (x, y) cm.

This module contains only pure, testable logic. The interactive CLI
that reads a points file and writes a config lives in
``scripts/calibrate_camera.py``.
"""

from __future__ import annotations

import numpy as np
import cv2


# Points closer than this (in either the image-pixel or world-cm
# coordinate system) to a straight line are considered degenerate.
# It's expressed as a fraction of the point spread (see
# `_max_point_spread`) so it scales sensibly with the units in use.
_COLLINEARITY_TOLERANCE_FRACTION = 1e-3

# A homography whose condition number exceeds this is treated as
# "effectively singular" -- numerically unstable and not trustworthy.
_MAX_CONDITION_NUMBER = 1e12


def _max_point_spread(points: np.ndarray) -> float:
    """Return the largest pairwise coordinate range across a point set,
    used to scale the collinearity tolerance to the point set's units."""
    spread_x = points[:, 0].max() - points[:, 0].min()
    spread_y = points[:, 1].max() - points[:, 1].min()
    return float(max(spread_x, spread_y, 1.0))


def _is_collinear(points: np.ndarray) -> bool:
    """Return True if all points in `points` (Nx2) lie (near) a single
    straight line, i.e. would produce a degenerate homography."""
    if len(points) < 3:
        return True

    tolerance = _COLLINEARITY_TOLERANCE_FRACTION * _max_point_spread(points)

    # Find two points that are not coincident to define the reference line.
    p0 = points[0]
    ref = None
    for p in points[1:]:
        if np.linalg.norm(p - p0) > 1e-9:
            ref = p
            break
    if ref is None:
        # All points coincide with p0.
        return True

    direction = ref - p0
    direction_norm = np.linalg.norm(direction)

    for p in points:
        # Perpendicular distance from p to the line through p0/ref, via
        # the magnitude of the 2D cross product divided by line length.
        cross = direction[0] * (p[1] - p0[1]) - direction[1] * (p[0] - p0[0])
        distance = abs(cross) / direction_norm
        if distance > tolerance:
            return False
    return True


def compute_homography(
    image_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]],
) -> np.ndarray:
    """Compute the 3x3 homography mapping image pixel -> ground-plane
    (x, y) cm, given corresponding point pairs.

    Args:
        image_points: pixel coordinates (x, y) in the camera frame.
        world_points: corresponding ground-plane coordinates (x, y) in cm.

    Returns:
        3x3 numpy array (dtype float64) such that, for a homogeneous
        image point p = [px, py, 1], H @ p (normalized by its third
        component) gives the ground-plane [x, y] in cm.

    Raises:
        ValueError: if fewer than 4 point pairs are given, the two
            lists have mismatched lengths, either point set is
            (near-)collinear/degenerate, or the resulting homography
            is None or numerically singular/ill-conditioned.
    """
    if len(image_points) != len(world_points):
        raise ValueError(
            f"image_points and world_points must have the same length "
            f"(got {len(image_points)} and {len(world_points)})"
        )
    if len(image_points) < 4:
        raise ValueError(
            f"Need at least 4 point correspondences to compute a homography, "
            f"got {len(image_points)}. Provide 4+ known image/world point pairs."
        )

    img_pts = np.array(image_points, dtype=np.float64)
    world_pts = np.array(world_points, dtype=np.float64)

    if _is_collinear(img_pts):
        raise ValueError(
            "Image points are (near-)collinear -- cannot compute a valid "
            "homography from points that all lie on a single line. Choose "
            "4 points that span the ground plane (e.g. corners of a "
            "rectangle), not points along a single row/line."
        )
    if _is_collinear(world_pts):
        raise ValueError(
            "World (ground-plane) points are (near-)collinear -- cannot "
            "compute a valid homography from points that all lie on a "
            "single line. Choose 4 points that span the ground plane "
            "(e.g. corners of a rectangle), not points along a single line."
        )

    img_pts_f32 = img_pts.astype(np.float32)
    world_pts_f32 = world_pts.astype(np.float32)

    if len(image_points) == 4:
        homography = cv2.getPerspectiveTransform(img_pts_f32, world_pts_f32)
    else:
        homography, _mask = cv2.findHomography(img_pts_f32, world_pts_f32, method=0)

    if homography is None:
        raise ValueError(
            "cv2 failed to compute a homography from the given points "
            "(returned None). Check that the points are correct and not "
            "degenerate."
        )

    homography = np.asarray(homography, dtype=np.float64)

    if not np.all(np.isfinite(homography)):
        raise ValueError(
            "Computed homography contains non-finite values (NaN/inf) -- "
            "the input points are likely degenerate or duplicated."
        )

    determinant = np.linalg.det(homography)
    if abs(determinant) < 1e-12:
        raise ValueError(
            "Computed homography is singular (near-zero determinant) -- "
            "the input points do not uniquely determine a valid mapping. "
            "Check for duplicate or degenerate points."
        )

    condition_number = np.linalg.cond(homography)
    if not np.isfinite(condition_number) or condition_number > _MAX_CONDITION_NUMBER:
        raise ValueError(
            f"Computed homography is ill-conditioned (condition number "
            f"{condition_number:.3g}) -- the input points are likely "
            f"(near-)degenerate. Choose points that are well-spread and "
            f"not close to collinear."
        )

    return homography


def apply_homography(homography: np.ndarray, image_point: tuple[float, float]) -> tuple[float, float]:
    """Map a single image pixel coordinate through `homography` to a
    ground-plane (x, y) coordinate in cm."""
    px, py = image_point
    vec = np.array([px, py, 1.0], dtype=np.float64)
    mapped = homography @ vec
    if abs(mapped[2]) < 1e-12:
        raise ValueError("Homography maps this point to infinity (w ~= 0).")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])
