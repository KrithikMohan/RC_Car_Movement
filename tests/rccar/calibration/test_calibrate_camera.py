"""Tests for camera calibration homography computation (T13).

Covers:
  - happy: 4 known image/world point correspondences (derived from a
    known ground-truth homography) compute a homography that
    correctly reprojects a 5th, held-out point within 2cm at ~100cm
    range.
  - edge: collinear (degenerate) image points raise a clear ValueError
    instead of producing a garbage homography.
"""

import numpy as np
import pytest

from rccar.calibration.homography import apply_homography, compute_homography


# A hand-picked, invertible 3x3 "ground truth" homography mapping
# image pixel coordinates -> ground-plane (x, y) cm. Roughly models a
# downward-angled camera: right/down in the image maps to further
# right/further away on the ground, with a touch of perspective
# (nonzero bottom row) so it's a genuine homography, not just an affine map.
H_TRUE = np.array(
    [
        [0.20, 0.00, -32.0],
        [0.00, -0.60, 130.0],
        [0.00, -0.0020, 1.0],
    ],
    dtype=np.float64,
)

# 5 ground-plane points (cm) spread around the ~100cm range in front of
# the car, not collinear.
WORLD_POINTS = [
    (-30.0, 40.0),
    (30.0, 40.0),
    (30.0, 100.0),
    (-30.0, 100.0),
    (0.0, 95.0),  # held out for the reprojection check
]


def _image_point_for_world_point(world_xy):
    """Invert H_TRUE to find the image pixel that maps to `world_xy`
    under the ground-truth homography."""
    h_inv = np.linalg.inv(H_TRUE)
    wx, wy = world_xy
    vec = h_inv @ np.array([wx, wy, 1.0])
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def test_compute_homography_reprojects_held_out_point_within_2cm():
    """Happy path: compute a homography from 4 correspondences and use
    it to reproject a 5th, held-out point; result should match the
    true world coordinates within 2cm at ~100cm range."""
    image_points = [_image_point_for_world_point(wp) for wp in WORLD_POINTS]

    calibration_image_points = image_points[:4]
    calibration_world_points = WORLD_POINTS[:4]
    held_out_image_point = image_points[4]
    held_out_world_point = WORLD_POINTS[4]

    homography = compute_homography(calibration_image_points, calibration_world_points)

    assert homography.shape == (3, 3)
    assert np.all(np.isfinite(homography))

    reprojected_x, reprojected_y = apply_homography(homography, held_out_image_point)

    assert reprojected_x == pytest.approx(held_out_world_point[0], abs=2.0)
    assert reprojected_y == pytest.approx(held_out_world_point[1], abs=2.0)


def test_compute_homography_rejects_collinear_points():
    """Edge case: all 4 image points lying on a straight line is a
    degenerate configuration that cannot determine a unique
    homography -- compute_homography should raise ValueError rather
    than returning a garbage matrix."""
    collinear_image_points = [(10.0, 100.0), (50.0, 100.0), (90.0, 100.0), (130.0, 100.0)]
    world_points = [(-30.0, 40.0), (30.0, 40.0), (30.0, 100.0), (-30.0, 100.0)]

    with pytest.raises(ValueError):
        compute_homography(collinear_image_points, world_points)


def test_compute_homography_requires_at_least_four_points():
    """Edge case: fewer than 4 correspondences cannot determine a
    homography (8 unknowns need 8 equations from 4 points)."""
    with pytest.raises(ValueError):
        compute_homography([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
