"""Tests for nearest-obstacle real-world distance (T17).

Covers:
  - happy: multiple Obstacles at different real-world distances, built
    via a synthetic ground-truth homography (same construction pattern
    as tests/calibration/test_homography_api.py). One case is chosen
    specifically so the pixel-closest centroid (nearest, in raw image
    pixels, to a reference point like the image's bottom-center, which
    is where "close to the camera" would naively look) is NOT the
    real-world-closest obstacle -- proving the function uses the
    homography-mapped ground distance rather than raw pixel distance.
  - edge: an empty obstacle list returns None cleanly.
"""

import numpy as np
import pytest

from rccar.obstacles.detect import Obstacle
from rccar.obstacles.distance import nearest_obstacle_real_distance

# Same ground-truth homography construction pattern as
# tests/calibration/test_homography_api.py: a downward-angled camera
# with non-uniform (projective) scaling, so cm-per-pixel differs
# across the image -- this is what makes the "pixel-closest is not
# real-world-closest" scenario possible.
H_TRUE = np.array(
    [
        [0.20, 0.00, -32.0],
        [0.00, -0.60, 130.0],
        [0.00, -0.0020, 1.0],
    ],
    dtype=np.float64,
)


def _obstacle_at(image_point):
    return Obstacle(centroid=image_point, area=100, bbox=(0, 0, 10, 10))


def test_nearest_obstacle_real_distance_uses_ground_distance_not_pixel_distance():
    """Happy path with three obstacles at known, distinct real-world
    distances (computed by applying H_TRUE forward to each pixel
    centroid):

      - pixel (0, 220)   -> world (-57.14, -3.57)  -> real dist ~57.25 cm
      - pixel (320, 200) -> world (53.33, 16.67)   -> real dist ~55.88 cm
      - pixel (160, 0)   -> world (0.0, 130.0)     -> real dist 130.0 cm

    Pixel (0, 220) is closer (in raw pixel Euclidean distance) to the
    image's bottom-center (160, 240) -- the naive "near the camera"
    reference point -- than pixel (320, 200) is:
      dist((0,220), (160,240))   ~= 161.2 px
      dist((320,200), (160,240)) ~= 164.9 px
    Yet pixel (320, 200) is the true nearest obstacle in real-world
    ground-plane distance (~55.88 cm vs ~57.25 cm). This confirms the
    function reduces via the homography-mapped ground distance, not
    raw pixel proximity.
    """
    pixel_closest_but_not_real_closest = _obstacle_at((0.0, 220.0))
    real_closest = _obstacle_at((320.0, 200.0))
    far_obstacle = _obstacle_at((160.0, 0.0))

    obstacles = [
        pixel_closest_but_not_real_closest,
        far_obstacle,
        real_closest,
    ]

    result = nearest_obstacle_real_distance(obstacles, H_TRUE)

    expected = (53.333333333333336**2 + 16.666666666666668**2) ** 0.5
    assert result == pytest.approx(expected, abs=1e-6)
    # Sanity: it's strictly less than the pixel-closest obstacle's real
    # distance -- i.e. the "pixel closer" obstacle did NOT win.
    pixel_closest_real_dist = (57.14285714285714**2 + 3.571428571428571**2) ** 0.5
    assert result < pixel_closest_real_dist


def test_nearest_obstacle_real_distance_empty_list_returns_none():
    assert nearest_obstacle_real_distance([], H_TRUE) is None
