"""Nearest-obstacle real-world distance (T17).

Thin adapter between T16's obstacle blob detection
(:mod:`rccar.obstacles.detect`) and T15's homography-based distance API
(:mod:`rccar.calibration.homography_api`): extracts each
:class:`~rccar.obstacles.detect.Obstacle`'s pixel centroid and feeds the
list through :func:`rccar.calibration.homography_api.nearest_obstacle_distance`
to get the nearest obstacle's real-world (ground-plane) distance in cm.
No distance math is reimplemented here -- it's all in the wrapped function.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from rccar.calibration.homography_api import nearest_obstacle_distance
from rccar.obstacles.detect import Obstacle


def nearest_obstacle_real_distance(
    obstacles: List[Obstacle],
    homography: np.ndarray,
    horizon_y: Optional[float] = None,
) -> Optional[float]:
    """Return the real-world (ground-plane) distance in cm to the nearest obstacle.

    Args:
        obstacles: list of :class:`~rccar.obstacles.detect.Obstacle` (e.g.
            from :func:`rccar.obstacles.detect.detect_obstacles`). Each
            obstacle's ``.centroid`` (pixel coordinates) is used.
        homography: 3x3 homography matrix, as returned by
            ``load_homography`` or ``compute_homography``.
        horizon_y: forwarded to ``nearest_obstacle_distance``; obstacle
            centroids above this image y-coordinate are ignored as not
            being on the ground plane.

    Returns:
        Minimum ground-plane distance in cm across all valid (below-
        horizon) obstacles, or None if ``obstacles`` is empty or every
        centroid is above the horizon.
    """
    centroids = [obstacle.centroid for obstacle in obstacles]
    return nearest_obstacle_distance(homography, centroids, horizon_y=horizon_y)
