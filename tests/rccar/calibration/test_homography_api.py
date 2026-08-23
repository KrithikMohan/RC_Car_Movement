"""Tests for the pixel-to-ground-plane distance API (T15).

Covers:
  - happy: a known image point (built from a synthetic ground-truth
    homography, same construction pattern as T13's test) maps to the
    expected ground-plane coordinate within 2cm.
  - edge: a point above the supplied horizon line returns None instead
    of a nonsense/garbage distance.
  - load_homography: missing file raises a clear FileNotFoundError;
    a real YAML file (matching config/homography.yaml.example) loads
    correctly.
  - nearest_obstacle_distance: picks the minimum valid distance and
    ignores points above the horizon; returns None when everything is
    filtered out or the input is empty.
"""

import numpy as np
import pytest
import yaml

from rccar.calibration.homography_api import (
    image_point_to_ground,
    load_homography,
    nearest_obstacle_distance,
)

# Same ground-truth homography construction pattern as
# tests/calibration/test_calibrate_camera.py: a downward-angled camera
# where right/down in the image maps to further right/further away on
# the ground.
H_TRUE = np.array(
    [
        [0.20, 0.00, -32.0],
        [0.00, -0.60, 130.0],
        [0.00, -0.0020, 1.0],
    ],
    dtype=np.float64,
)


def _image_point_for_world_point(world_xy):
    """Invert H_TRUE to find the image pixel that maps to `world_xy`."""
    h_inv = np.linalg.inv(H_TRUE)
    wx, wy = world_xy
    vec = h_inv @ np.array([wx, wy, 1.0])
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


def test_image_point_to_ground_maps_center_point_within_2cm():
    """Happy path: a point at (roughly) image center maps to the
    expected ground-plane forward distance within 2cm."""
    expected_world_point = (0.0, 95.0)
    image_point = _image_point_for_world_point(expected_world_point)

    result = image_point_to_ground(H_TRUE, image_point)

    assert result is not None
    x, y = result
    assert x == pytest.approx(expected_world_point[0], abs=2.0)
    assert y == pytest.approx(expected_world_point[1], abs=2.0)


def test_image_point_to_ground_above_horizon_returns_none():
    """Edge case: a point above the horizon (sky/far background) isn't
    on the ground plane -- should return None, not a bogus distance."""
    horizon_y = 50.0
    sky_point = (160.0, 10.0)  # py well above horizon_y

    result = image_point_to_ground(H_TRUE, sky_point, horizon_y=horizon_y)

    assert result is None


def test_image_point_to_ground_below_horizon_is_unaffected():
    """Points below (or at) the horizon are unaffected by the guard."""
    expected_world_point = (0.0, 95.0)
    image_point = _image_point_for_world_point(expected_world_point)
    px, py = image_point

    # horizon well above this point -> should pass through untouched.
    result = image_point_to_ground(H_TRUE, image_point, horizon_y=py - 10.0)

    assert result is not None


def test_image_point_to_ground_skips_check_when_horizon_not_given():
    """If horizon_y is None (default), no filtering happens."""
    sky_point = (160.0, 10.0)
    result = image_point_to_ground(H_TRUE, sky_point)
    assert result is not None


def test_load_homography_missing_file_raises_clear_error(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.yaml")

    with pytest.raises(FileNotFoundError, match="calibration"):
        load_homography(missing_path)


def test_load_homography_loads_valid_config(tmp_path):
    config_path = tmp_path / "homography.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "homography": H_TRUE.tolist(),
                "calibration_date": "2026-08-10",
                "image_width": 320,
                "image_height": 240,
                "units": "cm",
            }
        )
    )

    loaded = load_homography(str(config_path))

    assert loaded.shape == (3, 3)
    np.testing.assert_allclose(loaded, H_TRUE)


def test_load_homography_rejects_malformed_config(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.dump({"not_homography": [[1, 2, 3]]}))

    with pytest.raises(ValueError):
        load_homography(str(config_path))


def test_nearest_obstacle_distance_returns_minimum():
    near_world = (0.0, 50.0)
    far_world = (0.0, 150.0)
    near_point = _image_point_for_world_point(near_world)
    far_point = _image_point_for_world_point(far_world)

    result = nearest_obstacle_distance(H_TRUE, [far_point, near_point])

    expected = (near_world[0] ** 2 + near_world[1] ** 2) ** 0.5
    assert result == pytest.approx(expected, abs=2.0)


def test_nearest_obstacle_distance_ignores_points_above_horizon():
    near_world = (0.0, 50.0)
    near_point = _image_point_for_world_point(near_world)
    sky_point = (160.0, 10.0)
    horizon_y = 50.0

    result = nearest_obstacle_distance(H_TRUE, [sky_point, near_point], horizon_y=horizon_y)

    expected = (near_world[0] ** 2 + near_world[1] ** 2) ** 0.5
    assert result == pytest.approx(expected, abs=2.0)


def test_nearest_obstacle_distance_empty_list_returns_none():
    assert nearest_obstacle_distance(H_TRUE, []) is None


def test_nearest_obstacle_distance_all_above_horizon_returns_none():
    sky_point = (160.0, 10.0)
    result = nearest_obstacle_distance(H_TRUE, [sky_point], horizon_y=50.0)
    assert result is None
