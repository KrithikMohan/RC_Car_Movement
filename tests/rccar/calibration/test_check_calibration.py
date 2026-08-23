"""Tests for the homography sanity-check tool (T14).

Covers:
  - happy: a synthetic ground-truth homography (same H_TRUE-style
    construction pattern as tests/calibration/test_calibrate_camera.py
    and tests/calibration/test_homography_api.py) is used to derive
    an image point that maps to a known 50cm ground distance; running
    it through `check_calibration` reports PASS within tolerance.
  - error: a missing/malformed homography file raises a clear error
    (propagated from `load_homography`) rather than silently
    producing a wrong number.
"""

import math

import numpy as np
import pytest
import yaml

from rccar.calibration.homography_api import load_homography

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from check_calibration import check_calibration, load_reference_points  # noqa: E402

# Same ground-truth homography construction pattern as
# tests/calibration/test_calibrate_camera.py and
# tests/calibration/test_homography_api.py.
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


def test_check_calibration_passes_within_tolerance():
    """Happy path: a reference point derived from H_TRUE with a known
    50cm ground distance should report PASS within +/-5cm tolerance."""
    # Pick a world point directly ahead on the ground plane with
    # Euclidean distance from the origin of exactly 50cm.
    world_point = (0.0, 50.0)
    known_distance_cm = math.hypot(*world_point)
    assert known_distance_cm == pytest.approx(50.0)

    image_point = _image_point_for_world_point(world_point)

    ref_points = [
        {
            "image_point": image_point,
            "known_distance_cm": known_distance_cm,
            "label": "synthetic 50cm test point",
        }
    ]

    results = check_calibration(H_TRUE, ref_points, tolerance_pct=10.0)

    assert len(results) == 1
    result = results[0]
    assert result["measured_distance_cm"] is not None
    assert result["measured_distance_cm"] == pytest.approx(known_distance_cm, abs=5.0)
    assert result["passed"] is True


def test_check_calibration_ref_file_roundtrip(tmp_path):
    """load_reference_points reads back a calibration_ref.yaml-format
    file and check_calibration consumes it end-to-end."""
    world_point = (0.0, 50.0)
    known_distance_cm = math.hypot(*world_point)
    image_point = _image_point_for_world_point(world_point)

    ref_path = tmp_path / "calibration_ref.yaml"
    ref_path.write_text(
        yaml.dump(
            {
                "reference_points": [
                    {
                        "image_point": [image_point[0], image_point[1]],
                        "known_distance_cm": known_distance_cm,
                        "label": "synthetic 50cm test point",
                    }
                ]
            }
        )
    )

    ref_points = load_reference_points(str(ref_path))
    results = check_calibration(H_TRUE, ref_points, tolerance_pct=10.0)

    assert results[0]["passed"] is True


def test_load_homography_missing_file_raises_clear_error(tmp_path):
    """Error case: a missing homography file raises a clear
    FileNotFoundError (propagated from load_homography), not a
    silent wrong number."""
    missing_path = str(tmp_path / "does_not_exist.yaml")

    with pytest.raises(FileNotFoundError, match="calibration"):
        load_homography(missing_path)


def test_load_homography_malformed_file_raises_clear_error(tmp_path):
    """Error case: a malformed homography file raises a clear
    ValueError (propagated from load_homography)."""
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.dump({"not_homography": [[1, 2, 3]]}))

    with pytest.raises(ValueError):
        load_homography(str(bad_path))


def test_load_reference_points_missing_file_raises_clear_error(tmp_path):
    """Error case: a missing ref file raises a clear FileNotFoundError."""
    missing_path = str(tmp_path / "does_not_exist.yaml")

    with pytest.raises(FileNotFoundError, match="reference"):
        load_reference_points(missing_path)


def test_load_reference_points_malformed_file_raises_clear_error(tmp_path):
    """Error case: a malformed ref file (missing reference_points key)
    raises a clear ValueError."""
    bad_path = tmp_path / "bad_ref.yaml"
    bad_path.write_text(yaml.dump({"not_reference_points": []}))

    with pytest.raises(ValueError):
        load_reference_points(str(bad_path))
