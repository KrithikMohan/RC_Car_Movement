#!/usr/bin/env python3
"""Homography sanity-check tool (T14).

Given a stored homography (`config/homography.yaml`, produced by
`scripts/calibrate_camera.py`) and a reference file of known physical
measurements (`tests/fixtures/calibration_ref.yaml`-format: pixel
coordinates of a marker in a test frame, paired with its
tape-measure-verified real-world distance from the camera), this tool
maps each reference pixel through the homography and reports the
measured vs. actual ground-plane distance, the error (absolute cm and
percentage), and a PASS/FAIL verdict against a configurable tolerance.

This is the *real-world calibration-accuracy* check -- distinct from
`tests/calibration/test_calibrate_camera.py` and
`tests/calibration/test_homography_api.py`, which validate the
homography math against synthetic, hand-constructed ground truth.
This tool is only as good as the physical measurements in the
reference file; see `tests/fixtures/calibration_ref.yaml` for the
placeholder that must be replaced with a real tape-measure reading
before this tool's output means anything.

Usage
-----
    python3 scripts/check_calibration.py \\
        --homography config/homography.yaml \\
        --ref-file tests/fixtures/calibration_ref.yaml \\
        --tolerance-pct 10.0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Allow running this script directly (`python3 scripts/check_calibration.py`)
# without having installed the `rccar` package or set PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rccar.calibration.homography_api import (  # noqa: E402
    image_point_to_ground,
    load_homography,
)


def load_reference_points(ref_path: str) -> list[dict[str, Any]]:
    """Load reference points from a `calibration_ref.yaml`-format file.

    The file must contain a top-level `reference_points` list, each
    entry a dict with `image_point` ([px, py]), `known_distance_cm`
    (float), and `label` (str).

    Args:
        ref_path: path to the reference YAML file.

    Returns:
        List of dicts, one per reference point, with keys
        `image_point` (tuple[float, float]), `known_distance_cm`
        (float), and `label` (str).

    Raises:
        FileNotFoundError: if `ref_path` does not exist.
        ValueError: if the file exists but doesn't contain a valid
            `reference_points` list.
    """
    path = Path(ref_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration reference file not found at '{ref_path}'. This "
            f"file must contain known physical measurements (pixel "
            f"coordinates + tape-measure distance) to check the homography "
            f"against -- see tests/fixtures/calibration_ref.yaml for the "
            f"expected format."
        )

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "reference_points" not in data:
        raise ValueError(
            f"'{ref_path}' does not contain a top-level 'reference_points' "
            f"list. Expected format matches tests/fixtures/calibration_ref.yaml."
        )

    raw_points = data["reference_points"]
    if not isinstance(raw_points, list) or len(raw_points) == 0:
        raise ValueError(
            f"'reference_points' in '{ref_path}' must be a non-empty list."
        )

    reference_points: list[dict[str, Any]] = []
    for i, entry in enumerate(raw_points):
        if not isinstance(entry, dict):
            raise ValueError(
                f"'reference_points[{i}]' in '{ref_path}' must be a mapping "
                f"with 'image_point', 'known_distance_cm', and 'label' keys."
            )
        try:
            px, py = entry["image_point"]
            known_distance_cm = float(entry["known_distance_cm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"'reference_points[{i}]' in '{ref_path}' is malformed -- "
                f"expected 'image_point: [px, py]' and 'known_distance_cm: "
                f"<float>'. Error: {exc}"
            ) from exc
        label = str(entry.get("label", f"reference_points[{i}]"))
        reference_points.append(
            {
                "image_point": (float(px), float(py)),
                "known_distance_cm": known_distance_cm,
                "label": label,
            }
        )

    return reference_points


def check_calibration(
    homography: np.ndarray,
    ref_points: list[dict[str, Any]],
    tolerance_pct: float = 10.0,
) -> list[dict[str, Any]]:
    """Compare homography-derived ground-plane distances against known
    physical measurements.

    For each reference point, maps `image_point` through the
    homography via `image_point_to_ground` and computes the Euclidean
    ground-plane distance from the origin (`math.hypot(x, y)`) -- the
    same convention `nearest_obstacle_distance` in
    `rccar.calibration.homography_api` uses -- then compares it to
    `known_distance_cm`.

    Args:
        homography: 3x3 matrix as returned by `load_homography`.
        ref_points: list of dicts as returned by
            `load_reference_points` (each with `image_point`,
            `known_distance_cm`, `label`).
        tolerance_pct: allowed error, as a percentage of
            `known_distance_cm`, for a point to PASS.

    Returns:
        List of per-point result dicts with keys: `label`,
        `image_point`, `known_distance_cm`, `measured_distance_cm`
        (None if the point is off the ground plane), `error_cm`,
        `error_pct`, and `passed` (bool).
    """
    results: list[dict[str, Any]] = []
    for point in ref_points:
        image_point = point["image_point"]
        known_distance_cm = point["known_distance_cm"]
        label = point["label"]

        ground = image_point_to_ground(homography, image_point)

        if ground is None:
            results.append(
                {
                    "label": label,
                    "image_point": image_point,
                    "known_distance_cm": known_distance_cm,
                    "measured_distance_cm": None,
                    "error_cm": None,
                    "error_pct": None,
                    "passed": False,
                }
            )
            continue

        x, y = ground
        measured_distance_cm = math.hypot(x, y)
        error_cm = measured_distance_cm - known_distance_cm
        error_pct = (
            (abs(error_cm) / known_distance_cm) * 100.0
            if known_distance_cm != 0
            else float("inf")
        )
        passed = error_pct <= tolerance_pct

        results.append(
            {
                "label": label,
                "image_point": image_point,
                "known_distance_cm": known_distance_cm,
                "measured_distance_cm": measured_distance_cm,
                "error_cm": error_cm,
                "error_pct": error_pct,
                "passed": passed,
            }
        )

    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['label']}")
        print(f"    image_point:       {result['image_point']}")
        print(f"    known distance:    {result['known_distance_cm']:.2f} cm")
        if result["measured_distance_cm"] is None:
            print("    measured distance: N/A (point is above the horizon / off the ground plane)")
        else:
            print(f"    measured distance: {result['measured_distance_cm']:.2f} cm")
            print(
                f"    error:             {result['error_cm']:+.2f} cm "
                f"({result['error_pct']:.1f}%)"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Homography sanity check: compare pixel->ground-plane distances "
            "computed from the stored homography against known physical "
            "measurements."
        )
    )
    parser.add_argument(
        "--homography",
        default="config/homography.yaml",
        help="Path to the homography YAML config (default: config/homography.yaml)",
    )
    parser.add_argument(
        "--ref-file",
        default="tests/fixtures/calibration_ref.yaml",
        help=(
            "Path to the calibration reference file with known physical "
            "measurements (default: tests/fixtures/calibration_ref.yaml)"
        ),
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=10.0,
        help="Allowed error as a percentage of the known distance (default: 10.0)",
    )
    args = parser.parse_args(argv)

    try:
        homography = load_homography(args.homography)
        ref_points = load_reference_points(args.ref_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"check_calibration failed: {exc}", file=sys.stderr)
        return 1

    results = check_calibration(homography, ref_points, tolerance_pct=args.tolerance_pct)
    _print_results(results)

    all_passed = all(r["passed"] for r in results)
    print("Overall: PASS" if all_passed else "Overall: FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
