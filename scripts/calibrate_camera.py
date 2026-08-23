#!/usr/bin/env python3
"""One-time camera calibration CLI (T13).

Computes the homography that maps camera pixel coordinates to
ground-plane coordinates (in cm), from operator-supplied point
correspondences, and writes the result to a YAML config file.

Usage
-----
1. Point the car's camera at the ground and measure at least 4 known
   points on the floor in front of it (e.g. tape-measure marks or
   corners of a rectangle taped to the ground).
2. For each point, record its pixel location in the image (e.g. by
   opening a saved frame in an image viewer) and its real-world
   ground-plane coordinates in cm (x = lateral offset from camera
   center, y = distance in front of the camera).
3. Write these correspondences to a small JSON or YAML file, e.g.::

       image_points:
         - [45, 210]
         - [275, 210]
         - [200, 120]
         - [120, 120]
       world_points:
         - [-30, 40]
         - [30, 40]
         - [15, 100]
         - [-15, 100]

4. Run::

       python3 scripts/calibrate_camera.py --points-file points.yaml \\
           --output config/homography.yaml

The core math (`compute_homography`) lives in
`rccar.calibration.homography` so it can be unit tested independently
of this CLI/file-IO wrapper.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import yaml

# Allow running this script directly (`python3 scripts/calibrate_camera.py`)
# without having installed the `rccar` package or set PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rccar.calibration.homography import compute_homography  # noqa: E402


def _load_points_file(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Load image/world point correspondences from a YAML or JSON file.

    The file must contain an `image_points` list and a `world_points`
    list of the same length, each entry being a 2-element [x, y] pair.
    """
    if not path.exists():
        raise FileNotFoundError(f"Points file not found: {path}")

    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict) or "image_points" not in data or "world_points" not in data:
        raise ValueError(
            f"{path} must contain top-level 'image_points' and 'world_points' lists"
        )

    image_points = [tuple(p) for p in data["image_points"]]
    world_points = [tuple(p) for p in data["world_points"]]
    return image_points, world_points


def _write_homography_config(
    output_path: Path,
    homography,
    frame_width: int,
    frame_height: int,
) -> None:
    """Write the computed homography and calibration metadata to a YAML config."""
    config = {
        "homography": homography.tolist(),
        "calibration_date": datetime.date.today().isoformat(),
        "image_width": frame_width,
        "image_height": frame_height,
        "units": "cm",
        "notes": (
            "Maps camera pixel coordinates (x, y) to ground-plane coordinates "
            "(x, y) in cm. Apply as: [X, Y, W]^T = homography @ [px, py, 1]^T, "
            "then divide by W."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-time camera calibration: compute a homography from image "
            "pixel <-> ground-plane (cm) point correspondences."
        )
    )
    parser.add_argument(
        "--points-file",
        required=True,
        type=Path,
        help=(
            "YAML or JSON file with 'image_points' and 'world_points' lists "
            "(each a list of [x, y] pairs, at least 4 entries)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/homography.yaml"),
        help="Path to write the resulting homography config (default: config/homography.yaml)",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=320,
        help="Image width (px) the calibration was performed at (default: 320)",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=240,
        help="Image height (px) the calibration was performed at (default: 240)",
    )
    args = parser.parse_args(argv)

    try:
        image_points, world_points = _load_points_file(args.points_file)
        homography = compute_homography(image_points, world_points)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    _write_homography_config(args.output, homography, args.frame_width, args.frame_height)
    print(f"Wrote homography to {args.output}")
    print(homography)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
