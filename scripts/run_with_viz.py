#!/usr/bin/env python3
"""CLI wrapper for the T26 debug/viz overlay tool.

Runs the perception/decision pipeline over a live camera or recorded video
file, rendering a debug overlay (segmentation mask, curb line, decision
text) on each frame, and either writes the result to an output video file
and/or shows it live via a ``cv2.imshow`` window.

Mirrors ``rccar.main``'s ``--source-file``/``--device-index`` mutually
exclusive CLI pattern.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rccar.viz.overlay import build_frame_source, run_viz


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_with_viz",
        description=(
            "Run the rccar perception/decision pipeline with a debug overlay "
            "(segmentation mask, curb line, decision text) rendered on each frame."
        ),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-file",
        type=str,
        default=None,
        help="Path to a video file to use as the frame source.",
    )
    source_group.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Camera device index to use as a live frame source.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the overlaid video to (e.g. out.mp4).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Never open a cv2.imshow window (headless mode).",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default="config/homography.yaml",
        help="Path to the homography config YAML (default: 'config/homography.yaml').",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    from rccar.calibration.homography_api import load_homography

    try:
        homography = load_homography(args.homography)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        source = build_frame_source(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        run_viz(
            source,
            output_path=args.output,
            show_window=not args.no_display,
            homography=homography,
            max_frames=args.max_frames,
        )
    finally:
        source.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
