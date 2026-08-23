"""Code-path parity re-check (T28) on the FULL wired pipeline.

``tests/capture/test_parity.py`` (T5) proved that ``VideoFileSource`` vs.
``LiveCameraSource`` (monkeypatched to open the same underlying file)
produce an identical sequence of raw frames, exercised through a trivial
stub ``process_frame``. That's necessary but not sufficient: it only proves
parity at the capture layer.

This test re-runs the same idea against the FULL wired pipeline from T25
(``rccar.main.run_pipeline`` / ``process_frame``), all the way through
decision output (speed/steer) and the encoded serial bytes actually written
to the wire. It confirms that swapping the frame source implementation
(live vs. file) never leaks into the pipeline's behavior when the
underlying frames are identical -- i.e. ``process_frame``/``run_pipeline``
depend only on the frames and the (per-run, fresh) ``PipelineState``/
``Watchdog``/serial client they're given, never on which ``FrameSource``
subclass produced those frames.

Cases
-----
- happy: the same synthetic video, run once through a plain
  ``VideoFileSource`` and once through a ``LiveCameraSource`` monkeypatched
  (same technique as T5's ``test_parity.py``) to transparently open that
  same file, produces byte-for-byte identical serial writes and equal
  (exact for discrete fields, tolerance for float fields) per-frame result
  dicts.
"""

import os
from unittest.mock import patch

import cv2
import numpy as np

from rccar.calibration.homography_api import load_homography
from rccar.capture.file import VideoFileSource
from rccar.capture.live import LiveCameraSource
from rccar.main import run_pipeline
from rccar.watchdog.watchdog import Watchdog

_REPO_ROOT_HOMOGRAPHY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "config",
    "homography.yaml",
)

FRAME_SIZE = (320, 240)  # width, height -- matches the resolution config/roi.yaml
# and config/homography.yaml were calibrated against; obstacle_distance_cm
# is only meaningful (and the horizon/max-range guards only line up
# correctly) at this resolution.
NUM_FRAMES = 7

# Fields compared with an exact equality check: these are discrete/integer
# valued by the time process_frame returns them (SpeedTier enum members,
# an already-int-cast steer value, a small fixed-vocabulary curb state
# string, and the curb side string/None) -- there's no floating point
# arithmetic left un-rounded in them, so any difference would indicate a
# genuine parity break rather than incidental float noise.
_EXACT_FIELDS = ("speed", "steer", "curb_state", "curb_side")

# Fields compared with a small numeric tolerance: these carry raw
# floating-point results (nearest-obstacle distance / homography-derived
# ground-plane offset) that, in principle, could differ in their last bit
# between two otherwise-identical runs due to e.g. differing memory
# layouts/temporaries introduced by which cv2.VideoCapture code path
# produced the frame array. In practice OpenCV decodes should be bit
# identical here too, but tolerance-checking these keeps the test robust
# without weakening the meaningful discrete-field checks above.
_TOLERANCE_FIELDS = ("obstacle_distance_cm", "current_offset_cm")
_FLOAT_TOLERANCE = 1e-6


class FakeSerialClient:
    """Records every write; never opens a real port.

    Mirrors ``SerialClient.write``'s contract (raise, don't return False)
    closely enough for ``Watchdog`` to use it, while letting the test
    inspect exactly what bytes were sent -- same pattern as
    ``tests/test_main.py``.
    """

    def __init__(self):
        self.writes = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)


def _make_varying_video(path: str, num_frames: int) -> None:
    """Short synthetic clip with per-frame varying solid colors and obstacles
    of different sizes, positioned to trigger both steer and speed variation
    across frames so the parity test is non-vacuous."""
    width, height = FRAME_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"failed to open VideoWriter for {path}"
    try:
        for i in range(num_frames):
            # Road-colored background (greenish)
            frame = np.full((height, width, 3), (80, 100, 80), dtype=np.uint8)

            # Frame 0 stays obstacle-free (it seeds the road-color model);
            # later frames place a saturated, clearly non-road obstacle low
            # in the frame -- within the ROI's near-field band and within
            # nearest_obstacle_real_distance's forward-range guard -- so
            # this fixture produces a genuine FULL -> SLOW/STOP transition
            # rather than relying on incidental video-codec noise.
            if i > 0:
                obstacle_size = 20 + (i * 8) % 20
                obstacle_x = 100 + (i * 25) % (width - 200 - obstacle_size)
                obstacle_y = 236 - obstacle_size
                cv2.rectangle(
                    frame,
                    (obstacle_x, obstacle_y),
                    (obstacle_x + obstacle_size, 236),
                    (0, 0, 200),  # saturated red, clearly non-road
                    -1,
                )

            # Draw a curb line that shifts side to trigger steer variation
            curb_x = 40 + (i * 20) % (width - 60)  # moves across frame
            cv2.line(
                frame,
                (curb_x, height - 1),
                (curb_x + 30, 0),
                (200, 200, 200),  # bright line (road edge marker)
                3,
            )

            writer.write(frame)
    finally:
        writer.release()


def _live_source_backed_by_file(video_path: str, fake_device_index: int = 3) -> LiveCameraSource:
    """Construct a LiveCameraSource whose underlying cv2.VideoCapture is
    actually pointed at `video_path`, regardless of the device index
    passed. Only cv2.VideoCapture as seen from rccar.capture.live is
    redirected -- same technique as tests/capture/test_parity.py."""
    real_video_capture = cv2.VideoCapture

    def fake_video_capture(_index, *args, **kwargs):
        return real_video_capture(video_path)

    with patch("rccar.capture.live.cv2.VideoCapture", side_effect=fake_video_capture):
        return LiveCameraSource(fake_device_index)


def _run_full_pipeline(source, homography):
    """Drive run_pipeline with a fresh FakeSerialClient/Watchdog (run_pipeline
    itself constructs a fresh PipelineState internally each call -- see
    rccar.main.run_pipeline), returning (results, serial_writes)."""
    fake_serial = FakeSerialClient()
    watchdog = Watchdog(fake_serial, frame_timeout_ms=10_000)
    results = run_pipeline(source, fake_serial, homography, watchdog, max_frames=NUM_FRAMES)
    return results, fake_serial.writes


def test_full_pipeline_output_and_serial_writes_identical_across_file_and_live_sources(tmp_path):
    """Happy path: the same fixture, run through the full pipeline once via
    VideoFileSource and once via a (monkeypatched) LiveCameraSource backed
    by the same underlying file, produces an identical decision/output
    sequence and an identical sequence of encoded serial write bytes."""
    video_path = str(tmp_path / "synthetic.mp4")
    _make_varying_video(video_path, NUM_FRAMES)

    homography = load_homography(_REPO_ROOT_HOMOGRAPHY)

    # --- Run 1: plain VideoFileSource, fresh serial client/watchdog/state ---
    file_source = VideoFileSource(video_path)
    try:
        file_results, file_writes = _run_full_pipeline(file_source, homography)
    finally:
        file_source.release()

    # --- Run 2: monkeypatched LiveCameraSource backed by the same file,
    # its OWN fresh serial client/watchdog/state (run_pipeline builds a
    # fresh PipelineState internally on every call) ---
    live_source = _live_source_backed_by_file(video_path)
    try:
        live_results, live_writes = _run_full_pipeline(live_source, homography)
    finally:
        live_source.release()

    # Sanity: both runs actually processed every frame, and the fixture
    # produces some variation (not trivially-identical constant output
    # across frames within a single run), so this is a meaningful check.
    assert len(file_results) == NUM_FRAMES
    assert len(live_results) == NUM_FRAMES
    distinct_speeds = {r["speed"] for r in file_results}
    distinct_steers = {r["steer"] for r in file_results}
    assert len(distinct_speeds) > 1 or len(distinct_steers) > 1, (
        "fixture produced trivially-constant output; parity check would be "
        "vacuous"
    )

    # Per-frame result dict parity: exact equality for discrete fields,
    # tolerance-based equality for float fields (see module docstring /
    # _EXACT_FIELDS / _TOLERANCE_FIELDS for which-and-why).
    for i, (file_result, live_result) in enumerate(zip(file_results, live_results)):
        for field in _EXACT_FIELDS:
            assert file_result[field] == live_result[field], (
                f"frame {i}: exact field {field!r} differs: "
                f"file={file_result[field]!r} live={live_result[field]!r}"
            )
        for field in _TOLERANCE_FIELDS:
            file_val = file_result[field]
            live_val = live_result[field]
            if file_val is None or live_val is None:
                assert file_val is None and live_val is None, (
                    f"frame {i}: field {field!r} None-ness differs: "
                    f"file={file_val!r} live={live_val!r}"
                )
            else:
                assert abs(file_val - live_val) <= _FLOAT_TOLERANCE, (
                    f"frame {i}: tolerance field {field!r} differs beyond "
                    f"tolerance: file={file_val!r} live={live_val!r}"
                )

    # Wire-protocol parity: the exact same sequence of encoded serial
    # write bytes must have been sent in both runs.
    assert file_writes == live_writes
    assert len(file_writes) == NUM_FRAMES
