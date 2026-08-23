import time
import threading
import numpy as np
import pytest
from rccar.capture.source import FrameSource
from rccar.main import PipelineState, process_frame
from rccar.watchdog.watchdog import Watchdog
from rccar.segmentation.classify import AdaptiveClassifier
from rccar.curb.confidence import CurbConfidenceTracker
from rccar.decision.smoothing import MajorityVoteSmoother
from rccar.decision.speed import load_thresholds
from pi_server import UGV02SerialAdapter, load_homography


class MockSyntheticFrameSource(FrameSource):
    """Generates synthetic video frames containing a road, curb, and approaching obstacle."""
    def __init__(self, num_frames=15):
        self.num_frames = num_frames
        self.current_frame = 0
        self._active = True

    def read(self):
        if not self._active or self.current_frame >= self.num_frames:
            return None
        self.current_frame += 1

        # 320x240 frame with road background (dark gray)
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)

        # Draw clear right-side curb line (bright diagonal)
        # From (240, 240) to (180, 150)
        import cv2
        cv2.line(frame, (240, 240), (180, 150), (200, 200, 200), thickness=4)

        # Draw an obstacle in the corridor in later frames
        if self.current_frame >= 8:
            cv2.rectangle(frame, (130, 180), (170, 220), (20, 20, 240), -1)

        return frame

    def is_live(self):
        return self._active

    def release(self):
        self._active = False


def test_autodrive_pipeline_end_to_end(monkeypatch):
    """Drives rccar.main.process_frame() the same way pi_server._worker()'s
    auto-mode loop does (rccar.main.run_pipeline() has no state_callback/
    stop_event hooks, so pi_server drives process_frame directly -- see that
    function's comment). Feeds synthetic frames with a curb and an obstacle
    that appears partway through, and checks the pipeline both produces
    telemetry every frame and eventually commands a stop once the obstacle
    is within range.
    """
    sent_commands = []
    def mock_send_serial(cmd):
        sent_commands.append(cmd)

    monkeypatch.setattr("pi_server.send_serial", mock_send_serial)

    homography = load_homography("config/homography.yaml")
    source = MockSyntheticFrameSource(num_frames=12)
    adapter = UGV02SerialAdapter()
    watchdog = Watchdog(adapter)

    stop_distance_cm, slow_distance_cm = load_thresholds("config/thresholds.yaml")
    state = PipelineState(
        classifier=AdaptiveClassifier(),
        curb_tracker=CurbConfidenceTracker(),
        homography=homography,
        speed_smoother=MajorityVoteSmoother(),
        steer_smoother=MajorityVoteSmoother(),
        stop_distance_cm=stop_distance_cm,
        slow_distance_cm=slow_distance_cm,
    )

    telemetry_records = []
    while True:
        frame = source.read()
        if frame is None:
            break
        watchdog.on_frame_received()
        result = process_frame(frame, state)
        watchdog.write_command(result["speed"], result["steer"])
        telemetry_records.append(result)

    assert len(telemetry_records) == 12
    assert len(sent_commands) > 0

    # Once the obstacle (present from frame 8 onward) is within
    # stop_distance_cm, the pipeline must command a full stop (X=0).
    stop_tiers = [r for r in telemetry_records if r["speed"].name == "STOP"]
    assert stop_tiers, f"expected at least one STOP decision, got: {[r['speed'] for r in telemetry_records]}"
    assert '{"T":13,"X":0.000,"Z":0.000}' in sent_commands[-1]
