import time
import threading
import numpy as np
import pytest
from rccar.capture.source import FrameSource
from rccar.main import run_pipeline
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
    sent_commands = []
    def mock_send_serial(cmd):
        sent_commands.append(cmd)

    monkeypatch.setattr("pi_server.send_serial", mock_send_serial)

    homography = load_homography("config/homography.yaml")
    source = MockSyntheticFrameSource(num_frames=12)
    adapter = UGV02SerialAdapter()

    telemetry_records = []
    def state_cb(meta):
        telemetry_records.append(meta)

    run_pipeline(
        source=source,
        serial_client=adapter,
        homography=homography,
        state_callback=state_cb,
        max_frames=12
    )

    assert len(telemetry_records) > 0
    assert len(sent_commands) > 0

    # Final command sent upon pipeline completion is stop
    assert '{"T":13,"X":0.000,"Z":0.000}' in sent_commands[-1]
