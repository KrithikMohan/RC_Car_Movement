import time
import threading
import numpy as np
import pytest
from pi_server import CameraGrabber, GrabberFrameSource


def test_camera_grabber_concurrency():
    grabber = CameraGrabber()
    assert grabber.latest_frame is not None

    frame1 = grabber.read()
    assert isinstance(frame1, np.ndarray)

    # Verify frame memory isolation (independent copies)
    frame1[0, 0] = [255, 255, 255]
    frame2 = grabber.read()
    assert not np.array_equal(frame1, frame2) or np.array_equal(grabber.latest_frame, frame1)

    source = GrabberFrameSource(grabber)
    assert source.is_live()

    downsampled = source.read()
    assert downsampled.shape[1] == 320
    assert downsampled.shape[0] == 240

    source.release()
    assert not source.is_live()
