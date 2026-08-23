"""Live camera frame source, backed by cv2.VideoCapture."""

from typing import Optional

import cv2
import numpy as np

from rccar.capture.source import FrameSource


class LiveCameraSource(FrameSource):
    """Frame source that reads from a live camera device via OpenCV.

    Raises RuntimeError at construction time if the device cannot be
    opened, so callers don't have to discover a bad device index deep in
    a read loop.
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(
                f"LiveCameraSource: failed to open camera device index "
                f"{device_index!r}. Check that the device exists and is "
                f"not in use by another process."
            )

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def is_live(self) -> bool:
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
