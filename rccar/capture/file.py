"""Video file frame source, backed by cv2.VideoCapture."""

import os
from typing import Optional

import cv2
import numpy as np

from rccar.capture.source import FrameSource


class VideoFileSource(FrameSource):
    """Frame source that reads frames from a video file via OpenCV.

    Raises FileNotFoundError at construction time (before ever touching
    cv2.VideoCapture) if the given path does not exist, so a bad path is
    caught immediately rather than deep in a read loop.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"VideoFileSource: video file not found: {filepath!r}"
            )
        self._cap = cv2.VideoCapture(filepath)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(
                f"VideoFileSource: failed to open video file: {filepath!r}. "
                f"The file may be corrupt or in an unsupported format."
            )

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def is_live(self) -> bool:
        return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
