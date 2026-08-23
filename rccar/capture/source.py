"""FrameSource abstraction: a common interface over live cameras and video files."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class FrameSource(ABC):
    """Common interface for anything that can hand out video frames.

    Implementations wrap ``cv2.VideoCapture`` (or something with an
    equivalent call surface) so the rest of the pipeline doesn't need to
    know whether frames are coming from a live camera or a recorded file.
    """

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next frame as a numpy array, or None at end-of-stream
        or on failure to read a frame."""
        raise NotImplementedError

    @abstractmethod
    def is_live(self) -> bool:
        """Return True if this source represents a live camera feed, False
        if it's a recorded/file-backed source."""
        raise NotImplementedError

    def release(self) -> None:
        """Optional hook for releasing underlying resources. Default no-op;
        implementations that hold a cv2.VideoCapture should override this."""
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
