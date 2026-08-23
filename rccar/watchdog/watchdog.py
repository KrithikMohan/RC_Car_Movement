"""Watchdog (T24): stale frame / processing stall / serial failure -> forced stop.

Three independent failure triggers all funnel into the same corrective
action -- attempt to send a STOP command to the MCU over the serial link:

  1. **Stale frame** -- no new frame has arrived within ``frame_timeout_ms``
     of the last :meth:`Watchdog.on_frame_received` call. Detected by
     :meth:`Watchdog.check_frame_staleness`, called periodically from the
     main loop.
  2. **Processing stall** -- the current loop iteration has been running
     longer than an allowed ``max_stall_ms``. Detected by
     :meth:`Watchdog.check_processing_stall`, a generic helper the caller
     drives with the timestamp its own iteration started at.
  3. **Serial write failure** -- a normal drive-command write raises
     ``SerialClientError``. :meth:`Watchdog.write_command` wraps
     ``serial_client.write`` and, on failure, makes a last-ditch attempt to
     send STOP via :meth:`Watchdog.send_stop` (which itself may also fail --
     that is tolerated, not re-raised).

All three paths funnel through :meth:`Watchdog.send_stop`, which never lets
a ``SerialClientError``/``ProtocolError`` escape -- the watchdog's job is to
*try* to stop the car and keep the process alive even if that attempt also
fails (e.g. the serial link is fully dead).

The clock is injectable (defaults to ``time.monotonic``) so tests can drive
the watchdog with a fully deterministic, mocked clock instead of sleeping in
real time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from rccar.serial_client.client import SerialClientError
from rccar.serial_client.protocol import ProtocolError, SpeedTier, encode_command

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "watchdog.yaml"

_DEFAULT_FRAME_TIMEOUT_MS = 500.0


def load_frame_timeout_ms(config_path: "str | None" = None) -> float:
    """Best-effort load of ``frame_timeout_ms`` from ``config/watchdog.yaml``.

    Reads from ``config_path`` if given, otherwise from the default
    ``config/watchdog.yaml`` (resolved relative to the repo root). Falls
    back to :data:`_DEFAULT_FRAME_TIMEOUT_MS` if the file is missing or
    malformed, mirroring the pattern used elsewhere in the codebase (e.g.
    ``rccar.decision.smoothing.load_window_size``,
    ``rccar.curb.confidence._load_defaults``).
    """
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("frame_timeout_ms", _DEFAULT_FRAME_TIMEOUT_MS))
    except (OSError, ValueError, yaml.YAMLError):
        return _DEFAULT_FRAME_TIMEOUT_MS


class Watchdog:
    """Forces a STOP command when frames go stale, processing stalls, or
    the serial link fails.

    Parameters
    ----------
    serial_client:
        A ``SerialClient``-like object exposing ``.write(data: bytes)``,
        which raises ``SerialClientError`` on failure/timeout (never
        return-False style -- see ``rccar.serial_client.client``).
    frame_timeout_ms:
        Max allowed gap between frames before :meth:`check_frame_staleness`
        trips. Defaults to a best-effort load from ``config/watchdog.yaml``
        (falling back to 500ms if the file is missing/malformed).
    clock:
        Injectable time source, defaults to ``time.monotonic``. Tests
        should pass a simple mutable counter/callable instead of relying on
        real elapsed time.
    """

    def __init__(
        self,
        serial_client,
        frame_timeout_ms: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.serial_client = serial_client
        self.frame_timeout_ms = (
            frame_timeout_ms if frame_timeout_ms is not None else load_frame_timeout_ms()
        )
        self.clock = clock

        # No frame has been received yet -- see check_frame_staleness for
        # why this must not immediately count as "stale".
        self._last_frame_time: Optional[float] = None

        self.last_stop_send_failed: bool = False
        self.last_stop_send_exception: Optional[Exception] = None

    def on_frame_received(self) -> None:
        """Record that a new frame has just arrived.

        Call this each time a frame is received/processed; it resets the
        staleness clock used by :meth:`check_frame_staleness`.
        """
        self._last_frame_time = self.clock()

    def check_frame_staleness(self) -> bool:
        """Return True (and force a STOP) if too long has passed since the
        last :meth:`on_frame_received` call.

        Before any frame has ever been received, ``_last_frame_time`` is
        ``None`` and this deliberately returns False ("not stale yet")
        rather than tripping immediately -- the watchdog should not
        false-trigger before the main loop has even gotten its first
        frame; callers are expected to call :meth:`on_frame_received` once
        the pipeline is up before this check becomes meaningful.
        """
        if self._last_frame_time is None:
            return False
        elapsed = self.clock() - self._last_frame_time
        if elapsed > self.frame_timeout_ms / 1000.0:
            self.send_stop()
            return True
        return False

    def check_processing_stall(self, loop_start_time: float, max_stall_ms: float) -> bool:
        """Return True (and force a STOP) if the current loop iteration --
        which the caller reports started at ``loop_start_time`` -- has run
        longer than ``max_stall_ms``.

        Generic helper: models "processing stalled mid-iteration" for any
        caller-defined notion of "this iteration/step started at time X".
        """
        elapsed = self.clock() - loop_start_time
        if elapsed > max_stall_ms / 1000.0:
            self.send_stop()
            return True
        return False

    def send_stop(self) -> bool:
        """Attempt to send a STOP command via the serial client.

        Never lets a ``SerialClientError``/``ProtocolError`` propagate --
        the watchdog's job is to try to stop the car, and to keep the
        process alive even if that attempt also fails (e.g. the serial
        link itself is dead). Records the failure on
        ``last_stop_send_failed``/``last_stop_send_exception`` for callers
        or logs to inspect. Returns True on success, False on failure.
        """
        try:
            self.serial_client.write(encode_command(SpeedTier.STOP, 0))
        except (SerialClientError, ProtocolError) as exc:
            self.last_stop_send_failed = True
            self.last_stop_send_exception = exc
            return False
        self.last_stop_send_failed = False
        self.last_stop_send_exception = None
        return True

    def write_command(self, speed: "SpeedTier | int", steer: int) -> bool:
        """Write a normal drive command, guarding against serial failure.

        On ``SerialClientError`` (a normal write failing mid-loop), catches
        it, makes a last-ditch attempt to force STOP via :meth:`send_stop`
        (which may itself also fail -- that is tolerated, not raised), and
        returns False. Returns True on a successful write.
        """
        try:
            self.serial_client.write(encode_command(speed, steer))
        except SerialClientError:
            self.send_stop()
            return False
        return True
