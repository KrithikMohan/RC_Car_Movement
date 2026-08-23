"""Thin wrapper around pyserial's `Serial` for the RC-car Pi <-> MCU link.

Provides `SerialClient`, a small open/write/close wrapper that:

  - Opens the serial port in `__init__`, translating pyserial's
    `serial.SerialException` (port busy, device missing, permission denied,
    etc.) into a clear `SerialClientError` instead of letting a cryptic
    pyserial traceback bubble up.
  - Uses pyserial's own `write_timeout` so a write that cannot complete
    within `timeout` seconds raises `serial.SerialTimeoutException`
    internally, rather than blocking forever. `SerialClient.write()` catches
    that (and `SerialException`) and re-raises as `SerialClientError`.

Failure-reporting style: **raise, not return False**. `write()` returns
`None` on success and raises `SerialClientError` on any failure (timeout,
port not open, or a lower-level serial error). This keeps the "did it
work?" check at the call site simple (try/except) and matches
`decode_command`/`encode_command` in `protocol.py`, which also signal
failure via exceptions rather than sentinel return values.
"""

from __future__ import annotations

import serial


class SerialClientError(Exception):
    """Raised when opening, writing to, or otherwise using the serial port fails."""

    pass


class SerialClient:
    """Minimal open/write/close wrapper around `serial.Serial`.

    The port is opened eagerly in `__init__` (no separate `.open()` step).
    `timeout` is applied as pyserial's `write_timeout`, bounding how long a
    single `.write()` call may block before it is reported as a failure.
    """

    def __init__(self, port: str, baud: int, timeout: float) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial: "serial.Serial | None" = None
        try:
            self._serial = serial.Serial(port, baud, write_timeout=timeout)
        except serial.SerialException as exc:
            raise SerialClientError(
                f"failed to open serial port {port!r} at {baud} baud: {exc}"
            ) from exc

    def write(self, data: bytes) -> None:
        """Write `data` to the serial port.

        Raises `SerialClientError` if the port is not open, the write times
        out (per the `timeout` given at construction), or pyserial reports
        any other serial error. Never blocks longer than `timeout` seconds.
        """
        if self._serial is None or not self._serial.is_open:
            raise SerialClientError("cannot write: serial port is not open")
        try:
            self._serial.write(data)
        except serial.SerialTimeoutException as exc:
            raise SerialClientError(
                f"write to {self.port!r} timed out after {self.timeout}s"
            ) from exc
        except serial.SerialException as exc:
            raise SerialClientError(f"write to {self.port!r} failed: {exc}") from exc

    def close(self) -> None:
        """Close the underlying port. Safe to call multiple times."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
