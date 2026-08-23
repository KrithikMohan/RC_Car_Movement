from unittest.mock import MagicMock, patch

import pytest
import serial

from rccar.serial_client.client import SerialClient, SerialClientError


class TestWriteHappyPath:
    @patch("rccar.serial_client.client.serial.Serial")
    def test_write_succeeds_against_mocked_port(self, mock_serial_cls):
        mock_port = MagicMock()
        mock_port.is_open = True
        mock_port.write.return_value = 5
        mock_serial_cls.return_value = mock_port

        client = SerialClient(port="/dev/ttyUSB0", baud=9600, timeout=1.0)
        result = client.write(b"S,2,0\n")

        assert result is None  # success signaled by no exception
        mock_port.write.assert_called_once_with(b"S,2,0\n")
        mock_serial_cls.assert_called_once_with(
            "/dev/ttyUSB0", 9600, write_timeout=1.0
        )


class TestOpenFailure:
    @patch("rccar.serial_client.client.serial.Serial")
    def test_port_busy_raises_clear_error(self, mock_serial_cls):
        mock_serial_cls.side_effect = serial.SerialException(
            "could not open port /dev/ttyUSB0: [Errno 16] busy"
        )

        with pytest.raises(SerialClientError) as exc_info:
            SerialClient(port="/dev/ttyUSB0", baud=9600, timeout=1.0)

        # Wraps, doesn't just pass through, the raw pyserial exception.
        assert "/dev/ttyUSB0" in str(exc_info.value)
        assert not isinstance(exc_info.value, serial.SerialException)


class TestWriteTimeout:
    @patch("rccar.serial_client.client.serial.Serial")
    def test_write_timeout_reports_failure(self, mock_serial_cls):
        mock_port = MagicMock()
        mock_port.is_open = True
        mock_port.write.side_effect = serial.SerialTimeoutException(
            "Write timeout"
        )
        mock_serial_cls.return_value = mock_port

        client = SerialClient(port="/dev/ttyUSB0", baud=9600, timeout=0.2)

        with pytest.raises(SerialClientError) as exc_info:
            client.write(b"S,2,0\n")

        assert not isinstance(exc_info.value, serial.SerialTimeoutException)
        assert "timed out" in str(exc_info.value)


class TestClose:
    @patch("rccar.serial_client.client.serial.Serial")
    def test_close_is_idempotent(self, mock_serial_cls):
        mock_port = MagicMock()
        mock_port.is_open = True
        mock_serial_cls.return_value = mock_port

        client = SerialClient(port="/dev/ttyUSB0", baud=9600, timeout=1.0)
        client.close()
        # Simulate the mock now reporting closed, as a real Serial would.
        mock_port.is_open = False
        client.close()  # should not raise

        mock_port.close.assert_called_once()
