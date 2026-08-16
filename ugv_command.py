#!/usr/bin/env python3
"""
UGV02 Serial Command Interface

Simple function-based interface for sending commands to UGV02 over serial.
Handles all connection, communication, and cleanup internally.
"""

import time
import serial
from typing import Optional

# Configuration
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 115200


def send_command(
    command: str,
    port: str = SERIAL_PORT,
    baudrate: int = BAUD_RATE,
    timeout: float = 0.5,
    response_wait: float = 0.05
) -> Optional[str]:
    """
    Send a command to UGV02 and return the response.

    Handles serial port connection, transmission, and response collection internally.

    Args:
        command: JSON command string (e.g., '{"T":13,"X":0.2,"Z":0.0}')
        port: Serial port path (default: '/dev/serial0')
        baudrate: Baud rate (default: 115200)
        timeout: Serial read timeout in seconds (default: 0.5)
        response_wait: Time to wait for response in seconds (default: 0.05)

    Returns:
        Response string from UGV02, or None if no response received.
        Multiple response lines are joined with newlines.

    Raises:
        serial.SerialException: If serial port cannot be opened or communication fails.
        ValueError: If command is empty.

    Example:
        >>> response = send_command('{"T":13,"X":0.2,"Z":0.0}')
        >>> print(response)
    """

    if not command or not command.strip():
        raise ValueError("Command cannot be empty")

    response_lines = []

    try:
        # Open serial port with UGV02 configurations
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )

        # Flush buffers to clear stale boot messages
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.1)

        # Send command with mandatory newline
        command_string = command.strip() + "\n"
        bytes_written = ser.write(command_string.encode('utf-8'))
        ser.flush()

        # Allow UGV02 hardware processor time to respond
        time.sleep(response_wait)

        # Collect response
        if ser.in_waiting > 0:
            while ser.in_waiting > 0:
                response_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if response_line:
                    response_lines.append(response_line)

        ser.close()

        # Return joined response or None if no response
        return "\n".join(response_lines) if response_lines else None

    except serial.SerialException as e:
        raise serial.SerialException(
            f"Serial port error on {port}: {e}\n"
            "Verify device path and permissions: 'sudo chmod 666 /dev/ttyUSB0'"
        )


def send_command_interactive(
    port: str = SERIAL_PORT,
    baudrate: int = BAUD_RATE
) -> None:
    """
    Interactive command interface for testing.

    Allows typing multiple commands without reconnecting between each.
    Type 'exit' to quit.
    """

    print(f"Initializing connection to UGV02 on {port}...")

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5
        )

        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.1)

        print("Connected! Type a JSON command and press Enter.")
        print("Example: {\"T\":13,\"X\":0.2,\"Z\":0.0}")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("UGV_CMD > ").strip()

            if user_input.lower() == 'exit':
                break

            if not user_input:
                continue

            try:
                response = send_command(
                    user_input,
                    port=port,
                    baudrate=baudrate
                )

                if response:
                    print("--- Response Received ---")
                    print(f"UGV_OUT: {response}")
                    print("-------------------------")
                else:
                    command_bytes = len(user_input.encode('utf-8'))
                    print(f"[Sent {command_bytes} bytes. No serial response received]")

            except serial.SerialException as e:
                print(f"Error: {e}")

    except serial.SerialException as e:
        print(f"\nSerial Port Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting command interface.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")


if __name__ == '__main__':
    # Run interactive mode
    send_command_interactive()
