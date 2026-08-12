#!/usr/bin/env python3
import sys
import time
import serial

# Define the serial port for your host computer connection
# Raspberry Pi / Jetson default internal: '/dev/ttyS0' or '/dev/ttyAMA0'
# USB to UART Cable connection default: '/dev/ttyUSB0'
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 115200


def main():
    print(f"Initializing connection to UGV02 on {SERIAL_PORT}...")

    try:
        # Open serial port with required UGV02 configurations
        # Added short timeout to prevent readline() from blocking indefinitely
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5
        )

        # Flush buffers to clear stale boot messages
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.1)

        print("Connected! Type a JSON command and press Enter.")
        print("Example: {\"T\":13,\"X\":0.2,\"Z\":0.0}")
        print("Type 'exit' to quit.\n")

        while True:
            # Capture command input from terminal line
            user_input = input("UGV_CMD > ").strip()

            if user_input.lower() == 'exit':
                break

            if not user_input:
                continue

            # Append the mandatory newline required by the ESP32 parser
            command_string = user_input + "\n"

            # Send the encoded bytes down the line
            bytes_written = ser.write(command_string.encode('utf-8'))
            ser.flush()  # Force transmission down the wire

            # Pause briefly to allow the UGV02 hardware processor to answer
            time.sleep(0.05)

            # Check for incoming feedback logs
            if ser.in_waiting > 0:
                print("--- Response Received ---")
                while ser.in_waiting > 0:
                    # Read lines until buffer empties out
                    response_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if response_line:
                        print(f"UGV_OUT: {response_line}")
                print("-------------------------")
            else:
                print(f"[Sent {bytes_written} bytes. No serial response feedback received]")

    except serial.SerialException as e:
        print(f"\nSerial Port Error: {e}")
        print("Verify your device path permissions: 'sudo chmod 666 /dev/ttyUSB0'")
    except KeyboardInterrupt:
        print("\nExiting command interface script.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port interface closed safely.")


if __name__ == '__main__':
    main()

