import serial
import threading

# Configuration for the SPI serial port
SPI_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200  # Adjust based on your SPI device specifications
ser = serial.Serial(SPI_PORT, BAUD_RATE, timeout=1)

def receive_image():
    img_data = bytearray()
    header_found = False
    while True:
        byte = ser.read(1)  # Read a byte from the serial buffer
        if not header_found:
            if byte == b'\xFF':
                next_byte = ser.read(1)
                if next_byte == b'\xD8':  # JPEG start marker
                    img_data.extend([0xFF, 0xD8])
                    header_found = True
                else:
                    continue
        else:
            img_data.append(ord(byte))  # Append the byte to image data
            if byte == b'\xFF':
                next_byte = ser.read(1)
                if next_byte == b'\xD9':  # JPEG end marker
                    img_data.extend([0xFF, 0xD9])
                    break
    return img_data

def read_spi(ser):
    """Continuously read data from the SPI device."""
    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)  # Read all available data
                print(f"Received: {data}")
    except Exception as e:
        print(f"Error reading SPI: {e}")

def save_image_to_file(img_data, filename="captured_image.jpg"):
    with open(filename, "wb") as f:
        f.write(img_data)
    print(f"Image saved as {filename}")

def main():
    # Open the SPI port
    try:

        print(f"Connected to {SPI_PORT} at {BAUD_RATE} baud.")
    except Exception as e:
        print(f"Failed to open SPI port: {e}")
        return

    # Start a thread to read from the SPI device
    read_thread = threading.Thread(target=read_spi, args=(ser,), daemon=True)
    read_thread.start()

    # Allow the user to send data interactively
    try:
        while True:
            user_input = input("Enter bytes to send (e.g., '0x01 0x02 0x03') or 'exit': ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break

            try:
                # Convert user input into bytes
                bytes_to_send = bytes(int(b, 16) for b in user_input.split())
                ser.write(bytes_to_send)
                print(f"Sent: {bytes_to_send}")
                #image_data = receive_image()
                #save_image_to_file(image_data)
            except ValueError:
                print("Invalid input. Please enter bytes in hex format (e.g., '0x01 0x02').")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        ser.close()
        print("SPI port closed.")

if __name__ == "__main__":
    main()
