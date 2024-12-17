import serial
import time

SERIAL_PORT = '/dev/ttyUSB0'  # USB-to-serial device path
BAUD_RATE = 2000000  # 2M baud rate
TIMEOUT = 1  # Timeout for serial read

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Function to send the trigger (value 1) to the Arduino to begin capture
def send_trigger():
    ser.write(b'1')  # Send the value '1' over serial to trigger capture
    print("Trigger sent to Arduino.")

# Function to receive and save the image data
def receive_image():
    img_data = bytearray()
    header_found = False

    while True:
        byte = ser.read(1)  # Read a byte from the serial buffer
        if not byte:
            continue  # Skip if no data is received
        byte = byte[0]  # Unwrap the byte

        # Print the byte received for debugging
        print(f"{byte:02x} ", end="")

        # Check for JPEG start marker (0xFF 0xD8)
        if not header_found:
            if byte == 0xFF:
                next_byte = ser.read(1)  # Read the next byte
                if next_byte == b'\xD8':  # JPEG start marker
                    img_data.extend([0xFF, 0xD8])
                    header_found = True
                    print("JPEG Start Marker Found.")
                else:
                    continue  # Skip any bytes until start marker is found
        else:
            img_data.append(byte)  # Append the byte to image data

            # Check for JPEG end marker (0xFF 0xD9)
            if byte == 0xFF:
                next_byte = ser.read(1)  # Check the next byte
                if next_byte == b'\xD9':  # JPEG end marker
                    img_data.extend([0xFF, 0xD9])
                    print("JPEG End Marker Found.")
                    break  # End the loop after finding the end marker

    return img_data

# Function to save the received image to a file
def save_image_to_file(img_data, filename="captured_image.jpg"):
    with open(filename, "wb") as f:
        f.write(img_data)
    print(f"Image saved as {filename}")

# Main loop
def main():
    triggered = False
    while True:
        if not triggered:
            # Send the trigger to the Arduino to start capturing
            send_trigger()

            # Wait for the image data to be received
            print("Listening for image data...")
            img_data = receive_image()
            if img_data:
                print("Image received, saving to file...")
                save_image_to_file(img_data)
                triggered = True  # Stop after one capture
                break  # Exit after capturing one image

        time.sleep(1)  # Wait before the next capture attempt

if __name__ == '__main__':
    main()
