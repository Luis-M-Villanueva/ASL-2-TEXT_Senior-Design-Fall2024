import spidev
import serial
import time
from PIL import Image
import io

SERIAL_PORT = '/dev/ttyUSB0'  # USB-to-serial device path
BAUD_RATE = 2000000  # 2M baud rate
TIMEOUT = 1  # Timeout for serial read

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Function to send the trigger (value 1) to the Arduino to begin capture
def send_trigger():
    ser.write(b'1')  # Send the value '1' over serial to trigger capture
    print("Trigger sent to MCU")

# Function to receive and save the image data
def receive_image():
    img_data = bytearray()
    header_found = False

    while True:
        byte = ser.read(1)  # Read a byte from the serial buffer
        if byte:
            if not header_found:
                if byte == b'\xFF':
                    next_byte = ser.read(1)
                    if next_byte == b'\xD8':
                        print("Receiving...")
                        img_data.append(byte[0])
                        img_data.append(next_byte[0])
                        header_found = True
            else:
                img_data.append(byte[0])  # Append the byte to image data
                if byte == b'\xFF':
                    next_byte = ser.read(1)
                    img_data.append(next_byte[0])
                    if next_byte == b'\xD9':
                        print("Finished...")
                        break

    return img_data

# Function to save the received image to a file
def save_image_to_file(img_data, filename="captured_image.jpg"):
    with open(filename, "wb") as f:
        f.write(img_data)
    print(f"Image saved as {filename}")

# Main loop
def main():
    while(ser):
        triggered = False
        send_trigger()
        while not triggered:
            if ser.in_waiting > 0:  # Check if data is available
                print("Receiving image...")
                img_data = receive_image()
                print("Image received, saving to file...")
                save_image_to_file(img_data)
                triggered = True
                break

        time.sleep(.1)  # Wait before the next capture

if __name__ == '__main__':
    main()
