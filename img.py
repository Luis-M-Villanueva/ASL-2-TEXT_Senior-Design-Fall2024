import spidev
import serial
import time
from PIL import Image
import io
import os 

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
    with open(os.path.join(os.getcwd(),'cap'),"wb") as f:
        while True:
            try:
                byte = ser.read(1)# Read a byte from the serial buffer
                print("saving:",byte)
                f.write(byte)
            except: 
                print("ERROR")
                exit

def main():
    while(ser):
        triggered = False
        send_trigger()
        while not triggered:
            if ser.in_waiting > 0:  # Check if data is available
                print("Receiving image...")
                receive_image()
                print("Image received, saving to file...")
                triggered = True
                break

        time.sleep(.1)  # Wait before the next capture

if __name__ == '__main__':
    main()
