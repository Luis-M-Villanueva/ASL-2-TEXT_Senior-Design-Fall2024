import serial
import cv2
import numpy as np
import time

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust this as needed
BAUD_RATE = 2000000          # Baud rate must match the ESP32 setting
TIMEOUT = 1                  # Timeout in seconds

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Function to send the trigger (value 1) to the Arduino to begin capture
def send_trigger():
    ser.write(b'2')  # Send the value '1' over serial to trigger capture
    print("Trigger sent to MCU")

def receive_stream():
    """
    Receives image data over serial and streams it using OpenCV.
    """
    img_data = bytearray()
    header_found = False
    frame_count = 0
    
    print("Waiting for video stream...")

    while True:
        try:
            byte = ser.read(1)  # Read one byte from serial
            if byte:
                if not header_found:
                    if byte == b'\xFF':
                        next_byte = ser.read(1)
                        if next_byte == b'\xD8':  # Start of JPEG
                            img_data.append(byte[0])
                            img_data.append(next_byte[0])
                            header_found = True
                else:
                    img_data.append(byte[0])
                    if byte == b'\xFF':
                        next_byte = ser.read(1)
                        img_data.append(next_byte[0])
                        if next_byte == b'\xD9':  # End of JPEG")
                            frame_count += 1
                            
                            # Convert the data into an image
                            np_img = np.frombuffer(img_data, dtype=np.uint8)
                            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Display the frame
                                cv2.imshow('Video Stream', frame)
                                
                                # Break on pressing 'q'
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    print("Stream stopped by user.")
                                    ser.close()
                                    cv2.destroyAllWindows()
                                    return
                            
                            # Reset for the next frame
                            img_data = bytearray()
                            header_found = False

        except KeyboardInterrupt:
            print("Stream interrupted by user.")
            break

        except Exception as e:
            print(f"Error: {e}")
            break

    ser.close()
    cv2.destroyAllWindows()

def main():
    send_trigger()
    receive_stream()


if __name__ == "__main__":
    main()

