import tkinter as tk
from PIL import Image, ImageTk
import requests
import cv2
import threading
from io import BytesIO
from urllib.request import urlopen

class CameraApp:
    def __init__(self, master):
        self.master = master
        self.master.title("ESP32-CAM GUI")
        self.video_url = "http://<ESP32-IP>/stream"  # MJPEG stream URL from ESP32-CAM
        self.command_url = "http://<ESP32-IP>/command"  # Command URL for controlling ESP32
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()

        self.button_start = tk.Button(self.master, text="Start Stream", command=self.start_stream)
        self.button_start.pack()

        self.button_stop = tk.Button(self.master, text="Stop Stream", command=self.stop_stream)
        self.button_stop.pack()

    def start_stream(self):
        """Start displaying the video stream"""
        self.capture = cv2.VideoCapture(self.video_url)
        self.update_frame()

    def stop_stream(self):
        """Stop the video stream"""
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

    def update_frame(self):
        """Continuously update the frame on the canvas"""
        ret, frame = self.capture.read()
        if ret:
            # Convert frame to PIL Image
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv2_image)
            photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.master.after(10, self.update_frame)

    def send_command(self, command):
        """Send command to ESP32"""
        response = requests.post(self.command_url, data={"command": command})
        print(response.text)

def run_gui():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
