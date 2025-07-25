import cv2
import numpy as np
import time
from sensor.vision_sensor import VisionSensor
from utils.data_handler import debug_print

class CvSensor(VisionSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.cap = None

    def set_up(self, channel=0):
        """
        设置摄像头通道
        Args:
            channel (int): 摄像头通道号，通常为0或1
        """
        self.cap = cv2.VideoCapture(channel)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Failed to open camera on channel {channel}")
        else:
            print(f"[{self.name}] Opened camera on channel {channel}")

        # 可选：设置图像分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_image(self):
        image = {}
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Camera is not opened.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"[{self.name}] Failed to read image from camera.")

        if "color" in self.collect_info:
            # BGR → RGB
            image["color"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if "depth" in self.collect_info:
            debug_print(self.name, "cv2 capture does not support depth image.", "WARNING")

        return image

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print(f"[{self.name}] Camera released.")

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    cam = CvSensor("cv_test")
    cam.set_up(channel=0)
    cam.set_collect_info(["color"])  # 你可以设置为 ["color"] 或 ["color", "depth"]

    cam_list = []
    for i in range(100):
        print(f"[Frame {i}]")
        data = cam.get_image()
        cam_list.append(data)
        time.sleep(0.1)
