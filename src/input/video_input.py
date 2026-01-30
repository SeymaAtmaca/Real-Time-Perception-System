import cv2
from .base_input import BaseInput

class VideoInput(BaseInput):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Video cannot be opened")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
