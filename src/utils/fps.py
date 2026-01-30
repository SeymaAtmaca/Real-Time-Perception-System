import time

class FPS:
    def __init__(self):
        self.last = time.time()

    def tick(self):
        now = time.time()
        fps = 1.0 / (now - self.last)
        self.last = now
        return fps
