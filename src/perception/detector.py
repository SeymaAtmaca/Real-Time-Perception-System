from ultralytics import YOLO
from .labels import TARGET_CLASSES

class Detector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if cls_name not in TARGET_CLASSES:
                continue

            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": TARGET_CLASSES[cls_name],
                "conf": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return detections
