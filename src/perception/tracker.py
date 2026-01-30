from .sort import Sort
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = Sort(
            max_age=20,
            min_hits=3,
            iou_threshold=0.3
        )

    def update(self, detections):
        if len(detections) == 0:
            tracks = self.tracker.update(np.empty((0,5)))
            return []

        dets = []
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            dets.append([x1,y1,x2,y2,d["conf"]])

        dets = np.array(dets)
        tracks = self.tracker.update(dets)

        results = []
        for t in tracks:
            x1,y1,x2,y2,track_id = map(int, t)
            results.append({
                "id": track_id,
                "bbox": (x1,y1,x2,y2)
            })
        return results
