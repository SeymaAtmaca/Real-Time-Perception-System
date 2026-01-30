import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# ---------------------
# LOAD MODELS
# ---------------------
model = YOLO("yolov8n.pt")
tracker = Sort()

# ---------------------
# VIDEO SOURCE
# ---------------------
cap = cv2.VideoCapture("assets/dji.mp4")

if not cap.isOpened():
    raise RuntimeError("Video açılamadı")

# ---------------------
# MAIN LOOP
# ---------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO DETECTION
    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Sadece insan / kuş / drone (şimdilik hepsi)
        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append([x1, y1, x2, y2, conf])

    if len(detections) == 0:
        dets = np.empty((0, 5))
    else:
        dets = np.array(detections)

    # SORT TRACKING
    tracks = tracker.update(dets)

    # DRAW
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

    cv2.imshow("YOLO + SORT", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
