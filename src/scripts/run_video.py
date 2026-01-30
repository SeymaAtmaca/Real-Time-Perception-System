# import cv2 
# import sys 

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     sys.exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     cv2.imshow('Video Feed', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# vision_detect.py
from ultralytics import YOLO
import cv2

# Pretrained model (COCO dataset)
model = YOLO("yolov8n.pt")  # nano = hızlı

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found")

TARGET_CLASSES = ["person", "bird", "airplane"]

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    cv2.imshow("Vision", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
