import cv2
from input.video_input import VideoInput
from perception.detector import Detector
from utils.fps import FPS
from perception.tracker import Tracker

input_src = VideoInput("assets/dji.mp4")
detector = Detector()
fps_counter = FPS()
tracker = Tracker()

while True:
    frame = input_src.read()
    if frame is None:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    for t in tracks:
        x1,y1,x2,y2 = t["bbox"]
        tid = t["id"]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,0,0),
            2
        )

    fps = fps_counter.tick()
    cv2.putText(frame,f"FPS:{fps:.1f}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Perception", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
